/* Copyright (c) Facebook, Inc. and its affiliates.
 * Licensed under the Apache License, Version 2.0. */
#pragma once

#include "velox/common/base/Exceptions.h"
#include <lz4.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <cuda_runtime_api.h>

namespace facebook::velox::cudf_velox {

inline void checkHostSpillCuda(cudaError_t status) {
  VELOX_CHECK(status == cudaSuccess, "Host spill CUDA operation: {}", cudaGetErrorString(status));
}

// Per-process bound: four 64 MiB CUDA-pinned staging buffers. Large retained
// spill datasets remain pageable and compressed. Acquire never holds the pool
// mutex while executing CUDA operations or compression.
class HostSpillStaging {
 public:
  static constexpr size_t kBytes = 64ULL << 20;
  struct Slot { void* data{nullptr}; bool busy{false}; };
  static HostSpillStaging& instance() {
    static HostSpillStaging pool;
    return pool;
  }
  Slot* acquire() {
    std::unique_lock lock(mutex_);
    available_.wait(lock, [&] {
      for (auto& slot : slots_) if (!slot.busy) return true;
      return false;
    });
    Slot* selected = nullptr;
    for (auto& slot : slots_) {
      if (!slot.busy) { selected = &slot; slot.busy = true; break; }
    }
    lock.unlock();
    if (!selected->data) {
      auto status = cudaHostAlloc(&selected->data, kBytes, cudaHostAllocPortable);
      if (status != cudaSuccess) {
        release(selected);
        VELOX_FAIL("Host spill staging allocation: {}", cudaGetErrorString(status));
      }
    }
    return selected;
  }
  void release(Slot* slot) {
    { std::lock_guard lock(mutex_); slot->busy = false; }
    available_.notify_one();
  }
  ~HostSpillStaging() {
    for (auto& slot : slots_) if (slot.data) (void)cudaFreeHost(slot.data);
  }
 private:
  std::array<Slot,4> slots_;
  std::mutex mutex_;
  std::condition_variable available_;
};

struct HostSpillStagingLease {
  HostSpillStaging::Slot* slot{HostSpillStaging::instance().acquire()};
  ~HostSpillStagingLease() { HostSpillStaging::instance().release(slot); }
};

// Lossless storage compression; metadata continues to describe the unpacked
// cuDF buffer. A zero rawBytes denotes an uncompressed payload.
inline void compressHostSpill(std::vector<uint8_t>& data, size_t& rawBytes) {
  if (data.empty() || data.size() > LZ4_MAX_INPUT_SIZE) {
    return;
  }
  const auto size = static_cast<int>(data.size());
  std::vector<uint8_t> scratch(LZ4_compressBound(size));
  const auto compressed = LZ4_compress_default(
      reinterpret_cast<const char*>(data.data()),
      reinterpret_cast<char*>(scratch.data()), size, static_cast<int>(scratch.size()));
  if (compressed > 0 && static_cast<size_t>(compressed) < data.size() * 9 / 10) {
    rawBytes = data.size();
    // Allocate at compressed size: resizing scratch would retain its raw-size capacity.
    std::vector<uint8_t> compact(scratch.begin(), scratch.begin() + compressed);
    data.swap(compact);
  }
}

inline std::vector<uint8_t> decompressHostSpill(
    const std::vector<uint8_t>& data, size_t rawBytes) {
  std::vector<uint8_t> out(rawBytes);
  const auto actual = LZ4_decompress_safe(
      reinterpret_cast<const char*>(data.data()), reinterpret_cast<char*>(out.data()),
      static_cast<int>(data.size()), static_cast<int>(rawBytes));
  VELOX_CHECK_EQ(actual, rawBytes, "Corrupt host-spill buffer");
  return out;
}

inline void downloadHostSpill(
    const void* device, size_t bytes, cudaStream_t stream,
    std::vector<uint8_t>& data, size_t& rawBytes, bool compress = true) {
  rawBytes = 0;
  if (bytes > HostSpillStaging::kBytes) {
    data.resize(bytes);
    checkHostSpillCuda(cudaMemcpyAsync(data.data(), device, bytes, cudaMemcpyDeviceToHost, stream));
    checkHostSpillCuda(cudaStreamSynchronize(stream));
    if (compress) compressHostSpill(data, rawBytes);
    return;
  }
  HostSpillStagingLease lease;
  checkHostSpillCuda(cudaMemcpyAsync(lease.slot->data, device, bytes, cudaMemcpyDeviceToHost, stream));
  checkHostSpillCuda(cudaStreamSynchronize(stream));
  if (!compress) {
    const auto* ptr = static_cast<const uint8_t*>(lease.slot->data);
    data.assign(ptr, ptr+bytes);
    return;
  }
  const auto bound = LZ4_compressBound(static_cast<int>(bytes));
  auto scratch = std::make_unique_for_overwrite<char[]>(bound);
  const auto compressed = LZ4_compress_default(static_cast<const char*>(lease.slot->data),
      scratch.get(), static_cast<int>(bytes), bound);
  if (compressed > 0 && static_cast<size_t>(compressed) < bytes * 9 / 10) {
    rawBytes = bytes;
    data.assign(scratch.get(), scratch.get()+compressed);
  } else {
    const auto* ptr = static_cast<const uint8_t*>(lease.slot->data);
    data.assign(ptr, ptr+bytes);
  }
}

inline void uploadHostSpill(
    void* device, const std::vector<uint8_t>& data, size_t rawBytes, cudaStream_t stream) {
  const auto bytes = rawBytes ? rawBytes : data.size();
  if (bytes > HostSpillStaging::kBytes) {
    auto decoded = rawBytes ? decompressHostSpill(data, rawBytes) : std::vector<uint8_t>{};
    checkHostSpillCuda(cudaMemcpyAsync(device, rawBytes ? decoded.data() : data.data(), bytes,
                                   cudaMemcpyHostToDevice, stream));
    checkHostSpillCuda(cudaStreamSynchronize(stream));
    return;
  }
  HostSpillStagingLease lease;
  if (rawBytes) {
    auto n = LZ4_decompress_safe(reinterpret_cast<const char*>(data.data()),
        static_cast<char*>(lease.slot->data), static_cast<int>(data.size()), static_cast<int>(rawBytes));
    VELOX_CHECK_EQ(n, rawBytes, "Corrupt host-spill buffer");
  } else {
    std::memcpy(lease.slot->data, data.data(), bytes);
  }
  checkHostSpillCuda(cudaMemcpyAsync(device, lease.slot->data, bytes, cudaMemcpyHostToDevice, stream));
  // The staging lease must outlive DMA. It may then be reused by another driver.
  checkHostSpillCuda(cudaStreamSynchronize(stream));
}

} // namespace facebook::velox::cudf_velox
