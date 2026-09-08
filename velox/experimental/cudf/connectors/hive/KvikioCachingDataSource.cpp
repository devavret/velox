/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/KvikioCachingDataSource.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/utilities/pinned_memory.hpp>

#include <folly/Executor.h>
#include <folly/executors/QueuedImmediateExecutor.h>
#include <folly/futures/Future.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

// Launches a blocking host read on the connector IO executor or on the
// eventual waiting thread, whichever starts first. The deferred fallback
// prevents executor starvation when all connector threads are preparing
// splits that depend on reads queued to the same executor.
std::future<size_t> submitRead(
    folly::Executor* executor,
    std::function<size_t()> read) {
  auto task = std::make_shared<std::packaged_task<size_t()>>(std::move(read));
  auto result = task->get_future();
  auto once = std::make_shared<std::once_flag>();
  auto run = [once, task]() { std::call_once(*once, [task]() { (*task)(); }); };
  executor->add(run);
  return std::async(
      std::launch::deferred,
      [run = std::move(run), result = std::move(result)]() mutable {
        run();
        return result.get();
      });
}

class PinnedStagingBuffer {
 public:
  explicit PinnedStagingBuffer(size_t size)
      : mr_(cudf::get_pinned_memory_resource()),
        size_(size),
        data_(static_cast<uint8_t*>(mr_.allocate_sync(size))) {}

  ~PinnedStagingBuffer() {
    mr_.deallocate_sync(data_, size_);
  }

  PinnedStagingBuffer(const PinnedStagingBuffer&) = delete;
  PinnedStagingBuffer& operator=(const PinnedStagingBuffer&) = delete;

  uint8_t* data() const {
    return data_;
  }

 private:
  rmm::host_device_async_resource_ref mr_;
  size_t size_;
  uint8_t* data_;
};

// One reusable pinned staging allocation per connector executor thread. These
// objects intentionally have process lifetime because CUDA resources cannot be
// safely destroyed by thread-local/static teardown after the CUDA context.
struct PinnedStagingSlot {
  PinnedStagingBuffer* buffer{nullptr};
  size_t capacity{0};
  cudaEvent_t event{nullptr};

  uint8_t* reserve(size_t size) {
    if (event != nullptr) {
      CUDF_CUDA_TRY(cudaEventSynchronize(event));
    }
    if (capacity < size) {
      delete buffer;
      buffer = new PinnedStagingBuffer(size);
      capacity = size;
    }
    return buffer->data();
  }

  void recordCopy(cudaStream_t stream) {
    if (event == nullptr) {
      CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    CUDF_CUDA_TRY(cudaEventRecord(event, stream));
  }
};

} // namespace

KvikioCachingDataSource::KvikioCachingDataSource(
    std::unique_ptr<cudf::io::datasource> delegate,
    std::string path,
    folly::Executor* executor)
    : delegate_(std::move(delegate)),
      fileSize_(delegate_->size()),
      executor_(executor),
      cache_(cache::AsyncDataCache::getInstance()),
      fileNum_(fileIds(), path) {}

KvikioCachingDataSource::~KvikioCachingDataSource() = default;

size_t KvikioCachingDataSource::size() const {
  return fileSize_;
}

bool KvikioCachingDataSource::supports_device_read() const {
  return true;
}

bool KvikioCachingDataSource::is_device_read_preferred(size_t size) const {
  return delegate_->is_device_read_preferred(size);
}

cache::CachePin KvikioCachingDataSource::pinRange(
    uint64_t offset,
    uint64_t size,
    bool prefetch) {
  const cache::RawFileCacheKey key{fileNum_.id(), offset};
  constexpr int kMaxAttempts = 32;
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    folly::SemiFuture<bool> wait(false);
    cache::CachePin pin;
    try {
      pin = cache_->findOrCreate(key, size, /*contiguous=*/true, &wait);
    } catch (const VeloxRuntimeError&) {
      return {};
    }
    if (pin.empty()) {
      if (!wait.valid()) {
        return {};
      }
      std::move(wait).via(&folly::QueuedImmediateExecutor::instance()).wait();
      continue;
    }

    auto* entry = pin.checkedEntry();
    if (!entry->isExclusive()) {
      if (!prefetch) {
        entry->getAndClearFirstUseFlag();
      }
      return pin;
    }
    if (!entry->hasContiguousData()) {
      return {};
    }
    if (prefetch) {
      entry->setPrefetch();
    }
    try {
      const auto bytesRead = delegate_->host_read(
          offset,
          size,
          reinterpret_cast<uint8_t*>(entry->contiguousData()));
      VELOX_CHECK_EQ(bytesRead, size, "Short KvikIO cache fill");
      entry->setExclusiveToShared();
      return pin;
    } catch (...) {
      // Releasing an exclusive entry removes its incomplete contents and
      // wakes any waiters.
      pin.clear();
      throw;
    }
  }
  return {};
}

void KvikioCachingDataSource::readThroughCache(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  auto pin = pinRange(offset, size, /*prefetch=*/false);
  if (pin.empty()) {
    const auto bytesRead = delegate_->host_read(offset, size, dst);
    VELOX_CHECK_EQ(bytesRead, size, "Short uncached KvikIO read");
    return;
  }
  std::memcpy(dst, pin.checkedEntry()->contiguousData(), size);
}

size_t KvikioCachingDataSource::prefetchRange(size_t offset, size_t size) {
  auto pin = pinRange(offset, size, /*prefetch=*/true);
  return pin.empty() ? 0 : size;
}

std::vector<std::future<size_t>> KvikioCachingDataSource::prefetchRanges(
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges) {
  std::vector<std::future<size_t>> tasks;
  if (cache_ == nullptr || executor_ == nullptr) {
    return tasks;
  }

  tasks.reserve(byteRanges.size());
  for (size_t range = 0; range < byteRanges.size();) {
    const auto offset = static_cast<size_t>(byteRanges[range].offset());
    auto size = static_cast<size_t>(byteRanges[range].size());
    size_t next = range + 1;
    while (next < byteRanges.size() &&
           static_cast<size_t>(byteRanges[next].offset()) == offset + size) {
      size += static_cast<size_t>(byteRanges[next].size());
      ++next;
    }
    if (offset < fileSize_ && size != 0) {
      const auto readSize = std::min(size, fileSize_ - offset);
      tasks.push_back(submitRead(executor_, [this, offset, readSize]() {
        return prefetchRange(offset, readSize);
      }));
    }
    range = next;
  }
  return tasks;
}

std::unique_ptr<cudf::io::datasource::buffer>
KvikioCachingDataSource::host_read(size_t offset, size_t size) {
  if (cache_ == nullptr) {
    return delegate_->host_read(offset, size);
  }
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const auto readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readThroughCache(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t KvikioCachingDataSource::host_read(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  if (cache_ == nullptr) {
    return delegate_->host_read(offset, size, dst);
  }
  if (offset >= fileSize_) {
    return 0;
  }
  const auto readSize = std::min(size, fileSize_ - offset);
  readThroughCache(offset, readSize, dst);
  return readSize;
}

std::future<size_t> KvikioCachingDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  if (cache_ == nullptr || executor_ == nullptr) {
    return delegate_->device_read_async(offset, size, dst, stream);
  }
  return submitRead(executor_, [this, offset, size, dst, stream]() {
    return device_read(offset, size, dst, stream);
  });
}

size_t KvikioCachingDataSource::device_read(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  if (cache_ == nullptr) {
    return delegate_->device_read(offset, size, dst, stream);
  }
  if (offset >= fileSize_) {
    return 0;
  }
  const auto readSize = std::min(size, fileSize_ - offset);
  if (readSize == 0) {
    return 0;
  }

  static thread_local PinnedStagingSlot slot;
  auto* staging = slot.reserve(readSize);
  readThroughCache(offset, readSize, staging);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      dst, staging, readSize, cudaMemcpyDefault, stream.value()));
  slot.recordCopy(stream.value());
  return readSize;
}

std::unique_ptr<cudf::io::datasource::buffer>
KvikioCachingDataSource::device_read(
    size_t offset,
    size_t size,
    rmm::cuda_stream_view stream) {
  return delegate_->device_read(offset, size, stream);
}

} // namespace facebook::velox::cudf_velox::connector::hive
