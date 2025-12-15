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

#pragma once

#include "velox/common/memory/MemoryAllocator.h"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <atomic>
#include <memory>

namespace facebook::velox::cudf_velox {

/// A Velox MemoryAllocator that delegates allocations to an RMM
/// device_memory_resource.
class RmmMemoryAllocator : public memory::MemoryAllocator {
 public:
  RmmMemoryAllocator(
      std::shared_ptr<rmm::mr::device_memory_resource> mr,
      size_t capacityBytes);

  ~RmmMemoryAllocator() override = default;

  Kind kind() const override {
    return Kind::kMalloc;
  }

  // TODO (dm): What is this for? Find out.
  void registerCache(const std::shared_ptr<memory::Cache>&) override;

  memory::Cache* cache() const override {
    return nullptr;
  }

  size_t capacity() const override {
    return capacity_;
  }

  bool allocateContiguousWithoutRetry(
      memory::MachinePageCount numPages,
      memory::Allocation* collateral,
      memory::ContiguousAllocation& allocation,
      memory::MachinePageCount maxPages = 0) override;

  bool allocateNonContiguousWithoutRetry(
      const memory::MemoryAllocator::SizeMix& sizeMix,
      memory::Allocation& out) override;

  void* allocateBytesWithoutRetry(uint64_t bytes, uint16_t alignment) override;

  void* allocateZeroFilledWithoutRetry(uint64_t bytes) override;

  void freeContiguous(memory::ContiguousAllocation& allocation) override;

  int64_t freeNonContiguous(memory::Allocation& allocation) override;

  void freeBytes(void* p, uint64_t size) noexcept override;

  bool growContiguousWithoutRetry(
      memory::MachinePageCount /*increment*/,
      memory::ContiguousAllocation& /*allocation*/) override {
    return false;
  }

  memory::MachinePageCount unmap(
      memory::MachinePageCount /*targetPages*/) override {
    return 0;
  }

  size_t totalUsedBytes() const override {
    return allocatedBytes_.load();
  }

  memory::MachinePageCount numAllocated() const override {
    return numAllocated_.load();
  }

  memory::MachinePageCount numMapped() const override {
    return numMapped_.load();
  }

  memory::MachinePageCount numExternalMapped() const override {
    return numMapped_.load();
  }

  bool checkConsistency() const override {
    return true;
  }

  std::string toString() const override;

 private:
  bool checkAndCharge(uint64_t bytes);

  void releaseCharge(uint64_t bytes) noexcept;

  std::shared_ptr<rmm::mr::device_memory_resource> mr_;
  const size_t capacity_;
  std::atomic<size_t> allocatedBytes_{0};
  std::atomic<memory::MachinePageCount> numAllocated_{0};
  std::atomic<memory::MachinePageCount> numMapped_{0};
};
} // namespace facebook::velox::cudf_velox
