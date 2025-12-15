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

#include "velox/experimental/cudf/exec/RmmMemoryAllocator.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/SuccinctPrinter.h"

#include <cuda_runtime_api.h>

namespace facebook::velox::cudf_velox {
namespace {
constexpr uint16_t kGpuAlignment = memory::MemoryAllocator::kMaxAlignment;
}

RmmMemoryAllocator::RmmMemoryAllocator(
    std::shared_ptr<rmm::mr::device_memory_resource> mr,
    size_t capacityBytes)
    : mr_(std::move(mr)), capacity_(capacityBytes) {
  VELOX_CHECK_NOT_NULL(mr_);
}

void RmmMemoryAllocator::registerCache(const std::shared_ptr<memory::Cache>&) {
  VELOX_UNSUPPORTED("Cache registration is not supported for RMM allocator");
}

bool RmmMemoryAllocator::checkAndCharge(uint64_t bytes) {
  if (capacity_ != 0) {
    const auto before = allocatedBytes_.fetch_add(bytes);
    if (before + bytes > capacity_) {
      allocatedBytes_.fetch_sub(bytes);
      return false;
    }
    return true;
  }
  allocatedBytes_.fetch_add(bytes);
  return true;
}

void RmmMemoryAllocator::releaseCharge(uint64_t bytes) noexcept {
  allocatedBytes_.fetch_sub(bytes);
}

void* RmmMemoryAllocator::allocateBytesWithoutRetry(
    uint64_t bytes,
    uint16_t alignment) {
  memory::MemoryAllocator::alignmentCheck(bytes, alignment);
  if (!checkAndCharge(bytes)) {
    setAllocatorFailureMessage(
        fmt::format(
            "Exceeded RMM allocator limit: requested {}, capacity {}, used {}",
            succinctBytes(bytes),
            succinctBytes(capacity_),
            succinctBytes(allocatedBytes_.load())));
    return nullptr;
  }
  try {
    auto* ptr = mr_->allocate(bytes, rmm::cuda_stream_default);
    numAllocated_.fetch_add(memory::AllocationTraits::numPages(bytes));
    numMapped_.fetch_add(memory::AllocationTraits::numPages(bytes));
    return ptr;
  } catch (const std::exception& e) {
    releaseCharge(bytes);
    setAllocatorFailureMessage(e.what());
    return nullptr;
  }
}

void* RmmMemoryAllocator::allocateZeroFilledWithoutRetry(uint64_t bytes) {
  VELOX_UNSUPPORTED(
      "allocateZeroFilledWithoutRetry is not supported for RMM allocator");
  return nullptr;
}

bool RmmMemoryAllocator::allocateNonContiguousWithoutRetry(
    const SizeMix& sizeMix,
    memory::Allocation& out) {
  VELOX_UNSUPPORTED(
      "allocateNonContiguousWithoutRetry is not supported for RMM allocator");
  return false;
}

bool RmmMemoryAllocator::allocateContiguousWithoutRetry(
    memory::MachinePageCount numPages,
    memory::Allocation* collateral,
    memory::ContiguousAllocation& allocation,
    memory::MachinePageCount maxPages) {
  VELOX_UNSUPPORTED(
      "allocateContiguousWithoutRetry is not supported for RMM allocator");
  return false;
}

void RmmMemoryAllocator::freeContiguous(
    memory::ContiguousAllocation& allocation) {
  VELOX_UNSUPPORTED("freeContiguous is not supported for RMM allocator");
}

int64_t RmmMemoryAllocator::freeNonContiguous(memory::Allocation& allocation) {
  VELOX_UNSUPPORTED("freeNonContiguous is not supported for RMM allocator");
  return 0;
}

void RmmMemoryAllocator::freeBytes(void* p, uint64_t bytes) noexcept {
  if (p == nullptr || bytes == 0) {
    return;
  }
  mr_->deallocate(p, bytes, rmm::cuda_stream_default);
  releaseCharge(bytes);
  numAllocated_.fetch_sub(memory::AllocationTraits::numPages(bytes));
  numMapped_.fetch_sub(memory::AllocationTraits::numPages(bytes));
}

std::string RmmMemoryAllocator::toString() const {
  return fmt::format(
      "RmmMemoryAllocator[capacity {}, used {}, pages {}]",
      capacity_ == 0 ? "UNLIMITED" : succinctBytes(capacity_),
      succinctBytes(allocatedBytes_.load()),
      numAllocated_.load());
}
} // namespace facebook::velox::cudf_velox
