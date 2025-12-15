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

#include <rmm/mr/device/device_memory_resource.hpp>
#include "velox/common/memory/MemoryPool.h"

namespace facebook::velox::cudf::memory {

class VeloxPoolMemoryResource : public rmm::mr::device_memory_resource {
 public:
  explicit VeloxPoolMemoryResource(facebook::velox::memory::MemoryPool* pool)
      : pool_(pool) {
    VELOX_CHECK_NOT_NULL(pool_);
  }

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
    // Velox memory pool allocation is currently synchronous and doesn't take a
    // stream. The underlying allocator (if it's RmmMemoryAllocator) might use a
    // default stream.
    return pool_->allocate(bytes);
  }

  void do_deallocate(void* p, std::size_t bytes, rmm::cuda_stream_view stream)
      override {
    pool_->free(p, bytes);
  }

  bool do_is_equal(
      const device_memory_resource& other) const noexcept override {
    if (auto* other_res =
            dynamic_cast<const VeloxPoolMemoryResource*>(&other)) {
      return pool_ == other_res->pool_;
    }
    return false;
  }

  facebook::velox::memory::MemoryPool* pool_;
};

} // namespace facebook::velox::cudf::memory
