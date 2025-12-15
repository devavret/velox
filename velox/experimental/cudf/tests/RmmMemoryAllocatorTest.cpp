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

#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"

#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::memory;

namespace {
std::shared_ptr<MemoryManager> makeGpuManager(
    std::shared_ptr<rmm::mr::device_memory_resource> mr,
    int64_t capacityBytes) {
  static std::once_flag once;
  std::call_once(once, []() { SharedArbitrator::registerFactory(); });

  auto allocator = std::make_shared<cudf_velox::RmmMemoryAllocator>(
      std::move(mr), capacityBytes);
  MemoryManager::Options opts;
  opts.allocatorCapacity = capacityBytes;
  opts.arbitratorCapacity = capacityBytes;
  opts.arbitratorKind = "SHARED";
  opts.checkUsageLeak = true;
  opts.extraArbitratorConfigs = {
      {std::string(
           memory::SharedArbitrator::ExtraConfig::kGlobalArbitrationEnabled),
       "false"}};
  return std::make_shared<MemoryManager>(opts, std::move(allocator));
}
} // namespace

TEST(RmmMemoryAllocatorTest, crossPoolCapacityEnforced) {
  // Use a modest capacity to keep the test fast.
  const int64_t capacity = 32 << 20; // 32MB
  auto mr = std::make_shared<rmm::mr::cuda_memory_resource>();
  auto manager = makeGpuManager(mr, capacity);

  auto rootA = manager->addRootPool("gpu_root_a");
  auto rootB = manager->addRootPool("gpu_root_b");
  auto leafA = rootA->addLeafChild("leafA");
  auto leafB = rootB->addLeafChild("leafB");

  // Reserve almost all capacity from leafA.
  const int64_t firstAlloc = capacity - (4 << 20);
  void* aBuf = leafA->allocate(firstAlloc);
  ASSERT_NE(aBuf, nullptr);

  // A subsequent allocation from a sibling pool should trigger arbitration and
  // fail due to lack of remaining capacity.
  EXPECT_THROW(leafB->allocate(8 << 20), VeloxRuntimeError);

  // Free and confirm the second pool can now allocate.
  leafA->free(aBuf, firstAlloc);
  EXPECT_NO_THROW({
    void* bBuf = leafB->allocate(8 << 20);
    leafB->free(bBuf, 8 << 20);
  });
}
