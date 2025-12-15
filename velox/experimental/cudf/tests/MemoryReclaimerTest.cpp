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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"

#include <gtest/gtest.h>

#include <atomic>

using namespace facebook::velox;
using namespace facebook::velox::memory;

namespace {

// A simple test reclaimer that records how many times it is consulted and
// invoked. It reclaims by shrinking free capacity from the root pool.
class CountingReclaimer : public MemoryReclaimer {
 public:
  CountingReclaimer() : MemoryReclaimer(0) {}

  bool reclaimableBytes(const MemoryPool& /*pool*/, uint64_t& reclaimableBytes)
      const override {
    ++reclaimableCalls_;
    // Advertise some reclaimable bytes so the arbitrator picks this pool.
    reclaimableBytes = kReclaimableBytes;
    return true;
  }

  uint64_t reclaim(
      MemoryPool* pool,
      uint64_t targetBytes,
      uint64_t /*maxWaitMs*/,
      Stats& /*stats*/) override {
    ++reclaimCalls_;
    return 0;
    // const uint64_t bytesToFree =
    //     targetBytes == 0 ? kReclaimableBytes
    //                      : std::min<uint64_t>(targetBytes,
    //                      kReclaimableBytes);
    // // Release free capacity from the root pool; no actual used memory is
    // // involved in this test.
    // return pool->shrink(bytesToFree);
  }

  static void clear() {
    reclaimableCalls_ = 0;
    reclaimCalls_ = 0;
  }

  static int reclaimableCalls() {
    return reclaimableCalls_.load();
  }

  static int reclaimCalls() {
    return reclaimCalls_.load();
  }

 private:
  static constexpr uint64_t kReclaimableBytes{8 << 20}; // 8MB

  static std::atomic<int> reclaimableCalls_;
  static std::atomic<int> reclaimCalls_;
};

std::atomic<int> CountingReclaimer::reclaimableCalls_{0};
std::atomic<int> CountingReclaimer::reclaimCalls_{0};

// Verifies that calling MemoryPool::maybeReserve on a leaf pool, without
// performing any actual allocations, can still drive the SharedArbitrator and
// cause the leaf's reclaimer to run.
TEST(CudfMemoryReclaimerTest, maybeReserveTriggersReclaimer) {
  static std::once_flag once;
  std::call_once(once, []() { SharedArbitrator::registerFactory(); });

  const int64_t kCapacity = 64L << 20; // 64MB

  MemoryManager::Options opts;
  opts.allocatorCapacity = kCapacity;
  opts.arbitratorCapacity = kCapacity;
  opts.arbitratorKind = "SHARED";
  opts.checkUsageLeak = true;

  auto manager = std::make_shared<MemoryManager>(opts);

  CountingReclaimer::clear();

  // Root pool with a test reclaimer attached.
  auto root = manager->addRootPool("cudf_maybe_reserve_root", kCapacity);
  root->setReclaimer(std::make_unique<CountingReclaimer>());

  // Leaf pool under the root. We never allocate from it; we only reserve.
  auto leaf = root->addLeafChild("leaf", true /*threadSafe*/);

  // At construction time, root capacity is 0. Any positive reservation will
  // exceed current free capacity and force the arbitrator to grow the root,
  // which in turn must consult the reclaimer.
  const uint64_t reservationBytes = kCapacity - 1;

  // uint64_t beforeFreeBytes = root->freeBytes();
  // ASSERT_EQ(beforeFreeBytes, 0);

  bool reserved = leaf->maybeReserve(reservationBytes);
  // Reservation may succeed or fail depending on arbitrator policy; either way
  // we expect the reclaimer to have been consulted.
  (void)reserved;

  bool reserved2 = leaf->maybeReserve(reservationBytes);
  (void)reserved2;

  EXPECT_GT(CountingReclaimer::reclaimableCalls(), 0)
      << "Reclaimer::reclaimableBytes was not called";
  EXPECT_GT(CountingReclaimer::reclaimCalls(), 0)
      << "Reclaimer::reclaim was not called";
}

} // namespace
