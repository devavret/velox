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

#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/CudfPiecewiseSpillHashJoin.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"

#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <folly/Unit.h>
#include <folly/futures/Future.h>

#include <memory>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Benchmark-only partitioned hash join build operator.
///
/// The build side is hash partitioned by the join key using cuDF's
/// HASH_MURMUR3 and DEFAULT_HASH_SEED. Each partition gets a cudf::hash_join,
/// then its payload and hash table are copied into host memory and the
/// GPU-resident state is released.
class CudfPartitionedHashJoinBuild : public CudfOperatorBase {
 public:
  CudfPartitionedHashJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode,
      int32_t numPartitions,
      SpillHostMemoryKind spillHostMemoryKind);

  bool needsInput() const override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;

 private:
  std::shared_ptr<const core::HashJoinNode> joinNode_;
  int32_t const numPartitions_;
  SpillHostMemoryKind const spillHostMemoryKind_;
  std::vector<CudfVectorPtr> inputs_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

/// Benchmark-only partitioned hash join probe operator.
///
/// Each probe batch is hash partitioned with the same settings as the build
/// side, then partition i is joined only with build partition i.
class CudfPartitionedHashJoinProbe : public CudfOperatorBase {
 public:
  CudfPartitionedHashJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode,
      int32_t numPartitions);

  bool needsInput() const override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  struct OutputColumnSource {
    bool fromLeft;
    cudf::size_type sourceIndex;
  };

  struct PartitionSlot {
    rmm::device_buffer payloadDevice;
    cudf::table_view buildTableView;
    std::unique_ptr<cudf::hash_join> hj;
    std::size_t partitionIdx{0};
    rmm::cuda_stream_view stream{};
    bool valid{false};
  };

  void loadPartitionInto(PartitionSlot& target, std::size_t partitionIdx);

  folly::Future<folly::Unit> makeLoadFuture(
      PartitionSlot& slot,
      std::size_t partitionIdx);

  std::unique_ptr<cudf::table> joinPartition(
      cudf::table_view probePartition,
      const PartitionSlot& buildSlot,
      rmm::cuda_stream_view probeStream);

  std::optional<std::size_t> nextBuildPartition(std::size_t start) const;
  bool hasActiveProbe() const;
  void initializeProbeBatch();
  void resetProbeBatch();
  void releaseSpillStore();

  std::shared_ptr<const core::HashJoinNode> joinNode_;
  RowTypePtr probeType_;
  RowTypePtr buildType_;
  int32_t const numPartitions_;

  std::vector<cudf::size_type> leftKeyIndices_;
  std::vector<cudf::size_type> rightKeyIndices_;
  std::vector<OutputColumnSource> outputColumns_;

  std::shared_ptr<std::vector<HostBuildPiece>> pieces_;
  bool spillStoreAcquired_{false};

  PartitionSlot slotA_;
  PartitionSlot slotB_;
  bool slotsInitialized_{false};

  CudfVectorPtr probe_;
  std::unique_ptr<cudf::table> partitionedProbe_;
  std::vector<cudf::table_view> probePartitions_;
  rmm::cuda_stream_view probeStream_{};
  std::optional<std::size_t> currentPartition_;
  std::optional<folly::Future<folly::Unit>> loadFuture_;
  bool residentIsA_{true};
  ContinueFuture future_{ContinueFuture::makeEmpty()};
  bool finished_{false};
};

} // namespace facebook::velox::cudf_velox
