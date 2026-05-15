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
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"

#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Benchmark-only partitioned hash join build operator.
///
/// The build side is hash partitioned by the join key using cuDF's
/// HASH_MURMUR3 and DEFAULT_HASH_SEED. Each partition stays resident on GPU and
/// gets its own cudf::hash_join. The partition tables and hash joins are
/// published via the existing CudfHashJoinBridge.
class CudfPartitionedHashJoinBuild : public CudfOperatorBase {
 public:
  CudfPartitionedHashJoinBuild(
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

 private:
  std::shared_ptr<const core::HashJoinNode> joinNode_;
  int32_t const numPartitions_;
  std::vector<CudfVectorPtr> inputs_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

/// Benchmark-only partitioned hash join probe operator.
///
/// Each probe batch is hash partitioned with the same settings as the build
/// side, then partition i is joined only with build partition i.
class CudfPartitionedHashJoinProbe : public CudfOperatorBase {
 public:
  using hash_type = CudfHashJoinBridge::hash_type;

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

  std::unique_ptr<cudf::table> joinPartition(
      cudf::table_view probePartition,
      const std::shared_ptr<cudf::table>& buildPartition,
      const std::shared_ptr<cudf::hash_join>& hashJoin,
      rmm::cuda_stream_view stream);

  std::shared_ptr<const core::HashJoinNode> joinNode_;
  RowTypePtr probeType_;
  RowTypePtr buildType_;
  int32_t const numPartitions_;

  std::optional<hash_type> hashObject_;

  std::vector<cudf::size_type> leftKeyIndices_;
  std::vector<cudf::size_type> rightKeyIndices_;
  std::vector<OutputColumnSource> outputColumns_;

  CudfVectorPtr probe_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
  bool finished_{false};
};

} // namespace facebook::velox::cudf_velox
