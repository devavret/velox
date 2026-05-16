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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox {

/// One spilled cuDF table piece. Unlike HostBuildPiece, this stores only
/// cudf::pack(table) payload bytes and metadata; no hash-map storage is kept.
struct CudfProbeSpillTablePiece {
  std::vector<std::uint8_t> packedMetadata;
  SpillHostBuffer payloadBytes;
  cudf::size_type numRows{0};
};

using CudfProbeSpillTablePartitions =
    std::vector<std::vector<CudfProbeSpillTablePiece>>;

/// Benchmark-only build operator for the partitioned probe-spill study.
///
/// Build rows are hash-partitioned by the join key and immediately spilled to
/// host memory as packed cuDF table bytes. No cudf::hash_join is constructed
/// during build.
class CudfPartitionedProbeSpillHashJoinBuild : public CudfOperatorBase {
 public:
  CudfPartitionedProbeSpillHashJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode,
      int32_t numPartitions,
      bool usePinnedHostMemory);

  bool needsInput() const override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;

 private:
  CudfProbeSpillTablePiece spillTableView(
      cudf::table_view tableView,
      rmm::cuda_stream_view stream) const;

  std::shared_ptr<const core::HashJoinNode> joinNode_;
  int32_t const numPartitions_;
  SpillHostMemoryKind const spillHostMemoryKind_;
  std::vector<CudfVectorPtr> inputs_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

/// Benchmark-only probe operator for the partitioned probe-spill study.
///
/// Probe batches are hash-partitioned and spilled into per-partition lists.
/// Once noMoreInput is received, one build partition is restored, a
/// cudf::hash_join is constructed, all matching probe pieces are restored and
/// joined, then the build partition is released.
class CudfPartitionedProbeSpillHashJoinProbe : public CudfOperatorBase {
 public:
  CudfPartitionedProbeSpillHashJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode,
      int32_t numPartitions,
      bool usePinnedHostMemory);

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

  struct DeviceTableView {
    rmm::device_buffer payloadDevice;
    cudf::table_view tableView;
  };

  struct BuildPartitionSlot {
    std::vector<rmm::device_buffer> payloadDevices;
    std::unique_ptr<cudf::table> table;
    cudf::table_view tableView;
    std::unique_ptr<cudf::hash_join> hashJoin;
    std::size_t partitionIdx{0};
    rmm::cuda_stream_view stream{};
    bool valid{false};
  };

  CudfProbeSpillTablePiece spillTableView(
      cudf::table_view tableView,
      rmm::cuda_stream_view stream) const;

  DeviceTableView loadTablePiece(
      const CudfProbeSpillTablePiece& piece,
      rmm::cuda_stream_view stream) const;

  void loadBuildPartitionInto(
      BuildPartitionSlot& slot,
      std::size_t partitionIdx);

  folly::Future<folly::Unit> makeLoadFuture(
      BuildPartitionSlot& slot,
      std::size_t partitionIdx);

  std::unique_ptr<cudf::table> joinProbePiece(
      const CudfProbeSpillTablePiece& probePiece,
      const BuildPartitionSlot& buildSlot);

  std::optional<std::size_t> nextJoinablePartition(std::size_t start) const;

  void maybeStartCurrentPartition();
  void maybeStartNextPartitionPrefetch();
  void advanceToPrefetchedPartition();
  void finishCurrentPartition();
  void resetSlot(BuildPartitionSlot& slot);
  void releaseSpillStore();

  std::shared_ptr<const core::HashJoinNode> joinNode_;
  RowTypePtr probeType_;
  RowTypePtr buildType_;
  int32_t const numPartitions_;
  SpillHostMemoryKind const spillHostMemoryKind_;

  std::vector<cudf::size_type> leftKeyIndices_;
  std::vector<cudf::size_type> rightKeyIndices_;
  std::vector<OutputColumnSource> outputColumns_;

  std::shared_ptr<CudfProbeSpillTablePartitions> buildPartitions_;
  CudfProbeSpillTablePartitions probePartitions_;
  bool spillStoreAcquired_{false};

  BuildPartitionSlot slotA_;
  BuildPartitionSlot slotB_;
  bool slotsInitialized_{false};
  bool residentIsA_{true};

  std::optional<std::size_t> currentPartition_;
  std::size_t currentProbePiece_{0};
  std::optional<folly::Future<folly::Unit>> currentLoadFuture_;
  std::optional<folly::Future<folly::Unit>> prefetchFuture_;
  std::optional<std::size_t> prefetchedPartition_;

  ContinueFuture future_{ContinueFuture::makeEmpty()};
  bool finished_{false};
};

} // namespace facebook::velox::cudf_velox
