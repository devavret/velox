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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"

#include <cudf/contiguous_split.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/hash_join_storage.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox {

/// RAII wrapper around a pinned (page-locked) host allocation backed by
/// `cudaHostAlloc`. Move-only.
class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;
  explicit PinnedHostBuffer(std::size_t bytes);

  PinnedHostBuffer(PinnedHostBuffer const&) = delete;
  PinnedHostBuffer& operator=(PinnedHostBuffer const&) = delete;

  PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
  PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

  ~PinnedHostBuffer();

  [[nodiscard]] void* data() const noexcept {
    return data_;
  }
  [[nodiscard]] std::size_t size() const noexcept {
    return size_;
  }

 private:
  void* data_{nullptr};
  std::size_t size_{0};
};

/// One spilled build piece, fully resident in host memory.
///
/// Contains both the packed build-table payload and the cuco hash-table
/// slot bytes for that piece, plus the metadata needed to reconstruct
/// each on device.
struct HostBuildPiece {
  /// `cudf::pack` metadata for `payloadBytes`. Host-resident; small.
  std::vector<std::uint8_t> packedMetadata;
  /// Pinned-host bytes for the packed build table payload (the
  /// `packed_columns::gpu_data` bytes after D-to-H).
  PinnedHostBuffer payloadBytes;
  /// Pinned-host bytes for the cuco hash-table slot storage.
  PinnedHostBuffer hashTableBytes;
  /// Number of slots in the hash table.
  std::size_t hashSlotCount{0};
  /// Size of one slot in bytes.
  std::size_t hashSlotBytes{0};
  /// Whether nulls compared equal during build.
  cudf::null_equality hashCompareNulls{cudf::null_equality::UNEQUAL};
  /// Whether the build/probe inputs may contain nulls in join keys.
  bool hashHasNulls{true};
  /// Load factor used to build the cuco multiset.
  double hashLoadFactor{0.5};
  /// Number of rows in the build piece.
  cudf::size_type numRows{0};
};

/// Process-wide store keyed by plan-node ID that ships the host-resident
/// build pieces from the build operator to the probe operator.
///
/// Velox's existing `CudfHashJoinBridge` is reused only for build/probe
/// synchronization (its hash_type payload is ignored by the piecewise
/// operators); the pieces themselves flow through this store. This keeps
/// the piecewise feature self-contained in the benchmark and avoids
/// registering a custom `PlanNodeTranslator`.
class PiecewiseSpillStore {
 public:
  static PiecewiseSpillStore& getInstance();

  /// Publishes pieces for `planNodeId`. May be called once per task per
  /// plan-node id (multiple build drivers cooperate to publish exactly
  /// once — see the last-driver barrier in
  /// `CudfPiecewiseSpillHashJoinBuild::doNoMoreInput`).
  void put(
      const core::PlanNodeId& planNodeId,
      std::shared_ptr<std::vector<HostBuildPiece>> pieces,
      RowTypePtr buildRowType);

  /// Returns the pieces for `planNodeId` without removing them. Safe to
  /// call from multiple probe drivers concurrently. Returns
  /// (nullptr, nullptr) if nothing was published for that id.
  std::pair<std::shared_ptr<std::vector<HostBuildPiece>>, RowTypePtr> get(
      const core::PlanNodeId& planNodeId);

  /// Removes the entry for `planNodeId`. Called once per task at probe
  /// completion (by the last probe driver) to release host memory.
  void erase(const core::PlanNodeId& planNodeId);

 private:
  PiecewiseSpillStore() = default;
  struct Entry {
    std::shared_ptr<std::vector<HostBuildPiece>> pieces;
    RowTypePtr buildRowType;
  };
  std::mutex mutex_;
  std::unordered_map<std::string, Entry> entries_;
};

/// Build operator for the piecewise host-spill hash join.
///
/// Accumulates incoming GPU build batches, slices them into pieces of at
/// most `pieceTargetRows_` rows, and for each piece:
///   1. Builds a `cudf::hash_join` on the join keys.
///   2. Calls `release_storage` to obtain the cuco hash-table slot bytes
///      as an `rmm::device_buffer`.
///   3. Packs the piece's full payload (key + payload columns) via
///      `cudf::pack` to obtain a single contiguous device buffer plus
///      host metadata.
///   4. Allocates pinned-host destination buffers and `cudaMemcpyAsync`
///      D-to-H copies the slot bytes and packed payload bytes.
///   5. Drops the device-resident state.
/// After all pieces are spilled, publishes the vector to the bridge.
class CudfPiecewiseSpillHashJoinBuild : public CudfOperatorBase {
 public:
  CudfPiecewiseSpillHashJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode,
      cudf::size_type pieceTargetRows);

  bool needsInput() const override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;

 private:
  std::shared_ptr<const core::HashJoinNode> joinNode_;
  cudf::size_type const pieceTargetRows_;
  std::vector<CudfVectorPtr> inputs_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

/// Probe operator for the piecewise host-spill hash join.
///
/// For each incoming probe batch, iterates over all build pieces. Each
/// piece is unspilled to device just-in-time, used to construct a fresh
/// `cudf::hash_join` via `cudf::hash_join::from_storage`, then joined and
/// emitted. v1 is sequential — no double-buffer pipelining yet.
class CudfPiecewiseSpillHashJoinProbe : public CudfOperatorBase {
 public:
  CudfPiecewiseSpillHashJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode);

  bool needsInput() const override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  /// Loads piece `pieceIdx_` to device, builds a hash_join, runs
  /// inner_join with the cached probe table, gathers the output, and
  /// returns it as a CudfVector. Releases all transient device state
  /// before returning.
  RowVectorPtr joinAgainstPiece(std::size_t pieceIdx, rmm::cuda_stream_view stream);

  std::shared_ptr<const core::HashJoinNode> joinNode_;
  RowTypePtr probeType_;
  RowTypePtr buildType_;

  /// Column indices into the probe table for the join keys.
  std::vector<cudf::size_type> leftKeyIndices_;
  /// Column indices into the build table for the join keys.
  std::vector<cudf::size_type> rightKeyIndices_;

  /// Output gather plan: for each output column, whether to read from
  /// the probe ("left") or build ("right") table and the source column
  /// index within that table.
  struct OutputColumnSource {
    bool fromLeft;
    cudf::size_type sourceIndex;
  };
  std::vector<OutputColumnSource> outputColumns_;

  /// Pieces shared from the bridge. Read-only after build completion.
  std::shared_ptr<std::vector<HostBuildPiece>> pieces_;
  /// Build-side row schema (used to feed `cudf::unpack` callers and to
  /// rebuild a CudfVector wrapper).
  RowTypePtr unpackedBuildType_;

  /// Cached current probe batch. Set in addInput, cleared after the
  /// last piece is joined.
  CudfVectorPtr probe_;
  /// Index of the next piece to join against the current probe batch.
  std::size_t pieceIdx_{0};

  ContinueFuture future_{ContinueFuture::makeEmpty()};
  bool finished_{false};
};

}  // namespace facebook::velox::cudf_velox
