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

#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfPiecewiseSpillHashJoin.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/exec/Task.h"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cstring>

namespace facebook::velox::cudf_velox {

namespace {

constexpr auto kOobPolicy = cudf::out_of_bounds_policy::NULLIFY;

/// Returns true if the join key has been validated as a simple inner equi
/// join with no filter. The piecewise variant only supports this case.
bool isSimpleInnerEquiJoin(const core::HashJoinNode& joinNode) {
  return joinNode.isInnerJoin() && joinNode.filter() == nullptr;
}

} // namespace

// =============================================================================
// PinnedHostBuffer
// =============================================================================

PinnedHostBuffer::PinnedHostBuffer(std::size_t bytes) {
  if (bytes == 0) {
    return;
  }
  void* ptr = nullptr;
  auto const err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
  VELOX_CHECK(
      err == cudaSuccess,
      "cudaHostAlloc failed for pinned host buffer of {} bytes: {}",
      bytes,
      cudaGetErrorString(err));
  data_ = ptr;
  size_ = bytes;
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(
    PinnedHostBuffer&& other) noexcept {
  if (this != &other) {
    if (data_ != nullptr) {
      cudaFreeHost(data_);
    }
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

PinnedHostBuffer::~PinnedHostBuffer() {
  if (data_ != nullptr) {
    cudaFreeHost(data_);
  }
}

// =============================================================================
// PiecewiseSpillStore
// =============================================================================

PiecewiseSpillStore& PiecewiseSpillStore::getInstance() {
  static PiecewiseSpillStore instance;
  return instance;
}

void PiecewiseSpillStore::put(
    const core::PlanNodeId& planNodeId,
    std::shared_ptr<std::vector<HostBuildPiece>> pieces,
    RowTypePtr buildRowType,
    InitialResidentPiece initialResident) {
  std::lock_guard<std::mutex> l(mutex_);
  auto& entry = entries_[planNodeId];
  VELOX_CHECK(
      entry.pieces == nullptr,
      "PiecewiseSpillStore already has pieces for plan node: {}",
      planNodeId);
  entry.pieces = std::move(pieces);
  entry.buildRowType = std::move(buildRowType);
  entry.initialResident = std::move(initialResident);
  entry.initialResidentTaken = false;
}

std::pair<std::shared_ptr<std::vector<HostBuildPiece>>, RowTypePtr>
PiecewiseSpillStore::get(const core::PlanNodeId& planNodeId) {
  std::lock_guard<std::mutex> l(mutex_);
  auto it = entries_.find(planNodeId);
  if (it == entries_.end()) {
    return {nullptr, nullptr};
  }
  return {it->second.pieces, it->second.buildRowType};
}

InitialResidentPiece PiecewiseSpillStore::takeInitialResident(
    const core::PlanNodeId& planNodeId) {
  std::lock_guard<std::mutex> l(mutex_);
  auto it = entries_.find(planNodeId);
  if (it == entries_.end() || it->second.initialResidentTaken) {
    return {};
  }
  it->second.initialResidentTaken = true;
  return std::move(it->second.initialResident);
}

void PiecewiseSpillStore::erase(const core::PlanNodeId& planNodeId) {
  std::lock_guard<std::mutex> l(mutex_);
  entries_.erase(planNodeId);
}

// =============================================================================
// Build
// =============================================================================

CudfPiecewiseSpillHashJoinBuild::CudfPiecewiseSpillHashJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode,
    cudf::size_type pieceTargetRows)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          nullptr,
          joinNode->id(),
          "CudfPiecewiseSpillHashJoinBuild",
          nvtx3::rgb{65, 105, 225}, // Royal Blue
          NvtxMethodFlag::kAll,
          std::nullopt,
          joinNode),
      joinNode_(joinNode),
      pieceTargetRows_(pieceTargetRows) {
  VELOX_CHECK(
      isSimpleInnerEquiJoin(*joinNode_),
      "CudfPiecewiseSpillHashJoinBuild only supports inner equi-join "
      "with no filter");
  VELOX_CHECK_GT(pieceTargetRows_, 0);
}

bool CudfPiecewiseSpillHashJoinBuild::needsInput() const {
  return !noMoreInput_;
}

void CudfPiecewiseSpillHashJoinBuild::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  inputs_.push_back(std::move(cudfInput));
}

RowVectorPtr CudfPiecewiseSpillHashJoinBuild::doGetOutput() {
  return nullptr;
}

void CudfPiecewiseSpillHashJoinBuild::doNoMoreInput() {
  Operator::noMoreInput();

  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }
  // Collect inputs from peer drivers.
  for (auto& peer : peers) {
    auto* op = peer->findOperator(planNodeId());
    auto* build = dynamic_cast<CudfPiecewiseSpillHashJoinBuild*>(op);
    VELOX_CHECK_NOT_NULL(build);
    inputs_.insert(inputs_.end(), build->inputs_.begin(), build->inputs_.end());
  }

  SCOPE_EXIT {
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  auto buildType = joinNode_->sources()[1]->outputType();
  auto rightKeys = joinNode_->rightKeys();

  std::vector<cudf::size_type> buildKeyIndices(rightKeys.size());
  for (size_t i = 0; i < rightKeys.size(); ++i) {
    buildKeyIndices[i] = static_cast<cudf::size_type>(
        buildType->getChildIdx(rightKeys[i]->name()));
  }

  auto pieces = std::make_shared<std::vector<HostBuildPiece>>();
  InitialResidentPiece initialResident;

  auto bridge = std::dynamic_pointer_cast<CudfHashJoinBridge>(
      operatorCtx_->task()->getCustomJoinBridge(
          operatorCtx_->driverCtx()->splitGroupId, planNodeId()));
  VELOX_CHECK_NOT_NULL(bridge);

  auto publish = [&](std::shared_ptr<std::vector<HostBuildPiece>>&& p,
                     RowTypePtr&& schema,
                     InitialResidentPiece&& resident) {
    PiecewiseSpillStore::getInstance().put(
        planNodeId(),
        std::move(p),
        std::move(schema),
        std::move(resident));
    // Signal build completion via the standard CudfHashJoinBridge. The
    // probe operator ignores the payload — it pulls data from
    // PiecewiseSpillStore — but uses the bridge's promise/future for
    // synchronization, so we publish an empty (but present) hash_type.
    bridge->setHashTable(
        std::make_optional(CudfHashJoinBridge::hash_type{{}, {}}));
  };

  // Empty build side is legal — publish an empty pieces vector.
  if (inputs_.empty()) {
    publish(
        std::move(pieces), std::move(buildType), std::move(initialResident));
    return;
  }

  auto stream = cudfGlobalStreamPool().get_stream();

  // Slice incoming batches into pieces no larger than pieceTargetRows_.
  // We greedily fill each piece by concatenating consecutive incoming
  // batches; an oversized incoming batch is itself sliced.
  auto flushPiece = [&](std::vector<cudf::table_view> const& views,
                        cudf::size_type numRows) {
    if (numRows == 0) {
      return;
    }
    auto pieceTable = (views.size() == 1)
        ? std::make_unique<cudf::table>(views[0], stream, get_output_mr())
        : cudf::concatenate(views, stream, get_output_mr());
    auto pieceView = pieceTable->view();

    // Pack the entire build table for this piece. cudf::pack returns
    // contiguous device data + host metadata.
    auto packed = cudf::pack(pieceView, stream, get_output_mr());
    auto metadataCopy = *packed.metadata;
    auto const payloadSize = packed.gpu_data->size();

    // Rebuild the table_view over packed.gpu_data so any reference held
    // by the hash_join (which we may keep alive past the end of this
    // lambda for piece 0) stays valid for the lifetime of
    // packed.gpu_data.
    auto packedView = cudf::unpack(
        metadataCopy.data(),
        reinterpret_cast<std::uint8_t const*>(packed.gpu_data->data()));

    // Build the cudf::hash_join over key columns of the piece.
    auto hj = std::make_unique<cudf::hash_join>(
        packedView.select(buildKeyIndices),
        cudf::null_equality::UNEQUAL,
        stream);

    // release_storage is non-destructive (const member); it returns a
    // freshly allocated D-to-D snapshot of the cuco slot bytes, and the
    // original hash_join remains usable.
    auto storage = hj->release_storage(stream);

    HostBuildPiece piece;
    piece.numRows = pieceView.num_rows();
    piece.packedMetadata = metadataCopy;

    piece.payloadBytes = PinnedHostBuffer{payloadSize};
    auto const slotsSize = storage.slots.size();
    piece.hashTableBytes = PinnedHostBuffer{slotsSize};
    piece.hashSlotCount = storage.slot_count;
    piece.hashSlotBytes = storage.slot_bytes;
    piece.hashCompareNulls = storage.compare_nulls;
    piece.hashHasNulls = storage.has_nulls;
    piece.hashLoadFactor = storage.load_factor;

    {
      auto const err = cudaMemcpyAsync(
          piece.payloadBytes.data(),
          packed.gpu_data->data(),
          payloadSize,
          cudaMemcpyDeviceToHost,
          stream.value());
      VELOX_CHECK(
          err == cudaSuccess,
          "cudaMemcpyAsync D2H of packed payload failed: {}",
          cudaGetErrorString(err));
    }
    {
      auto const err = cudaMemcpyAsync(
          piece.hashTableBytes.data(),
          storage.slots.data(),
          slotsSize,
          cudaMemcpyDeviceToHost,
          stream.value());
      VELOX_CHECK(
          err == cudaSuccess,
          "cudaMemcpyAsync D2H of hash slots failed: {}",
          cudaGetErrorString(err));
    }

    bool const isFirstPiece = pieces->empty();
    if (isFirstPiece) {
      // Keep piece 0's device state alive so the probe operator can
      // start its first join without paying any H-to-D cost. Move out
      // packed.gpu_data (the rmm::device_buffer underlying packedView)
      // and the hash_join itself. The probe will free them when it
      // wraps past piece 0 the first time.
      initialResident.payloadDevice = std::move(*packed.gpu_data);
      initialResident.packedMetadata = std::move(metadataCopy);
      initialResident.buildTableView = packedView;
      initialResident.hj = std::move(hj);
      initialResident.pieceIdx = 0;
    } else {
      // Drop hash_join (and its cuco storage) and packed.gpu_data
      // once the D-to-H copy is scheduled. The actual free is
      // stream-ordered and runs after the copy completes.
      hj.reset();
      packed.gpu_data.reset();
    }
    pieces->push_back(std::move(piece));
  };

  std::vector<cudf::table_view> pendingViews;
  std::vector<std::unique_ptr<cudf::table>> pendingOwned;
  cudf::size_type pendingRows = 0;

  auto flushPending = [&]() {
    flushPiece(pendingViews, pendingRows);
    pendingViews.clear();
    pendingOwned.clear();
    pendingRows = 0;
  };

  for (auto& input : inputs_) {
    auto view = input->getTableView();
    auto const numRows = view.num_rows();
    cudf::size_type consumed = 0;
    while (consumed < numRows) {
      auto const remainingInPiece = pieceTargetRows_ - pendingRows;
      auto const take = std::min(numRows - consumed, remainingInPiece);
      // Slice produces a view referencing the original buffers.
      auto slice = cudf::slice(view, {consumed, consumed + take}, stream)[0];
      pendingViews.push_back(slice);
      pendingRows += take;
      consumed += take;
      if (pendingRows >= pieceTargetRows_) {
        flushPending();
      }
    }
  }
  flushPending();

  // Synchronize before publishing so all D-to-H copies have landed and
  // pinned-host buffers contain the final bytes.
  stream.synchronize();
  inputs_.clear();

  publish(
      std::move(pieces), std::move(buildType), std::move(initialResident));
}

exec::BlockingReason CudfPiecewiseSpillHashJoinBuild::isBlocked(
    ContinueFuture* future) {
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfPiecewiseSpillHashJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

// =============================================================================
// Probe
// =============================================================================

CudfPiecewiseSpillHashJoinProbe::CudfPiecewiseSpillHashJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          joinNode->outputType(),
          joinNode->id(),
          "CudfPiecewiseSpillHashJoinProbe",
          nvtx3::rgb{0, 128, 128}, // Teal
          NvtxMethodFlag::kAll,
          std::nullopt,
          joinNode),
      joinNode_(joinNode),
      probeType_(joinNode->sources()[0]->outputType()),
      buildType_(joinNode->sources()[1]->outputType()) {
  VELOX_CHECK(
      isSimpleInnerEquiJoin(*joinNode_),
      "CudfPiecewiseSpillHashJoinProbe only supports inner equi-join "
      "with no filter");

  auto const& leftKeys = joinNode_->leftKeys();
  auto const& rightKeys = joinNode_->rightKeys();
  leftKeyIndices_.resize(leftKeys.size());
  for (size_t i = 0; i < leftKeys.size(); ++i) {
    leftKeyIndices_[i] = static_cast<cudf::size_type>(
        probeType_->getChildIdx(leftKeys[i]->name()));
  }
  rightKeyIndices_.resize(rightKeys.size());
  for (size_t i = 0; i < rightKeys.size(); ++i) {
    rightKeyIndices_[i] = static_cast<cudf::size_type>(
        buildType_->getChildIdx(rightKeys[i]->name()));
  }

  // Map each output column to its source side and index.
  auto outputType = joinNode_->outputType();
  outputColumns_.reserve(outputType->size());
  for (size_t i = 0; i < outputType->size(); ++i) {
    auto const& name = outputType->nameOf(i);
    if (auto idx = probeType_->getChildIdxIfExists(name)) {
      outputColumns_.push_back(
          {true, static_cast<cudf::size_type>(idx.value())});
      continue;
    }
    if (auto idx = buildType_->getChildIdxIfExists(name)) {
      outputColumns_.push_back(
          {false, static_cast<cudf::size_type>(idx.value())});
      continue;
    }
    VELOX_FAIL("Output column not found in probe or build: {}", name);
  }
}

bool CudfPiecewiseSpillHashJoinProbe::needsInput() const {
  return !noMoreInput_ && probe_ == nullptr;
}

void CudfPiecewiseSpillHashJoinProbe::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  probe_ = std::move(cudfInput);
  piecesProcessedThisBatch_ = 0;
}

exec::BlockingReason CudfPiecewiseSpillHashJoinProbe::isBlocked(
    ContinueFuture* future) {
  if (pieces_ != nullptr) {
    return exec::BlockingReason::kNotBlocked;
  }
  auto bridge = std::dynamic_pointer_cast<CudfHashJoinBridge>(
      operatorCtx_->task()->getCustomJoinBridge(
          operatorCtx_->driverCtx()->splitGroupId, planNodeId()));
  VELOX_CHECK_NOT_NULL(bridge);
  auto signal = bridge->hashOrFuture(future);
  if (!signal.has_value()) {
    return exec::BlockingReason::kWaitForJoinBuild;
  }
  // Build has signaled completion via the bridge; pull the host pieces
  // and the initial resident piece from the spill store, then erase
  // the entry so a repeat task with the same plan-node id starts
  // clean.
  auto entry = PiecewiseSpillStore::getInstance().get(planNodeId());
  VELOX_CHECK_NOT_NULL(
      entry.first, "PiecewiseSpillStore has no entry for {}", planNodeId());
  pieces_ = std::move(entry.first);
  unpackedBuildType_ = std::move(entry.second);
  auto initialResident =
      PiecewiseSpillStore::getInstance().takeInitialResident(planNodeId());
  PiecewiseSpillStore::getInstance().erase(planNodeId());

  // Bind each slot to its own stream so successive pieces' lifecycles
  // run on independent streams. Done lazily here, not in the
  // constructor, so the streams are allocated after cuDF init.
  if (!slotsInitialized_) {
    slotA_.stream = cudfGlobalStreamPool().get_stream();
    slotB_.stream = cudfGlobalStreamPool().get_stream();
    slotsInitialized_ = true;
  }

  // Seat the initial resident piece (piece 0, GPU-resident) into
  // slot A so the first join doesn't have to pay an unspill stall.
  if (initialResident.hj != nullptr) {
    slotA_.payloadDevice = std::move(initialResident.payloadDevice);
    slotA_.buildTableView = initialResident.buildTableView;
    slotA_.hj = std::move(initialResident.hj);
    slotA_.pieceIdx = initialResident.pieceIdx;
    slotA_.valid = true;
    residentIsA_ = true;
  }
  return exec::BlockingReason::kNotBlocked;
}

void CudfPiecewiseSpillHashJoinProbe::loadPieceInto(
    PieceSlot& target,
    std::size_t pieceIdx) {
  VELOX_CHECK_LT(pieceIdx, pieces_->size());
  auto& piece = (*pieces_)[pieceIdx];
  auto stream = target.stream;

  // Replace any prior tenant. rmm::device_buffer move-assignment is
  // stream-ordered on the destination stream's allocator, so the old
  // contents' free is queued on `stream` before the new allocation.
  target.hj.reset();
  target.payloadDevice =
      rmm::device_buffer{piece.payloadBytes.size(), stream};
  {
    auto const err = cudaMemcpyAsync(
        target.payloadDevice.data(),
        piece.payloadBytes.data(),
        piece.payloadBytes.size(),
        cudaMemcpyHostToDevice,
        stream.value());
    VELOX_CHECK(
        err == cudaSuccess,
        "cudaMemcpyAsync H2D of packed payload failed: {}",
        cudaGetErrorString(err));
  }

  cudf::hash_join_storage storage;
  storage.slots = rmm::device_buffer{piece.hashTableBytes.size(), stream};
  storage.slot_count = piece.hashSlotCount;
  storage.slot_bytes = piece.hashSlotBytes;
  storage.compare_nulls = piece.hashCompareNulls;
  storage.has_nulls = piece.hashHasNulls;
  storage.load_factor = piece.hashLoadFactor;
  {
    auto const err = cudaMemcpyAsync(
        storage.slots.data(),
        piece.hashTableBytes.data(),
        piece.hashTableBytes.size(),
        cudaMemcpyHostToDevice,
        stream.value());
    VELOX_CHECK(
        err == cudaSuccess,
        "cudaMemcpyAsync H2D of hash slots failed: {}",
        cudaGetErrorString(err));
  }

  // unpack is a pure CPU operation that builds column_views over the
  // device-resident packed bytes.
  target.buildTableView = cudf::unpack(
      piece.packedMetadata.data(),
      reinterpret_cast<std::uint8_t const*>(target.payloadDevice.data()));

  target.hj = cudf::hash_join::from_storage(
      std::move(storage),
      target.buildTableView.select(rightKeyIndices_),
      stream);
  target.pieceIdx = pieceIdx;
  target.valid = true;
}

std::unique_ptr<cudf::table> CudfPiecewiseSpillHashJoinProbe::runJoinOnSlot(
    PieceSlot const& slot) {
  auto stream = slot.stream;
  auto probeView = probe_->getTableView();
  auto [leftIdx, rightIdx] = slot.hj->inner_join(
      probeView.select(leftKeyIndices_),
      std::nullopt,
      stream,
      get_temp_mr());

  cudf::column_view leftIdxCol{
      cudf::device_span<cudf::size_type const>{*leftIdx}};
  cudf::column_view rightIdxCol{
      cudf::device_span<cudf::size_type const>{*rightIdx}};

  std::vector<cudf::size_type> leftGather;
  std::vector<cudf::size_type> rightGather;
  std::vector<size_t> leftOutputAt;
  std::vector<size_t> rightOutputAt;
  for (size_t i = 0; i < outputColumns_.size(); ++i) {
    if (outputColumns_[i].fromLeft) {
      leftGather.push_back(outputColumns_[i].sourceIndex);
      leftOutputAt.push_back(i);
    } else {
      rightGather.push_back(outputColumns_[i].sourceIndex);
      rightOutputAt.push_back(i);
    }
  }

  std::vector<std::unique_ptr<cudf::column>> joined(outputColumns_.size());
  if (!leftGather.empty()) {
    auto gathered = cudf::gather(
        probeView.select(leftGather),
        leftIdxCol,
        kOobPolicy,
        stream,
        get_output_mr());
    auto cols = gathered->release();
    for (size_t i = 0; i < cols.size(); ++i) {
      joined[leftOutputAt[i]] = std::move(cols[i]);
    }
  }
  if (!rightGather.empty()) {
    auto gathered = cudf::gather(
        slot.buildTableView.select(rightGather),
        rightIdxCol,
        kOobPolicy,
        stream,
        get_output_mr());
    auto cols = gathered->release();
    for (size_t i = 0; i < cols.size(); ++i) {
      joined[rightOutputAt[i]] = std::move(cols[i]);
    }
  }

  auto outputTable = std::make_unique<cudf::table>(std::move(joined));
  if (outputTable->num_rows() == 0) {
    return nullptr;
  }
  return outputTable;
}

RowVectorPtr CudfPiecewiseSpillHashJoinProbe::doGetOutput() {
  if (probe_ == nullptr) {
    if (noMoreInput_) {
      finished_ = true;
    }
    return nullptr;
  }
  if (pieces_ == nullptr) {
    return nullptr;
  }
  // Empty build side → no output rows ever.
  if (pieces_->empty()) {
    probe_.reset();
    return nullptr;
  }

  // Walk pieces until we find one that produces rows, or exhaust them.
  // Each iteration: (1) launch H-to-D + from_storage of the next piece
  // into the OTHER slot on its stream; (2) launch inner_join + gather
  // on the resident slot's stream; (3) sync resident stream then next
  // stream; (4) swap slots so the freshly-loaded piece becomes
  // resident.
  auto const numPieces = pieces_->size();
  while (piecesProcessedThisBatch_ < numPieces) {
    auto& resident = residentIsA_ ? slotA_ : slotB_;
    auto& next = residentIsA_ ? slotB_ : slotA_;
    VELOX_CHECK(
        resident.valid,
        "Resident slot is empty in piecewise probe");

    bool prefetched = false;
    if (numPieces > 1) {
      auto const nextIdx = (resident.pieceIdx + 1) % numPieces;
      loadPieceInto(next, nextIdx);
      prefetched = true;
    }

    auto const joinStreamUsed = resident.stream;
    auto outputTable = runJoinOnSlot(resident);

    // Sync the resident's stream so the output table is safe to wrap.
    // The other slot's stream keeps running its H-to-D in parallel.
    resident.stream.synchronize();

    if (prefetched) {
      // Wait for the prefetch to finish before promoting it; usually
      // near-instant because it overlapped the join.
      next.stream.synchronize();
      residentIsA_ = !residentIsA_;
    }

    ++piecesProcessedThisBatch_;

    if (outputTable != nullptr) {
      auto const numRows =
          static_cast<vector_size_t>(outputTable->num_rows());
      auto pool = operatorCtx_->pool();
      return std::make_shared<CudfVector>(
          pool,
          outputType_,
          numRows,
          std::move(outputTable),
          joinStreamUsed);
    }
  }
  probe_.reset();
  piecesProcessedThisBatch_ = 0;
  return nullptr;
}

void CudfPiecewiseSpillHashJoinProbe::doNoMoreInput() {
  Operator::noMoreInput();
}

void CudfPiecewiseSpillHashJoinProbe::doClose() {
  Operator::close();
  // Drain both slot streams so any in-flight prefetches / joins
  // finish before we destroy the slot state (which issues
  // stream-ordered frees of its device buffers).
  if (slotsInitialized_) {
    slotA_.stream.synchronize();
    slotB_.stream.synchronize();
  }
  slotA_.hj.reset();
  slotB_.hj.reset();
  slotA_.payloadDevice = rmm::device_buffer{};
  slotB_.payloadDevice = rmm::device_buffer{};
  slotA_.valid = false;
  slotB_.valid = false;
  pieces_.reset();
  probe_.reset();
}

bool CudfPiecewiseSpillHashJoinProbe::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
