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
    RowTypePtr buildRowType) {
  std::lock_guard<std::mutex> l(mutex_);
  auto& entry = entries_[planNodeId];
  VELOX_CHECK(
      entry.pieces == nullptr,
      "PiecewiseSpillStore already has pieces for plan node: {}",
      planNodeId);
  entry.pieces = std::move(pieces);
  entry.buildRowType = std::move(buildRowType);
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

  auto bridge = std::dynamic_pointer_cast<CudfHashJoinBridge>(
      operatorCtx_->task()->getCustomJoinBridge(
          operatorCtx_->driverCtx()->splitGroupId, planNodeId()));
  VELOX_CHECK_NOT_NULL(bridge);

  auto publish = [&](std::shared_ptr<std::vector<HostBuildPiece>>&& p,
                     RowTypePtr&& schema) {
    PiecewiseSpillStore::getInstance().put(
        planNodeId(), std::move(p), std::move(schema));
    // Signal build completion via the standard CudfHashJoinBridge. The
    // probe operator ignores the payload — it pulls data from
    // PiecewiseSpillStore — but uses the bridge's promise/future for
    // synchronization, so we publish an empty (but present) hash_type.
    bridge->setHashTable(
        std::make_optional(CudfHashJoinBridge::hash_type{{}, {}}));
  };

  // Empty build side is legal — publish an empty pieces vector.
  if (inputs_.empty()) {
    publish(std::move(pieces), std::move(buildType));
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

    // Build the cudf::hash_join over key columns of the piece.
    auto hj = std::make_shared<cudf::hash_join>(
        pieceView.select(buildKeyIndices),
        cudf::null_equality::UNEQUAL,
        stream);

    auto storage = hj->release_storage(stream);

    HostBuildPiece piece;
    piece.numRows = pieceView.num_rows();
    piece.packedMetadata = std::move(*packed.metadata);

    auto const payloadSize = packed.gpu_data->size();
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
    // Drop hash_join (and its cuco storage) once the D-to-H copy is
    // scheduled. The actual free is stream-ordered and runs after the
    // copy completes.
    hj.reset();
    packed.gpu_data.reset();
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

  publish(std::move(pieces), std::move(buildType));
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
  pieceIdx_ = 0;
  nextPrefetchIdx_ = 0;

  // Lazily allocate the two pipeline streams on the very first probe
  // input. Doing this here (rather than in the constructor) keeps
  // CUDA-context-touching code off the velox operator construction
  // path, which can run before cuDF initialization on some setups.
  if (!joinStream_.has_value()) {
    joinStream_ = cudfGlobalStreamPool().get_stream();
    loadStream_ = cudfGlobalStreamPool().get_stream();
  }

  // Queue the first round of prefetches so the first getOutput()
  // call doesn't pay the full unspill latency.
  topUpPrefetches();
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
  // Build has signaled completion via the bridge; pull the real pieces
  // from the spill store and immediately erase the store entry. The
  // probe holds its own shared_ptr to the pieces vector, so the data
  // stays alive for the lifetime of this operator while freeing the
  // store slot for future benchmark repetitions that reuse plan-node
  // ids.
  auto entry = PiecewiseSpillStore::getInstance().get(planNodeId());
  VELOX_CHECK_NOT_NULL(
      entry.first, "PiecewiseSpillStore has no entry for {}", planNodeId());
  pieces_ = std::move(entry.first);
  unpackedBuildType_ = std::move(entry.second);
  PiecewiseSpillStore::getInstance().erase(planNodeId());
  return exec::BlockingReason::kNotBlocked;
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
    pieceIdx_ = 0;
    return nullptr;
  }
  // Walk pieces until we find one that produces rows, or exhaust them.
  // Pieces with no matching keys produce empty join output; velox requires
  // getOutput() to return nullptr or a non-empty vector.
  while (pieceIdx_ < pieces_->size()) {
    auto out = joinAgainstNextPiece();
    ++pieceIdx_;
    if (out != nullptr) {
      return out;
    }
  }
  probe_.reset();
  pieceIdx_ = 0;
  nextPrefetchIdx_ = 0;
  return nullptr;
}

void CudfPiecewiseSpillHashJoinProbe::topUpPrefetches() {
  while (inFlight_.size() < kPrefetchDepth &&
         nextPrefetchIdx_ < pieces_->size()) {
    prefetchPiece(nextPrefetchIdx_);
    ++nextPrefetchIdx_;
  }
}

void CudfPiecewiseSpillHashJoinProbe::prefetchPiece(std::size_t pieceIdx) {
  VELOX_CHECK_LT(pieceIdx, pieces_->size());
  auto& piece = (*pieces_)[pieceIdx];
  auto loadStream = loadStream_.value();

  PrefetchedPiece in;
  in.pieceIdx = pieceIdx;
  in.ready = std::make_unique<CudaEvent>(cudaEventDisableTiming);

  in.payloadDevice =
      rmm::device_buffer{piece.payloadBytes.size(), loadStream};
  {
    auto const err = cudaMemcpyAsync(
        in.payloadDevice.data(),
        piece.payloadBytes.data(),
        piece.payloadBytes.size(),
        cudaMemcpyHostToDevice,
        loadStream.value());
    VELOX_CHECK(
        err == cudaSuccess,
        "cudaMemcpyAsync H2D of packed payload failed: {}",
        cudaGetErrorString(err));
  }

  cudf::hash_join_storage storage;
  storage.slots = rmm::device_buffer{piece.hashTableBytes.size(), loadStream};
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
        loadStream.value());
    VELOX_CHECK(
        err == cudaSuccess,
        "cudaMemcpyAsync H2D of hash slots failed: {}",
        cudaGetErrorString(err));
  }

  // unpack is a pure CPU operation that builds column_views over the
  // device-resident packed bytes; safe to invoke before the H-to-D
  // copy has completed because the views only reference the address.
  in.buildTableView = cudf::unpack(
      piece.packedMetadata.data(),
      reinterpret_cast<std::uint8_t const*>(in.payloadDevice.data()));

  // from_storage does one stream-ordered D-to-D copy of the slot
  // bytes into a freshly-allocated cuco multiset on `loadStream`.
  in.hj = cudf::hash_join::from_storage(
      std::move(storage),
      in.buildTableView.select(rightKeyIndices_),
      loadStream);

  in.ready->recordFrom(loadStream);
  inFlight_.push_back(std::move(in));
}

RowVectorPtr CudfPiecewiseSpillHashJoinProbe::joinAgainstNextPiece() {
  VELOX_CHECK(!inFlight_.empty());
  VELOX_CHECK_EQ(
      inFlight_.front().pieceIdx,
      pieceIdx_,
      "Front of in-flight queue is piece {}, expected {}",
      inFlight_.front().pieceIdx,
      pieceIdx_);

  auto piece = std::move(inFlight_.front());
  inFlight_.pop_front();

  // Top up the prefetch queue BEFORE we submit the join so the next
  // H-to-D copy is queued on loadStream ahead of the join's wait on
  // the current piece's `ready` event. That way the H-to-D copy and
  // the next inner_join can overlap on their respective streams.
  topUpPrefetches();

  auto joinStream = joinStream_.value();

  // Wait for the current piece's H-to-D + from_storage to complete.
  piece.ready->waitOn(joinStream);

  auto probeView = probe_->getTableView();
  auto [leftIdx, rightIdx] = piece.hj->inner_join(
      probeView.select(leftKeyIndices_),
      std::nullopt,
      joinStream,
      get_temp_mr());

  cudf::column_view leftIdxCol{
      cudf::device_span<cudf::size_type const>{*leftIdx}};
  cudf::column_view rightIdxCol{
      cudf::device_span<cudf::size_type const>{*rightIdx}};

  // Gather just the columns we actually output.
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
        joinStream,
        get_output_mr());
    auto cols = gathered->release();
    for (size_t i = 0; i < cols.size(); ++i) {
      joined[leftOutputAt[i]] = std::move(cols[i]);
    }
  }
  if (!rightGather.empty()) {
    auto gathered = cudf::gather(
        piece.buildTableView.select(rightGather),
        rightIdxCol,
        kOobPolicy,
        joinStream,
        get_output_mr());
    auto cols = gathered->release();
    for (size_t i = 0; i < cols.size(); ++i) {
      joined[rightOutputAt[i]] = std::move(cols[i]);
    }
  }

  // Sync the join stream so the returned table is safe to wrap into a
  // CudfVector and hand to downstream. The load stream continues to
  // run in parallel — it isn't gated by this sync. The current
  // piece's device buffers are freed when `piece` goes out of scope
  // below; the rmm::device_buffer destructors issue stream-ordered
  // frees on loadStream, which is fine because joinStream is fully
  // drained at this point.
  joinStream.synchronize();

  auto outputTable = std::make_unique<cudf::table>(std::move(joined));
  auto const numRows = static_cast<vector_size_t>(outputTable->num_rows());
  if (numRows == 0) {
    return nullptr;
  }

  auto pool = operatorCtx_->pool();
  return std::make_shared<CudfVector>(
      pool, outputType_, numRows, std::move(outputTable), joinStream);
}

void CudfPiecewiseSpillHashJoinProbe::doNoMoreInput() {
  Operator::noMoreInput();
}

void CudfPiecewiseSpillHashJoinProbe::doClose() {
  Operator::close();
  // Drain both streams so any in-flight prefetches / joins finish
  // before we destroy the in-flight queue entries (which issue
  // stream-ordered frees of their device buffers).
  if (joinStream_.has_value()) {
    joinStream_->synchronize();
  }
  if (loadStream_.has_value()) {
    loadStream_->synchronize();
  }
  inFlight_.clear();
  pieces_.reset();
  probe_.reset();
}

bool CudfPiecewiseSpillHashJoinProbe::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
