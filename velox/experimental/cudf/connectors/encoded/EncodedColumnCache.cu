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

#include "velox/experimental/cudf/connectors/encoded/EncodedColumnCache.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <cuda/std/algorithm>
#include <cuda/std/limits>

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>

namespace facebook::velox::cudf_velox::connector::encoded {

namespace {

constexpr int kThreads = 256;

// This file is compiled by nvcc, which cannot parse Velox's folly-based
// headers on aarch64, so errors are reported with standard exceptions.
void check(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

int64_t numBlocks(int64_t n) {
  return std::max<int64_t>(1, (n + kThreads - 1) / kThreads);
}

void checkCuda(cudaError_t error, const char* what) {
  check(
      error == cudaSuccess,
      std::string(what) + " failed: " + cudaGetErrorString(error));
}

template <typename T>
__global__ void segmentMinMaxKernel(
    const T* data,
    int64_t numRows,
    int32_t segmentRows,
    int64_t* outMin,
    int64_t* outMax) {
  using BlockReduce = cub::BlockReduce<int64_t, kThreads>;
  __shared__ typename BlockReduce::TempStorage minStorage;
  __shared__ typename BlockReduce::TempStorage maxStorage;
  const int64_t begin = static_cast<int64_t>(blockIdx.x) * segmentRows;
  const int64_t end = cuda::std::min<int64_t>(begin + segmentRows, numRows);
  int64_t localMin = cuda::std::numeric_limits<int64_t>::max();
  int64_t localMax = cuda::std::numeric_limits<int64_t>::min();
  for (int64_t row = begin + threadIdx.x; row < end; row += blockDim.x) {
    const int64_t value = static_cast<int64_t>(data[row]);
    localMin = min(localMin, value);
    localMax = max(localMax, value);
  }
  const int64_t blockMin =
      BlockReduce(minStorage).Reduce(localMin, cuda::minimum<>{});
  const int64_t blockMax =
      BlockReduce(maxStorage).Reduce(localMax, cuda::maximum<>{});
  if (threadIdx.x == 0) {
    outMin[blockIdx.x] = blockMin;
    outMax[blockIdx.x] = blockMax;
  }
}

template <typename T>
__global__ void packKernel(
    const T* data,
    int64_t numRows,
    int32_t segmentRows,
    const SegmentParams* params,
    unsigned long long* words) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= numRows) {
    return;
  }
  const auto& p = params[row / segmentRows];
  if (p.bitWidth == 0) {
    return;
  }
  const uint64_t value =
      static_cast<uint64_t>(static_cast<int64_t>(data[row]) - p.base);
  const uint64_t bit = p.bitOffset + (row % segmentRows) * p.bitWidth;
  const uint64_t word = bit >> 6;
  const uint32_t shift = bit & 63;
  atomicOr(&words[word], static_cast<unsigned long long>(value << shift));
  if (shift + p.bitWidth > 64) {
    atomicOr(
        &words[word + 1],
        static_cast<unsigned long long>(value >> (64 - shift)));
  }
}

__device__ __forceinline__ int64_t
unpack(const uint64_t* words, const SegmentParams& p, int64_t index) {
  if (p.bitWidth == 0) {
    return p.base;
  }
  const uint64_t bit = p.bitOffset + index * p.bitWidth;
  const uint64_t word = bit >> 6;
  const uint32_t shift = bit & 63;
  uint64_t value = words[word] >> shift;
  if (shift + p.bitWidth > 64) {
    value |= words[word + 1] << (64 - shift);
  }
  const uint64_t mask = p.bitWidth == 64 ? ~0ULL : ((1ULL << p.bitWidth) - 1);
  return p.base + static_cast<int64_t>(value & mask);
}

/// Decodes 'numOut' values. Output j is row 'rows[j]' of the concatenated
/// segments, or row j without 'rows'.
template <typename OutT>
__global__ void unpackKernel(
    const uint64_t* words,
    const SegmentParams* params,
    const int32_t* segments,
    int32_t segmentRows,
    const int32_t* rows,
    int64_t numOut,
    OutT* out) {
  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= numOut) {
    return;
  }
  const int64_t row = rows ? rows[j] : j;
  const auto& p = params[segments[row / segmentRows]];
  out[j] = static_cast<OutT>(unpack(words, p, row % segmentRows));
}

/// Maps output j to the table row it reads, for gathers from kPlain columns.
__global__ void globalRowsKernel(
    const int32_t* segments,
    int32_t segmentRows,
    const int32_t* rows,
    int64_t numOut,
    int32_t* out) {
  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= numOut) {
    return;
  }
  const int64_t row = rows ? rows[j] : j;
  out[j] = static_cast<int32_t>(
      static_cast<int64_t>(segments[row / segmentRows]) * segmentRows +
      row % segmentRows);
}

bool isIntegerLike(cudf::data_type type) {
  switch (type.id()) {
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
      return true;
    default:
      return false;
  }
}

uint32_t bitWidthOf(uint64_t range) {
  return range == 0 ? 0 : 64 - __builtin_clzll(range);
}

/// Fills the kForBitpack streams of 'column' from 'values', a 4- or 8-byte
/// integer-like column.
void encodeForBitpack(
    EncodedColumn& column,
    cudf::column_view values,
    int32_t segmentRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const int64_t numRows = values.size();
  const int32_t numSegments =
      static_cast<int32_t>((numRows + segmentRows - 1) / segmentRows);
  const bool wide = cudf::size_of(values.type()) == 8;
  check(
      wide || cudf::size_of(values.type()) == 4,
      "Unsupported width for bit-packing: " +
          std::to_string(cudf::size_of(values.type())));

  rmm::device_uvector<int64_t> deviceMin(numSegments, stream, mr);
  rmm::device_uvector<int64_t> deviceMax(numSegments, stream, mr);
  if (numSegments > 0 && numRows > 0) {
    if (wide) {
      segmentMinMaxKernel<int64_t><<<numSegments, kThreads, 0, stream.get()>>>(
          values.data<int64_t>(),
          numRows,
          segmentRows,
          deviceMin.data(),
          deviceMax.data());
    } else {
      segmentMinMaxKernel<int32_t><<<numSegments, kThreads, 0, stream.get()>>>(
          values.data<int32_t>(),
          numRows,
          segmentRows,
          deviceMin.data(),
          deviceMax.data());
    }
    checkCuda(cudaGetLastError(), "segmentMinMaxKernel");
  }
  column.segmentMin.resize(numSegments);
  column.segmentMax.resize(numSegments);
  checkCuda(
      cudaMemcpyAsync(
          column.segmentMin.data(),
          deviceMin.data(),
          numSegments * sizeof(int64_t),
          cudaMemcpyDeviceToHost,
          stream.get()),
      "copy segment minimums");
  checkCuda(
      cudaMemcpyAsync(
          column.segmentMax.data(),
          deviceMax.data(),
          numSegments * sizeof(int64_t),
          cudaMemcpyDeviceToHost,
          stream.get()),
      "copy segment maximums");
  stream.sync();

  std::vector<SegmentParams> params(numSegments);
  column.segmentBitWidth.resize(numSegments);
  uint64_t bitOffset = 0;
  for (int32_t s = 0; s < numSegments; ++s) {
    const uint64_t range = static_cast<uint64_t>(column.segmentMax[s]) -
        static_cast<uint64_t>(column.segmentMin[s]);
    const uint32_t width = bitWidthOf(range);
    const int64_t rows =
        std::min<int64_t>(segmentRows, numRows - int64_t{s} * segmentRows);
    params[s] = SegmentParams{column.segmentMin[s], bitOffset, width, 0};
    column.segmentBitWidth[s] = width;
    bitOffset += ((rows * width + 63) / 64) * 64;
  }
  const int64_t numWords = static_cast<int64_t>(bitOffset / 64) + 1;

  column.segmentParams =
      rmm::device_uvector<SegmentParams>(numSegments, stream, mr);
  checkCuda(
      cudaMemcpyAsync(
          column.segmentParams.data(),
          params.data(),
          numSegments * sizeof(SegmentParams),
          cudaMemcpyHostToDevice,
          stream.get()),
      "copy segment params");
  column.packed = rmm::device_uvector<uint64_t>(numWords, stream, mr);
  checkCuda(
      cudaMemsetAsync(
          column.packed.data(), 0, numWords * sizeof(uint64_t), stream.get()),
      "clear packed words");
  if (numRows > 0) {
    auto* words = reinterpret_cast<unsigned long long*>(column.packed.data());
    if (wide) {
      packKernel<int64_t><<<numBlocks(numRows), kThreads, 0, stream.get()>>>(
          values.data<int64_t>(),
          numRows,
          segmentRows,
          column.segmentParams.data(),
          words);
    } else {
      packKernel<int32_t><<<numBlocks(numRows), kThreads, 0, stream.get()>>>(
          values.data<int32_t>(),
          numRows,
          segmentRows,
          column.segmentParams.data(),
          words);
    }
    checkCuda(cudaGetLastError(), "packKernel");
  }
  stream.sync();
}

/// Decodes the bit-packed stream of 'column' into a new column of 'type'.
std::unique_ptr<cudf::column> unpackColumn(
    const EncodedTable& table,
    const EncodedColumn& column,
    cudf::data_type type,
    const rmm::device_uvector<int32_t>& deviceSegments,
    const int32_t* rows,
    int64_t numOut,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto out = cudf::make_fixed_width_column(
      type,
      static_cast<cudf::size_type>(numOut),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  if (numOut == 0) {
    return out;
  }
  auto view = out->mutable_view();
  if (cudf::size_of(type) == 8) {
    unpackKernel<int64_t><<<numBlocks(numOut), kThreads, 0, stream.get()>>>(
        column.packed.data(),
        column.segmentParams.data(),
        deviceSegments.data(),
        table.segmentRows,
        rows,
        numOut,
        view.data<int64_t>());
  } else {
    check(cudf::size_of(type) == 4, "Unsupported decode width");
    unpackKernel<int32_t><<<numBlocks(numOut), kThreads, 0, stream.get()>>>(
        column.packed.data(),
        column.segmentParams.data(),
        deviceSegments.data(),
        table.segmentRows,
        rows,
        numOut,
        view.data<int32_t>());
  }
  checkCuda(cudaGetLastError(), "unpackKernel");
  return out;
}

/// Converts a column read from Parquet to the type it is stored and decoded
/// as. Short decimals are kept in 64 bits, as Velox holds them.
std::unique_ptr<cudf::column> normalize(
    std::unique_ptr<cudf::column> column,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (column->type().id() == cudf::type_id::DECIMAL128) {
    return cudf::cast(
        column->view(),
        cudf::data_type{cudf::type_id::DECIMAL64, column->type().scale()},
        stream,
        mr);
  }
  return column;
}

} // namespace

EncodedColumn::EncodedColumn(
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr)
    : codec(Codec::kPlain),
      packed(0, stream, mr),
      segmentParams(0, stream, mr) {}

int64_t EncodedColumn::hostBytes() const {
  int64_t bytes = 0;
  for (const auto& words : hostPacked) {
    bytes += words.size() * sizeof(uint64_t);
  }
  for (const auto& piece : hostValues) {
    bytes += piece->alloc_size();
  }
  return bytes;
}

int64_t EncodedColumn::deviceBytes() const {
  int64_t bytes = packed.size() * sizeof(uint64_t) +
      segmentParams.size() * sizeof(SegmentParams);
  if (values) {
    bytes += values->alloc_size();
  }
  return bytes;
}

int64_t EncodedTable::hostBytes() const {
  int64_t bytes = 0;
  for (const auto& column : columns) {
    bytes += column->hostBytes();
  }
  return bytes;
}

int64_t EncodedTable::deviceBytes() const {
  int64_t bytes = 0;
  for (const auto& column : columns) {
    bytes += column->deviceBytes();
  }
  return bytes;
}

void* BoundedPinnedHostResource::allocate_sync(
    std::size_t bytes,
    std::size_t /*alignment*/) {
  if (bytes == 0) {
    return nullptr;
  }
  auto& state = *state_;
  const uint64_t limit = state.limitBytes.load();
  const uint64_t used = state.usedBytes.fetch_add(bytes) + bytes;
  if (limit != 0 && used > limit) {
    state.usedBytes.fetch_sub(bytes);
    throw EncodedHostBudgetExceeded(
        "Encoded cache host memory limit exceeded: limit " +
        std::to_string(limit) + " bytes, in use " +
        std::to_string(used - bytes) + " bytes, requested " +
        std::to_string(bytes) + " bytes");
  }
  void* pointer = nullptr;
  const auto error = cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault);
  if (error != cudaSuccess) {
    state.usedBytes.fetch_sub(bytes);
    throw std::runtime_error(
        "cudaHostAlloc of " + std::to_string(bytes) +
        " bytes failed: " + cudaGetErrorString(error));
  }
  uint64_t peak = state.peakBytes.load();
  while (used > peak && !state.peakBytes.compare_exchange_weak(peak, used)) {
  }
  return pointer;
}

void BoundedPinnedHostResource::deallocate_sync(
    void* pointer,
    std::size_t bytes,
    std::size_t /*alignment*/) noexcept {
  if (pointer == nullptr || bytes == 0) {
    return;
  }
  cudaFreeHost(pointer);
  state_->usedBytes.fetch_sub(bytes);
}

const EncodedColumn& EncodedTable::column(const std::string& columnName) const {
  auto it = columnIndex.find(columnName);
  check(
      it != columnIndex.end(),
      "Column " + columnName + " not found in encoded table " + name);
  return *columns[it->second];
}

int32_t EncodedTable::segmentSize(int32_t segment) const {
  return static_cast<int32_t>(std::min<int64_t>(
      segmentRows, numRows - static_cast<int64_t>(segment) * segmentRows));
}

std::unique_ptr<cudf::column> decodeColumn(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments,
    const rmm::device_uvector<int32_t>& deviceSegments,
    std::optional<cudf::column_view> rows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  int64_t numOut = 0;
  if (rows.has_value()) {
    numOut = rows->size();
  } else {
    for (auto segment : segments) {
      numOut += table.segmentSize(segment);
    }
  }
  const int32_t* rowData = rows.has_value() ? rows->data<int32_t>() : nullptr;

  switch (column.codec) {
    case Codec::kForBitpack:
      return unpackColumn(
          table,
          column,
          column.type,
          deviceSegments,
          rowData,
          numOut,
          stream,
          mr);
    case Codec::kDictionary: {
      auto codes = unpackColumn(
          table,
          column,
          cudf::data_type{cudf::type_id::INT32},
          deviceSegments,
          rowData,
          numOut,
          stream,
          mr);
      auto gathered = cudf::gather(
          cudf::table_view({column.values->view()}),
          codes->view(),
          cudf::out_of_bounds_policy::DONT_CHECK,
          stream,
          mr);
      return std::move(gathered->release()[0]);
    }
    case Codec::kPlain: {
      auto map = cudf::make_fixed_width_column(
          cudf::data_type{cudf::type_id::INT32},
          static_cast<cudf::size_type>(numOut),
          cudf::mask_state::UNALLOCATED,
          stream,
          mr);
      if (numOut > 0) {
        globalRowsKernel<<<numBlocks(numOut), kThreads, 0, stream.get()>>>(
            deviceSegments.data(),
            table.segmentRows,
            rowData,
            numOut,
            map->mutable_view().data<int32_t>());
        checkCuda(cudaGetLastError(), "globalRowsKernel");
      }
      auto gathered = cudf::gather(
          cudf::table_view({column.values->view()}),
          map->view(),
          cudf::out_of_bounds_policy::DONT_CHECK,
          stream,
          mr);
      return std::move(gathered->release()[0]);
    }
  }
  throw std::logic_error("Unknown codec");
}

EncodedColumnCache& EncodedColumnCache::instance() {
  static EncodedColumnCache cache;
  return cache;
}

namespace {

constexpr cudf::size_type kMaxDictionaryKeys = 1 << 16;

// Rows to read per read_parquet() call. Small row groups are read together,
// as each call parses the file footer again.
constexpr int64_t kReadRows = 1 << 22;

// Reads 'columnNames' of 'files', or every column if that is empty, in runs
// of consecutive row groups of at least kReadRows rows, or one row group if
// it is larger. Calls 'consume' with each run after normalize(), and whether
// it is the last.
template <typename Consume>
void forEachRowGroup(
    const std::vector<std::string>& files,
    const std::vector<std::string>& columnNames,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    Consume consume) {
  std::vector<std::pair<size_t, std::vector<cudf::size_type>>> runs;
  for (size_t f = 0; f < files.size(); ++f) {
    const auto metadata =
        cudf::io::read_parquet_metadata(cudf::io::source_info(files[f]));
    const auto rowGroups = metadata.rowgroup_metadata();
    // Runs do not span files.
    int64_t rows = 0;
    for (cudf::size_type g = 0; g < metadata.num_rowgroups(); ++g) {
      if (rows == 0) {
        runs.emplace_back(f, std::vector<cudf::size_type>{});
      }
      runs.back().second.push_back(g);
      rows += rowGroups[g].at("num_rows");
      if (rows >= kReadRows) {
        rows = 0;
      }
    }
  }
  for (size_t i = 0; i < runs.size(); ++i) {
    const auto& [file, groups] = runs[i];
    auto options = cudf::io::parquet_reader_options::builder(
                       cudf::io::source_info(files[file]))
                       .row_groups({groups})
                       .build();
    if (!columnNames.empty()) {
      options.set_column_names(columnNames);
    }
    auto read = cudf::io::read_parquet(options, stream, mr);
    auto columns = read.tbl->release();
    for (auto& column : columns) {
      check(
          column->null_count() == 0, "Encoded tables do not support nulls yet");
      column = normalize(std::move(column), stream, mr);
    }
    consume(
        std::make_unique<cudf::table>(std::move(columns)),
        read.metadata,
        i + 1 == runs.size());
  }
}

// One piece of a bit-packed stream, stored in the storage resource.
struct PackedPiece {
  // The packed words, without a padding word.
  rmm::device_uvector<uint64_t> words;
  // Params of the piece's segments, 'bitOffset' counted from the first word
  // of the piece.
  std::vector<SegmentParams> params;
  int64_t numRows;
};

// A column being loaded. Its pieces line up with those of the other columns
// of the same load.
struct ColumnLoad {
  std::unique_ptr<EncodedColumn> column;
  // kForBitpack values or kDictionary codes.
  std::vector<PackedPiece> packed;
  // kPlain values.
  std::vector<std::unique_ptr<cudf::column>> plain;
  // kDictionary: the distinct values in order of first appearance, the
  // permutation that sorts them, and the sorted values. A code is an index
  // into 'keys', so earlier pieces stay valid as 'keys' grows.
  std::unique_ptr<cudf::column> keys;
  std::unique_ptr<cudf::column> sortedOrder;
  std::unique_ptr<cudf::column> sortedKeys;
};

// Bit-packs 'values' on the device and stores the words in 'storageMr'.
void appendPacked(
    ColumnLoad& load,
    cudf::column_view values,
    int32_t segmentRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref storageMr) {
  EncodedColumn piece(stream, mr);
  encodeForBitpack(piece, values, segmentRows, stream, mr);
  std::vector<SegmentParams> params(piece.segmentParams.size());
  checkCuda(
      cudaMemcpyAsync(
          params.data(),
          piece.segmentParams.data(),
          params.size() * sizeof(SegmentParams),
          cudaMemcpyDeviceToHost,
          stream.get()),
      "copy piece segment params");
  const uint64_t words = piece.packed.size() - 1;
  rmm::device_uvector<uint64_t> stored(words, stream, storageMr);
  if (words > 0) {
    checkCuda(
        cudaMemcpyAsync(
            stored.data(),
            piece.packed.data(),
            words * sizeof(uint64_t),
            cudaMemcpyDefault,
            stream.get()),
        "store packed words");
  }
  stream.sync();
  auto& column = *load.column;
  column.segmentMin.insert(
      column.segmentMin.end(),
      piece.segmentMin.begin(),
      piece.segmentMin.end());
  column.segmentMax.insert(
      column.segmentMax.end(),
      piece.segmentMax.begin(),
      piece.segmentMax.end());
  column.segmentBitWidth.insert(
      column.segmentBitWidth.end(),
      piece.segmentBitWidth.begin(),
      piece.segmentBitWidth.end());
  load.packed.push_back(
      PackedPiece{std::move(stored), std::move(params), values.size()});
}

// Decodes the INT32 codes of one stored piece on the device.
std::unique_ptr<cudf::column> unpackPiece(
    const PackedPiece& piece,
    int32_t segmentRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto numSegments = piece.params.size();
  rmm::device_uvector<uint64_t> words(piece.words.size() + 1, stream, mr);
  if (piece.words.size() > 0) {
    checkCuda(
        cudaMemcpyAsync(
            words.data(),
            piece.words.data(),
            piece.words.size() * sizeof(uint64_t),
            cudaMemcpyDefault,
            stream.get()),
        "copy piece words");
  }
  checkCuda(
      cudaMemsetAsync(
          words.data() + piece.words.size(), 0, sizeof(uint64_t), stream.get()),
      "clear padding word");
  rmm::device_uvector<SegmentParams> params(numSegments, stream, mr);
  checkCuda(
      cudaMemcpyAsync(
          params.data(),
          piece.params.data(),
          numSegments * sizeof(SegmentParams),
          cudaMemcpyHostToDevice,
          stream.get()),
      "copy piece params");
  std::vector<int32_t> ids(numSegments);
  for (size_t s = 0; s < numSegments; ++s) {
    ids[s] = static_cast<int32_t>(s);
  }
  rmm::device_uvector<int32_t> segments(numSegments, stream, mr);
  checkCuda(
      cudaMemcpyAsync(
          segments.data(),
          ids.data(),
          numSegments * sizeof(int32_t),
          cudaMemcpyHostToDevice,
          stream.get()),
      "copy piece segment ids");
  auto out = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(piece.numRows),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  if (piece.numRows > 0) {
    unpackKernel<int32_t>
        <<<numBlocks(piece.numRows), kThreads, 0, stream.get()>>>(
            words.data(),
            params.data(),
            segments.data(),
            segmentRows,
            nullptr,
            piece.numRows,
            out->mutable_view().data<int32_t>());
    checkCuda(cudaGetLastError(), "unpackKernel");
  }
  stream.sync();
  return out;
}

// Turns a kDictionary column being loaded into kPlain: its coded pieces
// become the strings they stand for, stored in 'storageMr'.
void toPlain(
    ColumnLoad& load,
    int32_t segmentRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref storageMr) {
  for (auto& piece : load.packed) {
    auto codes = unpackPiece(piece, segmentRows, stream, mr);
    auto strings = cudf::gather(
        cudf::table_view({load.keys->view()}),
        codes->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        storageMr);
    load.plain.push_back(std::move(strings->release()[0]));
    stream.sync();
    // Release the coded piece before decoding the next one.
    piece.words = rmm::device_uvector<uint64_t>(0, stream, storageMr);
  }
  load.packed.clear();
  load.keys.reset();
  load.sortedOrder.reset();
  load.sortedKeys.reset();
  auto& column = *load.column;
  column.codec = Codec::kPlain;
  column.segmentMin.clear();
  column.segmentMax.clear();
  column.segmentBitWidth.clear();
}

// Codes one piece of a kDictionary column against the values seen so far,
// adding its new values to them. Returns false, with the column turned into
// kPlain and the piece not stored, if the column has more distinct values
// than a dictionary may hold.
bool appendDictionaryCoded(
    ColumnLoad& load,
    cudf::column_view values,
    int32_t segmentRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref storageMr) {
  auto distinct = cudf::distinct(
      cudf::table_view({values}),
      {0},
      cudf::duplicate_keep_option::KEEP_ANY,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream,
      mr);
  if (distinct->num_rows() > kMaxDictionaryKeys) {
    toPlain(load, segmentRows, stream, mr, storageMr);
    return false;
  }
  // The values so far come first, so they keep their codes.
  std::vector<cudf::column_view> parts;
  if (load.keys) {
    parts.push_back(load.keys->view());
  }
  parts.push_back(distinct->view().column(0));
  auto all = cudf::concatenate(parts, stream, mr);
  auto merged = cudf::stable_distinct(
      cudf::table_view({all->view()}),
      {0},
      cudf::duplicate_keep_option::KEEP_FIRST,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream,
      mr);
  if (merged->num_rows() > kMaxDictionaryKeys) {
    toPlain(load, segmentRows, stream, mr, storageMr);
    return false;
  }
  if (!load.keys || merged->num_rows() != load.keys->size()) {
    load.keys = std::move(merged->release()[0]);
    load.sortedOrder = cudf::sorted_order(
        cudf::table_view({load.keys->view()}), {}, {}, stream, mr);
    auto sorted = cudf::gather(
        cudf::table_view({load.keys->view()}),
        load.sortedOrder->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        mr);
    load.sortedKeys = std::move(sorted->release()[0]);
  }
  auto positions = cudf::lower_bound(
      cudf::table_view({load.sortedKeys->view()}),
      cudf::table_view({values}),
      {cudf::order::ASCENDING},
      {cudf::null_order::AFTER},
      stream,
      mr);
  auto codes = cudf::gather(
      cudf::table_view({load.sortedOrder->view()}),
      positions->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      mr);
  appendPacked(
      load, codes->view().column(0), segmentRows, stream, mr, storageMr);
  return true;
}

// Moves the pieces of a loaded column into its final form: kept as pieces for
// a host-resident table, joined into one device buffer otherwise.
void finishColumn(
    ColumnLoad& load,
    const std::vector<int64_t>& pieceBeginRow,
    int64_t numRows,
    int32_t segmentRows,
    bool hostResident,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref storageMr) {
  auto& column = *load.column;
  if (column.codec == Codec::kDictionary &&
      int64_t{load.keys->size()} * 8 >= numRows) {
    toPlain(load, segmentRows, stream, mr, storageMr);
  }
  if (column.codec == Codec::kDictionary) {
    column.values = std::move(load.keys);
  }
  if (column.codec == Codec::kPlain) {
    if (hostResident) {
      column.hostValues = std::move(load.plain);
      column.hostPieceBeginRow = pieceBeginRow;
    } else if (load.plain.size() == 1) {
      column.values = std::move(load.plain[0]);
    } else {
      std::vector<cudf::column_view> parts;
      for (const auto& part : load.plain) {
        parts.push_back(part->view());
      }
      column.values = cudf::concatenate(parts, stream, storageMr);
    }
    load.plain.clear();
    stream.sync();
    return;
  }
  if (hostResident) {
    uint64_t word = 0;
    for (auto& piece : load.packed) {
      column.hostPackedBeginWord.push_back(word);
      for (auto p : piece.params) {
        p.bitOffset += word * 64;
        column.hostSegmentParams.push_back(p);
      }
      word += piece.words.size();
      column.hostPacked.push_back(std::move(piece.words));
    }
    column.hostPieceBeginRow = pieceBeginRow;
    load.packed.clear();
    return;
  }
  uint64_t numWords = 0;
  std::vector<SegmentParams> params;
  for (const auto& piece : load.packed) {
    for (auto p : piece.params) {
      p.bitOffset += numWords * 64;
      params.push_back(p);
    }
    numWords += piece.words.size();
  }
  column.packed =
      rmm::device_uvector<uint64_t>(numWords + 1, stream, storageMr);
  uint64_t offset = 0;
  for (const auto& piece : load.packed) {
    if (piece.words.size() > 0) {
      checkCuda(
          cudaMemcpyAsync(
              column.packed.data() + offset,
              piece.words.data(),
              piece.words.size() * sizeof(uint64_t),
              cudaMemcpyDefault,
              stream.get()),
          "concatenate packed words");
    }
    offset += piece.words.size();
  }
  checkCuda(
      cudaMemsetAsync(
          column.packed.data() + numWords, 0, sizeof(uint64_t), stream.get()),
      "clear padding word");
  column.segmentParams =
      rmm::device_uvector<SegmentParams>(params.size(), stream, mr);
  checkCuda(
      cudaMemcpyAsync(
          column.segmentParams.data(),
          params.data(),
          params.size() * sizeof(SegmentParams),
          cudaMemcpyHostToDevice,
          stream.get()),
      "copy segment params");
  stream.sync();
  load.packed.clear();
}

} // namespace

void EncodedColumnCache::loadColumns(
    const std::string& name,
    const std::vector<std::string>& parquetFiles,
    const std::vector<std::string>& columnNames,
    int32_t segmentRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref storageMr,
    bool hostResident) {
  check(segmentRows > 0, "segmentRows must be positive");
  EncodedTable* existing = nullptr;
  if (auto it = tables_.find(name); it != tables_.end()) {
    existing = it->second.get();
    check(
        existing->segmentRows == segmentRows &&
            existing->hostResident == hostResident,
        "Encoded table " + name + " is loaded with other settings");
  }
  std::vector<std::string> missing;
  for (const auto& column : columnNames) {
    if (existing == nullptr || !existing->hasColumn(column)) {
      missing.push_back(column);
    }
  }
  if (!columnNames.empty() && missing.empty()) {
    return;
  }

  // New columns, and for each column read its index in 'loads', or -1 for a
  // column the table already has.
  std::vector<ColumnLoad> loads;
  std::vector<int32_t> loadOf;
  std::vector<int64_t> pieceBeginRow;
  int64_t numRows = 0;
  int64_t encodedRows = 0;

  auto encodePiece = [&](cudf::table_view piece) {
    pieceBeginRow.push_back(encodedRows);
    for (size_t i = 0; i < loadOf.size(); ++i) {
      if (loadOf[i] < 0) {
        continue;
      }
      auto& load = loads[loadOf[i]];
      const auto values = piece.column(i);
      if (load.column->codec == Codec::kDictionary &&
          appendDictionaryCoded(
              load, values, segmentRows, stream, mr, storageMr)) {
        continue;
      }
      if (load.column->codec == Codec::kForBitpack) {
        appendPacked(load, values, segmentRows, stream, mr, storageMr);
      } else {
        std::vector<cudf::column_view> parts{values};
        load.plain.push_back(cudf::concatenate(parts, stream, storageMr));
      }
    }
    stream.sync();
    encodedRows += piece.num_rows();
  };

  // Encode whole segments as they fill, carrying the rest of each row group
  // into the next.
  std::unique_ptr<cudf::table> carry;
  forEachRowGroup(
      parquetFiles,
      missing,
      stream,
      mr,
      [&](std::unique_ptr<cudf::table> rows, const auto& metadata, bool last) {
        if (loadOf.empty()) {
          for (size_t i = 0; i < metadata.schema_info.size(); ++i) {
            const auto& columnName = metadata.schema_info[i].name;
            if (existing != nullptr && existing->hasColumn(columnName)) {
              loadOf.push_back(-1);
              continue;
            }
            auto encoded = std::make_unique<EncodedColumn>(stream, mr);
            encoded->name = columnName;
            encoded->type = rows->view().column(i).type();
            encoded->codec = isIntegerLike(encoded->type) ? Codec::kForBitpack
                : encoded->type.id() == cudf::type_id::STRING
                ? Codec::kDictionary
                : Codec::kPlain;
            loadOf.push_back(static_cast<int32_t>(loads.size()));
            loads.emplace_back();
            loads.back().column = std::move(encoded);
          }
        }
        check(
            static_cast<size_t>(rows->num_columns()) == loadOf.size(),
            "Row groups of " + name + " differ in their columns");
        numRows += rows->num_rows();
        if (carry) {
          std::vector<cudf::table_view> parts{carry->view(), rows->view()};
          rows = cudf::concatenate(parts, stream, mr);
          carry.reset();
        }
        const auto total = rows->num_rows();
        const auto whole = last ? total : total / segmentRows * segmentRows;
        if (whole > 0) {
          encodePiece(cudf::slice(rows->view(), {0, whole}, stream)[0]);
        }
        if (whole < total) {
          std::vector<cudf::table_view> rest{
              cudf::slice(rows->view(), {whole, total}, stream)[0]};
          carry = cudf::concatenate(rest, stream, mr);
        }
      });
  check(!loadOf.empty(), "No row groups in " + name);
  check(
      existing == nullptr || existing->numRows == numRows,
      "Encoded table " + name + " changed its row count");
  for (auto& load : loads) {
    finishColumn(
        load,
        pieceBeginRow,
        numRows,
        segmentRows,
        hostResident,
        stream,
        mr,
        storageMr);
  }

  if (existing == nullptr) {
    auto table = std::make_unique<EncodedTable>();
    table->name = name;
    table->segmentRows = segmentRows;
    table->hostResident = hostResident;
    table->numRows = numRows;
    table->numSegments =
        static_cast<int32_t>((numRows + segmentRows - 1) / segmentRows);
    existing = table.get();
    tables_[name] = std::move(table);
  }
  for (auto& load : loads) {
    existing->columnIndex.emplace(
        load.column->name, static_cast<int32_t>(existing->columns.size()));
    existing->columns.push_back(std::move(load.column));
  }
}

void EncodedColumnCache::retainColumns(
    const std::string& name,
    const std::unordered_set<std::string>& keep) {
  auto it = tables_.find(name);
  if (it == tables_.end()) {
    return;
  }
  auto& table = *it->second;
  std::vector<std::unique_ptr<EncodedColumn>> kept;
  for (auto& column : table.columns) {
    if (keep.count(column->name) != 0) {
      kept.push_back(std::move(column));
    }
  }
  if (kept.empty()) {
    tables_.erase(it);
    return;
  }
  table.columns = std::move(kept);
  table.columnIndex.clear();
  for (size_t i = 0; i < table.columns.size(); ++i) {
    table.columnIndex.emplace(table.columns[i]->name, static_cast<int32_t>(i));
  }
}

std::vector<std::string> EncodedColumnCache::tableNames() const {
  std::vector<std::string> names;
  for (const auto& [name, _] : tables_) {
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

int64_t EncodedColumnCache::hostBytes() const {
  int64_t bytes = 0;
  for (const auto& [_, table] : tables_) {
    bytes += table->hostBytes();
  }
  return bytes;
}

int64_t EncodedColumnCache::deviceBytes() const {
  int64_t bytes = 0;
  for (const auto& [_, table] : tables_) {
    bytes += table->deviceBytes();
  }
  return bytes;
}

const EncodedTable* EncodedColumnCache::findTable(
    const std::string& name) const {
  auto it = tables_.find(name);
  return it == tables_.end() ? nullptr : it->second.get();
}

const EncodedTable& EncodedColumnCache::table(const std::string& name) const {
  const auto* table = findTable(name);
  check(table != nullptr, "Encoded table " + name + " is not loaded");
  return *table;
}

namespace {

template <typename T>
__global__ void addToOffsetsKernel(T* offsets, int64_t n, T delta) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    offsets[i] += delta;
  }
}

void copyToDevice(
    void* dst,
    const void* src,
    size_t bytes,
    cuda::stream_ref stream) {
  if (bytes > 0) {
    checkCuda(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream.get()),
        "stage to device");
  }
}

// Work queued after every copy from pageable memory. A copy from pageable
// memory waits for the work already on the stream, so the bulk copies from
// pinned memory are queued last to keep them asynchronous.
using Deferred = std::vector<std::function<void()>>;

// A range of adjacent segments stored in one piece of a host-resident column.
struct PieceRun {
  size_t piece;
  int32_t firstSegment;
  int32_t lastSegment;
  // Rows of the run, counted from the first row of the piece.
  int64_t beginRow;
  int64_t endRow;
};

// Splits 'segments' into runs of adjacent segments of the same piece.
std::vector<PieceRun> pieceRuns(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments) {
  const auto& begins = column.hostPieceBeginRow;
  check(!begins.empty(), "Host pieces missing for " + column.name);
  std::vector<PieceRun> runs;
  for (auto segment : segments) {
    const int64_t row = int64_t{segment} * table.segmentRows;
    const size_t piece = std::upper_bound(begins.begin(), begins.end(), row) -
        begins.begin() - 1;
    const int64_t begin = row - begins[piece];
    const int64_t end = begin + table.segmentSize(segment);
    if (!runs.empty() && runs.back().piece == piece &&
        runs.back().lastSegment + 1 == segment) {
      runs.back().lastSegment = segment;
      runs.back().endRow = end;
    } else {
      runs.push_back(PieceRun{piece, segment, segment, begin, end});
    }
  }
  return runs;
}

// Stages the packed words and params of 'segments' of a bit-packed stream.
void stagePacked(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments,
    EncodedColumn& staged,
    Deferred& deferred,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto& params = column.hostSegmentParams;
  check(
      params.size() == static_cast<size_t>(table.numSegments),
      "Host segment params missing for " + column.name);
  auto wordsOf = [&](int32_t segment) {
    return (static_cast<uint64_t>(table.segmentSize(segment)) *
                params[segment].bitWidth +
            63) /
        64;
  };
  std::vector<SegmentParams> stagedParams;
  uint64_t numWords = 0;
  for (auto segment : segments) {
    auto p = params[segment];
    p.bitOffset = numWords * 64;
    stagedParams.push_back(p);
    numWords += wordsOf(segment);
  }
  staged.segmentParams =
      rmm::device_uvector<SegmentParams>(stagedParams.size(), stream, mr);
  copyToDevice(
      staged.segmentParams.data(),
      stagedParams.data(),
      stagedParams.size() * sizeof(SegmentParams),
      stream);
  staged.packed = rmm::device_uvector<uint64_t>(numWords + 1, stream, mr);
  checkCuda(
      cudaMemsetAsync(
          staged.packed.data() + numWords, 0, sizeof(uint64_t), stream.get()),
      "clear padding word");
  uint64_t word = 0;
  for (const auto& run : pieceRuns(table, column, segments)) {
    uint64_t runWords = 0;
    for (int32_t s = run.firstSegment; s <= run.lastSegment; ++s) {
      runWords += wordsOf(s);
    }
    const uint64_t* src = column.hostPacked[run.piece].data() +
        (params[run.firstSegment].bitOffset / 64 -
         column.hostPackedBeginWord[run.piece]);
    deferred.push_back([dst = staged.packed.data() + word,
                        src,
                        bytes = runWords * sizeof(uint64_t),
                        stream] { copyToDevice(dst, src, bytes, stream); });
    word += runWords;
  }
}

// Each run copies one offset past its end, which the next run overwrites, so
// the last offset of the staged column is its character count.
template <typename OffsetT>
std::unique_ptr<cudf::column> stageStrings(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments,
    Deferred& deferred,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto runs = pieceRuns(table, column, segments);
  const auto offsetType = cudf::data_type{cudf::type_to_id<OffsetT>()};
  int64_t numRows = 0;
  int64_t numChars = 0;
  for (const auto& run : runs) {
    cudf::strings_column_view strings(column.hostValues[run.piece]->view());
    check(
        strings.offsets().type() == offsetType,
        "Pieces of " + column.name + " differ in their offset width");
    const auto* offsets = strings.offsets().data<OffsetT>();
    numRows += run.endRow - run.beginRow;
    numChars += offsets[run.endRow] - offsets[run.beginRow];
  }
  check(
      numChars <= static_cast<int64_t>(std::numeric_limits<OffsetT>::max()),
      "Staged batch of " + column.name + " has too many characters");
  auto stagedOffsets = cudf::make_fixed_width_column(
      offsetType,
      static_cast<cudf::size_type>(numRows + 1),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* outOffsets = stagedOffsets->mutable_view().data<OffsetT>();
  rmm::device_buffer stagedChars(numChars, stream, mr);
  auto* outChars = static_cast<char*>(stagedChars.data());
  int64_t row = 0;
  int64_t charBase = 0;
  for (const auto& run : runs) {
    cudf::strings_column_view strings(column.hostValues[run.piece]->view());
    const auto* offsets = strings.offsets().data<OffsetT>();
    const char* chars = strings.parent().data<char>();
    const int64_t begin = run.beginRow;
    const int64_t runRows = run.endRow - run.beginRow;
    const int64_t runChars = offsets[run.endRow] - offsets[begin];
    const auto delta = static_cast<OffsetT>(charBase - offsets[begin]);
    deferred.push_back([=] {
      copyToDevice(
          outChars + charBase, chars + offsets[begin], runChars, stream);
      copyToDevice(
          outOffsets + row,
          offsets + begin,
          (runRows + 1) * sizeof(OffsetT),
          stream);
      addToOffsetsKernel<OffsetT>
          <<<numBlocks(runRows + 1), kThreads, 0, stream.get()>>>(
              outOffsets + row, runRows + 1, delta);
      checkCuda(cudaGetLastError(), "addToOffsetsKernel");
    });
    row += runRows;
    charBase += runChars;
  }
  return cudf::make_strings_column(
      static_cast<cudf::size_type>(numRows),
      std::move(stagedOffsets),
      std::move(stagedChars),
      0,
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
}

std::unique_ptr<cudf::column> stageFixedWidth(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments,
    Deferred& deferred,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto type = column.hostValues.front()->type();
  const auto width = cudf::size_of(type);
  int64_t numRows = 0;
  for (auto segment : segments) {
    numRows += table.segmentSize(segment);
  }
  auto staged = cudf::make_fixed_width_column(
      type,
      static_cast<cudf::size_type>(numRows),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* out = static_cast<char*>(staged->mutable_view().head());
  int64_t row = 0;
  for (const auto& run : pieceRuns(table, column, segments)) {
    const auto values = column.hostValues[run.piece]->view();
    const auto* in =
        static_cast<const char*>(values.head()) + values.offset() * width;
    deferred.push_back([dst = out + row * width,
                        src = in + run.beginRow * width,
                        bytes = (run.endRow - run.beginRow) * width,
                        stream] { copyToDevice(dst, src, bytes, stream); });
    row += run.endRow - run.beginRow;
  }
  return staged;
}

} // namespace

std::unique_ptr<EncodedTable> stageSegments(
    const EncodedTable& table,
    const std::vector<std::string>& columnNames,
    std::span<const int32_t> segments,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  check(table.hostResident, "stageSegments needs a host-resident table");
  check(!segments.empty(), "stageSegments needs segments");
  for (size_t i = 0; i + 1 < segments.size(); ++i) {
    check(
        segments[i] < segments[i + 1] &&
            table.segmentSize(segments[i]) == table.segmentRows,
        "Staged segments must be ascending with only the last one short");
  }
  auto staged = std::make_unique<EncodedTable>();
  staged->name = table.name;
  staged->segmentRows = table.segmentRows;
  staged->numSegments = static_cast<int32_t>(segments.size());
  staged->numRows = int64_t{table.segmentRows} * (staged->numSegments - 1) +
      table.segmentSize(segments.back());
  Deferred deferred;
  for (const auto& name : columnNames) {
    if (staged->columnIndex.count(name)) {
      continue;
    }
    const auto& column = table.column(name);
    auto out = std::make_unique<EncodedColumn>(stream, mr);
    out->name = column.name;
    out->type = column.type;
    out->codec = column.codec;
    switch (column.codec) {
      case Codec::kForBitpack:
        stagePacked(table, column, segments, *out, deferred, stream, mr);
        break;
      case Codec::kDictionary:
        stagePacked(table, column, segments, *out, deferred, stream, mr);
        out->values =
            std::make_unique<cudf::column>(column.values->view(), stream, mr);
        break;
      case Codec::kPlain: {
        check(
            !column.hostValues.empty(),
            "Host values missing for " + column.name);
        const auto first = column.hostValues.front()->view();
        if (first.type().id() == cudf::type_id::STRING) {
          cudf::strings_column_view strings(first);
          out->values = strings.offsets().type().id() == cudf::type_id::INT64
              ? stageStrings<int64_t>(
                    table, column, segments, deferred, stream, mr)
              : stageStrings<int32_t>(
                    table, column, segments, deferred, stream, mr);
        } else {
          out->values =
              stageFixedWidth(table, column, segments, deferred, stream, mr);
        }
        break;
      }
    }
    staged->columnIndex.emplace(
        name, static_cast<int32_t>(staged->columns.size()));
    staged->columns.push_back(std::move(out));
  }
  for (auto& work : deferred) {
    work();
  }
  return staged;
}

void EncodedColumnCache::clear() {
  tables_.clear();
}

} // namespace facebook::velox::cudf_velox::connector::encoded
