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
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <nvcomp/lz4.h>

#include <algorithm>
#include <cstring>
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

/// Per segment: whether it is non-decreasing, its largest difference between
/// consecutive rows, and its first value. A difference is taken in unsigned
/// 64-bit arithmetic only between rows in order, so it is exact.
template <typename T>
__global__ void segmentDeltaKernel(
    const T* data,
    int64_t numRows,
    int32_t segmentRows,
    int32_t* outSorted,
    unsigned long long* outMaxDelta,
    int64_t* outFirst) {
  using DeltaReduce = cub::BlockReduce<unsigned long long, kThreads>;
  using OrderReduce = cub::BlockReduce<int32_t, kThreads>;
  __shared__ typename DeltaReduce::TempStorage deltaStorage;
  __shared__ typename OrderReduce::TempStorage orderStorage;
  const int64_t begin = static_cast<int64_t>(blockIdx.x) * segmentRows;
  const int64_t end = cuda::std::min<int64_t>(begin + segmentRows, numRows);
  unsigned long long localMax = 0;
  int32_t localUnsorted = 0;
  for (int64_t row = begin + 1 + threadIdx.x; row < end; row += blockDim.x) {
    const int64_t previous = static_cast<int64_t>(data[row - 1]);
    const int64_t current = static_cast<int64_t>(data[row]);
    if (current < previous) {
      localUnsorted = 1;
    } else {
      const unsigned long long delta =
          static_cast<uint64_t>(current) - static_cast<uint64_t>(previous);
      localMax = localMax < delta ? delta : localMax;
    }
  }
  const unsigned long long blockMax =
      DeltaReduce(deltaStorage).Reduce(localMax, cuda::maximum<>{});
  const int32_t blockUnsorted =
      OrderReduce(orderStorage).Reduce(localUnsorted, cuda::maximum<>{});
  if (threadIdx.x == 0) {
    outSorted[blockIdx.x] = blockUnsorted == 0 ? 1 : 0;
    outMaxDelta[blockIdx.x] = blockMax;
    outFirst[blockIdx.x] = static_cast<int64_t>(data[begin]);
  }
}

/// Per segment: 1 if every value minus 'segmentMin', the segment minimum, is
/// a multiple of 100, else 0. The difference is taken in unsigned 64-bit
/// arithmetic from a value no smaller than the minimum, so it is exact even
/// where the signed one would overflow.
template <typename T>
__global__ void segmentScale100Kernel(
    const T* data,
    int64_t numRows,
    int32_t segmentRows,
    const int64_t* segmentMin,
    int32_t* outDivisible) {
  using Reduce = cub::BlockReduce<int32_t, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  const int64_t begin = static_cast<int64_t>(blockIdx.x) * segmentRows;
  const int64_t end = cuda::std::min<int64_t>(begin + segmentRows, numRows);
  const uint64_t base = static_cast<uint64_t>(segmentMin[blockIdx.x]);
  int32_t localIndivisible = 0;
  for (int64_t row = begin + threadIdx.x; row < end; row += blockDim.x) {
    const uint64_t offset =
        static_cast<uint64_t>(static_cast<int64_t>(data[row])) - base;
    if (offset % 100 != 0) {
      localIndivisible = 1;
    }
  }
  const int32_t blockIndivisible =
      Reduce(storage).Reduce(localIndivisible, cuda::maximum<>{});
  if (threadIdx.x == 0) {
    outDivisible[blockIdx.x] = blockIndivisible == 0 ? 1 : 0;
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
  const int64_t index = row % segmentRows;
  const uint64_t current =
      static_cast<uint64_t>(static_cast<int64_t>(data[row]));
  // Unsigned differences wrap instead of overflowing at the signed extremes.
  uint64_t value;
  if ((p.flags & kSegmentDelta) != 0) {
    value = index == 0
        ? 0
        : current - static_cast<uint64_t>(static_cast<int64_t>(data[row - 1]));
  } else {
    value = current - static_cast<uint64_t>(p.base);
    if ((p.flags & kSegmentScale100) != 0) {
      // Proven a multiple of 100 when the segment was chosen.
      value /= 100;
    }
  }
  const uint64_t bit = p.bitOffset + index * p.bitWidth;
  const uint64_t word = bit >> 6;
  const uint32_t shift = bit & 63;
  atomicOr(&words[word], static_cast<unsigned long long>(value << shift));
  if (shift + p.bitWidth > 64) {
    atomicOr(
        &words[word + 1],
        static_cast<unsigned long long>(value >> (64 - shift)));
  }
}

/// The value packed for row 'index' of the segment of 'p'.
__device__ __forceinline__ uint64_t
unpackRaw(const uint64_t* words, const SegmentParams& p, int64_t index) {
  if (p.bitWidth == 0) {
    return 0;
  }
  const uint64_t bit = p.bitOffset + index * p.bitWidth;
  const uint64_t word = bit >> 6;
  const uint32_t shift = bit & 63;
  uint64_t value = words[word] >> shift;
  if (shift + p.bitWidth > 64) {
    value |= words[word + 1] << (64 - shift);
  }
  const uint64_t mask = p.bitWidth == 64 ? ~0ULL : ((1ULL << p.bitWidth) - 1);
  return value & mask;
}

/// The bits of row 'index' of a frame-of-reference segment. For a delta
/// segment, the term the row adds to its segment's prefix sum: the first value
/// for row 0, else the difference from the previous row.
__device__ __forceinline__ uint64_t
unpackTerm(const uint64_t* words, const SegmentParams& p, int64_t index) {
  const uint64_t raw = unpackRaw(words, p, index);
  if ((p.flags & kSegmentDelta) != 0 && index != 0) {
    return raw;
  }
  // A scaled segment packs (value - base) / 100. raw * 100 is at most the
  // segment's range, and the sum wraps back to the value's bits.
  return static_cast<uint64_t>(p.base) +
      ((p.flags & kSegmentScale100) != 0 ? raw * 100 : raw);
}

/// Decodes 'numOut' values, or prefix-sum terms for rows of delta segments.
/// Output j is row 'rows[j]' of the concatenated segments, or row j without
/// 'rows'. Narrowing to a 4-byte OutT keeps the low bits, which a modular sum
/// of 4-byte terms reproduces.
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
  out[j] = static_cast<OutT>(static_cast<cuda::std::make_unsigned_t<OutT>>(
      unpackTerm(words, p, row % segmentRows)));
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
/// integer-like column. With 'allowDelta', a non-decreasing segment whose
/// consecutive differences need fewer bits than its range is delta-coded.
/// With 'allowScale100' and 'values' a DECIMAL32 or DECIMAL64 column with
/// scale -2, a segment whose values minus its minimum are all multiples of 100
/// packs those differences divided by 100 when narrower. Each segment takes
/// the narrowest of frame of reference, scaling and delta coding, never two.
void encodeForBitpack(
    EncodedColumn& column,
    cudf::column_view values,
    int32_t segmentRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    bool allowDelta,
    bool allowScale100) {
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
  const int32_t deltaSegments = allowDelta ? numSegments : 0;
  rmm::device_uvector<int32_t> deviceSorted(deltaSegments, stream, mr);
  rmm::device_uvector<unsigned long long> deviceMaxDelta(
      deltaSegments, stream, mr);
  rmm::device_uvector<int64_t> deviceFirst(deltaSegments, stream, mr);
  // Only decimals with scale -2: never dictionary codes or plain integers.
  const bool scale100 = allowScale100 &&
      (values.type().id() == cudf::type_id::DECIMAL32 ||
       values.type().id() == cudf::type_id::DECIMAL64) &&
      values.type().scale() == -2;
  const int32_t scaleSegments = scale100 ? numSegments : 0;
  rmm::device_uvector<int32_t> deviceDivisible(scaleSegments, stream, mr);
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
    if (scale100) {
      // Reads the minimums written above, in stream order.
      if (wide) {
        segmentScale100Kernel<int64_t>
            <<<numSegments, kThreads, 0, stream.get()>>>(
                values.data<int64_t>(),
                numRows,
                segmentRows,
                deviceMin.data(),
                deviceDivisible.data());
      } else {
        segmentScale100Kernel<int32_t>
            <<<numSegments, kThreads, 0, stream.get()>>>(
                values.data<int32_t>(),
                numRows,
                segmentRows,
                deviceMin.data(),
                deviceDivisible.data());
      }
      checkCuda(cudaGetLastError(), "segmentScale100Kernel");
    }
    if (allowDelta) {
      if (wide) {
        segmentDeltaKernel<int64_t><<<numSegments, kThreads, 0, stream.get()>>>(
            values.data<int64_t>(),
            numRows,
            segmentRows,
            deviceSorted.data(),
            deviceMaxDelta.data(),
            deviceFirst.data());
      } else {
        segmentDeltaKernel<int32_t><<<numSegments, kThreads, 0, stream.get()>>>(
            values.data<int32_t>(),
            numRows,
            segmentRows,
            deviceSorted.data(),
            deviceMaxDelta.data(),
            deviceFirst.data());
      }
      checkCuda(cudaGetLastError(), "segmentDeltaKernel");
    }
  }
  std::vector<int32_t> divisible(scaleSegments);
  if (scaleSegments > 0) {
    checkCuda(
        cudaMemcpyAsync(
            divisible.data(),
            deviceDivisible.data(),
            scaleSegments * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream.get()),
        "copy segment scale proofs");
  }
  std::vector<int32_t> sorted(deltaSegments);
  std::vector<unsigned long long> maxDelta(deltaSegments);
  std::vector<int64_t> first(deltaSegments);
  if (deltaSegments > 0) {
    checkCuda(
        cudaMemcpyAsync(
            sorted.data(),
            deviceSorted.data(),
            deltaSegments * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream.get()),
        "copy segment orders");
    checkCuda(
        cudaMemcpyAsync(
            maxDelta.data(),
            deviceMaxDelta.data(),
            deltaSegments * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost,
            stream.get()),
        "copy segment deltas");
    checkCuda(
        cudaMemcpyAsync(
            first.data(),
            deviceFirst.data(),
            deltaSegments * sizeof(int64_t),
            cudaMemcpyDeviceToHost,
            stream.get()),
        "copy segment first values");
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
  column.segmentDelta.assign(numSegments, 0);
  column.segmentScaled.assign(numSegments, 0);
  uint64_t bitOffset = 0;
  for (int32_t s = 0; s < numSegments; ++s) {
    const uint64_t range = static_cast<uint64_t>(column.segmentMax[s]) -
        static_cast<uint64_t>(column.segmentMin[s]);
    uint32_t width = bitWidthOf(range);
    const int64_t rows =
        std::min<int64_t>(segmentRows, numRows - int64_t{s} * segmentRows);
    params[s] = SegmentParams{column.segmentMin[s], bitOffset, width, 0};
    // The maximum minus the minimum is one of the differences proven
    // multiples of 100, so range / 100 is exact.
    if (scale100 && divisible[s] != 0) {
      const uint32_t scaledWidth = bitWidthOf(range / 100);
      if (scaledWidth < width) {
        width = scaledWidth;
        params[s] = SegmentParams{
            column.segmentMin[s], bitOffset, width, kSegmentScale100};
        column.segmentScaled[s] = 1;
      }
    }
    // A constant segment has width 0 either way and stays frame-of-reference.
    if (allowDelta && sorted[s] != 0) {
      const uint32_t deltaWidth = bitWidthOf(maxDelta[s]);
      if (deltaWidth < width) {
        width = deltaWidth;
        params[s] = SegmentParams{first[s], bitOffset, width, kSegmentDelta};
        column.segmentDelta[s] = 1;
        column.segmentScaled[s] = 0;
      }
    }
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

/// Scan key of output row j of a batch: one key per delta segment, so the
/// prefix sum restarts at each, and a key of its own for every other row, so
/// its decoded value passes through unchanged.
struct DeltaScanKey {
  const SegmentParams* params;
  const int32_t* segments;
  int32_t segmentRows;

  __host__ __device__ int64_t operator()(int64_t j) const {
    const int64_t s = j / segmentRows;
    return (params[segments[s]].flags & kSegmentDelta) != 0 ? -1 - s : j;
  }
};

/// Unsigned addition, which wraps instead of overflowing.
template <typename U>
struct WrappingPlus {
  __host__ __device__ U operator()(U a, U b) const {
    return a + b;
  }
};

/// Decodes all 'numRows' rows of the segments in 'deviceSegments' into 'out'
/// where some are delta segments: unpacks values and prefix-sum terms into a
/// temporary, then sums the terms of each delta segment in 'OutT''s unsigned
/// counterpart.
template <typename OutT>
void decodeWithDeltas(
    const EncodedTable& table,
    const EncodedColumn& column,
    const rmm::device_uvector<int32_t>& deviceSegments,
    int64_t numRows,
    OutT* out,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  using U = cuda::std::make_unsigned_t<OutT>;
  rmm::device_uvector<U> terms(numRows, stream, mr);
  unpackKernel<U><<<numBlocks(numRows), kThreads, 0, stream.get()>>>(
      column.packed.data(),
      column.segmentParams.data(),
      deviceSegments.data(),
      table.segmentRows,
      nullptr,
      numRows,
      terms.data());
  checkCuda(cudaGetLastError(), "unpackKernel");
  auto keys = cuda::make_transform_iterator(
      cuda::make_counting_iterator(int64_t{0}),
      DeltaScanKey{
          column.segmentParams.data(),
          deviceSegments.data(),
          table.segmentRows});
  const auto numItems = static_cast<uint32_t>(numRows);
  size_t tempBytes = 0;
  checkCuda(
      cub::DeviceScan::InclusiveScanByKey(
          nullptr,
          tempBytes,
          keys,
          terms.data(),
          reinterpret_cast<U*>(out),
          WrappingPlus<U>{},
          numItems,
          cuda::std::equal_to<>{},
          stream.get()),
      "size delta prefix sum");
  rmm::device_buffer temp(tempBytes, stream, mr);
  checkCuda(
      cub::DeviceScan::InclusiveScanByKey(
          temp.data(),
          tempBytes,
          keys,
          terms.data(),
          reinterpret_cast<U*>(out),
          WrappingPlus<U>{},
          numItems,
          cuda::std::equal_to<>{},
          stream.get()),
      "delta prefix sum");
}

/// Decodes the bit-packed stream of 'column' for 'segments' into a new column
/// of 'type': every row of 'segments', or with 'rows' only those. A batch
/// without delta segments decodes each output row on its own, as before.
std::unique_ptr<cudf::column> decodePacked(
    const EncodedTable& table,
    const EncodedColumn& column,
    cudf::data_type type,
    std::span<const int32_t> segments,
    const rmm::device_uvector<int32_t>& deviceSegments,
    std::optional<cudf::column_view> rows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  int64_t numRows = 0;
  bool anyDelta = false;
  for (auto segment : segments) {
    numRows += table.segmentSize(segment);
    anyDelta = anyDelta ||
        (static_cast<size_t>(segment) < column.segmentDelta.size() &&
         column.segmentDelta[segment] != 0);
  }
  if (!anyDelta || (rows.has_value() && rows->size() == 0)) {
    return unpackColumn(
        table,
        column,
        type,
        deviceSegments,
        rows.has_value() ? rows->data<int32_t>() : nullptr,
        rows.has_value() ? rows->size() : numRows,
        stream,
        mr);
  }
  if (rows.has_value()) {
    // Rows of a delta segment need the rows before them: decode the batch's
    // segments in full, then gather.
    auto all = decodePacked(
        table,
        column,
        type,
        segments,
        deviceSegments,
        std::nullopt,
        stream,
        mr);
    auto gathered = cudf::gather(
        cudf::table_view({all->view()}),
        *rows,
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        mr);
    return std::move(gathered->release()[0]);
  }
  check(
      numRows <= std::numeric_limits<cudf::size_type>::max(),
      "Too many rows to decode in one batch: " + std::to_string(numRows));
  auto out = cudf::make_fixed_width_column(
      type,
      static_cast<cudf::size_type>(numRows),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto view = out->mutable_view();
  if (cudf::size_of(type) == 8) {
    decodeWithDeltas<int64_t>(
        table,
        column,
        deviceSegments,
        numRows,
        view.data<int64_t>(),
        stream,
        mr);
  } else {
    check(cudf::size_of(type) == 4, "Unsupported decode width");
    decodeWithDeltas<int32_t>(
        table,
        column,
        deviceSegments,
        numRows,
        view.data<int32_t>(),
        stream,
        mr);
  }
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

int64_t EncodedColumn::encodedBytes() const {
  int64_t bytes = packed.size() * sizeof(uint64_t) +
      segmentParams.size() * sizeof(SegmentParams);
  if (values) {
    bytes += values->alloc_size();
  }
  if (stringOffsets) {
    bytes += stringOffsets->alloc_size();
  }
  if (compressedChars) {
    bytes += compressedChars->blocks->size();
  }
  return bytes;
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

namespace {

// Uncompressed bytes per LZ4 block of plain string characters.
constexpr int64_t kCharBlockBytes = int64_t{64} << 10;
// Each compressed block starts at a multiple of this in its buffer.
constexpr int64_t kCompressedBlockAlignment = 8;
// Blocks compressed per round at load.
constexpr int64_t kCompressRoundBlocks = 4096;
// Alignment of every rmm allocation used for nvCOMP arguments.
constexpr size_t kAllocationAlignment = 256;

void checkNvcomp(nvcompStatus_t status, const char* what) {
  check(
      status == nvcompSuccess,
      std::string(what) + " failed with nvCOMP status " +
          std::to_string(static_cast<int>(status)));
}

int64_t alignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Decompression on the CUDA backend, whose statuses report a block that is
// corrupt or does not fit its output. On the hardware backend nvCOMP documents
// corrupt input as undefined behavior with statuses always nvcompSuccess.
nvcompBatchedLZ4DecompressOpts_t decompressOptions() {
  auto options = nvcompBatchedLZ4DecompressDefaultOpts;
  options.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
  return options;
}

// Synchronizes 'stream' on leaving its scope, normally or by an exception, so
// the buffers declared before it outlive every copy queued on the stream.
// rmm's pinned host resource frees without waiting on any stream.
struct StreamSyncGuard {
  cudaStream_t stream;

  ~StreamSyncGuard() {
    cudaStreamSynchronize(stream);
  }
};

struct CudaEventDeleter {
  void operator()(cudaEvent_t event) const {
    cudaEventDestroy(event);
  }
};

} // namespace

/// The host side of the decompression staged for one batch of a compressed
/// strings column. One block of pinned host memory from cuDF's pinned pool
/// holds the per-block decompression arguments, which staging fills on the
/// host and copies to the device asynchronously, and the per-block statuses
/// and actual sizes, which it copies back after decompressing. The block goes
/// back to the pool only after the host has seen every copy that reads or
/// writes it complete, so the pool never hands it to another user, on any
/// thread or stream, while a copy is in flight. decodeColumn() calls verify()
/// before reading the staged column.
///
/// Layout, in the pinned block and in the device buffer that mirrors it: input
/// pointers, input sizes, output pointers and output capacities, 8 bytes per
/// block each, then the statuses, 4 bytes per block padded to 8, then the
/// actual sizes, 8 bytes per block. The four argument arrays are contiguous,
/// so one copy moves them.
class DecompressionCheck {
 public:
  DecompressionCheck(
      std::string column,
      std::vector<size_t> expectedSizes,
      cuda::stream_ref stream)
      : column_(std::move(column)),
        expected_(std::move(expectedSizes)),
        blocks_(expected_.size()),
        stream_(stream),
        mr_(cudf::get_pinned_memory_resource()) {
    bytes_ =
        std::max<size_t>(actualSizesOffset() + blocks_ * sizeof(size_t), 8);
    pinned_ = static_cast<char*>(mr_.allocate(stream_, bytes_, kAlignment));
    cudaEvent_t event = nullptr;
    const auto status =
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (status != cudaSuccess) {
      // Nothing was queued on the block: return it before reporting.
      mr_.deallocate(stream_, pinned_, bytes_, kAlignment);
      checkCuda(status, "create decompression event");
    }
    done_.reset(event);
  }

  DecompressionCheck(const DecompressionCheck&) = delete;
  DecompressionCheck& operator=(const DecompressionCheck&) = delete;

  ~DecompressionCheck() {
    // The copies that read the arguments and write the statuses and sizes
    // must be complete before the pool may hand the block out again: wait for
    // the recorded event or, if recording failed after a copy may have been
    // queued, for the stream they were queued on.
    if (recorded_) {
      cudaEventSynchronize(done_.get());
    } else if (pending_) {
      cudaStreamSynchronize(stream_.get());
    }
    try {
      mr_.deallocate(stream_, pinned_, bytes_, kAlignment);
    } catch (...) {
      // A destructor must not throw; the pool keeps the block on failure.
    }
  }

  /// Host views of the argument arrays, filled before anything is queued.
  const void** inputPointers() {
    return reinterpret_cast<const void**>(pinned_ + inputPointersOffset());
  }
  size_t* inputSizes() {
    return reinterpret_cast<size_t*>(pinned_ + inputSizesOffset());
  }
  void** outputPointers() {
    return reinterpret_cast<void**>(pinned_ + outputPointersOffset());
  }
  size_t* capacities() {
    return reinterpret_cast<size_t*>(pinned_ + capacitiesOffset());
  }

  /// Bytes of the pinned block, for the device buffer that mirrors it.
  size_t bytes() const {
    return bytes_;
  }

  size_t inputPointersOffset() const {
    return 0;
  }
  size_t inputSizesOffset() const {
    return blocks_ * sizeof(void*);
  }
  size_t outputPointersOffset() const {
    return inputSizesOffset() + blocks_ * sizeof(size_t);
  }
  size_t capacitiesOffset() const {
    return outputPointersOffset() + blocks_ * sizeof(void*);
  }
  size_t statusesOffset() const {
    return capacitiesOffset() + blocks_ * sizeof(size_t);
  }
  size_t actualSizesOffset() const {
    return statusesOffset() +
        static_cast<size_t>(alignUp(
            static_cast<int64_t>(blocks_ * sizeof(nvcompStatus_t)),
            sizeof(size_t)));
  }

  /// Queues the copy of the four argument arrays to 'deviceMirror', a device
  /// buffer with the same layout. Called before the decompression is queued.
  void copyArguments(char* deviceMirror) {
    pending_ = true;
    checkCuda(
        cudaMemcpyAsync(
            deviceMirror,
            pinned_,
            statusesOffset(),
            cudaMemcpyHostToDevice,
            stream_.get()),
        "copy decompression arguments");
  }

  /// Queues the copies of the statuses and sizes from 'deviceMirror', after
  /// the decompression that writes them, and records their completion.
  void record(const char* deviceMirror) {
    pending_ = true;
    checkCuda(
        cudaMemcpyAsync(
            pinned_ + statusesOffset(),
            deviceMirror + statusesOffset(),
            bytes_ - statusesOffset(),
            cudaMemcpyDeviceToHost,
            stream_.get()),
        "copy decompression statuses and sizes");
    checkCuda(
        cudaEventRecord(done_.get(), stream_.get()),
        "record decompression event");
    recorded_ = true;
  }

  /// Throws if a block failed or did not decompress to its expected size.
  void verify() {
    if (verified_) {
      return;
    }
    check(recorded_, "Decompression of column " + column_ + " was not issued");
    checkCuda(cudaEventSynchronize(done_.get()), "wait for decompression");
    const auto* statuses =
        reinterpret_cast<const nvcompStatus_t*>(pinned_ + statusesOffset());
    const auto* sizes =
        reinterpret_cast<const size_t*>(pinned_ + actualSizesOffset());
    for (size_t i = 0; i < blocks_; ++i) {
      check(
          statuses[i] == nvcompSuccess,
          "LZ4 block " + std::to_string(i) + " staged for column " + column_ +
              " failed with nvCOMP status " +
              std::to_string(static_cast<int>(statuses[i])));
      check(
          sizes[i] == expected_[i],
          "LZ4 block " + std::to_string(i) + " staged for column " + column_ +
              " decompressed to " + std::to_string(sizes[i]) +
              " bytes, expected " + std::to_string(expected_[i]));
    }
    verified_ = true;
  }

 private:
  static constexpr size_t kAlignment = kAllocationAlignment;

  const std::string column_;
  const std::vector<size_t> expected_;
  const size_t blocks_;
  const cuda::stream_ref stream_;
  rmm::host_device_async_resource_ref mr_;
  size_t bytes_{0};
  char* pinned_{nullptr};
  std::unique_ptr<CUevent_st, CudaEventDeleter> done_;
  // Set before the first copy that touches the pinned block is queued.
  bool pending_{false};
  bool recorded_{false};
  bool verified_{false};
};

std::unique_ptr<cudf::column> decodeDictionaryCodes(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments,
    const rmm::device_uvector<int32_t>& deviceSegments,
    std::optional<cudf::column_view> rows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  check(
      column.codec == Codec::kDictionary,
      "Dictionary codes requested for a non-dictionary column");
  return decodePacked(
      table,
      column,
      cudf::data_type{cudf::type_id::INT32},
      segments,
      deviceSegments,
      rows,
      stream,
      mr);
}

std::unique_ptr<cudf::column> decodeColumn(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments,
    const rmm::device_uvector<int32_t>& deviceSegments,
    std::optional<cudf::column_view> rows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (column.decompressionCheck) {
    column.decompressionCheck->verify();
  }
  check(
      column.compressedChars == nullptr,
      "Column " + column.name +
          " holds compressed characters; stage it with stageSegments()");
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
      return decodePacked(
          table,
          column,
          column.type,
          segments,
          deviceSegments,
          rows,
          stream,
          mr);
    case Codec::kDictionary: {
      auto codes = decodePacked(
          table,
          column,
          cudf::data_type{cudf::type_id::INT32},
          segments,
          deviceSegments,
          rows,
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

std::optional<cudf::column_view> stagedPlainValues(
    const EncodedColumn& column) {
  if (column.codec != Codec::kPlain || column.values == nullptr ||
      column.compressedChars != nullptr) {
    return std::nullopt;
  }
  if (column.decompressionCheck) {
    column.decompressionCheck->verify();
  }
  return column.values->view();
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

// Reads 'files' in runs of consecutive row groups of at least kReadRows rows,
// or one row group if it is larger, and calls 'consume' with each run after
// normalize(), and whether it is the last.
template <typename Consume>
void forEachRowGroup(
    const std::vector<std::string>& files,
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

// Bit-packed segments encoded piece by piece and stored in 'storageMr'.
struct PackedPieces {
  std::vector<rmm::device_uvector<uint64_t>> words;
  std::vector<SegmentParams> params;
  uint64_t numWords{0};
};

void appendPacked(
    EncodedColumn& column,
    PackedPieces& pieces,
    cudf::column_view values,
    int32_t segmentRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref storageMr,
    bool allowDelta,
    bool allowScale100) {
  EncodedColumn piece(stream, mr);
  encodeForBitpack(
      piece, values, segmentRows, stream, mr, allowDelta, allowScale100);
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
  checkCuda(
      cudaMemcpyAsync(
          stored.data(),
          piece.packed.data(),
          words * sizeof(uint64_t),
          cudaMemcpyDefault,
          stream.get()),
      "store packed words");
  stream.sync();
  for (auto p : params) {
    p.bitOffset += pieces.numWords * 64;
    pieces.params.push_back(p);
  }
  pieces.words.push_back(std::move(stored));
  pieces.numWords += words;
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
  column.segmentDelta.insert(
      column.segmentDelta.end(),
      piece.segmentDelta.begin(),
      piece.segmentDelta.end());
  column.segmentScaled.insert(
      column.segmentScaled.end(),
      piece.segmentScaled.begin(),
      piece.segmentScaled.end());
}

void finishPacked(
    EncodedColumn& column,
    PackedPieces& pieces,
    bool hostResident,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref storageMr) {
  column.packed =
      rmm::device_uvector<uint64_t>(pieces.numWords + 1, stream, storageMr);
  uint64_t offset = 0;
  for (const auto& words : pieces.words) {
    checkCuda(
        cudaMemcpyAsync(
            column.packed.data() + offset,
            words.data(),
            words.size() * sizeof(uint64_t),
            cudaMemcpyDefault,
            stream.get()),
        "concatenate packed words");
    offset += words.size();
  }
  const uint64_t zero = 0;
  checkCuda(
      cudaMemcpyAsync(
          column.packed.data() + pieces.numWords,
          &zero,
          sizeof(uint64_t),
          cudaMemcpyDefault,
          stream.get()),
      "clear padding word");
  column.segmentParams =
      rmm::device_uvector<SegmentParams>(pieces.params.size(), stream, mr);
  checkCuda(
      cudaMemcpyAsync(
          column.segmentParams.data(),
          pieces.params.data(),
          pieces.params.size() * sizeof(SegmentParams),
          cudaMemcpyHostToDevice,
          stream.get()),
      "copy segment params");
  stream.sync();
  if (hostResident) {
    column.hostSegmentParams = std::move(pieces.params);
  }
  pieces = PackedPieces{};
}

// A string column coded piece by piece before its dictionary is known: each
// piece holds codes into its own sorted distinct values, recoded to the
// table's dictionary once every piece is read.
struct DictionaryPieces {
  std::vector<std::unique_ptr<cudf::column>> keys;
  std::vector<std::unique_ptr<cudf::column>> codes;
  std::unique_ptr<cudf::column> allKeys;
};

// The strings of each piece of 'pieces', stored in 'storageMr'.
void decodePieces(
    DictionaryPieces& pieces,
    std::vector<std::unique_ptr<cudf::column>>& plain,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref storageMr) {
  for (size_t p = 0; p < pieces.keys.size(); ++p) {
    auto strings = cudf::gather(
        cudf::table_view({pieces.keys[p]->view()}),
        pieces.codes[p]->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        storageMr);
    plain.push_back(std::move(strings->release()[0]));
  }
  stream.sync();
  pieces = DictionaryPieces{};
}

// Replaces the characters of 'column', a kPlain strings column in pinned host
// memory, by LZ4 blocks of kCharBlockBytes compressed with nvCOMP, if the
// blocks and their metadata are smaller. Each round copies up to
// kCompressRoundBlocks blocks to the device, compresses them and copies the
// result back. The offsets stay as they are.
void compressStringChars(
    EncodedColumn& column,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref storageMr) {
  cudf::strings_column_view strings(column.values->view());
  const int64_t numRows = strings.size();
  if (numRows == 0 || column.values->null_count() != 0) {
    return;
  }
  const auto offsets = strings.offsets();
  const int64_t numChars = offsets.type().id() == cudf::type_id::INT64
      ? offsets.data<int64_t>()[numRows]
      : offsets.data<int32_t>()[numRows];
  if (numChars == 0) {
    return;
  }
  const char* chars = strings.parent().data<char>();

  const auto compressOptions = nvcompBatchedLZ4CompressDefaultOpts;
  nvcompAlignmentRequirements_t compressAlignments{};
  checkNvcomp(
      nvcompBatchedLZ4CompressGetRequiredAlignments(
          compressOptions, &compressAlignments),
      "LZ4 compression alignments");
  nvcompAlignmentRequirements_t decompressAlignments{};
  checkNvcomp(
      nvcompBatchedLZ4DecompressGetRequiredAlignments(
          decompressOptions(), &decompressAlignments),
      "LZ4 decompression alignments");
  // Staging places blocks at multiples of kCompressedBlockAlignment and
  // decompresses into kCharBlockBytes strides of 256-byte aligned buffers.
  if (compressAlignments.input > kAllocationAlignment ||
      compressAlignments.output > kAllocationAlignment ||
      compressAlignments.temp > kAllocationAlignment ||
      decompressAlignments.input >
          static_cast<size_t>(kCompressedBlockAlignment) ||
      decompressAlignments.output > kAllocationAlignment ||
      decompressAlignments.temp > kAllocationAlignment) {
    return;
  }
  size_t maxOut = 0;
  checkNvcomp(
      nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
          kCharBlockBytes, compressOptions, &maxOut),
      "LZ4 output bound");
  const int64_t outStride =
      alignUp(static_cast<int64_t>(maxOut), kAllocationAlignment);
  const int64_t blockCount = (numChars + kCharBlockBytes - 1) / kCharBlockBytes;
  const int64_t roundBlocks = std::min(blockCount, kCompressRoundBlocks);
  size_t tempBytes = 0;
  checkNvcomp(
      nvcompBatchedLZ4CompressGetTempSize(
          roundBlocks,
          kCharBlockBytes,
          compressOptions,
          &tempBytes,
          roundBlocks * kCharBlockBytes,
          stream.get()),
      "LZ4 compression scratch size");

  rmm::device_buffer input(roundBlocks * kCharBlockBytes, stream, mr);
  rmm::device_buffer output(roundBlocks * outStride, stream, mr);
  rmm::device_buffer temp(tempBytes, stream, mr);
  rmm::device_uvector<const void*> inputPointers(roundBlocks, stream, mr);
  rmm::device_uvector<size_t> inputSizes(roundBlocks, stream, mr);
  rmm::device_uvector<void*> outputPointers(roundBlocks, stream, mr);
  rmm::device_uvector<size_t> outputSizes(roundBlocks, stream, mr);
  rmm::device_uvector<nvcompStatus_t> statuses(roundBlocks, stream, mr);
  // Pinned, so the copy back of each round is asynchronous.
  rmm::device_buffer hostOutput(roundBlocks * outStride, stream, storageMr);

  std::vector<const void*> hostInputPointers(roundBlocks);
  std::vector<void*> hostOutputPointers(roundBlocks);
  for (int64_t k = 0; k < roundBlocks; ++k) {
    hostInputPointers[k] =
        static_cast<const char*>(input.data()) + k * kCharBlockBytes;
    hostOutputPointers[k] = static_cast<char*>(output.data()) + k * outStride;
  }
  checkCuda(
      cudaMemcpyAsync(
          inputPointers.data(),
          hostInputPointers.data(),
          roundBlocks * sizeof(void*),
          cudaMemcpyHostToDevice,
          stream.get()),
      "copy compression inputs");
  checkCuda(
      cudaMemcpyAsync(
          outputPointers.data(),
          hostOutputPointers.data(),
          roundBlocks * sizeof(void*),
          cudaMemcpyHostToDevice,
          stream.get()),
      "copy compression outputs");

  std::vector<size_t> hostInputSizes(roundBlocks);
  std::vector<size_t> hostOutputSizes(roundBlocks);
  std::vector<nvcompStatus_t> hostStatuses(roundBlocks);
  std::vector<char> compressed;
  std::vector<int64_t> blockStarts;
  std::vector<int64_t> blockSizes;
  blockStarts.reserve(blockCount);
  blockSizes.reserve(blockCount);
  // Declared after every buffer the rounds copy into or out of, so leaving by
  // an error waits for the queued copies before those buffers are freed.
  const StreamSyncGuard syncOnExit{stream.get()};
  for (int64_t first = 0; first < blockCount; first += roundBlocks) {
    const int64_t count = std::min(roundBlocks, blockCount - first);
    const int64_t begin = first * kCharBlockBytes;
    const int64_t bytes =
        std::min(numChars, begin + count * kCharBlockBytes) - begin;
    checkCuda(
        cudaMemcpyAsync(
            input.data(),
            chars + begin,
            bytes,
            cudaMemcpyDefault,
            stream.get()),
        "stage characters to compress");
    for (int64_t k = 0; k < count; ++k) {
      hostInputSizes[k] = static_cast<size_t>(
          std::min(kCharBlockBytes, numChars - (first + k) * kCharBlockBytes));
    }
    checkCuda(
        cudaMemcpyAsync(
            inputSizes.data(),
            hostInputSizes.data(),
            count * sizeof(size_t),
            cudaMemcpyHostToDevice,
            stream.get()),
        "copy block sizes");
    checkNvcomp(
        nvcompBatchedLZ4CompressAsync(
            inputPointers.data(),
            inputSizes.data(),
            kCharBlockBytes,
            count,
            temp.data(),
            tempBytes,
            outputPointers.data(),
            outputSizes.data(),
            compressOptions,
            statuses.data(),
            stream.get()),
        "LZ4 compression");
    checkCuda(
        cudaMemcpyAsync(
            hostOutputSizes.data(),
            outputSizes.data(),
            count * sizeof(size_t),
            cudaMemcpyDeviceToHost,
            stream.get()),
        "copy compressed sizes");
    checkCuda(
        cudaMemcpyAsync(
            hostStatuses.data(),
            statuses.data(),
            count * sizeof(nvcompStatus_t),
            cudaMemcpyDeviceToHost,
            stream.get()),
        "copy compression statuses");
    checkCuda(
        cudaMemcpyAsync(
            hostOutput.data(),
            output.data(),
            count * outStride,
            cudaMemcpyDefault,
            stream.get()),
        "copy compressed blocks");
    stream.sync();
    for (int64_t k = 0; k < count; ++k) {
      check(
          hostStatuses[k] == nvcompSuccess,
          "LZ4 compression of block " + std::to_string(first + k) +
              " of column " + column.name + " failed with nvCOMP status " +
              std::to_string(static_cast<int>(hostStatuses[k])));
      check(
          hostOutputSizes[k] > 0 && hostOutputSizes[k] <= maxOut,
          "LZ4 compression of block " + std::to_string(first + k) +
              " of column " + column.name + " returned an invalid size");
      const int64_t start = alignUp(
          static_cast<int64_t>(compressed.size()), kCompressedBlockAlignment);
      compressed.resize(start + hostOutputSizes[k]);
      std::memcpy(
          compressed.data() + start,
          static_cast<const char*>(hostOutput.data()) + k * outStride,
          hostOutputSizes[k]);
      blockStarts.push_back(start);
      blockSizes.push_back(static_cast<int64_t>(hostOutputSizes[k]));
    }
  }
  // Keep the characters plain unless the blocks and their metadata are
  // smaller.
  const int64_t metadataBytes =
      blockCount * static_cast<int64_t>(2 * sizeof(int64_t));
  if (static_cast<int64_t>(compressed.size()) + metadataBytes >= numChars) {
    return;
  }
  auto stored = std::make_unique<CompressedChars>();
  stored->blockBytes = kCharBlockBytes;
  stored->numChars = numChars;
  stored->blocks = std::make_unique<rmm::device_buffer>(
      compressed.size(), stream, storageMr);
  // Both buffers are host memory: copy on the host, leaving nothing in flight.
  std::memcpy(stored->blocks->data(), compressed.data(), compressed.size());
  stored->blockStarts = std::move(blockStarts);
  stored->blockSizes = std::move(blockSizes);
  auto contents = column.values->release();
  column.values.reset();
  column.stringOffsets = std::move(
      contents.children[cudf::strings_column_view::offsets_column_index]);
  column.compressedChars = std::move(stored);
  // 'contents' frees the plain characters.
}

} // namespace

void EncodedColumnCache::loadTable(
    const std::string& name,
    const std::vector<std::string>& parquetFiles,
    int32_t segmentRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref storageMr,
    bool hostResident,
    bool deltaEncodeSortedSegments,
    bool compressPlainStrings,
    bool scale100Decimals) {
  check(segmentRows > 0, "segmentRows must be positive");

  auto table = std::make_unique<EncodedTable>();
  table->name = name;
  table->segmentRows = segmentRows;
  table->hostResident = hostResident;
  // String columns start as kDictionary and become kPlain once they have more
  // distinct values than a dictionary may hold.
  std::vector<PackedPieces> packed;
  std::vector<DictionaryPieces> dictionaries;
  std::vector<std::vector<std::unique_ptr<cudf::column>>> plain;
  auto toPlain = [&](size_t i) {
    decodePieces(dictionaries[i], plain[i], stream, storageMr);
    table->columns[i]->codec = Codec::kPlain;
  };

  auto encodePiece = [&](cudf::table_view piece) {
    for (size_t i = 0; i < table->columns.size(); ++i) {
      auto& column = *table->columns[i];
      const auto values = piece.column(i);
      if (column.codec == Codec::kDictionary) {
        auto& dictionary = dictionaries[i];
        auto distinct = cudf::distinct(
            cudf::table_view({values}),
            {0},
            cudf::duplicate_keep_option::KEEP_ANY,
            cudf::null_equality::EQUAL,
            cudf::nan_equality::ALL_EQUAL,
            stream,
            mr);
        if (distinct->num_rows() > kMaxDictionaryKeys) {
          toPlain(i);
        } else {
          auto sorted = cudf::sort(distinct->view(), {}, {}, stream, mr);
          auto keys = std::move(sorted->release()[0]);
          auto codes = cudf::lower_bound(
              cudf::table_view({keys->view()}),
              cudf::table_view({values}),
              {cudf::order::ASCENDING},
              {cudf::null_order::AFTER},
              stream,
              storageMr);
          std::vector<cudf::column_view> parts{keys->view()};
          if (dictionary.allKeys) {
            parts.push_back(dictionary.allKeys->view());
          }
          auto all = cudf::concatenate(parts, stream, mr);
          auto allDistinct = cudf::distinct(
              cudf::table_view({all->view()}),
              {0},
              cudf::duplicate_keep_option::KEEP_ANY,
              cudf::null_equality::EQUAL,
              cudf::nan_equality::ALL_EQUAL,
              stream,
              mr);
          dictionary.allKeys = std::move(allDistinct->release()[0]);
          dictionary.keys.push_back(std::move(keys));
          dictionary.codes.push_back(std::move(codes));
          if (dictionary.allKeys->size() > kMaxDictionaryKeys) {
            toPlain(i);
          }
          continue;
        }
      }
      if (column.codec == Codec::kForBitpack) {
        appendPacked(
            column,
            packed[i],
            values,
            segmentRows,
            stream,
            mr,
            storageMr,
            deltaEncodeSortedSegments,
            scale100Decimals);
      } else {
        std::vector<cudf::column_view> parts{values};
        plain[i].push_back(cudf::concatenate(parts, stream, storageMr));
      }
    }
    stream.sync();
  };

  // Encode whole segments as they fill, carrying the rest of each row group
  // into the next.
  std::unique_ptr<cudf::table> carry;
  forEachRowGroup(
      parquetFiles,
      stream,
      mr,
      [&](std::unique_ptr<cudf::table> rows, const auto& metadata, bool last) {
        if (table->columns.empty()) {
          for (size_t i = 0; i < metadata.schema_info.size(); ++i) {
            auto encoded = std::make_unique<EncodedColumn>(stream, mr);
            encoded->name = metadata.schema_info[i].name;
            encoded->type = rows->view().column(i).type();
            encoded->codec = isIntegerLike(encoded->type) ? Codec::kForBitpack
                : encoded->type.id() == cudf::type_id::STRING
                ? Codec::kDictionary
                : Codec::kPlain;
            table->columnIndex.emplace(encoded->name, static_cast<int32_t>(i));
            table->columns.push_back(std::move(encoded));
          }
          packed.resize(table->columns.size());
          dictionaries.resize(table->columns.size());
          plain.resize(table->columns.size());
        }
        table->numRows += rows->num_rows();
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
  check(!table->columns.empty(), "No row groups in " + name);
  table->numSegments =
      static_cast<int32_t>((table->numRows + segmentRows - 1) / segmentRows);

  for (size_t i = 0; i < table->columns.size(); ++i) {
    auto& column = *table->columns[i];
    if (column.codec == Codec::kDictionary &&
        int64_t{dictionaries[i].allKeys->size()} * 8 >= table->numRows) {
      toPlain(i);
    }
    if (column.codec == Codec::kDictionary) {
      auto& dictionary = dictionaries[i];
      auto sorted = cudf::stable_sort(
          cudf::table_view({dictionary.allKeys->view()}), {}, {}, stream, mr);
      column.values = std::move(sorted->release()[0]);
      for (size_t p = 0; p < dictionary.keys.size(); ++p) {
        auto recode = cudf::lower_bound(
            cudf::table_view({column.values->view()}),
            cudf::table_view({dictionary.keys[p]->view()}),
            {cudf::order::ASCENDING},
            {cudf::null_order::AFTER},
            stream,
            mr);
        auto codes = cudf::gather(
            cudf::table_view({recode->view()}),
            dictionary.codes[p]->view(),
            cudf::out_of_bounds_policy::DONT_CHECK,
            stream,
            mr);
        appendPacked(
            column,
            packed[i],
            codes->view().column(0),
            segmentRows,
            stream,
            mr,
            storageMr,
            /*allowDelta=*/false,
            /*allowScale100=*/false);
        dictionary.keys[p].reset();
        dictionary.codes[p].reset();
      }
      dictionaries[i] = DictionaryPieces{};
    }
    if (column.codec == Codec::kPlain) {
      if (plain[i].size() == 1) {
        column.values = std::move(plain[i][0]);
      } else {
        std::vector<cudf::column_view> parts;
        for (const auto& part : plain[i]) {
          parts.push_back(part->view());
        }
        column.values = cudf::concatenate(parts, stream, storageMr);
      }
      plain[i].clear();
      if (compressPlainStrings && hostResident &&
          column.values->type().id() == cudf::type_id::STRING) {
        compressStringChars(column, stream, mr, storageMr);
      }
    } else {
      finishPacked(column, packed[i], hostResident, stream, mr, storageMr);
    }
    stream.sync();
  }
  tables_[name] = std::move(table);
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

// Ascending runs of adjacent segments, as [first, last] segment ids.
std::vector<std::pair<int32_t, int32_t>> segmentRuns(
    std::span<const int32_t> segments) {
  std::vector<std::pair<int32_t, int32_t>> runs;
  for (auto segment : segments) {
    if (!runs.empty() && runs.back().second + 1 == segment) {
      runs.back().second = segment;
    } else {
      runs.emplace_back(segment, segment);
    }
  }
  return runs;
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
    staged.segmentDelta.push_back((p.flags & kSegmentDelta) != 0 ? 1 : 0);
    staged.segmentScaled.push_back((p.flags & kSegmentScale100) != 0 ? 1 : 0);
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
  for (const auto& [first, last] : segmentRuns(segments)) {
    uint64_t runWords = 0;
    for (int32_t s = first; s <= last; ++s) {
      runWords += wordsOf(s);
    }
    deferred.push_back(
        [dst = staged.packed.data() + word,
         src = column.packed.data() + params[first].bitOffset / 64,
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
    cudf::strings_column_view strings,
    std::span<const int32_t> segments,
    Deferred& deferred,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto* offsets = strings.offsets().data<OffsetT>();
  const char* chars = strings.parent().data<char>();
  std::vector<std::pair<int64_t, int64_t>> rowRuns;
  int64_t numRows = 0;
  int64_t numChars = 0;
  for (const auto& [first, last] : segmentRuns(segments)) {
    const int64_t begin = int64_t{first} * table.segmentRows;
    const int64_t end =
        int64_t{last} * table.segmentRows + table.segmentSize(last);
    rowRuns.emplace_back(begin, end);
    numRows += end - begin;
    numChars += offsets[end] - offsets[begin];
  }
  auto stagedOffsets = cudf::make_fixed_width_column(
      strings.offsets().type(),
      static_cast<cudf::size_type>(numRows + 1),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* outOffsets = stagedOffsets->mutable_view().data<OffsetT>();
  rmm::device_buffer stagedChars(numChars, stream, mr);
  auto* outChars = static_cast<char*>(stagedChars.data());
  int64_t row = 0;
  int64_t charBase = 0;
  for (const auto& [begin, end] : rowRuns) {
    const int64_t runRows = end - begin;
    const int64_t runChars = offsets[end] - offsets[begin];
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

// Stages the rows of 'segments' of a strings column whose characters are LZ4
// blocks. Copies the compressed blocks each run of segments covers, then
// decompresses them on 'stream' into a scratch buffer of whole blocks and
// copies each run's characters out of it. Only the covering blocks of the
// staged rows are copied and decompressed. The decompression arguments are
// written on the host into pinned memory from cuDF's pool, owned by the
// DecompressionCheck that 'staged' keeps, and copied with the other pinned
// copies: nothing here allocates pinned memory directly or copies from
// pageable memory. decodeColumn() verifies the check before reading 'staged'.
template <typename OffsetT>
std::unique_ptr<cudf::column> stageCompressedStrings(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments,
    EncodedColumn& staged,
    Deferred& deferred,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto& compressed = *column.compressedChars;
  const auto* offsets = column.stringOffsets->view().data<OffsetT>();
  const auto* blocks = static_cast<const char*>(compressed.blocks->data());
  const int64_t blockBytes = compressed.blockBytes;

  struct Run {
    int64_t beginRow;
    int64_t endRow;
    int64_t charBegin;
    int64_t charEnd;
    // Covering blocks, none when lastBlock < firstBlock.
    int64_t firstBlock{0};
    int64_t lastBlock{-1};
    // Where the run's compressed blocks go in the device copy, and the index
    // of its first block in the scratch buffer.
    int64_t compressedBase{0};
    int64_t compressedBytes{0};
    int64_t scratchBlock{0};
  };
  std::vector<Run> runs;
  int64_t numRows = 0;
  int64_t numChars = 0;
  int64_t compressedBytes = 0;
  int64_t blockCount = 0;
  for (const auto& [first, last] : segmentRuns(segments)) {
    Run run;
    run.beginRow = int64_t{first} * table.segmentRows;
    run.endRow = int64_t{last} * table.segmentRows + table.segmentSize(last);
    run.charBegin = offsets[run.beginRow];
    run.charEnd = offsets[run.endRow];
    numRows += run.endRow - run.beginRow;
    numChars += run.charEnd - run.charBegin;
    if (run.charEnd > run.charBegin) {
      run.firstBlock = run.charBegin / blockBytes;
      run.lastBlock = (run.charEnd - 1) / blockBytes;
      run.compressedBase = compressedBytes;
      run.compressedBytes = compressed.blockStarts[run.lastBlock] +
          compressed.blockSizes[run.lastBlock] -
          compressed.blockStarts[run.firstBlock];
      compressedBytes +=
          alignUp(run.compressedBytes, kCompressedBlockAlignment);
      run.scratchBlock = blockCount;
      blockCount += run.lastBlock - run.firstBlock + 1;
    }
    runs.push_back(run);
  }
  auto stagedOffsets = cudf::make_fixed_width_column(
      column.stringOffsets->type(),
      static_cast<cudf::size_type>(numRows + 1),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* outOffsets = stagedOffsets->mutable_view().data<OffsetT>();
  rmm::device_buffer stagedChars(numChars, stream, mr);
  auto* outChars = static_cast<char*>(stagedChars.data());

  // Device buffers, released in stream order on 'stream' after the work the
  // deferred lambdas queue on it.
  auto deviceBlocks =
      std::make_shared<rmm::device_buffer>(compressedBytes, stream, mr);
  auto scratch =
      std::make_shared<rmm::device_buffer>(blockCount * blockBytes, stream, mr);
  std::shared_ptr<DecompressionCheck> decompressionCheck;
  std::shared_ptr<rmm::device_buffer> deviceArguments;
  std::shared_ptr<rmm::device_buffer> temp;
  size_t tempBytes = 0;
  if (blockCount > 0) {
    std::vector<size_t> expectedSizes;
    expectedSizes.reserve(blockCount);
    for (const auto& run : runs) {
      for (int64_t b = run.firstBlock; b <= run.lastBlock; ++b) {
        expectedSizes.push_back(
            static_cast<size_t>(
                std::min(blockBytes, compressed.numChars - b * blockBytes)));
      }
    }
    decompressionCheck = std::make_shared<DecompressionCheck>(
        column.name, std::move(expectedSizes), stream);
    // The pinned arguments are written before any copy that reads them is
    // queued, and not written again.
    auto* inputPointers = decompressionCheck->inputPointers();
    auto* inputSizes = decompressionCheck->inputSizes();
    auto* outputPointers = decompressionCheck->outputPointers();
    auto* capacities = decompressionCheck->capacities();
    int64_t i = 0;
    for (const auto& run : runs) {
      for (int64_t b = run.firstBlock; b <= run.lastBlock; ++b, ++i) {
        inputPointers[i] = static_cast<const char*>(deviceBlocks->data()) +
            run.compressedBase +
            (compressed.blockStarts[b] -
             compressed.blockStarts[run.firstBlock]);
        inputSizes[i] = static_cast<size_t>(compressed.blockSizes[b]);
        outputPointers[i] = static_cast<char*>(scratch->data()) +
            (run.scratchBlock + b - run.firstBlock) * blockBytes;
        capacities[i] = static_cast<size_t>(blockBytes);
      }
    }
    deviceArguments = std::make_shared<rmm::device_buffer>(
        decompressionCheck->bytes(), stream, mr);
    checkNvcomp(
        nvcompBatchedLZ4DecompressGetTempSize(
            blockCount,
            blockBytes,
            decompressOptions(),
            &tempBytes,
            blockCount * blockBytes,
            stream.get()),
        "LZ4 decompression scratch size");
    temp = std::make_shared<rmm::device_buffer>(tempBytes, stream, mr);
    staged.decompressionCheck = decompressionCheck;
  }

  // Copies from pinned memory, then the decompression and trims that read
  // them, all queued last on 'stream'.
  int64_t row = 0;
  int64_t charBase = 0;
  for (const auto& run : runs) {
    const int64_t runRows = run.endRow - run.beginRow;
    const auto delta = static_cast<OffsetT>(charBase - run.charBegin);
    const char* source = run.lastBlock >= run.firstBlock
        ? blocks + compressed.blockStarts[run.firstBlock]
        : nullptr;
    deferred.push_back([=] {
      if (source != nullptr) {
        copyToDevice(
            static_cast<char*>(deviceBlocks->data()) + run.compressedBase,
            source,
            run.compressedBytes,
            stream);
      }
      copyToDevice(
          outOffsets + row,
          offsets + run.beginRow,
          (runRows + 1) * sizeof(OffsetT),
          stream);
      addToOffsetsKernel<OffsetT>
          <<<numBlocks(runRows + 1), kThreads, 0, stream.get()>>>(
              outOffsets + row, runRows + 1, delta);
      checkCuda(cudaGetLastError(), "addToOffsetsKernel");
    });
    row += runRows;
    charBase += run.charEnd - run.charBegin;
  }
  if (blockCount > 0) {
    deferred.push_back([=] {
      auto* mirror = static_cast<char*>(deviceArguments->data());
      decompressionCheck->copyArguments(mirror);
      checkNvcomp(
          nvcompBatchedLZ4DecompressAsync(
              reinterpret_cast<const void* const*>(
                  mirror + decompressionCheck->inputPointersOffset()),
              reinterpret_cast<const size_t*>(
                  mirror + decompressionCheck->inputSizesOffset()),
              reinterpret_cast<const size_t*>(
                  mirror + decompressionCheck->capacitiesOffset()),
              reinterpret_cast<size_t*>(
                  mirror + decompressionCheck->actualSizesOffset()),
              blockCount,
              temp->data(),
              tempBytes,
              reinterpret_cast<void* const*>(
                  mirror + decompressionCheck->outputPointersOffset()),
              decompressOptions(),
              reinterpret_cast<nvcompStatus_t*>(
                  mirror + decompressionCheck->statusesOffset()),
              stream.get()),
          "LZ4 decompression");
      int64_t base = 0;
      for (const auto& run : runs) {
        const int64_t runChars = run.charEnd - run.charBegin;
        if (runChars > 0) {
          checkCuda(
              cudaMemcpyAsync(
                  outChars + base,
                  static_cast<const char*>(scratch->data()) +
                      run.scratchBlock * blockBytes +
                      (run.charBegin - run.firstBlock * blockBytes),
                  runChars,
                  cudaMemcpyDeviceToDevice,
                  stream.get()),
              "trim decompressed characters");
        }
        base += runChars;
      }
      decompressionCheck->record(mirror);
    });
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
    cudf::column_view values,
    std::span<const int32_t> segments,
    Deferred& deferred,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto width = cudf::size_of(values.type());
  int64_t numRows = 0;
  for (auto segment : segments) {
    numRows += table.segmentSize(segment);
  }
  auto staged = cudf::make_fixed_width_column(
      values.type(),
      static_cast<cudf::size_type>(numRows),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* out = static_cast<char*>(staged->mutable_view().head());
  const auto* in =
      static_cast<const char*>(values.head()) + values.offset() * width;
  int64_t row = 0;
  for (const auto& [first, last] : segmentRuns(segments)) {
    const int64_t begin = int64_t{first} * table.segmentRows;
    const int64_t end =
        int64_t{last} * table.segmentRows + table.segmentSize(last);
    deferred.push_back([dst = out + row * width,
                        src = in + begin * width,
                        bytes = (end - begin) * width,
                        stream] { copyToDevice(dst, src, bytes, stream); });
    row += end - begin;
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
      case Codec::kPlain:
        if (column.compressedChars != nullptr) {
          out->values =
              column.stringOffsets->type().id() == cudf::type_id::INT64
              ? stageCompressedStrings<int64_t>(
                    table, column, segments, *out, deferred, stream, mr)
              : stageCompressedStrings<int32_t>(
                    table, column, segments, *out, deferred, stream, mr);
        } else if (column.values->type().id() == cudf::type_id::STRING) {
          cudf::strings_column_view strings(column.values->view());
          out->values = strings.offsets().type().id() == cudf::type_id::INT64
              ? stageStrings<int64_t>(
                    table, strings, segments, deferred, stream, mr)
              : stageStrings<int32_t>(
                    table, strings, segments, deferred, stream, mr);
        } else {
          out->values = stageFixedWidth(
              table, column.values->view(), segments, deferred, stream, mr);
        }
        break;
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
