/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/cudf/exec/DenseIntegerGroupby.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <limits>
#include <type_traits>

namespace facebook::velox::cudf_velox {
namespace {

// Inspired by the direct-address count accumulation observed in Sirius's
// dense_count_join_impl.cu. Here the bounds are measured on each input batch,
// and the result obeys the ordinary cuDF groupby request/result interface.
template <typename CountT, bool Sum>
__global__ void accumulateDense(
    const int32_t* keys,
    const int64_t* values,
    int32_t rows,
    int32_t minKey,
    CountT* bins) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < rows;
       i += stride) {
    const auto offset =
        static_cast<uint32_t>(keys[i]) - static_cast<uint32_t>(minKey);
    if constexpr (Sum) {
      atomicAdd(
          reinterpret_cast<unsigned long long*>(bins + offset),
          static_cast<unsigned long long>(values[i]));
    } else {
      atomicAdd(bins + offset, int32_t{1});
    }
  }
}

template <typename T>
std::pair<T, T> bounds(
    cudf::column_view column,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto [low, high] = cudf::minmax(column, stream, mr);
  return {
      static_cast<const cudf::numeric_scalar<T>&>(*low).value(stream),
      static_cast<const cudf::numeric_scalar<T>&>(*high).value(stream)};
}

} // namespace

std::optional<DenseIntegerGroupbyResult> tryDenseIntegerGroupby(
    cudf::table_view keys,
    const std::vector<cudf::groupby::aggregation_request>& requests,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref tempMr,
    rmm::device_async_resource_ref outputMr) {
  if (keys.num_columns() != 1 || keys.num_rows() == 0 ||
      keys.column(0).type().id() != cudf::type_id::INT32 ||
      keys.column(0).null_count() != 0 || requests.size() != 1 ||
      requests[0].aggregations.size() != 1 ||
      requests[0].values.size() != keys.num_rows() ||
      requests[0].values.null_count() != 0) {
    return std::nullopt;
  }
  const auto kind = requests[0].aggregations[0]->kind;
  const bool sum = kind == cudf::aggregation::SUM;
  if (!sum && kind != cudf::aggregation::COUNT_VALID &&
      kind != cudf::aggregation::COUNT_ALL) {
    return std::nullopt;
  }
  if (sum && requests[0].values.type().id() != cudf::type_id::INT64) {
    return std::nullopt;
  }
  const auto [minKey, maxKey] = bounds<int32_t>(keys.column(0), stream, tempMr);
  const int64_t slots = static_cast<int64_t>(maxKey) - minKey + 1;
  // Bound the state to 64 MiB for COUNT or 128 MiB for SUM. Avoid allocating
  // a mostly empty domain when a hash aggregation is a better fit.
  if (slots <= 0 || slots > (int64_t{1} << 24) ||
      slots > int64_t{2} * keys.num_rows()) {
    return std::nullopt;
  }
  if (sum) {
    const auto [minValue, maxValue] =
        bounds<int64_t>(requests[0].values, stream, tempMr);
    // Positive values make zero an unambiguous empty-bin marker. Prove that
    // even the sum of the entire batch fits, so atomics cannot wrap to zero.
    if (minValue <= 0 ||
        static_cast<__int128>(maxValue) * keys.num_rows() >
            std::numeric_limits<int64_t>::max()) {
      return std::nullopt;
    }
  }

  const auto dtype =
      cudf::data_type{sum ? cudf::type_id::INT64 : cudf::type_id::INT32};
  auto bins = cudf::make_numeric_column(
      dtype,
      static_cast<cudf::size_type>(slots),
      cudf::mask_state::UNALLOCATED,
      stream,
      tempMr);
  CUDF_CUDA_TRY(cudaMemsetAsync(
      bins->mutable_view().head<void>(),
      0,
      slots * (sum ? sizeof(int64_t) : sizeof(int32_t)),
      stream.get()));
  constexpr int threads = 256;
  const auto blocks = static_cast<int>(std::min<int64_t>(
      65535, (static_cast<int64_t>(keys.num_rows()) + threads - 1) / threads));
  if (sum) {
    accumulateDense<int64_t, true><<<blocks, threads, 0, stream.get()>>>(
        keys.column(0).data<int32_t>(),
        requests[0].values.data<int64_t>(),
        keys.num_rows(),
        minKey,
        bins->mutable_view().data<int64_t>());
  } else {
    accumulateDense<int32_t, false><<<blocks, threads, 0, stream.get()>>>(
        keys.column(0).data<int32_t>(),
        nullptr,
        keys.num_rows(),
        minKey,
        bins->mutable_view().data<int32_t>());
  }
  CUDF_CUDA_TRY(cudaGetLastError());

  cudf::numeric_scalar<int32_t> initial(minKey, true, stream, tempMr);
  cudf::numeric_scalar<int32_t> step(1, true, stream, tempMr);
  auto denseKeys = cudf::sequence(
      static_cast<cudf::size_type>(slots), initial, step, stream, tempMr);
  std::unique_ptr<cudf::column> occupied;
  if (sum) {
    cudf::numeric_scalar<int64_t> zero(0, true, stream, tempMr);
    occupied = cudf::binary_operation(
        bins->view(),
        zero,
        cudf::binary_operator::GREATER,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        tempMr);
  } else {
    cudf::numeric_scalar<int32_t> zero(0, true, stream, tempMr);
    occupied = cudf::binary_operation(
        bins->view(),
        zero,
        cudf::binary_operator::GREATER,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        tempMr);
  }
  auto selected = cudf::apply_retention_mask(
                      cudf::table_view{{denseKeys->view(), bins->view()}},
                      occupied->view(),
                      stream,
                      outputMr)
                      ->release();
  std::vector<std::unique_ptr<cudf::column>> keyColumns;
  keyColumns.push_back(std::move(selected[0]));
  std::vector<cudf::groupby::aggregation_result> results(1);
  results[0].results.push_back(std::move(selected[1]));
  return DenseIntegerGroupbyResult{
      std::make_unique<cudf::table>(std::move(keyColumns)), std::move(results)};
}

} // namespace facebook::velox::cudf_velox
