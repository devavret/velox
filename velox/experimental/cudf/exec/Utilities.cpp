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

#include "velox/experimental/cudf/exec/Utilities.h"

#include <cudf/concatenate.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <common/base/Exceptions.h>

#include <cstdlib>
#include <memory>
#include <string_view>

namespace facebook::velox::cudf_velox {

namespace {
[[nodiscard]] auto makeCudaMr() {
  return std::make_shared<rmm::mr::cuda_memory_resource>();
}

[[nodiscard]] auto makePoolMr() {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      makeCudaMr(), rmm::percent_of_free_device_memory(50));
}

[[nodiscard]] auto makeAsyncMr() {
  return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

[[nodiscard]] auto makeManagedMr() {
  return std::make_shared<rmm::mr::managed_memory_resource>();
}

[[nodiscard]] auto makeArenaMr() {
  return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(
      makeCudaMr());
}

[[nodiscard]] auto makeManagedPoolMr() {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      makeManagedMr(), rmm::percent_of_free_device_memory(50));
}

template <typename Upstream>
[[nodiscard]] auto makeStatisticsMr(std::shared_ptr<Upstream> upstream) {
  return rmm::mr::make_owning_wrapper<rmm::mr::statistics_resource_adaptor>(
      std::move(upstream));
}

} // namespace

std::shared_ptr<rmm::mr::device_memory_resource> createMemoryResource(
    std::string_view mode) {
  if (mode == "cuda")
    return makeCudaMr();
  if (mode == "pool")
    return makePoolMr();
  if (mode == "async")
    return makeAsyncMr();
  if (mode == "arena")
    return makeArenaMr();
  if (mode == "managed")
    return makeManagedMr();
  if (mode == "managed_pool")
    return makeManagedPoolMr();
  if (mode == "stats_cuda")
    return makeStatisticsMr(makeCudaMr());
  if (mode == "stats_pool")
    return makeStatisticsMr(makePoolMr());
  if (mode == "stats_async")
    return makeStatisticsMr(makeAsyncMr());
  if (mode == "stats_arena")
    return makeStatisticsMr(makeArenaMr());
  if (mode == "stats_managed")
    return makeStatisticsMr(makeManagedMr());
  if (mode == "stats_managed_pool")
    return makeStatisticsMr(makeManagedPoolMr());
  VELOX_FAIL(
      "Unknown memory resource mode: " + std::string(mode) +
      "\nExpecting: cuda, pool, async, arena, managed, managed_pool, " +
      "stats_cuda, stats_pool, stats_async, stats_arena, stats_managed, or stats_managed_pool");
}

template <typename Upstream>
std::shared_ptr<rmm::mr::device_memory_resource> createStatisticsMemoryResource(
    std::shared_ptr<Upstream> upstream) {
  return makeStatisticsMr(upstream);
}

// Explicit template instantiations for common resource types
template std::shared_ptr<rmm::mr::device_memory_resource>
    createStatisticsMemoryResource<rmm::mr::cuda_memory_resource>(
        std::shared_ptr<rmm::mr::cuda_memory_resource>);
template std::shared_ptr<rmm::mr::device_memory_resource>
    createStatisticsMemoryResource<
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
        std::shared_ptr<
            rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>);
template std::shared_ptr<rmm::mr::device_memory_resource>
    createStatisticsMemoryResource<rmm::mr::cuda_async_memory_resource>(
        std::shared_ptr<rmm::mr::cuda_async_memory_resource>);
template std::shared_ptr<rmm::mr::device_memory_resource>
    createStatisticsMemoryResource<
        rmm::mr::arena_memory_resource<rmm::mr::cuda_memory_resource>>(
        std::shared_ptr<
            rmm::mr::arena_memory_resource<rmm::mr::cuda_memory_resource>>);
template std::shared_ptr<rmm::mr::device_memory_resource>
    createStatisticsMemoryResource<rmm::mr::managed_memory_resource>(
        std::shared_ptr<rmm::mr::managed_memory_resource>);

template <typename Upstream>
rmm::mr::statistics_resource_adaptor<Upstream>* getStatisticsResourceAdaptor(
    rmm::mr::device_memory_resource* resource) {
  if (resource == nullptr) {
    return nullptr;
  }

  // First try direct cast
  auto directStats =
      dynamic_cast<rmm::mr::statistics_resource_adaptor<Upstream>*>(resource);
  if (directStats != nullptr) {
    return directStats;
  }

  // Then try to get it from owning_wrapper
  using WrappedStats = rmm::mr::
      owning_wrapper<rmm::mr::statistics_resource_adaptor<Upstream>, Upstream>;
  auto wrappedStats = dynamic_cast<WrappedStats*>(resource);
  if (wrappedStats != nullptr) {
    return &wrappedStats->wrapped();
  }

  return nullptr;
}

// Explicit template instantiations for common resource types
template rmm::mr::statistics_resource_adaptor<rmm::mr::cuda_memory_resource>*
getStatisticsResourceAdaptor<rmm::mr::cuda_memory_resource>(
    rmm::mr::device_memory_resource*);
template rmm::mr::statistics_resource_adaptor<
    rmm::mr::cuda_async_memory_resource>*
getStatisticsResourceAdaptor<rmm::mr::cuda_async_memory_resource>(
    rmm::mr::device_memory_resource*);
template rmm::mr::statistics_resource_adaptor<rmm::mr::managed_memory_resource>*
getStatisticsResourceAdaptor<rmm::mr::managed_memory_resource>(
    rmm::mr::device_memory_resource*);
template rmm::mr::statistics_resource_adaptor<
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>*
getStatisticsResourceAdaptor<
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
    rmm::mr::device_memory_resource*);
template rmm::mr::statistics_resource_adaptor<
    rmm::mr::arena_memory_resource<rmm::mr::cuda_memory_resource>>*
getStatisticsResourceAdaptor<
    rmm::mr::arena_memory_resource<rmm::mr::cuda_memory_resource>>(
    rmm::mr::device_memory_resource*);

void* tryGetStatisticsResourceAdaptor(
    rmm::mr::device_memory_resource* resource) {
  if (resource == nullptr) {
    return nullptr;
  }

  // Try each common upstream type
  if (auto stats = getStatisticsResourceAdaptor<rmm::mr::cuda_memory_resource>(
          resource)) {
    return stats;
  }
  if (auto stats =
          getStatisticsResourceAdaptor<rmm::mr::cuda_async_memory_resource>(
              resource)) {
    return stats;
  }
  if (auto stats =
          getStatisticsResourceAdaptor<rmm::mr::managed_memory_resource>(
              resource)) {
    return stats;
  }
  if (auto stats = getStatisticsResourceAdaptor<
          rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
          resource)) {
    return stats;
  }
  if (auto stats = getStatisticsResourceAdaptor<
          rmm::mr::arena_memory_resource<rmm::mr::cuda_memory_resource>>(
          resource)) {
    return stats;
  }

  return nullptr;
}

cudf::detail::cuda_stream_pool& cudfGlobalStreamPool() {
  return cudf::detail::global_cuda_stream_pool();
};

std::unique_ptr<cudf::table> concatenateTables(
    std::vector<std::unique_ptr<cudf::table>> tables,
    rmm::cuda_stream_view stream) {
  // Check for empty vector
  VELOX_CHECK_GT(tables.size(), 0);

  if (tables.size() == 1) {
    return std::move(tables[0]);
  }
  std::vector<cudf::table_view> tableViews;
  tableViews.reserve(tables.size());
  std::transform(
      tables.begin(),
      tables.end(),
      std::back_inserter(tableViews),
      [&](const auto& tbl) { return tbl->view(); });
  return cudf::concatenate(
      tableViews, stream, cudf::get_current_device_resource_ref());
}

std::unique_ptr<cudf::table> getConcatenatedTable(
    std::vector<CudfVectorPtr>& tables,
    rmm::cuda_stream_view stream) {
  // Check for empty vector
  VELOX_CHECK_GT(tables.size(), 0);

  auto inputStreams = std::vector<rmm::cuda_stream_view>();
  auto tableViews = std::vector<cudf::table_view>();

  inputStreams.reserve(tables.size());
  tableViews.reserve(tables.size());

  for (const auto& table : tables) {
    VELOX_CHECK_NOT_NULL(table);
    tableViews.push_back(table->getTableView());
    inputStreams.push_back(table->stream());
  }

  cudf::detail::join_streams(inputStreams, stream);

  if (tables.size() == 1) {
    return tables[0]->release();
  }

  auto output = cudf::concatenate(
      tableViews, stream, cudf::get_current_device_resource_ref());
  stream.synchronize();
  return output;
}

} // namespace facebook::velox::cudf_velox
