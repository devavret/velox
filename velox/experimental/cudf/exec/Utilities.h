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

#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/table/table.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>
#include <string_view>

namespace rmm {
namespace mr {
template <typename Upstream>
class statistics_resource_adaptor;
}
} // namespace rmm

namespace facebook::velox::cudf_velox {

/**
 * @brief Creates a memory resource based on the given mode.
 */
[[nodiscard]] std::shared_ptr<rmm::mr::device_memory_resource>
createMemoryResource(std::string_view mode);

/**
 * @brief Create a statistics resource adapter that wraps around an existing
 * memory resource
 *
 * @tparam Upstream Type of the upstream resource
 * @param upstream The upstream memory resource to wrap with statistics tracking
 * @return std::shared_ptr<rmm::mr::device_memory_resource> A statistics
 * resource adapter
 */
template <typename Upstream>
std::shared_ptr<rmm::mr::device_memory_resource> createStatisticsMemoryResource(
    std::shared_ptr<Upstream> upstream);

/**
 * @brief Extract the statistics_resource_adaptor from a device_memory_resource
 * pointer
 *
 * This works whether the statistics_resource_adaptor is directly the resource
 * or is wrapped in an owning_wrapper.
 *
 * @tparam Upstream The upstream resource type of the
 * statistics_resource_adaptor
 * @param resource Pointer to a device_memory_resource that might be or contain
 * a statistics_resource_adaptor
 * @return rmm::mr::statistics_resource_adaptor<Upstream>* Pointer to the
 * statistics_resource_adaptor if found, nullptr otherwise
 */
template <typename Upstream>
rmm::mr::statistics_resource_adaptor<Upstream>* getStatisticsResourceAdaptor(
    rmm::mr::device_memory_resource* resource);

/**
 * @brief Try to extract the statistics_resource_adaptor from a
 * device_memory_resource pointer without knowing the upstream type
 *
 * This attempts to extract a statistics resource adaptor with common upstream
 * types.
 *
 * @param resource Pointer to a device_memory_resource that might be or contain
 * a statistics_resource_adaptor
 * @return void* Pointer to the statistics_resource_adaptor if found (must be
 * cast to appropriate type), nullptr otherwise
 */
void* tryGetStatisticsResourceAdaptor(
    rmm::mr::device_memory_resource* resource);

/**
 * @brief Returns the global CUDA stream pool used by cudf.
 */
[[nodiscard]] cudf::detail::cuda_stream_pool& cudfGlobalStreamPool();

// Concatenate a vector of cuDF tables into a single table
[[nodiscard]] std::unique_ptr<cudf::table> concatenateTables(
    std::vector<std::unique_ptr<cudf::table>> tables,
    rmm::cuda_stream_view stream);

// Concatenate a vector of cuDF tables into a single table.
// This function joins the streams owned by individual tables on the passed
// stream. Inputs are not safe to use after calling this function.
[[nodiscard]] std::unique_ptr<cudf::table> getConcatenatedTable(
    std::vector<CudfVectorPtr>& tables,
    rmm::cuda_stream_view stream);

} // namespace facebook::velox::cudf_velox
