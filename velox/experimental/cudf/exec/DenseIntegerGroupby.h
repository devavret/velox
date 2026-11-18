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
#pragma once

#include <cudf/groupby.hpp>

#include <optional>

namespace facebook::velox::cudf_velox {

using DenseIntegerGroupbyResult = std::pair<
    std::unique_ptr<cudf::table>,
    std::vector<cudf::groupby::aggregation_result>>;

// A bounded direct-address aggregation for one non-null INT32 key and one
// non-null COUNT or positive INT64 SUM. Unsupported or sparse domains return
// nullopt. Input ordering and output ordering are not assumed.
std::optional<DenseIntegerGroupbyResult> tryDenseIntegerGroupby(
    cudf::table_view keys,
    const std::vector<cudf::groupby::aggregation_request>& requests,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref tempMr,
    rmm::device_async_resource_ref outputMr);

} // namespace facebook::velox::cudf_velox
