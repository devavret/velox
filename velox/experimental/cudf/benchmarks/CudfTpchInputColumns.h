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

#include "velox/core/PlanNode.h"

#include <set>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Columns required by one scan, including filter-only columns.
struct TpchScanColumns {
  core::PlanNodeId scanId;
  std::string table;
  std::set<std::string> columns;
};

/// Inspects the existing plan without changing it. Cache preparation takes the
/// union per table; filtering and aggregation remain part of query execution.
std::vector<TpchScanColumns> tpchScanColumns(const core::PlanNodePtr& plan);

} // namespace facebook::velox::cudf_velox
