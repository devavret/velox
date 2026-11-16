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

#include "velox/experimental/cudf/benchmarks/CudfTpchInputColumns.h"

#include "velox/connectors/common/FileColumnHandle.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/Expressions.h"

namespace facebook::velox::cudf_velox {

namespace {

// Adds the top-level input columns 'expr' reads to 'columns'.
void addInputColumns(
    const core::TypedExprPtr& expr,
    std::set<std::string>& columns) {
  if (expr == nullptr) {
    return;
  }
  if (const auto* field =
          dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get());
      field != nullptr && field->isInputColumn()) {
    columns.insert(field->name());
  }
  for (const auto& input : expr->inputs()) {
    addInputColumns(input, columns);
  }
}

void addScanColumns(
    const core::PlanNodePtr& node,
    std::vector<TpchScanColumns>& scans) {
  if (const auto* scan = dynamic_cast<const core::TableScanNode*>(node.get())) {
    const auto handle =
        std::dynamic_pointer_cast<const connector::hive::HiveTableHandle>(
            scan->tableHandle());
    VELOX_CHECK_NOT_NULL(
        handle, "Table scan {} does not have a Hive table handle", scan->id());
    TpchScanColumns result;
    result.scanId = scan->id();
    result.table = handle->tableName();
    for (const auto& [_, column] : scan->assignments()) {
      result.columns.insert(
          checkedPointerCast<const connector::FileColumnHandle>(column)
              ->name());
    }
    for (const auto& [subfield, _] : handle->subfieldFilters()) {
      result.columns.insert(subfield.baseName());
    }
    addInputColumns(handle->remainingFilter(), result.columns);
    scans.push_back(std::move(result));
  }
  for (const auto& source : node->sources()) {
    addScanColumns(source, scans);
  }
}

} // namespace

std::vector<TpchScanColumns> tpchScanColumns(const core::PlanNodePtr& plan) {
  std::vector<TpchScanColumns> scans;
  addScanColumns(plan, scans);
  return scans;
}

} // namespace facebook::velox::cudf_velox
