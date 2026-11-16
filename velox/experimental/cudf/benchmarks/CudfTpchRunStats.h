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

#include "velox/exec/TaskStats.h"

#include <folly/dynamic.h>

#include <string>
#include <string_view>

namespace facebook::velox::cudf_velox {

/// Converts the statistics a task recorded for one run of a query to JSON, for
/// the per-repeat files velox_cudf_tpch_benchmark writes. Every counter is
/// copied as the exact integer the engine recorded, in the unit the engine
/// recorded it in: nothing is rounded, scaled or derived from another counter.
/// Keys that hold a quantity end in its unit (_bytes, _nanos, _ms).
class CudfTpchRunStats {
 public:
  /// Returns {"unit", "sum", "count", "min", "max"} of 'metric', "unit" being
  /// "none", "nanos" or "bytes". "sum", "min" and "max" are in that unit.
  static folly::dynamic toJson(const RuntimeMetric& metric);

  /// Returns the statistics of one operator of one pipeline, summed over the
  /// drivers that ran it: its "operator_id", "pipeline_id", "plan_node_id" and
  /// "operator_type", its row, byte and timing counters, and every entry of
  /// its runtime statistics under "runtime_stats", keyed by name.
  static folly::dynamic toJson(const velox::exec::OperatorStats& stats);

  /// Returns the task-level counters of 'stats' and, under "pipelines", one
  /// entry per pipeline with its "operators" and the runtime statistics of
  /// each of its "drivers".
  static folly::dynamic toJson(const velox::exec::TaskStats& stats);

  /// Returns the sum of runtime statistic 'name' over every operator of
  /// 'stats', zero if no operator recorded it.
  static int64_t operatorRuntimeStatSum(
      const velox::exec::TaskStats& stats,
      std::string_view name);

  /// Returns the sum of runtime statistic 'name' over every driver of 'stats',
  /// zero if no driver recorded it.
  static int64_t driverRuntimeStatSum(
      const velox::exec::TaskStats& stats,
      std::string_view name);

  /// Returns 'document' as JSON text with sorted keys, one per line.
  static std::string serialize(const folly::dynamic& document);
};

} // namespace facebook::velox::cudf_velox
