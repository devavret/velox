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

#include "velox/experimental/cudf/benchmarks/CudfTpchQueryBuilder.h"

#include "velox/benchmarks/tpch/TpchBenchmark.h"
#include "velox/common/base/Exceptions.h"

#include <folly/executors/CPUThreadPoolExecutor.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

/// Runs the TPC-H plans of cudf_velox::CudfTpchQueryBuilder. Every entry point
/// runs those plans: --run_query_verbose and --run_queries run the given
/// queries, and without either all 22 queries run, each --num_repeats times.
/// TpchBenchmark's own query builder, runQuery() and folly benchmarks, which
/// run the plans of exec::test::TpchQueryBuilder, are not used.
///
/// Every query runs in its own QueryCtx with a cuDF GPU root pool, so GPU
/// memory is arbitrated against the process GPU budget and operators spill to
/// pinned host memory when cudf.spill_host_memory_bytes is set. Nothing spills
/// to disk.
///
/// With --encoded and --encoded_load=query, the columns each query's scans
/// read are loaded into the encoded cache before the query is timed, and the
/// columns it does not read are dropped. Loading only projects columns: no row
/// is filtered and nothing is aggregated ahead of the timed run.
///
/// For each query it prints the wall time of each of the --num_repeats runs
/// ('Repeat N wall time us: T') followed by that run's memory counters
/// ('Repeat N memory: ...') and the sums of its operators' spill counters
/// ('Repeat N spill counters: ...'). With --run_query_verbose or
/// --run_queries it also prints the plan with its runtime statistics, and with
/// --include_results every column of every result row as '{a, b, ...}'.
///
/// With --results_json_dir, each run's result rows go to
/// <dir>/repeat<i>/q<N>.json and its counters to
/// <dir>/repeat<i>/q<N>.stats.json: every operator with its plan node id and
/// each of its runtime statistics as recorded, the drivers' statistics, and
/// the GPU and host spill memory counters. The files are written after the
/// run's wall time is taken. The last run's rows are also written to
/// <dir>/q<N>.json once every run of the query has succeeded.
///
/// A run that fails ('Repeat N failed after us: T') still writes its
/// counters, fails its query and ends its repeats. With --run_queries the
/// queries after a failed one still run, <dir>/run_summary.json lists the
/// failed queries with their errors, and the process exits with 1.
class CudfTpchBenchmark : public TpchBenchmark {
 public:
  void initialize() override;

  void runMain(std::ostream& out, facebook::velox::RunStats& runStats) override;

  std::shared_ptr<facebook::velox::config::ConfigBase> makeConnectorProperties()
      override;

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
  listSplits(
      const std::string& path,
      int32_t numSplitsPerFile,
      const facebook::velox::exec::test::TpchPlan& plan) override;

  void shutdown() override;

 protected:
  void configureCursorParameters(
      facebook::velox::exec::CursorParameters& params) const;

 private:
  // Where the scans of the query being run read from.
  enum class InputMode {
    // The encoded column cache.
    kEncoded,
    // Parquet files under --data_path, through the cuDF Hive connector.
    kParquet,
  };

  // One run of a query.
  struct QueryRun {
    // Owns the task and the pool 'results' are allocated from, so it is
    // declared first to be destroyed last. Null if the run failed before
    // its task was made.
    std::unique_ptr<facebook::velox::exec::TaskCursor> cursor;

    std::vector<facebook::velox::RowVectorPtr> results;

    // From before the query context is made until the task has completed
    // or, for a failed run, until its failure was seen.
    int64_t wallMicros{0};

    bool failed{false};

    // Why the run failed. Empty if it succeeded.
    std::string error;
  };

  // Runs 'queryId' --num_repeats times and prints the time and counters of
  // each run, writing them and the results under --results_json_dir if set.
  // With 'verbose', also prints the results and statistics of the last run.
  // Throws if a run fails, after reporting that run.
  void executeQuery(int32_t queryId, bool verbose, std::ostream& out);

  // Runs 'plan' once and waits for its task. Does not throw if the query
  // fails: the failed run is returned with its 'error' set, once the task's
  // drivers have closed and added their counters to the task.
  QueryRun runOnce(const facebook::velox::exec::test::TpchPlan& plan);

  // Makes the encoded cache hold the columns the scans of 'plan' read, and
  // with --encoded_load=query nothing else. Throws
  // encoded::EncodedHostBudgetExceeded if they do not fit the host cache
  // limit.
  void prepareEncodedInput(
      int32_t queryId,
      const facebook::velox::exec::test::TpchPlan& plan,
      std::ostream& out);

  // Loads 'columns' of 'table' from 'files', every column if empty.
  void loadEncodedColumns(
      const std::string& table,
      const std::vector<std::string>& files,
      const std::vector<std::string>& columns,
      std::ostream& out);

  // Registers the connector of 'mode' under the Hive connector id.
  void setInputMode(InputMode mode);

  // Prints the resolved GPU and host budgets.
  void printBudgets(std::ostream& out) const;

  std::unique_ptr<facebook::velox::cudf_velox::CudfTpchQueryBuilder>
      queryBuilder_;
  std::shared_ptr<folly::CPUThreadPoolExecutor> driverExecutor_;

  std::shared_ptr<facebook::velox::connector::Connector> encodedConnector_;
  std::shared_ptr<facebook::velox::connector::Connector> parquetConnector_;
  InputMode inputMode_{InputMode::kParquet};

  // Pinned host bytes the encoded cache may hold; zero means no limit.
  uint64_t hostCacheLimitBytes_{0};
};

namespace facebook::velox::cudf_velox {

/// Parse a properties file into a key-value map.
/// Each line should be key=value. Lines starting with '#' and blank lines are
/// skipped. Lines without '=' are logged as warnings and ignored.
inline std::unordered_map<std::string, std::string> loadPropertiesFile(
    const std::string& path) {
  auto fsPath = std::filesystem::path(path);
  VELOX_USER_CHECK(
      std::filesystem::exists(fsPath), "Properties file not found: {}", path);
  std::unordered_map<std::string, std::string> properties;
  std::string line;
  std::ifstream configFile(fsPath);
  while (std::getline(configFile, line)) {
    line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
    if (line.empty() || line[0] == '#') {
      continue;
    }
    LOG(INFO) << "Setting property " << line;
    const auto delimiterPos = line.find('=');
    if (delimiterPos == std::string::npos) {
      LOG(WARNING) << "Skipping malformed config line (no '='): " << line;
      continue;
    }
    properties.emplace(
        line.substr(0, delimiterPos), line.substr(delimiterPos + 1));
  }
  return properties;
}

} // namespace facebook::velox::cudf_velox
