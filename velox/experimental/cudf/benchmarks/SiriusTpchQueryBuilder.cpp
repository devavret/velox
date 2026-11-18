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
#include "velox/experimental/cudf/benchmarks/SiriusTpchQueryBuilder.h"
#include "velox/experimental/cudf/connectors/encoded/EncodedColumnCache.h"

#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/tpch/gen/TpchGen.h"

#include <gflags/gflags.h>

#include <fstream>
#include <iostream>
#include <limits>

DECLARE_bool(sirius_q22_window);
DECLARE_bool(sirius_q17_filtered_aggregate);
DECLARE_bool(sirius_q13_preaggregate);
DECLARE_bool(sirius_q10_preaggregate);
DECLARE_bool(sirius_q10_topn);
DECLARE_bool(sirius_q21_order_summary);
DECLARE_bool(sirius_q17_window);
DECLARE_bool(sirius_q18_unscaled_sum);
DECLARE_bool(sirius_q21_pairs);
DECLARE_bool(sirius_q15_window_max);
DECLARE_bool(sirius_q4_inner_distinct);
DECLARE_bool(sirius_filter_only_scans);
DECLARE_bool(sirius_prune_scan_outputs);
DECLARE_bool(sirius_q5_numeric_group);
DECLARE_bool(sirius_q5_late_revenue);
DECLARE_bool(sirius_q9_numeric_group);
DECLARE_bool(sirius_q9_partsupp_first);

namespace facebook::velox::exec::test {

namespace {

/// Parses literals such as 0.05 as DECIMAL so they compare with DECIMAL
/// columns.
const parse::ParseOptions& decimalParseOptions() {
  static const parse::ParseOptions options = [] {
    parse::ParseOptions result;
    result.parseDecimalAsDouble = false;
    return result;
  }();
  return options;
}

/// Q11's FRACTION is 0.0001 / SF. TPCH_Q11_FRACTION overrides the SF1 value.
std::string q11Fraction() {
  const char* fraction = std::getenv("TPCH_Q11_FRACTION");
  return fraction != nullptr ? fraction : "0.0001";
}

/// DWRF does not support Date type and Varchar is used.
/// Return the Date filter expression as per data format.
std::string formatDateFilter(
    const std::string& stringDate,
    const RowTypePtr& rowType,
    const std::string& lowerBound,
    const std::string& upperBound) {
  bool isDwrf = rowType->findChild(stringDate)->isVarchar();
  auto suffix = isDwrf ? "" : "::DATE";

  if (!lowerBound.empty() && !upperBound.empty()) {
    return fmt::format(
        "{} between {}{} and {}{}",
        stringDate,
        lowerBound,
        suffix,
        upperBound,
        suffix);
  } else if (!lowerBound.empty()) {
    return fmt::format("{} > {}{}", stringDate, lowerBound, suffix);
  } else if (!upperBound.empty()) {
    return fmt::format("{} < {}{}", stringDate, upperBound, suffix);
  }

  VELOX_FAIL(
      "Date range check expression must have either a lower or an upper bound");
}

std::vector<std::string> mergeColumnNames(
    const std::vector<std::string>& firstColumnVector,
    const std::vector<std::string>& secondColumnVector) {
  std::vector<std::string> mergedColumnVector = std::move(firstColumnVector);
  mergedColumnVector.insert(
      mergedColumnVector.end(),
      secondColumnVector.begin(),
      secondColumnVector.end());
  return mergedColumnVector;
};
} // namespace

void SiriusTpchQueryBuilder::readFileSchema(
    const std::string& tableName,
    const std::string& filePath,
    const std::vector<std::string>& columns) {
  dwio::common::ReaderOptions readerOptions(pool_.get());
  readerOptions.setDataIoStats(dataIoStats_);
  readerOptions.setMetadataIoStats(metadataIoStats_);
  readerOptions.setFileFormat(format_);
  auto uniqueReadFile =
      filesystems::getFileSystem(filePath, nullptr)->openFileForRead(filePath);
  std::shared_ptr<ReadFile> readFile;
  readFile.reset(uniqueReadFile.release());
  auto input = std::make_unique<dwio::common::BufferedInput>(
      readFile, readerOptions.memoryPool());
  std::unique_ptr<dwio::common::Reader> reader =
      dwio::common::getReaderFactory(readerOptions.fileFormat())
          ->createReader(std::move(input), readerOptions);
  const auto fileType = reader->rowType();
  const auto fileColumnNames = fileType->names();
  // There can be extra columns in the file towards the end.
  VELOX_CHECK_GE(fileColumnNames.size(), columns.size());
  std::unordered_map<std::string, std::string> fileColumnNamesMap(
      columns.size());
  std::transform(
      columns.begin(),
      columns.end(),
      fileColumnNames.begin(),
      std::inserter(fileColumnNamesMap, fileColumnNamesMap.begin()),
      [](std::string a, std::string b) { return std::make_pair(a, b); });
  auto columnNames = columns;
  auto types = fileType->children();
  types.resize(columnNames.size());
  tableMetadata_[tableName].type =
      std::make_shared<RowType>(std::move(columnNames), std::move(types));
  tableMetadata_[tableName].fileColumnNames = std::move(fileColumnNamesMap);
}

void SiriusTpchQueryBuilder::initialize(const std::string& dataPath) {
  for (const auto& [tableName, columns] : kTables_) {
    const fs::path tablePath{dataPath + "/" + tableName};
    std::error_code error;
    bool anyFound = false;
    for (auto const& dirEntry : fs::directory_iterator{
             tablePath, std::filesystem::directory_options(), error}) {
      if (!dirEntry.is_regular_file()) {
        continue;
      }
      // Ignore hidden files.
      if (dirEntry.path().filename().c_str()[0] == '.') {
        continue;
      }
      if (tableMetadata_[tableName].dataFiles.empty()) {
        anyFound = true;
        readFileSchema(tableName, dirEntry.path().string(), columns);
      }
      tableMetadata_[tableName].dataFiles.push_back(dirEntry.path());
    }
    if (!anyFound && error) {
      std::ifstream file(tablePath);
      std::string line;
      while (std::getline(file, line)) {
        if (tableMetadata_[tableName].dataFiles.empty()) {
          readFileSchema(tableName, line, columns);
        }
        tableMetadata_[tableName].dataFiles.push_back(line);
      }
    }
  }
}

const std::vector<std::string>& SiriusTpchQueryBuilder::getTableNames() {
  return kTableNames_;
}

TpchPlan SiriusTpchQueryBuilder::getQueryPlan(int queryId) const {
  switch (queryId) {
    case 1:
      return getQ1Plan();
    case 2:
      return getQ2Plan();
    case 3:
      return getQ3Plan();
    case 4:
      return FLAGS_sirius_q4_inner_distinct ? getQ4InnerDistinctPlan()
                                            : getQ4Plan();
    case 5:
      return FLAGS_sirius_q5_numeric_group ? getQ5NumericPlan()
          : FLAGS_sirius_q5_late_revenue   ? getQ5LateRevenuePlan()
                                           : getQ5Plan();
    case 6:
      return getQ6Plan();
    case 7:
      return getQ7Plan();
    case 8:
      return getQ8Plan();
    case 9:
      return (FLAGS_sirius_q9_numeric_group || FLAGS_sirius_q9_partsupp_first)
          ? getQ9NumericPlan()
          : getQ9Plan("p_name like '%yellow%'");
    case 10:
      return FLAGS_sirius_q10_preaggregate ? getQ10PreAggregatePlan()
                                           : getQ10Plan();
    case 11:
      return getQ11Plan();
    case 12:
      return getQ12Plan();
    case 13:
      return FLAGS_sirius_q13_preaggregate ? getQ13PreAggregatePlan()
                                           : getQ13Plan();
    case 14:
      return getQ14Plan();
    case 15:
      return FLAGS_sirius_q15_window_max ? getQ15WindowMaxPlan() : getQ15Plan();
    case 16:
      return getQ16Plan();
    case 17:
      return FLAGS_sirius_q17_window ? getQ17WindowPlan() : getQ17Plan();
    case 18:
      return getQ18Plan();
    case 19:
      return getQ19Plan();
    case 20:
      return getQ20Plan();
    case 21:
      return FLAGS_sirius_q21_pairs
          ? getQ21PairsPlan()
          : (FLAGS_sirius_q21_order_summary ? getQ21OrderSummaryPlan()
                                            : getQ21Plan());
    case 22:
      return FLAGS_sirius_q22_window ? getQ22WindowPlan() : getQ22Plan();
    default:
      VELOX_NYI("TPC-H query {} is not supported yet", queryId);
  }
}

TpchPlan SiriusTpchQueryBuilder::getAltPlan(int queryId) const {
  switch (queryId) {
    case 9:
      return getQ9Plan("p_size <= 3");
    default:
      VELOX_NYI("TPC-H query {} has no alternate plan", queryId);
  }
}

TpchPlan SiriusTpchQueryBuilder::getQ1Plan() const {
  std::vector<std::string> selectedColumns = {
      "l_returnflag",
      "l_linestatus",
      "l_quantity",
      "l_extendedprice",
      "l_discount",
      "l_tax",
      "l_shipdate"};

  const auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);

  // shipdate <= '1998-09-02'
  const auto shipDate = "l_shipdate";
  auto filter = formatDateFilter(shipDate, selectedRowType, "", "'1995-08-20'");

  core::PlanNodeId lineitemPlanNodeId;

  auto plan =
      PlanBuilder(pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs ? getRowType(
                                                    kLineitem,
                                                    {"l_returnflag",
                                                     "l_linestatus",
                                                     "l_quantity",
                                                     "l_extendedprice",
                                                     "l_discount",
                                                     "l_tax"})
                                              : selectedRowType,
              fileColumnNames,
              {filter},
              "",
              selectedRowType)
          .captureScanNodeId(lineitemPlanNodeId)
          .project(
              {"l_returnflag",
               "l_linestatus",
               "l_quantity",
               "l_extendedprice",
               "l_extendedprice * (1.0 - l_discount) AS l_sum_disc_price",
               "l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax) AS l_sum_charge",
               "l_discount"})
          .project(
              {"l_returnflag",
               "l_linestatus",
               "l_quantity",
               "l_extendedprice",
               "l_sum_disc_price",
               "l_sum_charge",
               "l_discount",
               "cast(l_quantity as DOUBLE) as qty_double",
               "cast(l_extendedprice as DOUBLE) as price_double",
               "cast(l_discount as DOUBLE) as discount_double"})
          .partialAggregation(
              {"l_returnflag", "l_linestatus"},
              {"sum(l_quantity)",
               "sum(l_extendedprice)",
               "sum(l_sum_disc_price)",
               "sum(l_sum_charge)",
               "avg(qty_double)",
               "avg(price_double)",
               "avg(discount_double)",
               "count(0)"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"l_returnflag", "l_linestatus"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ2Plan() const {
  std::vector<std::string> supplierColumnsSubQuery = {
      "s_suppkey", "s_nationkey"};
  std::vector<std::string> nationColumnsSubQuery = {
      "n_nationkey", "n_regionkey"};
  std::vector<std::string> supplierColumns = {
      "s_acctbal",
      "s_name",
      "s_address",
      "s_phone",
      "s_comment",
      "s_suppkey",
      "s_nationkey"};
  std::vector<std::string> partColumns = {
      "p_partkey", "p_mfgr", "p_size", "p_type"};
  std::vector<std::string> partsuppColumns = {
      "ps_partkey", "ps_suppkey", "ps_supplycost"};
  std::vector<std::string> nationColumns = {
      "n_nationkey", "n_name", "n_regionkey"};
  std::vector<std::string> regionColumns = {"r_regionkey", "r_name"};

  auto supplierSelectedRowTypeSubQuery =
      getRowType(kSupplier, supplierColumnsSubQuery);
  const auto& supplierFileColumnsSubQuery = getFileColumnNames(kSupplier);
  auto nationSelectedRowTypeSubQuery =
      getRowType(kNation, nationColumnsSubQuery);
  const auto& nationFileColumnsSubQuery = getFileColumnNames(kNation);
  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto partsuppSelectedRowType = getRowType(kPartsupp, partsuppColumns);
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  auto regionSelectedRowType = getRowType(kRegion, regionColumns);
  const auto& regionFileColumns = getFileColumnNames(kRegion);

  const std::string regionNameFilter = "r_name = 'EUROPE'";

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId supplierScanIdSubQuery;
  core::PlanNodeId partsuppScanIdSubQuery;
  core::PlanNodeId nationScanIdSubQuery;
  core::PlanNodeId regionScanIdSubQuery;
  core::PlanNodeId partScanId;
  core::PlanNodeId supplierScanId;
  core::PlanNodeId partsuppScanId;
  core::PlanNodeId nationScanId;
  core::PlanNodeId regionScanId;

  auto regionSubQuery = PlanBuilder(planNodeIdGenerator)
                            .setParseOptions(decimalParseOptions())
                            .filtersAsNode(filtersAsNode_)
                            .tableScan(
                                kRegion,
                                FLAGS_sirius_prune_scan_outputs
                                    ? getRowType(kRegion, {"r_regionkey"})
                                    : regionSelectedRowType,
                                regionFileColumns,
                                {regionNameFilter},
                                "",
                                regionSelectedRowType)
                            .captureScanNodeId(regionScanIdSubQuery)
                            .planNode();

  auto nationJoinRegionSubQuery =
      PlanBuilder(planNodeIdGenerator)
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationSelectedRowTypeSubQuery, nationFileColumnsSubQuery)
          .captureScanNodeId(nationScanIdSubQuery)
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              regionSubQuery,
              "",
              {"n_nationkey"})
          .planNode();

  auto supplierJoinNationJoinRegionSubQuery =
      PlanBuilder(planNodeIdGenerator)
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kSupplier,
              supplierSelectedRowTypeSubQuery,
              supplierFileColumnsSubQuery)
          .captureScanNodeId(supplierScanIdSubQuery)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationJoinRegionSubQuery,
              "",
              {"s_suppkey"})
          .planNode();

  auto part = PlanBuilder(planNodeIdGenerator)
                  .setParseOptions(decimalParseOptions())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      FLAGS_sirius_prune_scan_outputs
                          ? getRowType(kPart, {"p_partkey", "p_mfgr", "p_size"})
                          : partSelectedRowType,
                      partFileColumns,
                      {},
                      "p_type like '%NICKEL'",
                      partSelectedRowType)
                  .captureScanNodeId(partScanId)
                  .filter("p_size = 41")
                  .planNode();

  auto region = PlanBuilder(planNodeIdGenerator)
                    .setParseOptions(decimalParseOptions())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kRegion,
                        FLAGS_sirius_prune_scan_outputs
                            ? getRowType(kRegion, {"r_regionkey"})
                            : regionSelectedRowType,
                        regionFileColumns,
                        {regionNameFilter},
                        "",
                        regionSelectedRowType)
                    .captureScanNodeId(regionScanId)
                    .planNode();

  auto nationJoinRegion =
      PlanBuilder(planNodeIdGenerator)
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanId)
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              region,
              "",
              {"n_nationkey", "n_name"})
          .planNode();

  auto supplierJoinNationJoinRegion =
      PlanBuilder(planNodeIdGenerator)
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationJoinRegion,
              "",
              mergeColumnNames(supplierColumns, {"s_suppkey", "n_name"}))
          .planNode();

  auto partsuppJoinPartJoinSupplierJoinNationJoinRegion =
      PlanBuilder(planNodeIdGenerator)
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanId)
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              part,
              "",
              {"ps_suppkey", "ps_supplycost", "p_partkey", "p_mfgr"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNationJoinRegion,
              "",
              mergeColumnNames(
                  supplierColumns,
                  {"ps_supplycost", "p_partkey", "p_mfgr", "n_name"}))
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanIdSubQuery)
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNationJoinRegionSubQuery,
              "",
              {"ps_supplycost", "ps_partkey"})
          .partialAggregation(
              {"ps_partkey"}, {"min(ps_supplycost) AS min_supplycost"})
          .localPartition({"ps_partkey"})
          .finalAggregation()
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              partsuppJoinPartJoinSupplierJoinNationJoinRegion,
              "ps_supplycost = min_supplycost",
              mergeColumnNames(
                  supplierColumns, {"p_partkey", "p_mfgr", "n_name"}))
          .orderBy({"s_acctbal DESC", "n_name", "s_name", "p_partkey"}, false)
          .project(
              {"s_acctbal",
               "s_name",
               "n_name",
               "p_partkey",
               "p_mfgr",
               "s_address",
               "s_phone",
               "s_comment"})
          .limit(0, 100, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[supplierScanIdSubQuery] = getTableFilePaths(kSupplier);
  context.dataFiles[partsuppScanIdSubQuery] = getTableFilePaths(kPartsupp);
  context.dataFiles[nationScanIdSubQuery] = getTableFilePaths(kNation);
  context.dataFiles[regionScanIdSubQuery] = getTableFilePaths(kRegion);
  context.dataFiles[partScanId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanId] = getTableFilePaths(kSupplier);
  context.dataFiles[partsuppScanId] = getTableFilePaths(kPartsupp);
  context.dataFiles[nationScanId] = getTableFilePaths(kNation);
  context.dataFiles[regionScanId] = getTableFilePaths(kRegion);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ3Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_shipdate", "l_orderkey", "l_extendedprice", "l_discount"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_shippriority", "o_custkey", "o_orderkey"};
  std::vector<std::string> customerColumns = {"c_custkey", "c_mktsegment"};

  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  const auto orderDate = "o_orderdate";
  const auto shipDate = "l_shipdate";
  auto orderDateFilter =
      formatDateFilter(orderDate, ordersSelectedRowType, "", "'1995-03-25'");
  auto shipDateFilter =
      formatDateFilter(shipDate, lineitemSelectedRowType, "'1995-03-25'", "");
  auto customerFilter = "c_mktsegment = 'HOUSEHOLD'";

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;
  core::PlanNodeId ordersPlanNodeId;
  core::PlanNodeId customerPlanNodeId;

  auto customers = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .setParseOptions(decimalParseOptions())
                       .filtersAsNode(filtersAsNode_)
                       .tableScan(
                           kCustomer,
                           FLAGS_sirius_prune_scan_outputs
                               ? getRowType(kCustomer, {"c_custkey"})
                               : customerSelectedRowType,
                           customerFileColumns,
                           {customerFilter},
                           "",
                           customerSelectedRowType)
                       .captureScanNodeId(customerPlanNodeId)
                       .planNode();

  auto custkeyJoinNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              ordersSelectedRowType,
              ordersFileColumns,
              {orderDateFilter})
          .captureScanNodeId(ordersPlanNodeId)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"o_orderdate", "o_shippriority", "o_orderkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(
                        kLineitem,
                        {"l_orderkey", "l_extendedprice", "l_discount"})
                  : lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter},
              "",
              lineitemSelectedRowType)
          .captureScanNodeId(lineitemPlanNodeId)
          .project(
              {"l_extendedprice * (1.0 - l_discount) AS part_revenue",
               "l_orderkey"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              custkeyJoinNode,
              "",
              {"l_orderkey", "o_orderdate", "o_shippriority", "part_revenue"})
          .partialAggregation(
              {"l_orderkey", "o_orderdate", "o_shippriority"},
              {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"l_orderkey", "revenue", "o_orderdate", "o_shippriority"})
          .orderBy({"revenue DESC", "o_orderdate"}, false)
          .limit(0, 10, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersPlanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerPlanNodeId] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ4Plan() const {
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_orderpriority", "o_orderkey"};
  std::vector<std::string> lineitemColumns = {
      "l_orderkey", "l_commitdate", "l_receiptdate"};

  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);

  std::string orderDateFilter = formatDateFilter(
      "o_orderdate", ordersSelectedRowType, "'1996-10-01'", "'1996-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;
  core::PlanNodeId ordersPlanNodeId;

  auto orders =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kOrders, {"o_orderkey", "o_orderpriority"})
                  : ordersSelectedRowType,
              ordersFileColumns,
              {orderDateFilter},
              "",
              ordersSelectedRowType)
          .captureScanNodeId(ordersPlanNodeId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kLineitem, {"l_orderkey"})
                  : lineitemSelectedRowType,
              lineitemFileColumns,
              {},
              "l_commitdate < l_receiptdate",
              lineitemSelectedRowType)
          .captureScanNodeId(lineitemPlanNodeId)
          .partialAggregation({"l_orderkey"}, {}, {})
          .localPartition({"l_orderkey"})
          .finalAggregation()
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"o_orderpriority"},
              core::JoinType::kRightSemiFilter)
          .partialAggregation({"o_orderpriority"}, {"count(0) AS order_count"})
          .finalAggregation()
          .orderBy({"o_orderpriority"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersPlanNodeId] = getTableFilePaths(kOrders);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ4InnerDistinctPlan() const {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId ordersScan, lineitemScan;
  auto orders =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kOrders,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kOrders, {"o_orderkey", "o_orderpriority"})
                  : getRowType(
                        kOrders,
                        {"o_orderkey", "o_orderpriority", "o_orderdate"}),
              getFileColumnNames(kOrders),
              {"o_orderdate between DATE '1996-10-01' and DATE '1996-12-31'"},
              "",
              getRowType(
                  kOrders, {"o_orderkey", "o_orderpriority", "o_orderdate"}))
          .captureScanNodeId(ordersScan)
          .planNode();
  // Restrict to the date-selected orders before deduplicating lineitems.
  // TPC-H's unique order key means each (key, priority) is counted once.
  auto plan =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kLineitem, {"l_orderkey"})
                  : getRowType(
                        kLineitem,
                        {"l_orderkey", "l_commitdate", "l_receiptdate"}),
              getFileColumnNames(kLineitem),
              {},
              "l_commitdate < l_receiptdate",
              getRowType(
                  kLineitem, {"l_orderkey", "l_commitdate", "l_receiptdate"}))
          .captureScanNodeId(lineitemScan)
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"l_orderkey", "o_orderpriority"})
          .partialAggregation({"l_orderkey", "o_orderpriority"}, {})
          .localPartition({"l_orderkey"})
          .finalAggregation()
          .partialAggregation({"o_orderpriority"}, {"count(0) as order_count"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"o_orderpriority"}, false)
          .planNode();
  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[ordersScan] = getTableFilePaths(kOrders);
  result.dataFiles[lineitemScan] = getTableFilePaths(kLineitem);
  result.dataFileFormat = format_;
  return result;
}

TpchPlan SiriusTpchQueryBuilder::getQ5Plan() const {
  std::vector<std::string> customerColumns = {"c_custkey", "c_nationkey"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_custkey", "o_orderkey"};
  std::vector<std::string> lineitemColumns = {
      "l_suppkey", "l_orderkey", "l_discount", "l_extendedprice"};
  std::vector<std::string> supplierColumns = {"s_nationkey", "s_suppkey"};
  std::vector<std::string> nationColumns = {
      "n_nationkey", "n_name", "n_regionkey"};
  std::vector<std::string> regionColumns = {"r_regionkey", "r_name"};

  auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  auto regionSelectedRowType = getRowType(kRegion, regionColumns);
  const auto& regionFileColumns = getFileColumnNames(kRegion);

  std::string regionNameFilter = "r_name = 'EUROPE'";
  const auto orderDate = "o_orderdate";
  std::string orderDateFilter = formatDateFilter(
      orderDate, ordersSelectedRowType, "'1997-01-01'", "'1997-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId nationScanNodeId;
  core::PlanNodeId regionScanNodeId;

  auto region = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setParseOptions(decimalParseOptions())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kRegion,
                        FLAGS_sirius_prune_scan_outputs
                            ? getRowType(kRegion, {"r_regionkey"})
                            : regionSelectedRowType,
                        regionFileColumns,
                        {regionNameFilter},
                        "",
                        regionSelectedRowType)
                    .captureScanNodeId(regionScanNodeId)
                    .planNode();

  auto orders = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setParseOptions(decimalParseOptions())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kOrders,
                        FLAGS_sirius_prune_scan_outputs
                            ? getRowType(kOrders, {"o_orderkey", "o_custkey"})
                            : ordersSelectedRowType,
                        ordersFileColumns,
                        {orderDateFilter},
                        "",
                        ordersSelectedRowType)
                    .captureScanNodeId(ordersScanNodeId)
                    .planNode();

  auto customer =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .planNode();

  auto nationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              region,
              "",
              {"n_nationkey", "n_name"})
          .planNode();

  auto supplierJoinNationRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationJoinRegion,
              "",
              {"s_suppkey", "n_name", "s_nationkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanNodeId)
          .project(
              {"l_extendedprice * (1.0 - l_discount) AS part_revenue",
               "l_orderkey",
               "l_suppkey"})
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplierJoinNationRegion,
              "",
              {"n_name", "part_revenue", "s_nationkey", "l_orderkey"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"n_name", "part_revenue", "s_nationkey", "o_custkey"})
          .hashJoin(
              {"s_nationkey", "o_custkey"},
              {"c_nationkey", "c_custkey"},
              customer,
              "",
              {"n_name", "part_revenue"})
          .partialAggregation({"n_name"}, {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"revenue DESC"}, false)
          .project({"n_name", "revenue"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[regionScanNodeId] = getTableFilePaths(kRegion);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ5LateRevenuePlan() const {
  std::vector<std::string> customerColumns = {"c_custkey", "c_nationkey"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_custkey", "o_orderkey"};
  std::vector<std::string> lineitemColumns = {
      "l_suppkey", "l_orderkey", "l_discount", "l_extendedprice"};
  std::vector<std::string> supplierColumns = {"s_nationkey", "s_suppkey"};
  std::vector<std::string> nationColumns = {
      "n_nationkey", "n_name", "n_regionkey"};
  std::vector<std::string> regionColumns = {"r_regionkey", "r_name"};

  auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  auto regionSelectedRowType = getRowType(kRegion, regionColumns);
  const auto& regionFileColumns = getFileColumnNames(kRegion);

  std::string regionNameFilter = "r_name = 'EUROPE'";
  const auto orderDate = "o_orderdate";
  std::string orderDateFilter = formatDateFilter(
      orderDate, ordersSelectedRowType, "'1997-01-01'", "'1997-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId nationScanNodeId;
  core::PlanNodeId regionScanNodeId;

  auto region = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setParseOptions(decimalParseOptions())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kRegion,
                        FLAGS_sirius_prune_scan_outputs
                            ? getRowType(kRegion, {"r_regionkey"})
                            : regionSelectedRowType,
                        regionFileColumns,
                        {regionNameFilter},
                        "",
                        regionSelectedRowType)
                    .captureScanNodeId(regionScanNodeId)
                    .planNode();

  auto orders = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setParseOptions(decimalParseOptions())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kOrders,
                        FLAGS_sirius_prune_scan_outputs
                            ? getRowType(kOrders, {"o_orderkey", "o_custkey"})
                            : ordersSelectedRowType,
                        ordersFileColumns,
                        {orderDateFilter},
                        "",
                        ordersSelectedRowType)
                    .captureScanNodeId(ordersScanNodeId)
                    .planNode();

  auto customer =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .planNode();

  auto nationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              region,
              "",
              {"n_nationkey", "n_name"})
          .planNode();

  auto supplierJoinNationRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationJoinRegion,
              "",
              {"s_suppkey", "n_name", "s_nationkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanNodeId)
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplierJoinNationRegion,
              "",
              {"n_name",
               "l_extendedprice",
               "l_discount",
               "s_nationkey",
               "l_orderkey"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"n_name",
               "l_extendedprice",
               "l_discount",
               "s_nationkey",
               "o_custkey"})
          .hashJoin(
              {"s_nationkey", "o_custkey"},
              {"c_nationkey", "c_custkey"},
              customer,
              "",
              {"n_name", "l_extendedprice", "l_discount"})
          .project(
              {"n_name",
               "l_extendedprice * (1.0 - l_discount) AS part_revenue"})
          .partialAggregation({"n_name"}, {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"revenue DESC"}, false)
          .project({"n_name", "revenue"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[regionScanNodeId] = getTableFilePaths(kRegion);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ5NumericPlan() const {
  std::vector<std::string> customerColumns = {"c_custkey", "c_nationkey"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_custkey", "o_orderkey"};
  std::vector<std::string> lineitemColumns = {
      "l_suppkey", "l_orderkey", "l_discount", "l_extendedprice"};
  std::vector<std::string> supplierColumns = {"s_nationkey", "s_suppkey"};
  std::vector<std::string> nationColumns = {
      "n_nationkey", "n_name", "n_regionkey"};
  std::vector<std::string> regionColumns = {"r_regionkey", "r_name"};

  auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  auto regionSelectedRowType = getRowType(kRegion, regionColumns);
  const auto& regionFileColumns = getFileColumnNames(kRegion);

  std::string regionNameFilter = "r_name = 'EUROPE'";
  const auto orderDate = "o_orderdate";
  std::string orderDateFilter = formatDateFilter(
      orderDate, ordersSelectedRowType, "'1997-01-01'", "'1997-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId nationScanNodeId;
  core::PlanNodeId regionScanNodeId;
  core::PlanNodeId finalRegionScan, finalNationScan;

  auto region = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setParseOptions(decimalParseOptions())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kRegion,
                        FLAGS_sirius_prune_scan_outputs
                            ? getRowType(kRegion, {"r_regionkey"})
                            : regionSelectedRowType,
                        regionFileColumns,
                        {regionNameFilter},
                        "",
                        regionSelectedRowType)
                    .captureScanNodeId(regionScanNodeId)
                    .planNode();

  auto orders = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setParseOptions(decimalParseOptions())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kOrders,
                        FLAGS_sirius_prune_scan_outputs
                            ? getRowType(kOrders, {"o_orderkey", "o_custkey"})
                            : ordersSelectedRowType,
                        ordersFileColumns,
                        {orderDateFilter},
                        "",
                        ordersSelectedRowType)
                    .captureScanNodeId(ordersScanNodeId)
                    .planNode();

  auto customer =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .planNode();

  auto nationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              region,
              "",
              {"n_nationkey", "n_name"})
          .planNode();

  auto supplierJoinNationRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationJoinRegion,
              "",
              {"s_suppkey", "s_nationkey"},
              core::JoinType::kLeftSemiFilter)
          .planNode();

  // Apply the tiny dimension restriction as a semi join above. Materialize
  // names only after aggregation, preserving duplicate dimension rows here.
  auto finalRegion = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .setParseOptions(decimalParseOptions())
                         .tableScan(
                             kRegion,
                             getRowType(kRegion, {"r_regionkey"}),
                             regionFileColumns,
                             {regionNameFilter},
                             "",
                             regionSelectedRowType)
                         .captureScanNodeId(finalRegionScan)
                         .planNode();
  auto finalNames =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(finalNationScan)
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              finalRegion,
              "",
              {"n_nationkey", "n_name"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanNodeId)
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplierJoinNationRegion,
              "",
              {"l_extendedprice", "l_discount", "s_nationkey", "l_orderkey"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"l_extendedprice", "l_discount", "s_nationkey", "o_custkey"})
          .hashJoin(
              {"s_nationkey", "o_custkey"},
              {"c_nationkey", "c_custkey"},
              customer,
              "",
              {"s_nationkey", "l_extendedprice", "l_discount"})
          .project(
              {"s_nationkey",
               "l_extendedprice * (1.0 - l_discount) AS part_revenue"})
          .partialAggregation({"s_nationkey"}, {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              finalNames,
              "",
              {"n_name", "revenue"})
          .partialAggregation({"n_name"}, {"sum(revenue) as revenue_sum"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"n_name", "revenue_sum as revenue"})
          .orderBy({"revenue DESC"}, false)
          .project({"n_name", "revenue"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[regionScanNodeId] = getTableFilePaths(kRegion);
  context.dataFiles[finalRegionScan] = getTableFilePaths(kRegion);
  context.dataFiles[finalNationScan] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ6Plan() const {
  std::vector<std::string> selectedColumns = {
      "l_shipdate", "l_extendedprice", "l_quantity", "l_discount"};

  const auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);

  const auto shipDate = "l_shipdate";
  auto shipDateFilter = formatDateFilter(
      shipDate, selectedRowType, "'1997-01-01'", "'1997-12-31'");

  core::PlanNodeId lineitemPlanNodeId;
  auto plan =
      PlanBuilder(pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kLineitem, {"l_extendedprice", "l_discount"})
                  : selectedRowType,
              fileColumnNames,
              {shipDateFilter,
               "l_discount between cast(0.02 as DECIMAL(15, 2)) and cast(0.04 as DECIMAL(15, 2))",
               "l_quantity < cast(24 as DECIMAL(15, 2))"},
              "",
              selectedRowType)
          .captureScanNodeId(lineitemPlanNodeId)
          .project({"l_extendedprice * l_discount"})
          .partialAggregation({}, {"sum(p0)"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();
  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ7Plan() const {
  std::vector<std::string> supplierColumns = {"s_nationkey", "s_suppkey"};
  std::vector<std::string> lineitemColumns = {
      "l_shipdate", "l_suppkey", "l_orderkey", "l_discount", "l_extendedprice"};
  std::vector<std::string> ordersColumns = {"o_custkey", "o_orderkey"};
  std::vector<std::string> customerColumns = {"c_custkey", "c_nationkey"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  const std::string nationFilter = "n_name IN ('EGYPT', 'UNITED STATES')";
  auto shipDateFilter = formatDateFilter(
      "l_shipdate", lineitemSelectedRowType, "'1995-01-01'", "'1996-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId suppNationScanNodeId;
  core::PlanNodeId custNationScanNodeId;

  auto custNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationSelectedRowType, nationFileColumns, {nationFilter})
          .captureScanNodeId(custNationScanNodeId)
          .planNode();

  auto customerJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              custNation,
              "",
              {"n_name", "c_custkey"})
          .project({"n_name as cust_nation", "c_custkey"})
          .planNode();

  auto ordersJoinCustomer =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customerJoinNation,
              "",
              {"cust_nation", "o_orderkey"})
          .planNode();

  auto suppNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationSelectedRowType, nationFileColumns, {nationFilter})
          .captureScanNodeId(suppNationScanNodeId)
          .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              suppNation,
              "",
              {"n_name", "s_suppkey"})
          .project({"n_name as supp_nation", "s_suppkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter})
          .captureScanNodeId(lineitemScanNodeId)
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"supp_nation",
               "l_extendedprice",
               "l_discount",
               "l_shipdate",
               "l_orderkey"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              ordersJoinCustomer,
              "(((cust_nation = 'EGYPT') AND (supp_nation = 'UNITED STATES')) OR "
              "((cust_nation = 'UNITED STATES') AND (supp_nation = 'EGYPT')))",
              {"supp_nation",
               "cust_nation",
               "l_extendedprice",
               "l_discount",
               "l_shipdate"})
          .project(
              {"cust_nation",
               "supp_nation",
               "l_extendedprice * (1.0 - l_discount) as part_revenue",
               "year(l_shipdate) as l_year"})
          .partialAggregation(
              {"supp_nation", "cust_nation", "l_year"},
              {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"supp_nation", "cust_nation", "l_year"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[suppNationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[custNationScanNodeId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ8Plan() const {
  std::vector<std::string> partColumns = {"p_partkey", "p_type"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_nationkey"};
  std::vector<std::string> lineitemColumns = {
      "l_suppkey", "l_orderkey", "l_partkey", "l_extendedprice", "l_discount"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_orderkey", "o_custkey"};
  std::vector<std::string> customerColumns = {"c_nationkey", "c_custkey"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_regionkey"};
  std::vector<std::string> nationColumnsWithName = {
      "n_name", "n_nationkey", "n_regionkey"};
  std::vector<std::string> regionColumns = {"r_name", "r_regionkey"};

  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  const auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  const auto nationSelectedRowTypeWithName =
      getRowType(kNation, nationColumnsWithName);
  const auto& nationFileColumnsWithName = getFileColumnNames(kNation);
  const auto regionSelectedRowType = getRowType(kRegion, regionColumns);
  const auto& regionFileColumns = getFileColumnNames(kRegion);

  const auto orderDateFilter = formatDateFilter(
      "o_orderdate", ordersSelectedRowType, "'1995-01-01'", "'1996-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId partScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId nationScanNodeId;
  core::PlanNodeId nationScanNodeIdWithName;
  core::PlanNodeId regionScanNodeId;

  auto nationWithName =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationSelectedRowTypeWithName, nationFileColumnsWithName)
          .captureScanNodeId(nationScanNodeIdWithName)
          .planNode();

  auto region = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setParseOptions(decimalParseOptions())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kRegion,
                        FLAGS_sirius_prune_scan_outputs
                            ? getRowType(kRegion, {"r_regionkey"})
                            : regionSelectedRowType,
                        regionFileColumns,
                        {"r_name = 'MIDDLE EAST'"},
                        "",
                        regionSelectedRowType)
                    .captureScanNodeId(regionScanNodeId)
                    .planNode();

  auto part =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kPart,
              FLAGS_sirius_prune_scan_outputs ? getRowType(kPart, {"p_partkey"})
                                              : partSelectedRowType,
              partFileColumns,
              {"p_type = 'PROMO BRUSHED COPPER'"},
              "",
              partSelectedRowType)
          .captureScanNodeId(partScanNodeId)
          .planNode();

  auto nationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .hashJoin(
              {"n_regionkey"}, {"r_regionkey"}, region, "", {"n_nationkey"})
          .planNode();

  auto customerJoinNationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              nationJoinRegion,
              "",
              {"c_custkey"})
          .planNode();

  auto ordersJoinCustomerJoinNationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              ordersSelectedRowType,
              ordersFileColumns,
              {orderDateFilter})
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customerJoinNationJoinRegion,
              "",
              {"o_orderkey", "o_orderdate"})
          .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationWithName,
              "",
              {"s_suppkey", "n_name"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanNodeId)
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              ordersJoinCustomerJoinNationJoinRegion,
              "",
              {"l_partkey",
               "l_suppkey",
               "o_orderdate",
               "l_extendedprice",
               "l_discount"})
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"n_name",
               "o_orderdate",
               "l_partkey",
               "l_extendedprice",
               "l_discount"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              "",
              {"n_name", "o_orderdate", "l_extendedprice", "l_discount"})
          .project(
              {"l_extendedprice * (1.0 - l_discount) as volume",
               "n_name",
               "o_orderdate"})
          .project(
              {"volume",
               "(CASE WHEN n_name = 'EGYPT' THEN volume ELSE cast(0 as DECIMAL(31, 4)) END) as brazil_volume",
               "year(o_orderdate) AS o_year"})
          .partialAggregation(
              {"o_year"},
              {"sum(brazil_volume) as volume_brazil",
               "sum(volume) as volume_all"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"o_year"}, false)
          .project(
              {"o_year",
               "(cast(volume_brazil as DOUBLE) / cast(volume_all as DOUBLE)) as mkt_share"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[nationScanNodeIdWithName] = getTableFilePaths(kNation);
  context.dataFiles[regionScanNodeId] = getTableFilePaths(kRegion);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ9Plan(
    const std::string& partFilter) const {
  std::vector<std::string> lineitemColumns = {
      "l_suppkey",
      "l_partkey",
      "l_discount",
      "l_extendedprice",
      "l_orderkey",
      "l_quantity"};
  std::vector<std::string> partColumns = {"p_name", "p_size", "p_partkey"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_nationkey"};
  std::vector<std::string> partsuppColumns = {
      "ps_partkey", "ps_suppkey", "ps_supplycost"};
  std::vector<std::string> ordersColumns = {"o_orderkey", "o_orderdate"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partsuppSelectedRowType = getRowType(kPartsupp, partsuppColumns);
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);
  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId partScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId partsuppScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId nationScanNodeId;

  const std::vector<std::string> lineitemCommonColumns = {
      "l_extendedprice", "l_discount", "l_quantity"};

  auto part =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kPart,
              FLAGS_sirius_prune_scan_outputs ? getRowType(kPart, {"p_partkey"})
                                              : partSelectedRowType,
              partFileColumns,
              {},
              partFilter,
              partSelectedRowType)
          .captureScanNodeId(partScanNodeId)
          .planNode();

  auto supplier =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .planNode();

  auto nation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .planNode();

  auto lineitemJoinPartJoinSupplier =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanNodeId)
          .hashJoin({"l_partkey"}, {"p_partkey"}, part, "", lineitemColumns)
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplier,
              "",
              mergeColumnNames(lineitemColumns, {"s_nationkey"}))
          .planNode();

  auto partsuppJoinLineitemJoinPartJoinSupplier =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanNodeId)
          .hashJoin(
              {"ps_partkey", "ps_suppkey"},
              {"l_partkey", "l_suppkey"},
              lineitemJoinPartJoinSupplier,
              "",
              mergeColumnNames(
                  lineitemCommonColumns,
                  {"l_orderkey", "s_nationkey", "ps_supplycost"}))
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              partsuppJoinLineitemJoinPartJoinSupplier,
              "",
              mergeColumnNames(
                  lineitemCommonColumns,
                  {"s_nationkey", "ps_supplycost", "o_orderdate"}))
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              mergeColumnNames(
                  lineitemCommonColumns,
                  {"ps_supplycost", "o_orderdate", "n_name"}))
          .project(
              {"n_name AS nation",
               "year(o_orderdate) AS o_year",
               "l_extendedprice * (1.0 - l_discount) - ps_supplycost * l_quantity AS amount"})
          .partialAggregation(
              {"nation", "o_year"}, {"sum(amount) AS sum_profit"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"nation", "o_year DESC"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[partsuppScanNodeId] = getTableFilePaths(kPartsupp);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ9NumericPlan() const {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId partScan, supplierScan, lineitemScan, partsuppScan,
      ordersScan, nationScan;
  auto part = PlanBuilder(ids, pool_.get())
                  .setParseOptions(decimalParseOptions())
                  .tableScan(
                      kPart,
                      getRowType(kPart, {"p_partkey"}),
                      getFileColumnNames(kPart),
                      {},
                      "p_name like '%yellow%'",
                      getRowType(kPart, {"p_partkey", "p_name"}))
                  .captureScanNodeId(partScan)
                  .planNode();
  auto suppliers = PlanBuilder(ids, pool_.get())
                       .setParseOptions(decimalParseOptions())
                       .tableScan(
                           kSupplier,
                           getRowType(kSupplier, {"s_suppkey", "s_nationkey"}),
                           getFileColumnNames(kSupplier))
                       .captureScanNodeId(supplierScan)
                       .planNode();
  const auto lineType = getRowType(
      kLineitem,
      {"l_orderkey",
       "l_partkey",
       "l_suppkey",
       "l_extendedprice",
       "l_discount",
       "l_quantity"});
  const auto psType =
      getRowType(kPartsupp, {"ps_partkey", "ps_suppkey", "ps_supplycost"});
  core::PlanNodePtr details;
  if (FLAGS_sirius_q9_partsupp_first) {
    auto ps = PlanBuilder(ids, pool_.get())
                  .setParseOptions(decimalParseOptions())
                  .tableScan(kPartsupp, psType, getFileColumnNames(kPartsupp))
                  .captureScanNodeId(partsuppScan)
                  .hashJoin(
                      {"ps_partkey"},
                      {"p_partkey"},
                      part,
                      "",
                      {"ps_partkey", "ps_suppkey", "ps_supplycost"})
                  .planNode();
    details = PlanBuilder(ids, pool_.get())
                  .setParseOptions(decimalParseOptions())
                  .tableScan(kLineitem, lineType, getFileColumnNames(kLineitem))
                  .captureScanNodeId(lineitemScan)
                  .hashJoin(
                      {"l_partkey", "l_suppkey"},
                      {"ps_partkey", "ps_suppkey"},
                      ps,
                      "",
                      {"l_orderkey",
                       "l_suppkey",
                       "l_extendedprice",
                       "l_discount",
                       "l_quantity",
                       "ps_supplycost"})
                  .hashJoin(
                      {"l_suppkey"},
                      {"s_suppkey"},
                      suppliers,
                      "",
                      {"l_orderkey",
                       "s_nationkey",
                       "l_extendedprice",
                       "l_discount",
                       "l_quantity",
                       "ps_supplycost"})
                  .planNode();
  } else {
    auto lines =
        PlanBuilder(ids, pool_.get())
            .setParseOptions(decimalParseOptions())
            .tableScan(kLineitem, lineType, getFileColumnNames(kLineitem))
            .captureScanNodeId(lineitemScan)
            .hashJoin(
                {"l_partkey"},
                {"p_partkey"},
                part,
                "",
                {"l_orderkey",
                 "l_partkey",
                 "l_suppkey",
                 "l_extendedprice",
                 "l_discount",
                 "l_quantity"})
            .hashJoin(
                {"l_suppkey"},
                {"s_suppkey"},
                suppliers,
                "",
                {"l_orderkey",
                 "l_partkey",
                 "l_suppkey",
                 "s_nationkey",
                 "l_extendedprice",
                 "l_discount",
                 "l_quantity"})
            .planNode();
    details = PlanBuilder(ids, pool_.get())
                  .setParseOptions(decimalParseOptions())
                  .tableScan(kPartsupp, psType, getFileColumnNames(kPartsupp))
                  .captureScanNodeId(partsuppScan)
                  .hashJoin(
                      {"ps_partkey", "ps_suppkey"},
                      {"l_partkey", "l_suppkey"},
                      lines,
                      "",
                      {"l_orderkey",
                       "s_nationkey",
                       "l_extendedprice",
                       "l_discount",
                       "l_quantity",
                       "ps_supplycost"})
                  .planNode();
  }
  auto profit =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kOrders,
              getRowType(kOrders, {"o_orderkey", "o_orderdate"}),
              getFileColumnNames(kOrders))
          .captureScanNodeId(ordersScan)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              details,
              "",
              {"s_nationkey",
               "o_orderdate",
               "l_extendedprice",
               "l_discount",
               "l_quantity",
               "ps_supplycost"})
          .project(
              {"s_nationkey",
               "year(o_orderdate) as o_year",
               "l_extendedprice * (1.0 - l_discount) - ps_supplycost * l_quantity as amount"})
          .partialAggregation(
              {"s_nationkey", "o_year"}, {"sum(amount) as nation_profit"})
          .localPartition({"s_nationkey", "o_year"})
          .finalAggregation()
          .planNode();
  auto plan =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kNation,
              getRowType(kNation, {"n_nationkey", "n_name"}),
              getFileColumnNames(kNation))
          .captureScanNodeId(nationScan)
          .hashJoin(
              {"n_nationkey"},
              {"s_nationkey"},
              profit,
              "",
              {"n_name", "o_year", "nation_profit"})
          .partialAggregation(
              {"n_name", "o_year"}, {"sum(nation_profit) as sum_profit"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"n_name as nation", "o_year", "sum_profit"})
          .orderBy({"nation", "o_year DESC"}, false)
          .planNode();
  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[partScan] = getTableFilePaths(kPart);
  result.dataFiles[supplierScan] = getTableFilePaths(kSupplier);
  result.dataFiles[lineitemScan] = getTableFilePaths(kLineitem);
  result.dataFiles[partsuppScan] = getTableFilePaths(kPartsupp);
  result.dataFiles[ordersScan] = getTableFilePaths(kOrders);
  result.dataFiles[nationScan] = getTableFilePaths(kNation);
  result.dataFileFormat = format_;
  return result;
}

TpchPlan SiriusTpchQueryBuilder::getQ10Plan() const {
  std::vector<std::string> customerColumns = {
      "c_nationkey",
      "c_custkey",
      "c_acctbal",
      "c_name",
      "c_address",
      "c_phone",
      "c_comment"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};
  std::vector<std::string> lineitemColumns = {
      "l_orderkey", "l_returnflag", "l_extendedprice", "l_discount"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_orderkey", "o_custkey"};

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  const auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  const auto lineitemReturnFlagFilter = "l_returnflag = 'R'";
  const auto orderDate = "o_orderdate";
  auto orderDateFilter = formatDateFilter(
      orderDate, ordersSelectedRowType, "'1994-03-01'", "'1994-05-31'");

  const std::vector<std::string> customerOutputColumns = {
      "c_name", "c_acctbal", "c_phone", "c_address", "c_custkey", "c_comment"};

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId nationScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId ordersScanNodeId;

  auto nation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .planNode();

  auto orders = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setParseOptions(decimalParseOptions())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kOrders,
                        FLAGS_sirius_prune_scan_outputs
                            ? getRowType(kOrders, {"o_orderkey", "o_custkey"})
                            : ordersSelectedRowType,
                        ordersFileColumns,
                        {orderDateFilter},
                        "",
                        ordersSelectedRowType)
                    .captureScanNodeId(ordersScanNodeId)
                    .planNode();

  auto partialPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .hashJoin(
              {"c_custkey"},
              {"o_custkey"},
              orders,
              "",
              mergeColumnNames(
                  customerOutputColumns, {"c_nationkey", "o_orderkey"}))
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              mergeColumnNames(customerOutputColumns, {"n_name", "o_orderkey"}))
          .planNode();

  PlanBuilder builder(planNodeIdGenerator, pool_.get());
  builder.setParseOptions(decimalParseOptions())
      .filtersAsNode(filtersAsNode_)
      .tableScan(
          kLineitem,
          FLAGS_sirius_prune_scan_outputs
              ? getRowType(
                    kLineitem, {"l_orderkey", "l_extendedprice", "l_discount"})
              : lineitemSelectedRowType,
          lineitemFileColumns,
          {lineitemReturnFlagFilter},
          "",
          lineitemSelectedRowType)
      .captureScanNodeId(lineitemScanNodeId)
      .project(
          {"l_extendedprice * (1.0 - l_discount) AS part_revenue",
           "l_orderkey"})
      .hashJoin(
          {"l_orderkey"},
          {"o_orderkey"},
          partialPlan,
          "",
          mergeColumnNames(customerOutputColumns, {"part_revenue", "n_name"}))
      .partialAggregation(
          {"c_custkey",
           "c_name",
           "c_acctbal",
           "n_name",
           "c_address",
           "c_phone",
           "c_comment"},
          {"sum(part_revenue) as revenue"})
      .localPartition(std::vector<std::string>{})
      .finalAggregation();
  const std::vector<std::string> outputColumns = {
      "c_custkey",
      "c_name",
      "revenue",
      "c_acctbal",
      "n_name",
      "c_address",
      "c_phone",
      "c_comment"};
  if (FLAGS_sirius_q10_topn) {
    // The final aggregation runs on one driver after the gather, so one TopN
    // keeps the 20 highest revenues, as the sort and limit do.
    builder.topN({"revenue DESC"}, 20, false).project(outputColumns);
  } else {
    builder.orderBy({"revenue DESC"}, false)
        .project(outputColumns)
        .limit(0, 20, false);
  }
  auto plan = builder.planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ10PreAggregatePlan() const {
  // Revenue per order customer key from the returned lineitems of the
  // selected orders, computed after that join and before any customer or
  // nation string is touched. The customer and nation payload then joins the
  // per-key revenue, and a final group by every SQL group column sums it
  // again, so a duplicated customer or nation row contributes the same
  // multiples as in the original joins. TopN replaces the sort and limit.
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScan;
  core::PlanNodeId nationScan;
  core::PlanNodeId lineitemScan;
  core::PlanNodeId ordersScan;
  const auto ordersType =
      getRowType(kOrders, {"o_orderdate", "o_orderkey", "o_custkey"});
  const auto lineitemType = getRowType(
      kLineitem,
      {"l_orderkey", "l_returnflag", "l_extendedprice", "l_discount"});
  auto orders =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kOrders, {"o_orderkey", "o_custkey"})
                  : ordersType,
              getFileColumnNames(kOrders),
              {formatDateFilter(
                  "o_orderdate", ordersType, "'1994-03-01'", "'1994-05-31'")},
              "",
              ordersType)
          .captureScanNodeId(ordersScan)
          .planNode();
  auto revenueByCustomer =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(
                        kLineitem,
                        {"l_orderkey", "l_extendedprice", "l_discount"})
                  : lineitemType,
              getFileColumnNames(kLineitem),
              {"l_returnflag = 'R'"},
              "",
              lineitemType)
          .captureScanNodeId(lineitemScan)
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"o_custkey", "l_extendedprice", "l_discount"})
          .project(
              {"o_custkey",
               "l_extendedprice * (1.0 - l_discount) AS part_revenue"})
          .partialAggregation(
              {"o_custkey"}, {"sum(part_revenue) AS customer_revenue"})
          .localPartition({"o_custkey"})
          .finalAggregation()
          .planNode();
  auto nation = PlanBuilder(ids, pool_.get())
                    .setParseOptions(decimalParseOptions())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kNation,
                        getRowType(kNation, {"n_nationkey", "n_name"}),
                        getFileColumnNames(kNation))
                    .captureScanNodeId(nationScan)
                    .planNode();
  const std::vector<std::string> groupColumns = {
      "c_custkey",
      "c_name",
      "c_acctbal",
      "n_name",
      "c_address",
      "c_phone",
      "c_comment"};
  auto plan =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kCustomer,
              getRowType(
                  kCustomer,
                  {"c_nationkey",
                   "c_custkey",
                   "c_acctbal",
                   "c_name",
                   "c_address",
                   "c_phone",
                   "c_comment"}),
              getFileColumnNames(kCustomer))
          .captureScanNodeId(customerScan)
          .hashJoin(
              {"c_custkey"},
              {"o_custkey"},
              revenueByCustomer,
              "",
              {"c_custkey",
               "c_name",
               "c_acctbal",
               "c_address",
               "c_phone",
               "c_comment",
               "c_nationkey",
               "customer_revenue"})
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              {"c_custkey",
               "c_name",
               "c_acctbal",
               "n_name",
               "c_address",
               "c_phone",
               "c_comment",
               "customer_revenue"})
          // c_custkey is a group column: rows of one group share a partition.
          .partialAggregation(
              groupColumns, {"sum(customer_revenue) AS revenue"})
          .localPartition({"c_custkey"})
          .finalAggregation()
          .topN({"revenue DESC"}, 20, true)
          .localPartition(std::vector<std::string>{})
          .topN({"revenue DESC"}, 20, false)
          .project(
              {"c_custkey",
               "c_name",
               "revenue",
               "c_acctbal",
               "n_name",
               "c_address",
               "c_phone",
               "c_comment"})
          .planNode();
  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[customerScan] = getTableFilePaths(kCustomer);
  result.dataFiles[nationScan] = getTableFilePaths(kNation);
  result.dataFiles[lineitemScan] = getTableFilePaths(kLineitem);
  result.dataFiles[ordersScan] = getTableFilePaths(kOrders);
  result.dataFileFormat = format_;
  return result;
}

TpchPlan SiriusTpchQueryBuilder::getQ11Plan() const {
  // Column definitions.
  std::vector<std::string> partsuppColumns = {
      "ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"};
  std::vector<std::string> partsuppColumnsSubQuery = {
      "ps_availqty", "ps_suppkey", "ps_supplycost"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_nationkey"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  // Row type definitions.
  auto partsuppRowType = getRowType(kPartsupp, partsuppColumns);
  auto partsuppRowTypeSubQuery = getRowType(kPartsupp, partsuppColumnsSubQuery);
  auto supplierRowType = getRowType(kSupplier, supplierColumns);
  auto nationRowType = getRowType(kNation, nationColumns);

  // File columns.
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);
  const auto& partsuppFileColumnsSubQuery = getFileColumnNames(kPartsupp);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  // Filter for nation name.
  const std::string nationNameFilter = "n_name = 'GERMANY'";

  // Plan node ID generator.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  // Subquery plan nodes.
  core::PlanNodeId partsuppScanIdSubQuery;
  core::PlanNodeId supplierScanIdSubQuery;
  core::PlanNodeId nationScanIdSubQuery;

  // Main query plan nodes.
  core::PlanNodeId partsuppScanId;
  core::PlanNodeId supplierScanId;
  core::PlanNodeId nationScanId;

  // Subquery plan to calculate filter cost.
  auto nationSubQuery = PlanBuilder(planNodeIdGenerator)
                            .setParseOptions(decimalParseOptions())
                            .filtersAsNode(filtersAsNode_)
                            .tableScan(
                                kNation,
                                FLAGS_sirius_prune_scan_outputs
                                    ? getRowType(kNation, {"n_nationkey"})
                                    : nationRowType,
                                nationFileColumns,
                                {nationNameFilter},
                                "",
                                nationRowType)
                            .captureScanNodeId(nationScanIdSubQuery)
                            .planNode();

  auto supplierJoinNationSubQuery =
      PlanBuilder(planNodeIdGenerator)
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanIdSubQuery)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationSubQuery,
              "",
              {"s_suppkey"})
          .planNode();

  auto subQueryPlan =
      PlanBuilder(planNodeIdGenerator)
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kPartsupp, partsuppRowTypeSubQuery, partsuppFileColumnsSubQuery)
          .captureScanNodeId(partsuppScanIdSubQuery)
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNationSubQuery,
              "",
              {"ps_availqty", "ps_suppkey", "ps_supplycost"})
          .project(
              {"ps_supplycost * cast(ps_availqty as DECIMAL(10, 0)) AS product_cost_qty"})
          .partialAggregation({}, {"sum(product_cost_qty) AS sum_cost_qty"})
          .localPartition({})
          .finalAggregation()
          .project({fmt::format(
              "cast(sum_cost_qty * {} as DECIMAL(38, 6)) AS filter_cost_qty",
              q11Fraction())})
          .planNode();

  // Query plan.
  auto nation = PlanBuilder(planNodeIdGenerator)
                    .setParseOptions(decimalParseOptions())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kNation,
                        FLAGS_sirius_prune_scan_outputs
                            ? getRowType(kNation, {"n_nationkey"})
                            : nationRowType,
                        nationFileColumns,
                        {nationNameFilter},
                        "",
                        nationRowType)
                    .captureScanNodeId(nationScanId)
                    .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator)
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanId)
          .hashJoin({"s_nationkey"}, {"n_nationkey"}, nation, "", {"s_suppkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanId)
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"ps_partkey", "ps_availqty", "ps_supplycost"})
          .project(
              {"ps_supplycost * cast(ps_availqty as DECIMAL(10, 0)) AS product_cost_qty",
               "ps_partkey"})
          .partialAggregation(
              {"ps_partkey"}, {"sum(product_cost_qty) AS value"})
          .localPartition({"ps_partkey"})
          .finalAggregation()
          .nestedLoopJoin(
              subQueryPlan, {"ps_partkey", "value", "filter_cost_qty"})
          .filter("cast(value as DECIMAL(38, 6)) > filter_cost_qty")
          .project({"ps_partkey", "value"})
          .orderBy({"value DESC"}, false)
          .planNode();

  // Create context and return.
  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[partsuppScanIdSubQuery] = getTableFilePaths(kPartsupp);
  context.dataFiles[supplierScanIdSubQuery] = getTableFilePaths(kSupplier);
  context.dataFiles[nationScanIdSubQuery] = getTableFilePaths(kNation);
  context.dataFiles[partsuppScanId] = getTableFilePaths(kPartsupp);
  context.dataFiles[supplierScanId] = getTableFilePaths(kSupplier);
  context.dataFiles[nationScanId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ12Plan() const {
  std::vector<std::string> ordersColumns = {"o_orderkey", "o_orderpriority"};
  std::vector<std::string> lineitemColumns = {
      "l_receiptdate",
      "l_orderkey",
      "l_commitdate",
      "l_shipmode",
      "l_shipdate"};

  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId lineitemScanNodeId;

  const std::string receiptDateFilter = formatDateFilter(
      "l_receiptdate", lineitemSelectedRowType, "'1994-01-01'", "'1994-12-31'");
  const std::string shipDateFilter = formatDateFilter(
      "l_shipdate", lineitemSelectedRowType, "", "'1995-01-01'");
  const std::string commitDateFilter = formatDateFilter(
      "l_commitdate", lineitemSelectedRowType, "", "'1995-01-01'");

  auto lineitem =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs ? getRowType(
                                                    kLineitem,
                                                    {"l_orderkey",
                                                     "l_shipmode",
                                                     "l_shipdate",
                                                     "l_commitdate"})
                                              : lineitemSelectedRowType,
              lineitemFileColumns,
              {receiptDateFilter,
               "l_shipmode IN ('TRUCK', 'REG AIR')",
               shipDateFilter,
               commitDateFilter},
              "l_commitdate < l_receiptdate",
              lineitemSelectedRowType)
          .captureScanNodeId(lineitemScanNodeId)
          .filter("l_shipdate < l_commitdate")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns, {})
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              lineitem,
              "",
              {"l_shipmode", "o_orderpriority"})
          .project(
              {"l_shipmode",
               "(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count_partial",
               "(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count_partial"})
          .partialAggregation(
              {"l_shipmode"},
              {"sum(high_line_count_partial) as high_line_count",
               "sum(low_line_count_partial) as low_line_count"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"l_shipmode"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ13Plan() const {
  std::vector<std::string> ordersColumns = {
      "o_custkey", "o_comment", "o_orderkey"};
  std::vector<std::string> customerColumns = {"c_custkey"};

  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId ordersScanNodeId;

  auto customers =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kOrders, {"o_custkey", "o_orderkey"})
                  : ordersSelectedRowType,
              ordersFileColumns,
              {},
              "o_comment not like '%special%requests%'",
              ordersSelectedRowType)
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"c_custkey", "o_orderkey"},
              core::JoinType::kRight)
          .partialAggregation({"c_custkey"}, {"count(o_orderkey) as c_count"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .singleAggregation({"c_count"}, {"count(0) as custdist"})
          .orderBy({"custdist DESC", "c_count DESC"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ13PreAggregatePlan() const {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId ordersScan;
  core::PlanNodeId customersScan;
  // Count qualifying orders once per customer before the outer join. Every
  // customer contributes one count (zero if unmatched) to the distribution.
  auto counts =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              FLAGS_sirius_filter_only_scans
                  ? getRowType(kOrders, {"o_custkey", "o_orderkey"})
                  : getRowType(
                        kOrders, {"o_custkey", "o_orderkey", "o_comment"}),
              getFileColumnNames(kOrders),
              {},
              "o_comment not like '%special%requests%'",
              getRowType(kOrders, {"o_custkey", "o_orderkey", "o_comment"}))
          .captureScanNodeId(ordersScan)
          .partialAggregation(
              {"o_custkey"}, {"count(o_orderkey) as order_count"})
          .localPartition({"o_custkey"})
          .finalAggregation()
          .planNode();
  auto plan =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kCustomer,
              getRowType(kCustomer, {"c_custkey"}),
              getFileColumnNames(kCustomer))
          .captureScanNodeId(customersScan)
          .hashJoin(
              {"c_custkey"},
              {"o_custkey"},
              counts,
              "",
              {"c_custkey", "order_count"},
              core::JoinType::kLeft)
          .project({"coalesce(order_count, cast(0 as BIGINT)) as c_count"})
          .partialAggregation({"c_count"}, {"count(0) as custdist"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"custdist DESC", "c_count DESC"}, false)
          .planNode();
  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[ordersScan] = getTableFilePaths(kOrders);
  result.dataFiles[customersScan] = getTableFilePaths(kCustomer);
  result.dataFileFormat = format_;
  return result;
}

TpchPlan SiriusTpchQueryBuilder::getQ14Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_partkey", "l_extendedprice", "l_discount", "l_shipdate"};
  std::vector<std::string> partColumns = {"p_partkey", "p_type"};

  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  const std::string shipDate = "l_shipdate";
  const std::string shipDateFilter = formatDateFilter(
      shipDate, lineitemSelectedRowType, "'1994-08-01'", "'1994-08-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId partScanNodeId;

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .setParseOptions(decimalParseOptions())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(kPart, partSelectedRowType, partFileColumns)
                  .captureScanNodeId(partScanNodeId)
                  .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(
                        kLineitem,
                        {"l_partkey", "l_extendedprice", "l_discount"})
                  : lineitemSelectedRowType,
              lineitemFileColumns,
              {},
              shipDateFilter,
              lineitemSelectedRowType)
          .captureScanNodeId(lineitemScanNodeId)
          .project(
              FLAGS_sirius_prune_scan_outputs
                  ? std::vector<
                        std::
                            string>{"l_extendedprice * (1.0 - l_discount) as part_revenue", "l_partkey"}
                  : std::vector<
                        std::
                            string>{"l_extendedprice * (1.0 - l_discount) as part_revenue", "l_shipdate", "l_partkey"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              "",
              {"part_revenue", "p_type"})
          .project(
              {"(CASE WHEN (p_type LIKE 'PROMO%') THEN part_revenue ELSE cast(0 as DECIMAL(31, 4)) END) as filter_revenue",
               "part_revenue"})
          .partialAggregation(
              {},
              {"sum(part_revenue) as total_revenue",
               "sum(filter_revenue) as total_promo_revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project(
              {"cast(100.00 * total_promo_revenue as DOUBLE)/cast(total_revenue as DOUBLE) as promo_revenue"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ15Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"};
  std::vector<std::string> supplierColumns = {
      "s_suppkey", "s_name", "s_address", "s_phone"};

  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  const std::string shipDateFilter = formatDateFilter(
      "l_shipdate", lineitemSelectedRowType, "'1993-05-01'", "'1993-07-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanNodeIdSubQuery;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId supplierScanNodeId;

  auto maxRevenue =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(
                        kLineitem,
                        {"l_suppkey", "l_extendedprice", "l_discount"})
                  : lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter},
              "",
              lineitemSelectedRowType)
          .captureScanNodeId(lineitemScanNodeId)
          .project(
              {"l_suppkey",
               "l_extendedprice * (1.0 - l_discount) as part_revenue"})
          .partialAggregation(
              {"l_suppkey"}, {"sum(part_revenue) as total_revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .singleAggregation({}, {"max(total_revenue) as max_revenue"})
          .planNode();

  auto supplierWithMaxRevenue =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(
                        kLineitem,
                        {"l_suppkey", "l_extendedprice", "l_discount"})
                  : lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter},
              "",
              lineitemSelectedRowType)
          .captureScanNodeId(lineitemScanNodeIdSubQuery)
          .project(
              {"l_suppkey as supplier_no",
               "l_extendedprice * (1.0 - l_discount) as part_revenue"})
          .partialAggregation(
              {"supplier_no"}, {"sum(part_revenue) as total_revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .hashJoin(
              {"total_revenue"},
              {"max_revenue"},
              maxRevenue,
              "",
              {"supplier_no", "total_revenue"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_suppkey"},
              {"supplier_no"},
              supplierWithMaxRevenue,
              "",
              {"s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"})
          .orderBy({"s_suppkey"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanNodeIdSubQuery] = getTableFilePaths(kLineitem);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ15WindowMaxPlan() const {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScan, supplierScan;
  auto winners =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(
                        kLineitem,
                        {"l_suppkey", "l_extendedprice", "l_discount"})
                  : getRowType(
                        kLineitem,
                        {"l_suppkey",
                         "l_shipdate",
                         "l_extendedprice",
                         "l_discount"}),
              getFileColumnNames(kLineitem),
              {"l_shipdate between DATE '1993-05-01' and DATE '1993-07-31'"},
              "",
              getRowType(
                  kLineitem,
                  {"l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"}))
          .captureScanNodeId(lineitemScan)
          .project(
              {"l_suppkey", "l_extendedprice * (1.0 - l_discount) as revenue"})
          .partialAggregation({"l_suppkey"}, {"sum(revenue) as total_revenue"})
          .localPartition({"l_suppkey"})
          .finalAggregation()
          .localPartition(std::vector<std::string>{})
          .window({"max(total_revenue) over () as max_revenue"})
          .filter("total_revenue = max_revenue")
          .project({"l_suppkey", "total_revenue"})
          .planNode();
  auto plan =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kSupplier,
              getRowType(
                  kSupplier, {"s_suppkey", "s_name", "s_address", "s_phone"}),
              getFileColumnNames(kSupplier))
          .captureScanNodeId(supplierScan)
          .hashJoin(
              {"s_suppkey"},
              {"l_suppkey"},
              winners,
              "",
              {"s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"})
          .localPartition(std::vector<std::string>{})
          .orderBy({"s_suppkey"}, false)
          .planNode();
  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[lineitemScan] = getTableFilePaths(kLineitem);
  result.dataFiles[supplierScan] = getTableFilePaths(kSupplier);
  result.dataFileFormat = format_;
  return result;
}

TpchPlan SiriusTpchQueryBuilder::getQ16Plan() const {
  std::vector<std::string> partColumns = {
      "p_brand", "p_type", "p_size", "p_partkey"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_comment"};
  std::vector<std::string> partsuppColumns = {"ps_partkey", "ps_suppkey"};

  const auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  const auto partsuppSelectedRowType = getRowType(kPartsupp, partsuppColumns);
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId partScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId partsuppScanNodeId;

  // Keep the IN-list literals as INTEGER so they match the INTEGER p_size
  // column; widening them to BIGINT would cast the column and block the
  // subfield filter pushdown.
  parse::ParseOptions parseOptions;
  parseOptions.parseIntegerAsBigint = false;
  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .setParseOptions(decimalParseOptions())
                  .setParseOptions(parseOptions)
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partSelectedRowType,
                      partFileColumns,
                      {"p_size in (38, 2, 8, 31, 44, 5, 14, 24)"},
                      "p_type NOT LIKE 'MEDIUM PLATED%'")
                  .captureScanNodeId(partScanNodeId)
                  // Neq is unsupported as a tableScan subfield filter for
                  // Parquet source.
                  .filter("p_brand <> 'Brand#21'")
                  .planNode();

  auto supplier = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .setParseOptions(decimalParseOptions())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kSupplier,
                          FLAGS_sirius_prune_scan_outputs
                              ? getRowType(kSupplier, {"s_suppkey"})
                              : supplierSelectedRowType,
                          supplierFileColumns,
                          {},
                          "s_comment LIKE '%Customer%Complaints%'",
                          supplierSelectedRowType)
                      .captureScanNodeId(supplierScanNodeId)
                      .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanNodeId)
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              part,
              "",
              {"ps_suppkey", "p_brand", "p_type", "p_size"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplier,
              "",
              {"ps_suppkey", "p_brand", "p_type", "p_size"},
              core::JoinType::kAnti,
              true /*nullAware*/)
          // Empty aggregate is used here to get the distinct count of
          // ps_suppkey.
          // approx_distinct could be used instead for getting the count of
          // distinct ps_suppkey but since approx_distinct is non deterministic
          // and the standard error can not be set to 0, it is not used here.
          .partialAggregation({"p_brand", "p_type", "p_size", "ps_suppkey"}, {})
          .localPartition({"p_brand", "p_type", "p_size", "ps_suppkey"})
          .finalAggregation()
          .partialAggregation(
              {"p_brand", "p_type", "p_size"},
              {"count(ps_suppkey) as supplier_cnt"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"supplier_cnt DESC", "p_brand", "p_type", "p_size"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[partsuppScanNodeId] = getTableFilePaths(kPartsupp);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ17Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_partkey", "l_extendedprice", "l_quantity"};
  std::vector<std::string> partColumns = {
      "p_partkey", "p_brand", "p_container"};

  auto lineitemRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanId;
  core::PlanNodeId lineitemAggScanId;
  core::PlanNodeId partScanId;
  core::PlanNodeId partAggScanId;

  auto part =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kPart,
              FLAGS_sirius_prune_scan_outputs ? getRowType(kPart, {"p_partkey"})
                                              : partRowType,
              partFileColumns,
              {"p_brand = 'Brand#13'", "p_container = 'JUMBO CAN'"},
              "",
              partRowType)
          .captureScanNodeId(partScanId)
          .planNode();

  auto partAgg =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kPart,
              FLAGS_sirius_prune_scan_outputs ? getRowType(kPart, {"p_partkey"})
                                              : partRowType,
              partFileColumns,
              FLAGS_sirius_q17_filtered_aggregate
                  ? std::vector<
                        std::
                            string>{"p_brand = 'Brand#13'", "p_container = 'JUMBO CAN'"}
                  : std::vector<std::string>{},
              "",
              partRowType)
          .captureScanNodeId(partAggScanId)
          .planNode();

  auto lineitemJoinPart =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanId)
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              "",
              {"l_quantity", "p_partkey", "l_extendedprice"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemAggScanId)
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              partAgg,
              "",
              {"l_partkey", "l_quantity"})
          .project({"l_partkey", "cast(l_quantity as DOUBLE) as qty_double"})
          .partialAggregation({"l_partkey"}, {"avg(qty_double) as avg_"})
          .localPartition({"l_partkey"})
          .finalAggregation()
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              lineitemJoinPart,
              "cast(l_quantity as DOUBLE) < cast(0.2 as DOUBLE) * avg_",
              {"l_extendedprice"})
          .partialAggregation({}, {"sum(l_extendedprice) as partial_sum"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project(
              {"(cast(partial_sum as DOUBLE) / cast(7.0 as DOUBLE)) as avg_yearly"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanId] = getTableFilePaths(kLineitem);
  context.dataFiles[lineitemAggScanId] = getTableFilePaths(kLineitem);
  context.dataFiles[partScanId] = getTableFilePaths(kPart);
  context.dataFiles[partAggScanId] = getTableFilePaths(kPart);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ17WindowPlan() const {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId partsScan, lineitemScan;
  auto parts =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kPart,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kPart, {"p_partkey"})
                  : getRowType(kPart, {"p_partkey", "p_brand", "p_container"}),
              getFileColumnNames(kPart),
              {"p_brand = 'Brand#13'", "p_container = 'JUMBO CAN'"},
              "",
              getRowType(kPart, {"p_partkey", "p_brand", "p_container"}))
          .captureScanNodeId(partsScan)
          .planNode();
  auto plan =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kLineitem,
              getRowType(
                  kLineitem, {"l_partkey", "l_quantity", "l_extendedprice"}),
              getFileColumnNames(kLineitem))
          .captureScanNodeId(lineitemScan)
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              parts,
              "",
              {"l_partkey", "l_quantity", "l_extendedprice"})
          .project(
              {"l_partkey",
               "l_extendedprice",
               "cast(l_quantity as DOUBLE) as qty_double"})
          .localPartition({"l_partkey"})
          .window(
              {"sum(qty_double) over (partition by l_partkey rows between unbounded preceding and unbounded following) as sum_qty",
               "count(qty_double) over (partition by l_partkey rows between unbounded preceding and unbounded following) as count_qty"})
          .filter(
              "qty_double < cast(0.2 as DOUBLE) * (sum_qty / cast(count_qty as DOUBLE))")
          .project({"l_extendedprice"})
          .partialAggregation({}, {"sum(l_extendedprice) as partial_sum"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project(
              {"cast(partial_sum as DOUBLE) / cast(7.0 as DOUBLE) as avg_yearly"})
          .planNode();
  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[partsScan] = getTableFilePaths(kPart);
  result.dataFiles[lineitemScan] = getTableFilePaths(kLineitem);
  result.dataFileFormat = format_;
  return result;
}

TpchPlan SiriusTpchQueryBuilder::getQ18Plan() const {
  std::vector<std::string> lineitemColumns = {"l_orderkey", "l_quantity"};
  std::vector<std::string> ordersColumns = {
      "o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"};
  std::vector<std::string> customerColumns = {"c_name", "c_custkey"};

  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);

  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  const bool unscaled = FLAGS_sirius_q18_unscaled_sum;
  if (unscaled) {
    const auto* table =
        cudf_velox::connector::encoded::EncodedColumnCache::instance()
            .findTable(kLineitem);
    VELOX_USER_CHECK(
        table != nullptr, "Unscaled Q18 requires encoded table metadata");
    const auto& column = table->column("l_quantity");
    VELOX_USER_CHECK(
        column.codec == cudf_velox::connector::encoded::Codec::kForBitpack &&
            column.segmentMin.size() ==
                static_cast<size_t>(table->numSegments) &&
            column.segmentMax.size() == column.segmentMin.size() &&
            lineitemSelectedRowType->findChild("l_quantity")
                ->isShortDecimal() &&
            lineitemSelectedRowType->findChild("l_quantity")
                    ->asShortDecimal()
                    .scale() == 2 &&
            static_cast<int32_t>(column.type.scale()) == -2,
        "Unscaled Q18 requires complete scale-2 encoded decimal bounds");
    __int128 maxAbs = 0;
    for (size_t i = 0; i < column.segmentMin.size(); ++i) {
      for (const auto value : {column.segmentMin[i], column.segmentMax[i]}) {
        const __int128 wide = value;
        maxAbs = std::max(maxAbs, wide < 0 ? -wide : wide);
      }
    }
    const __int128 bound = maxAbs * table->numRows;
    VELOX_USER_CHECK(
        bound <= std::numeric_limits<int64_t>::max(),
        "Encoded bounds do not prove that the Q18 integer sum fits BIGINT");
    std::cout << "Q18 unscaled sum absolute bound: "
              << static_cast<int64_t>(bound) << std::endl;
  }

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId lineitemScanNodeId;

  auto bigOrders =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanNodeId)
          .project(
              unscaled
                  ? std::vector<
                        std::
                            string>{"l_orderkey", "parity_unscaled_value(l_quantity) as qty_units"}
                  : std::vector<std::string>{"l_orderkey", "l_quantity"})
          .partialAggregation(
              {"l_orderkey"},
              {unscaled ? "sum(qty_units) AS quantity_units"
                        : "sum(l_quantity) AS quantity"})
          .localPartition({"l_orderkey"})
          .finalAggregation()
          .filter(
              unscaled ? "quantity_units > cast(30000 as BIGINT)"
                       : "quantity > cast(300 as DECIMAL(38, 2))")
          .project(
              unscaled
                  ? std::vector<
                        std::
                            string>{"l_orderkey", "cast(quantity_units as DECIMAL(38, 2)) / cast(100 as DECIMAL(3, 0)) as quantity"}
                  : std::vector<std::string>{"l_orderkey", "quantity"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              bigOrders,
              "",
              {"o_orderkey",
               "o_custkey",
               "o_orderdate",
               "o_totalprice",
               "l_orderkey",
               "quantity"})
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              PlanBuilder(planNodeIdGenerator, pool_.get())
                  .setParseOptions(decimalParseOptions())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kCustomer, customerSelectedRowType, customerFileColumns)
                  .captureScanNodeId(customerScanNodeId)
                  .planNode(),
              "",
              {"c_name",
               "c_custkey",
               "o_orderkey",
               "o_orderdate",
               "o_totalprice",
               "quantity"})
          .localPartition(std::vector<std::string>{})
          .orderBy({"o_totalprice DESC", "o_orderdate"}, false)
          .limit(0, 100, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ19Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_partkey",
      "l_shipmode",
      "l_shipinstruct",
      "l_extendedprice",
      "l_discount",
      "l_quantity"};
  std::vector<std::string> partColumns = {
      "p_partkey", "p_brand", "p_container", "p_size"};

  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId partScanNodeId;

  const std::string shipModeFilter = "l_shipmode IN ('AIR', 'AIR REG')";
  const std::string shipInstructFilter =
      "(l_shipinstruct = 'DELIVER IN PERSON')";
  const std::string joinFilterExpr =
      "     ((p_brand = 'Brand#41')"
      "     AND (l_quantity between cast(2 as DECIMAL(15, 2)) and cast(12 as DECIMAL(15, 2)))"
      "     AND (p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG'))"
      "     AND (p_size BETWEEN 1 AND 5))"
      " OR  ((p_brand ='Brand#13')"
      "     AND (p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK'))"
      "     AND (l_quantity between cast(14 as DECIMAL(15, 2)) and cast(24 as DECIMAL(15, 2)))"
      "     AND (p_size BETWEEN 1 AND 10))"
      " OR  ((p_brand = 'Brand#55')"
      "     AND (p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'))"
      "     AND (l_quantity between cast(23 as DECIMAL(15, 2)) and cast(33 as DECIMAL(15, 2)))"
      "     AND (p_size BETWEEN 1 AND 15))";

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .setParseOptions(decimalParseOptions())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(kPart, partSelectedRowType, partFileColumns)
                  .captureScanNodeId(partScanNodeId)
                  .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs ? getRowType(
                                                    kLineitem,
                                                    {"l_partkey",
                                                     "l_extendedprice",
                                                     "l_discount",
                                                     "l_quantity"})
                                              : lineitemSelectedRowType,
              lineitemFileColumns,
              {shipModeFilter, shipInstructFilter},
              "",
              lineitemSelectedRowType)
          .captureScanNodeId(lineitemScanNodeId)
          .project(
              FLAGS_sirius_prune_scan_outputs
                  ? std::vector<
                        std::
                            string>{"l_extendedprice * (1.0 - l_discount) as part_revenue", "l_partkey", "l_quantity"}
                  : std::vector<
                        std::
                            string>{"l_extendedprice * (1.0 - l_discount) as part_revenue", "l_shipmode", "l_shipinstruct", "l_partkey", "l_quantity"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              joinFilterExpr,
              {"part_revenue"})
          .partialAggregation({}, {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ20Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_shipdate", "l_suppkey", "l_partkey", "l_quantity"};
  std::vector<std::string> partColumns = {"p_partkey", "p_name"};
  std::vector<std::string> supplierColumns = {
      "s_nationkey", "s_address", "s_name", "s_suppkey"};
  std::vector<std::string> partsuppColumns = {
      "ps_availqty", "ps_partkey", "ps_suppkey"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto partsuppSelectedRowType = getRowType(kPartsupp, partsuppColumns);
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  const std::string shipDateFilter = formatDateFilter(
      "l_shipdate", lineitemSelectedRowType, "'1993-01-01'", "'1993-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanId;
  core::PlanNodeId partScanId;
  core::PlanNodeId partAggScanId;
  core::PlanNodeId supplierScanId;
  core::PlanNodeId partsuppScanId;
  core::PlanNodeId nationScanId;

  auto part =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kPart,
              FLAGS_sirius_prune_scan_outputs ? getRowType(kPart, {"p_partkey"})
                                              : partSelectedRowType,
              partFileColumns,
              {},
              "p_name like 'antique%'",
              partSelectedRowType)
          .captureScanNodeId(partScanId)
          .planNode();

  auto partAgg =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kPart,
              FLAGS_sirius_prune_scan_outputs ? getRowType(kPart, {"p_partkey"})
                                              : partSelectedRowType,
              partFileColumns,
              {},
              "p_name like 'antique%'",
              partSelectedRowType)
          .captureScanNodeId(partAggScanId)
          .planNode();

  auto nation = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setParseOptions(decimalParseOptions())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kNation,
                        nationSelectedRowType,
                        nationFileColumns,
                        {"n_name = 'KENYA'"})
                    .captureScanNodeId(nationScanId)
                    .planNode();

  auto partsuppJoinPart =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanId)
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              part,
              "",
              {"ps_partkey", "ps_suppkey", "ps_availqty"},
              core::JoinType::kLeftSemiFilter)
          .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              {"s_name", "s_address", "s_suppkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(
                        kLineitem, {"l_partkey", "l_suppkey", "l_quantity"})
                  : lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter},
              "",
              lineitemSelectedRowType)
          .captureScanNodeId(lineitemScanId)
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              partAgg,
              "",
              {"l_partkey", "l_suppkey", "l_quantity"})
          .partialAggregation(
              {"l_partkey", "l_suppkey"}, {"sum(l_quantity) AS sum_qty"})
          .localPartition({"l_partkey", "l_suppkey"})
          .finalAggregation()
          .project({"l_partkey", "l_suppkey", "0.5 * sum_qty AS filter_qty"})
          .hashJoin(
              {"l_partkey", "l_suppkey"},
              {"ps_partkey", "ps_suppkey"},
              partsuppJoinPart,
              "cast(ps_availqty as DECIMAL(38, 3)) > filter_qty",
              {"ps_suppkey"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"s_name", "s_address"},
              core::JoinType::kRightSemiFilter)
          .orderBy({"s_name"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanId] = getTableFilePaths(kLineitem);
  context.dataFiles[partScanId] = getTableFilePaths(kPart);
  context.dataFiles[partAggScanId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanId] = getTableFilePaths(kSupplier);
  context.dataFiles[partsuppScanId] = getTableFilePaths(kPartsupp);
  context.dataFiles[nationScanId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ21Plan() const {
  std::vector<std::string> supplierColumns = {
      "s_nationkey", "s_name", "s_suppkey"};
  std::vector<std::string> lineitemColumnsWithDates = {
      "l_suppkey", "l_commitdate", "l_orderkey", "l_receiptdate"};
  std::vector<std::string> lineitemColumns = {"l_suppkey", "l_orderkey"};
  std::vector<std::string> ordersColumns = {"o_orderkey", "o_orderstatus"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  auto supplierRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto lineitem1RowType = getRowType(kLineitem, lineitemColumnsWithDates);
  const auto& lineitem1FileColumns = getFileColumnNames(kLineitem);
  auto lineitem2RowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitem2FileColumns = getFileColumnNames(kLineitem);
  auto lineitem3RowType = getRowType(kLineitem, lineitemColumnsWithDates);
  const auto& lineitem3FileColumns = getFileColumnNames(kLineitem);
  auto ordersRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto nationRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId lineitem1ScanNodeId;
  core::PlanNodeId lineitem2ScanNodeId;
  core::PlanNodeId lineitem3ScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId nationScanNodeId;
  const std::string receiptCommitFilter = "l_receiptdate > l_commitdate";

  // Order keys of the late lineitems of BRAZIL suppliers: every order
  // that can qualify. Semi joins on these keep the lineitem inputs of the
  // EXISTS and NOT EXISTS joins to those orders, a few million rows instead of
  // the whole table.
  std::vector<core::PlanNodeId> keyLineitemScanNodeIds;
  std::vector<core::PlanNodeId> keySupplierScanNodeIds;
  std::vector<core::PlanNodeId> keyNationScanNodeIds;
  auto candidateOrderKeys = [&](const std::string& name) {
    auto nationKeys =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .setParseOptions(decimalParseOptions())
            .filtersAsNode(filtersAsNode_)
            .tableScan(
                kNation,
                FLAGS_sirius_prune_scan_outputs
                    ? getRowType(kNation, {"n_nationkey"})
                    : nationRowType,
                nationFileColumns,
                {"n_name = 'BRAZIL'"},
                "",
                nationRowType)
            .captureScanNodeId(keyNationScanNodeIds.emplace_back())
            .planNode();
    auto suppliers =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .setParseOptions(decimalParseOptions())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kSupplier, supplierRowType, supplierFileColumns)
            .captureScanNodeId(keySupplierScanNodeIds.emplace_back())
            .hashJoin(
                {"s_nationkey"}, {"n_nationkey"}, nationKeys, "", {"s_suppkey"})
            .planNode();
    return PlanBuilder(planNodeIdGenerator, pool_.get())
        .setParseOptions(decimalParseOptions())
        .filtersAsNode(filtersAsNode_)
        .tableScan(
            kLineitem,
            FLAGS_sirius_prune_scan_outputs
                ? getRowType(kLineitem, {"l_orderkey", "l_suppkey"})
                : lineitem1RowType,
            lineitem1FileColumns,
            {},
            receiptCommitFilter,
            lineitem1RowType)
        .captureScanNodeId(keyLineitemScanNodeIds.emplace_back())
        .hashJoin({"l_suppkey"}, {"s_suppkey"}, suppliers, "", {"l_orderkey"})
        .project({fmt::format("l_orderkey as {}", name)})
        .planNode();
  };

  auto lineitem3 =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kLineitem, {"l_orderkey", "l_suppkey"})
                  : lineitem3RowType,
              lineitem3FileColumns,
              {},
              receiptCommitFilter,
              lineitem3RowType)
          .captureScanNodeId(lineitem3ScanNodeId)
          .project({"l_orderkey as l_orderkey_3", "l_suppkey as l_suppkey_3"})
          .hashJoin(
              {"l_orderkey_3"},
              {"key_3"},
              candidateOrderKeys("key_3"),
              "",
              {"l_orderkey_3", "l_suppkey_3"},
              core::JoinType::kLeftSemiFilter)
          .planNode();

  auto nation = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setParseOptions(decimalParseOptions())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kNation,
                        FLAGS_sirius_prune_scan_outputs
                            ? getRowType(kNation, {"n_nationkey"})
                            : nationRowType,
                        nationFileColumns,
                        {"n_name = 'BRAZIL'"},
                        "",
                        nationRowType)
                    .captureScanNodeId(nationScanNodeId)
                    .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              {"s_suppkey", "s_name", "s_nationkey"})
          .planNode();

  auto lineitemJoinSupplier =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kLineitem, {"l_orderkey", "l_suppkey"})
                  : lineitem1RowType,
              lineitem1FileColumns,
              {},
              receiptCommitFilter,
              lineitem1RowType)
          .captureScanNodeId(lineitem1ScanNodeId)
          .project({"l_orderkey as l_orderkey_1", "l_suppkey as l_suppkey_1"})
          .hashJoin(
              {"l_suppkey_1"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"l_orderkey_1", "s_nationkey", "l_suppkey_1", "s_name"})
          .planNode();

  auto ordersJoinLineitem1 =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kOrders, {"o_orderkey"})
                  : ordersRowType,
              ordersFileColumns,
              {"o_orderstatus = 'F'"},
              "",
              ordersRowType)
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey_1"},
              lineitemJoinSupplier,
              "",
              {"s_nationkey", "l_orderkey_1", "l_suppkey_1", "s_name"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitem2RowType, lineitem2FileColumns)
          .captureScanNodeId(lineitem2ScanNodeId)
          .project({"l_orderkey as l_orderkey_2", "l_suppkey as l_suppkey_2"})
          .hashJoin(
              {"l_orderkey_2"},
              {"key_2"},
              candidateOrderKeys("key_2"),
              "",
              {"l_orderkey_2", "l_suppkey_2"},
              core::JoinType::kLeftSemiFilter)
          .hashJoin(
              {"l_orderkey_2"},
              {"l_orderkey_1"},
              ordersJoinLineitem1,
              "l_suppkey_2 <> l_suppkey_1",
              {"l_orderkey_1", "l_suppkey_1", "s_name"},
              core::JoinType::kRightSemiFilter)
          .hashJoin(
              {"l_orderkey_1"},
              {"l_orderkey_3"},
              lineitem3,
              "l_suppkey_3 <> l_suppkey_1",
              {"s_name"},
              core::JoinType::kAnti,
              false /*nullAware*/)
          .partialAggregation({"s_name"}, {"count(1) as numwait"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"numwait DESC", "s_name"}, false)
          .limit(0, 100, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[lineitem1ScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[lineitem2ScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[lineitem3ScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  for (const auto& id : keyLineitemScanNodeIds) {
    context.dataFiles[id] = getTableFilePaths(kLineitem);
  }
  for (const auto& id : keySupplierScanNodeIds) {
    context.dataFiles[id] = getTableFilePaths(kSupplier);
  }
  for (const auto& id : keyNationScanNodeIds) {
    context.dataFiles[id] = getTableFilePaths(kNation);
  }
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getQ21PairsPlan() const {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId nationScan, supplierScan, supplierNameScan, lateScan,
      otherScan, ordersScan;
  auto nations = PlanBuilder(ids, pool_.get())
                     .setParseOptions(decimalParseOptions())
                     .tableScan(
                         kNation,
                         FLAGS_sirius_prune_scan_outputs
                             ? getRowType(kNation, {"n_nationkey"})
                             : getRowType(kNation, {"n_nationkey", "n_name"}),
                         getFileColumnNames(kNation),
                         {"n_name = 'BRAZIL'"},
                         "",
                         getRowType(kNation, {"n_nationkey", "n_name"}))
                     .captureScanNodeId(nationScan)
                     .planNode();
  auto suppliers =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kSupplier,
              getRowType(kSupplier, {"s_suppkey", "s_nationkey"}),
              getFileColumnNames(kSupplier))
          .captureScanNodeId(supplierScan)
          .hashJoin(
              {"s_nationkey"}, {"n_nationkey"}, nations, "", {"s_suppkey"})
          .planNode();
  auto late =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kLineitem,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kLineitem, {"l_orderkey", "l_suppkey"})
                  : getRowType(
                        kLineitem,
                        {"l_orderkey",
                         "l_suppkey",
                         "l_commitdate",
                         "l_receiptdate"}),
              getFileColumnNames(kLineitem),
              {},
              "l_receiptdate > l_commitdate",
              getRowType(
                  kLineitem,
                  {"l_orderkey", "l_suppkey", "l_commitdate", "l_receiptdate"}))
          .captureScanNodeId(lateScan)
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"l_orderkey", "l_suppkey"})
          .project(
              {"l_orderkey as candidate_order",
               "l_suppkey as candidate_supplier"})
          .planNode();
  auto candidates =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kOrders,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kOrders, {"o_orderkey"})
                  : getRowType(kOrders, {"o_orderkey", "o_orderstatus"}),
              getFileColumnNames(kOrders),
              {"o_orderstatus = 'F'"},
              "",
              getRowType(kOrders, {"o_orderkey", "o_orderstatus"}))
          .captureScanNodeId(ordersScan)
          .hashJoin(
              {"o_orderkey"},
              {"candidate_order"},
              late,
              "",
              {"candidate_order", "candidate_supplier"})
          .partialAggregation(
              {"candidate_order", "candidate_supplier"},
              {"count(0) as late_rows"})
          .localPartition({"candidate_order"})
          .finalAggregation()
          .planNode();
  // An inner join establishes EXISTS(other supplier). max(other_late)=0
  // establishes NOT EXISTS(other late supplier). late_rows preserves the
  // multiplicity of qualifying l1 rows instead of counting orders/pairs.
  auto waiting =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kLineitem,
              getRowType(
                  kLineitem,
                  {"l_orderkey", "l_suppkey", "l_commitdate", "l_receiptdate"}),
              getFileColumnNames(kLineitem))
          .captureScanNodeId(otherScan)
          .project(
              {"l_orderkey as other_order",
               "l_suppkey as other_supplier",
               "if(l_receiptdate > l_commitdate, 1, 0) as other_late"})
          .hashJoin(
              {"other_order"},
              {"candidate_order"},
              candidates,
              "",
              {"candidate_order",
               "candidate_supplier",
               "late_rows",
               "other_supplier",
               "other_late"})
          .filter("other_supplier <> candidate_supplier")
          .project(
              {"candidate_order",
               "candidate_supplier",
               "late_rows",
               "other_late"})
          .partialAggregation(
              {"candidate_order", "candidate_supplier", "late_rows"},
              {"max(other_late) as any_late"})
          .localPartition({"candidate_order"})
          .finalAggregation()
          .filter("any_late = 0")
          .partialAggregation(
              {"candidate_supplier"}, {"sum(late_rows) as waiting_rows"})
          .localPartition({"candidate_supplier"})
          .finalAggregation()
          .planNode();
  auto plan =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kSupplier,
              getRowType(kSupplier, {"s_suppkey", "s_name"}),
              getFileColumnNames(kSupplier))
          .captureScanNodeId(supplierNameScan)
          .hashJoin(
              {"s_suppkey"},
              {"candidate_supplier"},
              waiting,
              "",
              {"s_name", "waiting_rows"})
          .partialAggregation({"s_name"}, {"sum(waiting_rows) as numwait"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"numwait DESC", "s_name"}, false)
          .limit(0, 100, false)
          .planNode();
  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[nationScan] = getTableFilePaths(kNation);
  result.dataFiles[supplierScan] = getTableFilePaths(kSupplier);
  result.dataFiles[supplierNameScan] = getTableFilePaths(kSupplier);
  result.dataFiles[lateScan] = getTableFilePaths(kLineitem);
  result.dataFiles[otherScan] = getTableFilePaths(kLineitem);
  result.dataFiles[ordersScan] = getTableFilePaths(kOrders);
  result.dataFileFormat = format_;
  return result;
}

TpchPlan SiriusTpchQueryBuilder::getQ21OrderSummaryPlan() const {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId nationScan, supplierScan, lineitemScan, ordersScan;
  auto nations = PlanBuilder(ids, pool_.get())
                     .setParseOptions(decimalParseOptions())
                     .tableScan(
                         kNation,
                         FLAGS_sirius_prune_scan_outputs
                             ? getRowType(kNation, {"n_nationkey"})
                             : getRowType(kNation, {"n_nationkey", "n_name"}),
                         getFileColumnNames(kNation),
                         {"n_name = 'BRAZIL'"},
                         "",
                         getRowType(kNation, {"n_nationkey", "n_name"}))
                     .captureScanNodeId(nationScan)
                     .planNode();
  auto suppliers =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kSupplier,
              getRowType(kSupplier, {"s_suppkey", "s_nationkey", "s_name"}),
              getFileColumnNames(kSupplier))
          .captureScanNodeId(supplierScan)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nations,
              "",
              {"s_suppkey", "s_name"})
          .planNode();
  const auto lineitemType = getRowType(
      kLineitem, {"l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"});
  // For TPC-H's non-null supplier keys: another supplier exists iff min != max.
  // No other late supplier exists iff all late rows have the same supplier.
  // Sum the late-row count (not the order count), preserving duplicate l1 rows.
  auto candidates =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(kLineitem, lineitemType, getFileColumnNames(kLineitem))
          .captureScanNodeId(lineitemScan)
          .project(
              {"l_orderkey",
               "l_suppkey",
               fmt::format(
                   "if(l_receiptdate > l_commitdate, l_suppkey, cast(null as {})) as late_suppkey",
                   lineitemType->findChild("l_suppkey")->toString()),
               "if(l_receiptdate > l_commitdate, cast(1 as BIGINT), cast(0 as BIGINT)) as late_count"})
          .partialAggregation(
              {"l_orderkey"},
              {"min(l_suppkey) as min_supp",
               "max(l_suppkey) as max_supp",
               "min(late_suppkey) as min_late",
               "max(late_suppkey) as max_late",
               "sum(late_count) as num_late"})
          .localPartition({"l_orderkey"})
          .finalAggregation()
          .filter("min_supp <> max_supp and min_late = max_late")
          .project({"l_orderkey", "min_late as late_supplier", "num_late"})
          .hashJoin(
              {"late_supplier"},
              {"s_suppkey"},
              suppliers,
              "",
              {"l_orderkey", "s_name", "num_late"})
          .planNode();
  auto plan =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kOrders,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kOrders, {"o_orderkey"})
                  : getRowType(kOrders, {"o_orderkey", "o_orderstatus"}),
              getFileColumnNames(kOrders),
              {"o_orderstatus = 'F'"},
              "",
              getRowType(kOrders, {"o_orderkey", "o_orderstatus"}))
          .captureScanNodeId(ordersScan)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              candidates,
              "",
              {"s_name", "num_late"})
          .partialAggregation({"s_name"}, {"sum(num_late) as numwait"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"numwait DESC", "s_name"}, false)
          .limit(0, 100, false)
          .planNode();
  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[nationScan] = getTableFilePaths(kNation);
  result.dataFiles[supplierScan] = getTableFilePaths(kSupplier);
  result.dataFiles[lineitemScan] = getTableFilePaths(kLineitem);
  result.dataFiles[ordersScan] = getTableFilePaths(kOrders);
  result.dataFileFormat = format_;
  return result;
}

TpchPlan SiriusTpchQueryBuilder::getQ22WindowPlan() const {
  // SUM in exact scaled integers, then convert the completed sum to DOUBLE
  // before dividing by count, matching the decimal AVG finalization used by
  // the reference. Summing rounded DOUBLE inputs can change this predicate.
  const auto* table =
      cudf_velox::connector::encoded::EncodedColumnCache::instance().findTable(
          kCustomer);
  VELOX_USER_CHECK(
      table != nullptr, "Window Q22 requires encoded table metadata");
  const auto& column = table->column("c_acctbal");
  const auto type = getRowType(kCustomer, {"c_acctbal"})->childAt(0);
  VELOX_USER_CHECK(
      column.codec == cudf_velox::connector::encoded::Codec::kForBitpack &&
          column.segmentMin.size() == static_cast<size_t>(table->numSegments) &&
          column.segmentMax.size() == column.segmentMin.size() &&
          type->isShortDecimal() && type->asShortDecimal().scale() == 2 &&
          static_cast<int32_t>(column.type.scale()) == -2,
      "Window Q22 requires complete scale-2 decimal bounds");
  __int128 maxAbs = 0;
  for (size_t i = 0; i < column.segmentMin.size(); ++i) {
    for (const auto value : {column.segmentMin[i], column.segmentMax[i]}) {
      const __int128 wide = value;
      maxAbs = std::max(maxAbs, wide < 0 ? -wide : wide);
    }
  }
  VELOX_USER_CHECK(
      maxAbs * table->numRows <= std::numeric_limits<int64_t>::max(),
      "Encoded bounds do not prove that the Q22 integer sum fits BIGINT");
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId ordersScan, customerScan;
  auto orders = PlanBuilder(ids, pool_.get())
                    .setParseOptions(decimalParseOptions())
                    .tableScan(
                        kOrders,
                        getRowType(kOrders, {"o_custkey"}),
                        getFileColumnNames(kOrders))
                    .captureScanNodeId(ordersScan)
                    .planNode();
  const std::string phoneFilter =
      "substr(c_phone, 1, 2) IN ('24', '31', '11', '16', '21', '20', '34')";
  auto plan =
      PlanBuilder(ids, pool_.get())
          .setParseOptions(decimalParseOptions())
          .tableScan(
              kCustomer,
              getRowType(kCustomer, {"c_custkey", "c_acctbal", "c_phone"}),
              getFileColumnNames(kCustomer),
              {},
              phoneFilter)
          .captureScanNodeId(customerScan)
          .project(
              {"c_custkey",
               "c_acctbal",
               "substr(c_phone, 1, 2) as country_code",
               "if(c_acctbal > cast(0 as DECIMAL(15, 2)), parity_unscaled_value(c_acctbal), cast(null as BIGINT)) as positive_balance"})
          .localPartition(std::vector<std::string>{})
          .window(
              {"sum(positive_balance) over () as positive_sum",
               "count(positive_balance) over () as positive_count"})
          .filter(
              "(cast(positive_balance as DOUBLE) / cast(100 as DOUBLE)) > if(positive_count > 0, (cast(positive_sum as DOUBLE) / cast(100 as DOUBLE)) / cast(positive_count as DOUBLE), cast(null as DOUBLE))")
          .hashJoin(
              {"c_custkey"},
              {"o_custkey"},
              orders,
              "",
              {"c_acctbal", "country_code"},
              core::JoinType::kAnti,
              false)
          .partialAggregation(
              {"country_code"},
              {"count(0) as numcust", "sum(c_acctbal) as totacctbal"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"country_code"}, false)
          .planNode();
  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[ordersScan] = getTableFilePaths(kOrders);
  result.dataFiles[customerScan] = getTableFilePaths(kCustomer);
  result.dataFileFormat = format_;
  return result;
}

TpchPlan SiriusTpchQueryBuilder::getQ22Plan() const {
  std::vector<std::string> ordersColumns = {"o_custkey"};
  std::vector<std::string> customerColumns = {"c_acctbal", "c_phone"};
  std::vector<std::string> customerColumnsWithKey = {
      "c_custkey", "c_acctbal", "c_phone"};

  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  const auto customerSelectedRowTypeWithKey =
      getRowType(kCustomer, customerColumnsWithKey);
  const auto& customerFileColumnsWithKey = getFileColumnNames(kCustomer);

  const std::string phoneFilter =
      "substr(c_phone, 1, 2) IN ('24', '31', '11', '16', '21', '20', '34')";

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId customerScanNodeIdWithKey;
  core::PlanNodeId ordersScanNodeId;

  auto orders =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
          .captureScanNodeId(ordersScanNodeId)
          .planNode();

  auto customerAvgAccountBalance =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kCustomer,
              FLAGS_sirius_prune_scan_outputs
                  ? getRowType(kCustomer, {"c_acctbal"})
                  : customerSelectedRowType,
              customerFileColumns,
              {"c_acctbal > cast(0 as DECIMAL(15, 2))"},
              phoneFilter,
              customerSelectedRowType)
          .captureScanNodeId(customerScanNodeId)
          .partialAggregation(
              {},
              {"sum(c_acctbal) as sum_acctbal",
               "count(c_acctbal) as positive_count"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project(
              {"if(positive_count > 0, cast(sum_acctbal as DOUBLE) / cast(positive_count as DOUBLE), cast(null as DOUBLE)) as avg_acctbal"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .setParseOptions(decimalParseOptions())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kCustomer,
              customerSelectedRowTypeWithKey,
              customerFileColumnsWithKey,
              {},
              phoneFilter)
          .captureScanNodeId(customerScanNodeIdWithKey)
          .nestedLoopJoin(
              customerAvgAccountBalance,
              {"c_acctbal", "avg_acctbal", "c_custkey", "c_phone"})
          .filter("cast(c_acctbal as DOUBLE) > avg_acctbal")
          .hashJoin(
              {"c_custkey"},
              {"o_custkey"},
              orders,
              "",
              {"c_acctbal", "c_phone"},
              core::JoinType::kAnti,
              false /*nullAware*/)
          .project({"substr(c_phone, 1, 2) AS country_code", "c_acctbal"})
          .partialAggregation(
              {"country_code"},
              {"count(0) AS numcust", "sum(c_acctbal) AS totacctbal"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"country_code"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[customerScanNodeIdWithKey] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan SiriusTpchQueryBuilder::getIoMeterPlan(int columnPct) const {
  VELOX_CHECK(columnPct > 0 && columnPct <= 100);
  auto columns = getFileColumnNames(kLineitem);
  std::vector<std::string> names;
  for (auto& pair : columns) {
    names.push_back(pair.first);
  }
  std::sort(names.begin(), names.end());
  names.resize(names.size() * columnPct / 100);
  if (std::find(names.begin(), names.end(), "l_partkey") == names.end()) {
    names.push_back("l_partkey");
  }

  const auto selectedRowType = getRowType(kLineitem, names);
  std::vector<std::string> aggregates;
  std::vector<std::string> projectExprs;

  for (auto i = 0; i < selectedRowType->size(); ++i) {
    if (selectedRowType->childAt(i)->kind() == TypeKind::VARCHAR) {
      projectExprs.push_back(
          fmt::format("length({}) as l{}", selectedRowType->nameOf(i), i));
      aggregates.push_back(fmt::format("max(l{})", i));
    } else {
      projectExprs.push_back(selectedRowType->nameOf(i));
      aggregates.push_back(fmt::format("max({})", selectedRowType->nameOf(i)));
    }
  }

  std::string filter = "l_partkey between 2000000 and 2500000";

  core::PlanNodeId lineitemPlanNodeId;
  std::unordered_map<std::string, std::string> aliases;
  for (auto& name : names) {
    aliases[name] = name;
  }
  auto plan = PlanBuilder(pool_.get())
                  .setParseOptions(decimalParseOptions())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(kLineitem, selectedRowType, aliases, {filter})
                  .captureScanNodeId(lineitemPlanNodeId)
                  .project(projectExprs)
                  .partialAggregation({}, aggregates)
                  .localPartition(std::vector<std::string>{})
                  .finalAggregation()
                  .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

const std::vector<std::string> SiriusTpchQueryBuilder::kTableNames_ = {
    kLineitem,
    kOrders,
    kCustomer,
    kNation,
    kRegion,
    kPart,
    kSupplier,
    kPartsupp};

const std::unordered_map<std::string, std::vector<std::string>>
    SiriusTpchQueryBuilder::kTables_ = {
        std::make_pair(
            "lineitem",
            tpch::getTableSchema(tpch::Table::TBL_LINEITEM)->names()),
        std::make_pair(
            "orders",
            tpch::getTableSchema(tpch::Table::TBL_ORDERS)->names()),
        std::make_pair(
            "customer",
            tpch::getTableSchema(tpch::Table::TBL_CUSTOMER)->names()),
        std::make_pair(
            "nation",
            tpch::getTableSchema(tpch::Table::TBL_NATION)->names()),
        std::make_pair(
            "region",
            tpch::getTableSchema(tpch::Table::TBL_REGION)->names()),
        std::make_pair(
            "part",
            tpch::getTableSchema(tpch::Table::TBL_PART)->names()),
        std::make_pair(
            "supplier",
            tpch::getTableSchema(tpch::Table::TBL_SUPPLIER)->names()),
        std::make_pair(
            "partsupp",
            tpch::getTableSchema(tpch::Table::TBL_PARTSUPP)->names())};

} // namespace facebook::velox::exec::test
