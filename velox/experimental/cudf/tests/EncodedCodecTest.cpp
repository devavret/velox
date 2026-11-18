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
#include "velox/experimental/cudf/connectors/encoded/EncodedConnector.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <cudf/column/column_factories.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox::exec::test;

namespace encoded = facebook::velox::cudf_velox::connector::encoded;

namespace {

constexpr const char* kConnectorId = "test-encoded-codec";
constexpr int32_t kSegmentRows = 1'024;
// Five full segments and a short last one of 7 rows.
constexpr int32_t kRows = 5 * kSegmentRows + 7;
constexpr int64_t kMin64 = std::numeric_limits<int64_t>::min();
constexpr int64_t kMax64 = std::numeric_limits<int64_t>::max();
constexpr int32_t kMin32 = std::numeric_limits<int32_t>::min();
constexpr int32_t kMax32 = std::numeric_limits<int32_t>::max();

// Bits to hold 0..range.
uint32_t bitWidth(uint64_t range) {
  return range == 0 ? 0 : 64 - __builtin_clzll(range);
}

// The encoding of one segment as the host computes it: its true minimum and
// maximum, its packed width and whether it is delta-coded.
struct SegmentChoice {
  int64_t min;
  int64_t max;
  uint32_t width;
  bool delta;
};

// Frame of reference unless delta coding is allowed, the segment is
// non-decreasing and its differences need fewer bits than its range.
std::vector<SegmentChoice> expectedChoices(
    const std::vector<int64_t>& values,
    bool allowDelta) {
  std::vector<SegmentChoice> choices;
  for (size_t begin = 0; begin < values.size(); begin += kSegmentRows) {
    const size_t end = std::min(values.size(), begin + kSegmentRows);
    int64_t min = values[begin];
    int64_t max = values[begin];
    bool sorted = true;
    uint64_t maxDelta = 0;
    for (size_t i = begin + 1; i < end; ++i) {
      min = std::min(min, values[i]);
      max = std::max(max, values[i]);
      if (values[i] < values[i - 1]) {
        sorted = false;
      } else {
        maxDelta = std::max(
            maxDelta,
            static_cast<uint64_t>(values[i]) -
                static_cast<uint64_t>(values[i - 1]));
      }
    }
    SegmentChoice choice{
        min,
        max,
        bitWidth(static_cast<uint64_t>(max) - static_cast<uint64_t>(min)),
        false};
    if (allowDelta && sorted && bitWidth(maxDelta) < choice.width) {
      choice.width = bitWidth(maxDelta);
      choice.delta = true;
    }
    choices.push_back(choice);
  }
  return choices;
}

// The rows of 'segments' of 'all', or with 'rows' those offsets into them.
std::vector<int64_t> expectedRows(
    const std::vector<int64_t>& all,
    const std::vector<int32_t>& segments,
    const std::vector<int32_t>* rows = nullptr) {
  std::vector<int64_t> batch;
  for (auto segment : segments) {
    const size_t begin = static_cast<size_t>(segment) * kSegmentRows;
    const size_t end = std::min(all.size(), begin + kSegmentRows);
    batch.insert(batch.end(), all.begin() + begin, all.begin() + end);
  }
  if (rows == nullptr) {
    return batch;
  }
  std::vector<int64_t> gathered;
  for (auto row : *rows) {
    gathered.push_back(batch[row]);
  }
  return gathered;
}

std::vector<int32_t> narrow(const std::vector<int64_t>& values) {
  return std::vector<int32_t>(values.begin(), values.end());
}

// BIGINT: sorted from INT64_MIN and up to INT64_MAX (delta), unsorted over the
// full range (64-bit frame of reference), constant (zero width), sorted with
// one jump as wide as its range (frame of reference), short and sorted
// (delta).
std::vector<int64_t> makeBigints() {
  std::vector<int64_t> values(kRows);
  for (int32_t i = 0; i < kRows; ++i) {
    const int64_t r = i % kSegmentRows;
    switch (i / kSegmentRows) {
      case 0:
        values[i] = kMin64 + 2 * r;
        break;
      case 1:
        values[i] = kMax64 - 3 * (kSegmentRows - 1 - r);
        break;
      case 2:
        values[i] = r % 2 == 0 ? kMin64 + r : kMax64 - r;
        break;
      case 3:
        values[i] = 42;
        break;
      case 4:
        values[i] = r < kSegmentRows / 2 ? 0 : int64_t{1} << 40;
        break;
      default:
        values[i] = r - 3;
        break;
    }
  }
  return values;
}

// INTEGER: sorted from INT32_MIN and up to INT32_MAX (delta), unsorted over
// the full range, TPC-H order keys each repeated three times (delta),
// decreasing, short constant.
std::vector<int64_t> makeIntegers() {
  std::vector<int64_t> values(kRows);
  for (int32_t i = 0; i < kRows; ++i) {
    const int64_t r = i % kSegmentRows;
    switch (i / kSegmentRows) {
      case 0:
        values[i] = kMin32 + r;
        break;
      case 1:
        values[i] = kMax32 - (kSegmentRows - 1 - r);
        break;
      case 2:
        values[i] = r % 2 == 0 ? kMin32 + r : kMax32 - r;
        break;
      case 3: {
        const int64_t k = r / 3;
        values[i] = 1 + (k / 8) * 32 + k % 8;
        break;
      }
      case 4:
        values[i] = kSegmentRows - r;
        break;
      default:
        values[i] = 5;
        break;
    }
  }
  return values;
}

// DATE days: sorted (delta), scattered, sorted (delta), constant, sorted in
// steps of 10 (delta), short decreasing.
std::vector<int64_t> makeDates() {
  std::vector<int64_t> values(kRows);
  for (int32_t i = 0; i < kRows; ++i) {
    const int64_t r = i % kSegmentRows;
    switch (i / kSegmentRows) {
      case 0:
        values[i] = 8'000 + r / 2;
        break;
      case 1:
        values[i] = 8'000 + (r * 7'919) % 2'557;
        break;
      case 2:
        values[i] = 9'000 + r;
        break;
      case 3:
        values[i] = 9'000;
        break;
      case 4:
        values[i] = 9'000 + 10 * r;
        break;
      default:
        values[i] = 10'000 - r;
        break;
    }
  }
  return values;
}

// DECIMAL(15, 2) unscaled values increasing by 0.25: every segment is delta.
std::vector<int64_t> makeDecimals() {
  std::vector<int64_t> values(kRows);
  for (int32_t i = 0; i < kRows; ++i) {
    values[i] = 100 + 25 * int64_t{i};
  }
  return values;
}

// BIGINT noise: no segment is sorted.
std::vector<int64_t> makeNoise() {
  std::mt19937_64 random(42);
  std::vector<int64_t> values(kRows);
  for (auto& value : values) {
    value = static_cast<int64_t>(random());
  }
  return values;
}

void checkCuda(cudaError_t error) {
  VELOX_CHECK(error == cudaSuccess, "{}", cudaGetErrorString(error));
}

// The coding expected for each segment of 'values': frame of reference;
// scaled when 'eligible', every value minus the minimum is a multiple of 100
// and that is narrower; delta when allowed, non-decreasing and narrower still.
struct ScaledChoice {
  int64_t min;
  int64_t max;
  uint32_t width;
  bool scaled;
  bool delta;
};

std::vector<ScaledChoice> expectedScaledChoices(
    const std::vector<int64_t>& values,
    bool eligible,
    bool allowDelta) {
  std::vector<ScaledChoice> choices;
  for (size_t begin = 0; begin < values.size(); begin += kSegmentRows) {
    const size_t end = std::min(values.size(), begin + kSegmentRows);
    const auto [low, high] =
        std::minmax_element(values.begin() + begin, values.begin() + end);
    const uint64_t range =
        static_cast<uint64_t>(*high) - static_cast<uint64_t>(*low);
    ScaledChoice choice{*low, *high, bitWidth(range), false, false};
    bool divisible = eligible;
    bool sorted = true;
    uint64_t maxDelta = 0;
    for (size_t i = begin; i < end; ++i) {
      const uint64_t offset =
          static_cast<uint64_t>(values[i]) - static_cast<uint64_t>(*low);
      divisible = divisible && offset % 100 == 0;
      if (i > begin) {
        if (values[i] < values[i - 1]) {
          sorted = false;
        } else {
          maxDelta = std::max(
              maxDelta,
              static_cast<uint64_t>(values[i]) -
                  static_cast<uint64_t>(values[i - 1]));
        }
      }
    }
    if (divisible && bitWidth(range / 100) < choice.width) {
      choice.width = bitWidth(range / 100);
      choice.scaled = true;
    }
    if (allowDelta && sorted && bitWidth(maxDelta) < choice.width) {
      choice.width = bitWidth(maxDelta);
      choice.scaled = false;
      choice.delta = true;
    }
    choices.push_back(choice);
  }
  return choices;
}

// DECIMAL(18, 2) hundredths: whole units over a negative and over a fractional
// base, one row a hundredth off, constant, near the type's bounds, and a short
// sorted last segment that scaling narrows more than delta coding.
std::vector<int64_t> makeScaledDecimals() {
  static constexpr int64_t kLast[] = {
      0, 100, 200, 300'000, 300'100, 300'200, 300'300};
  std::vector<int64_t> values(kRows);
  for (int32_t row = 0; row < kRows; ++row) {
    const int64_t i = row % kSegmentRows;
    switch (row / kSegmentRows) {
      case 0:
        values[row] = -500'000 + 100 * ((i * 7) % 50);
        break;
      case 1:
        values[row] = 37 + 100 * ((i * 13) % 50);
        break;
      case 2:
        values[row] = 100 * ((i * 7) % 50) + (i == 3 ? 1 : 0);
        break;
      case 3:
        values[row] = 4'200;
        break;
      case 4:
        values[row] =
            i % 2 == 0 ? -999'999'999'999'999'900 : 999'999'999'999'999'900;
        break;
      default:
        values[row] = kLast[i];
    }
  }
  return values;
}

// Whole units: the first segment rises one unit per row, where delta coding is
// narrower than scaling; the others are unsorted, the short last one sorted.
std::vector<int64_t> makeSteppedDecimals() {
  std::vector<int64_t> values(kRows);
  for (int32_t row = 0; row < kRows; ++row) {
    const int64_t i = row % kSegmentRows;
    values[row] = row < kSegmentRows ? 100 * i : 100 * ((i * 11) % 90) - 4'500;
  }
  return values;
}

struct ScaledColumn {
  std::string name;
  const std::vector<int64_t>* values;
  bool eligible;
};

/// Loads tables into the encoded cache with and without delta coding and
/// decodes them as scans do. The parameter keeps the table in pinned host
/// memory, staged to the device for each decode, instead of device memory.
class EncodedCodecTest : public CudfHiveConnectorTestBase,
                         public testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();
    facebook::velox::connector::ConnectorRegistry::global().insert(
        kConnectorId,
        std::make_shared<encoded::EncodedConnector>(kConnectorId));
    // registerCudf() in the base SetUp() reset the adapters.
    encoded::registerEncodedTableScanAdapter();
    expected_ = {
        {"a", makeBigints()},
        {"b", makeIntegers()},
        {"d", makeDates()},
        {"m", makeDecimals()},
        {"u", makeNoise()}};
    data_ = makeRowVector(
        {"a", "b", "d", "m", "u"},
        {makeFlatVector<int64_t>(expected_.at("a")),
         makeFlatVector<int32_t>(narrow(expected_.at("b"))),
         makeFlatVector<int32_t>(narrow(expected_.at("d")), DATE()),
         makeFlatVector<int64_t>(expected_.at("m"), DECIMAL(15, 2)),
         makeFlatVector<int64_t>(expected_.at("u"))});
  }

  void TearDown() override {
    encoded::EncodedColumnCache::instance().clear();
    facebook::velox::connector::ConnectorRegistry::global().erase(kConnectorId);
    data_.reset();
    CudfHiveConnectorTestBase::TearDown();
  }

  const encoded::EncodedTable& load(
      const std::string& name,
      const RowVectorPtr& data,
      bool delta,
      bool scale100 = false) {
    auto file = TempFilePath::create();
    writeToFile(file->getPath(), data);
    cudf_velox::ensureCudaContextForThread();
    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    const auto mr = cudf_velox::get_output_mr();
    static rmm::mr::pinned_host_memory_resource pinnedHost;
    const bool hostResident = GetParam();
    encoded::EncodedColumnCache::instance().loadTable(
        name,
        {file->getPath()},
        kSegmentRows,
        stream,
        mr,
        hostResident ? rmm::device_async_resource_ref(pinnedHost) : mr,
        hostResident,
        delta,
        /*compressPlainStrings=*/false,
        scale100);
    stream.sync();
    return encoded::EncodedColumnCache::instance().table(name);
  }

  struct Decoded {
    cudf::data_type type;
    // Raw values, 4-byte ones sign-extended.
    std::vector<int64_t> values;
  };

  // Decodes 'name' of 'table' for 'segments', all rows or those of 'rows', as
  // a scan does: a host-resident table is staged first.
  Decoded decode(
      const encoded::EncodedTable& table,
      const std::string& name,
      std::vector<int32_t> segments,
      const std::vector<int32_t>* rows = nullptr) {
    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    const auto mr = cudf_velox::get_output_mr();
    std::unique_ptr<encoded::EncodedTable> staged;
    const encoded::EncodedTable* source = &table;
    if (table.hostResident) {
      staged = encoded::stageSegments(table, {name}, segments, stream, mr);
      source = staged.get();
      std::iota(segments.begin(), segments.end(), 0);
    }
    rmm::device_uvector<int32_t> deviceSegments(segments.size(), stream, mr);
    checkCuda(cudaMemcpyAsync(
        deviceSegments.data(),
        segments.data(),
        segments.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream.get()));
    std::unique_ptr<cudf::column> rowColumn;
    std::optional<cudf::column_view> rowView;
    if (rows != nullptr) {
      rowColumn = cudf::make_fixed_width_column(
          cudf::data_type{cudf::type_id::INT32},
          static_cast<cudf::size_type>(rows->size()),
          cudf::mask_state::UNALLOCATED,
          stream,
          mr);
      checkCuda(cudaMemcpyAsync(
          rowColumn->mutable_view().data<int32_t>(),
          rows->data(),
          rows->size() * sizeof(int32_t),
          cudaMemcpyHostToDevice,
          stream.get()));
      rowView = rowColumn->view();
    }
    auto decoded = encoded::decodeColumn(
        *source,
        source->column(name),
        segments,
        deviceSegments,
        rowView,
        stream,
        mr);
    Decoded result{decoded->type(), std::vector<int64_t>(decoded->size())};
    if (cudf::size_of(decoded->type()) == 8) {
      checkCuda(cudaMemcpyAsync(
          result.values.data(),
          decoded->view().data<int64_t>(),
          result.values.size() * sizeof(int64_t),
          cudaMemcpyDeviceToHost,
          stream.get()));
      stream.sync();
    } else {
      std::vector<int32_t> values(decoded->size());
      checkCuda(cudaMemcpyAsync(
          values.data(),
          decoded->view().data<int32_t>(),
          values.size() * sizeof(int32_t),
          cudaMemcpyDeviceToHost,
          stream.get()));
      stream.sync();
      std::copy(values.begin(), values.end(), result.values.begin());
    }
    return result;
  }

  std::map<std::string, std::vector<int64_t>> expected_;
  RowVectorPtr data_;
};

// Each segment keeps its true minimum and maximum for pruning, and is
// delta-coded only when allowed, non-decreasing and narrower that way.
TEST_P(EncodedCodecTest, choosesDeltaOnlyWhenNarrower) {
  const auto& plain = load("plain", data_, false);
  const auto& delta = load("delta", data_, true);
  for (const auto& [name, values] : expected_) {
    for (const bool allowDelta : {false, true}) {
      const auto& column = (allowDelta ? delta : plain).column(name);
      const auto choices = expectedChoices(values, allowDelta);
      ASSERT_EQ(column.codec, encoded::Codec::kForBitpack) << name;
      ASSERT_EQ(column.segmentDelta.size(), choices.size()) << name;
      for (size_t s = 0; s < choices.size(); ++s) {
        SCOPED_TRACE(
            fmt::format("{} delta {} segment {}", name, allowDelta, s));
        EXPECT_EQ(column.segmentMin[s], choices[s].min);
        EXPECT_EQ(column.segmentMax[s], choices[s].max);
        EXPECT_EQ(column.segmentBitWidth[s], choices[s].width);
        EXPECT_EQ(column.segmentDelta[s] != 0, choices[s].delta);
      }
    }
    // Delta coding never widens a column and shrinks one it applies to.
    const auto choices = expectedChoices(values, true);
    const bool anyDelta = std::any_of(
        choices.begin(), choices.end(), [](const auto& c) { return c.delta; });
    if (anyDelta) {
      EXPECT_LT(
          delta.column(name).encodedBytes(), plain.column(name).encodedBytes())
          << name;
    } else {
      EXPECT_EQ(
          delta.column(name).encodedBytes(), plain.column(name).encodedBytes())
          << name;
    }
  }
  EXPECT_EQ(
      delta.column("a").segmentDelta, (std::vector<uint8_t>{1, 1, 0, 0, 0, 1}));
  EXPECT_EQ(
      delta.column("b").segmentDelta, (std::vector<uint8_t>{1, 1, 0, 1, 0, 0}));
  EXPECT_EQ(
      delta.column("d").segmentDelta, (std::vector<uint8_t>{1, 0, 1, 0, 1, 0}));
  EXPECT_EQ(delta.column("m").segmentDelta, std::vector<uint8_t>(6, 1));
  EXPECT_EQ(delta.column("u").segmentDelta, std::vector<uint8_t>(6, 0));
}

// Whole batches of segments, mixing delta and frame-of-reference segments and
// starting anywhere, decode to the exact values with and without delta coding.
TEST_P(EncodedCodecTest, decodesBitExact) {
  const std::vector<std::vector<int32_t>> batches{
      {0, 1, 2, 3, 4, 5}, {1, 2, 3}, {5}, {4, 5}, {0}, {2}};
  for (const bool delta : {false, true}) {
    const auto& table = load(delta ? "delta" : "plain", data_, delta);
    for (const auto& [name, values] : expected_) {
      for (const auto& segments : batches) {
        SCOPED_TRACE(
            fmt::format(
                "{} delta {} segments {}",
                name,
                delta,
                fmt::join(segments, ",")));
        EXPECT_EQ(
            decode(table, name, segments).values,
            expectedRows(values, segments));
      }
    }
  }
}

// Gathers rows across segment boundaries, repeated and out of order, and no
// rows at all.
TEST_P(EncodedCodecTest, gathersRowsBitExact) {
  const std::vector<int32_t> segments{1, 2, 3, 4, 5};
  const int32_t batchRows = 4 * kSegmentRows + 7;
  const std::vector<int32_t> rows{
      batchRows - 1,
      0,
      kSegmentRows - 1,
      kSegmentRows,
      2 * kSegmentRows + 17,
      17,
      17,
      4 * kSegmentRows,
      3 * kSegmentRows - 1};
  const std::vector<int32_t> none;
  for (const bool delta : {false, true}) {
    const auto& table = load(delta ? "delta" : "plain", data_, delta);
    for (const auto& [name, values] : expected_) {
      SCOPED_TRACE(fmt::format("{} delta {}", name, delta));
      EXPECT_EQ(
          decode(table, name, segments, &rows).values,
          expectedRows(values, segments, &rows));
      EXPECT_TRUE(decode(table, name, segments, &none).values.empty());
    }
  }
}

// A last segment of one row stays frame-of-reference with width 0.
TEST_P(EncodedCodecTest, singleRowLastSegment) {
  const int32_t numRows = 2 * kSegmentRows + 1;
  std::vector<int64_t> values(numRows);
  for (int32_t i = 0; i < numRows; ++i) {
    values[i] = 7 * int64_t{i} - 5'000;
  }
  const auto& table = load(
      "single", makeRowVector({"s"}, {makeFlatVector<int64_t>(values)}), true);
  EXPECT_EQ(table.column("s").segmentDelta, (std::vector<uint8_t>{1, 1, 0}));
  EXPECT_EQ(decode(table, "s", {0, 1, 2}).values, values);
  EXPECT_EQ(
      decode(table, "s", {2}).values, std::vector<int64_t>{values.back()});
  const std::vector<int32_t> rows{2 * kSegmentRows, 0, 5};
  EXPECT_EQ(
      decode(table, "s", {0, 1, 2}, &rows).values,
      (std::vector<int64_t>{values[2 * kSegmentRows], values[0], values[5]}));
}

// Decoded columns keep their physical type and decimal scale.
TEST_P(EncodedCodecTest, preservesPhysicalTypes) {
  const auto& table = load("delta", data_, true);
  EXPECT_EQ(decode(table, "a", {0}).type.id(), cudf::type_id::INT64);
  EXPECT_EQ(decode(table, "b", {0}).type.id(), cudf::type_id::INT32);
  EXPECT_EQ(decode(table, "d", {0}).type.id(), cudf::type_id::TIMESTAMP_DAYS);
  const auto decimal = decode(table, "m", {0}).type;
  EXPECT_EQ(decimal.id(), cudf::type_id::DECIMAL64);
  EXPECT_EQ(decimal.scale(), -2);
}

// A scan of a delta-coded table filters on one column, prunes by the true
// minimum and maximum, and gathers the other columns at the passing rows.
TEST_P(EncodedCodecTest, scansDeltaCodedTable) {
  load("delta", data_, true);
  createDuckDbTable({data_});
  auto plan = PlanBuilder(pool())
                  .startTableScan(kConnectorId)
                  .tableName("delta")
                  .outputType(asRowType(data_->type()))
                  .subfieldFilters({"a > 0"})
                  .endTableScan()
                  .planNode();
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .splits(encoded::makeEncodedSplits(kConnectorId, "delta", 2))
      .maxDrivers(2)
      .assertResults("SELECT * FROM tmp WHERE a > 0");
}

// Scaling applies only to proven segments of scale -2 decimals, keeps the
// true minimum and maximum, and each segment takes its narrowest coding.
TEST_P(EncodedCodecTest, scale100CodesOnlyProvenNarrowerSegments) {
  const auto scaled = makeScaledDecimals();
  const auto stepped = makeSteppedDecimals();
  std::vector<int64_t> hundreds(kRows);
  std::vector<std::string> tags(kRows);
  for (int32_t row = 0; row < kRows; ++row) {
    hundreds[row] = 100 * ((row * 7) % 50);
    tags[row] = "tag" + std::to_string(row % 4);
  }
  auto data = makeRowVector(
      {"q", "p", "w", "i", "k", "t"},
      {makeFlatVector<int64_t>(scaled, DECIMAL(18, 2)),
       makeFlatVector<int64_t>(stepped, DECIMAL(18, 2)),
       makeFlatVector<int64_t>(stepped, DECIMAL(9, 2)),
       makeFlatVector<int64_t>(hundreds),
       makeFlatVector<int64_t>(hundreds, DECIMAL(15, 3)),
       makeFlatVector<std::string>(tags)});
  const std::vector<ScaledColumn> columns = {
      {"q", &scaled, true},
      {"p", &stepped, true},
      {"w", &stepped, true},
      {"i", &hundreds, false},
      {"k", &hundreds, false}};
  std::vector<int32_t> all(6);
  std::iota(all.begin(), all.end(), 0);
  for (const bool delta : {false, true}) {
    SCOPED_TRACE(delta ? "delta allowed" : "no delta");
    const auto& table =
        load(delta ? "scaledDelta" : "scaled", data, delta, true);
    ASSERT_EQ(table.numSegments, 6);
    for (const auto& column : columns) {
      SCOPED_TRACE(column.name);
      const auto& encodedColumn = table.column(column.name);
      const auto choices =
          expectedScaledChoices(*column.values, column.eligible, delta);
      ASSERT_EQ(encodedColumn.segmentScaled.size(), choices.size());
      ASSERT_EQ(encodedColumn.segmentDelta.size(), choices.size());
      for (size_t s = 0; s < choices.size(); ++s) {
        EXPECT_EQ(encodedColumn.segmentMin[s], choices[s].min) << s;
        EXPECT_EQ(encodedColumn.segmentMax[s], choices[s].max) << s;
        EXPECT_EQ(encodedColumn.segmentBitWidth[s], choices[s].width) << s;
        EXPECT_EQ(encodedColumn.segmentScaled[s] != 0, choices[s].scaled) << s;
        EXPECT_EQ(encodedColumn.segmentDelta[s] != 0, choices[s].delta) << s;
      }
      EXPECT_EQ(decode(table, column.name, all).values, *column.values);
    }
    EXPECT_EQ(
        table.column("q").segmentScaled,
        (std::vector<uint8_t>{1, 1, 0, 0, 1, 1}));
    EXPECT_EQ(table.column("p").segmentScaled[0] != 0, !delta);
    EXPECT_EQ(table.column("p").segmentDelta[0] != 0, delta);
    EXPECT_EQ(table.column("i").segmentScaled, std::vector<uint8_t>(6, 0));
    EXPECT_EQ(table.column("k").segmentScaled, std::vector<uint8_t>(6, 0));
    // Dictionary codes are never scaled.
    EXPECT_EQ(table.column("t").codec, encoded::Codec::kDictionary);
    EXPECT_EQ(table.column("t").segmentScaled, std::vector<uint8_t>(6, 0));
    // DECIMAL(9, 2) is read as DECIMAL32 or DECIMAL64; either qualifies.
    const auto shortDecimal = table.column("w").type;
    EXPECT_TRUE(
        shortDecimal.id() == cudf::type_id::DECIMAL32 ||
        shortDecimal.id() == cudf::type_id::DECIMAL64);
    EXPECT_EQ(shortDecimal.scale(), -2);
  }
}

// Noncontiguous segments and row gathers decode scaled segments exactly,
// next to frame-of-reference, constant and delta-coded ones.
TEST_P(EncodedCodecTest, scale100DecodesSelectedSegmentsAndRows) {
  const auto scaled = makeScaledDecimals();
  const auto stepped = makeSteppedDecimals();
  auto data = makeRowVector(
      {"q", "p"},
      {makeFlatVector<int64_t>(scaled, DECIMAL(18, 2)),
       makeFlatVector<int64_t>(stepped, DECIMAL(18, 2))});
  const auto& table = load("scaledRows", data, true, true);
  EXPECT_NE(table.column("p").segmentDelta[0], 0);
  EXPECT_NE(table.column("p").segmentScaled[1], 0);
  // Ascending noncontiguous full segments, with the short segment last, as
  // required by stageSegments and decodeColumn.
  const std::vector<int32_t> segments{0, 2, 3, 4, 5};
  const std::vector<int32_t> rows{
      4102, 0, 6, 7, 1030, 1031, 2058, 3079, 3080, 3};
  const std::vector<ScaledColumn> columns = {
      {"q", &scaled, true}, {"p", &stepped, true}};
  for (const auto& column : columns) {
    SCOPED_TRACE(column.name);
    EXPECT_EQ(
        decode(table, column.name, segments).values,
        expectedRows(*column.values, segments));
    EXPECT_EQ(
        decode(table, column.name, segments, &rows).values,
        expectedRows(*column.values, segments, &rows));
  }
}

// A scan filters a scaled column and prunes segments by their true minimums
// and maximums.
TEST_P(EncodedCodecTest, scansScaledTableWithOriginalBounds) {
  auto data = makeRowVector(
      {"q", "p"},
      {makeFlatVector<int64_t>(makeScaledDecimals(), DECIMAL(18, 2)),
       makeFlatVector<int64_t>(makeSteppedDecimals(), DECIMAL(18, 2))});
  load("scaledScan", data, false, true);
  createDuckDbTable({data});
  core::PlanNodeId scanId;
  auto plan = PlanBuilder(pool())
                  .startTableScan(kConnectorId)
                  .tableName("scaledScan")
                  .outputType(asRowType(data->type()))
                  .subfieldFilters({"q < cast(-1000 as DECIMAL(18, 2))"})
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .planNode();
  auto task =
      AssertQueryBuilder(plan, duckDbQueryRunner_)
          .splits(encoded::makeEncodedSplits(kConnectorId, "scaledScan", 2))
          .maxDrivers(2)
          .assertResults(
              "SELECT * FROM tmp WHERE q < cast(-1000 as DECIMAL(18, 2))");
  // Only segments 0 and 4 hold values below -1000.00.
  const auto stats = exec::toPlanStats(task->taskStats());
  EXPECT_EQ(stats.at(scanId).customStats.at("encodedSegmentsPruned").sum, 4);
}

INSTANTIATE_TEST_SUITE_P(
    EncodedCodecTest,
    EncodedCodecTest,
    testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) -> std::string {
      return info.param ? "pinnedHost" : "device";
    });

} // namespace
