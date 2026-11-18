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
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdlib>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox::exec::test;

namespace encoded = facebook::velox::cudf_velox::connector::encoded;

namespace {

constexpr const char* kConnectorId = "test-encoded-lz4";
constexpr int32_t kSegmentRows = 1'024;
// Five full segments and a short last one of 7 rows.
constexpr int32_t kRows = 5 * kSegmentRows + 7;
// Longer than two LZ4 blocks of 64 KiB.
constexpr int32_t kLongString = 150'000;

void checkCuda(cudaError_t error) {
  VELOX_CHECK(error == cudaSuccess, "{}", cudaGetErrorString(error));
}

// c: text with repeated words; segment 1 is multi-byte UTF-8, segment 2 holds
// one string spanning several blocks, segment 3 only empty strings and
// segment 4 every third one empty. r: random characters LZ4 cannot shrink.
std::vector<std::string> makeComments() {
  std::vector<std::string> values(kRows);
  for (int32_t i = 0; i < kRows; ++i) {
    const int32_t r = i % kSegmentRows;
    switch (i / kSegmentRows) {
      case 1:
        values[i] = fmt::format("héllo wörld ✓ {} δοκιμή 测试 {}", i, r % 7);
        break;
      case 2:
        values[i] = r == 5
            ? std::string(kLongString, 'a') + std::to_string(i)
            : fmt::format("carefully final deposits {} sleep {}", i, r % 11);
        break;
      case 3:
        values[i] = "";
        break;
      case 4:
        values[i] = r % 3 == 0 ? "" : fmt::format("ironic packages {} nag", i);
        break;
      default:
        values[i] = fmt::format(
            "order {} special requests about the {} deposits haggle",
            i,
            r % 13);
        break;
    }
  }
  return values;
}

std::vector<std::string> makeNoise() {
  std::mt19937 random(7);
  std::uniform_int_distribution<int> printable(33, 126);
  std::vector<std::string> values(kRows);
  for (auto& value : values) {
    value.resize(64);
    for (auto& c : value) {
      c = static_cast<char>(printable(random));
    }
  }
  return values;
}

// The rows of 'segments' of 'all', or with 'rows' those offsets into them.
std::vector<std::string> expectedRows(
    const std::vector<std::string>& all,
    const std::vector<int32_t>& segments,
    const std::vector<int32_t>* rows = nullptr) {
  std::vector<std::string> batch;
  for (auto segment : segments) {
    const size_t begin = static_cast<size_t>(segment) * kSegmentRows;
    const size_t end = std::min(all.size(), begin + kSegmentRows);
    batch.insert(batch.end(), all.begin() + begin, all.begin() + end);
  }
  if (rows == nullptr) {
    return batch;
  }
  std::vector<std::string> gathered;
  for (auto row : *rows) {
    gathered.push_back(batch[row]);
  }
  return gathered;
}

// Copies a device strings column to the host.
std::vector<std::string> toHost(
    cudf::column_view column,
    cuda::stream_ref stream) {
  cudf::strings_column_view strings(column);
  const auto numRows = strings.size();
  std::vector<std::string> values(numRows);
  if (numRows == 0) {
    return values;
  }
  std::vector<int64_t> offsets(numRows + 1);
  const auto offsetsView = strings.offsets();
  if (offsetsView.type().id() == cudf::type_id::INT64) {
    checkCuda(cudaMemcpyAsync(
        offsets.data(),
        offsetsView.data<int64_t>(),
        offsets.size() * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.get()));
  } else {
    std::vector<int32_t> narrow(numRows + 1);
    checkCuda(cudaMemcpyAsync(
        narrow.data(),
        offsetsView.data<int32_t>(),
        narrow.size() * sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream.get()));
    stream.sync();
    std::copy(narrow.begin(), narrow.end(), offsets.begin());
  }
  stream.sync();
  std::vector<char> chars(offsets[numRows] - offsets[0]);
  checkCuda(cudaMemcpyAsync(
      chars.data(),
      strings.parent().data<char>() + offsets[0],
      chars.size(),
      cudaMemcpyDeviceToHost,
      stream.get()));
  stream.sync();
  for (cudf::size_type i = 0; i < numRows; ++i) {
    values[i].assign(
        chars.data() + (offsets[i] - offsets[0]), offsets[i + 1] - offsets[i]);
  }
  return values;
}

/// Loads string columns into the encoded cache with LZ4 compression of their
/// characters and stages and decodes them as scans do.
class EncodedLz4Test : public CudfHiveConnectorTestBase {
 protected:
  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();
    facebook::velox::connector::ConnectorRegistry::global().insert(
        kConnectorId,
        std::make_shared<encoded::EncodedConnector>(kConnectorId));
    // registerCudf() in the base SetUp() reset the adapters.
    encoded::registerEncodedTableScanAdapter();
    comments_ = makeComments();
    noise_ = makeNoise();
    data_ = makeRowVector(
        {"k", "c", "r"},
        {makeFlatVector<int32_t>(kRows, [](auto row) { return row; }),
         makeFlatVector<std::string>(comments_),
         makeFlatVector<std::string>(noise_)});
  }

  void TearDown() override {
    unsetenv("LIBCUDF_LARGE_STRINGS_THRESHOLD");
    encoded::EncodedColumnCache::instance().clear();
    facebook::velox::connector::ConnectorRegistry::global().erase(kConnectorId);
    data_.reset();
    CudfHiveConnectorTestBase::TearDown();
  }

  const encoded::EncodedTable&
  load(const std::string& name, bool hostResident, bool compress) {
    auto file = TempFilePath::create();
    writeToFile(file->getPath(), data_);
    cudf_velox::ensureCudaContextForThread();
    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    const auto mr = cudf_velox::get_output_mr();
    static rmm::mr::pinned_host_memory_resource pinnedHost;
    encoded::EncodedColumnCache::instance().loadTable(
        name,
        {file->getPath()},
        kSegmentRows,
        stream,
        mr,
        hostResident ? rmm::device_async_resource_ref(pinnedHost) : mr,
        hostResident,
        /*deltaEncodeSortedSegments=*/false,
        compress);
    stream.sync();
    return encoded::EncodedColumnCache::instance().table(name);
  }

  // Stages 'segments' of a host-resident table as a scan does, then decodes
  // 'name', all rows or those of 'rows'. Returns the staged column's offsets
  // type in 'stagedOffsets' when given.
  std::vector<std::string> decode(
      const encoded::EncodedTable& table,
      const std::string& name,
      std::vector<int32_t> segments,
      const std::vector<int32_t>* rows = nullptr,
      cudf::type_id* stagedOffsets = nullptr) {
    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    const auto mr = cudf_velox::get_output_mr();
    std::unique_ptr<encoded::EncodedTable> staged;
    const encoded::EncodedTable* source = &table;
    if (table.hostResident) {
      staged = encoded::stageSegments(table, {name}, segments, stream, mr);
      source = staged.get();
      std::iota(segments.begin(), segments.end(), 0);
      if (stagedOffsets != nullptr) {
        *stagedOffsets =
            source->column(name)
                .values->view()
                .child(cudf::strings_column_view::offsets_column_index)
                .type()
                .id();
      }
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
    return toHost(decoded->view(), stream);
  }

  std::vector<std::string> comments_;
  std::vector<std::string> noise_;
  RowVectorPtr data_;
};

// Compressible characters are stored as LZ4 blocks instead of a strings
// column, and any batch of segments stages back to the exact strings.
TEST_F(EncodedLz4Test, stagesCompressedStringsExactly) {
  const auto& plain = load("plain", true, false);
  const auto& table = load("lz4", true, true);
  const auto& column = table.column("c");
  ASSERT_EQ(column.codec, encoded::Codec::kPlain);
  ASSERT_TRUE(column.compressedChars != nullptr);
  EXPECT_TRUE(column.values == nullptr);
  ASSERT_TRUE(column.stringOffsets != nullptr);
  EXPECT_EQ(column.stringOffsets->type().id(), cudf::type_id::INT32);
  EXPECT_GT(column.compressedChars->blockStarts.size(), 2u);
  EXPECT_LT(column.encodedBytes(), plain.column("c").encodedBytes());

  const std::vector<std::vector<int32_t>> batches{
      {0, 1, 2, 3, 4, 5}, {1, 2, 3}, {0, 2, 5}, {3}, {2}, {5}, {3, 4}};
  for (const auto& segments : batches) {
    SCOPED_TRACE(fmt::format("segments {}", fmt::join(segments, ",")));
    EXPECT_EQ(decode(table, "c", segments), expectedRows(comments_, segments));
  }
  // Rows across runs, repeated and out of order.
  const std::vector<int32_t> segments{0, 2, 5};
  const std::vector<int32_t> rows{
      2 * kSegmentRows + 6,
      0,
      kSegmentRows + 5,
      kSegmentRows + 5,
      17,
      2 * kSegmentRows - 1,
      kSegmentRows};
  EXPECT_EQ(
      decode(table, "c", segments, &rows),
      expectedRows(comments_, segments, &rows));
}

// Offsets above the large strings threshold are 64-bit, in the cache and in
// every staged column.
TEST_F(EncodedLz4Test, stagesLargeOffsets) {
  setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", "4096", 1);
  const auto& table = load("lz4", true, true);
  unsetenv("LIBCUDF_LARGE_STRINGS_THRESHOLD");
  const auto& column = table.column("c");
  ASSERT_TRUE(column.compressedChars != nullptr);
  ASSERT_EQ(column.stringOffsets->type().id(), cudf::type_id::INT64);
  for (const auto& segments : std::vector<std::vector<int32_t>>{
           {0, 1, 2, 3, 4, 5}, {0, 2, 4, 5}, {3}}) {
    SCOPED_TRACE(fmt::format("segments {}", fmt::join(segments, ",")));
    cudf::type_id stagedOffsets{};
    EXPECT_EQ(
        decode(table, "c", segments, nullptr, &stagedOffsets),
        expectedRows(comments_, segments));
    EXPECT_EQ(stagedOffsets, cudf::type_id::INT64);
  }
}

// Characters LZ4 cannot shrink stay a plain strings column.
TEST_F(EncodedLz4Test, keepsIncompressibleStringsPlain) {
  const auto& table = load("lz4", true, true);
  const auto& column = table.column("r");
  EXPECT_TRUE(column.compressedChars == nullptr);
  ASSERT_TRUE(column.values != nullptr);
  EXPECT_EQ(decode(table, "r", {0, 3, 5}), expectedRows(noise_, {0, 3, 5}));
}

// Device-resident tables are never compressed.
TEST_F(EncodedLz4Test, leavesDeviceResidentTablesPlain) {
  const auto& table = load("device", false, true);
  const auto& column = table.column("c");
  EXPECT_TRUE(column.compressedChars == nullptr);
  ASSERT_TRUE(column.values != nullptr);
  EXPECT_EQ(decode(table, "c", {1, 2}), expectedRows(comments_, {1, 2}));
}

// Compressed characters are read only through stageSegments().
TEST_F(EncodedLz4Test, rejectsDecodingUnstagedCompressedStrings) {
  const auto& table = load("lz4", true, true);
  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  const auto mr = cudf_velox::get_output_mr();
  const std::vector<int32_t> segments{0};
  rmm::device_uvector<int32_t> deviceSegments(1, stream, mr);
  checkCuda(cudaMemcpyAsync(
      deviceSegments.data(),
      segments.data(),
      sizeof(int32_t),
      cudaMemcpyHostToDevice,
      stream.get()));
  stream.sync();
  EXPECT_THROW(
      encoded::decodeColumn(
          table,
          table.column("c"),
          segments,
          deviceSegments,
          std::nullopt,
          stream,
          mr),
      std::runtime_error);
}

// A block that decompresses to a size other than its metadata implies fails
// the decode of the staged column instead of yielding its strings. The
// compressed data stays intact: one more character in the metadata makes the
// last, partial block expect one byte more than it decompresses to, still
// within the output allocated for it. Only the size check is exercised; no
// malformed data reaches nvCOMP.
TEST_F(EncodedLz4Test, reportsBlockSizeMismatch) {
  const auto& table = load("lz4", true, true);
  auto& column = const_cast<encoded::EncodedColumn&>(table.column("c"));
  ASSERT_TRUE(column.compressedChars != nullptr);
  auto& compressed = *column.compressedChars;
  // The last block is partial, so one more byte stays within it.
  ASSERT_NE(compressed.numChars % compressed.blockBytes, 0);
  compressed.numChars += 1;
  // The last segment's rows end at the last character, in the last block.
  try {
    decode(table, "c", {5});
    ADD_FAILURE() << "Expected the size mismatch to be reported";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("decompressed to"), std::string::npos)
        << e.what();
  }
  compressed.numChars -= 1;
  // With the metadata restored, the same segment stages exactly.
  EXPECT_EQ(decode(table, "c", {5}), expectedRows(comments_, {5}));
}

// Staging from several threads at once, many times, reuses pinned memory
// from cuDF's pool across threads and streams; each batch still stages its
// exact strings.
TEST_F(EncodedLz4Test, restagesConcurrentlyAndRepeatedly) {
  const auto& table = load("lz4", true, true);
  ASSERT_TRUE(table.column("c").compressedChars != nullptr);
  const std::vector<std::vector<int32_t>> batches{
      {0, 1, 2, 3, 4, 5}, {1, 2, 3}, {0, 2, 5}, {3}, {2}, {5}, {3, 4}};
  std::vector<std::thread> threads;
  for (int32_t t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      cudf_velox::ensureCudaContextForThread();
      for (int32_t i = 0; i < 25; ++i) {
        const auto& segments = batches[(t + i) % batches.size()];
        EXPECT_EQ(
            decode(table, "c", segments), expectedRows(comments_, segments))
            << "thread " << t << " iteration " << i;
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

// A staged batch dropped without being decoded waits for its copies before
// its pinned memory goes back to the pool; staging afterwards is exact.
TEST_F(EncodedLz4Test, dropsStagedBatchesWithoutDecoding) {
  const auto& table = load("lz4", true, true);
  ASSERT_TRUE(table.column("c").compressedChars != nullptr);
  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  const auto mr = cudf_velox::get_output_mr();
  for (int32_t i = 0; i < 10; ++i) {
    auto staged = encoded::stageSegments(
        table, {"c"}, std::vector<int32_t>{0, 1, 2, 3, 4, 5}, stream, mr);
    EXPECT_TRUE(staged->column("c").decompressionCheck != nullptr);
  }
  EXPECT_EQ(decode(table, "c", {0, 2, 5}), expectedRows(comments_, {0, 2, 5}));
}

// A scan filters on and returns a compressed column.
TEST_F(EncodedLz4Test, scansCompressedStrings) {
  load("lz4", true, true);
  createDuckDbTable({data_});
  auto plan = PlanBuilder(pool())
                  .startTableScan(kConnectorId)
                  .tableName("lz4")
                  .outputType(ROW({"k", "c"}, {INTEGER(), VARCHAR()}))
                  .remainingFilter("c like '%special%' or c like '%δοκιμή%'")
                  .endTableScan()
                  .planNode();
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .splits(encoded::makeEncodedSplits(kConnectorId, "lz4", 2))
      .maxDrivers(2)
      .assertResults(
          "SELECT k, c FROM tmp "
          "WHERE c LIKE '%special%' OR c LIKE '%δοκιμή%'");
}

} // namespace
