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

// Tests for the multi-literal LIKE fast path (cudf.like_multiliteral_enabled).
//
// Rows that are not valid UTF-8 only ever reach likeMultiLiteral() directly.
// cudf::strings::like, the fallback, can misbehave on such rows, so they are
// never evaluated through LikeFunction or with a pattern that falls back.

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/LikeMultiLiteral.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

using Rows = std::vector<std::optional<std::string>>;
using Results = std::vector<std::optional<bool>>;

std::string likeSql(const std::string& pattern) {
  return "like(c0, '" + pattern + "')";
}

std::string patternOf(const std::vector<std::string>& literals) {
  std::string pattern = "%";
  for (const auto& literal : literals) {
    pattern += literal + "%";
  }
  return pattern;
}

// The CPU substrings search (functions/lib/Re2Functions.cpp,
// matchSubstringsPattern): the leftmost byte-wise occurrence of each literal,
// each search starting after the previous match.
bool referenceMatch(
    std::string_view value,
    const std::vector<std::string>& literals) {
  size_t pos = 0;
  for (const auto& literal : literals) {
    const auto found = value.find(literal, pos);
    if (found == std::string_view::npos) {
      return false;
    }
    pos = found + literal.size();
  }
  return true;
}

Results reference(const Rows& rows, const std::vector<std::string>& literals) {
  Results out;
  out.reserve(rows.size());
  for (const auto& row : rows) {
    out.push_back(
        row.has_value() ? std::optional<bool>(referenceMatch(*row, literals))
                        : std::nullopt);
  }
  return out;
}

std::string printable(const std::optional<std::string>& row) {
  if (!row.has_value()) {
    return "NULL";
  }
  std::string out;
  for (const unsigned char ch : *row) {
    if (ch >= 0x20 && ch < 0x7f && ch != '\\') {
      out.push_back(static_cast<char>(ch));
    } else {
      char escaped[8];
      std::snprintf(escaped, sizeof(escaped), "\\x%02x", ch);
      out += escaped;
    }
  }
  return out;
}

std::string printable(const std::optional<bool>& value) {
  return value.has_value() ? (*value ? "true" : "false") : "NULL";
}

void expectResults(
    const Results& actual,
    const Results& expected,
    const Rows& rows) {
  ASSERT_EQ(actual.size(), expected.size());
  ASSERT_EQ(rows.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      ADD_FAILURE() << "row " << i << " '" << printable(rows[i]) << "': got "
                    << printable(actual[i]) << ", expected "
                    << printable(expected[i]);
      return;
    }
  }
}

// Builds a strings column from host bytes with 'offsetType' offsets.
std::unique_ptr<cudf::column> makeStrings(
    const Rows& rows,
    cudf::type_id offsetType,
    cuda::stream_ref stream) {
  const auto mr = get_output_mr();
  const auto numRows = static_cast<cudf::size_type>(rows.size());
  std::string chars;
  std::vector<int64_t> offsets64{0};
  std::vector<int32_t> offsets32{0};
  std::vector<cudf::bitmask_type> mask(
      cudf::num_bitmask_words(numRows), ~cudf::bitmask_type{0});
  cudf::size_type nullCount = 0;
  for (cudf::size_type i = 0; i < numRows; ++i) {
    if (rows[i].has_value()) {
      chars += *rows[i];
    } else {
      ++nullCount;
      mask[i / 32] &= ~(cudf::bitmask_type{1} << (i % 32));
    }
    offsets64.push_back(static_cast<int64_t>(chars.size()));
    offsets32.push_back(static_cast<int32_t>(chars.size()));
  }

  auto offsets = cudf::make_numeric_column(
      cudf::data_type{offsetType},
      numRows + 1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  if (offsetType == cudf::type_id::INT64) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        offsets->mutable_view().data<int64_t>(),
        offsets64.data(),
        offsets64.size() * sizeof(int64_t),
        cudaMemcpyHostToDevice,
        stream.get()));
  } else {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        offsets->mutable_view().data<int32_t>(),
        offsets32.data(),
        offsets32.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream.get()));
  }
  rmm::device_buffer charsBuffer(chars.size(), stream, mr);
  if (!chars.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        charsBuffer.data(),
        chars.data(),
        chars.size(),
        cudaMemcpyHostToDevice,
        stream.get()));
  }
  auto nullMask = cudf::create_null_mask(
      nullCount > 0 ? numRows : 0,
      nullCount > 0 ? cudf::mask_state::ALL_VALID
                    : cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  if (nullCount > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        nullMask.data(),
        mask.data(),
        mask.size() * sizeof(cudf::bitmask_type),
        cudaMemcpyHostToDevice,
        stream.get()));
  }
  // The host vectors above are the sources of the copies.
  stream.sync();
  return cudf::make_strings_column(
      numRows,
      std::move(offsets),
      std::move(charsBuffer),
      nullCount,
      std::move(nullMask));
}

Results copyResults(cudf::column_view result, cuda::stream_ref stream) {
  EXPECT_EQ(result.type().id(), cudf::type_id::BOOL8);
  const auto size = result.size();
  std::vector<uint8_t> values(size);
  std::vector<cudf::bitmask_type> mask;
  if (size > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        values.data(),
        result.data<bool>(),
        size,
        cudaMemcpyDeviceToHost,
        stream.get()));
    if (result.nullable()) {
      mask.resize(cudf::num_bitmask_words(result.offset() + size));
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          mask.data(),
          result.null_mask(),
          mask.size() * sizeof(cudf::bitmask_type),
          cudaMemcpyDeviceToHost,
          stream.get()));
    }
  }
  stream.sync();
  Results out(size);
  cudf::size_type nulls = 0;
  for (cudf::size_type i = 0; i < size; ++i) {
    const auto bit = result.offset() + i;
    const bool valid = mask.empty() || ((mask[bit / 32] >> (bit % 32)) & 1);
    if (valid) {
      out[i] = values[i] != 0;
    } else {
      ++nulls;
    }
  }
  EXPECT_EQ(result.null_count(), nulls);
  return out;
}

Rows validRows() {
  return {
      std::nullopt,
      "",
      "a",
      "aa",
      "ab",
      "ba",
      "aba",
      "abab",
      "ababa",
      "abaaba",
      "special requests",
      "requests special",
      "specialrequests",
      "special\nrequests",
      "x special\n\nrequests y",
      std::string("special\0requests", 16),
      std::string("a\0b\0a\0b", 7),
      "caf\xc3\xa9 special requests",
      "sp\xc3\xa9"
      "cial requests",
      "\xe2\x82\xac"
      "special"
      "\xe2\x82\xac"
      "requests"
      "\xf0\x9f\x98\x80",
      "SPECIAL requests",
      "specia lrequests",
      "pecial requests special requests",
      "xxxxxxxa",
      "xxxxxxxab",
      "xxxxxxxxab",
      "abxxxxxxba",
      std::string(70, 'x') + "special" + std::string(70, 'y') + "requests",
      std::string(3000, 'q') + "special" + std::string(5, 'z') + "requests",
      std::string(40, 'q'),
      std::nullopt,
      "ec",
  };
}

// Eligible patterns, all of which the fast path must take.
std::vector<std::string> eligiblePatterns() {
  return {
      "%special%requests%",
      "%requests%special%",
      "%special%",
      "%a%",
      "%a%a%",
      "%aba%aba%",
      "%%ab%%%ba%%",
      "%ab%ba%",
      "%a%b%a%b%",
      "%x%a%",
      "%\n%",
      "%ec%",
      "%l r%",
      "%" + std::string(9, 'x') + "%",
      "%" + std::string(32, 'q') + "%",
  };
}

// Patterns the fast path must leave to cudf::strings::like. The NUL pattern
// is covered by classifyRejects only: on the fallback cuDF treats NUL as an
// escape character when no ESCAPE is given, which differs from the CPU and is
// left as it is.
std::vector<std::string> fallbackPatterns() {
  return {
      "",
      "%",
      "%%%",
      "special requests",
      "special%",
      "%requests",
      "spec%req%",
      "%spec%ests",
      "%sp_cial%",
      "%_%",
      "%caf\xc3\xa9%",
      "%" + std::string(33, 'q') + "%",
      "%a%b%c%d%e%",
  };
}

class LikeMultiLiteralTest : public CudfFunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    savedAllowCpuFallback_ = CudfConfig::getInstance().allowCpuFallback;
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
  }

  static void TearDownTestCase() {
    unregisterCudf();
    CudfConfig::getInstance().allowCpuFallback = savedAllowCpuFallback_;
  }

  void SetUp() override {
    CudfFunctionBaseTest::SetUp();
    savedLikeMultiLiteral_ = CudfConfig::getInstance().likeMultiLiteralEnabled;
    CudfConfig::getInstance().likeMultiLiteralEnabled = true;
  }

  void TearDown() override {
    CudfConfig::getInstance().likeMultiLiteralEnabled = savedLikeMultiLiteral_;
    CudfFunctionBaseTest::TearDown();
  }

  RowVectorPtr makeInput(const Rows& rows) {
    return makeRowVector({makeNullableFlatVector<std::string>(rows)});
  }

  // Evaluates 'sql' on the CPU.
  Results evaluateCpu(const std::string& sql, const Rows& rows) {
    auto input = makeInput(rows);
    auto exprSet = compileExpression(sql, asRowType(input->type()));
    auto result = functions::test::FunctionBaseTest::evaluate(*exprSet, input);
    const auto* values = result->as<SimpleVector<bool>>();
    VELOX_CHECK_NOT_NULL(values);
    Results out(rows.size());
    for (vector_size_t i = 0; i < static_cast<vector_size_t>(rows.size());
         ++i) {
      if (!result->isNullAt(i)) {
        out[i] = values->valueAt(i);
      }
    }
    return out;
  }

  // Evaluates 'sql' through the cuDF expression evaluator over 'input' as c0.
  Results evaluateGpu(
      const std::string& sql,
      cudf::column_view input,
      cuda::stream_ref stream) {
    const auto rowType = ROW({"c0"}, {VARCHAR()});
    auto optimized = expression::optimize(
        makeTypedExpr(sql, rowType), execCtx_.queryCtx(), execCtx_.pool());
    auto expression = createCudfExpression(
        optimized, rowType, pool_.get(), execCtx_.queryCtx()->queryConfig());
    auto result = expression->eval({input}, stream, get_output_mr());
    return copyResults(asView(result), stream);
  }

  // Checks likeMultiLiteral() on 'rows' and on a slice of them, with both
  // offset widths, against 'expected'.
  void expectDirect(
      const std::vector<std::string>& literals,
      const Rows& rows,
      const Results& expected,
      cuda::stream_ref stream) {
    const auto pattern =
        classifyLikeMultiLiteral(patternOf(literals), std::nullopt);
    ASSERT_TRUE(pattern.has_value());
    const auto numRows = static_cast<cudf::size_type>(rows.size());
    const auto begin = std::min<cudf::size_type>(3, numRows);
    const auto end = std::max<cudf::size_type>(begin, numRows - 2);
    for (const auto offsetType : {cudf::type_id::INT32, cudf::type_id::INT64}) {
      SCOPED_TRACE(
          offsetType == cudf::type_id::INT32 ? "INT32 offsets"
                                             : "INT64 offsets");
      auto column = makeStrings(rows, offsetType, stream);
      auto full = likeMultiLiteral(
          cudf::strings_column_view(column->view()),
          *pattern,
          stream,
          get_output_mr());
      ASSERT_TRUE(full != nullptr);
      expectResults(copyResults(full->view(), stream), expected, rows);

      auto slice = cudf::slice(column->view(), {begin, end}, stream)[0];
      auto sliced = likeMultiLiteral(
          cudf::strings_column_view(slice), *pattern, stream, get_output_mr());
      ASSERT_TRUE(sliced != nullptr);
      expectResults(
          copyResults(sliced->view(), stream),
          Results(expected.begin() + begin, expected.begin() + end),
          Rows(rows.begin() + begin, rows.begin() + end));
    }
  }

  inline static bool savedAllowCpuFallback_{true};
  bool savedLikeMultiLiteral_{false};
};

std::optional<std::vector<std::string>> literalsOf(
    std::string_view pattern,
    std::optional<std::string_view> escape = std::nullopt) {
  auto parsed = classifyLikeMultiLiteral(pattern, escape);
  if (!parsed.has_value()) {
    return std::nullopt;
  }
  return parsed->literals();
}

using Literals = std::vector<std::string>;

TEST(LikeMultiLiteralClassifyTest, accepts) {
  EXPECT_EQ(
      literalsOf("%special%requests%"), (Literals{"special", "requests"}));
  EXPECT_EQ(literalsOf("%a%"), (Literals{"a"}));
  // Consecutive '%' collapse.
  EXPECT_EQ(literalsOf("%%ab%%%ba%%"), (Literals{"ab", "ba"}));
  EXPECT_EQ(literalsOf("%a\nb%"), (Literals{"a\nb"}));
  EXPECT_EQ(literalsOf("%a%b%c%d%"), (Literals{"a", "b", "c", "d"}));
  EXPECT_EQ(literalsOf("%a%a%"), (Literals{"a", "a"}));
  const std::string longest(kLikeMultiLiteralMaxLiteralBytes, 'x');
  EXPECT_EQ(literalsOf("%" + longest + "%"), (Literals{longest}));
  // '#' and '\' are ordinary bytes when they are not the escape character.
  EXPECT_EQ(literalsOf("%a#b%"), (Literals{"a#b"}));
  EXPECT_EQ(literalsOf("%a\\b%"), (Literals{"a\\b"}));
  // An escape character that the pattern never uses has no effect.
  EXPECT_EQ(literalsOf("%a%b%", "#"), (Literals{"a", "b"}));
  EXPECT_EQ(literalsOf("%a%b%", "_"), (Literals{"a", "b"}));
}

TEST(LikeMultiLiteralClassifyTest, rejects) {
  // Anchored, exact and empty patterns.
  EXPECT_FALSE(literalsOf(""));
  EXPECT_FALSE(literalsOf("abc"));
  EXPECT_FALSE(literalsOf("a%"));
  EXPECT_FALSE(literalsOf("%a"));
  EXPECT_FALSE(literalsOf("a%b%"));
  EXPECT_FALSE(literalsOf("%a%b"));
  // Only '%'.
  EXPECT_FALSE(literalsOf("%"));
  EXPECT_FALSE(literalsOf("%%"));
  EXPECT_FALSE(literalsOf("%%%"));
  // '_' wildcard.
  EXPECT_FALSE(literalsOf("%a_b%"));
  EXPECT_FALSE(literalsOf("%_%"));
  // NUL and non-ASCII bytes.
  EXPECT_FALSE(literalsOf(std::string_view("%a\0b%", 5)));
  EXPECT_FALSE(literalsOf(std::string_view("%\0%", 3)));
  EXPECT_FALSE(literalsOf("%caf\xc3\xa9%"));
  EXPECT_FALSE(literalsOf("%\x80%"));
  // Escape characters the pattern uses, or that are not one byte.
  EXPECT_FALSE(literalsOf("%a#%b%", "#"));
  EXPECT_FALSE(literalsOf("%a%b%", "%"));
  EXPECT_FALSE(literalsOf("%a%b%", "a"));
  EXPECT_FALSE(literalsOf("%a%b%", "##"));
  EXPECT_FALSE(literalsOf("%a%b%", ""));
  // Descriptors larger than the kernel supports.
  EXPECT_FALSE(literalsOf("%a%b%c%d%e%"));
  EXPECT_FALSE(literalsOf(
      "%" + std::string(kLikeMultiLiteralMaxLiteralBytes + 1, 'x') + "%"));
}

TEST_F(LikeMultiLiteralTest, eligiblePatternsMatchCpu) {
  const auto rows = validRows();
  const auto input = makeInput(rows);
  const auto rowType = asRowType(input->type());
  for (const auto& pattern : eligiblePatterns()) {
    SCOPED_TRACE(printable(pattern));
    const auto literals = literalsOf(pattern);
    ASSERT_TRUE(literals.has_value());
    const auto before = likeMultiLiteralLaunchCountForTesting();
    assertExpressionMatchesCpu(likeSql(pattern), input, rowType);
    EXPECT_EQ(likeMultiLiteralLaunchCountForTesting(), before + 1);
    // The CPU result is also the reference search.
    expectResults(
        evaluateCpu(likeSql(pattern), rows), reference(rows, *literals), rows);
  }
}

TEST_F(LikeMultiLiteralTest, fallbackPatternsMatchCpu) {
  const auto rows = validRows();
  const auto input = makeInput(rows);
  const auto rowType = asRowType(input->type());
  for (const auto& pattern : fallbackPatterns()) {
    SCOPED_TRACE(printable(pattern));
    const auto before = likeMultiLiteralLaunchCountForTesting();
    assertExpressionMatchesCpu(likeSql(pattern), input, rowType);
    EXPECT_EQ(likeMultiLiteralLaunchCountForTesting(), before);
  }
}

TEST_F(LikeMultiLiteralTest, notLikeStaysSeparate) {
  const auto input = makeInput(validRows());
  const auto before = likeMultiLiteralLaunchCountForTesting();
  assertExpressionMatchesCpu(
      "c0 not like '%special%requests%'", input, asRowType(input->type()));
  EXPECT_EQ(likeMultiLiteralLaunchCountForTesting(), before + 1);
}

TEST_F(LikeMultiLiteralTest, nullScalarsKeepAllNullResult) {
  const auto input = makeInput(validRows());
  const auto rowType = asRowType(input->type());
  const auto before = likeMultiLiteralLaunchCountForTesting();
  assertExpressionMatchesCpu("like(c0, cast(null as varchar))", input, rowType);
  assertExpressionMatchesCpu(
      "like(c0, '%a%', cast(null as varchar))", input, rowType);
  EXPECT_EQ(likeMultiLiteralLaunchCountForTesting(), before);
}

TEST_F(LikeMultiLiteralTest, escapes) {
  const auto input = makeInput(validRows());
  const auto rowType = asRowType(input->type());

  // Unused escape characters qualify after validation.
  auto before = likeMultiLiteralLaunchCountForTesting();
  assertExpressionMatchesCpu("like(c0, '%a%b%', '#')", input, rowType);
  assertExpressionMatchesCpu("like(c0, '%a%b%', '_')", input, rowType);
  EXPECT_EQ(likeMultiLiteralLaunchCountForTesting(), before + 2);

  // An escape the pattern uses keeps cudf::strings::like.
  before = likeMultiLiteralLaunchCountForTesting();
  assertExpressionMatchesCpu("like(c0, '%a#%b%', '#')", input, rowType);
  assertExpressionMatchesCpu("like(c0, '%a#_b%', '#')", input, rowType);
  EXPECT_EQ(likeMultiLiteralLaunchCountForTesting(), before);

  // Escape validation errors are unchanged.
  VELOX_ASSERT_THROW(
      evaluate(makeTypedExpr("like(c0, '%a%b%', '##')", rowType), input),
      "Escape string must be a single character");
  VELOX_ASSERT_THROW(
      evaluate(makeTypedExpr("like(c0, '%a#x%', '#')", rowType), input),
      "Escape character must be followed by '%', '_' or the escape character itself");
  EXPECT_EQ(likeMultiLiteralLaunchCountForTesting(), before);
}

TEST_F(LikeMultiLiteralTest, disabledKeepsCudfLike) {
  CudfConfig::getInstance().likeMultiLiteralEnabled = false;
  const auto input = makeInput(validRows());
  const auto before = likeMultiLiteralLaunchCountForTesting();
  assertExpressionMatchesCpu(
      likeSql("%special%requests%"), input, asRowType(input->type()));
  EXPECT_EQ(likeMultiLiteralLaunchCountForTesting(), before);
}

TEST_F(LikeMultiLiteralTest, slicesAndOffsetWidths) {
  auto stream = cudfGlobalStreamPool().get_stream();
  const auto rows = validRows();
  const auto numRows = static_cast<cudf::size_type>(rows.size());
  const Literals literals{"special", "requests"};
  const auto sql = likeSql(patternOf(literals));
  const auto pattern = classifyLikeMultiLiteral(patternOf(literals), {});
  ASSERT_TRUE(pattern.has_value());
  const std::vector<std::pair<cudf::size_type, cudf::size_type>> ranges = {
      {0, numRows},
      {3, 17},
      {15, 16},
      {numRows - 1, numRows},
      {1, numRows - 1},
      {numRows, numRows},
  };
  for (const auto offsetType : {cudf::type_id::INT32, cudf::type_id::INT64}) {
    auto column = makeStrings(rows, offsetType, stream);
    for (const auto& [begin, end] : ranges) {
      SCOPED_TRACE(
          (offsetType == cudf::type_id::INT32 ? std::string("INT32 [")
                                              : std::string("INT64 [")) +
          std::to_string(begin) + ", " + std::to_string(end) + ")");
      auto slice = cudf::slice(column->view(), {begin, end}, stream)[0];
      const Rows sliceRows(rows.begin() + begin, rows.begin() + end);
      const auto expected = reference(sliceRows, literals);

      const auto before = likeMultiLiteralLaunchCountForTesting();
      auto direct = likeMultiLiteral(
          cudf::strings_column_view(slice), *pattern, stream, get_output_mr());
      ASSERT_TRUE(direct != nullptr);
      expectResults(copyResults(direct->view(), stream), expected, sliceRows);
      expectResults(evaluateGpu(sql, slice, stream), expected, sliceRows);
      // An empty slice needs no kernel.
      EXPECT_EQ(
          likeMultiLiteralLaunchCountForTesting(),
          before + (begin == end ? 0 : 2));
    }
  }
}

TEST_F(LikeMultiLiteralTest, tailsAndRowBoundaries) {
  auto stream = cudfGlobalStreamPool().get_stream();
  const std::vector<Literals> patterns = {
      {"a"}, {"ab"}, {"b"}, {"a", "b"}, {"xab"}};
  // The last row ends the characters at every alignment and is a few bytes
  // long, so the kernel reads a partial final word.
  for (int prefix = 0; prefix < 16; ++prefix) {
    for (const std::string last : {"a", "ab", "b", "ba", "xab", ""}) {
      const Rows rows = {std::string(prefix, 'x'), last};
      for (const auto& literals : patterns) {
        SCOPED_TRACE(
            "prefix " + std::to_string(prefix) + ", last '" + last +
            "', pattern " + patternOf(literals));
        const auto expected = reference(rows, literals);
        expectResults(
            evaluateCpu(likeSql(patternOf(literals)), rows), expected, rows);
        auto column = makeStrings(rows, cudf::type_id::INT64, stream);
        expectResults(
            evaluateGpu(likeSql(patternOf(literals)), column->view(), stream),
            expected,
            rows);
        auto lastRow = cudf::slice(column->view(), {1, 2}, stream)[0];
        expectResults(
            evaluateGpu(likeSql(patternOf(literals)), lastRow, stream),
            {expected[1]},
            {rows[1]});
      }
    }
  }

  // Candidates that straddle two rows, or that lie in the part of a word
  // before the row starts, must not match.
  const Rows boundaries = {
      "xa",
      "bx",
      "ab",
      "x",
      "a",
      "b",
      std::string(7, 'x') + "a",
      "b" + std::string(7, 'x'),
      "",
      "a",
      std::nullopt,
      "b",
  };
  for (const auto& literals : patterns) {
    SCOPED_TRACE(patternOf(literals));
    expectDirect(literals, boundaries, reference(boundaries, literals), stream);
    auto column = makeStrings(boundaries, cudf::type_id::INT32, stream);
    expectResults(
        evaluateGpu(likeSql(patternOf(literals)), column->view(), stream),
        evaluateCpu(likeSql(patternOf(literals)), boundaries),
        boundaries);
  }
}

TEST_F(LikeMultiLiteralTest, longStrings) {
  auto stream = cudfGlobalStreamPool().get_stream();
  std::mt19937 rng(7);
  const std::vector<Literals> patterns = {
      {"special", "requests"},
      {"abcdefghijklmnopqrstuvwxyz012345"},
      {"abcdefghi", "jklmnopqr", "stuvwxyz"},
      {"q", "r", "s", "t"},
  };
  Rows rows;
  for (int i = 0; i < 64; ++i) {
    std::string row(100 + rng() % 6000, '.');
    for (const auto& literals : patterns) {
      for (const auto& literal : literals) {
        if (rng() % 3 != 0) {
          const auto at = rng() % (row.size() - literal.size());
          row.replace(at, literal.size(), literal);
        }
      }
    }
    rows.push_back(i % 9 == 4 ? std::nullopt : std::optional(row));
  }
  const auto input = makeInput(rows);
  for (const auto& literals : patterns) {
    SCOPED_TRACE(patternOf(literals));
    const auto expected = evaluateCpu(likeSql(patternOf(literals)), rows);
    expectResults(reference(rows, literals), expected, rows);
    assertExpressionMatchesCpu(
        likeSql(patternOf(literals)), input, asRowType(input->type()));
    expectDirect(literals, rows, expected, stream);
  }
}

TEST_F(LikeMultiLiteralTest, unalignedCharactersFallBack) {
  auto stream = cudfGlobalStreamPool().get_stream();
  const auto mr = get_output_mr();
  const Rows rows = {"special requests", "x", "requests special", "special"};
  std::string chars;
  std::vector<int32_t> offsets{0};
  for (const auto& row : rows) {
    chars += *row;
    offsets.push_back(static_cast<int32_t>(chars.size()));
  }
  // Characters that start one byte past an 8-byte boundary.
  rmm::device_buffer charsBuffer(chars.size() + 1, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      static_cast<char*>(charsBuffer.data()) + 1,
      chars.data(),
      chars.size(),
      cudaMemcpyHostToDevice,
      stream.get()));
  auto offsetsColumn = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(offsets.size()),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      offsetsColumn->mutable_view().data<int32_t>(),
      offsets.data(),
      offsets.size() * sizeof(int32_t),
      cudaMemcpyHostToDevice,
      stream.get()));
  stream.sync();
  const cudf::column_view unaligned(
      cudf::data_type{cudf::type_id::STRING},
      static_cast<cudf::size_type>(rows.size()),
      static_cast<const char*>(charsBuffer.data()) + 1,
      nullptr,
      0,
      0,
      {offsetsColumn->view()});

  const Literals literals{"special", "requests"};
  const auto pattern = classifyLikeMultiLiteral(patternOf(literals), {});
  ASSERT_TRUE(pattern.has_value());
  const auto before = likeMultiLiteralLaunchCountForTesting();
  EXPECT_TRUE(
      likeMultiLiteral(
          cudf::strings_column_view(unaligned), *pattern, stream, mr) ==
      nullptr);
  // LikeFunction falls back to cudf::strings::like.
  expectResults(
      evaluateGpu(likeSql(patternOf(literals)), unaligned, stream),
      reference(rows, literals),
      rows);
  EXPECT_EQ(likeMultiLiteralLaunchCountForTesting(), before);
}

TEST_F(LikeMultiLiteralTest, invalidUtf8OnFastPathOnly) {
  auto stream = cudfGlobalStreamPool().get_stream();
  const Rows rows = {
      // A lead byte whose declared width covers ASCII bytes.
      "\xe0special requests",
      // A stray continuation byte.
      "x\x80special requests",
      std::string(8, '\x80') + "special requests",
      "\xc3(special\xa9requests",
      "spe\xff"
      "cial requests",
      std::string("\x80special\0\xa0requests", 18),
      std::nullopt,
      // A lead byte truncated by the end of the row, then of the characters.
      "special requests\xf0",
      "\xf0",
  };
  const std::vector<Literals> patterns = {
      {"special", "requests"}, {"s"}, {"ec"}, {"e", "e"}, {"("}};
  for (const auto& literals : patterns) {
    SCOPED_TRACE(patternOf(literals));
    // The CPU path searches bytes too, so it is the oracle here as well.
    const auto expected = evaluateCpu(likeSql(patternOf(literals)), rows);
    expectResults(reference(rows, literals), expected, rows);
    expectDirect(literals, rows, expected, stream);
  }
}

TEST_F(LikeMultiLiteralTest, fixedSeedFuzz) {
  auto stream = cudfGlobalStreamPool().get_stream();
  std::mt19937 rng(20261111);
  auto randomLiterals = [&]() {
    Literals literals(1 + rng() % kLikeMultiLiteralMaxLiterals);
    for (auto& literal : literals) {
      const auto length = rng() % 10 == 0 ? 9 + rng() % 4 : 1 + rng() % 3;
      for (size_t i = 0; i < length; ++i) {
        literal.push_back("abc"[rng() % 3]);
      }
    }
    return literals;
  };
  auto randomRows = [&](const std::vector<std::string>& pieces) {
    Rows rows(3000);
    for (auto& row : rows) {
      if (rng() % 10 == 0) {
        continue;
      }
      const auto length = rng() % 30 == 0 ? 64 + rng() % 256 : rng() % 40;
      std::string value;
      while (value.size() < length) {
        value += pieces[rng() % pieces.size()];
      }
      row = value;
    }
    return rows;
  };

  // Any bytes, including NUL, newline and invalid UTF-8: likeMultiLiteral()
  // only, against the CPU.
  const auto anyBytes = randomRows(
      {"a", "b", "c", "x", "\n", std::string(1, '\0'), "\x80", "\xe0", "\xf0"});
  // Valid UTF-8 only: also through LikeFunction.
  const auto validUtf8 = randomRows(
      {"a",
       "b",
       "c",
       "x",
       "\n",
       std::string(1, '\0'),
       "\xc3\xa9",
       "\xe2\x82\xac"});
  for (int i = 0; i < 40; ++i) {
    const auto literals = randomLiterals();
    // Doubled '%' collapse.
    const auto pattern =
        i % 4 == 0 ? "%" + patternOf(literals) + "%" : patternOf(literals);
    SCOPED_TRACE(pattern);
    const auto sql = likeSql(pattern);

    const auto expectedAny = evaluateCpu(sql, anyBytes);
    expectResults(reference(anyBytes, literals), expectedAny, anyBytes);
    expectDirect(literals, anyBytes, expectedAny, stream);

    const auto expectedValid = evaluateCpu(sql, validUtf8);
    auto column = makeStrings(validUtf8, cudf::type_id::INT32, stream);
    const auto before = likeMultiLiteralLaunchCountForTesting();
    expectResults(
        evaluateGpu(sql, column->view(), stream), expectedValid, validUtf8);
    EXPECT_EQ(likeMultiLiteralLaunchCountForTesting(), before + 1);
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox
