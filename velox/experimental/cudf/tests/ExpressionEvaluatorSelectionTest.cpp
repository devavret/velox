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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/JitExpression.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/type/Type.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::test_utils;

namespace {

class CudfExpressionSelectionTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    facebook::velox::functions::sparksql::registerFunctions();
    facebook::velox::functions::prestosql::registerAllScalarFunctions();
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool("", false);
    queryCtx_ = core::QueryCtx::create();
    execCtx_ = std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get());
    cudf_velox::registerCudf();
    cudf_velox::registerSparkFunctions("");
    rowType_ = ROW({
        {"a", BIGINT()},
        {"b", BIGINT()},
        {"c", INTEGER()},
        {"name", VARCHAR()},
        {"date", TIMESTAMP()},
        {"c", INTEGER()},
    });

    parse::registerTypeResolver();
  }

  void TearDown() override {
    cudf_velox::unregisterFunctions();
    cudf_velox::unregisterCudf();
    execCtx_.reset();
    queryCtx_.reset();
    pool_.reset();
  }

  template <typename T>
  std::unique_ptr<cudf::column> makeFixedWidthColumn(
      cudf::type_id type,
      const std::vector<T>& data) {
    auto column = cudf::make_fixed_width_column(
        cudf::data_type{type},
        static_cast<cudf::size_type>(data.size()),
        cudf::mask_state::UNALLOCATED,
        stream_,
        mr());
    auto status = cudaMemcpy(
        column->mutable_view().data<T>(),
        data.data(),
        data.size() * sizeof(T),
        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      VELOX_FAIL("cudaMemcpy failed: {}", cudaGetErrorString(status));
    }
    return column;
  }

  rmm::device_async_resource_ref mr() {
    return cudf::get_current_device_resource_ref();
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<core::ExecCtx> execCtx_;
  RowTypePtr rowType_;
  rmm::cuda_stream_view stream_ = rmm::cuda_stream_default;
};

class FunctionExpressionOnlyEvaluatorGuard {
 public:
  FunctionExpressionOnlyEvaluatorGuard() {
    unregisterCudfExpressionEvaluator(kAstEvaluatorName);
    unregisterCudfExpressionEvaluator(kJitEvaluatorName);
  }

  ~FunctionExpressionOnlyEvaluatorGuard() {
    auto& config = CudfConfig::getInstance();
    if (config.astExpressionEnabled) {
      registerAstEvaluator(config.astExpressionPriority);
    }
    if (config.jitExpressionEnabled) {
      registerJitEvaluator(config.jitExpressionPriority);
    }
  }
};

TEST_F(CudfExpressionSelectionTest, astRoot) {
  auto prevAst = CudfConfig::getInstance().astExpressionEnabled;
  auto prevJit = CudfConfig::getInstance().jitExpressionEnabled;
  CudfConfig::getInstance().astExpressionEnabled = true;
  CudfConfig::getInstance().jitExpressionEnabled = true;
  auto expr = compileExecExpr("a + c", rowType_, execCtx_.get());
  auto cudfExpr = createCudfExpression(expr, rowType_);
  auto* ast = dynamic_cast<ASTExpression*>(cudfExpr.get());
  auto* jit = dynamic_cast<JitExpression*>(cudfExpr.get());
  ASSERT_TRUE(ast != nullptr || jit != nullptr);
  CudfConfig::getInstance().astExpressionEnabled = prevAst;
  CudfConfig::getInstance().jitExpressionEnabled = prevJit;
}

TEST_F(CudfExpressionSelectionTest, functionRoot) {
  auto expr = compileExecExpr("lower(name)", rowType_, execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/false));
  auto cudfExpr = createCudfExpression(expr, rowType_);
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
}

TEST_F(CudfExpressionSelectionTest, astTopLevelWithFunctionPrecompute) {
  auto prevAst = CudfConfig::getInstance().astExpressionEnabled;
  auto prevJit = CudfConfig::getInstance().jitExpressionEnabled;
  CudfConfig::getInstance().astExpressionEnabled = true;
  CudfConfig::getInstance().jitExpressionEnabled = true;
  auto expr = compileExecExpr(
      "(year(date) > 2020) AND (length(name) < 10)", rowType_, execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/false));
  auto cudfExpr = createCudfExpression(expr, rowType_);
  auto* ast = dynamic_cast<ASTExpression*>(cudfExpr.get());
  auto* jit = dynamic_cast<JitExpression*>(cudfExpr.get());
  ASSERT_TRUE(ast != nullptr || jit != nullptr);
  CudfConfig::getInstance().astExpressionEnabled = prevAst;
  CudfConfig::getInstance().jitExpressionEnabled = prevJit;
}

TEST_F(CudfExpressionSelectionTest, functionTopLevelWithNestedFunction) {
  auto expr =
      compileExecExpr("lower(substr(name, 1, 5))", rowType_, execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/false));
  auto cudfExpr = createCudfExpression(expr, rowType_);

  // Top level should be Function
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
}

// Disabled because this test segfaults in CI in compileExecExpr step which does
// not use cudf code.
TEST_F(CudfExpressionSelectionTest, DISABLED_functionTopLevelWithNestedAst) {
  auto expr = compileExecExpr(
      "hash_with_seed(42, add(a, b))",
      rowType_,
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  auto cudfExpr = createCudfExpression(expr, rowType_);
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
}

// Disabled because this test segfaults in CI in compileExecExpr step which does
// not use cudf code.
TEST_F(
    CudfExpressionSelectionTest,
    DISABLED_signatureEnforcesConstantArgsSplit) {
  // OK: delimiter and limit are constants
  auto ok = compileExecExpr(
      "split(name, ',', 3)",
      rowType_,
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  ASSERT_TRUE(canBeEvaluatedByCudf(ok, /*deep=*/true));

  // Bad: delimiter is not a constant
  auto bad = compileExecExpr(
      "split(name, name, 3)",
      rowType_,
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  ASSERT_FALSE(canBeEvaluatedByCudf(bad, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, signatureEnforcesConstantArgsLike) {
  // OK: pattern is a constant
  auto ok = compileExecExpr("like(name, '%abc%')", rowType_, execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok, /*deep=*/true));

  // Bad: pattern is not a constant
  auto bad = compileExecExpr("like(name, name)", rowType_, execCtx_.get());
  ASSERT_FALSE(FunctionExpression::canEvaluate(bad));
}

TEST_F(CudfExpressionSelectionTest, signatureArityAndConstantsSubstr) {
  // OK: 2-arg substr with constant start
  auto ok2 = compileExecExpr("substr(name, 1)", rowType_, execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok2, /*deep=*/true));

  // OK: 3-arg substr with constant start and length
  auto ok3 = compileExecExpr("substr(name, 1, 5)", rowType_, execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok3, /*deep=*/true));

  // Bad: start must be constant
  auto badConst = compileExecExpr("substr(name, a)", rowType_, execCtx_.get());
  ASSERT_FALSE(FunctionExpression::canEvaluate(badConst));
}

TEST_F(CudfExpressionSelectionTest, signatureCastsInDivide) {
  // OK: numeric args are castable to double
  auto ok = compileExecExpr("divide(a, b)", rowType_, execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, cpuFallbackRoot) {
  FunctionExpressionOnlyEvaluatorGuard functionExpressionOnly;

  auto rowType = ROW({
      {"a", BIGINT()},
      {"b", BIGINT()},
      {"c", INTEGER()},
  });
  auto expr = compileExecExpr("bit_count(a, b)", rowType, execCtx_.get());
  ASSERT_FALSE(FunctionExpression::canEvaluate(expr));
  ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/true));

  auto cudfExpr = createCudfExpression(expr, rowType);
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_EQ(functionExpr, nullptr);

  auto colA = makeFixedWidthColumn<int64_t>(cudf::type_id::INT64, {1, 3, 7});
  auto colB = makeFixedWidthColumn<int64_t>(cudf::type_id::INT64, {64, 64, 64});
  auto colC = makeFixedWidthColumn<int32_t>(cudf::type_id::INT32, {1, 2, 3});
  std::vector<cudf::column_view> inputViews = {
      colA->view(), colB->view(), colC->view()};

  auto result = cudfExpr->eval(inputViews, stream_, mr(), /*finalize=*/true);
  auto resultView = asView(result);
  ASSERT_EQ(resultView.size(), 3);
  ASSERT_EQ(resultView.type().id(), cudf::type_id::INT64);
}

TEST_F(
    CudfExpressionSelectionTest,
    prestoGpuSimpleFunctionsUseFunctionExpression) {
  FunctionExpressionOnlyEvaluatorGuard functionExpressionOnly;

  struct FunctionCase {
    const char* name;
    const char* sql;
  };

  const std::vector<FunctionCase> cases = {
      {"plus", "a + b"},
      {"minus", "a - b"},
      {"multiply", "a * b"},
      {"divide", "cast(a as double) / cast(b as double)"},
      {"modulus", "mod(a, b)"},
      {"negate", "negate(a)"},
      {"abs", "abs(a)"},
      {"ceil", "ceil(cast(a as double))"},
      {"floor", "floor(cast(a as double))"},
      {"truncate", "truncate(cast(a as double))"},
      {"exp", "exp(cast(a as double))"},
      {"ln", "ln(cast(a as double))"},
      {"log2", "log2(cast(a as double))"},
      {"log10", "log10(cast(a as double))"},
      {"sqrt", "sqrt(cast(a as double))"},
      {"cbrt", "cbrt(cast(a as double))"},
      {"sin", "sin(cast(a as double))"},
      {"cos", "cos(cast(a as double))"},
      {"tan", "tan(cast(a as double))"},
      {"asin", "asin(cast(a as double))"},
      {"acos", "acos(cast(a as double))"},
      {"atan", "atan(cast(a as double))"},
      {"atan2", "atan2(cast(a as double), cast(b as double))"},
      {"cosh", "cosh(cast(a as double))"},
      {"tanh", "tanh(cast(a as double))"},
      {"power", "power(cast(a as double), cast(b as double))"},
      {"sign", "sign(a)"},
      {"radians", "radians(cast(a as double))"},
      {"degrees", "degrees(cast(a as double))"},
      {"is_finite", "is_finite(cast(a as double))"},
      {"is_infinite", "is_infinite(cast(a as double))"},
      {"is_nan", "is_nan(cast(a as double))"},
      {"clamp", "clamp(a, b, a)"},
      {"lt", "a < b"},
      {"gt", "a > b"},
      {"lte", "a <= b"},
      {"gte", "a >= b"},
      {"between", "a between b and a"},
      {"bitwise_and", "bitwise_and(a, b)"},
      {"bitwise_or", "bitwise_or(a, b)"},
      {"bitwise_xor", "bitwise_xor(a, b)"},
      {"bitwise_not", "bitwise_not(a)"},
      {"bitwise_left_shift", "bitwise_left_shift(a, c)"},
      {"bitwise_right_shift", "bitwise_right_shift(a, c)"},
      {"bitwise_arithmetic_shift_right",
       "bitwise_right_shift_arithmetic(a, c)"},
  };

  auto colA = makeFixedWidthColumn<int64_t>(cudf::type_id::INT64, {1, 1, 1});
  auto colB = makeFixedWidthColumn<int64_t>(cudf::type_id::INT64, {1, 2, 3});
  auto colC = makeFixedWidthColumn<int32_t>(cudf::type_id::INT32, {1, 2, 3});
  std::vector<cudf::column_view> inputViews = {
      colA->view(), colB->view(), colC->view()};

  for (const auto& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    auto expr = compileExecExpr(testCase.sql, rowType_, execCtx_.get());
    ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/true)) << testCase.sql;

    auto cudfExpr = createCudfExpression(expr, rowType_);
    auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
    ASSERT_NE(functionExpr, nullptr) << testCase.sql;

    auto result = cudfExpr->eval(inputViews, stream_, mr(), /*finalize=*/true);
    ASSERT_EQ(asView(result).size(), 3) << testCase.sql;
  }
}

TEST_F(CudfExpressionSelectionTest, signatureVarargsHashWithSeed) {
  facebook::velox::functions::sparksql::registerFunctions();

  // Multi-column hash_with_seed is not supported by FunctionExpression because
  // cudf's murmurhash3_x86_32 combines columns via hash_combine(h(col0, seed),
  // h(col1, seed)), while Spark hashes iteratively: h(col1, h(col0, seed)).
  // The CPU fallback evaluator can still evaluate the expression.
  auto multiCol = compileExecExpr(
      "hash_with_seed(42, a, b)",
      rowType_,
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  ASSERT_FALSE(FunctionExpression::canEvaluate(multiCol));
  ASSERT_TRUE(canBeEvaluatedByCudf(multiCol, /*deep=*/true));

  // Single-column hash_with_seed is supported (no column combining needed).
  auto singleCol = compileExecExpr(
      "hash_with_seed(42, a)",
      rowType_,
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  ASSERT_TRUE(canBeEvaluatedByCudf(singleCol, /*deep=*/true));

  // Bad: first arg must be constant seed
  try {
    auto bad = compileExecExpr(
        "hash_with_seed(c, b)",
        rowType_,
        execCtx_.get(),
        {.parseIntegerAsBigint = false, .functionPrefix = ""});
    // If compilation succeeds, the compiled check must fail.
    ASSERT_FALSE(FunctionExpression::canEvaluate(bad));
  } catch (const VeloxUserError&) {
    // Treat compile-time validation failure as unsupported.
    SUCCEED();
  }
}

TEST_F(CudfExpressionSelectionTest, signatureTypeVariableCoalesce) {
  // OK: same type BIGINT
  auto ok1 = compileExecExpr("coalesce(a, b)", rowType_, execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok1, /*deep=*/true));

  // OK: VARCHAR with literal
  auto ok2 = compileExecExpr("coalesce(name, 'x')", rowType_, execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok2, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, signatureTypeVariableSwitchIf) {
  // OK: boolean + same type BIGINT
  auto ok1 = compileExecExpr("if(true, a, b)", rowType_, execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok1, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, DISABLED_castAndTryCast) {
  // TODO (dm): This is required for passing of castAndTryCast test but breaks
  // others. This is because ASTExpr agrees to support bad casts. remove after
  // ASTExpr checks cast types
  // CudfConfig::getInstance().astExpressionEnabled = false;

  // OK: cast bigint -> double (supported by cuDF)
  auto okCast = compileExecExpr("cast(a AS double)", rowType_, execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okCast, /*deep=*/true));

  // OK: try_cast bigint -> double (supported by cuDF)
  auto okTryCast =
      compileExecExpr("try_cast(a AS double)", rowType_, execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okTryCast, /*deep=*/true));

  // BAD: cast boolean -> date (expected unsupported by cuDF)
  auto badCast = compileExecExpr(
      "cast(length(name) < 10 AS date)", rowType_, execCtx_.get());
  ASSERT_FALSE(canBeEvaluatedByCudf(badCast, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, constantFoldingStringAllocatesOnCompile) {
  // Build a constant string expression that should be folded at compile time.
  // This triggers creation of a ConstantExpr and allocates memory for the
  // folded string using the ExecCtx pool.
  auto typed =
      parseAndInferTypedExpr("lower('ABCDEF')", rowType_, execCtx_.get());

  std::vector<core::TypedExprPtr> exprs;
  exprs.push_back(typed);

  auto exprSet = exec::makeExprSetFromFlag(
      std::move(exprs), execCtx_.get(), /*lazyDereference=*/false);

  auto compiled = exprSet->expr(0);
  auto* c = dynamic_cast<facebook::velox::exec::ConstantExpr*>(compiled.get());
  ASSERT_NE(c, nullptr);
  // Verify the constant vector was created using the provided ExecCtx pool.
  ASSERT_EQ(c->value()->pool(), execCtx_->pool());
}

} // namespace
