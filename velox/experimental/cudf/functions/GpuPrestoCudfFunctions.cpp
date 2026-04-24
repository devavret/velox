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

#include "velox/experimental/cudf/functions/GpuPrestoCudfFunctions.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/functions/GpuPrestoFunctions.h"
#include "velox/experimental/cudf/functions/GpuVectorFunction.h"

#include "velox/common/base/Exceptions.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/type/Type.h"

#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

struct GpuSimpleCudfVariant {
  std::string gpuName;
  cudf::type_id returnType;
  std::vector<cudf::type_id> argTypes;
  GpuSimpleCudfFunctionFactory factory;
  exec::FunctionSignaturePtr signature;
};

enum class ArgumentKind {
  kRuntimeColumn,
  kLiteral,
};

struct ArgumentBinding {
  ArgumentKind kind;
  size_t index;
};

struct CudfRegistrationState {
  bool enabled{false};
  std::string prefix;
  std::unordered_map<std::string, std::vector<GpuSimpleCudfVariant>> variants;
};

std::mutex& registrationMutex() {
  static std::mutex mutex;
  return mutex;
}

CudfRegistrationState& registrationState() {
  static CudfRegistrationState state;
  return state;
}

std::string typeName(cudf::type_id type) {
  switch (type) {
    case cudf::type_id::BOOL8:
      return "boolean";
    case cudf::type_id::INT8:
      return "tinyint";
    case cudf::type_id::INT16:
      return "smallint";
    case cudf::type_id::INT32:
      return "integer";
    case cudf::type_id::INT64:
      return "bigint";
    case cudf::type_id::FLOAT32:
      return "real";
    case cudf::type_id::FLOAT64:
      return "double";
    default:
      return "cudf::type_id(" +
          std::to_string(static_cast<int32_t>(type)) + ")";
  }
}

cudf::type_id veloxPrimitiveTypeToCudfTypeId(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return cudf::type_id::BOOL8;
    case TypeKind::TINYINT:
      return cudf::type_id::INT8;
    case TypeKind::SMALLINT:
      return cudf::type_id::INT16;
    case TypeKind::INTEGER:
      VELOX_CHECK(
          !type->isDate() && !type->isIntervalYearMonth(),
          "Unsupported logical type for native GPU simple function: {}",
          type->toString());
      return cudf::type_id::INT32;
    case TypeKind::BIGINT:
      VELOX_CHECK(
          !type->isDecimal() && !type->isIntervalDayTime(),
          "Unsupported logical type for native GPU simple function: {}",
          type->toString());
      return cudf::type_id::INT64;
    case TypeKind::REAL:
      return cudf::type_id::FLOAT32;
    case TypeKind::DOUBLE:
      return cudf::type_id::FLOAT64;
    default:
      VELOX_FAIL(
          "Unsupported type for native GPU simple function: {}",
          type->toString());
  }
}

exec::FunctionSignaturePtr makeSignature(
    cudf::type_id returnType,
    const std::vector<cudf::type_id>& argTypes) {
  exec::FunctionSignatureBuilder builder;
  builder.returnType(typeName(returnType));
  for (auto argType : argTypes) {
    builder.argumentType(typeName(argType));
  }
  return builder.build();
}

std::vector<std::string> cudfNamesForGpuName(const std::string& name) {
  if (name == "ceil") {
    return {"ceil", "ceiling"};
  }
  if (name == "modulus") {
    return {"modulus", "mod"};
  }
  if (name == "power") {
    return {"power", "pow"};
  }
  if (name == "bitwise_arithmetic_shift_right") {
    return {"bitwise_arithmetic_shift_right", "bitwise_right_shift_arithmetic"};
  }
  return {name};
}

bool sameSignature(
    const GpuSimpleCudfVariant& variant,
    cudf::type_id returnType,
    const std::vector<cudf::type_id>& argTypes) {
  return variant.returnType == returnType && variant.argTypes == argTypes;
}

const GpuSimpleCudfVariant* findMatchingVariant(
    const std::shared_ptr<velox::exec::Expr>& expr,
    const std::vector<GpuSimpleCudfVariant>& variants) {
  const auto returnType = veloxPrimitiveTypeToCudfTypeId(expr->type());
  const auto& inputs = expr->inputs();

  for (const auto& variant : variants) {
    if (variant.returnType != returnType ||
        variant.argTypes.size() != inputs.size()) {
      continue;
    }

    bool matches = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (veloxPrimitiveTypeToCudfTypeId(inputs[i]->type()) !=
          variant.argTypes[i]) {
        matches = false;
        break;
      }
    }

    if (matches) {
      return &variant;
    }
  }

  return nullptr;
}

class RegisteredGpuSimpleCudfFunction final : public CudfFunction {
 public:
  RegisteredGpuSimpleCudfFunction(
      std::string registeredName,
      const std::shared_ptr<velox::exec::Expr>& expr,
      const GpuSimpleCudfVariant& variant)
      : registeredName_(std::move(registeredName)),
        gpuName_(variant.gpuName),
        returnType_(variant.returnType),
        argTypes_(variant.argTypes),
        function_(variant.factory()) {
    VELOX_CHECK_EQ(
        expr->inputs().size(),
        argTypes_.size(),
        "{} expects {} inputs",
        registeredName_,
        argTypes_.size());
    VELOX_CHECK_NOT_NULL(function_);

    const auto actualReturnType = veloxPrimitiveTypeToCudfTypeId(expr->type());
    VELOX_CHECK(
        actualReturnType == returnType_,
        "Function {} expected return type {}, got {}",
        registeredName_,
        typeName(returnType_),
        typeName(actualReturnType));

    size_t nextRuntimeInput = 0;
    for (size_t i = 0; i < expr->inputs().size(); ++i) {
      const auto& input = expr->inputs()[i];
      const auto actualArgType = veloxPrimitiveTypeToCudfTypeId(input->type());
      VELOX_CHECK(
          actualArgType == argTypes_[i],
          "Function {} expected argument {} type {}, got {}",
          registeredName_,
          i,
          typeName(argTypes_[i]),
          typeName(actualArgType));

      if (std::dynamic_pointer_cast<velox::exec::ConstantExpr>(input)) {
        auto literalIndex = literalScalars_.size();
        literalScalars_.push_back(makeScalarFromConstantExpr(input));
        argBindings_.push_back({ArgumentKind::kLiteral, literalIndex});
      } else {
        argBindings_.push_back(
            {ArgumentKind::kRuntimeColumn, nextRuntimeInput++});
      }
    }
    runtimeInputCount_ = nextRuntimeInput;
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(
        inputColumns.size(),
        runtimeInputCount_,
        "Unexpected number of runtime inputs for {}",
        registeredName_);

    bool hasRuntimeRows = false;
    cudf::size_type numRows = 0;
    for (auto& inputColumn : inputColumns) {
      auto view = asView(inputColumn);
      if (!hasRuntimeRows) {
        numRows = view.size();
        hasRuntimeRows = true;
      } else {
        VELOX_CHECK_EQ(
            view.size(),
            numRows,
            "Mismatched input sizes for {}",
            registeredName_);
      }
    }

    VELOX_CHECK(
        hasRuntimeRows,
        "Native GPU simple FunctionExpression cannot evaluate literal-only "
        "call {} because CudfFunction::eval does not receive the input row "
        "count",
        registeredName_);

    std::vector<cudf::column_view> fullInputs;
    fullInputs.reserve(argBindings_.size());
    std::vector<std::unique_ptr<cudf::column>> literalColumns;
    literalColumns.reserve(literalScalars_.size());

    for (const auto& binding : argBindings_) {
      if (binding.kind == ArgumentKind::kRuntimeColumn) {
        fullInputs.push_back(asView(inputColumns[binding.index]));
      } else {
        auto literalColumn = cudf::make_column_from_scalar(
            *literalScalars_[binding.index], numRows, stream, mr);
        fullInputs.push_back(literalColumn->view());
        literalColumns.push_back(std::move(literalColumn));
      }
    }

    return function_->apply(fullInputs, numRows, nullptr, stream, mr);
  }

 private:
  std::string registeredName_;
  std::string gpuName_;
  cudf::type_id returnType_;
  std::vector<cudf::type_id> argTypes_;
  std::vector<ArgumentBinding> argBindings_;
  std::vector<std::unique_ptr<cudf::scalar>> literalScalars_;
  size_t runtimeInputCount_{0};
  std::unique_ptr<gpu::GpuVectorFunction> function_;
};

void publishCudfFunction(
    const std::string& registeredName,
    const std::vector<GpuSimpleCudfVariant>& variants) {
  std::vector<exec::FunctionSignaturePtr> signatures;
  signatures.reserve(variants.size());
  for (const auto& variant : variants) {
    signatures.push_back(variant.signature);
  }

  registerCudfFunction(
      registeredName,
      [registeredName, variants](
          const std::string&,
          const std::shared_ptr<velox::exec::Expr>& expr) {
        auto* variant = findMatchingVariant(expr, variants);
        VELOX_CHECK_NOT_NULL(
            variant,
            "No native GPU simple FunctionExpression variant registered for {}",
            expr->toString());
        return std::make_shared<RegisteredGpuSimpleCudfFunction>(
            registeredName, expr, *variant);
      },
      signatures);
}

void registerVariantForCudfName(
    const std::string& registeredName,
    const std::string& gpuName,
    cudf::type_id returnType,
    const std::vector<cudf::type_id>& argTypes,
    const GpuSimpleCudfFunctionFactory& factory) {
  auto& state = registrationState();
  auto& variants = state.variants[registeredName];
  if (std::none_of(variants.begin(), variants.end(), [&](const auto& variant) {
        return sameSignature(variant, returnType, argTypes);
      })) {
    variants.push_back(GpuSimpleCudfVariant{
        gpuName,
        returnType,
        argTypes,
        factory,
        makeSignature(returnType, argTypes)});
  }

  publishCudfFunction(registeredName, variants);
}

} // namespace

void registerGpuSimpleCudfFunction(
    const std::string& name,
    cudf::type_id returnType,
    std::vector<cudf::type_id> argTypes,
    GpuSimpleCudfFunctionFactory factory) {
  std::lock_guard<std::mutex> lock(registrationMutex());
  auto& state = registrationState();
  if (!state.enabled) {
    return;
  }

  for (const auto& cudfName : cudfNamesForGpuName(name)) {
    registerVariantForCudfName(
        state.prefix + cudfName, name, returnType, argTypes, factory);
  }
}

void registerPrestoGpuSimpleCudfFunctions(const std::string& prefix) {
  {
    std::lock_guard<std::mutex> lock(registrationMutex());
    auto& state = registrationState();
    state.enabled = true;
    state.prefix = prefix;
  }

  gpu::registerPrestoArithmetic();
  gpu::registerPrestoComparisons();
  gpu::registerPrestoBitwise();

  {
    std::lock_guard<std::mutex> lock(registrationMutex());
    registrationState().enabled = false;
  }
}

} // namespace facebook::velox::cudf_velox
