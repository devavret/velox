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

#include "velox/core/Expressions.h"
#include "velox/expression/ExprConstants.h"

#include <optional>
#include <unordered_set>

namespace facebook::velox::cudf_velox {

inline std::string exprName(const core::TypedExprPtr& expr) {
  VELOX_CHECK_NOT_NULL(expr);

  if (expr->isCallKind()) {
    return expr->asUnchecked<core::CallTypedExpr>()->name();
  }
  if (expr->isCastKind()) {
    return expr->asUnchecked<core::CastTypedExpr>()->isTryCast()
        ? expression::kTryCast
        : expression::kCast;
  }
  if (expr->isConstantKind()) {
    return "literal";
  }
  if (expr->isFieldAccessKind()) {
    return expr->asUnchecked<core::FieldAccessTypedExpr>()->name();
  }
  if (expr->isDereferenceKind()) {
    return expr->asUnchecked<core::DereferenceTypedExpr>()->name();
  }
  if (expr->isConcatKind()) {
    return expression::kRowConstructor;
  }

  return expr->toString();
}

inline bool isLiteralExpr(const core::TypedExprPtr& expr) {
  return expr != nullptr && expr->isConstantKind();
}

inline std::optional<std::vector<std::string>> extractFieldPath(
    const core::TypedExprPtr& expr) {
  if (expr == nullptr) {
    return std::nullopt;
  }

  if (expr->isFieldAccessKind()) {
    const auto* field = expr->asUnchecked<core::FieldAccessTypedExpr>();
    if (field->inputs().empty() || field->inputs()[0]->isInputKind()) {
      return std::vector<std::string>{field->name()};
    }

    auto path = extractFieldPath(field->inputs()[0]);
    if (!path.has_value()) {
      return std::nullopt;
    }
    path->push_back(field->name());
    return path;
  }

  if (expr->isDereferenceKind()) {
    const auto* dereference = expr->asUnchecked<core::DereferenceTypedExpr>();
    auto path = extractFieldPath(dereference->inputs()[0]);
    if (!path.has_value()) {
      return std::nullopt;
    }
    path->push_back(dereference->name());
    return path;
  }

  return std::nullopt;
}

inline std::optional<std::string> rootFieldName(const core::TypedExprPtr& expr) {
  auto path = extractFieldPath(expr);
  if (!path.has_value() || path->empty()) {
    return std::nullopt;
  }
  return path->front();
}

inline std::optional<std::string> leafFieldName(const core::TypedExprPtr& expr) {
  auto path = extractFieldPath(expr);
  if (!path.has_value() || path->empty()) {
    return std::nullopt;
  }
  return path->back();
}

inline std::vector<int> nestedFieldIndices(
    const core::TypedExprPtr& expr,
    const RowTypePtr& inputType) {
  auto path = extractFieldPath(expr);
  if (!path.has_value() || path->size() < 2) {
    return {};
  }

  VELOX_CHECK_NOT_NULL(inputType);
  VELOX_CHECK(
      inputType->containsChild(path->front()),
      "Field {} not found in input schema {}",
      path->front(),
      inputType->toString());

  TypePtr current = inputType->findChild(path->front());
  std::vector<int> indices;
  indices.reserve(path->size() - 1);

  for (size_t i = 1; i < path->size(); ++i) {
    VELOX_CHECK(
        current->isRow(),
        "Field {} is not a row type in path {}",
        path->at(i - 1),
        expr->toString());
    const auto& row = current->asRow();
    VELOX_CHECK(
        row.containsChild(path->at(i)),
        "Nested field {} not found in {}",
        path->at(i),
        row.toString());
    const auto index = row.getChildIdx(path->at(i));
    indices.push_back(index);
    current = row.childAt(index);
  }

  return indices;
}

inline bool isInputFieldReference(const core::TypedExprPtr& expr) {
  return rootFieldName(expr).has_value();
}

inline void collectReferencedInputFields(
    const core::TypedExprPtr& expr,
    std::unordered_set<std::string>& fields,
    const std::unordered_set<std::string>& lambdaInputs = {}) {
  if (expr == nullptr) {
    return;
  }

  if (auto root = rootFieldName(expr);
      root.has_value() && !lambdaInputs.count(*root)) {
    fields.insert(*root);
  }

  if (expr->isLambdaKind()) {
    const auto* lambda = expr->asUnchecked<core::LambdaTypedExpr>();
    auto scopedLambdaInputs = lambdaInputs;
    for (const auto& name : lambda->signature()->names()) {
      scopedLambdaInputs.insert(name);
    }
    collectReferencedInputFields(lambda->body(), fields, scopedLambdaInputs);
    return;
  }

  for (const auto& input : expr->inputs()) {
    collectReferencedInputFields(input, fields, lambdaInputs);
  }
}

inline std::unordered_set<std::string> referencedInputFields(
    const core::TypedExprPtr& expr) {
  std::unordered_set<std::string> fields;
  collectReferencedInputFields(expr, fields);
  return fields;
}

} // namespace facebook::velox::cudf_velox
