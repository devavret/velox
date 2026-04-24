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

#include <cudf/types.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace facebook::velox::gpu {
class GpuVectorFunction;
} // namespace facebook::velox::gpu

namespace facebook::velox::cudf_velox {

using GpuSimpleCudfFunctionFactory =
    std::function<std::unique_ptr<gpu::GpuVectorFunction>()>;

// Called by gpu::registerGpuFunction while Presto GPU simple functions are
// being registered through registerPrestoGpuSimpleCudfFunctions().
void registerGpuSimpleCudfFunction(
    const std::string& name,
    cudf::type_id returnType,
    std::vector<cudf::type_id> argTypes,
    GpuSimpleCudfFunctionFactory factory) __attribute__((weak));

// Registers the native GPU simple functions compiled in GpuPrestoFunctions.cu.
// Each registerGpuFunction call registers both the native GPU function and the
// matching CudfFunction variant consumable by FunctionExpression.
void registerPrestoGpuSimpleCudfFunctions(const std::string& prefix);

} // namespace facebook::velox::cudf_velox
