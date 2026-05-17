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

#include <cstdint>

namespace facebook::velox::cudf_velox {

enum class SpillHostMemoryKind;

/// Registers benchmark-only adapters that replace the default cuDF hash join
/// with CudfPartitionedHashJoinBuild / CudfPartitionedHashJoinProbe.
///
/// Must be called after registerCudf() so the default cuDF hash-join adapters
/// are present and can be overwritten.
void registerPartitionedHashJoinAdapter(
    int32_t numPartitions,
    SpillHostMemoryKind spillHostMemoryKind);

} // namespace facebook::velox::cudf_velox
