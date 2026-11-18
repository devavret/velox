#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
data=$(realpath "${1:?SF100 dataset directory}")
results=${2:?New result directory}
test ! -e "$results" || { echo 'Result directory already exists' >&2; exit 2; }
mkdir -p "$results"
results=$(realpath "$results")
cd "$repo"
sha256sum -c repro/sf100-9p71/source-sha256.txt > "$results/source-verification.log"
(cd "$data" && sha256sum -c "$repo/repro/sf100-9p71/data-sha256.txt") > "$results/data-verification.log"
docker compose -f docker-compose.yml -f repro/sf100-9p71/image.yml run --rm -T --no-deps \
  -v "$data":/benchmark-input:ro -v "$results":/benchmark-results \
  -e TPCH_Q11_FRACTION=0.0000010000 adapters-cuda \
  /velox/_build_docker/release/velox/experimental/cudf/benchmarks/velox_cudf_tpch_benchmark \
  --data_path=/benchmark-input \
  --data_format=parquet \
  --num_drivers=1 \
  --encoded \
  --encoded_memory=pinned_host \
  --cudf_properties=/velox/repro/sf100-9p71/optimized.properties \
  --cudf_local_exchange_buffer_size=268435456 \
  --num_repeats=6 \
  --sirius_plans \
  --encoded_scale100_decimals \
  --encoded_lz4_strings \
  --encoded_delta \
  --encoded_dictionary_filters \
  --encoded_plain_filter_views \
  --sirius_prune_scan_outputs \
  --sirius_q5_numeric_group \
  --sirius_q9_partsupp_first \
  --sirius_q10_preaggregate \
  --sirius_q22_window \
  --sirius_q4_inner_distinct \
  --sirius_q13_preaggregate \
  --sirius_filter_only_scans \
  --sirius_q15_window_max \
  --sirius_q17_window \
  --sirius_q18_unscaled_sum \
  --sirius_q21_pairs \
  --include_custom_stats \
  --results_json_dir=/benchmark-results \
  --run_queries=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 2>&1 | tee "$results/benchmark.log"
