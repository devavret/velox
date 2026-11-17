# SF1000 host-spill measurements on one L4

All 22 queries have complete series of six executions that match the independent CPU reference, including result ordering. Spilling was enabled for every query. Q9, Q13, Q18 and Q22 actually spilled in the measured runs; the other queries fit without spilling. No disk-spill bytes or files were recorded.

The **21 encoded-host-input queries sum to 156.758195s** in warm medians. **Q9 is a separate Parquet-input case: 29.461726s**. Their mixed-input sum is 186.219921s; it is not an all-cached-input result or an end-to-end elapsed time.

## Source and method

- Based on [PR10](https://github.com/GregoryKimball/velox/pull/10) at `7de0eaf6dc3d74ab047bfad2bb9f2ea9d73a1b44`; query-builder C++ and header blobs are unchanged.
- Measured engine commit: `4744dde01acff543929a17f03ea4b582f8158798`. These measurements predate the compact decimal SUM state addition and do not measure its performance. To reproduce this state layout with the newer benchmark, pass `--cudf_compact_decimal_sum_state=false`.
- Binary SHA256: `dbc8b680f1700de42948141dbe222e72f2218eea92a7858299ac490af100f20b`.
- cuDF: `02f215d2bb7a4840ebbd5b81532ae717abea5054`; CUDA 12.9.86, GCC 14.2.1, driver 595.112.07, existing `adapters-cuda` service.
- One L4 on g6.8xlarge, 32 logical CPUs. Observed VRAM 23,034 MiB; Linux MemTotal 130,088,366,080 bytes (121.15 GiB). The container digest is in the adjacent JSON.
- Same 90 SF1000 decimal Parquet files throughout; all archived [SHA256 checks](sf1000-data-sha256.txt) matched. No data regeneration or reordering was done for this measurement.
- Each query executes six times in one process. Repeat 0 is excluded; the median of repeats 1–5 is reported. All 132 accepted executions passed CPU-reference checks. Decimal comparison is exact; floating tolerances remain 1e-12 relative/1e-10 absolute.
- The timer includes query-context creation, execution, native CPU result collection and task completion. Plan construction, encoded input preparation and writing result/statistics files are outside it. Q9 includes Parquet reading/decoding; warm runs can benefit from the OS file cache.
- No competing build, test, profile or archive jobs ran during measurements. CPU fallback was disabled.

## Results

Peak and transfer columns below are the maximum across the five warm executions. Peak live spill storage and cumulative bytes written are different quantities. GPU peaks measure accounted query allocations, not total device residency.

| Query | Warm median (s) | Input | Peak host spill (GiB) | Written to host (GiB) |
|---|---:|---|---:|---:|
| Q1 | 8.989224 | Encoded host | 0.000 | 0.000 |
| Q2 | 1.516593 | Encoded host | 0.000 | 0.000 |
| Q3 | 4.935493 | Encoded host | 0.000 | 0.000 |
| Q4 | 3.972410 | Encoded host | 0.000 | 0.000 |
| Q5 | 5.599893 | Encoded host | 0.000 | 0.000 |
| Q6 | 3.043723 | Encoded host | 0.000 | 0.000 |
| Q7 | 6.294618 | Encoded host | 0.000 | 0.000 |
| Q8 | 7.146741 | Encoded host | 0.000 | 0.000 |
| Q9 | 29.461726 | Parquet | 15.064 | 35.337 |
| Q10 | 7.900577 | Encoded host | 0.000 | 0.000 |
| Q11 | 1.333816 | Encoded host | 0.000 | 0.000 |
| Q12 | 5.029136 | Encoded host | 0.000 | 0.000 |
| Q13 | 16.279508 | Encoded host | 3.112 | 4.540 |
| Q14 | 4.345147 | Encoded host | 0.000 | 0.000 |
| Q15 | 10.133148 | Encoded host | 0.000 | 0.000 |
| Q16 | 1.791635 | Encoded host | 0.000 | 0.000 |
| Q17 | 3.939471 | Encoded host | 0.000 | 0.000 |
| Q18 | 35.471268 | Encoded host | 57.360 | 57.546 |
| Q19 | 7.169373 | Encoded host | 0.000 | 0.000 |
| Q20 | 6.243198 | Encoded host | 0.000 | 0.000 |
| Q21 | 10.303311 | Encoded host | 0.000 | 0.000 |
| Q22 | 5.319912 | Encoded host | 5.588 | 17.430 |

The [JSON](host_spill_sf1000_l4.json) contains every warm sample, GPU/spill peaks, transfer counters and exact per-query settings.

## Configuration

Defaults: 4 drivers, 16 GiB accounted GPU budget, 99 GiB combined host budget, 16 GiB host-spill allowance. Use [l4-default.properties](l4-default.properties): 16M target rows, independent 16M aggregation spill-leaf limit, no global concatenation row cap. All sizes below are GiB.

| Query | Drivers | GPU budget | Combined host budget | Spill allowance | Other change |
|---|---:|---:|---:|---:|---|
| Q9 | 1 | 18 | 99 | 64 | Parquet input; 512 MiB chunk/1 GiB pass limits |
| Q10 | 1 | 20 | 99 | 16 | — |
| Q13 | 1 | 16 | 99 | 16 | — |
| Q18 | 4 | 18 | 108 | 64 | — |
| Q22 | 4 | 20 | 99 | 16 | 8 GiB join-partition estimate limit |

Q22 uses [l4-q22.properties](l4-q22.properties). Q9 reads the same Parquet files from NVMe and still spills only to host RAM. Its encoded cache alone required 105,190,945,388 bytes; the attempted host-cache run exhausted physical RAM during execution. It is deliberately reported separately.

Example for a default-settings query (adjust the paths and binary build directory):

```bash
./_build_docker/release/velox/experimental/cudf/benchmarks/velox_cudf_tpch_benchmark \
  --data_path=/path/to/tpch_sf1000 --data_format=parquet --scale_factor=1000 \
  --encoded --encoded_memory=pinned_host --encoded_load=query --input_overflow=fail \
  --gpu_budget_bytes=17179869184 --host_budget_bytes=106300440576 \
  --spill_host_bytes=17179869184 --num_drivers=4 --num_repeats=6 \
  --cudf_properties=velox/experimental/cudf/benchmarks/results/l4-default.properties \
  --cudf_local_exchange_buffer_size=268435456 \
  --results_json_dir=/path/to/results --include_custom_stats --run_queries=1
```

For Q9 omit the three encoded-input options, set the budgets/driver count from the table, and add `--cudf_chunk_read_limit=536870912 --cudf_pass_read_limit=1073741824`. The JSON configuration records all flags.

## Q13 recovery and validation

Q13’s first 18 GiB-budget series returned five correct results, then failed its sixth execution with a physical CUDA allocation error. Logical allocations returned to zero, but the CUDA pool retained a growing physical footprint; a leak was not established. The reported Q13 result is an entirely new series of six executions at 16 GiB, with the same source, binary, plans, input, driver count and batch size. The other 21 query series are unchanged. No minimum was selected between successful series; the failed Q13 series is excluded and retained in the experiment archive.

Targeted enabled tests passed: 594 core-memory, 47 task-lifetime, 4 host-budget, 7 pinned-pool, 5 vector, 5 configuration, 6 forced aggregation-spill, 7 forced hash-join-spill, 124 TPC-H, 151 aggregation, 3 GPU-resource and 146 existing hash-join cases (1,099 total). The clean PR10 control also passed its 75 plan tests. This is targeted validation, not the full repository CI matrix.

## Included scope

This branch adds query/operator GPU accounting, bounded pinned-host storage, aggregation and supported hash-join spilling, spill-related execution/lifetime fixes, query-column input preparation and per-repeat telemetry. It includes the actual [reclaimer-factory PR #19156](https://github.com/facebookincubator/velox/pull/19156). Existing plain/FOR/dictionary input encodings are retained. Later plan variants, specialized low-cardinality/string/decimal kernels, new input codecs, bitmap filtering and unrelated exchange/storage functionality are excluded.

Partitioned joins cover supported inner/left/semi/anti cases; build-side-output and null-aware builds still need to fit. Aggregation spilling uses its own leaf-row bound so the global batch cap can remain unset.
