# CuDF Benchmarks

Benchmark binaries for TPC-H and TPC-DS queries with optional CuDF GPU acceleration.

## Binaries

| Binary | Benchmark | Mode | Source |
|--------|-----------|------|--------|
| `velox_tpch_benchmark` | TPC-H (Q1-Q22) | CPU | `velox/benchmarks/tpch/` |
| `velox_cudf_tpch_benchmark` | TPC-H (Q1-Q22) | GPU | `CudfTpchBenchmark.cpp` |
| `velox_tpcds_benchmark` | TPC-DS (Q1-Q99) | CPU | `velox/benchmarks/tpcds/` |
| `velox_cudf_tpcds_benchmark` | TPC-DS (Q1-Q99) | GPU | `CudfTpcdsBenchmark.cpp` |

CPU binaries use HiveConnector. GPU binaries use CudfHiveConnector and register
cuDF GPU operator replacements.

## Build

```bash
# GPU binaries (requires CUDA)
CUDA_ARCHITECTURES="native" EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_BENCHMARKS=ON" make cudf
cd _build/release
ninja velox_cudf_tpch_benchmark velox_cudf_tpcds_benchmark

# CPU-only binaries (no CUDA required)
ninja velox_tpch_benchmark velox_tpcds_benchmark
```

---

## TPC-H Benchmark

### Data

Generate TPC-H parquet data using the [velox-testing](https://github.com/rapidsai/velox-testing) data generation tool (see [Data Generation](#data-generation) below), or the standard `dbgen` tool and convert decimal columns to float.

`velox_cudf_tpch_benchmark` reads decimal TPC-H data instead: DECIMAL(15, 2)
amounts and DATE dates, as `tpchgen-cli` and DuckDB write them
(`prepare_tpch_encoded.py` below).

### Plans

`velox_cudf_tpch_benchmark` runs the plans of `CudfTpchQueryBuilder`, which use
one fixed set of query parameters, listed in `CudfTpchQueryBuilder.h`, and
several rewrites of the SQL described there. `--scale_factor` gives the scale
factor of the data, which sets Q11's FRACTION, `0.0001 / SF`.

### Run

```bash
# CPU - all queries
./velox_tpch_benchmark --data_path=/path/to/tpch/sf100 --data_format=parquet

# GPU (CuDF) - all queries
./velox_cudf_tpch_benchmark --data_path=/path/to/tpch/sf100 --data_format=parquet \
  --scale_factor=100
```

Without `--run_query_verbose` or `--run_queries`, `velox_cudf_tpch_benchmark`
runs each query `--num_repeats` times and prints the wall time of each run.
`--results_json_dir` writes each query's result rows to `<dir>/q<N>.json`.

### Encoded Column Cache

`--encoded` loads the tables into the in-memory encoded column cache and scans
them from there instead of Parquet (`--encoded_memory=pinned_host` keeps them in
pinned host memory). The cache encodes rows in file order, in segments of
`--encoded_segment_rows` rows, and prunes segments by min/max, so sort the files
beforehand: `prepare_tpch_encoded.py` writes decimal TPC-H with lineitem sorted
by `l_shipdate` and orders by `o_orderdate`. The same files give the same cache.

```bash
pip install tpchgen-cli duckdb pyarrow
python3 prepare_tpch_encoded.py --scale-factor 100 --out /path/to/tpch_sf100

echo "cudf.batch_size_min_threshold=16777216" > cudf.properties
./velox_cudf_tpch_benchmark \
  --data_path=/path/to/tpch_sf100 --data_format=parquet --scale_factor=100 \
  --num_drivers=4 \
  --encoded --encoded_memory=pinned_host --cudf_properties=cudf.properties \
  --num_repeats=4 --run_queries=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
```

With `--encoded_load=query` (the default), the cache retains only the columns
needed by the next query. Columns shared with the previous query are reused;
other columns are released. `--encoded_load=all` retains whole tables instead.
Preparation loads and encodes rows in file order. It does not filter or aggregate
rows ahead of the timed query.

### Bounded GPU memory and host spilling

The benchmark gives every execution its own GPU query pool. Set
`--gpu_budget_bytes` to limit accounted query allocations and
`--spill_host_bytes` to enable spilling into a bounded pinned-host pool.
There is no disk spill destination.

For host-resident encoded input, `--host_budget_bytes` reserves a fixed share
for spilling and gives the remainder to the input cache:

```
input cache limit = host_budget_bytes - spill_host_bytes
```

This is an allocation budget for cached input and spilled state, not a limit on
whole-process RSS. Leave room for other CPU allocations, CUDA bookkeeping and
pinned-pool allocation rounding. The GPU budget similarly excludes CUDA runtime
overhead and device allocations made while preparing the input cache.

For example, the following runs Q1 on SF1000 data with a 16 GiB GPU budget,
16 GiB host-spill budget and 99 GiB combined input/spill budget:

```bash
cat > host-spill.properties <<'EOF'
cudf.memory_resource=async
cudf.allow_cpu_fallback=false
cudf.batch_size_min_threshold=16777216
cudf.spill_groupby_max_rows_per_leaf=16777216
EOF

./velox_cudf_tpch_benchmark \
  --data_path=/path/to/tpch_sf1000 --data_format=parquet --scale_factor=1000 \
  --encoded --encoded_memory=pinned_host --encoded_load=query \
  --gpu_budget_bytes=17179869184 --host_budget_bytes=106300440576 \
  --spill_host_bytes=17179869184 --input_overflow=fail \
  --cudf_properties=host-spill.properties --num_drivers=4 \
  --num_repeats=6 --run_queries=1 --results_json_dir=/path/to/results
```

The aggregation spill-leaf bound is independent of the global batch maximum.
Leave `cudf.batch_size_max_threshold` unset for these runs: setting it also splits
join builds into multiple tables, which can add probe work and is unsupported by
some join types. Encoded scans still cap decoding at 16M rows by default.
`cudf.spill_groupby_max_rows_per_leaf` keeps aggregation leaves bounded without
changing join builds.

By default the benchmark gives each grouped `SUM` over a short decimal column a
compact intermediate state: one `DECIMAL(38, scale)` column of 16 bytes for each
group, in place of the 32-byte `VARBINARY` state and its 4-byte offsets. That
is what such a `SUM` exchanges between its partial and final aggregation, and
what the final aggregation buffers and spills. The query builder's plans are
not changed: the benchmark applies `CudfAggregationNode::rewritePlan` to the
plan of each query, which keeps the nodes, their ids, the SQL parameters and the
result types. A `SUM` is eligible only if it reads a short decimal column
without a mask, `DISTINCT` or `ORDER BY`, in a grouped partial and final
aggregation; every other aggregate keeps its ordinary state. The compact state
is exact: fewer than 2^63 values below 10^18 in magnitude cannot overflow 128
bits.

Only cuDF operators read and write this state, and a query fails instead of
running such an aggregation on the CPU. The benchmark therefore applies it only
with `cudf.allow_cpu_fallback=false`. `--cudf_compact_decimal_sum_state=false`
turns it off. Each run prints the number of aggregations that use the state and
records it as `compact_decimal_sum_aggregations` in `q<Q>.stats.json`, next to
the `compact_decimal_sum_state` setting.

The appropriate input/spill split depends on the query. `--input_overflow=fail`
reports input that cannot fit instead of changing its source. The optional
`--input_overflow=parquet` explicitly falls back to reading input Parquet files;
the log and per-repeat metadata identify that input mode. This concerns input
loading only and does not enable disk spilling.

Each repeat writes `repeat<N>/q<Q>.json` and `repeat<N>/q<Q>.stats.json`.
Statistics include operator counters, GPU pool peaks, spill-pool peaks and the
input mode. The first repeat is available as a warmup; reporting scripts should
exclude it when computing warm-query medians. The timer includes query-context
creation, execution, result materialization and task completion. Input-cache
preparation and writing JSON files are outside the timer. A failed repeat is
recorded and makes the benchmark exit unsuccessfully.

Use a fresh process when changing GPU or host-spill limits. The process GPU
resource is initialized once; query pools share its capacity.

---

## TPC-DS Benchmark

TPC-DS plans are loaded from pre-dumped Velox plan JSON files (serialized from Presto).

### 1. Get Plan JSON Files

Clone the plans repository:

```bash
git clone https://github.com/karthikeyann/VeloxPlans.git
# Plans are at: VeloxPlans/presto/tpcds/sf100/
```

The directory contains `Q1.json`, `Q2.json`, ..., `Q99.json`.

### 2. Get TPC-DS Data

Generate TPC-DS parquet data using the [velox-testing](https://github.com/rapidsai/velox-testing) data generation tool (see [Data Generation](#data-generation) below). The data directory must have one subdirectory per table:

```
/path/to/tpcds/sf100/
  store_sales/
  customer/
  date_dim/
  item/
  ...
```

Each subdirectory contains parquet files for that table.

### 3. Run

**CPU - all queries (folly benchmark mode):**

```bash
./velox_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet
```

**CPU - single query with stats:**

```bash
./velox_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet \
  --run_query_verbose=1
```

**GPU (CuDF) - all queries:**

```bash
./velox_cudf_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet
```

**GPU (CuDF) - single query with stats:**

```bash
./velox_cudf_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet \
  --run_query_verbose=1
```

### TPC-DS Flags

These flags are shared by both CPU and GPU binaries:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | (required) | Root directory of TPC-DS table data |
| `--plan_path` | (required) | Directory containing Q*.json plan files |
| `--data_format` | `parquet` | Data file format |
| `--run_query_verbose` | `-1` | Run single query with stats (`-1` = run all) |
| `--num_drivers` | `4` | Number of parallel drivers |
| `--include_results` | `false` | Print query results |

### CuDF Flags (GPU binaries only)

These flags apply to `velox_cudf_tpch_benchmark` and `velox_cudf_tpcds_benchmark`:

| Flag | Default | Description |
|------|---------|-------------|
| `--cudf_chunk_read_limit` | `0` | Chunk read limit for cuDF parquet reader |
| `--cudf_pass_read_limit` | `0` | Pass read limit for cuDF parquet reader |
| `--cudf_gpu_batch_size_rows` | `100000` | GPU batch size in rows |
| `--velox_cudf_table_scan` | `true` | Use CuDF table scan |
| `--cudf_properties` | `""` | Path to a CudfConfig properties file (key=value per line). See `CudfConfig.h` for available keys |

---

## Data Generation

Both TPC-H and TPC-DS parquet data can be generated using the
[velox-testing](https://github.com/rapidsai/velox-testing) repository.
Full instructions are also available in the
[VeloxPlans TPC-DS README](https://github.com/karthikeyann/VeloxPlans/tree/main/presto/tpcds/sf100).

### Quick Start

```bash
# 1. Clone velox-testing
git clone https://github.com/rapidsai/velox-testing.git
cd velox-testing

# 2. Install Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r benchmark_data_tools/requirements.txt

# 3. Generate TPC-DS data (sf100)
python benchmark_data_tools/generate_data_files.py \
  --benchmark-type tpcds \
  --data-dir-path /path/to/tpcds/sf100/data \
  --scale-factor 100 \
  --convert-decimals-to-floats

# 4. Generate TPC-H data (sf100)
python benchmark_data_tools/generate_data_files.py \
  --benchmark-type tpch \
  --data-dir-path /path/to/tpch/sf100/data \
  --scale-factor 100 \
  --convert-decimals-to-floats
```

### Key Flags

| Flag | Description |
|------|-------------|
| `--benchmark-type` | `tpcds` or `tpch` |
| `--data-dir-path` | Output directory for parquet files |
| `--scale-factor` | Scale factor (e.g. `1`, `10`, `100`) |
| `--convert-decimals-to-floats` | Convert decimal columns to double (recommended for Velox) |

The output directory will contain one subdirectory per table, each with `.parquet` files.
For a quick sanity check, use `--scale-factor 1` first.
