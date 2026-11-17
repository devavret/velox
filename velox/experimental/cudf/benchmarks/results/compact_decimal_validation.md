# Compact decimal SUM state validation

Validated on 2026-11-18 on the existing single-L4 g6.8xlarge, using the `adapters-cuda` service and cuDF `02f215d2bb7a4840ebbd5b81532ae717abea5054`.

Tested engine commit: `9132a4d084de1c45caef0b0fd2022e9f513967df`. This report adds documentation only.

The benchmark enables `--cudf_compact_decimal_sum_state=true` by default when cuDF is enabled and CPU fallback is disabled. Eligible grouped short-decimal SUMs exchange a 16-byte DECIMAL128 state instead of a 32-byte VARBINARY state plus offsets. SQL parameters, grouping keys, join order, final result types, and query-builder source remain unchanged. Long-decimal, masked, distinct, ordered, global, and unsupported plan shapes keep their ordinary states. The optional streaming decimal group-by does not consume this representation; compact aggregations use the existing non-streaming, host-spill-capable path.

## GPU correctness results

| Suite | Passed | Skipped / disabled |
|---|---:|---|
| Compact aggregation node | 11 | None |
| Aggregation selection | 27 | None |
| Decimal aggregation | 65 | None |
| Group-by spilling | 7 | None |
| Aggregation | 151 | 1 pre-existing disabled test |
| TPC-H query builder and run statistics | 129 | 18 compact-only cases without eligible SUMs |
| Total | 390 | 18 skipped; 1 disabled |

All suites exited successfully. The new tests cover NULL groups, empty input, multiple scales, sums exceeding 64-bit range, mixed compact/ordinary aggregates, CPU-fallback rejection, and plan-shape preservation. The host-spill test verifies exact query results, spill/restore row counts, repartitioning, and fewer bytes written than the same query with ordinary states.

On the TPC-H correctness fixture, Q1, Q18, Q20 and Q22 use the compact rewrite and match reference SQL with host spilling enabled. The other 18 queries are unchanged and covered by the existing suites. These are correctness-fixture results, not SF1000 performance measurements.

The benchmark and all six test binaries were built successfully. Building the existing decimal regression suite exposed a stale null-mask helper declaration; it was corrected to `cuda::device_buffer<cudf::bitmask_type>` for the pinned API.

## Performance scope

No SF1000 timings were rerun for this addition. The archived [SF1000 report](host_spill_sf1000_l4.md) predates it. Use `--cudf_compact_decimal_sum_state=false` to select the earlier state representation for a comparison. New benchmark statistics record the setting and the number of compact aggregation nodes per query.
