# Ascend dynamic EPLB: validation scope

Status as of 2026-10-26, after NPU migration validation and copy refinements.
Community base: `de7055f59645ebe03784b0980e1f222789f468a3`.

## Implementation scope

The PR adapts EPLB to Ascend sender-route statistics, a dedicated HCCL
communication group, and ND staging for internal-format expert weights.
It preserves upstream rebalance scheduling; the earlier automatic 64-step
initial learning policy is not included.

Before migrating a chunk, NPU validates corresponding tensor counts, shapes,
dtypes, devices, and staging requirements. Ordinary tensors bypass ND staging;
unchanged local placements still participate in sends but skip ND writeback.
Receive buffers are reused within a chunk and are not cached between rebalances.
ND staging retains each layer's conversion results directly. Writeback converts
them to the destination format before copying into the original model storage.
NPU target forwards skip the draft guard's device lookup, and empty route
tensors return before creating a buffer view or issuing a copy.
Graphs without buffered routes omit inactive capture flags, letting replay skip
the statistics-state restoration hook.

Six production files differ from the base: three NPU files and the shared
recorder factory, weight updater, and ModelRunner entry points. The shared
EPLB manager and decode graph runner are unchanged from the base.

## Validation status

| Check | Status after migration refinements |
| --- | --- |
| Source and call-chain review | Completed; this is not execution validation. |
| Formatting and static checks during cleanup | Completed; no functional result is implied. |
| CPU unit tests and GPU regression after migration refinements | Not run. |
| NPU full decode graph regression after cleanup | Not run. |
| HCCL migration with the new staging paths | Passed a 16-rank synthetic migration and reverse migration; full tensor values, original addresses, formats, ND scales, and unchanged local layouts checked. |
| Full-model correctness, PD acceptance, and performance after cleanup | Not run. |

An isolated 30-round paired check on Ascend 910C with torch_npu 2.10.0.post6
used the observed INT8 shapes `[17,4096,4096]` and `[17,2048,4096]` (408 MiB).
Median device Event time was 2.083 -> 1.457 ms for load and 3.383 -> 2.201 ms
for store. These are staging microbenchmark results, not serving throughput.
Full-size values and destination addresses were checked. A direct preallocated
format-cast overload was slower at these shapes and was not adopted.
Debug workspace bytes describe planned logical tensor storage, excluding
allocator padding/caching and transient format-conversion allocations. Graph-statistics hooks
are wired to the NPU full decode backend; other graph modes are not validated.

The added unit tests and manual probes have been moved out of this PR and
remain available in the [historical test tree](https://github.com/AkiYangDev/sglang/tree/46393a1ce8359fc226704702b14a157dfa77c7ee/test).
Only a one-line fixture adjustment remains in the existing NPU graph-backend
test: its manually registered graph also receives an empty EPLB state.
No new test files are included; the copy probes and 16-rank checks ran outside
the PR. Earlier cleanup steps did not execute their archived tests.

## Historical evidence

The [archived report and its linked data](https://github.com/AkiYangDev/sglang/blob/4fa073523c5f1113375fac08242cbe99571f1cf8/test/manual/ep/reports/ascend_dynamic_eplb_upstream_20261024.md)
retain earlier source identities, CPU/NPU results, performance tables,
screenshots, and failure records. The report's references to "current source"
describe those historical revisions, not this PR head.

Those runs preceded subsequent cleanup, adapter consolidation, and removal of the
initial learning policy. They are not validation of this implementation, and
no throughput gain or full-model accuracy result is claimed for it.
