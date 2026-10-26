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
Outside Elastic EP, a rank also skips staging and migration when its local
placement is unchanged and no changed destination requests a local expert.
Communication group initialization still precedes this per-rank early return.
Receive buffers are reused within a chunk and are not cached between rebalances.
ND staging retains each layer's conversion results directly. Writeback converts
them to the destination format before copying into the original model storage.
NPU target forwards skip the draft guard's device lookup, and empty route
tensors return before creating a buffer view or issuing a copy.
Graphs without buffered routes omit inactive capture flags, letting replay skip
the statistics-state restoration hook.

Eight production files differ from the base: four NPU files and the shared
recorder, weight updater, ModelRunner, and prefill backend factory. The shared
EPLB manager and decode graph runner are unchanged from the base.

NPU prefill capture temporarily uses direct route counting, so target graphs
record count updates without depending on replay-time Python batching flags.
The scope includes warmups and restores the previous policy on exceptions;
decode retains buffered counting. Draft prefill capture remains isolated.

## Validation status

| Check | Status after migration refinements |
| --- | --- |
| Source and call-chain review | Completed; this is not execution validation. |
| Formatting and static checks during cleanup | Completed; no functional result is implied. |
| CPU unit suite and GPU regression after migration refinements | Not run. |
| NPU full decode graph regression after cleanup | Not run. |
| HCCL migration with the new staging paths | Passed a 16-rank synthetic migration and reverse migration before the idle-rank guard; values, addresses, formats, ND scales, and unchanged local layouts checked. |
| Full-model correctness, PD acceptance, and performance after cleanup | Not run. |

The idle-rank guard passed an external CPU check against the unchanged shared
migration planner across 18,370 rank cases. Values, addresses, and send/receive
lists matched, including unchanged local senders and bypassing the guard for
Elastic EP. Transport and group creation were mocked; this is not NPU/HCCL or
service validation of the guard.

An isolated 30-round paired check on Ascend 910C with torch_npu 2.10.0.post6
used the observed INT8 shapes `[17,4096,4096]` and `[17,2048,4096]` (408 MiB).
Median device Event time was 2.083 -> 1.457 ms for load and 3.383 -> 2.201 ms
for store. These are staging microbenchmark results, not serving throughput.
Full-size values and destination addresses were checked. A direct preallocated
format-cast overload was slower at these shapes and was not adopted.
Debug workspace bytes describe planned logical tensor storage, excluding
allocator padding/caching and transient format-conversion allocations. Graph-statistics hooks
are wired to the NPU full decode backend; other graph modes are not validated.

CPU regression tests now cover direct prefill counts, restoration of decode
batching, nested/exception scopes, and disabled draft recorders. The existing
NPU backend fixture supplies the model-runner identity and an empty EPLB state.
These tests require the normal SGLang CPU test environment; they were not run
locally because the local Python has no PyTorch installation. A separate
production-method extraction probe with simulated graph operations passed
prefill replay counting, draft isolation, and policy restoration. This is not
an actual NPU graph or full-package integration test.

Earlier migration tests and manual probes remain available in the
[historical test tree](https://github.com/AkiYangDev/sglang/tree/46393a1ce8359fc226704702b14a157dfa77c7ee/test).
The copy probes and 16-rank checks ran outside the PR.

## Historical evidence

The [archived report and its linked data](https://github.com/AkiYangDev/sglang/blob/4fa073523c5f1113375fac08242cbe99571f1cf8/test/manual/ep/reports/ascend_dynamic_eplb_upstream_20261024.md)
retain earlier source identities, CPU/NPU results, performance tables,
screenshots, and failure records. The report's references to "current source"
describe those historical revisions, not this PR head.

Those runs preceded subsequent cleanup, adapter consolidation, and removal of the
initial learning policy. They are not validation of this implementation, and
no throughput gain or full-model accuracy result is claimed for it.
