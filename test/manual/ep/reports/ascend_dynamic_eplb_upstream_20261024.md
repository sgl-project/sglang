# Ascend dynamic EPLB: validation scope

Status as of 2026-10-26, after isolating NPU adapters from CUDA execution.
Community base: `de7055f59645ebe03784b0980e1f222789f468a3`.

## Implementation scope

The PR adapts EPLB to Ascend sender-route statistics, a dedicated HCCL
communication group, and ND staging for internal-format expert weights.
It preserves upstream rebalance scheduling; the earlier automatic 64-step
initial learning policy is not included.

Six production files differ from the base: three NPU files and the shared
recorder factory, weight updater, and ModelRunner entry points. The shared
EPLB manager and decode graph runner are unchanged from the base.

## Validation status

| Check | Status after GPU-path isolation |
| --- | --- |
| Source and call-chain review | Completed; this is not execution validation. |
| Formatting and static checks during cleanup | Completed; no functional result is implied. |
| CPU unit tests and GPU regression after GPU-path isolation | Not run. |
| NPU graph execution and HCCL weight migration after cleanup | Not run. |
| Full-model correctness, PD acceptance, and performance after cleanup | Not run. |

The added unit tests and manual probes have been moved out of this PR and
remain available in the [historical test tree](https://github.com/AkiYangDev/sglang/tree/46393a1ce8359fc226704702b14a157dfa77c7ee/test).
Only a one-line fixture adjustment remains in the existing NPU graph-backend
test: its manually registered graph also receives an empty EPLB state.
No new test cases are included, and no tests were executed for this cleanup.

## Historical evidence

The [archived report and its linked data](https://github.com/AkiYangDev/sglang/blob/4fa073523c5f1113375fac08242cbe99571f1cf8/test/manual/ep/reports/ascend_dynamic_eplb_upstream_20261024.md)
retain earlier source identities, CPU/NPU results, performance tables,
screenshots, and failure records. The report's references to "current source"
describe those historical revisions, not this PR head.

Those runs preceded subsequent cleanup, GPU-path isolation, and removal of the
initial learning policy. They are not validation of this implementation, and
no throughput gain or full-model accuracy result is claimed for it.
