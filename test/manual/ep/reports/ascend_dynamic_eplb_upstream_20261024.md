# Ascend dynamic EPLB: validation scope

Status as of 2026-10-26, for code and tests at
`4fa073523c5f1113375fac08242cbe99571f1cf8`.
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

| Check | Status for the revision above |
| --- | --- |
| Source and call-chain review | Completed; this is not execution validation. |
| Formatting and static checks during cleanup | Completed; no functional result is implied. |
| CPU unit tests after the latest code and fixture changes | Not run. |
| NPU graph execution and HCCL weight migration after cleanup | Not run. |
| Full-model correctness, PD acceptance, and performance after cleanup | Not run. |

Existing tests cover route counting, padding, overflow, capture/replay state,
draft isolation, communication-group selection, and weight registration.
The manual migration and online-loop scripts remain available. Their presence
does not establish that this revision passes them. No tests were added or
executed for this documentation cleanup.

## Historical evidence

The [archived report and its linked data](https://github.com/AkiYangDev/sglang/blob/4fa073523c5f1113375fac08242cbe99571f1cf8/test/manual/ep/reports/ascend_dynamic_eplb_upstream_20261024.md)
retain earlier source identities, CPU/NPU results, performance tables,
screenshots, and failure records. The report's references to "current source"
describe those historical revisions, not this PR head.

Those runs preceded subsequent cleanup and removal of the initial learning
policy. They are not validation of the revision above, and no throughput gain
or full-model accuracy result is claimed for the current implementation.
