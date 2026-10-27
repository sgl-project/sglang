# Ascend dynamic EPLB: validation scope

Status as of 2026-10-27. Comparison base:
`de7055f59645ebe03784b0980e1f222789f468a3`.

## Implementation

This PR enables Ascend EPLB with valid sender-route statistics, a dedicated
WORLD-membership HCCL group, and ND staging for internal-format expert weights.
It retains upstream rebalance scheduling and thresholds; the earlier automatic
64-step initial learning policy is excluded.

- NPU recording reuses the upstream direct sender-route counter. Routing inputs
  exclude graph padding and shared slots. Draft forwards/capture are isolated
  from target counts. There is no batched-route buffer, capacity configuration,
  custom counting kernel, or replay-time statistics state.
- Supported NPU policies aggregate the statistics window before logical mapping
  and communication. File dumps and policies requiring per-step loads retain
  the full window.
- Migration validates each local chunk before allocating workspaces. Unchanged
  layers are skipped outside Elastic EP. Unchanged local placements still send
  when needed, but skip writeback; ranks with provably no work skip migration.
  Receive buffers are reused within a chunk.
- Each weight tensor migrates separately. Load retains the NZ-to-ND conversion
  result. Store converts to the original format and copies into original model
  storage, preserving graph addresses. Only the current layer's conversion
  results are retained; auxiliary weights are not packed together.
- Non-elastic target recorder initialization prewarms statistics and all-peer
  P2P communication, then synchronizes before releasing the tiny buffers. Reuse
  is cached by group identity. This moves channel setup cost to startup.

Eight production files differ from the comparison base: four NPU files and four
shared integration files. Shared EPLBManager, placement metadata and the decode
graph runner are unchanged. GPU retains its original recorder and device
migration path; no NPU forward wrapper is installed on GPU.

## Validation

Historical component checks cover dedicated communication warmup, internal-format
weight conversion and in-place writeback. These precede the latest simplification
and do not validate the current full-model service. Results for removed batched
statistics, auxiliary packing and static-dispatch optimizations are excluded
from the current implementation's evidence. Detailed probes and performance
materials remain outside this PR.

No new test files are included. The existing graph-backend fixture retains the
required model-runner identity; the removed statistics capture state is no longer
needed. Local source-extracted checks with stand-ins do not establish torch/NPU
execution or full-package import compatibility.

The current production revision still needs full-model startup, migration followed
by NPU graph replay, explicit prefill graph modes, GPU regression, and matched
PD throughput/accuracy validation. No serving-throughput or model-accuracy
improvement is claimed.
