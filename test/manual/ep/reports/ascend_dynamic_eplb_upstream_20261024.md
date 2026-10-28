# Ascend dynamic EPLB: validation scope

Status as of 2026-10-28. Comparison base:
`de7055f59645ebe03784b0980e1f222789f468a3`.

## Implementation

This PR enables Ascend EPLB with valid sender-route statistics, reuse of the
existing TP device group, and ND staging for internal-format expert weights.
It retains upstream rebalance scheduling and thresholds; the earlier automatic
64-step initial learning policy is excluded.

- TP must cover WORLD in rank order because shared statistics and migration use
  WORLD ranks. TP subgroups are rejected before communication. EPLB inherits
  TP's existing configuration; it does not create another ProcessGroup or
  change TP's communication options.
- NPU recording reuses the upstream direct sender-route counter. Routing inputs
  exclude graph padding and shared slots. NPU recorder and full-graph capture
  isolate draft execution from target statistics without modifying model files.
  Ordinary forward and full-graph capture share one NPU entry guard. ModelRunner
  selects it only on NPU; the GPU forward function and shared recorder lifecycle
  retain their upstream implementations and signatures.
  Generic breakable prefill graph capture remains outside this adapter's coverage.
  There is no batched-route buffer,
  capacity configuration, custom counting kernel, or replay-time statistics state.
- Supported NPU policies aggregate the statistics window before logical mapping
  and communication. File dumps and policies requiring per-step loads retain
  the full window.
- Migration validates each local chunk before allocating workspaces. Unchanged
  layers are skipped outside Elastic EP. Unchanged local placements still send
  when needed, but skip writeback; ranks with provably no work skip migration.
  Receive buffers are allocated on the first participating layer and reused
  within a chunk; ranks with no work allocate none.
- Each weight tensor migrates separately. Load retains NZ-to-ND conversion
  results and packs strided ND tensors into contiguous migration storage.
  Contiguous ND tensors are reused directly. Store writes staged tensors back
  into original model storage, preserving formats, strides and graph addresses.
  Only the current layer's staging results are retained; auxiliary weights are
  not packed together.
- Non-elastic target recorder initialization prewarms statistics and all-peer
  P2P communication, then synchronizes before releasing the tiny buffers. Reuse
  is cached by group identity. This moves channel setup cost to startup.

Six production files differ from the comparison base: three NPU files and three
shared integration files. Shared EPLBManager, placement metadata, model definitions
and the decode graph runner are unchanged. GPU retains its original recorder and
device migration path. No draft-worker parameter is added to the shared recorder.

## Validation

Historical component checks used a dedicated migration group and cover warmup,
internal-format conversion and in-place writeback. They do not validate sharing
TP between forward communication and migration. Results for removed batched
statistics, auxiliary packing and static-dispatch optimizations are excluded
from the current implementation's evidence. Detailed probes and performance
materials remain outside this PR.

No new test files are included. The existing NPU graph-backend fixture supplies
the model-runner draft identity. Local source-extracted checks cover target/draft
statistics, both entry guards, exception-state restoration, and the unchanged GPU
forward implementation. Scalar gatherer and graph-engine stand-ins do not
establish device graph execution or full-package import compatibility.
Targeted CPU checks also cover strided-ND send/receive ordering and original
storage writeback, deferred allocation, unchanged senders and chunk reuse.
The wire-order check follows the inspected HCCL pointer/count contract; it is
not a hardware HCCL execution result.

Before TP reuse, the production sources passed real-package imports, controlled
NPU mixed-format multi-layer migration with communicating graph replay, and a short
non-speculative PD ON startup/rebalance smoke check followed by decode graph
replay. These are bounded correctness checks, not full model-accuracy validation.
The shared-TP path and refactored entry guard still require complete service
validation. Generic prefill/draft graph coverage, real GPU
regression, and matched PD throughput/accuracy validation remain pending. No serving-throughput or
model-accuracy improvement is claimed. Detailed evidence remains outside the PR.
