# Ascend dynamic EPLB: validation scope

Status as of 2026-10-27. Production changes reviewed through `7d5854b5`;
comparison base: `de7055f59645ebe03784b0980e1f222789f468a3`.

## Implementation

This PR enables Ascend EPLB with valid sender-route statistics, a dedicated
WORLD-membership HCCL group, and ND staging for internal-format expert weights.
It retains upstream rebalance scheduling and thresholds; the earlier automatic
64-step initial learning policy is excluded.

- NPU statistics buffer small forwards across layers and count overflow directly.
  Decode capture/replay restores the matching batching state. Prefill capture
  uses direct counts, including warmups, and restores the previous policy on exit.
  Draft forwards/capture are isolated from target counts.
- Migration validates each local chunk before allocating workspaces. Unchanged
  layers are skipped outside Elastic EP. Unchanged local placements still send
  when needed, but skip writeback; ranks with provably no work skip migration.
  Receive buffers are reused within a chunk.
- Load retains the NZ-to-ND conversion result. Store converts to the original
  format and copies into original model storage, preserving graph addresses.
  Only the current layer's conversion results are retained.
- Outside Elastic EP, small two-dimensional BF16/FP32 auxiliary matrices, up to
  64 KiB per expert row, are grouped by dtype per layer. Large weights remain
  separate. Group membership depends on shape/dtype, not rank-local NPU format.
- Non-elastic target recorder initialization prewarms statistics and all-peer
  P2P communication, then synchronizes before releasing the tiny buffers. Reuse
  is cached by group identity. This moves channel setup to startup; it does not
  remove its cost.
- CPU static-dispatch construction converts candidates once and stores only the
  current rank's result. Replica choice order and random-state consumption are
  preserved; this adds no GPU forward work.

Nine production files differ from the comparison base: four NPU files, plus
shared recorder construction/signatures, placement metadata, the weight updater,
ModelRunner's worker-identity argument, and the prefill-backend factory. Shared
EPLBManager and the decode graph runner are unchanged. GPU retains its original
recorder and device migration path; no NPU forward wrapper is installed on GPU.

## Validation

These checks validate specific components, not the current full-model service.
Source-extracted checks do not establish full-package import compatibility.

| Evidence | Scope and result |
| --- | --- |
| Latest source review | Reviewed initialization order, WORLD group caching, sender participation, auxiliary wire grouping, and in-place writeback. Static review only. |
| CPU static-dispatch equivalence | 3,936 real PyTorch CPU cases and 36 output/random-state comparisons passed. Synthetic 43-layer layouts: median function time 116.1 -> 14.1 ms and 109.2 -> 10.1 ms. |
| Communication warmup | Standalone 16-rank Ascend 910C check, torch_npu 2.10.0.post6: payloads and cached reuse passed. Warmup cost 2,736.9 ms; first transfer 2.30 ms versus 2,175.0 ms in an independent cold group. |
| Latest auxiliary coalescing | Source-extracted 16-rank weight update with real NZ weights, sparse moves and rank-dependent auxiliary formats passed value/address/format checks. Per-rank median one-layer time 24.99 -> 23.61 ms; slowest-rank median 26.45 -> 25.02 ms. |
| Staging copy refinement | Earlier isolated 30-round check at INT8 shapes `[17,4096,4096]` and `[17,2048,4096]`: load Event median 2.083 -> 1.457 ms; store 3.383 -> 2.201 ms. Values and addresses checked. |
| Other external checks | Idle-rank guard checked across 18,370 mocked CPU rank cases. Twelve auxiliary-staging CPU cases and a two-rank Gloo call-chain check passed. Prefill counting/draft/exception scopes passed a simulated-graph production-method probe. |
| Formatting | Pinned Ruff 0.15.1 and isort 7.0.0 checks passed for the production revisions above. |

No new test files are included. New prefill regression-test source and diagnostic
probes are kept outside this PR. The existing graph-backend fixture retains only
the required model-runner identity and empty EPLB capture state. The normal CPU
unit suite was not run locally because the local Python has no PyTorch installation.

The current production revision has **not** passed full-model startup with the
new warmup, migration followed by NPU graph replay, GPU regression, or matched
PD throughput/accuracy testing. Explicit prefill graph modes also need device
validation. No serving-throughput or model-accuracy improvement is claimed.
Load/store helper timings now include auxiliary packing/unpacking, so their
envelopes must not be reported as pure format-conversion time. Debug workspace
bytes exclude allocator padding/caching and transient conversion allocations.

## Historical evidence

The [archived report](https://github.com/AkiYangDev/sglang/blob/4fa073523c5f1113375fac08242cbe99571f1cf8/test/manual/ep/reports/ascend_dynamic_eplb_upstream_20261024.md)
and [historical test tree](https://github.com/AkiYangDev/sglang/tree/46393a1ce8359fc226704702b14a157dfa77c7ee/test)
retain older runs and probes. They precede subsequent cleanup and do not validate
this production revision. Full-model acceptance remains pending.
