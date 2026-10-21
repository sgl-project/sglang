# V4.1 CP encoder replay: source and runtime acceptance

This extends upstream folded encoder replay to CUDA SM90 interleave Prefill CP.
It reuses the community request window, DSpark draft ring, window-to-paged PD
handoff, HiCache components, and Mooncake backend. CP1 retains upstream model
dispatch and GEMM/MoE backends. This self-contained series also includes two
common H20 repairs used for both CP1 and CP8: a stable
Hopper BF16x3 mHC pipeline configuration and deterministic TopK cutoff ties.
It does not add wide-precision projection or blanket fusion/reduction overrides.

## Support envelope

- DeepSeek V4.1, CUDA SM90; initial runtime target is H20.
- Prefill: interleave CP with attention TP=1; PP=DP=DCP=1. The initial H20
  topology is TP8/EP8/CP8. CP size derives from the resolved parallel config.
- Prefill CUDA graphs disabled. No CP-TP group sharing, cache layer split,
  logical-page KV sharding, mixed chunks, sessions, LoRA, or multimodal requests.
  A checkpoint with vision layers requires `--language-model-only`; image
  requests are rejected by the tokenizer entry point before reaching workers.
- PD uses Mooncake and DSpark static verify with matching block size and KV
  layouts. Decode remains CP1 and does not enable encoder replay. Enable decoder
  SWA bounded replay explicitly when exercising the compact-tail path; its
  auto-default can be disabled by upstream under CP.
- HiCache uses `cache` host memory mode; optional L3 uses Mooncake Store.
  `--hicache-storage-backend mooncake` and
  `--disaggregation-transfer-backend mooncake` select separate facilities.
  The direct external L3 linker is outside this envelope. Runtime storage
  attachment also enforces the Mooncake backend for CP replay pools.

This source envelope is not an H20 acceptance result. Other GPU architectures,
parallel layouts, and backend performance are not certified by CPU tests.

## Execution and ownership

1. Reuse upstream folded extend and its replay floor, keep rows and write mask.
   Replay rows rebuild request-local SWA; padding commits nowhere. Compact tails
   retain their original CP owners and global output-token indices.
2. Window writes/history remain global while query metadata is CP-local.
   Existing KV gathers materialize the supported replicated window and low-ratio
   source state. The native Torch indexer receives local per-request row counts.
3. DSpark hidden/aux gathers temporarily use tail-specific CP metadata and restore
   the full metadata even on failure. The upstream fold drop removes replay rows
   before draft injection. Keep upstream ring `valid_from` semantics.
   Only FULL hidden capture carries the compact row map; LAST capture is already
   one row per request. A model-side output hook exposes this map only for CP
   encoder replay. Encoder-off and CP1 callers retain their original output layout.
4. Decode derives the required Prefill fan-in from the remote CP layout. Existing
   partitioning gives each KV/indexer page one sender; replicated state uses the
   existing CP0 handoff. Prefill advertises `cp_encoder_replay_version=1` inside
   the DSpark layout only for this new mode. New Decode accepts upstream legacy
   layouts with their original fan-in; legacy Decode rejects the new mode during
   bootstrap. Unversioned experimental CP replay builds are not compatible peers:
   upgrade both P/D roles together rather than mixing earlier PR images with v1.
   Unknown versions and other KV/layout mismatches are rejected before transfer.
   Validate committed logical target-window history once, using the pool's window
   size, before publishing metadata or paged SWA indices. This check synchronizes
   GPU tags and a CP CPU vote. A missing window on any rank concludes the room as
   Failed on every rank; existing inflight failure handling releases KV/metadata.
5. Replayed request-owned KV is not deduplicated into an unadopted snapshot,
   including Host-only nodes. Existing request locks and transfer lifetimes own
   cleanup. Layer loading events remain the dependency for KV/indexer readers;
   no additional model-wide device barrier is added.
6. CP>1 Mooncake objects use a versioned producer/layout/topology namespace.
   Supported persistent pools are replicated: CP0 writes, and every rank restores
   from the same component identity. Request windows, draft rings and temporary
   hidden are not additional L3 objects. CP1 cache keys stay unchanged; cross-CP
   cache sharing is not assumed.
   The effective namespace and producer recipe summary are logged. Dependency or
   numerical recipe changes can require a new namespace and cache warmup; old
   objects remain until Store eviction. The fingerprint does not include actual
   FlashInfer autotune tactics: pin those tactics for strict comparisons and
   change the operator's `extra_backend_tag` if they change. Do not assume sharing
   between different recipes during rolling upgrades.
7. Storage probes intersect actual restorable endpoint sets. The scheduler and
   prefetch thread reuse the same algorithm on their respective groups. Every
   rank follows the same protocol on hit, miss, cancellation, or query failure.

Replayed requests with an overlapping unadopted radix snapshot skip that
checkpoint's insertion instead of rebinding live KV. This incurs an additional
prefix lookup and CPU vote per affected checkpoint and can reduce suffix reuse
while overlap persists. The guard is CP-only; it does not establish a CP1 bug.

## CPU checks

With the repository test dependencies installed, from the checkout root:

```bash
python -m pytest -q test/registered/unit/model_executor/test_dsv41_cp_replay_*.py
```

The files also have individual CI registrations and direct pytest entrypoints.
The tests exercise production tensor/control-flow definitions with explicit CPU
fixtures instead of importing CUDA-bound model modules. Gloo tests use two local
processes and separate scheduler/prefetch groups. They do not validate compiled
kernels, CUDA streams, RDMA, full-model numerics, or serving performance.

## H20 acceptance gates (separate, not run by CPU checks)

Freeze source/image/dependencies, model, effective argv, inputs and sampling.
Preserve the existing strict token/logprob/top-2 equality rules; do not replace
them with tolerance checks.

1. Check CP1 dispatch and paired runs against the same community base. Record
   inherited community A/A drift separately from newly introduced differences.
   Preserving native CP1 math does not guarantee the base is deterministic.
   Test startup refusal for vision+CP, image rejection in text-only mode and a
   subsequent valid request. Cover new/new and mixed-version P/D bootstrap, plus
   an injected missing-window handoff with failure notification and cleanup.
2. Compare community and candidate at the same CP size when that community
   configuration is available. Independently check repeated cold requests and
   L1 replay at each CP using 257/4097/8193-token inputs with 256 outputs.
   Cross-CP bitwise equality is not required: native CP1 and CP8 may use
   different numerical schedules. Inspect Prefill selection, PD metadata, and
   the first DSpark
   verify block before expanding. Include mixed hits/misses and noncontinuous
   request slots; add a long chunk-boundary case where needed.
3. Force L2 restoration of a representative request and prove actual object
   restores, layer readiness and valid windows. Compare against the L1 path.
   Hold the adopted prefix endpoint and replay floor fixed: different endpoints
   invoke different bounded-history computations. Use a fixed Prefill topology
   per cache lifetime; mixed CP producers sharing Decode prefix state are outside
   this acceptance envelope.
4. Force Mooncake L3 restoration with the target objects absent from L1/L2,
   without a global Store purge. Prove backend reads, every-rank availability,
   restored bytes/state and output equality against L1. Counters alone are not
   sufficient evidence. Do not use a cache flush as proof of L2-only eviction.
5. Measure cold and hit performance separately with the same frozen workload.
   Native CP1 and CP8 need fresh evidence for this revision; old candidate
   timings/quality are not acceptance for a branch with different numerics.
6. Assess bounded replay quality separately, reporting paired accuracy difference
   and 95% interval without inventing a business threshold. Cold and hit outputs
   need not be identical. Same-CP repetition and same-CP L1/L2/L3 restoration
   must meet strict equality for the same adopted prefix and replay floor.
   Report unavailable community configurations explicitly; do not substitute
   a candidate-only control for a community comparison.

On a difference, stop the affected lane at its earliest divergent boundary.
CPU contract success does not close a strict numerical or GPU/RDMA acceptance gap.

## Portable validation attachment

The accompanying validation package accepts an explicit JSON configuration for
endpoints, Kubernetes context/namespace, source heads, immutable image digests,
Pod/node identities and result roots. The strict finite-bit comparator and the
idle-only cache-control hook are unchanged from the H20 acceptance clients.
Normal requests never install or activate the cache controls automatically.
See the attachment README for cold/L1, forced L2 and fresh-namespace L3 commands.

The rebased source and common H20 repairs need a matching rebuilt image before
claiming new GPU acceptance. H20 results for candidate `6674e87cfb` remain historical.
The quality plan uses official EvalScope scoring, frozen full GSM8K and a
preselected complete-question LongBench-v2 subset, with paired accuracy differences
and 95% intervals. Cold and replay quality are evaluated separately; no business
threshold is invented.
