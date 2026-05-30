# DS on DeepSeek-V3.2: Recall R&D Gate

## Decision

Pursue DS long-context-recall R&D on DeepSeek-V3.2, but only after the Loop-6 engineering spine has landed and passed validation:

1. Reduce TokenLabelTable footprint.
2. Lift admitted KV capacity without generation-time OOM.
3. Re-run the client SLO benchmark with admission-wait vs prefill-compute attribution.
4. Harden the selected footprint path.

The selected recall-R&D direction is a custom sparse-matmul DECODE kernel that mirrors the native NSA/DSA sparse decode path but exposes an adjustable `top_k`, removing the current hard cap that requires `indices.shape[-1] == dsa_index_topk`. A learned or query-aware DS selector is the secondary alternative.

DSA remains the production default. The DS compact-table path remains opt-in and reversible.

## Rationale

The current V3.2 DS recall failure is not evidence that sparse decode is mathematically broken. It is evidence that DS is selecting the wrong 2048 positions.

Evidence:

- V3.2’s sparse decode budget is kernel-locked to the model-native DSA `index_topk = 2048`.
- The shared `flashmla_kv` decode path asserts `indices.shape[-1] == self.dsa_index_topk` during the decode path, including CUDA graph capture. `SGLANG_DS_ALLOW_TOPK_MISMATCH=1` does not bypass that kernel contract.
- DS NIAH recall at `top_k=2048` is poor: 4K = 75%, 16K = 5%, 64K = 0%.
- DSA recall is 100% at every tested length with the same 2048 budget and same decode kernel.
- Dense DS for `seq <= 2048` recalls 100%, which proves the DS decode path can return the needle when selection includes it.

Therefore the DS-vs-DSA recall gap is selection quality against V3.2’s trained DSA indexer, not raw sparse attention budget alone.

## Why Adjustable `top_k` Needs a Kernel

Increasing DS `top_k` above 2048 is not a config-only change on V3.2. The current `flashmla_kv` decode kernel and its metadata path are shaped around the model-native DSA top-k. The backend asserts the final indices dimension equals `dsa_index_topk`; CUDA graph capture, metadata sizing, scheduling, and kernel assumptions all depend on that shape.

A valid top-k relaxation therefore requires a new decode kernel path, or a kernel variant, with explicit support for adjustable top-k and corresponding performance validation. The correct R&D target is a custom sparse-matmul DECODE kernel that preserves the native sparse path’s serving semantics while making top-k a real runtime/configurable parameter.

The learned/query-aware selector alternative is still credible because DSA achieves 100% recall at the same 2048 budget. If DS can learn or infer a better selector, it may close recall without widening top-k. It is secondary because it adds data/training/integration complexity and does not remove the kernel cap.

## Sequencing Consequence

Recall R&D is gated behind this decision and behind the landed engineering spine. It must not block, destabilize, or regress the footprint-reduction, memory-lift, client-SLO, or hardening work.

That sequencing is intentional. Loop 6 exists because DS currently misses P99 TTFT through admission and queueing, not because per-request generation speed is below the TPS SLO. Long-context recall R&D is legitimate, but it belongs in its own loop after DS can admit the target concurrency with HBM headroom.

## Non-V3.2 Note

On a model without a trained sparse indexer, DS has a stronger value proposition. The V3.2 comparison is unusually harsh because DSA already has a trained selector that places the needle inside the same 2048 decode budget. This matters for deferred GLM-5.1 / 128k work, where DS may be more attractive if no equivalent native trained sparse indexer exists.
