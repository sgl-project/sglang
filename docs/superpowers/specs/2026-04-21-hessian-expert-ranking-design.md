# Hessian-based Expert Ranking — Design

**Date:** 2026-04-21
**Author:** huanchen@usc.edu
**Scope:** `expert_precision_assignment/hessian/` — a new per-expert sensitivity scorer that uses the second-order Taylor term `½·dᵀHd` to rank experts for BF16-vs-INT4 assignment.

## Motivation

The current ranking in `expert_precision_assignment/policy/heter_assign/assign_experts.py` combines two signals:

```
score(L, E) = max(0, ppl_increase[L]) × l2[L][E] / Σ_E l2[L]
```

- `ppl_increase[L]` (from `sensitivity/per_moe_layer/`) is a **global** metric — full-model WikiText PPL delta when all experts in layer L are quantized.
- `l2[L][E]` (from `sensitivity/per_expert/`) is a **local** metric — `‖MoE_out(quant E) − MoE_out(baseline)‖₂` measured at the MoE block only.

Observed failure: **increasing the number of BF16 experts does not monotonically improve accuracy.** The hybrid score assumes each expert's contribution to the layer's end-to-end PPL delta is proportional to its local L2 share within the layer. That assumption is wrong — different experts in the same layer route different tokens, which then traverse different downstream paths with different Jacobian amplification. The local L2 can't see this.

## Approach: per-expert second-order Taylor term

For a quantization perturbation `d_E = W_E^BF16 − dequant(W_E^INT4)`, the end-to-end loss perturbation is:

```
ΔL_E ≈ (∂L/∂W_E)ᵀ d_E  +  ½ · d_Eᵀ · H_{W_E} · d_E
       └── first-order ──┘   └─── second-order ──┘
            ≈ 0 at minimum        dominant term
```

`H_{W_E}` is the Hessian of the loss w.r.t. `W_E`, computed via standard autograd double-backward. The chain rule baked into backprop propagates the perturbation through all downstream layers to the loss — no local approximation. This yields a per-expert scalar in units of loss, directly **cross-layer comparable**.

### Computation: Hessian-vector product (HVP)

To compute `d_Eᵀ · H_{W_E} · d_E` without materializing the Hessian:

1. Forward → NLL loss
2. `grads = autograd.grad(loss, expert_params, create_graph=True)` — first-order
3. `gd = Σ (gᵢ · dᵢ)` — scalar
4. `Hd = autograd.grad(gd, expert_params)` — HVP via double-backward
5. Per-expert: `score[L,E] = ½ · (d_E · Hd_E).sum()`

Also record:
- `first_order[L,E] = (d_E · g_E).sum()` — sanity check, should be ≈ 0 at trained minimum
- `d_norm_sq[L,E] = ‖d_E‖²` — quantization perturbation magnitude, for diagnostics

## Hardware and execution plan

**Target box:** 8× A100 80 GB (DGX A100). All GPUs idle during probe run.

**Distributed strategy:** Data parallelism across calibration samples (`DP=8`). Each rank:
- Holds a full BF16 model replica (loaded direct to GPU, no CPU staging)
- Processes its subset of calibration samples (samples split round-robin by rank)
- AllReduces per-expert scalars at the end (trivial — 6144 floats)

Layer-splitting does **not** parallelize end-to-end Hessian computation, because the backward chain must traverse all layers. DP across samples does, because `E_x[dᵀHd]` is a linear expectation.

**CPU-RAM discipline:**
- Model loaded via `from_pretrained(device_map=f"cuda:{rank}")` — bypasses CPU staging.
- INT4 checkpoints streamed via `safetensors.safe_open(device=f"cuda:{rank}")` — direct to GPU, reusing existing `_load_int4_dequantized` helper from `sensitivity/per_expert/sensitivity.py`.
- No intermediate `.cpu()` calls in the hot path.
- Expected CPU RAM per rank: ~200 MB (tokenizer + datasets lib overhead, fixed regardless of `N`).

**GPU memory discipline (layer chunking):**

Per-rank peak memory grows with the number of expert params holding gradients. For all 6144 experts: ~58 GB of gradient storage would push peak over 80 GB. Solution: process one layer-chunk at a time with only that chunk's expert params unfrozen.

Per-chunk working set (2-layer chunks):
| Item | Size |
|---|---|
| Model weights (resident) | ~60 GB |
| Chunk `d` tensors | ~2.4 GB |
| First-order grads (chunk) | ~2.4 GB |
| HVP output `Hd` (chunk) | ~2.4 GB |
| Activations + 2nd-order graph | ~8 GB |
| **Peak** | **~75 GB** |

Falls back to 1-layer chunks if OOM. Chunk size is a CLI flag.

## Probe configuration

Minimum viable probe to validate plumbing and produce a first comparison:

| Param | Value |
|---|---|
| `--nsamples` | 8 (1 sample per rank) |
| `--seqlen` | 512 |
| `--chunk_size` | 2 (layers per HVP) |
| Calibration data | WikiText-2 train (matches existing per_expert/per_moe_layer) |
| Seed | 42 |

**Wall-time estimate:** ~3–5 min end-to-end.

**Caveat logged in output:** N=8 is a Monte-Carlo estimator with ~0.35σ noise floor on any single expert's score. Sufficient to validate the pipeline and detect large disagreements with the hybrid score; insufficient for final threshold tuning. Scaling up requires only changing `--nsamples` (no code or memory changes) — see scaling roadmap below.

## Scaling roadmap (post-probe)

VRAM is independent of `N` (samples are iterated, scalars accumulated on-GPU). CPU RAM is independent of `N`. Wall-time scales linearly.

| Stage | Config | Wall time | Purpose |
|---|---|---|---|
| Probe | N=8 × 512 WikiText | ~3–5 min | Plumbing validation |
| Statistical | N=128 × 512 WikiText | ~15–25 min | Conclusive comparison vs. hybrid |
| Workload-real | N=128 × 512 ShareGPT/GSM8K | ~15–25 min | Production ranking |
| Final | N=128 × 2048 ShareGPT | ~45–60 min | Matches per-expert calibration |

Calibration data source is a CLI flag — no re-architecting between stages.

## File layout

```
expert_precision_assignment/hessian/
├── hessian_score.py   # HVP-based scorer, torchrun-compatible
├── compare.py         # Ranking comparison vs. hybrid score
├── run.sh             # torchrun launcher
└── results/
    ├── hessian_scores.json
    └── comparison.json
```

## Output schemas

`results/hessian_scores.json`:
```json
{
  "model": "...", "int4_checkpoint": "...",
  "nsamples": 8, "seqlen": 512, "chunk_size": 2, "world_size": 8,
  "avg_loss": 2.31,
  "per_layer": {
    "0": {"experts": {"0": {"hessian_score": <float>, "first_order_score": <float>, "d_norm_sq": <float>}, ...}},
    ...
  }
}
```

`results/comparison.json`:
```json
{
  "num_experts_compared": 6144,
  "spearman_corr": <float>,
  "top_k_overlap": {"0.10": <float>, "0.25": <float>, "0.50": <float>},
  "biggest_disagreements": [{"layer": L, "expert": E, "hybrid_score": ..., "hessian_score": ...,
                              "hybrid_rank": ..., "hessian_rank": ..., "rank_delta": ...}, ...],
  "note": "N=8 is a probe; disagreements may reflect MC noise — re-run with N≥64 for conclusive results."
}
```

## Non-goals for this iteration

- Integrating the Hessian score into `assign_experts.py` (separate follow-up once the probe + statistical runs confirm the approach).
- Full trace of `H` via Hutchinson (the deterministic `dᵀHd` needs no stochastic estimator).
- Cross-layer Hessian coupling `H_{W_E, W_{E'}}` (block-diagonal approximation — ignoring cross-block terms — suffices per the cross-expert independence assumption in the current hybrid score).
- Calibration on workload-specific data (supported via flag, but defaults to WikiText for probe comparability).
