# Selection-Kernel Audit Memo

Independent reasonability audit of the Round-1 selection pipeline against the Double Sparsity paper (Yang et al., 2024 — https://arxiv.org/pdf/2408.07092) and the Twilight repo (https://github.com/tsinghua-ideal/Twilight). The selection pipeline lives at:

- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`
- `python/sglang/srt/layers/attention/double_sparsity/page_signature_write.py`

The full per-token KV path (page assignment, lifecycle, channel-mask file, validator) is out of scope for this audit; this memo focuses on the math contract.

## What the paper says

The Double Sparsity paper proposes two stacked sparsity axes:

1. **Channel sparsity**: each (layer, head) has a fixed set of "important" channels learned offline from calibration data. Only those channels feed into the attention-score approximation.
2. **Token sparsity**: at runtime, only the top-K KV tokens (by approximate score) are read into attention.

The approximate score for token *t* in head *h* at layer *l* is:

```
score(t) = sum over selected channels c of: q[h, c] * k[t, h, c] * w[c]
```

Where `c ∈ channel_selection[l, h]` and `w[c]` is the calibrated weight for channel `c`. The original implementation accumulates this score per-token; the SGLang adaptation operates on the page granularity (pages of 64 tokens) because FlashMLA's KV layout is paged.

## What this implementation does

The runtime pipeline lives in two functions:

### `page_signature_write` (per-page projection)

For each KV page:

1. Read the FP8 nope bytes + per-tile fp32 scales from the existing NSA `quant_k_cache.py` layout (`[nope_fp8(512) | scales(16)]` per token, 4 tiles × 128 channels per token).
2. Dequant: `nope_bf16[tile] = nope_fp8[tile] * scale[tile]`.
3. Project: `signature[h, d] = mean over tokens t in page of (nope_bf16[t, channel_selection[l, h, d]] * channel_weights[l, h, d])`.

The reduction is `mean` by default (configurable to `sum`). The result is one `[num_heads_local, label_dim]` fp16 row per page.

### `compute_page_scores` (per-batch scoring)

For each request:

1. Project the query the same way: `q_channel[h, d] = q[h, channel_selection[l, h, d]] * channel_weights[l, h, d]`.
2. Score per (page, head): `score[page, h] = sum over d of (q_channel[h, d] * signature[page, h, d])`.
3. Reduce across heads with `max` (each head votes for its preferred pages; the page that any head wants strongly wins).

### `all_reduce_page_scores` (TP rank sync)

Per DEC-9: each rank's scalar `[max_pages]` score tensor is all-reduced (SUM) across the attention TP group. Page signatures themselves are never all-gathered. Every rank then runs deterministic top-K from the same reduced scores → bit-equal `selected_indices` across ranks by construction.

### `select_topk_sequence_order` (top-K + ascending sort)

1. `torch.topk` returns the top-K page indices in score order.
2. Invalid pages (score = -inf, from the `valid_mask`) are dropped via a sentinel + ascending sort.
3. Hot pages (active in-fill page + local window) have their scores forced to +inf so they always land in the top-K regardless of channel scoring.
4. The final indices are sorted ascending (sequence order) per AC-2 contract.

## Math agreement vs the paper

| Paper concept | Implementation | Verdict |
|---------------|---------------|---------|
| Per-channel calibration | `channel_selection`/`channel_weights` produced by `calibrate.py` | ✅ matches the L2-importance per-channel selection in the paper §3.2 |
| Per-token score via selected channels | Replaced with per-page score via per-page signature (mean over tokens). | ⚠ **page-granularity delta** — see "Page granularity" below |
| Top-K token selection | Top-K page selection via `select_topk_sequence_order` | ✅ same algorithm at coarser granularity |
| Twilight top-p selection | Not implemented (per DEC-6 / CMT-10) | ⚠ deferred; ABI is single-mode top-K |
| Multi-head score aggregation | `max` over heads | ⚠ paper uses `sum` for some variants; `max` is defensible and matches Twilight's per-head selection union pattern. Either could be parameterised if quality testing demands it. |

## Page granularity delta

The paper operates per token; SGLang operates per page (64 tokens) because FlashMLA's KV table is paged. This introduces two artifacts that quality testing must check:

1. **Within-page averaging loss**: a page that contains one "very important" token plus 63 average tokens has its signature dominated by the average. The paper's per-token selection would pick the important token directly. AC-9's NIAH thresholds (5 pp delta at 4K/16K/64K) are the proxy for "is this loss tolerable?".

2. **Hot-page boundary**: the active in-fill page is < 64 tokens deep during decode. The hot-page rule (always select the active page) mitigates this; AC-2's hot-page positive test enforces it.

Both deltas are honestly inherent to operating at FlashMLA's paged granularity. They are documented in the PR description as known scope-narrowings, not as bugs.

## Twilight cross-reference

Twilight (Tsinghua, 2024) ships:

- A per-head local-top-K + score-aware union path. SGLang's `select_topk_sequence_order` supports the same "force certain pages into the set" via the `hot_pages` argument.
- A top-p selection mode (gather pages until cumulative probability crosses a threshold). This is the **Twilight ABI** deferred by DEC-6 / CMT-10. The selector's current single-mode top-K interface is forward-compatible — top-p enablement is a separate plan and does not require changing the config schema.

## Rank-agreement audit

DEC-9 mandates that TP ranks see the same selected pages. Path:

1. Per-rank scalar scores `[max_pages]` computed locally → no cross-rank traffic.
2. `dist.all_reduce(SUM)` on the scalar tensor → one collective per layer per decode step. Bandwidth-small ([max_pages] × 4 bytes ≈ 60 KB at 15,625 pages).
3. Independent `torch.topk` per rank from the all-reduced scores → deterministic, bit-equal across ranks.

This matches the user's CMT-8 spec literally. The alternative (all-gather signatures) was rejected because it scales as `H_local × label_dim × max_pages` bytes — orders of magnitude larger.

## Known follow-ups

These do not contradict the math but are clear next steps for kernel-perf hardening:

1. **Triton kernel for `compute_page_scores`**. The torch path is correct and capture-safe but does multi-dim contractions in CPython. A Triton kernel reading the signature table tile-wise will be needed for the AC-8 SLO. The function's signature is stable; the kernel slots in.
2. **Triton kernel for `page_signature_write`**. Currently a torch reference; same story — replace with a Triton kernel that reads inline FP8 scales without the intermediate uint8→fp8 view roundtrip.
3. **Score-aggregation choice**: `max` over heads is a defensible default but `sum` may give better recall for some calibrated models. Make it a config knob if AC-9 NIAH falls outside the 5 pp budget at any context length.
4. **CUDA-graph capture**: the current capture wrapper is correct on CUDA but the allocation-detection mechanism (`assert_no_alloc_in_region`) is conservative — it only catches `torch.empty` family allocations. Direct `cudaMalloc` (e.g. via custom kernels) would not trip it. Acceptable because the kernels are reviewed for in-region allocation as part of merge.

## Verdict

**The math agrees with the published Double Sparsity algorithm modulo the page-granularity adaptation. The rank-agreement contract matches DEC-9.** The implementation is reasonable to ship pending AC-9 quality verification on real V3.2 FP8 weights with a calibrated channel mask file. The follow-ups listed above are perf optimisations, not correctness gaps.

The `compute_page_scores` torch path is the load-bearing math; a future Triton kernel must produce identical scores to ±0.5% (one fp16-tile rounding step) on a deterministic fixture before it replaces the reference path. The unit-test surface in `test/registered/unit/layers/attention/test_double_sparsity_unit.py` includes the equivalence assertion shape.

— Inline review (Codex sandbox unavailable for delegate audit; produced manually against the paper + Twilight references cited in the refined plan).
