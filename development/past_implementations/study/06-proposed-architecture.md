# 06 — Proposed Standalone Double Sparsity for DeepSeek-V3.2 (FP8)

**Status:** as-built design synthesized from `development/loop{1,2,3}/` + the in-tree DS package at `python/sglang/srt/layers/attention/double_sparsity/`. This document captures what we are building and — most importantly — **the performance knobs the previous sglang DS (commit `29f56cb2304`, removed by #23009) could not support, and how this design enables them.**

**Audience:** anyone reviewing the design or picking up the standalone-DS branch (`dev/double-sparsity-standalone`).

---

## 1. TL;DR

Replace the per-step token-selection role of DeepSeek-V3.2's NSA `Indexer` (`python/sglang/srt/layers/attention/nsa/nsa_indexer.py`) with an **offline-calibrated channel-importance projection + runtime page-signature top-K**, hooked at one named edit site in `DeepseekV2AttentionMLA._select_topk_indices`. Everything downstream stays untouched: paged FlashMLA, FP8 KV cache, NSA's quant/dequant kernels, the rest of the model code.

The design is deliberately **standalone**: no HiSparse coupling, no PD-disagg coupling, no monkey-patching, no model fork.

Two artifacts ship per deployment:
1. **Channel mask file** — `safetensors`, offline, once per model revision, content-hash + startup sanity probe.
2. **Page signature table** — GPU tensor, allocator-owned metadata next to the KV page table, written on page assign by an FP8-scale-aware Triton kernel, invalidated on free/evict/abort.

The proposed pipeline is in [`diagrams/03-proposed-architecture.mmd`](diagrams/03-proposed-architecture.mmd).

---

## 2. Performance gaps in the previous sglang DS (commit `29f56cb2304`)

Verbatim from the read of `development/past_implementations/sglang-last-with-double-sparsity/python/sglang/srt/`:

| # | Knob | sglang-last status | Why it didn't work | Evidence |
|---|------|--------------------|--------------------|----------|
| **G1** | **CUDA graphs** | **Force-disabled** at model-runner startup | `topk` runs on the host; routing branch reads `.item()` to decide dense vs sparse | `model_runner.py:927-932` `disable_cuda_graph = True`; `triton_ops/double_sparsity_attention.py:759` `torch.topk(att_out_approx, ...)` returns on host; backend.py:211-214 routes on `min_seq_len < heavy_token_num` (host ints) |
| **G2** | **Attention backend** | **Triton-only**, force-set | DS pipeline assumes raw paged-Triton attention; no FlashAttention / FlashInfer / FlashMLA path | `model_runner.py:931` `attention_backend = "triton"` |
| **G3** | **MLA / DeepSeek-V3.2** | **Not supported** | Selection assumes `model.layers.<i>.self_attn.{q,k}_proj` naming + standard Q,K,V; no MLA absorb-prepare hook | Channel-config schema in `model_runner.py:2156-2171`; selection forward in `double_sparsity_backend.py:128-256` operates on raw Q/K |
| **G4** | **FP8 KV cache** | **Not supported** | `K_label` produced via `torch.gather` on a BF16 K tensor; no FP8 scale-aware projection | `double_sparsity_backend.py:128-134, 196-202` |
| **G5** | **Radix cache** | **Implicitly broken under DS** | Label cache lives in a *separate* `label_buffer` indexed by physical token slots; radix prefix-match works on token positions but DS would re-extract from the K side every layer, never gets a cache hit | `memory_pool.py:1972-2060` `DoubleSparseTokenToKVPool`'s triple-buffer (`k`, `v`, `label`) — no prefix-stable signature artifact |
| **G6** | **Page size 64** | **Per-token, not per-page** | Whole pipeline operates on token-flat `req_to_token` + `[num_head, batch, seq_len]` attention; no page-table emission, no FlashMLA `block_table` adapter | `triton_ops/double_sparsity_attention.py:329-398` (`_sparse_fwd_kernel_flash_decode_stage1` indexes per token) |
| **G7** | **TP rank synchronization** | **Not validated** | Top-K is per-rank from per-rank scores; with TP>1, ranks can pick different tokens — divergent attention output | `triton_ops/double_sparsity_attention.py:759`; no `all_reduce` anywhere in the DS path |
| **G8** | **Mixed batches (DS + dense)** | **Per-batch routing, not per-request** | Whole batch takes either dense or sparse path based on `min`/`max` of `seq_lens` | `double_sparsity_backend.py:211-214` |
| **G9** | **Prefill sparsity** | **Always dense** | Prefill path calls `extend_attention_fwd` regardless | `double_sparsity_backend.py:113-165` |
| **G10** | **`heavy_token_num` adaptivity** | **Static per-launch CLI flag** | Number of selected tokens cannot vary by request/length | `server_args.py:597` `--ds-heavy-token-num 256` |
| **G11** | **CPU offload (paper's DS-Offload)** | **Not ported** | Full K,V always on GPU; no double-buffer prefetch | None in repo |
| **G12** | **Static buffers for graph capture** | **Per-step `torch.empty`** | `att_out_approx`, `mid_out`, `mid_o_logexpsum` reallocated each `init_forward_metadata` | `double_sparsity_backend.py:60-96` |
| **G13** | **Hot-page / fresh-token handling** | **N/A (per-token)** | The token-granularity design doesn't have the problem — but loses it on the way to page granularity unless the page-aware redesign handles it | — |
| **G14** | **Observability** | **None DS-specific** | No `dense_fallback`/`selected_pages`/`channel_mask_valid` exports | No metrics file in DS package |
| **G15** | **Channel config validation** | **Schema-naive JSON** | No content hash, no sanity probe, no dtype/page-size/label-dim guards; LoRA-finetune mismatches go undetected | `model_runner.py:2156-2171` |

**Headline answer to the user's question:**
- Yes, lack of CUDA graphs (G1) was the central perf problem, and it was *caused* by a chain of host-side decisions (G1+G7+G8+G12).
- Radix cache (G5) was never explicitly disabled but DS's per-layer K-side re-extraction sat *outside* the radix-cache prefix contract.
- Beyond those two: no FlashMLA (G2), no MLA model (G3), no FP8 (G4), no paged attention (G6), no TP sync (G7).
- For DeepSeek-V3.2 specifically, **the previous DS code wouldn't even boot** (G2, G3, G4, G6).

---

## 3. Proposed architecture (what closes each gap)

### 3.1 Component map

```
python/sglang/srt/layers/attention/double_sparsity/
├── __init__.py
├── config.py                 # DoubleSparsityConfig dataclass
├── validator.py              # validate_double_sparsity(server_args)
├── channel_mask.py           # safetensors loader + sha256 + sanity probe
├── calibrate.py              # python -m ...calibrate (CLI entrypoint)
├── page_signature_table.py   # allocator-owned table, lifecycle hooks
├── page_signature_write.py   # FP8-scale-aware Triton write kernel
├── selection_kernel.py       # device-side stage-1 + stage-2 top-K
├── selector.py               # DoubleSparsitySelector.retrieve_topk
├── page_table_adapter.py     # logical → physical → FlashMLA block_table
├── cuda_graph.py             # static buffers + assert_no_alloc_in_region
├── metrics.py                # sglang_double_sparsity_* Prometheus surface
└── error_containment.py      # per-request abort, no silent dense fallback
```

Edit site in the model: `python/sglang/srt/models/deepseek_v2.py::DeepseekV2AttentionMLA._select_topk_indices`. One config-gated branch:

```python
if self.use_double_sparsity:
    return self.double_sparsity_selector.retrieve_topk(...)
return self.indexer(...)
```

No monkey-patching, no model class fork, no new attention backend. The downstream FlashMLA call is unchanged.

### 3.2 Selector ABI (locked, AC-11)

```python
DoubleSparsitySelector.retrieve_topk(
    queries, layer_id, req_pool_indices, sparse_mask, seq_lens, hot_pages,
) -> (selected_indices: int32[bs, max_top_k],   # -1 padded, sequence-ascending
      valid_lengths:    int32[bs])
```

The order is **sequence-ascending**, not score-order — FlashMLA's `block_table` wants sequence order. Score order is consumed internally to choose the top-K set; the ABI hides it.

`hot_pages` carries the active in-fill page (plus configurable local window, default 1) → force-included regardless of score. This is the page-granularity replacement for sglang-last's implicit per-token freshness.

### 3.3 Two-artifact design

**Channel mask file** (`safetensors`):
- Tensors: `channel_selection[L, H, label_dim]` (int32), `channel_weights[L, H, label_dim]` (fp32).
- Metadata: `dtype`, `head_dim`, `page_size`, `label_dim`, `created_at`, `schema_version`, `content_sha256`.
- Validated against the running config at boot; sanity probe runs one NIAH-min prompt through DS and verifies the needle. Failure aborts the server.
- Schema-version field plus zero length-dependent fields — forward-compatible with GLM-5.1, 128K ISL, nvfp4/mxfp4 (deferred client asks).

**Page signature table** (GPU tensor):
- Shape per TP rank: `[num_layers_local, max_pages, num_heads_local, label_dim]`, fp16.
- Budget at V3.2 / page=64 / 1M ctx / TP=8 / head-sharded: ~480 MB / rank. Documented; not estimated at boot.
- **Lifecycle is allocator-owned**: KV page allocator gets `on_assign` / `on_free` / `on_evict` / `on_reuse` hooks; the signature table updates `valid_mask[L, max_pages]` in lockstep. Selection ignores invalid pages.
- Hot pages (the active in-fill page) get their signature refreshed every decode step at low cost — one page's K vectors.

### 3.4 How each gap closes

| Gap | This design |
|-----|-------------|
| **G1 CUDA graphs** | Stage-1 score kernel + stage-2 device-side top-K are both Triton, no host roundtrip. Static `selected_indices [max_bs, max_top_k]` (-1 pad) + `valid_lengths [max_bs]` buffers reused across capture/replay (`cuda_graph.py::DSGraphState`). Branching is `tl.where` / mask multiply only. `assert_no_alloc_in_region` is a regression probe. |
| **G2 attention backend** | DS is **not** an attention backend; it's a selector that hands a `block_table` to the existing FlashMLA backend (`flashmla_kv` for FP8, `flashmla_sparse` for BF16). No backend force-set. |
| **G3 MLA / V3.2** | Hook is in `DeepseekV2AttentionMLA._select_topk_indices`. NSA's quant/dequant/MTP/precompute path is untouched. |
| **G4 FP8 KV** | `page_signature_write` Triton kernel reads inline FP8 scales (NSA's `[nope_fp8(512) \| scales(16)]` layout from `quant_k_cache.py`), dequants per-tile to BF16, projects through `channel_selection` / `channel_weights`. RoPE half stays BF16. Validated against an FP32 reference at ±0.5% on a deterministic fixture (per `kernel_audit_memo.md`). |
| **G5 Radix cache** | Two changes make radix cache work under DS: (a) page signatures are **content-derived from K**, not from the selection state — same KV bytes → same signature regardless of which request wrote them; (b) the M3-B page-stability fixture (synthetic CI hook today, hardware run in Phase 5) asserts cold-prefix vs warm-prefix signatures are bit-stable. Until that fixture passes on real hardware, the loop-1 DEC-2 default is **radix cache off under DS** (operator flips after Phase 5). |
| **G6 Page size** | Selection unit is the page (64 by default, 32 also tested). `page_table_adapter` converts logical page IDs → physical via `req_to_token` and emits FlashMLA `block_table` in sequence order. |
| **G7 TP sync** | DEC-9 contract: each rank computes scalar page scores from its local head shard; one `dist.all_reduce(SUM)` on `[max_pages]`-shaped scores (~60 KB at 15,625 pages); each rank runs independent `topk` from the all-reduced scores → bit-equal `selected_indices` by construction. **No signature all-gather** (which would scale as `H_local × label_dim × max_pages` bytes). |
| **G8 Mixed batches** | Per-request page ownership mask `sparse_mask: [bs, max_pages]` built from `req_to_token_pool.req_to_token`, `req_pool_indices`, `seq_lens` — attached to `ForwardBatch.sparse_mask`. `retrieve_topk` consumes it before argmax → picks never escape the request's own KV range. Per-request dense fallback path is available via the `error_containment.py` abort surface, not via batch-wide routing. |
| **G9 Prefill** | Prefill stays dense — but page signatures are *written during prefill* by the KV-write hook, so the first decode step sees a fully populated table. This is the loop-3 M1 work item. |
| **G10 `heavy_token_num` adaptivity** | Out of initial scope. Single-mode top-K (`top_k`, `max_top_k` configurable per server) per AC-11. Adaptive selection (top-p / Twilight) is the documented downstream follow-on. |
| **G11 CPU offload** | Deferred (Phase 4+). Schema and lifecycle hooks are CPU-offload-shaped; not implemented. |
| **G12 Static buffers** | `cuda_graph.allocate_graph_state(max_bs, max_top_k, num_score_blocks, partial_topk)` allocates everything before capture. |
| **G13 Hot-page / fresh tokens** | (a) active-page signature refreshed every decode step; (b) active page (+ configurable local window) force-included via `hot_pages` argument to `retrieve_topk`. Both are explicit AC-2 positive tests. |
| **G14 Observability** | `sglang_double_sparsity_channel_mask_valid` gauge, `sglang_double_sparsity_dense_fallback_total` counter, `sglang_double_sparsity_selected_pages_{sum,count}` histogram. Per-request `meta_info.sparsity_rate / selected_pages / dense_fallback`. |
| **G15 Channel config validation** | `channel_mask.py` validates `dtype`, `head_dim`, `page_size`, `label_dim`, `content_sha256`. Startup sanity probe is a real DS forward against an NIAH-min prompt. Fail-fast at boot. `model_revision_sha` is dropped — content hash supersedes it (catches LoRA-finetune miscalibration too). |

### 3.5 Selection math (single contract)

For a request `r`, page `p`, head `h`, layer `l`:

```
signature[l, p, h, d] = mean over tokens t in page p of (
    nope_bf16[t, channel_selection[l, h, d]] * channel_weights[l, h, d]
)

q_channel[r, l, h, d] = q[r, l, h, channel_selection[l, h, d]] * channel_weights[l, h, d]

page_score[r, l, p] = max over h of (
    sum over d of q_channel[r, l, h, d] * signature[l, p, h, d]
)

page_score_reduced[r, l, p] = all_reduce_SUM_TP(page_score[r, l, p])

selected_pages[r, l] = ascending(topk(page_score_reduced[r, l, ·], max_top_k))
                     ∪ hot_pages[r]
```

`max` over heads (vs `sum`) per `kernel_audit_memo.md` — defensible default; `sum` is a config knob if AC-9 NIAH delta exceeds budget.

### 3.6 CUDA graph contract (per AC-6)

The captured region writes into pre-allocated `DSGraphState` buffers:

1. `selected_indices: int32[max_bs, max_top_k]` (-1 padded)
2. `valid_lengths:    int32[max_bs]`
3. `scratch_partial_scores`, `scratch_partial_indices` if two-stage selection is used

Rules enforced by `assert_no_alloc_in_region`:
- No `torch.empty/zeros/tensor` calls inside the region.
- No host-readable branches (`if tensor.item()` is rejected by review).
- `max_top_k` is the static-shape parameter; setting `max_top_k < top_k` fails at startup, not at capture.

This is the structural antidote to sglang-last G1.

---

## 4. Status of the in-tree DS package (as of `dev/double-sparsity-standalone @ fc5f6b200`)

Per loop1/loop2/loop3 + the existing files under `python/sglang/srt/layers/attention/double_sparsity/`:

| Component | LOC | Status | Loop |
|-----------|-----|--------|------|
| `config.py` | 107 | ✅ ABI locked (no `selection_mode`/`top_p`) | L1 |
| `validator.py` | 207 | ✅ HiSparse mutex + missing-config + page-size + backend/dtype | L1 |
| `channel_mask.py` | 570 | ✅ safetensors + sha256 + value-domain rejection | L1+L2 R1 |
| `calibrate.py` | 441 | ✅ CLI + tiny CI fixture | L1 |
| `page_signature_table.py` | 185 | ✅ alloc + lifecycle scaffold | L2 |
| `page_signature_write.py` | 498 | ✅ FP8-scale-aware torch reference; ⚠ **not yet called from KV-write sites** (Loop 3 M1) | L2/L3 |
| `selection_kernel.py` | 488 | ✅ torch reference (project_query, compute_page_scores, select_topk_sequence_order); ⚠ Triton port pending | L2 |
| `selector.py` | 327 | ✅ ABI + placeholder mode + bind_runtime_data | L1 |
| `page_table_adapter.py` | 404 | ✅ logical → physical → FlashMLA block_table | L2 R0 |
| `cuda_graph.py` | 221 | ✅ DSGraphState + capture wrapper + alloc-detection probe | L2 R1 |
| `metrics.py` | 299 | ✅ Prometheus + meta_info | L1 |
| `error_containment.py` | 92 | ✅ per-request abort | L2 R1 |
| **Total** | **3,887** | | |

Tests: 150 unit tests passing at `test/registered/unit/layers/attention/test_double_sparsity_unit.py` (Loop 2 R9, commit `ba7d55d64`).

**The work remaining (Loop 3 hard scope):**
1. **M1** — Hook `page_signature_write` at the real `set_mla_kv_buffer` call sites in `nsa_backend.py` so `valid_mask` actually transitions False→True at serve time. Today it stays all-False and the selector either skips DS or scores against zero signatures.
2. **M2** — Build `sparse_mask` from `req_to_token` and attach to `ForwardBatch`. Today selection can score across page boundaries between requests.
3. **M3** — End-to-end DeepSeek-V3.2 (FP8) `bench_serving` run on 8×H200 with DS enabled, side-by-side vs DS-off. This is the **done** criterion. Loop 2 hit a circuit breaker at R9 because everything else required hardware.

After M1–M3 land, the path to AC-8 (30 tok/s P50, 22s P99 TTFT @ conc=64 on 2-node H200) is a kernel-perf hardening pass:
- Triton port of `compute_page_scores` (the torch path is correct + capture-safe but does multi-dim contractions in CPython; see `kernel_audit_memo.md` §"Known follow-ups").
- Triton port of `page_signature_write` reading FP8 scales without the uint8→fp8 view roundtrip.
- Tune `max_top_k`, `device_buffer_size`, score-aggregation choice (`max` vs `sum`) if NIAH delta exceeds 5 pp at any context length.

---

## 5. Decisions that govern the design

From `loop1/refined_plan_v3.md`:

- **DEC-1** — Hardware: 2-node H200 cluster, 8-way TP default (16-way cross-node acceptable but slower per DEC-9's per-step all-reduce cost).
- **DEC-2** — Radix cache default under DS: **off** until M3-B hardware fixture passes; operator flips it after Phase 5 of the runbook.
- **DEC-6** — Initial scope is single-mode top-K only. Top-p (Twilight ABI) is a separate follow-on, separate plan.
- **DEC-8** — DS and HiSparse are mutually exclusive at runtime (startup error). Both ship in the codebase. No plans to integrate.
- **DEC-9** — TP sync is per-step `all_reduce(SUM)` on scalar `[max_pages]` scores; never an all-gather of signatures.
- **DEC-10** — Capability check is V3.2-specific initially (`is_deepseek_nsa(hf_config)` proxy); GLM-5.1 generalization once it ships the same indexer interface.

---

## 6. Open architectural questions

1. **Score aggregation across heads** — `max` vs `sum`. `kernel_audit_memo.md` says `max` is the defensible default but `sum` may give better NIAH recall for some calibrated models. Decide after AC-9 measurement, then lock the choice in `DoubleSparsityConfig.extra` (don't add a top-level field).
2. **Page-stability under radix-cache eviction** — M3-B is the right fixture but the synthetic CI version is a shape-only probe. The real cold-vs-warm bit-equality on hardware is still owed (Phase 5).
3. **CPU-offload-shaped lifecycle hooks** — the signature-table lifecycle (`on_assign` / `on_free` / `on_evict` / `on_reuse`) is intentionally shaped to admit CPU offload later. Confirm this shape survives once we have measured perf evidence; otherwise simplify.
4. **GLM-5.1 capability seam** — `is_deepseek_nsa(hf_config)` is a V3.2-shaped proxy. When GLM-5.1 lands, decide whether the seam is `(model_arch, has_indexer)` or `(use_mla, has_paged_kv, supports_fp8)`. Don't prematurely generalize.
5. **`label_dim`** — the design picks 16 at fp16 (~480 MB/rank @ 1M ctx, TP=8). Whether this is the right operating point or 8/24/32 wins depends on AC-9 NIAH recall vs the memory budget on 64K and 128K. Open until measured.
6. **Triton port priority** — `compute_page_scores` vs `page_signature_write`. The audit memo flags both; tracing of M3 should pick the actual bottleneck before we port either.

---

## 7. What this design explicitly is not

- **Not a HiSparse algorithm.** Zero entries under `mem_cache/sparsity/algorithms/`, zero `_ALGORITHM_REGISTRY` registrations.
- **Not coupled to PD-disagg.** Works on a single-instance server.
- **Not a port of sglang-last's DS code.** The previous code's three-stage Triton kernel was per-token; this design is per-page from the math up.
- **Not a port of Twilight.** Twilight is a selector framework (Quest/DS/SparQ/StreamingLLM + a pruner). This is the DS selector alone, page-aware, FlashMLA-targeted.
- **Not adaptive selection.** Top-p / Twilight enablement is a separate follow-on with its own ABI design.
- **Not multi-model in v1.** V3.2-only capability check; GLM-5.1 is deferred-but-hard.

---

## 8. Where to read more

- The three reference implementations: [`study/00-survey.md`](00-survey.md), [`study/.scratch/survey-{doublesparse,sglang-last,twilight}.md`](.scratch/).
- The kernel audit (math vs paper + page-granularity delta): [`development/kernel_audit_memo.md`](../../kernel_audit_memo.md).
- The refined Phase 1 plan (acceptance criteria, hook site, hardware): [`development/loop1/refined_plan_v3.md`](../../loop1/refined_plan_v3.md).
- The Phase 2 runbook (calibrate → boot → bench → comparator → M3-B): [`development/loop2/RUNBOOK.md`](../../loop2/RUNBOOK.md).
- The Phase 3 hard scope (M1/M2/M3): [`development/loop3/draft.md`](../../loop3/draft.md).
- The previous DS code (for comparison): [`past_implementations/sglang-last-with-double-sparsity/`](../sglang-last-with-double-sparsity/).
