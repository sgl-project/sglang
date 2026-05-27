# 06 — Proposed Standalone Double Sparsity for DeepSeek-V3.2 (FP8) — v2

**Status:** as-built design synthesized from `development/loop{1,2,3}/` + the in-tree DS package at `python/sglang/srt/layers/attention/double_sparsity/`, **plus the minimal-viable roadmap to a DSv3.2 deployment that matches/beats the default DSA + radix-cache + CUDA-graph baseline** (§9–§11).

**v2 changes vs v1:** added §9 (MVP roadmap, Phases A/B/C), §10 (calibration recipe — reused from references), §11 (kernel-fusion roadmap — deferred until MVP works), §12 (deferred items + ongoing issues bucketed must-address / should-address / defer, plus §12.7 already-resolved items), **§13 (architecture rotation: token-level signatures at page_size=64 — supersedes page-level signatures in §3.3 and §3.5)**. The relaxed performance target lives in §9.2.

> **⚠ §13 supersedes the page-level design in §3.3, §3.4 G6, and §3.5.** Read §13 before applying any decision from §3 — page-level signatures are no longer the design point. The rest of §3 (hook site, selector ABI shape, FP8 dequant, TP sync, CUDA-graph contract) is unchanged.

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

---

## 9. Minimal-viable roadmap to a DSv3.2 deployment

This section answers: **how far are we from a minimal working DSv3.2 deployment, and what is the smallest path to one that matches/beats the default DSA + radix-cache + CUDA-graph baseline?**

### 9.1 Where we are right now (commit `fc5f6b200`)

- ✅ 3,887 LOC of DS package, ABI locked, validator + config + mask loader green.
- ✅ 150 unit tests passing.
- ✅ CUDA-graph capture wrapper exists (`cuda_graph.py::DSGraphState`).
- ✅ Page-table adapter emits FlashMLA `block_table` from logical IDs (`page_table_adapter.py`).
- ⚠ **Selector returns placeholder mode unless `bind_runtime_data` runs at boot for every DS-enabled layer.** Without that, no DS forward happens.
- ⚠ **`PageSignatureTable.valid_mask` is all-False at serve time** — `page_signature_write` is never called from the live KV-write sites (Loop 3 M1 gap).
- ⚠ **No per-request page-ownership mask on `ForwardBatch`** — selection can score against pages from other requests in the same batch (Loop 3 M2 gap).
- ⚠ **Zero end-to-end V3.2 FP8 runs on real hardware** — Loop 2 circuit-broke at R9 before any `bench_serving` against a real model.
- ⚠ **No calibrated channel mask exists** for DSv3.2 yet. The `calibrate.py` CLI works against synthetic tiny fixtures (CI path) but has not been run against a real V3.2 checkpoint.

**Honest summary:** the structural plumbing is in. The data isn't flowing through it. There is **zero measured behavior** on real V3.2.

### 9.2 Relaxed performance target (replaces AC-8's hard SLO)

The original client SLO (P50 ≥ 30 tok/s per request, P99 TTFT ≤ 22 s at conc=64, ISL≈4096, OSL=512, ~55% prefix-cache hit) is **kept as the aspiration**, but the working target is now:

> **Match or beat DSv3.2 running with its default DSA (NSA `Indexer`) on TP=8, radix cache ON, CUDA graphs ON, on the same workload, without significant accuracy reduction.**

"Significant" = the AC-9 budgets: NIAH retrieval @ 4K/16K/64K within **5 percentage points** of the DSA baseline; MMLU 5-shot within **1.0 percentage point**. Both retain the `kernel_audit_memo.md` defensibility.

Why this relaxation is honest:
- The 30 tok/s P50 SLO is a *client* number, not a derived number. We don't know yet whether DSA itself reaches it on this workload at conc=64 — bench it first.
- DSA is the right baseline because that's what the V3.2 indexer ships with. Beating DSA at the same TP/radix/CUDA settings is a clean, falsifiable comparison.
- If DSA on TP=8 + radix + CUDA-graphs already meets the 30/22 SLO, then matching DSA also meets the client SLO. If DSA misses it, no individual selector swap can save us — that becomes a stack-level conversation, not a DS conversation.

### 9.3 Phase A — Make DS actually run on DSv3.2 (no perf target)

**Goal:** a DS-enabled `bench_serving` run on 8×H200 against DSv3.2 FP8 that does not crash, produces non-garbage output, and demonstrates the selector is doing real work (`selected_pages < total_pages`, `valid_mask` is non-zero, hot-page rule fires).

| Step | What | Where | Owner trigger |
|------|------|-------|---------------|
| **A1** | Wire `page_signature_write` to KV-write sites in `nsa_backend.py` (search for `set_mla_kv_buffer` — Codex flagged ~L1383, ~L1583, ~L2108 in Loop 2). Retract on KV-free in `req_to_token_pool` deallocation. | `python/sglang/srt/layers/attention/nsa_backend.py` + `double_sparsity/page_signature_table.py` | Loop 3 M1 |
| **A2** | Build `sparse_mask: [bs, max_pages]` from `req_to_token_pool.req_to_token`, `req_pool_indices`, `seq_lens`. Attach to `ForwardBatch.sparse_mask`. `retrieve_topk` consumes it before argmax. | `python/sglang/srt/model_executor/forward_batch_info.py` + `double_sparsity/selector.py` | Loop 3 M2 |
| **A3** | Add a V3.2-shaped calibration mode to `calibrate.py` (see §10). Run it against `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2/` to produce `dsv32-fp8-channel-mask.safetensors`. | `double_sparsity/calibrate.py` | new |
| **A4** | Boot DS with the real mask (`serve_double_sparsity.sh`). Validator passes; `bind_runtime_data` runs for all layers; `/metrics` shows `channel_mask_valid=1`. | `development/serve_double_sparsity.sh` + `validator.py` | Loop 2 RUNBOOK Phase 2 |
| **A5** | **Lightweight quality smoke** (see §9.4 — this is the user's "minimal valid correctness" gate). | `test/manual/test_dsv32_quality_smoke.py` (new) | new |
| **A6** | `bench_serving` smoke: ≥ 64 requests, ISL ≈ 4096, mixed lengths, conc=16/32/64, **DS-on**. Crash-free; non-trivial selection (selected_pages < total_pages on at least 90% of requests); `dense_fallback_total` matches `error_containment` accounting. | `development/benchmark.sh` | Loop 3 M3 |

**Done criterion for Phase A:** a committed round summary showing a successful `bench_serving` run of DSv3.2 FP8 with DS-on on 8×H200, with the quality smoke (§9.4) passing. No comparison vs DSA yet — that's Phase B.

### 9.4 Lightweight quality smoke (Phase A5)

The full NIAH/MMLU regression suite (Phase B6) is too heavy for the inner loop. Phase A needs a **fast, deterministic, hardware-cheap check** that answers a single question: *did DS-on break generation quality vs the default DSv3.2 output?*

**Fixture:** ~20 deterministic prompts (committed alongside the test). Mix of:
- 5 short QA prompts ("What is the capital of …?" type — single-token answers).
- 5 code-completion prompts (~32-token continuations of a fenced Python snippet).
- 5 summarization prompts (~3-sentence summaries of a 256-token passage).
- 5 needle-in-haystack mini prompts at ISL ≈ 4K with the needle deliberately placed past the local window (forces the selector to retrieve it).

**Generation:**
1. Run the fixture with **DSA-on (default V3.2)** → reference outputs, cached on disk (one-time, recompute on V3.2 weight bump).
2. Run the fixture with **DS-on** → candidate outputs.
3. Both runs use `temperature=0`, `top_p=1`, identical sampling seeds.

**Pass criteria (all must hold):**
- **Prefix-match:** ≥ 80 % of the 20 prompts have **first 32 tokens identical** between DSA-on reference and DS-on candidate.
- **ROUGE-L:** mean ROUGE-L over the 20 candidates vs the 20 references ≥ 0.85.
- **Needle recall:** ≥ 4 of 5 NIAH-mini prompts return the planted needle string (the same threshold the AC-4 sanity probe uses, just on 5 prompts instead of 1).
- **No catastrophic divergence:** no single prompt's first 8 tokens differ entirely (a "DS picked nothing useful" failure mode).

**Failure modes this catches:**
- `valid_mask` all-False (DS would pick empty page set → garbage output → all four thresholds fail).
- Channel mask wrong shape / wrong dtype matched the validator but is semantically corrupt (passes value-domain but fails sanity probe).
- TP rank divergence (selected_indices differ across ranks → mid-sequence quality cliff).
- Hot-page rule wired wrong (recent context invisible → NIAH-mini fails).

**Why this is the right level for Phase A:** it runs in < 5 minutes on 8×H200, requires no labeled eval data, and is **fully deterministic** — a failure is reproducible without standing up an LM-Eval-Harness pipeline. It is **not a quality release gate**; that's Phase B6.

### 9.5 Phase B — Match the DSA baseline

**Goal:** demonstrate DS-on ≥ DSA-on throughput at the same TP/radix/CUDA settings on the same workload, with NIAH/MMLU deltas inside the §9.2 budgets.

| Step | What | Done criterion |
|------|------|----------------|
| **B1** | Record the DSA baseline: DSv3.2 default, TP=8, radix cache ON, CUDA graphs ON, on `gsp_isl4096_osl512_c64.jsonl`. Save TPS / TTFT / TPOT / goodput. | `development/benchmark_baseline.sh` produces a JSON; comparator (`benchmark_compare.py`) accepts it. |
| **B2** | Turn on **CUDA graphs** under DS at conc 16/32/64. Validate capture succeeds, replay matches eager output on a deterministic fixture, `assert_no_alloc_in_region` does not trip. | AC-6 positive tests pass on real hardware. |
| **B3** | Turn on **radix cache** under DS. Run the M3-B hardware fixture (`page_signature_write --m3b-fixture-hardware-run`) — cold-prefix vs warm-prefix signatures must be bit-stable. Flip `_double_sparsity_radix_fixture_passed = True` (DEC-2). | RUNBOOK Phase 5 passes. |
| **B4** | Run `bench_serving` with DS-on at conc 16/32/64 with both knobs on. | JSON produced; comparator accepts the row. |
| **B5** | Run the comparator: `python development/benchmark_compare.py --ds-results ... --baseline-results ...` → side-by-side report. | DS-on TPS ≥ DSA-on TPS at conc=64 (the gate row); P99 TTFT not worse than +10 % vs DSA. |
| **B6** | Full quality gate: NIAH @ 4K / 16K / 64K + MMLU 5-shot via `test/manual/test_double_sparsity_v32.py`. | NIAH ≤ 5 pp delta at each length; MMLU ≤ 1.0 pp delta. |

**Done criterion for Phase B:** comparator emits a green row at conc=64, AC-9 quality passes, and the only knobs differing between baseline and DS columns are `--enable-double-sparsity` + `--double-sparsity-config`.

### 9.6 Phase C — Kernel-perf hardening (only if Phase B is slower)

**Trigger:** Phase B5 shows DS-on < DSA-on TPS at conc=64. If we already match/beat DSA after Phase B, **Phase C is deferred work, not blocker work.**

Phase C is the kernel-fusion roadmap (§11). It is gated on Phase B evidence so we don't over-port without a profile.

### 9.7 Effort estimate (very rough)

| Phase | Items | Estimated bench-day effort |
|-------|-------|----------------------------|
| A1 | M1 hook | 1 day (3 KV-write sites, lifecycle retract) |
| A2 | M2 sparse_mask | 0.5 day |
| A3 | V3.2 calibration | 1–2 days (recipe is settled, V3.2 MLA layer enumeration is new) |
| A4 | Boot + validate | 0.5 day |
| A5 | Quality smoke fixture + runner | 1 day |
| A6 | bench_serving smoke run | 0.5 day |
| **Phase A total** | | **~4–5 days** |
| B1 | DSA baseline run | 0.5 day |
| B2 | CUDA graph on DS @ conc 16/32/64 | 1 day (debugging capture-time gotchas is likely) |
| B3 | Radix cache + M3-B hardware | 1 day |
| B4 | DS bench run | 0.5 day |
| B5 | Comparator + iterate | 1 day |
| B6 | NIAH/MMLU quality | 1 day |
| **Phase B total** | | **~5 days** |
| C | Kernel ports (each, see §11) | 1–3 days per kernel; only if needed |

Numbers are best-case-no-surprises; double them for the realistic budget.

---

## 10. Calibration recipe (reused from references)

**Principle (per user direction):** the **calibration dataset and scoring criteria stay identical** to DoubleSparse / sglang-last / Twilight. We only add the code that adapts the recipe to DSv3.2's MLA layer enumeration and channel-name map. No new dataset, no new scoring metric.

### 10.1 Shared recipe across the three references

Verified across `DoubleSparse/config/offline_calibration.py:18-22,91-93,152`, `sglang-last/python/sglang/srt/.../init_double_sparsity_channel_config`, and Twilight's "configs from external DoubleSparse repo" pointer:

| Knob | Value | Source |
|------|-------|--------|
| Dataset | Pile validation split | `DoubleSparse/config/offline_calibration.py:91` (`load_dataset("monology/pile-uncopyrighted", split="validation")`) |
| Sample seed | `42` | `offline_calibration.py:18` |
| Number of samples | `256` | `offline_calibration.py:20` |
| Sequence length | `512` tokens | `offline_calibration.py:21` |
| Scoring method | **Method 1**: `mean(abs(Q · K))` per channel, post-RoPE | `offline_calibration.py:152` |
| Projection | Per-(layer, head, q_proj_or_k_proj) | per-head sort of channels by score |
| Output | JSON dict keyed by `model.layers.<i>.self_attn.<q|k>_proj` → list of per-head channel-index lists | `offline_calibration.py` save block |

For our DSv3.2 calibration we **keep all of the above** and only adapt the *layer enumeration* and the *projection-name keys*.

### 10.2 What changes for DSv3.2

DSv3.2 uses Multi-head Latent Attention (MLA) with absorb-prepare projections, so the projection-name map is different:

- **Standard LLaMA / Mistral (references):** keys are `model.layers.{i}.self_attn.q_proj` and `model.layers.{i}.self_attn.k_proj`, channel-axis = `head_dim` (typically 128).
- **DeepSeek-V3.2 MLA:** the K side that DS scores against is the **nope half of the latent-projected K** that NSA's `quant_k_cache.py` lays out. The channel axis for DS is the post-RoPE nope dimension (512 in V3.2). Per-(layer, head) means per-(layer, MLA-head); head count after TP shard = `H_local`.

Concrete adaptations to `calibrate.py`:

1. **Layer enumeration:** walk `model.model.layers` like the references, but read `self_attn.kv_b_proj` (MLA's K↑proj) for the K side rather than `self_attn.k_proj`. The Q side comes from `self_attn.q_b_proj` after the absorb-prepare math (or from probing the post-RoPE Q tensor at inference time, depending on which is easier).
2. **Channel axis:** 512 (nope dim) — picked by V3.2 config, not hard-coded.
3. **Channel selection size:** unchanged conceptually — pick top `label_dim` (default 16) channels per (layer, head) by mean-abs-product score.
4. **Output key naming:** keep the `model.layers.{i}.self_attn.{q,k}_proj` convention for downstream compatibility — DS's channel-mask loader treats keys as opaque labels into the per-(layer, head) tensor, so the *name* doesn't need to match V3.2's actual module names. The `channel_selection[L, H, label_dim]` tensor in the resulting `safetensors` is what `selector.py` actually consumes.

### 10.3 What the existing `calibrate.py` already does

`python/sglang/srt/layers/attention/double_sparsity/calibrate.py` (441 LOC):
- ✅ CLI entrypoint: `python -m sglang.srt.layers.attention.double_sparsity.calibrate --model … --output … --dtype fp8_e4m3 --page-size 64 --label-dim 16`.
- ✅ Writes `safetensors` with the AC-4 schema (tensors + metadata + `content_sha256`).
- ✅ Tiny CI fixture path that runs in < 1 min on a synthetic NSA-shaped tiny model.
- ⚠ V3.2 path is untested — needs the layer enumeration and projection-name adaptations from §10.2.

### 10.4 Calibration step list (Phase A3)

| Step | What | Output |
|------|------|--------|
| A3.1 | Adapt `calibrate.py` to walk MLA layers (`self_attn.kv_b_proj` instead of `self_attn.k_proj`); add `--model-arch deepseek_v3` flag that selects the MLA path. | code change |
| A3.2 | Load Pile val with seed=42, 256 samples × 512 tokens (unchanged from references). | recipe match |
| A3.3 | Forward-pass through V3.2 with hooks on the post-RoPE Q and the K-side projection; accumulate `mean(abs(Q · K))` per (layer, head, channel). | tensor `[L, H, 512]` |
| A3.4 | Sort each per-(layer, head) row descending; take top `label_dim` indices and the corresponding scores → `channel_selection[L, H, label_dim]`, `channel_weights[L, H, label_dim]`. | tensors |
| A3.5 | Write `dsv32-fp8-channel-mask.safetensors` with metadata (`dtype=fp8_e4m3`, `head_dim=512`, `page_size=64`, `label_dim=16`, `schema_version=1`, `content_sha256`). | `/models/dsv32-fp8-channel-mask.safetensors` |
| A3.6 | Run `channel_mask.py::load_channel_mask` against it; verify the AC-4 sanity probe passes (one NIAH-min prompt retrieves the needle). | green |

**No new dataset, no new scoring criterion.** This is intentional — diverging from Pile-val-256x512-Method-1 would mean we can no longer reuse the reference implementations' calibration evidence (§3 of `00-survey.md`, the open question on calibration sensitivity in §9.1 of that doc).

### 10.5 Calibration sensitivity (open question, deferred)

Whether Method 1 with this exact dataset is optimal for DSv3.2's MLA-shaped projection is **explicitly an open question** (`study/00-survey.md` §9.1). Resolving it is *not* in MVP scope. Once Phase B is green, an ablation can revisit:
- Calibration set choice (Pile vs C4 vs domain-matched).
- Method 1 (`mean|Q·K|`) vs Method 2 (`mean|Q|·mean|K|`) vs Method 3 (mean activation magnitude).
- Sample count (256 vs 64 vs 1024).
- Sequence length (512 vs 2048).

That ablation is its own follow-on plan; it should not gate the first DSv3.2 deployment.

---

## 11. Kernel-fusion roadmap (deferred until MVP works)

**Principle (per user direction):** we **do not port reference kernels until Phase B identifies an actual bottleneck.** Premature porting risks (a) correctness divergence we don't have a baseline to detect, (b) sunk-cost lock-in to a kernel that turns out not to be the bottleneck, (c) blocking the Phase A/B execution behind kernel work that may be unnecessary.

Phase C is gated on a profile from Phase B5. The kernels listed below are the **catalog we draw from** when Phase C fires.

### 11.1 Fusion opportunities ranked by likely impact

| Rank | Fusion | Before | After | Borrow from | Risk |
|------|--------|--------|-------|-------------|------|
| **1** | **Fused FP8-dequant + channel projection** in `page_signature_write` | dequant FP8 K → BF16 intermediate → gather heavy channels → mul `channel_weights` → mean across page | one Triton kernel: load FP8 + per-tile scale, dequant inline (no BF16 materialization), gather + project + reduce in-register | `nsa/dequant_k_cache.py` for the FP8 layout; sglang-last's `_sparse_fwd_kernel_flash_decode_stage1` (lines 329-398) for the tile-wise pattern; Twilight's `bgemv_int8_kernel` for the quantized-K BGEMV layout | Med — FP8 + page granularity is new, no reference exactly matches; ±0.5 % tolerance against torch ref is the gate |
| **2** | **Fused score + top-K** in `compute_page_scores` + `select_topk_sequence_order` | matmul → materialize `[bs, max_pages]` → `torch.topk` (host on sglang-last, device-side via stage-1+stage-2 in current torch ref) | one kernel: BGEMV-style score → device radix-select top-K, no intermediate `[bs, max_pages]` allocation | ftka's `raft_topk` (vendored RAFT radix-select; see [`study/05-flash-topk-attention.md`](05-flash-topk-attention.md)); ftka's flash-topk-attention fused score+select pattern | Low if `raft_topk` is the only piece adopted (it's CUDA-graph friendly by construction). Higher if the full flash-topk fused pipeline is adopted (more surface to validate) |
| **3** | **Fused score + sparse_mask AND** | score all pages → multiply by sparse_mask (zero out invalid) → reduce | mask in the inner loop, skip the MAC for invalid pages entirely | new (no precedent; trivial Triton predicate) | Low |
| **4** | **Fused hot-page force-include + top-K** | top-K → check hot-page set → insert/override scores → re-sort | inject hot-page scores as `+inf` in stage-1, single top-K | sglang-last's existing `select_topk_sequence_order` (post-topk override pattern); the new fused form is straightforward in Triton | Low |
| **5** | **Fused Q-side projection + score** | `q_channel = q[..., channel_selection] * channel_weights` (gather + mul) → BGEMV with signature | one kernel: load Q + channel indices + weights, project on-the-fly, BGEMV against signature, write scores | Twilight's `get_label_tensor_kernel` (channel.py:11-79) for the gather + scale pattern | Low |
| **6** | **Fused signature update for hot page** | every decode step: re-run `page_signature_write` on the single active page | persistent kernel that updates one page-row per decode step without re-launching the full write kernel | new; hot-page update is single-page-per-step so launch overhead matters | Low |

### 11.2 Kernels we have ready (from literature review)

From [`study/05-flash-topk-attention.md`](05-flash-topk-attention.md) and `study/.scratch/survey-{doublesparse,sglang-last,twilight}.md`:

| Kernel | Repo | File | Purpose | Likely role in Phase C |
|--------|------|------|---------|------------------------|
| `_sparse_fwd_kernel_flash_decode_stage1` | sglang-last | `triton_ops/double_sparsity_attention.py:329-398` | BGEMV Q_label · K_label_bufferᵀ for per-token approximate scores | **Reference** for the BGEMV pattern; the page-aware variant is a rewrite, but the tiling and dtype-coercion choices port directly |
| `get_label_tensor_kernel` | Twilight | `twilight/kernel/triton/channel.py:11-79` | Gather heavy channels per head with per-token gather indices | **Direct reuse** for the Q-side channel projection (rank 5 above) |
| `bgemv_int8_kernel` | Twilight | `twilight/kernel/triton/bgemv_int8.py:12-73` | INT8 quantized BGEMV with per-token scales | **Pattern reference** for the FP8 fused kernel (rank 1); shows how to fold per-tile scales into the BGEMV |
| `raft_topk` | ftka (vendored RAFT) | `flash-topk-attention/csrc/include/raft_topk.cuh:984-1010` | Single-kernel radix-select top-K, CUDA-graph safe | **Direct adoption** for `select_topk_sequence_order` device-side top-K (rank 2) — solves the host-sync class of bugs the sglang-last design had |
| `top_p_fp16_return_mask` | Twilight | `csrc/src/sampling.cu:22-42` | CUB-based top-p prefix scan returning a bool mask | **Not in MVP scope** — this is for the Twilight selector ABI (deferred per DEC-6). Catalog only. |
| `_attn_fwd` (per-block int8 K) | Twilight | `twilight/kernel/triton/qk_int8_per_block.py:23-97` | Fused attention with per-block int8 K | **Not in MVP scope** — FlashMLA is doing this for us; we don't fork the attention kernel itself |
| `fwd_sparse_no_mask` | DoubleSparse | `DoubleSparse/models/model.py:265-294` Triton call | Per-token sparse attention | **Not in MVP scope** — per-token, not paged; superseded by FlashMLA |

### 11.3 What is explicitly NOT to be ported

Even if Phase B shows DS-on slower than DSA, **the following are out of scope** for the initial MVP iteration:

- The **DSA / NSA Indexer** itself — we keep V3.2's stock indexer as the *fallback* (the `_select_topk_indices` branch's `else` arm), not something to optimize.
- **Twilight's pruner** (top-p, threshold) — the AC-11 ABI explicitly rejects `selection_mode`; this is a separate plan per DEC-6.
- **DoubleSparse's DS-Offload** (DGL `gather_pinned_tensor_rows`) — paper §DS-Offload is deferred. The signature-table lifecycle hooks are *shaped* to admit it later (§3.3, §6 open question 3).
- **Fused attention kernel rewrites** — FlashMLA owns the softmax(QK)·V step. We do not fork or fuse around FlashMLA in v1.

### 11.4 Phase C entry condition (precise)

Phase C fires if and only if **at conc=64**:
- DS-on TPS < DSA-on TPS by more than 5 %, **and**
- Profile evidence (from `sglang-torch-profiler-analysis`) names a specific kernel as the bottleneck.

If the bottleneck is *not* in the DS code path (e.g. it's FlashMLA itself, the all-reduce, or radix-cache eviction), Phase C does not fire — that becomes a different conversation.

The profile-first discipline avoids the sglang-last failure mode where the DS kernels were ported eagerly *before* the host-sync problem (the actual perf killer per §2 G1) was understood.

---

## 12. What is still deferred + ongoing issues (and whether MVP must address them)

This section enumerates everything the previous loops explicitly deferred plus everything that is currently a known issue in the in-tree code at `dev/double-sparsity-standalone @ e32ec2b4b`. Each item is classified into one of three buckets:

- 🛑 **Must address in MVP** — blocks any V3.2 DS deployment from working at all.
- ⚠ **Should address in MVP** — would surface as silent bugs at deployment scale; cheap to fix while we're here.
- 🟰 **Defer past MVP** — downstream feature or optimization; not a blocker.

### 12.1 Deferred from Loop 1 (refined_plan_v3.md scope decisions)

| # | Item | Bucket | Why |
|---|------|--------|-----|
| 1.1 | **GLM-5.1 support** | 🟰 Defer | Deferred client requirement; schema already shaped for it (`channel_mask.py` is dtype/length-agnostic). Capability check stays V3.2-specific (DEC-10) until GLM-5.1 ships its indexer interface. |
| 1.2 | **128K ISL workload** | 🟰 Defer | Deferred client requirement; no length-dependent fields in the artifact schema. Memory budget at 1M context already documented in §3.3 — 128K is well inside it. |
| 1.3 | **nvfp4 / mxfp4 weights** | 🟰 Defer | Deferred client requirement. Channel mask is dtype-agnostic (FP8 path validated; FP4 needs new dequant kernel). |
| 1.4 | **DP Attention** | 🟰 Defer | Newly added to `development/CLIENT_SLOS.md` as deferred item 4. No DS-specific code change needed; the selector's TP rank-sync (DEC-9) generalizes — but DP changes the all-reduce group topology, so an explicit test is owed before declaring support. |
| 1.5 | **Twilight / top-p selection** | 🟰 Defer | AC-11 ABI explicitly rejects `selection_mode` / `top_p`. Downstream task — separate plan, separate ABI design. |
| 1.6 | **"Extensions" engine knob** | 🟰 Defer | Downstream framework feature. Not blocked by DS shape; DS would be a consumer once the knob exists. |
| 1.7 | **PD-Disagg / HiCache integration** | 🟰 Defer | Downstream; no customer ask on DS yet. The HiSparse mutex (DEC-8) explicitly *prevents* HiSparse coexistence at runtime — DS-on-PD-Disagg without HiSparse remains an open shape question. |
| 1.8 | **DS-Offload (paper §DS-Offload)** | 🟰 Defer | CPU pinned K,V + double-buffer prefetch. Signature-table lifecycle hooks (`on_assign/free/evict/reuse`) are shaped to admit it later; not implemented. |
| 1.9 | **DEC-2 radix-cache default flip** | ⚠ Should address | Loop 1 left this as an explicit operator decision: flip `_double_sparsity_radix_fixture_passed = True` *after* Phase 5 M3-B hardware run passes. Without this flip, the baseline comparison in §9.5 B5 cannot turn radix cache on under DS — which would break the apples-to-apples requirement (baseline uses radix). **Required for Phase B.** |

### 12.2 Deferred from Loop 2 (R9 retro — explicit non-goals carried forward)

These were in Loop 2's R9 summary and got copied verbatim into the Loop 3 draft non-goals list:

| # | Item | Bucket | Why |
|---|------|--------|-----|
| 2.1 | **AC-8 captured-path zero-allocation** — Triton kernel for value-domain assertions | 🟰 Defer | Today's `assert_no_alloc_in_region` is a host-side allocator counter (`cuda_graph.py:191-221`). It detects `torch.empty` but not direct `cudaMalloc` from inside Triton. Loop 2 acknowledged this gap as acceptable because the kernels are reviewed for in-region allocation at merge time. **Not blocking** unless a Phase C Triton port introduces a direct `cudaMalloc`. |
| 2.2 | **AC-8 wrapper / multi-step backend metadata fixup** | 🟰 Defer | Multi-step decode (overlap scheduling) integration. Adds new metadata mutation paths that the static-buffer contract has to absorb. Not blocking the first DSv3.2 deployment (V3.2 doesn't use overlap-scheduling by default per Loop 1 setup). |
| 2.3 | **M3-B perturbation negative** (fixture redesign) | 🟰 Defer | The synthetic CI M3-B fixture only proves *positive* cold/warm equivalence on a shape-only probe. A perturbation negative (deliberately mutate the channel mask between cold and warm; the test should fail) is needed for confidence. Not blocking — it's a test gap, not a runtime gap. |
| 2.4 | **Real two-rank TP divergence test** (multi-process harness) | 🛑 **Must address** | This one is *important.* AC-7 today is single-rank; a synthetic two-rank fixture exists in unit tests but a real multi-process harness has never been wired. **At TP=8 we cannot ship without verifying rank-bit-equal `selected_indices`** — divergence would produce silently wrong attention output at scale. **Required for Phase A6 before any quality call is trusted.** |
| 2.5 | **`transform_index_page_table_decode_fast` 2048 hard-assert** | ⚠ Should address | Loop 2 left this as a "review caught it but not gated by test." If a request exceeds `max_top_k=2048`, the fast-path may produce wrong page tables silently. Add a `tl.device_assert` (Phase A2 work, cheap) so the failure mode is loud. |
| 2.6 | **Operator phases 2–3** (real calibration / benchmark / M3-B hardware) | 🛑 **Must address** | Loop 2 explicitly punted these as "operator-phase 2/3" work; that operator phase **is exactly Phase A of the MVP roadmap (§9.3)**. They are now blocker, not deferred. |

### 12.3 Loop 3 hard scope (= Phase A of the MVP roadmap)

Listed for completeness — all three are 🛑 **must address in MVP**. They are restated here so the deferred-list reader doesn't have to flip back to §9.

- **M1** — Hook `page_signature_write` into NSA backend's `set_mla_kv_buffer` sites (§9.3 A1). Without this, `valid_mask` is all-False; every DS forward picks against zero signatures → garbage output.
- **M2** — Build per-request `sparse_mask` and attach to `ForwardBatch` (§9.3 A2). Without this, selection can pick pages owned by other requests in the same batch.
- **M3** — End-to-end `bench_serving` run on V3.2 FP8 (§9.3 A6). The done criterion for the loop; Loop 3 draft's single-sentence done definition.

### 12.4 Ongoing issues in the current code state (commit `e32ec2b4b`)

These are not "deferred" in the loop-plan sense — they are current code-state issues that need triage as part of executing the MVP roadmap.

| # | Issue | Where | Bucket | Notes |
|---|-------|-------|--------|-------|
| 4.1 | **`page_signature_write` is never called from a live KV-write site** | `python/sglang/srt/layers/attention/nsa_backend.py` — `set_mla_kv_buffer` call sites have no DS hook today (verified: `grep` returns zero hits for `page_signature_write` in that file) | 🛑 Must address | Same as Loop 3 M1. Mentioned twice because it is the load-bearing gap. |
| 4.2 | **Selector boots in placeholder mode when channel mask is misconfigured** | `double_sparsity/selector.py:60-72` `IS_PLACEHOLDER=True` flips to `False` only when `_bind_double_sparsity_runtime_data` (`deepseek_v2.py:1832`) succeeds | ⚠ Should address | `bind_runtime_data` IS wired at per-layer init (`deepseek_v2.py:1541`), but it requires `server_args._double_sparsity_channel_mask` to be pre-populated by the validator. If the validator order is wrong or path is missing, every layer crashes loudly at init. Verify on Phase A4 boot that boot logs show "bind_runtime_data completed" for every DS-enabled layer (RUNBOOK Phase 2 already documents this check). |
| 4.3 | **`compute_page_scores` is a torch reference, not Triton** | `double_sparsity/selection_kernel.py:488` torch reference, multi-dim contractions in CPython | 🟰 Defer (Phase C) | Captures fine under CUDA graphs because buffers are static and ops are PyTorch primitives (no Python-level branching on tensor values). But the wall-clock cost may be material — that's the Phase B5 question. |
| 4.4 | **`page_signature_write` is a torch reference, not Triton** | `double_sparsity/page_signature_write.py:498` | 🟰 Defer (Phase C) | Same story as 4.3. Phase B profile will say whether porting this matters before `compute_page_scores`. |
| 4.5 | **Hardware-level CUDA-graph capture has never run at conc=64** | `cuda_graph.py:109-188` `capture_decode_step` only exercised on synthetic shapes in unit tests | 🛑 Must address (Phase B2) | The capture machinery passes synthetic shape tests, but no one has captured against a real V3.2 conc=64 batch yet. This is the AC-6 positive test from Loop 1 that's still owed. |
| 4.6 | **Two-rank TP harness still synthetic** | Restated from 2.4 above | 🛑 Must address | The DEC-9 `all_reduce(SUM)` path is unit-tested in single-process simulation only. Multi-process two-rank test is the canonical TP correctness check. |
| 4.7 | **M3-B is synthetic CI only; never run on hardware** | `test_double_sparsity_unit.py::test_ds_m3b_synthetic_ci_hook` is shape-only | ⚠ Should address (Phase B3) | Hardware run is RUNBOOK Phase 5; required before DEC-2 default flip (gates radix cache under DS). |
| 4.8 | **`SGLANG_DS_RADIX_OVERRIDE` env var remains** | `validator.py` (separate concern from the removed `SGLANG_DS_ALLOW_*` gates per Loop 2 task 11 deliberation) | 🟰 Defer | Loop 2 explicitly kept this in place. Operator override knob for testing; not a runtime path concern. |
| 4.9 | **`skip_topk` gate fix lives in `forward_absorb_prepare`, not the helper** | `models/deepseek_common/attention_forward_methods/forward_mla.py:245-277` — gate is `or not self.skip_topk or prev_topk_indices is None`, which mirrors Loop 2 R0's `skip_topk` fix | ✅ Already addressed | Loop 2 closed this. Listed here so the reader doesn't think it's still open. |
| 4.10 | **DS branch return type uses unified `page_table_1` Tensor** | `deepseek_v2.py:2060` and the Loop 2 task 2 work | ✅ Already addressed | Loop 2 closed the typed-union vs unified-shape design question (Design C resolution). Listed here for the same reason as 4.9. |

### 12.5 Carry-forward lessons from Loop 2 retro (operating guidance, not code work)

These are not code items — they are operating-process bookkeeping that the MVP execution must observe:

| Lesson | What it says | When it bites |
|--------|--------------|---------------|
| **BL-20260520-read-fields-before-abort-mutation** | Capture batch-wide cursor spans BEFORE invoking abort helpers — `set_finish_with_abort` rewrites `req.origin_input_ids = [0]`. | If a DS error in mid-decode triggers per-request abort (the `error_containment` path), and the surrounding code reads `req.origin_input_ids` *after* abort, it reads `[0]` not the original. Could bite us during Phase A6 if any prompt hits a DS error mid-decode. |
| **BL-20260520-symbol-vs-test-fixture-drift** | Test fixtures must reference live dataclass field names (`forward_batch.rids` not `req_ids`; verify with `dataclasses.fields(ForwardBatch)`). | Only affects test correctness, not runtime. Mention it because Phase A5 (quality smoke) and Phase B6 (NIAH/MMLU) both touch `ForwardBatch` and could drift if a field is renamed upstream. |
| **2-vs-6 budget mismatch (Loop 2 retro)** | If 2 consecutive rounds open more gaps than they close, **stop the loop manually** with `/humanize:cancel-rlcr-loop` and reassess scope. Don't wait for the circuit breaker. | Phase A is small (6 items, ~5 days). If we hit Day 3 with only A1 closed and three new gaps open, that's the cancel signal — exactly the failure mode Loop 2 hit at R9. |

### 12.6 Summary — what must be in the MVP

**🛑 Must address in MVP (6 items) — blocks boot, blocks any DS forward, or would be silently-wrong-at-scale:**

1. **M1** — `page_signature_write` hook (12.4 #4.1 = Loop 3 M1). Verified with `grep`: zero hits in `nsa_backend.py` today → `valid_mask` is all-False at serve time.
2. **M2** — `sparse_mask` on `ForwardBatch` (Loop 3 M2). Without it, requests in a batch score against each other's pages.
3. **M3** — end-to-end V3.2 FP8 `bench_serving` run (Loop 3 M3).
4. **Real V3.2 calibration** (12.2 #2.6 — operator phases 2-3 now in scope). `calibrate.py` works on tiny CI fixture; V3.2 MLA layer enumeration path is untested.
5. **Two-rank TP multi-process harness** (12.2 #2.4 / 12.4 #4.6). AC-7 is single-rank today; at TP=8, rank-divergent `selected_indices` produces silently wrong attention. Non-negotiable.
6. **Hardware CUDA-graph capture at conc=64** (12.4 #4.5) — Loop 1's AC-6 positive test on real hardware, still owed.

**⚠ Should address in MVP (4 items) — cheap silent-bug guards:**

7. **DEC-2 radix-cache default flip** after Phase 5 (12.1 #1.9) — gates radix-on in the §9.5 B5 baseline comparison.
8. **Placeholder-mode boot trip-wire** (12.4 #4.2). `bind_runtime_data` IS wired at `deepseek_v2.py:1541`, but it depends on the validator pre-populating `server_args._double_sparsity_channel_mask`. Verify boot logs match RUNBOOK Phase 2.
9. **`transform_index_page_table_decode_fast` device assert at `max_top_k=2048`** (12.2 #2.5) — Loop 2 caught this in review but never gated by test; make silent overflow loud.
10. **M3-B hardware fixture run** (12.4 #4.7) — synthetic CI exists; real run gates #7.

**🟰 Defer past MVP — explicit not-in-scope list:**

- **Loop 1 deferred client requirements:** GLM-5.1 (1.1), 128K ISL (1.2), nvfp4 / mxfp4 (1.3), DP Attention (1.4).
- **Downstream features:** Twilight / top-p (1.5), "Extensions" engine knob (1.6), PD-Disagg / HiCache (1.7), CPU offload / paper §DS-Offload (1.8).
- **Phase C kernel ports (gated on Phase B evidence):** Triton port of `compute_page_scores` (4.3), Triton port of `page_signature_write` (4.4), fused FP8-dequant + projection, fused score + top-K via `raft_topk`, etc. — full catalog in §11.
- **Test gaps that aren't runtime gaps:** AC-8 captured-path zero-allocation device-side assert kernel (2.1), AC-8 multi-step backend metadata fixup (2.2 — V3.2 doesn't use overlap-scheduling by default), M3-B perturbation negative (2.3).
- **Operator knob kept on purpose:** `SGLANG_DS_RADIX_OVERRIDE` env var (4.8) — Loop 2 explicitly kept this.

Any of these gets elevated to MVP only if Phase B5 evidence promotes it.

### 12.7 Already addressed (do not re-do)

Two items from earlier loop discussions are **already closed in the current code state** — listed so future readers don't re-open them:

- **`skip_topk` gate fix in `forward_absorb_prepare`** (12.4 #4.9). Loop 2 R0 closed this; gate is `or not self.skip_topk or prev_topk_indices is None` at `models/deepseek_common/attention_forward_methods/forward_mla.py:245-277`. The fix lives in `forward_absorb_prepare`, not inside `_select_topk_indices` (which is the resolved disagreement from Loop 2's plan).
- **Unified `page_table_1` return type from DS branch** (12.4 #4.10). Loop 2's Design C resolution: both DS and NSA branches of `_select_topk_indices` return the same `page_table_1` Tensor type; the downstream `_forward_flashmla_kv` call has no `isinstance` branch. Lives at `deepseek_v2.py:2060`.

Both of these were live debates in the Loop 2 plan deliberations; both are now resolved structurally. Reviewers picking up the branch should treat them as load-bearing invariants, not open design questions.

---

## 13. Architecture rotation — token-level signatures at page_size=64

This section supersedes the page-level design in §3.3 ("Two-artifact design / Page signature table"), §3.4 G6 (Page size), and §3.5 (Selection math). The rest of §3 — hook site, selector ABI shape, FP8 dequant approach, TP `all_reduce(SUM)`, CUDA-graph contract — is unchanged.

### 13.1 The rotation in one paragraph

**Selection is token-level. Storage is token-level. FlashMLA still reads at page granularity (page_size=64).** A per-(layer, token, head) label cache holds the offline-calibrated channel projection of K. The selector scores tokens, picks the top-K **tokens**, then hands the token indices to NSA's existing `transform_index_page_table_decode` helper, which emits the page-aligned `block_table` FlashMLA needs. Page_size=64 is FlashMLA's KV layout requirement — not a selection-granularity choice.

This is exactly what NSA's `Indexer` does today: token-level top-K → page-table conversion. DS swaps NSA's lightning indexer for an offline-calibrated channel projection, but the surrounding pipeline (selection unit, table conversion, FlashMLA call) stays identical.

### 13.2 Why this matters for the MVP

The page-level design was chosen for memory budget at 1M context (~30 GB/rank per-token vs ~480 MB/rank per-page). Token-level memory at the client workloads:

| Context | Per-token cache / rank (60 layers × H_local=16 × label_dim=16 × 2 B) |
|---|---|
| 4K ISL (client target) | **120 MB** |
| 32K | 960 MB |
| 128K (deferred client) | 3.8 GB |
| 256K | 7.6 GB |
| 1M (no client ask) | 30 GB ← only here does page-level help |

At every context length the client has asked for, token-level is comfortable.

The page-level memory win was paying for itself with complexity nobody asked for: custom `page_table_adapter` (404 LOC), chunked-prefill multi-chunk-per-page accumulation, page-granularity quality artifacts, custom hot-page rule, custom page lifecycle hooks.

### 13.3 What changes in the package

| File | Before (page-level) | After (token-level) | Net LOC |
|---|---|---|---|
| `page_signature_table.py` → `token_label_table.py` (185 LOC) | `[L_local, max_pages, H_local, label_dim]`, allocator-owned page lifecycle | `[L_local, max_tokens, H_local, label_dim]`, slot-indexed by `out_cache_loc` exactly like K/V — no separate lifecycle | similar size, simpler internals |
| `page_signature_write.py` → `token_label_write.py` (498 LOC) | FP8 dequant per page → mean across 64 tokens → write 1 row | FP8 dequant per token → write 1 row, no page reduction | **smaller** (~300 LOC est.) |
| `selection_kernel.py` (488 LOC) | `compute_page_scores` over `[bs, max_pages]`, page top-K | `compute_token_scores` (BGEMV Q_label · K_label_bufferᵀ) over `[bs, max_tokens]`, token top-K | similar LOC; uses the sglang-last 3-stage Triton pattern directly |
| `page_table_adapter.py` (404 LOC) | Custom logical_page → physical_page → FlashMLA block_table | **Mostly deleted** — call NSA's existing `transform_index_page_table_decode` from `nsa_indexer.py` neighborhood | **~100 LOC** (the call wrapper) |
| `cuda_graph.py` (221 LOC) | Static `selected_indices: [max_bs, max_top_k_pages]` | Static `selected_token_indices: [max_bs, max_top_k_tokens]`. Same machinery. | unchanged |
| `selector.py` (327 LOC) | Returns `(selected_page_indices, valid_lengths)` | Returns `(selected_token_indices, valid_lengths)` — matches NSA's existing return shape | unchanged |
| `config.py` (107 LOC) | `top_k` interpreted as max pages | `top_k` interpreted as max tokens (matches sglang-last's `--ds-heavy-token-num` semantics) | unchanged |
| `validator.py`, `channel_mask.py`, `calibrate.py`, `metrics.py`, `error_containment.py` | unchanged | unchanged | — |

**Net effect: ~1500 LOC reshape, but more deletion than addition** because the custom adapter collapses.

### 13.4 Why chunked prefill becomes implicit

sglang's chunked prefill already routes each chunk's `out_cache_loc` to fresh token slots in the KV pool. The token-label cache slots in by the same `out_cache_loc` lookup. Each chunk writes its own token slots; **no cross-chunk accumulation needed.**

The DS write hook (the §3.4 G9 / Loop 3 M1 hook) becomes a per-chunk per-layer call:

```python
# At each chunk's set_mla_kv_buffer site:
token_label_buffer[layer.layer_id, forward_batch.out_cache_loc, :, :] = \
    project_k_to_label(k, channel_selection[layer.layer_id], channel_weights[layer.layer_id])
```

That's the entire M1 hook. No multi-chunk-per-page accumulation. No partial-page state. No "is this page complete yet" bookkeeping.

This is the design point sglang-last incidentally got right by using per-token slot indexing. We're back-porting that observation.

### 13.5 Selection math (replaces §3.5)

For request `r`, layer `l`, head `h`, decode-step query `q`:

```
label[l, t, h, d] = K[l, t, h, channel_selection[l, h, d]] * channel_weights[l, h, d]
                  (written once per token at KV-write time; per-tile FP8 dequant inline)

q_label[r, l, h, d] = q[r, l, h, channel_selection[l, h, d]] * channel_weights[l, h, d]

token_score[r, l, t] = max over h of (
    sum over d of q_label[r, l, h, d] * label[l, t, h, d]
)

token_score_reduced[r, l, t] = all_reduce_SUM_TP(token_score[r, l, t])

selected_tokens[r, l] = ascending(topk(token_score_reduced[r, l, ·], max_top_k_tokens))
                      ∪ hot_tokens[r]    # local-window: last N tokens
```

The TP `all_reduce(SUM)` (DEC-9) reduces a `[max_tokens]` scalar tensor per request per layer — bigger than `[max_pages]` (factor of 64) but still tiny: at 128K context that's 128K × 4 B = 512 KB / request / layer. Acceptable.

The output `selected_tokens[r, l]` is **token indices**, sorted ascending. These feed `transform_index_page_table_decode(selected_tokens, page_size=64)` (the existing NSA helper at `nsa/utils.py` or `nsa_indexer.py` neighborhood — same one NSA uses) to produce the page-aligned block_table FlashMLA reads.

### 13.6 Memory budget summary (replaces §3.3's page table block)

**Token-label cache** (per TP rank, head-sharded):
- Shape: `[num_layers, max_tokens, H_local, label_dim]`, dtype fp16.
- At V3.2 with 60 layers, H=128, TP=8 → H_local=16, label_dim=16, fp16: **30 KB / token / rank**.
- Budget at client workloads: 120 MB at 4K, 3.8 GB at 128K. Documented; not estimated at boot.
- Lifetime is identical to the KV cache — slot-indexed by `out_cache_loc`, no separate lifecycle hooks. Free-on-evict is handled by the existing KV allocator.

### 13.7 Hot-token rule (replaces page-level hot-page rule)

Force-include the last N tokens of the request (default N = 64, i.e. one page worth) regardless of score. This is the token-level equivalent of NSA's local-window override and matches the "recent context is always relevant" inductive bias.

No "active page is partially filled" edge case — at the token granularity, the active token is just the most recent index.

### 13.8 Where page_size=64 still matters

Only at the FlashMLA boundary:
1. `transform_index_page_table_decode(selected_tokens, page_size=64)` rounds token indices up to page boundaries for the `block_table`. Per the conversation: page_size=64 is preferred but not hard-required; the helper supports other sizes.
2. The KV pool itself stores K and V in pages of 64. The token-label buffer is allocated *parallel to* this pool, slot-indexed identically — but its conceptual unit is one token, not one page.

No DS code "knows about" pages. The page concept lives entirely in FlashMLA's KV layout and the existing NSA helper.

### 13.9 What this rotation deletes from earlier scope

These problems **disappear** with the rotation:

| Problem (page-level) | Disposition (token-level) |
|---|---|
| Chunked-prefill multi-chunk-per-page accumulation | ✂ Gone — per-token slot indexing inherits sglang's existing chunked prefill |
| Custom `page_table_adapter.py` (404 LOC) | ✂ Mostly deleted — call NSA's `transform_index_page_table_decode` |
| Page lifecycle hooks (`on_assign`/`on_free`/`on_evict`/`on_reuse`) | ✂ Gone — token-label cache shares the KV cache's allocator lifecycle |
| `valid_mask` per-page invalidation | ✂ Gone — KV pool's free-list is the truth |
| Within-page averaging quality delta (kernel_audit_memo §"Page granularity delta") | ✂ Gone — selection is at paper's natural granularity |
| Custom hot-page rule + "page partially filled" state | ✂ Simplified — hot-token rule mirrors NSA |
| Loop 2 Design C "unified page_table_1 Tensor" return-type contract | Replaced by token-indices return; both DS and NSA produce token indices, both go through `transform_index_page_table_decode` |

These were real engineering items in earlier scope. They become free-or-deleted with the rotation.

### 13.10 What this rotation adds back

Nothing major. The only new thing vs sglang-last is that DS still has to **write the token-label cache from FP8 KV inputs** (sglang-last's K was BF16). The FP8-aware write kernel from the page-level design (§3.4 G4) ports directly — it's per-page-mean code minus the page-mean reduction. Strictly simpler.

### 13.11 1M-context recoverability

If a future client ask brings 1M context into scope, the page-level design from §3.3 is recoverable from git history. The token-level → page-level migration would be:
1. Add the page-aware signature table (the §3.3 design).
2. Add a config knob `signature_granularity = "token" | "page"`.
3. At init, route to one or the other based on `max_context_length * H_local * label_dim * 2 < SIGNATURE_BUDGET_BYTES`.

The token-level design doesn't preclude the page-level one — it just defers it until somebody actually needs it.
