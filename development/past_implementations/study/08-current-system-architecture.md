# 08 — Current System Architecture: Double Sparsity on DeepSeek-V3.2 FP8 (as-built after Loop 6)

**Status:** AS-BUILT state of the standalone Double Sparsity (DS) path after **Loop 6** closed at its Minimum
Acceptable Scope (`.humanize/rlcr/2026-05-30_06-27-19`). Where [`07-mvp-proposed-architecture.md`](07-mvp-proposed-architecture.md)
was the *design intent* for the Loop-4 MVP, this doc records **what is actually implemented, measured, and
shipped on hardware** — the operating point, the data structures, the decode hot path, the kernel constraints,
the measured performance/recall, and what remains deferred. Read this first when picking up the branch.

**Branch:** `dev/double-sparsity-standalone`. **Hardware validated:** single node, 8×H200 (TP=8), V3.2 FP8.

---

## 1. TL;DR — what DS-on-V3.2 is today

DS replaces the per-step token-selection role of V3.2's native NSA `Indexer` with an **offline-calibrated
channel-importance projection + a runtime per-token signature table + a top-2048 selection**, hooked at
`DeepseekV2AttentionMLA._select_topk_indices`. Everything downstream (paged FlashMLA, FP8 KV cache, NSA's
quant/dequant) is untouched. DS is **opt-in and flag-gated**; **DSA is the production default**.

After Loop 6, DS **serves the client workload (4096 ISL / 512 OSL) within the tail-latency SLO at conc-16**
(P99 TTFT **13.13 s < 22**) at a lifted, full-context operating point, with **admission restored** by an
**int8-compacted signature table**. Two things remain characterized rather than strictly met: per-request
decode **TPS** (a structural decode-batch ceiling, DS ≤ DSA) and **long-context recall** (4K/16K/64K =
75/5/0% vs DSA 100% — the deferred Tier-2 R&D, now Loop 7).

---

## 2. Operating point (the lifted "Option B" point)

| Knob | Value | Notes |
|------|-------|-------|
| Topology | single node, TP=8 | DEC-5: multi-node deferred to its own loop |
| Model | DeepSeek-V3.2, FP8 (e4m3) | ~84 GB/rank weights |
| `page_size` | 64 | locked; page-size flexibility is roadmap §5 |
| KV cache | fp8 | |
| Attention backend | `flashmla_kv` prefill + decode | overlap-schedule + piecewise-cuda-graph **disabled** |
| CUDA graph | regular capture on | fixed shapes — central constraint on the selection path (§5) |
| Radix cache | **on**, via config-bound fixture | `--double-sparsity-radix-fixture-artifact` (no env override) |
| **Signature dtype** | **int8** (compact) | `SIGNATURE_DTYPE=int8`; fp16 is the default/fallback |
| **`mem_fraction_static`** | **0.7** | lifted from Loop-5's 0.6; the int8 table is what makes 0.7 boot+serve without generation-time OOM |
| `max_total_num_tokens` | **396096** | the admitted KV pool at this point |
| `context_len` | 163840 | drives `req_to_token.shape[1]` = the selection scan width (§5) |

This is the **fixed operating point** for all Loop-6 acceptance; Loop 7 inherits it as the baseline.

---

## 3. The footprint fix — compact int8 TokenLabelTable (the Loop-6 spine)

**Why it exists.** DS reserves a per-rank `TokenLabelTable` *on top of* the FP8 weights, allocated from
runtime headroom *after* weights + KV pool. Its size scales with the KV pool, so raising `mem_fraction_static`
grows both the pool *and* the table — a memory fixed point. At fp16 the table forced `mem_fraction_static`=0.6
(KV pool admits only ~14.5/24.6/35.7 of nominal conc 16/32/64 → requests queue → P99 TTFT 57.7/132.9/292.0 s
in Loop 5). Raising past 0.6 generation-OOM'd.

**Footprint formula** (`token_label_table.py`):
```
table_bytes_per_rank = num_layers_local × max_tokens × num_heads_local × label_dim × elem_size
V3.2 @ TP=8:           61          × max_tokens ×        16          ×    16     × elem_size
max_tokens = max_total_num_tokens + page_size(64)
```

**The lever (int8, DEC-4 primary):** symmetric **int8** signatures (`elem_size` 1 vs fp16's 2) at the **same
`label_dim`=16**, plus one per-`(layer, slot, head)` **fp16 scale** applied at scoring. Net:
`int8+scale = 0.5×(1 + 2/(16·1)) = 0.5625×` the fp16 table ≈ **1.78× smaller**. At the lifted point this is a
compact **~6.48 GB/rank** table (vs the fp16 table that gen-OOM'd), which is what lets DS boot and *serve* at
`mem_fraction_static`=0.7 with generation headroom. (`label_dim` narrowing was explicitly NOT chosen; the
page-level/two-stage structural lever was the escalation path if int8 had been predicted insufficient — it
was not needed.)

**Implemented surfaces (all int8-aware, fp16-default, fail-closed):**
- `token_label_table.py` — compact int8 storage + the per-`(layer,slot,head)` fp16 scale sidecar.
- `token_label_write.py` — **quantize-on-write** (fp16 signature → symmetric int8 + scale on page assign).
- `selection_kernel.py` — **scale-aware scoring** (applies the int8 scale at score time; §5).
- `serve_double_sparsity.sh` — `SIGNATURE_DTYPE` launcher env (config + launch-log echo; default fp16).
- **Radix fixture fail-closed:** `signature_dtype` is part of the radix fingerprint, so an fp16 fixture cannot
  authorize an int8 boot (and vice-versa); the radix capture records + compares per-`(layer,token)` scale
  SHAs; `startup_sanity_probe` is scale-aware.

**Selection-equivalence gate (int8 vs fp16):** top-k overlap@2048 **≥ 0.99** (binding); selected-token recall
and score-error distribution recorded as diagnostics. Real-mask NIAH non-regression: int8 DS recall ≥ the
fp16 Loop-5 baseline at every length (1024/1536 = 100, 4K = 85 ≥ 75, 16K = 5, 64K = 0).

---

## 4. Pipeline (write → select → attend)

```
prefill/decode page assign
        │
        ▼
 token_label_write.py  ──(FP8-scale-aware Triton; quantize-on-write fp16→int8+scale)──►  TokenLabelTable
        │                                                                                 [L, max_tokens, H_local, 16] int8
        │                                                                                 + scales [L, max_tokens, H_local] fp16
        ▼
DeepseekV2AttentionMLA._select_topk_indices            ◄── the one named hook site (replaces NSA Indexer)
        │
        ▼
 selection_kernel.py : retrieve_topk_graph_safe        ◄── graph-safe Triton path, per layer (×61) per decode step
        │   _logical_score_triton  (scale-aware scores over the scan width; R17 early-exit past seq_len)
        │   torch.topk(scores, 2048)                   ◄── the kernel-locked budget
        │   2nd topk / deterministic tie-break (R23)
        │   logical_to_physical
        ▼
 indices [.., 2048]  ──►  flashmla_kv DECODE kernel  (asserts indices.shape[-1] == dsa_index_topk == 2048)
        ▼
 paged FlashMLA over the selected 2048 tokens (unchanged)
```

Two artifacts ship per deployment: the **channel mask** (`/models/dsv32-fp8-channel-mask.safetensors`,
offline, content-hashed, startup sanity probe) and the **TokenLabelTable** (GPU tensor, allocator-owned,
written on page assign, invalidated on free/evict/abort).

---

## 5. Decode selection hot path + the CUDA-graph constraint (the perf axis)

Per decode step, **for every one of the 61 layers**, `retrieve_topk_graph_safe` runs:
1. `_logical_score_triton` — scores the request's tokens from the int8 signatures (scale-aware).
2. `torch.topk(scores, 2048)` — the dominant cost.
3. a 2nd topk / sort for the **deterministic tie-break**, then `logical_to_physical`.

**The over-scan.** The score/topk runs over `max_seq_len = ds_graph_state.max_seq_len =
req_to_token.shape[1] = model_config.context_len = 163840`, even though a client request is only ~4096 tokens.
Under **CUDA-graph fixed shapes** the topk **cannot skip** to the live width — the shape is captured at
163840. This full-width topk is the per-step selection floor.

**R17 score early-exit (shipped).** `_logical_score_kernel` (Triton) early-exits per token-block once
`tok_blk × TOKEN_BLOCK ≥ seq_len_i`, storing `-inf` for out-of-range blocks and skipping the per-head loads.
Bit-identical, CUDA-graph-safe — it removes the dead-block *scoring* cost but **not** the `torch.topk` width
(topk still scans 163840).

**R23 deterministic tie-break (shipped).** A shared `_topk_by_score_then_pos(vals, pos, k)` selects top-K by
**(score DESC, then logical position ASC)** via a stable position-asc sort then a stable score-desc argsort, so
the monolithic selector, the blocked oracle, and any future kernel agree **bit-identically including on finite
ties**. (As a full-argsort it is an *oracle*, slower than topk — a hot-path kernel would need a fast
position-asc-tie top-k.)

**The kernel ABI lock.** The shared `flashmla_kv` **decode** kernel asserts
`indices.shape[-1] == self.dsa_index_topk == 2048` (`dsa_backend.py`, `_forward_flashmla_kv`) at CUDA-graph
capture. `SGLANG_DS_ALLOW_TOPK_MISMATCH=1` does not bypass it. **DS top_k cannot widen beyond 2048 without a
new decode kernel** — the central Tier-2 (Loop 7) constraint.

**R24 design microbench (decisive, no full-context top-k design wins).** Timed at 61L/bs16/seq4096/maxlen163840:
A monolithic-163840 = 6.56 ms/step (→ implied conc-16 27.1 TPS); B live-region-4096 = 2.38 ms (→ 30.6, **but
this CAPS context to the live width**); C blocked bw=8192/pk=2048 = 8.50 ms (→ 25.7, **worse** than monolithic);
C′ torch-blocked = 12.33 ms (→ 23.4). **Conclusion: no graph-safe FULL-CONTEXT blocked-top-k design reaches
conc-16 ≥ 30** — under fixed shapes a two-stage merge processes `num_blocks × partial_k` candidates and two
topk passes cost more than one monolithic topk. The only design that reaches ≥ 30 caps the scan/merge to the
live region (= the bounded-context op-point, which was not adopted as the strict pass).
(`runs/20260530_dsv32_loop6/ac5_topk_design/`.)

---

## 6. Measured state (the client SLO bar: `development/CLIENT_SLOS.md` — P99 TTFT < 22 s AND ≥ 30 TPS/req)

Full-context Option-B point, np64 steady-state methodology (warmup 120 s / 300 s window), fail-closed verifier
that recomputes every headline from committed raw arrays (`runs/20260530_dsv32_loop6/ac5_fullctx/`):

| conc | P99 TTFT | TTFT SLO (<22 s) | per-req TPS | TPS SLO (≥30) |
|------|---------:|:----------------:|------------:|:-------------:|
| 16 | **13.13 s** | ✅ **PASS** | 24.9 | ❌ |
| 32 | 25.33 s | ❌ | 19.5 | ❌ |
| 64 | 77.90 s | ❌ | 17.3 | ❌ |

- **TTFT (the Loop-5 hard blocker) is fixed at conc-16** — the footprint→admission→TTFT spine works; conc-16
  meets the tail-latency SLO at **full context** (no context cap), with measured queue-vs-prefill attribution.
- **Per-req TPS is a structural decode-batch ceiling, not a DS defect.** TPS grows with decode batch; DS ≤ DSA
  on the same path, and **conc-64 ≥ 30 is unattainable even for DSA** (DSA = 46.1/37.0/**29.4** at conc
  16/32/64). No top-k kernel changes this at full context (§5).
- **AC-5 is closed DIRECTIONAL** (DEC-3, owner R24): the spine is validated and the strict misses are recorded
  with attribution — accepted MVP progress, **not** a strict/shippable all-concurrency pass.

**Recall (NIAH, the deferred axis):** DS 4K **75%** / 16K **5%** / 64K **0%** vs DSA **100%** everywhere at the
same 2048 budget + same kernel. Dense DS (seq ≤ 2048) = 100%; DS MMLU 89.00% == DSA. ⇒ decode is sound; the
gap is **selection quality + the 2048 cap**, not a bug. This is Loop-7 / Tier-2.

**Quality / servability:** AC-Q paired smoke PASS; AC-12 DS-fair gate PASS (MMLU Δ0.00, within-budget NIAH
100%); 64K `/generate` **serves** at the lifted point (servability win, recall separate).

---

## 7. What is NOT done (deferred, with owners)

| Item | Status | Where |
|------|--------|-------|
| **DS long-context recall (Tier-2 / AC-10)** | **deferred — Loop 7 high priority** | adjustable-`top_k` decode kernel (relax the ABI lock) + learned/query-aware selector; gate already open (`ds_on_v32_decision.md`). See [`development/loop7/draft.md`](../../loop7/draft.md). |
| Strict all-concurrency client SLO (≥30 TPS @ every conc) | downstream | structural (DS ≤ DSA; conc-64 ≥30 unattainable even for DSA) — operating-point/DSA-side, not recall R&D |
| 128k ISL servability + recall | roadmap §6 Loop 7 | extends the 64K servability work |
| Multi-node / TP scaling | deferred (DEC-5) | dedicated future loop; SLO numbers unchanged across topologies |
| Page-size flexibility, env-gate cleanup, CI registration, upstreaming/PR hygiene, mask provenance | roadmap §5 | productionization tech-debt |
| nvfp4/mxfp4 weights, knob×DS compat matrix, GLM-5.1 | roadmap §6/§9/§10 | downstream client requirements |

---

## 8. Key files (re-derivation index)

- **Footprint lever / table:** `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py`
- **Quantize-on-write:** `…/double_sparsity/token_label_write.py`
- **Selection (scoring + topk + tie-break):** `…/double_sparsity/selection_kernel.py`
  (`retrieve_topk_graph_safe`, `_logical_score_kernel` R17 early-exit, `_topk_by_score_then_pos` R23 tie-break,
  `blocked_topk_sequence_order` R22 exact oracle)
- **Kernel ABI lock (Tier-2 target):** `indices.shape[-1] == dsa_index_topk` in
  `python/sglang/srt/layers/attention/dsa_backend.py` (`_forward_flashmla_kv`)
- **Launcher:** `development/serve_double_sparsity.sh` (`SIGNATURE_DTYPE`); **bench:** `development/benchmark.sh`
- **Channel mask (on disk, untracked):** `/models/dsv32-fp8-channel-mask.safetensors`
- **Loop-6 evidence:** `runs/20260530_dsv32_loop6/` — `ds_on_v32_decision.md` (strategic gate),
  `footprint_feasibility.md`, `ac5_fullctx/` (verifier + arrays + attribution), `ac5_topk_design/` (R24
  microbench), `ac8_servability/` (64K)
- **Loop-6 process record:** `.humanize/rlcr/2026-05-30_06-27-19/` (goal-tracker, round summaries/reviews)
- **Client bar:** `development/CLIENT_SLOS.md`; **roadmap:** `development/roadmap.md`
- **Design intent (prior):** [`07-mvp-proposed-architecture.md`](07-mvp-proposed-architecture.md),
  [`06-proposed-architecture.md`](06-proposed-architecture.md)
