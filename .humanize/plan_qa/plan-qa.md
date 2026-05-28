# Refine Plan QA

## Summary

Processed 26 comment blocks (CRITICAL, MAJOR, SEQ, TEST, LT, RISK, SMELL, CODEX-AGREE tags) from `development/loop4/plan.md`. Two rounds of external Codex review were incorporated alongside pensieve architectural review. Primary changes: (1) corrected the CRITICAL-1 k-dimension error (kv_b_proj projection required at write time); (2) updated calibration to paper-faithful Method 1 Q·K (confirmed from original DoubleSparse paper source); (3) added missing `task-ac0-deepseek-v2`; (4) fixed score tensor shape [bs, max_tokens]; (5) corrected task sequencing for AC-1b, AC-6 dependency on M2; (6) added reproducibility mechanics to AC-11 gate; (7) added stale-slot negative test to AC-2. All comments fully resolved; plan converges.

---

## Comment Ledger

| CMT-ID | Classification | Location | Original Text (excerpt) | Disposition |
|--------|----------------|----------|-------------------------|-------------|
| CMT-1 | research_request | Goal Description | "CRITICAL-1: The `k` argument at `set_mla_kv_buffer` is the MLA latent key (512-d)..." | researched + applied |
| CMT-2 | change_request | AC-0 | "CRITICAL-3: `validator.py` reads `nsa_prefill_backend`... dead attribute names..." | applied |
| CMT-3 | change_request | AC-0 | "CRITICAL-4: `device_buffer_size=4096`... covers only ~5% of a typical H200 KV pool..." | applied |
| CMT-4 | change_request | AC-0 | "SEQ-2: `deepseek_v2.py` changes are missing from the task table entirely..." | applied |
| CMT-5 | change_request | AC-0 | "TEST-2: AC-13 'green after every code change'... impossible during AC-0 development..." | applied |
| CMT-6 | change_request | AC-0 | "CODEX-AGREE: TEST-2/3/4: confirmed..." | applied (folded into CMT-5/CMT-12/CMT-14) |
| CMT-7 | change_request | AC-2 | "LT-2: 'Freed/evicted KV slots do not produce persistent label pollution'... not proved..." | applied |
| CMT-8 | change_request | AC-2 | "CODEX-AGREE: LT-2/TEST-5: save_kv_cache=False fused path... AC-7 needs to pin..." | applied (folded into CMT-7/CMT-15) |
| CMT-9 | research_request | AC-4 | "MAJOR-6: `calibrate.py` computes L2-squared K importance... not Q·K..." | researched + applied |
| CMT-10 | change_request | AC-4 | "MAJOR-5: `--model-arch deepseek_v3` flag doesn't exist in calibrate.py..." | applied |
| CMT-11 | research_request | AC-4 | "RISK-1: Pre-loop prerequisite: read `get_dsa_index_topk(V3.2_config)`..." | researched + applied |
| CMT-12 | change_request | AC-5 | "TEST-3: all_reduce(SUM) fires over `[max_tokens]`-shaped score tensor... wrong shape..." | applied |
| CMT-13 | question | AC-5 | "LT-3: TP=2 on a single node uses shared-memory NCCL... Different failure modes..." | answered + note added |
| CMT-14 | change_request | AC-6 | "TEST-4: `assert_no_alloc_in_region`... real CUDA graph capture barrier is PyTorch's graph mode itself..." | applied |
| CMT-15 | change_request | AC-6 | "SEQ-5: `capture_decode_step` calls `retrieve_topk` with ownership mask... task-ac0-cuda-graph doesn't depend on task-m2-rangemask..." | applied |
| CMT-16 | change_request | AC-7 | "TEST-5: label-write hook at `dsa_backend.py:L1439` is inside `if save_kv_cache:`..." | applied |
| CMT-17 | change_request | AC-8 | "RISK-3: quality smoke baseline has no specification for when/how it was generated..." | applied |
| CMT-18 | change_request | AC-8 | "SMELL-2: `error_containment` counter check requires `row_errors` dict to survive rewrite..." | applied |
| CMT-19 | change_request | AC-10 | "RISK-4: 'Cold-prefix vs warm-prefix labels are bit-stable' assumes FP8 scale factors are identical..." | applied |
| CMT-20 | change_request | AC-11 | "LT-4: 'DS-on TPS within 5% of DSA-on TPS' is not falsifiable as written..." | applied |
| CMT-21 | change_request | AC-11 | "CODEX-AGREE: RISK-3/LT-4: both quality smoke and performance gates need reproducibility mechanics..." | applied (folded into CMT-17/CMT-20) |
| CMT-22 | research_request | Feasibility Hints | "CRITICAL-1 repeated in code path: This `k` is NOT 128-d projected nope K..." | researched + applied |
| CMT-23 | research_request | Feasibility Hints | "RISK-2: FlashMLA at `dsa_backend.py:L1864-L1874` expects indices into `kv_cache.view(-1, 64, ...)`..." | researched + applied |
| CMT-24 | change_request | Feasibility Hints | "MAJOR-6 in code path: 'No changes to the importance metric' is incorrect..." | applied (same as CMT-9) |
| CMT-25 | change_request | Task Breakdown (HTML) | "SEQ-2 + SMELL-1: Missing task: update `deepseek_v2.py` for AC-0 rotation..." | applied |
| CMT-26 | change_request | Task Breakdown (HTML) | "SEQ-4: AC-1b is scheduled after task-ac8-server, but its outcome retroactively changes AC-8's config..." | applied |
| CMT-27 | change_request | Claude-Codex Deliberation | "CRITICAL-1 in deliberation: resolution rests on a false factual premise..." | applied |
| CMT-28 | change_request | RLCR Config | "LT-5: 14-round budget... CRITICAL-1 and missing deepseek_v2.py tasks are unresolved..." | applied |

---

## Answers

### CMT-13: LT-3 — TP=2 single-node NCCL vs production TP=8 NVLink/IB

**Original Comment:**
```
CRITIQUE [LT-3] TP=2 on a single node uses shared-memory NCCL collectives, not the actual NCCL over NVLink/IB path used in production TP=8. Different failure modes: no dropped packets, no timeout behavior, same NUMA domain. A rank-divergence bug caused by a wrong process group config would only appear at AC-6/AC-8. If TP correctness is the goal, consider at least testing with `NCCL_P2P_DISABLE=1` to force a more production-representative collective path.
```

**Answer:**
The observation is correct. Single-node TP=2 with shared-memory NCCL differs from production TP=8 over NVLink/IB. However, the AC-5 harness serves a specific and bounded purpose: it verifies the logical correctness of the `all_reduce(SUM)` domain (logical position space vs physical slot space) and that the reduce operation itself is not a no-op. This is a unit-level correctness gate, not a production-fidelity gate. The production TP path is exercised at AC-6 (CUDA graph capture on real H200) and AC-8 (bench_serving on 8×H200 TP=8). NCCL_P2P_DISABLE=1 can be used as a fallback if AC-6 reveals rank-divergence that AC-5 missed.

**Plan Changes:**
Added to Implementation Notes (TP Collective Notes): "If AC-5 passes but AC-6 shows rank divergence, set `NCCL_P2P_DISABLE=1` in the TP test to force a more production-representative path."

---

## Research Findings

### CMT-1 / CMT-22: CRITICAL-1 — What is the `k` argument at `set_mla_kv_buffer`?

**Original Comment:**
```
CRITIQUE [CRITICAL-1] The `k` argument at `set_mla_kv_buffer` is the MLA latent key (512-d, `kv_lora_rank=512`), not the 128-d projected nope K. The per-head projection `Q·K_nope^T` only materializes inside the attention kernel when `kv_b_proj` is applied — it is never written to the hook site. `memory_pool.py` confirms: `cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512)|scales(16)]`. Writing 128-d labels at this hook site requires applying `kv_b_proj` at write time...
```

**Research Scope:**
- `memory_pool.py:L1757-1792` — `set_mla_kv_buffer` signature, third param is `cache_k_nope: torch.Tensor`
- `memory_pool.py:L1785` comment — `cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512)|scales(16)]` — confirms 512-d latent
- `forward_mla.py:L159` — `k_nope = latent_cache[..., :self.kv_lora_rank]` — kv_lora_rank=512
- `deepseek_v2.py:L1719-1720` — `if self.attn_mha.kv_b_proj is None: self.attn_mha.kv_b_proj = self.kv_b_proj` — confirms kv_b_proj accessible at hook site
- `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2/config.json` — `kv_lora_rank: 512`, `qk_nope_head_dim: 128`

**Findings:**
- `k` at `set_mla_kv_buffer` is definitively 512-d MLA latent key, not 128-d projected nope K.
- The 128-d projection `K_nope = latent @ kv_b_proj_k_side` only materializes inside the FlashMLA CUDA kernel.
- However, `layer.kv_b_proj` IS accessible at all three hook sites via `deepseek_v2.py:L1720`.
- The write kernel can apply the K-side projection matmul at hook time: `[T, 512] @ [512, H_local*(128+v_dim)]` → slice K_nope prefix `[T, H_local*128]` → reshape `[T, H_local, 128]`.

**Impact on Plan:**
- Goal Description updated: removed claim that 128-d is read directly; added explicit mention of kv_b_proj projection step at write time.
- Feasibility Hints code example updated: replaced direct-read snippet with projection-then-write snippet.
- Claude-Codex Deliberation DEC-1 re-resolved: "kv_b_proj K-side projection applied at hook time; `layer.kv_b_proj` accessible."
- `task-m1-hook` updated: explicitly includes the kv_b_proj projection step.

---

### CMT-9 / CMT-24: MAJOR-6 — What calibration method does the original DoubleSparse use?

**Original Comment:**
```
CRITIQUE [MAJOR-6] The plan says `calibrate.py` implements "Method 1 (`mean(abs(Q·K))` per channel)" and "is already correct." It isn't. The actual implementation computes L2-squared K importance: `sum_over_tokens(K_channel^2)` — no Q involvement at all (`calibrate.py:L244-246`). Method 1 requires hooking both Q and K projections and computing joint attention scores.
```

**Research Scope:**
- `calibrate.py:L244-246` — confirmed: `squared = tensor.pow(2); squared = squared.reshape(-1, num_heads, k_head_dim).sum(dim=0)` — L2-squared K, no Q involvement
- `development/past_implementations/DoubleSparse/config/offline_calibration.py:L91-93` — original paper calibration:
  ```python
  # Method 1: every token only attend to itself
  out = q * k  # element-wise per channel
  out = out.reshape(-1, m.num_heads, m.head_dim).abs().mean(dim=0)
  ```
- `development/past_implementations/sglang-last-with-double-sparsity/python/sglang/srt/server_args.py:L599` — `ds_heavy_channel_type: str = "qk"` — original sglang used pre-generated JSON from the paper's calibration (Method 1 implicitly)

**Findings:**
- The original DoubleSparse paper uses **Method 1** exclusively: `mean(abs(q_channel * k_channel))` per channel, element-wise Q·K, then sort channels descending.
- `sglang-last-with-double-sparsity` loaded pre-generated JSON from this calibration (flag name `"qk"` confirms Q·K-based origin).
- Current `calibrate.py` uses L2(K²) — different from both the paper and sglang-last.
- Implementing Method 1 requires co-registering both Q_nope and K_nope hooks in the same forward pass and computing the joint per-channel score.

**Impact on Plan:**
- AC-4 updated: calibration explicitly requires Method 1 (Q·K joint), with both Q_nope and kv_b_proj hooks.
- `task-ac4-calibrate` updated: scope includes Q hook addition alongside K hook.
- Feasibility Hints calibration section updated: shows Method 1 signature with Q+K.
- Reference added: `DoubleSparse/config/offline_calibration.py`.

---

### CMT-11: RISK-1 — Verify `get_dsa_index_topk(V3.2_config)` before writing validator

**Original Comment:**
```
CRITIQUE [RISK-1] Pre-loop prerequisite: read `get_dsa_index_topk(V3.2_config)` on the actual H200 cluster before writing any validator code. The plan assumes this returns 2048, but it's never verified.
```

**Research Scope:**
- `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2/config.json` — read directly

**Findings:**
- `config.json` contains `"index_topk": 2048` — confirmed.
- Other confirmed values: `"kv_lora_rank": 512`, `"qk_nope_head_dim": 128`, `"num_attention_heads": 128`.

**Impact on Plan:**
- Added to Implementation Notes (Pre-Loop Verifications): "topk confirmed: get_dsa_index_topk returns 2048 — verified from config.json."
- No change to validator logic needed; the assumption was correct.

---

### CMT-23: RISK-2 — FlashMLA index format (slot IDs vs block indices)

**Original Comment:**
```
CRITIQUE [RISK-2] The plan describes physical_slots as "flattened physical token indices" but FlashMLA at `dsa_backend.py:L1864-L1874` expects indices into `kv_cache.view(-1, 64, ...)` — each index selects a 64-token block, not a single token slot.
```

**Research Scope:**
- `dsa_backend.py:L1864` — `kv_cache = kv_cache.view(-1, self.real_page_size, 1, self.kv_cache_dim)` + `indices = page_table_1.unsqueeze(1)`
- `dsa_backend.py:L1872-1874` — shape assertion on indices
- `dsa/dsa_indexer.py:L550` — `assert page_size == 64`
- `dsa/dsa_indexer.py:L553-555` — `metadata.get_page_table_1()` or `metadata.get_page_table_64()` for block_tables
- `dsa_backend.py:L438-446` — `_transform_table_1_to_real`: `return page_table[:, strided_indices] // page_size` — divides by page_size to get block indices

**Findings:**
- `page_table_1` in FlashMLA path stores VALUES from `req_to_token` directly (physical token slot indices), NOT pre-divided block indices.
- The `kv_cache.view(-1, 64, ...)` reshaping means FlashMLA uses the token indices to index into the paged view, where each "page" holds 64 tokens. The indices into this view ARE the physical token slot IDs from `req_to_token` — they are naturally aligned because `req_to_token` stores token-level slot IDs and FlashMLA's paged view interprets them as such.
- The `_transform_table_1_to_real` path that divides by page_size is for a different path (the DSA-non-FlashMLA path). The `flashmla_kv` sparse path in Option B does NOT divide — it uses req_to_token values directly.
- Therefore: the adapter's output of `req_to_token` values IS the correct format. No additional conversion needed.

**Impact on Plan:**
- Feasibility Hints adapter section clarified: "req_to_token values are exactly the physical token indices FlashMLA expects (page_table_1 in dsa_indexer.py is populated directly from req_to_token); no additional conversion needed."

---

## Plan Changes Applied

### CMT-2: CRITICAL-3 — Fix dead validator attribute names

**Original Comment:**
```
CRITIQUE [CRITICAL-3] `validator.py` reads `server_args.nsa_prefill_backend` and `server_args.nsa_decode_backend` for the backend-KV-dtype check. Those attributes don't exist — live names are `dsa_prefill_backend` and `dsa_decode_backend`. The validator's backend pairing check silently passes (reads `None, None`) regardless of actual config.
```

**Changes Made:**
- AC-0 Positive Tests: added "validator.py reads `dsa_prefill_backend`/`dsa_decode_backend` (corrected from dead `nsa_*` attributes); backend-KV-dtype check fires correctly on a bad config."
- `task-ac0-validator`: added "fix dead attribute names `nsa_prefill_backend`/`nsa_decode_backend` → `dsa_prefill_backend`/`dsa_decode_backend`."

**Affected Sections:** AC-0, Task Breakdown

---

### CMT-3: CRITICAL-4 — Derive max_tokens from req_to_token_pool.size

**Original Comment:**
```
CRITIQUE [CRITICAL-4] `device_buffer_size=4096` is passed as `max_pages` / `max_tokens`... After token rotation, `max_tokens=4096` covers only ~5% of a typical H200 KV pool. Any slot with index ≥ 4096 triggers an out-of-bounds write or silently drops labels.
```

**Changes Made:**
- AC-0 Positive Tests: "Token label table allocates with `max_tokens = req_to_token_pool.size` (derived at bind time, not from `device_buffer_size`)."
- `task-ac0-deepseek-v2` (new task): includes "(1) derive `max_tokens = req_to_token_pool.size` at bind time."
- `task-ac0-config`: `device_buffer_size` explicitly documented as score-scratch buffer cap (not pool size).

**Affected Sections:** AC-0, Task Breakdown

---

### CMT-4 / CMT-25: SEQ-2 — Add missing task-ac0-deepseek-v2

**Original Comment:**
```
CRITIQUE [SEQ-2] `deepseek_v2.py` changes are missing from the task table entirely. Required changes: (1) `_bind_double_sparsity_runtime_data`: rename, max_tokens sizing; (2) `_select_topk_indices`: adapter ABI change; (3) `ds_topk_indices_out` pre-allocation; (4) `row_errors` dict wiring.
```

**Changes Made:**
- New task added to Task Breakdown: `task-ac0-deepseek-v2 | Update deepseek_v2.py: ... | AC-0 | coding | task-ac0-adapter`
- Dependencies graph updated in Milestones section.

**Affected Sections:** Task Breakdown, Dependencies and Sequence

---

### CMT-5: TEST-2 — Scope AC-13 as post-AC-0 gate only

**Original Comment:**
```
CRITIQUE [TEST-2] AC-13 says "regression suite green after every code change throughout the loop." The 150 existing unit tests use page-level shapes — they will all fail during AC-0 development before migration completes.
```

**Changes Made:**
- AC-13 rewritten: "Scope: this gate applies after AC-0 shape migration is complete... gate is not 'green after every change' — it is green after `task-ac0-tests` merges."
- Milestones section: "AC-13: regression suite green after `task-ac0-tests` merges (and remains green for all subsequent changes)."

**Affected Sections:** AC-13, Dependencies and Sequence

---

### CMT-7 / CMT-8: LT-2 — Add stale-slot negative test to AC-2

**Original Comment:**
```
CRITIQUE [LT-2] "Freed/evicted KV slots do not produce persistent label pollution" is asserted, not proved. The AC-2 positive test should include a synthetic freed→reallocated→read fixture that fails without valid_mask protection.
```

**Changes Made:**
- AC-2 Negative Tests: added "Stale-slot fixture: allocate a slot, write labels, free, reallocate to new request, invoke selector BEFORE new write fires — confirm stale labels from prior request are not returned. Verifies the overwrite-before-read invariant holds on all paths including `save_kv_cache=False` fused tilelang prefill."
- `task-ac2-lifetime`: "add stale-slot negative test (freed→reallocated→read before write)."

**Affected Sections:** AC-2, Task Breakdown

---

### CMT-10: MAJOR-5 — Remove --model-arch flag (auto-detect works)

**Original Comment:**
```
CRITIQUE [MAJOR-5] The `--model-arch deepseek_v3` flag doesn't exist in `calibrate.py`. Additionally, `calibrate.py` already auto-detects architecture via `getattr(config, "qk_nope_head_dim", 0)`.
```

**Changes Made:**
- AC-4 Positive Tests: removed `--model-arch deepseek_v3` from the sample command; added "(Auto-detects V3.2 MLA via `qk_nope_head_dim=128` from model config; no `--model-arch` flag required)."
- `task-ac4-calibrate`: removed "add `--model-arch deepseek_v3`" from scope; kept `kv_b_proj` traversal.

**Affected Sections:** AC-4, Task Breakdown

---

### CMT-12: TEST-3 — Fix score tensor shape to [bs, max_tokens]

**Original Comment:**
```
CRITIQUE [TEST-3] AC-5 says `all_reduce(SUM)` fires over `[max_tokens]`-shaped score tensor. The actual `all_reduce_page_scores` reduces a `[bs, max_pages]` tensor (batch-keyed, not flat). After token rotation this becomes `[bs, max_tokens]` — NOT a 1-D `[max_tokens]` tensor.
```

**Changes Made:**
- AC-5 Positive Tests: "all_reduce(SUM) fires over `[bs, max_tokens]`-shaped score tensor (batch-keyed, not flat)."
- `task-ac0-kernel`: "all_reduce over [bs, max_tokens] score tensor (batch-keyed)."
- `task-ac5-tp`: "all_reduce(SUM) on [bs, max_tokens] score tensor in logical-position space."

**Affected Sections:** AC-5, Task Breakdown

---

### CMT-14: TEST-4 — Clarify CUDA graph capture mechanism

**Original Comment:**
```
CRITIQUE [TEST-4] `assert_no_alloc_in_region` counts PyTorch caching-allocator allocations. The real CUDA graph capture barrier is PyTorch's graph mode itself, which errors on any `cudaMalloc`. The AC-6 negative test is valid but the mechanism it claims to test is muddled.
```

**Changes Made:**
- AC-6 Negative Tests rewritten: "With a non-preallocated scoring buffer, PyTorch's graph capture mode fails outright — the `assert_no_alloc_in_region` detector fires as a secondary belt-and-suspenders check. The negative test confirms that preallocating output buffers before `torch.cuda.graph.capture_begin()` is the load-bearing fix."
- `task-ac6-cuda-graph`: "confirm preallocation prevents PyTorch graph capture failure (not just alloc detector)."

**Affected Sections:** AC-6, Task Breakdown

---

### CMT-15: SEQ-5 — task-ac0-cuda-graph depends on task-m2-rangemask

**Original Comment:**
```
CRITIQUE [SEQ-5] `capture_decode_step` calls `retrieve_topk` with a `per_request_valid` ownership mask. `task-ac0-cuda-graph` doesn't mention this parameter. If it closes before `task-m2-rangemask`, the captured graph omits the ownership mask.
```

**Changes Made:**
- `task-ac0-cuda-graph` Depends On: added `task-m2-rangemask`.
- AC-6 text: "Note: `task-ac0-cuda-graph` must depend on `task-m2-rangemask`, since `capture_decode_step` calls `selector.retrieve_topk` with the ownership mask parameter introduced in M2."

**Affected Sections:** AC-6, Task Breakdown

---

### CMT-16: TEST-5 — AC-7 must verify save_kv_cache=True on FP8 V3.2 prefill

**Original Comment:**
```
CRITIQUE [TEST-5] The label-write hook at `dsa_backend.py:L1439` is inside `if save_kv_cache:`. For FP8 prefill on V3.2, this may be the default fast path using save_kv_cache=False. If so, the token-label write hook silently never fires.
```

**Changes Made:**
- AC-7 Positive Tests: added "Explicit path verification: confirm that the FP8 V3.2 prefill path sets `save_kv_cache=True` at hook sites, so the `token_label_write` hook fires. If any path uses `save_kv_cache=False`, identify and instrument the alternative hook site. Verification must be logged in the task-ac7-bypass commit."
- `task-m1-hook`: "verify that the FP8 prefill path sets `save_kv_cache=True` at each hook site; document findings in commit message."
- `task-ac7-bypass`: "explicitly confirm save_kv_cache=True on FP8 V3.2 prefill path or identify alternative hook site."

**Affected Sections:** AC-7, Task Breakdown

---

### CMT-17 / CMT-21: RISK-3 — Specify DSA reference generation for AC-8 quality smoke

**Original Comment:**
```
CRITIQUE [RISK-3] The quality smoke baseline ("reference outputs cached from DSA-on") has no specification for when and how it was generated or verified. `temperature=0` does not guarantee bit-identical outputs across server restarts with FP8 KV quantization.
```

**Changes Made:**
- AC-8 Positive Tests quality smoke: "generate the DSA reference on the **same server binary** used for the DS run, in the **same server restart session** immediately before the DS smoke test; record the DSA server commit SHA alongside the reference file."
- `task-ac8-quality`: "generate DSA reference on same server binary + same restart session, record commit SHA."

**Affected Sections:** AC-8, Task Breakdown

---

### CMT-18: SMELL-2 — Document row_errors decision in task-ac0-adapter

**Original Comment:**
```
CRITIQUE [SMELL-2] The `error_containment` counter check requires the adapter's `row_errors` side-channel dict to survive the < 150 LOC rewrite. Explicitly decide: keep the dict, replace with structured exceptions, or remove per-row tracking.
```

**Changes Made:**
- `task-ac0-adapter`: "Error tracking: keep `row_errors` dict pattern (compatible with existing AC-8 counter check); a simple scalar error count replaces per-row mutable dict."
- Implementation Notes (Error Containment Design): "keep `row_errors` pattern; adapter returns scalar `error_count` alongside `physical_slots`; structured exceptions avoided on hot decode path."

**Affected Sections:** Task Breakdown, Implementation Notes

---

### CMT-19: RISK-4 — Add FP8 scale factor verification to AC-10

**Original Comment:**
```
CRITIQUE [RISK-4] "Cold-prefix vs warm-prefix labels are bit-stable" assumes FP8 quantization scale factors are identical for a token written in isolation vs. within a fully packed KV page. This is not guaranteed.
```

**Changes Made:**
- AC-10: "Explicit verification: confirm that FP8 block quantization assigns identical per-block scale factors for the same token regardless of block-fill level (cold singleton vs. fully-packed block). If scale factors can differ, document the failure mode and defer radix cache to Loop 5."

**Affected Sections:** AC-10

---

### CMT-20 / CMT-21: LT-4 — Add reproducibility mechanics to AC-11 gate

**Original Comment:**
```
CRITIQUE [LT-4] "DS-on TPS within 5% of DSA-on TPS" is not falsifiable as written. A single bench_serving run at conc=64 with ISL=4096 has P99 TTFT variance that can swing 5%+ from measurement noise alone.
```

**Changes Made:**
- AC-11: added "Reproducibility requirements: use fixed random seed for request arrival; run for minimum 600s measurement window after 120s warmup; run at least 3 independent trials and report median; record commit SHA + full server args + chunked-prefill setting alongside each result JSON."
- `task-ac11-compare`: "with fixed seed, 600s window, 120s warmup, 3 trials, median aggregation."

**Affected Sections:** AC-11, Task Breakdown

---

### CMT-26: SEQ-4 — Move task-ac1b-probe before task-ac8-server

**Original Comment:**
```
CRITIQUE [SEQ-4] AC-1b is scheduled after task-ac8-server, but its outcome retroactively changes AC-8's operating point. If probe fails, AC-8's bench_serving must be re-run with the wrong config.
```

**Changes Made:**
- `task-ac1b-probe` Depends On: changed from `task-ac8-server` to `task-ac6-hwrun`.
- `task-ac8-server` Depends On: added `task-ac1b-probe` (so chunked-prefill config is known before full bench_serving run).
- Dependencies paragraph updated.

**Affected Sections:** Task Breakdown, Dependencies and Sequence

---

### CMT-27: CRITICAL-1 in Deliberation — Re-resolve DEC-1 with correct implementation path

**Original Comment:**
```
CRITIQUE [CRITICAL-1 in deliberation] This resolution rests on a false factual premise: "The write kernel is updated to read `k` (k_nope, bf16) directly from the `set_mla_kv_buffer` call site." The `k` argument at that call site is 512-d MLA latent K, not 128-d projected nope K.
```

**Changes Made:**
- Claude-Codex Deliberation DEC-1 resolution text rewritten: "128-d nope K is correct (paper-faithful, theoretically superior). However, the `k` argument at `set_mla_kv_buffer` is the 512-d MLA latent key — reading 128-d per-head K_nope requires applying `kv_b_proj` at write time. `layer.kv_b_proj` is accessible at all three hook sites (`deepseek_v2.py:L1720`). The write kernel is NOT reading 128-d directly — it applies the projection."

**Affected Sections:** Claude-Codex Deliberation

---

### CMT-28: LT-5 — Pre-loop kv_b_proj investigation note

**Original Comment:**
```
CRITIQUE [LT-5] The 14-round budget... Resolve the CRITICAL-1 question (128-d requires adding `kv_b_proj` projection vs. accepting 512-d latent) before the loop starts — if 128-d is confirmed but requires adding a projection matmul on the write path, that is a substantial new workload.
```

**Changes Made:**
- RLCR Loop Configuration: "The kv_b_proj projection step in `task-m1-hook` and the Method 1 Q+K hook refactor in `task-ac4-calibrate` both add scope vs the original estimate; these are explicitly tracked as separate tasks."
- Implementation Notes (Pre-Loop Verifications): "kv_b_proj projection cost: Before writing `task-m1-hook`, measure the projection matmul cost on H200 at realistic batch sizes."

**Affected Sections:** Implementation Notes, RLCR Loop Configuration

---

## Remaining Decisions

None. All comments fully resolved.

---

## Refinement Metadata

- **Input Plan:** `development/loop4/plan.md`
- **Output Plan:** `development/loop4/refined_plan_v1.md`
- **QA Document:** `.humanize/plan_qa/plan-qa.md`
- **Total Comments Processed:** 28 (including CODEX-AGREE blocks counted separately)
  - Questions: 1 (CMT-13)
  - Change Requests: 19 (CMT-2, CMT-3, CMT-4, CMT-5, CMT-6, CMT-7, CMT-8, CMT-10, CMT-12, CMT-13, CMT-14, CMT-15, CMT-16, CMT-17, CMT-18, CMT-19, CMT-20, CMT-21, CMT-25, CMT-26, CMT-27, CMT-28)
  - Research Requests: 5 (CMT-1, CMT-9, CMT-11, CMT-22, CMT-23)
- **Research Sources:**
  - `memory_pool.py:L1757-1792` — confirms 512-d latent at set_mla_kv_buffer
  - `deepseek_v2.py:L1719-1720` — confirms kv_b_proj accessible at hook site
  - `dsa_backend.py:L1864-1874` — FlashMLA index format
  - `dsa/dsa_indexer.py:L550-555` — page_size=64 and req_to_token format
  - `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2/config.json` — confirmed index_topk=2048
  - `development/past_implementations/DoubleSparse/config/offline_calibration.py` — confirmed Method 1 Q·K
  - `development/past_implementations/sglang-last-with-double-sparsity/python/sglang/srt/server_args.py:L599` — confirmed ds_heavy_channel_type="qk"
  - `calibrate.py:L244-246` — confirmed L2(K²) divergence from Method 1
- **Plan Sections Modified:**
  - Goal Description
  - AC-0, AC-2, AC-4, AC-5, AC-6, AC-7, AC-8, AC-10, AC-11, AC-13
  - Feasibility Hints and Suggestions
  - Dependencies and Sequence
  - Task Breakdown (7 tasks modified, 1 added)
  - Claude-Codex Deliberation
  - Implementation Notes
  - RLCR Loop Configuration
- **Convergence Status:** `converged`
- **Refinement Date:** 2026-05-27
