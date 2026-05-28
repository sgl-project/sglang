Read and execute below with ultrathink

## Goal Tracker Setup (REQUIRED FIRST STEP)

Before starting implementation, you MUST initialize the Goal Tracker:

1. Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md
2. If the "Ultimate Goal" section says "[To be extracted...]", extract a clear goal statement from the plan
3. If the "Acceptance Criteria" section says "[To be defined...]", define 3-7 specific, testable criteria
4. Populate the "Active Tasks" table with MAINLINE tasks from the plan, mapping each to an AC and filling Tag/Owner
5. Record any already-known side issues in either "Blocking Side Issues" or "Queued Side Issues"
6. Write the updated goal-tracker.md

## Round Contract Setup (REQUIRED BEFORE CODING)

Before starting implementation, create @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-0-contract.md with:

1. **One mainline objective** for this round
2. **Target ACs** (1-2 ACs only)
3. **Blocking side issues in scope** for this round
4. **Queued side issues out of scope** for this round
5. **Round success criteria**

Use this contract to keep the round focused. Do NOT let non-blocking bugs or cleanup work replace the mainline objective.

**IMPORTANT**: The IMMUTABLE SECTION can only be modified in Round 0. After this round, it becomes read-only.

---

## Implementation Plan

For all tasks that need to be completed, please use the Task system (TaskCreate, TaskUpdate, TaskList).

Every task MUST start with exactly one lane tag:
- `[mainline]` for plan-derived work that directly advances the round objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` tasks are the primary success condition for the round
- `[blocking]` tasks may be resolved in the round only if they truly block mainline progress
- `[queued]` tasks must NOT become the round objective and do NOT need to be cleared before moving on
- If a new issue is not blocking the current objective, tag it `[queued]` and keep moving on the mainline

## Task Tag Routing (MUST FOLLOW)

Each task must have one routing tag from the plan: `coding` or `analyze`.

- Tag `coding`: Claude executes the task directly.
- Tag `analyze`: Claude must execute via `/humanize:ask-codex`, then integrate Codex output.
- Keep Goal Tracker "Active Tasks" columns **Tag** and **Owner** aligned with execution (`coding -> claude`, `analyze -> codex`).
- If a task has no explicit tag, default to `coding` (Claude executes directly).

# Loop 4 Plan — Double Sparsity MVP on DeepSeek-V3.2 (FP8): Token-Level Rotation + Option B Benchmark

## Goal Description

Deliver two coupled changes on `dev/double-sparsity-standalone`, anchored at the §13-committed head (base tag: `loop4-base`).

**First:** Rotate the Double Sparsity (DS) implementation from page-level to token-level label storage. Labels are stored per physical KV slot (slot-indexed by `out_cache_loc`), written by hooking the three `set_mla_kv_buffer` call sites in `dsa_backend.py`. At each hook site, the `kv_b_proj` K-side projection is applied to the 512-d MLA latent key to produce 128-d projected nope K per head (`layer.kv_b_proj` is accessible at each hook site via `deepseek_v2.py:L1720`). The custom page lifecycle, within-page averaging quality delta, and most of the adapter indirection are deleted in a single rename + reshape operation. The selector emits logical sequence positions; the adapter converts them to physical token indices via `req_to_token` for FlashMLA's sparse input.

**Second:** Prove the Option B MVP — at the locked operating point (FP8, `flashmla_kv` backends, overlap off, piecewise off), DS-on must pass a full quality gate (AC-12: NIAH-Δ ≤ 5 pp at 4K/16K/64K, MMLU-Δ ≤ 1 pp), survive end-to-end `bench_serving` without crash (AC-8), and leave a Phase B stretch comparator row (AC-11) open for TPS parity measurement.

**Why token-level is the right shape:** K labels written per-slot at the `set_mla_kv_buffer` hook site are automatically slot-consistent with K and V. The KV allocator's lifecycle applies without any separate page hooks; chunked prefill writes chunk-by-chunk into the same slots; radix cache slot reuse is transparent. The adapter collapses to a `req_to_token` gather plus -1 padding.

**Why 128-d nope K:** Paper-faithful. Channel selection operates in the same space as the actual attention computation (`Q_nope · K_nope^T`). Channel importance measured in nope space directly predicts attention score contribution, unlike 512-d latent K which is an indirect proxy through the fixed `kv_b_proj` projection. The `kv_b_proj` K-side projection is applied at write time to produce 128-d K_nope per head; `layer.kv_b_proj` is accessible at all three hook sites.

**Carry-forward lessons applied:**
- `BL-20260520-read-fields-before-abort-mutation`: capture `out_cache_loc` spans before any abort helper; relevant to AC-8 error_containment path.
- `BL-20260520-symbol-vs-test-fixture-drift`: use `dataclasses.fields(ForwardBatch)` to verify field names before writing fixtures; relevant to AC-3, AC-5, AC-8 fixtures.
- Loop 3 lesson: hardware verification is a first-class AC, not a bonus. AC-1, AC-4, AC-5, AC-6, and AC-8 all require real H200 runs.

---

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- **AC-0** (Architecture rotation — token-level signatures, `page_size=64` stays):
  - Positive Tests (expected to PASS):
    - `from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, token_label_write, retrieve_topk` succeeds.
    - `retrieve_topk` returns `(selected_token_indices: int32[bs, max_top_k], valid_lengths: int32[bs])`, sequence-ascending, `-1` padded.
    - `page_table_adapter.py` is < 150 LOC; it accepts logical token positions and returns `int32[bs, get_dsa_index_topk]` physical token indices via `req_to_token` gather.
    - Token label table allocates with shape `[L_local, max_tokens, H_local, label_dim]` where `max_tokens = req_to_token_pool.size` (derived at bind time, not from `device_buffer_size`).
    - `DoubleSparsityConfig.top_k` defaults to 2048 (max tokens, not pages). Validator boots without error when `top_k == get_dsa_index_topk(hf_config)`.
    - `DSGraphState.selected_indices` shape is `int32[max_bs, max_top_k]`. `__init__.py` re-exports use the new names.
    - A non-contiguous physical-slot fixture (e.g., `out_cache_loc = [7, 64, 200, 512]`) is correctly scored and the adapter maps the logical top-K result to those physical slots via `req_to_token`.
    - `validator.py` reads `server_args.dsa_prefill_backend` and `server_args.dsa_decode_backend` (corrected from the dead `nsa_prefill_backend`/`nsa_decode_backend` attributes); the backend-KV-dtype check fires correctly on a bad config.
    - `_bind_double_sparsity_runtime_data` in `deepseek_v2.py` derives `max_tokens = req_to_token_pool.size` at bind time and passes it to `TokenLabelTable.__init__`.
  - Negative Tests (expected to FAIL):
    - Importing `PageSignatureTable` or calling `page_signature_write` raises `ImportError` or `AttributeError`.
    - Validator refuses to start when `top_k != get_dsa_index_topk(hf_config)` and `SGLANG_DS_ALLOW_TOPK_MISMATCH` is unset.

- **AC-1** (M1 — live token-label cache population from KV-write path):
  - Positive Tests (expected to PASS):
    - After `forward_extend` in `dsa_backend.py`, `token_label_table.signatures[layer_id, out_cache_loc, :, :]` is non-zero for each newly written token slot.
    - After `forward_decode`, decode-step slots are updated.
    - AC-8 `bench_serving` does not crash on selector reads (integration smoke).
  - Negative Tests (expected to FAIL):
    - Without the hook, `token_label_table.signatures[layer_id, out_cache_loc, :, :]` remains at initialization default — confirms the hook is the sole population path.

- **AC-1b** (Chunked-prefill probe — NOT a code AC; one-time measurement):
  - Run AC-8 once with `chunked_prefill_size=4096`, forcing a ≥4096-token prompt across two chunks. Assert that per-token labels for tokens 0..4095 (written by chunk 1) are byte-equal to the non-chunked baseline.
  - **Pass:** leave H200 default (`chunked_prefill_size=8192`); AC-8 and AC-9 proceed as normal.
  - **Fail:** append `--chunked-prefill-size -1` to **both** DSA and DS launch commands. Explicit chunked-prefill support is Loop 5 scope. No code is written in Loop 4 to fix the failure.

- **AC-2** (Token-label cache shares KV pool allocator lifetime — no label leaks):
  - Positive Tests (expected to PASS):
    - Boot-time log emits `token_label_table: X.XX GB/rank` (HBM footprint from `[L_local, T, H_local, label_dim]` at fp16).
    - 2× KV-slot budget of requests processed through the same server: slot count consumed by token labels never exceeds KV pool slot count.
    - Freed/evicted KV slots do not produce persistent label pollution: the label at a reused slot is overwritten on the next write before the selector reads it.
  - Negative Tests (expected to FAIL):
    - If label table were sized independently of KV pool and allocated more slots, the boot-time fail-fast gate (HBM budget check) would trigger — confirming the gate is exercised.
    - Stale-slot fixture: allocate a slot, write labels, free the slot, reallocate the same slot to a new request, then invoke the selector BEFORE the new write fires — the test must confirm stale labels from the prior request are not returned. This verifies that the overwrite-before-read invariant holds (or that valid_mask protection catches it) on all code paths including the `save_kv_cache=False` fused tilelang prefill path.

- **AC-3** (M2 — per-request token range ownership prevents cross-request picks):
  - Positive Tests (expected to PASS):
    - Multi-request batch with disjoint KV regions: `retrieve_topk` picks for request B are all within request B's `req_to_token[req_pool_idx_B, :seq_len_B]` slots — no slot from request A appears.
    - Kernel-level fixture with known cross-request boundaries: inspecting `selected_token_indices` directly confirms no boundary violations.
  - Negative Tests (expected to FAIL):
    - Without the ownership mask, a fixture with two requests sharing adjacent token index ranges produces cross-request picks — confirming the mask is load-bearing.

- **AC-4** (V3.2 calibration — `dsv32-fp8-channel-mask.safetensors` generated):
  - Positive Tests (expected to PASS):
    - `python -m sglang.srt.layers.attention.double_sparsity.calibrate --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2 --output /models/dsv32-fp8-channel-mask.safetensors --dtype bfloat16 --page-size 64 --label-dim 16` runs to completion on H200 cluster. (Auto-detects V3.2 MLA via `qk_nope_head_dim=128` from model config; no `--model-arch` flag required.)
    - Calibration uses `k_head_dim = qk_nope_head_dim = 128` (projected nope K). Dataset: Pile validation, seed=42, 256 × 512 tokens. Importance method: **Method 1 as in the original DoubleSparse paper** — `mean(abs(Q_nope · K_nope))` per channel, requiring hooks on both `kv_b_proj` (K-side, 128-d nope K prefix) and the query nope projection (Q_nope, 128-d) in the same forward pass.
    - Generated file passes `channel_mask.py::load_channel_mask` validation: correct shape (`[L, H, 16]` channel selections from 128-d space), content-hash, and AC-4 sanity probe against real V3.2 model.
    - CI tiny-fixture path stays green.
  - Negative Tests (expected to FAIL):
    - `load_channel_mask` rejects a file generated with 512-d channel indices (out-of-range for 128-d nope K space).
    - Calibration with mismatched `--label-dim` vs model's `qk_nope_head_dim` rejects with a clear error.
  - **The generated mask file is NOT committed to git.** It lives at `/models/dsv32-fp8-channel-mask.safetensors` on the H200 cluster.

- **AC-5** (Multi-process two-rank TP harness — bit-equal `selected_token_indices`):
  - `test/registered/integration/test_double_sparsity_tp_multiprocess.py` spawns 2 processes via `torch.multiprocessing`, initializes a process group.
  - Positive Tests (expected to PASS):
    - TP=2 processes score a deterministic fixture; `all_reduce(SUM)` fires over `[bs, max_tokens]`-shaped score tensor (batch-keyed, not flat); both ranks produce bit-equal `selected_token_indices` (in logical position domain, before `req_to_token` conversion).
    - Physical-slot permutation case: rank-0 and rank-1 have different physical slot assignments for the same logical sequence; after `all_reduce` in logical space and `req_to_token` conversion, physical indices are rank-specific but logical positions agree — confirms domain separation is correct.
  - Negative Tests (expected to FAIL):
    - Without `all_reduce` (mock no-op): perturbed rank produces divergent `selected_token_indices` — confirms `all_reduce` is load-bearing.

- **AC-6** (CUDA graph hardware capture at conc=64):
  - Positive Tests (expected to PASS):
    - `capture_decode_step` completes against a real V3.2 conc=64 decode batch at Option B operating point (piecewise off).
    - 100 replay steps run without `CUDA error: launch failed`.
    - Eager path output matches graph replay on a deterministic fixture (`max_abs_diff <= 1e-6`).
  - Negative Tests (expected to FAIL):
    - With a non-preallocated scoring buffer (allocation inside capture region), PyTorch's graph capture mode fails outright — the `assert_no_alloc_in_region` detector fires as a secondary belt-and-suspenders check. The negative test confirms that preallocating output buffers before `torch.cuda.graph.capture_begin()` is the load-bearing fix.
  - Note: `task-ac0-cuda-graph` must depend on `task-m2-rangemask`, since `capture_decode_step` calls `selector.retrieve_topk` with the ownership mask parameter introduced in M2. Capturing without the mask produces a stale graph that must be discarded when M2 lands.

- **AC-7** (Short-seq MHA bypass for DS):
  - Positive Tests (expected to PASS):
    - For a prefill below `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`, `DoubleSparsitySelector.retrieve_topk` is NOT invoked (verified by forward-pass hook).
    - Token-label write hook DOES fire during the dense prefill path → labels for short-prefill tokens are populated.
    - First decode step after short prefill: `retrieve_topk` IS invoked and returns a non-empty, non-trivial selection.
    - **Explicit path verification**: confirm that the FP8 V3.2 prefill path (`dsa_backend.py` at the hook sites) sets `save_kv_cache=True`, so the `token_label_write` hook inside the `if save_kv_cache:` guard fires. If any code path uses `save_kv_cache=False` (tilelang fused prefill), identify and instrument the alternative hook site. This verification must be logged in the task-ac7-bypass commit.
  - Negative Tests (expected to FAIL):
    - Without the bypass gate, `retrieve_topk` is called at prefill below the threshold — the test catches it.

- **AC-8** (M3 — end-to-end `bench_serving` + lightweight quality smoke):
  - Boot DSv3.2 FP8 on 8×H200 at Option B with `--enable-double-sparsity --double-sparsity-config '{"top_k":2048,"page_size":64,"channel_mask_path":"/models/dsv32-fp8-channel-mask.safetensors","device_buffer_size":4096}'` and `--disable-radix-cache` (until AC-10 flips it).
  - Positive Tests (expected to PASS):
    - `bench_serving` with ≥64 requests, ISL ≈ 4096, mixed lengths, conc 16/32/64: no crash for the benchmark duration.
    - `selected_tokens.shape[1] < total_seq_len` on ≥90% of decode steps (non-trivial sparsity active).
    - `dense_fallback_total` matches `error_containment` counter accounting (no silent fallback).
    - Lightweight quality smoke (20 deterministic prompts, `temperature=0`): generate the DSA reference on the **same server binary** used for the DS run, in the **same server restart session** immediately before the DS smoke test; record the DSA server commit SHA alongside the reference file. Then compare DS outputs: prefix-match ≥ 80%; mean ROUGE-L ≥ 0.85; NIAH-mini needle recall ≥ 4/5; no prompt with first-8-tokens entirely different.
  - Negative Tests (expected to FAIL):
    - DSA-only baseline run: `error_containment` counter reads 0 — confirms it is not always-firing.

- **AC-9** (STRETCH — DSA baseline at Option B):
  - DSv3.2 default, TP=8, FP8 explicit, `flashmla_kv` both sides, overlap off, piecewise off, radix ON, CUDA graphs ON. TPS/TTFT/TPOT/goodput at conc 16/32/64 saved to `development/results/native_dsa_<timestamp>.json`.

- **AC-10** (STRETCH — radix cache ON under DS):
  - M3-B hardware fixture (`token_label_write --m3b-fixture-hardware-run`) passes against real V3.2 + generated mask. Cold-prefix vs warm-prefix labels are bit-stable. Explicit verification: confirm that FP8 block quantization assigns identical per-block scale factors for the same token regardless of block-fill level (cold singleton vs. fully-packed block). If scale factors can differ, warm-prefix hits will use misscaled labels — in that case, document the failure mode and defer radix cache to Loop 5. Operator config: `_double_sparsity_radix_fixture_passed = True`. `serve_double_sparsity.sh` removes `--disable-radix-cache`.

- **AC-11** (STRETCH — DS vs DSA comparator row):
  - `python development/benchmark_compare.py --ds-results ... --baseline-results ...`. Gate: **directional with ≥5% tolerance** — DS-on TPS within 5% of DSA-on TPS passes; regression ≥5% triggers a profiling obligation (not an automatic loop failure). P99 TTFT ≤ DSA-on P99 TTFT × 1.10. Only `--enable-double-sparsity` and `--double-sparsity-config` differ between columns.
  - Reproducibility requirements: use a fixed random seed for the request arrival process; run for minimum 600s measurement window after a 120s warmup; run at least 3 independent trials and report the median; record commit SHA + full server args + chunked-prefill setting alongside each result JSON. Without these, measurement noise alone can swing the 5% gate.

- **AC-12** (HARD — full quality gate; moved from stretch):
  - `test/manual/test_double_sparsity_v32.py`. Runs both DS and DSA in quality-test mode (no separate AC-9 JSON required for quality comparison). NIAH @ 4K/16K/64K: DS-on delta ≤ 5 pp at each length vs DSA baseline. MMLU 5-shot: DS-on delta ≤ 1.0 pp vs DSA baseline. **Loop does not close without AC-12 passing.**

- **AC-13** (Regression — 150 Loop-2 unit tests pass after shape migration):
  - Scope: this gate applies **after AC-0 shape migration is complete**. The 150 existing tests use page-level shapes and will fail during AC-0 development before `task-ac0-tests` migrates them. The gate is not "green after every change throughout the loop" — it is green after `task-ac0-tests` merges, and remains green for all subsequent changes.
  - Positive Tests (expected to PASS):
    - `pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py` — all 150 tests pass, same count, after AC-0 shape updates land.
  - Negative Tests (expected to FAIL):
    - Running the suite against the pre-AC-0 codebase (page-shaped) fails on shape assertions — confirms tests actually check the shapes being migrated.

---

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The implementation delivers AC-0 through AC-8 (architecture rotation + end-to-end bench_serving), AC-12 (full quality gate), AC-13 (regression), plus all Phase B stretch ACs (AC-9 through AC-11) if Phase A + AC-12 close before round 8. The AC-0 rotation covers all six affected files in the `double_sparsity/` package, the adapter rewrite to < 150 LOC, all three `dsa_backend.py` hook sites, and the 150-test migration. Phase B stretch is fully attempted: DSA baseline JSON, radix cache flip, and comparator row.

### Lower Bound (Minimum Acceptable Scope)

AC-0 through AC-8, AC-12, and AC-13 are closed. Phase B stretch ACs (AC-9 through AC-11) are deferred to a future loop if Phase A + AC-12 exhaust the round budget. The AC-1b probe result is recorded regardless of outcome (chunked-prefill behavior documented). AC-12 is hard: the loop is not declared done without it, even if it requires extending past AC-8.

### Allowed Choices

- **Hook implementation:** Python-level hook at `set_mla_kv_buffer` call sites; the hook applies `kv_b_proj` K-side projection to produce 128-d nope K before writing labels. Triton kernel for `token_label_write` acceptable if eager is a bottleneck.
- **Scorer implementation:** Adapt existing `_compute_page_scores_kernel` (Triton) for token-level shapes `[L,T,H,D]`, or use a torch fallback initially and optimize if AC-6 shows allocation issues.
- **Adapter:** `req_to_token`-based gather in Python for initial correctness (target < 150 LOC). Triton kernel acceptable if Python gather is a capture-time bottleneck.
- **TP process harness:** `torch.multiprocessing` for AC-5.
- **Cannot use:** `nsa_backend.py`, `nsa/nsa_indexer.py`, `nsa/transform_index.py` (all deprecated re-exports of `dsa/` equivalents). `dsa/transform_index.py::transform_index_page_table_decode` (asserts `page_size==1`, incompatible with FlashMLA `page_size=64`). Page-level shapes or page lifecycle hooks in any new code.
- **Operating point is locked:** Both DSA baseline and DS use the exact Option B commands (see draft §operating-point-cheatsheet). Any deviation breaks the apples-to-apples requirement for AC-11.

---

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

**Token-label write (AC-0, AC-1 hook site):**

At each `set_mla_kv_buffer(layer, cache_loc, k, k_rope)` call in `dsa_backend.py`, `k` is the 512-d MLA latent key (`kv_lora_rank=512`) with shape `[num_tokens, 1, 512]`. To get 128-d projected nope K per head, apply the `kv_b_proj` K-side projection at write time. `layer.kv_b_proj` is accessible at all three hook sites via `deepseek_v2.py:L1720`. The label write then projects to `label_dim=16` selected channels:

```python
# Conceptual token_label_write — k_latent is [T, 1, 512] MLA latent from hook site
# Apply kv_b_proj K-side to produce 128-d nope K per head
k_nope = apply_kv_b_proj_k_side(k_latent, layer.kv_b_proj, H_local, nope_dim=128)
# k_nope: [T, H_local, 128]
labels = k_nope[:, :, channel_selection[layer_id]]   # [T, H_local, label_dim]
table.signatures[layer_id, cache_loc, :, :] = labels
```

The `kv_b_proj` output concatenates `[K_nope | V]` per head; slice the first `H_local * nope_dim` columns, reshape to `[T, H_local, 128]`, then select label channels.

**Selector flow with logical → physical conversion (page_table_adapter, AC-0):**

The scorer operates in physical-slot space (iterating `req_to_token[req_pool_idx, :seq_len]` slots), reduces scores per request, and runs top-K. The selector returns **logical positions** (0-indexed within the request's sequence). The adapter then resolves to physical token indices:

```python
# retrieve_topk returns logical positions [bs, K], sequence-ascending, -1 padded
logical_topk, valid_lens = selector.retrieve_topk(...)

# Adapter: logical → physical via req_to_token
# req_to_token values are exactly the physical token indices FlashMLA expects
# (page_table_1 in dsa_indexer.py is populated directly from req_to_token)
physical_slots = req_to_token[req_pool_indices[:, None], logical_topk]  # [bs, K]
physical_slots = torch.where(logical_topk == -1, torch.tensor(-1), physical_slots)

# FlashMLA sparse path (Option B, flashmla_kv) consumes physical_slots as
# int32[bs, get_dsa_index_topk] physical token indices, -1 padded
# These values index into kv_cache.view(-1, 64, 1, dim) at the page level;
# req_to_token already stores indices in this format — no additional conversion needed.
```

**Calibration (AC-4) — Method 1, channel axis is 128:**

The original DoubleSparse paper (`config/offline_calibration.py`) uses Method 1: `mean(abs(q_channel * k_channel))` per channel. Implementing this in `calibrate.py` requires hooking both Q_nope and K_nope in the same forward pass:

```python
# Method 1 calibration: hook kv_b_proj (K-nope 128-d) AND q_nope (128-d) together
# Per-channel importance = mean_over_tokens(abs(q_nope_channel * k_nope_channel))
# This requires two registered hooks per layer capturing activations from the
# same forward pass, then computing the joint importance score.
importance += (q_nope * k_nope).abs().reshape(-1, H, 128).mean(dim=0)  # [H, 128]
```

The existing `k_head_dim = qk_nope_head_dim = 128` derivation in `calibrate.py` is correct. V3.2 model config is auto-detected via `getattr(config, "qk_nope_head_dim", 0)` — no `--model-arch` flag required.

### Relevant References

- `python/sglang/srt/layers/attention/double_sparsity/` — AC-0 rename targets: `page_signature_table.py`, `page_signature_write.py`, `page_table_adapter.py`, `selection_kernel.py`, `selector.py`, `cuda_graph.py`, `config.py`, `validator.py`
- `python/sglang/srt/layers/attention/dsa_backend.py` L1439, L1637, L2162 — three live `set_mla_kv_buffer` hook sites for AC-1
- `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` — reference for FlashMLA sparse token index input format (Option B adapter target)
- `python/sglang/srt/models/deepseek_v2.py` L1720 (`kv_b_proj` accessible at hook site), L1832 (`_bind_double_sparsity_runtime_data`), L2018 (`_select_topk_indices`) — selector dispatch wiring
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:245` (`forward_absorb_prepare`) — short-seq MHA bypass gate (AC-7)
- `python/sglang/srt/model_executor/forward_batch_info.py` — `ForwardBatch`, M2 `req_to_token` attachment (AC-3)
- `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` — update for Method 1 Q+K joint hooks, `kv_b_proj` K-side traversal (AC-4)
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — 150-test regression suite (AC-13)
- `test/manual/test_double_sparsity_v32.py` — full quality suite (AC-12)
- `test/manual/test_dsv32_quality_smoke.py` — lightweight quality smoke fixture for AC-8 (new file, 20 prompts)
- `development/serve_double_sparsity.sh`, `development/benchmark.sh`, `development/benchmark_compare.py` — bench harness (AC-8, AC-9, AC-11)
- `development/past_implementations/study/07-mvp-proposed-architecture.md` — design intent, §9.4 quality smoke spec, §10 calibration contract, §12 non-goals, §13 token-level rotation spec
- `development/past_implementations/DoubleSparse/config/offline_calibration.py` — original paper's calibration implementation (Method 1: `mean(abs(q*k))` per channel)
- `development/CLIENT_SLOS.md` — client SLOs

---

## Dependencies and Sequence

### Milestones

1. **Architecture Rotation (Phase A1 — prerequisite for everything)**
   - AC-0: file renames, shape updates, adapter rewrite, validator, config, deepseek_v2.py wiring, 150-test migration
   - AC-13: regression suite green after `task-ac0-tests` merges (and remains green for all subsequent changes)

2. **Data Flow and Selection (Phase A2 — parallel after A1)**
   - AC-1 + AC-1b: M1 hook at dsa_backend.py (3 sites, with kv_b_proj projection), hardware population test, chunked-prefill probe
   - AC-2: token-label cache lifetime, HBM boot log
   - AC-3: M2 per-request token range ownership
   - AC-7: short-seq MHA bypass

3. **Calibration + TP (Phase A3 — requires A2)**
   - AC-4: update calibrate.py (Method 1 Q+K hooks), hardware run on H200, generate mask file
   - AC-5: multi-process TP=2 test (requires M1 + M2 wired)

4. **Graph Capture + End-to-End (Phase A4 — requires A3)**
   - AC-6: CUDA graph hardware capture at conc=64 (requires full decode path: M1, M2, AC-7; must run after M2 to capture the ownership mask parameter)
   - AC-8: end-to-end bench_serving + lightweight quality smoke (requires AC-6 + calibrated mask)

5. **Quality Gate (Phase A5 — requires A4)**
   - AC-12: full NIAH/MMLU quality gate (hard; requires bench infra from AC-8 and DSA quality baseline)

6. **Phase B Stretch (requires Phase A5)**
   - AC-9: DSA benchmark JSON (can run in parallel with AC-12 quality gate)
   - AC-10: radix cache ON (requires AC-8 passing)
   - AC-11: comparator row (requires AC-9 + AC-10)

Dependencies: AC-0 → everything. AC-1 → AC-2, AC-3, AC-7, AC-5, AC-6. AC-4 → AC-6, AC-8. AC-6 → AC-8. AC-8 → AC-12, AC-9, AC-10. AC-1b → AC-6 (run before full bench_serving). AC-10 → AC-11. AC-9 → AC-11.

**Cancel signals (from Loops 1–3 lessons):**
- If AC-0 takes > 2 rounds, the rename + reshape estimate is wrong — stop and re-scope before continuing.
- If Day 3 of the loop hasn't closed AC-0, stop and reassess.
- If 2 consecutive rounds open more gaps than they close, invoke `/humanize:cancel-rlcr-loop` and reassess scope.

---

## Task Breakdown

Each task has exactly one routing tag: `coding` (implemented by Claude) or `analyze` (executed via Codex `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag | Depends On |
|---|---|---|---|---|
| task-ac0-rename | Rename page_signature_table→token_label_table, page_signature_write→token_label_write; update shapes [L,P,H,D]→[L,T,H,D]; update `__init__.py` re-exports; remove old names | AC-0 | coding | — |
| task-ac0-kernel | selection_kernel.py: compute_page_scores→compute_token_scores; valid_mask [L,P]→[L,T]; select_topk_sequence_order operates on tokens; all_reduce over [bs, max_tokens] score tensor (batch-keyed) | AC-0 | coding | task-ac0-rename |
| task-ac0-adapter | Rewrite page_table_adapter.py to < 150 LOC: retrieve_topk emits logical positions; adapter converts via `req_to_token` gather to `int32[bs, get_dsa_index_topk]` physical token indices for FlashMLA sparse input. Error tracking: keep `row_errors` dict pattern (compatible with existing AC-8 counter check); a simple scalar error count replaces per-row mutable dict. | AC-0 | coding | task-ac0-kernel |
| task-ac0-config | DoubleSparsityConfig: top_k = max tokens (not pages), default 2048; device_buffer_size remains as score-scratch buffer cap; update parse_double_sparsity_config | AC-0 | coding | task-ac0-rename |
| task-ac0-validator | Boot assert: `top_k == get_dsa_index_topk(hf_config)`; respect `SGLANG_DS_ALLOW_TOPK_MISMATCH=1` override; fix dead attribute names `nsa_prefill_backend`/`nsa_decode_backend` → `dsa_prefill_backend`/`dsa_decode_backend` so the backend-KV-dtype check fires correctly | AC-0 | coding | task-ac0-config |
| task-ac0-deepseek-v2 | Update `deepseek_v2.py`: (1) `_bind_double_sparsity_runtime_data` — rename PageSignatureTable import, derive `max_tokens = req_to_token_pool.size` at bind time, pass to TokenLabelTable; (2) `_select_topk_indices` — update call to new adapter ABI; (3) `ds_topk_indices_out` pre-allocation — resize to token-level `max_top_k`; (4) `row_errors` usage — update to new scalar error count | AC-0 | coding | task-ac0-adapter |
| task-ac0-cuda-graph | DSGraphState: selected_indices shape `[max_bs, max_top_k]`; update capture_decode_step to new selector ABI (logical positions, req_to_token adapter, ownership mask from M2) | AC-0 | coding | task-ac0-kernel, task-m2-rangemask |
| task-ac0-tests | Migrate 150 Loop-2 unit tests for page→token shape changes; verify count preserved (AC-13 gate activates after this task merges) | AC-0, AC-13 | coding | task-ac0-rename |
| task-m1-hook | Wire token_label_write at dsa_backend.py L1439 (forward_extend), L1637 (forward_decode), L2162 (TRT-LLM MLA); at each site: apply kv_b_proj K-side projection to 512-d latent key to produce 128-d k_nope per head, then write labels indexed by `out_cache_loc`. Also verify that the FP8 prefill path sets `save_kv_cache=True` at each hook site; document findings in commit message. | AC-1 | coding | task-ac0-rename |
| task-ac1-hwtest | Hardware test: real forward_extend pass on H200; assert `token_label_table.signatures[layer_id, out_cache_loc]` non-zero for each written token slot | AC-1 | coding | task-m1-hook |
| task-ac2-lifetime | Add boot-time GB/rank log; 2× slot-budget test; verify slot count bounded by KV pool slot count; confirm reused-slot overwrite semantics; add stale-slot negative test (freed→reallocated→read before write) | AC-2 | coding | task-m1-hook |
| task-m2-rangemask | ForwardBatch: attach per-request token range from req_to_token_pool; scorer masks out-of-range slots to −inf before top-K | AC-3 | coding | task-ac0-kernel |
| task-ac3-test | Multi-request boundary kernel test: confirm picks never cross request KV range; negative fixture confirms mask is load-bearing | AC-3 | coding | task-m2-rangemask |
| task-ac7-bypass | Short-seq MHA bypass: DS selector not invoked for prefill below SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD; token-label write hook still fires; first decode runs DS; explicitly confirm save_kv_cache=True on FP8 V3.2 prefill path or identify alternative hook site | AC-7 | coding | task-m1-hook |
| task-ac4-calibrate | Update calibrate.py: implement Method 1 (mean(abs(Q_nope · K_nope)) per channel) by adding Q_nope hook alongside the existing kv_b_proj K-side hook; confirm k_head_dim = qk_nope_head_dim = 128; V3.2 auto-detected via config (no --model-arch flag needed); Pile-val-256×512 dataset; CI tiny-fixture green | AC-4 | coding | task-ac0-rename |
| task-ac4-hwrun | Hardware run on H200 cluster: generate `/models/dsv32-fp8-channel-mask.safetensors`; validate with load_channel_mask; NOT committed to git | AC-4 | analyze | task-ac4-calibrate |
| task-ac5-tp | Add test/registered/integration/test_double_sparsity_tp_multiprocess.py: TP=2 multiprocess, all_reduce(SUM) on [bs, max_tokens] score tensor in logical-position space, bit-equal logical selected_token_indices; physical-slot permutation negative case | AC-5 | coding | task-m2-rangemask |
| task-ac6-cuda-graph | Decode-path full-graph capture with preallocated buffers; confirm preallocation prevents PyTorch graph capture failure (not just alloc detector); 100-step replay; eager==graph deterministic; alloc-detector negative test | AC-6 | coding | task-m1-hook, task-m2-rangemask |
| task-ac6-hwrun | Hardware run: execute full-graph capture at conc=64 against real V3.2, Option B | AC-6 | analyze | task-ac6-cuda-graph, task-ac4-hwrun |
| task-ac1b-probe | Chunked-prefill probe: boot same server config as AC-6 hwrun; run with chunked_prefill_size=4096; compare token labels 0..4095 vs non-chunked baseline | AC-1b | analyze | task-ac6-hwrun |
| task-ac8-server | Boot DSv3.2 FP8 TP=8 8×H200 at Option B; bench_serving conc 16/32/64, ≥64 requests, ISL ≈ 4096 | AC-8 | analyze | task-ac6-hwrun, task-ac1b-probe |
| task-ac8-quality | Lightweight quality smoke: generate DSA reference on same server binary + same restart session, record commit SHA; then run 20 deterministic prompts temperature=0 vs reference; assert 4 quality gates | AC-8 | analyze | task-ac8-server |
| task-ac12-quality | Full quality gate: run DS and DSA in quality-test mode (NIAH @ 4K/16K/64K + MMLU 5-shot); assert deltas ≤ 5 pp / 1.0 pp | AC-12 | analyze | task-ac8-server |
| task-ac9-baseline | DSA baseline: boot DSA at Option B; bench_serving conc 16/32/64; save JSON to development/results/ | AC-9 | analyze | task-ac8-server |
| task-ac10-radix | M3-B hardware fixture; verify FP8 scale factor stability across cold/warm prefix writes; flip `_double_sparsity_radix_fixture_passed`; update serve_double_sparsity.sh | AC-10 | coding | task-ac9-baseline |
| task-ac11-compare | Run benchmark_compare.py with fixed seed, 600s window, 120s warmup, 3 trials, median aggregation; emit comparator row; check TPS directional gate (≥5% tolerance), P99 TTFT ≤ 1.10× | AC-11 | analyze | task-ac9-baseline, task-ac10-radix |

---

## Claude-Codex Deliberation

### Agreements

- Slot-indexed label rotation (slot = `out_cache_loc`) is the correct architecture for token-level DS.
- `dsa_backend.py` (not deprecated `nsa_backend.py`) contains the three live `set_mla_kv_buffer` hook sites at L1439, L1637, L2162.
- `nsa_backend.py`, `nsa/nsa_indexer.py`, and `nsa/transform_index.py` are deprecated re-export shims; no new code may reference them.
- `dsa/transform_index.py::transform_index_page_table_decode` asserts `page_size==1` and hard-codes `TOPK=2048`; it is unsuitable for the adapter path.
- `selected_token_indices` must be **logical sequence positions** (not physical KV slots) to ensure consistent range-mask semantics, TP sync domain, and sequence-ascending ordering.
- The adapter converts logical positions to physical token indices via `req_to_token` gather; the Option B `flashmla_kv` sparse path consumes `int32[bs, get_dsa_index_topk]` physical token indices, not a block_table. `req_to_token` values are already in the correct format for FlashMLA (same values `dsa_indexer.py` uses for `page_table_1`).
- An HBM budget gate (boot-time GB/rank log + fail-fast) is required for the `[L,T,H,D]` token label table.
- The `top_k == get_dsa_index_topk(hf_config)` boot assert is correct and keeps Phase B comparison apples-to-apples.

### Resolved Disagreements

- **Calibration/write-space dimension (512-d vs 128-d) — re-resolved with correct implementation path:** Codex v1 and v2 both flagged a mismatch between `page_signature_write.py`'s `_NOPE_DIM=512` (MLA latent K, what `set_mla_kv_buffer` receives) and `calibrate.py`'s `k_head_dim = qk_nope_head_dim = 128` (projected nope K). **Resolution: 128-d nope K is correct (paper-faithful, theoretically superior).** However, the `k` argument at `set_mla_kv_buffer` is the 512-d MLA latent key — reading 128-d per-head K_nope requires applying `kv_b_proj` at write time. `layer.kv_b_proj` is accessible at all three hook sites (`deepseek_v2.py:L1720`). The matmul cost is `[T, 512] @ [512, H_local*(128+v_dim)]` → slice K prefix → `[T, H_local, 128]`; acceptable on the prefill write path. The write kernel is NOT reading 128-d directly from the hook site argument — it applies the projection.

- **Calibration importance method (L2(K²) vs Method 1 Q·K):** `calibrate.py` currently computes L2-squared K importance (`sum(K_channel²)`) without Q involvement. The original DoubleSparse paper (`config/offline_calibration.py`) and by implication `sglang-last-with-double-sparsity` use **Method 1**: `mean(abs(q_channel * k_channel))` per channel, with both Q and K from the same forward pass. **Resolution: implement Method 1 in `calibrate.py`** — add Q_nope hook alongside the K hook. This requires refactoring calibrate.py to collect matching Q/K activations in each forward pass and compute the joint importance. This is the paper-faithful approach.

- **AC-11/AC-12 scope (stretch vs hard):** Codex recommended both hard; draft had both stretch. **User resolved: AC-11 stays stretch; AC-12 moves to hard.** Rationale: quality correctness (NIAH/MMLU deltas) is a hard MVP requirement — shipping DS with unknown quality degradation is not acceptable. TPS parity (AC-11) is a performance target with directional tolerance.
- **AC-11 TPS gate type:** Codex and Claude both treated it as binary in first pass. **User confirmed directional with ≥5% tolerance** — regression ≥5% triggers profiling obligation, not automatic loop failure.
- **`device_buffer_size` semantics:** After token rotation, the original "page buffer count" semantics are obsolete. **Resolved: repurpose as score-scratch buffer cap** (maximum concurrently live tokens for the decode scoring scratch tensor). The config value `4096` is retained as default.

### Convergence Status

- Final Status: `converged`
- Rounds: 2 Codex passes + post-critique refinement
- No REQUIRED_CHANGES remain unaddressed.

---

## Pending User Decisions

All decisions resolved. No PENDING items.

- DEC-1: Label dimension source (512-d latent K vs 128-d nope K) + write-path implementation
  - Claude Position: 128-d nope K (paper-faithful, theoretically superior); projection via `kv_b_proj` at write time
  - Codex Position: Required resolution; flagged both as viable with different tradeoffs
  - Decision Status: **RESOLVED — 128-d nope K; kv_b_proj K-side projection applied at hook time; `layer.kv_b_proj` accessible at all three sites**

- DEC-2: AC-11/AC-12 scope
  - Claude Position: Draft scope (both stretch)
  - Codex Position: Both hard for MVP credibility
  - Decision Status: **RESOLVED — AC-11 stretch, AC-12 hard (user choice)**

- DEC-3: AC-11 TPS gate strictness
  - Claude Position: Binary as written in draft
  - Codex Position: Needs explicit definition
  - Decision Status: **RESOLVED — directional, ≥5% regression triggers investigation (user choice)**

---

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers. These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code: `token_label_table` (not "AC-0 table"), `write_token_label` (not "M1 hook"), `select_topk_tokens` (not "AC-0 selection").
- Field names in test fixtures must be verified against live dataclass definitions before writing (see carry-forward lesson `BL-20260520-symbol-vs-test-fixture-drift`).

### Pre-Loop Verifications (complete before first RLCR round)

- **topk confirmed:** `get_dsa_index_topk(DeepSeek-V3.2-config)` returns `2048` — verified from `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2/config.json` (`index_topk=2048`). Safe to hard-wire in validator and boot assert.
- **kv_b_proj projection cost:** Before writing `task-m1-hook`, measure the `kv_b_proj` K-side projection matmul cost on H200 at realistic batch sizes (e.g., 64 tokens prefill chunk). If per-token cost exceeds ~5 µs, consider a Triton kernel for the projection+label-write path as a Phase C item.
- **save_kv_cache path for FP8 V3.2:** Confirm at boot that FP8 V3.2 prefill goes through `save_kv_cache=True` at each `set_mla_kv_buffer` hook site. Document the specific code path in `task-m1-hook` commit message before proceeding to `task-ac7-bypass`.

### TP Collective Notes

- AC-5 uses `torch.multiprocessing` TP=2 on a single node (shared-memory NCCL). This uses a different collective path than production TP=8 over NVLink/IB. A rank-divergence bug caused by a misconfigured process group would surface at AC-6/AC-8, not AC-5. If AC-5 passes but AC-6 shows rank divergence, set `NCCL_P2P_DISABLE=1` in the TP test to force a more production-representative path.

### Error Containment Design

- The `row_errors` pattern in the adapter is kept: the adapter returns a scalar `error_count` alongside `physical_slots`. This is simpler than per-row mutable dict while remaining compatible with the AC-8 `error_containment` counter check. Structured exceptions are avoided on the hot decode path.

### Explicit Non-Goals (Loop 4)

Per `07-mvp-proposed-architecture.md §12` and draft scoping:
- Default cookbook bf16 path (`flashmla_sparse` + `fa3`) — future loop.
- Piecewise CUDA graphs and overlap scheduler under DS — Phase B explicitly disables both.
- MTP/EAGLE speculative decoding under DS.
- GLM-5.1, 128K ISL, FP4 weights, DP Attention.
- Twilight top-p selection, Extensions, PD-Disagg, HiCache, CPU offload.
- Phase C kernel ports (Triton ports of token_label_write / compute_token_scores; fused FP8-dequant; raft_topk adoption) — gated on Phase B showing DS-on < DSA-on TPS by > 5% with profile evidence.
- Explicit chunked-prefill support code — Loop 4 only probes (AC-1b).
- Page-level signature design at 1M context — recoverable from git history; not in scope until client asks.

### Operating-Point Cheatsheet (Option B, locked)

Both DSA baseline and DS runs use this point. Any deviation breaks the apples-to-apples requirement.

```bash
# DSA baseline (AC-9):
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
  --kv-cache-dtype fp8_e4m3 \
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule --disable-piecewise-cuda-graph \
  --page-size 64 --trust-remote-code

# DS (AC-8 / AC-11):
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
  --kv-cache-dtype fp8_e4m3 \
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule --disable-piecewise-cuda-graph \
  --page-size 64 --trust-remote-code \
  --enable-double-sparsity \
  --double-sparsity-config '{"top_k":2048,"page_size":64,"channel_mask_path":"/models/dsv32-fp8-channel-mask.safetensors","device_buffer_size":4096}' \
  --disable-radix-cache  # Remove after AC-10 flips it
```

Chunked prefill (conditional on AC-1b outcome): if probe passes, leave H200 auto-default (`chunked_prefill_size=8192`) on both. If probe fails, append `--chunked-prefill-size -1` to both.

### RLCR Loop Configuration

- Anchor base branch: `loop4-base` (create at §13-committed head at loop start)
- Working branch: `dev/double-sparsity-standalone`
- Round budget: ≤ 14 rounds. Phase A (AC-0 through AC-8 + AC-12): 9+1 hard ACs; Phase B: 3 stretch. The kv_b_proj projection step in `task-m1-hook` and the Method 1 Q+K hook refactor in `task-ac4-calibrate` both add scope vs the original estimate; these are explicitly tracked as separate tasks with their own round budget expectation.
- Plan budget cap (advisory): if a round closes < 2 ACs AND opens > 2 new gaps, escalate immediately.
- Cancel signal: if Day 3 of the loop hasn't closed AC-0, the rename + reshape estimate is wrong; stop and re-scope.

--- Original Design Draft Start ---

# Loop 4 Draft — Reach the Double Sparsity MVP on DeepSeek-V3.2 (FP8)

## Why this loop exists

**Framing:** the implementation we are landing is **sglang-last's DS algorithm (per-token sparsity, dense prefill, sparse decode, per-token K_label cache slot-indexed by `out_cache_loc`) — but performant**. The performance knobs sglang-last couldn't support (FlashMLA backend, FP8 KV cache, MLA model, CUDA graphs, radix cache, page sizes including 64, multi-process TP rank sync, mixed batches via per-request range masks) come "for free" from inheriting the rest of sglang's modern infrastructure once the selection granularity is right. The win vs sglang-last is not a different algorithm — it's the same algorithm at the modern operating point.

Loop 3 set the scope at 3 items (M1/M2/M3) and never ran. The structural plumbing landed in Loops 1–2 (3,887 LOC, 150 unit tests, ABI locked, `bind_runtime_data` wired at `deepseek_v2.py:1541`), but the **data isn't flowing through it on real V3.2** and the **selection granularity was page-level, paying complexity for memory savings that only matter at 1M context** (no client ask).

Loop 4 makes two changes at once:

1. **Rotates the architecture to token-level signatures at page_size=64** (per `07-mvp-proposed-architecture.md` §13). Selection is per-token, storage is per-token (slot-indexed by `out_cache_loc` exactly like K and V), FlashMLA still reads at page granularity via NSA's existing `transform_index_page_table_decode`. This deletes the custom page adapter, the page lifecycle hooks, and the within-page averaging quality delta — all in one move, by becoming the same shape sglang-last had.
2. **Reaches the MVP at the Option B operating point** (FP8 + `flashmla_kv` + overlap off + piecewise off, both DSA baseline and DS at the same operating point): DS-on matches or beats DSA-on TPS at conc=64 with NIAH-Δ ≤ 5 pp and MMLU-Δ ≤ 1 pp.

**Chunked prefill is *probed*, not actively supported.** sglang-last got chunked prefill for free because its K_label cache was slot-indexed by `out_cache_loc`. The token-level rotation preserves that property, so chunked prefill *should* work implicitly. **Phase A includes a probe test** that asserts implicit support. **If the probe fails**, the Phase B comparison disables chunked prefill on both DSA and DS (`--chunked-prefill-size -1`) and chunked-prefill explicit support becomes Loop 5 scope. **No explicit chunked-prefill code lands in Loop 4 regardless of the probe outcome** — either it works implicitly and we leave it alone, or it doesn't and the baseline gets adjusted.

**Anchor:** start from `dev/double-sparsity-standalone` at the head with §13 committed.

## Hard scope — Phase A (9 ACs: AC-0 through AC-8, must close; AC-1b is a one-time probe)

### AC-0 — Architecture rotation: token-level signatures, page_size=64 stays

The first round closes the rotation. Subsequent ACs assume token-level.

- Rename `page_signature_table.py` → `token_label_table.py`. Shape changes from `[L_local, max_pages, H_local, label_dim]` to `[L_local, max_tokens, H_local, label_dim]`. The "max_tokens" is the KV pool's slot count — no separate lifecycle (the KV allocator already manages it).
- Rename `page_signature_write.py` → `token_label_write.py`. Per-token FP8 dequant + channel projection — no page-mean reduction. Slot-indexed by `out_cache_loc`, exactly parallel to K/V writes.
- `selection_kernel.py`: `compute_page_scores` → `compute_token_scores` (BGEMV `Q_label · K_label_bufferᵀ`). `select_topk_sequence_order` operates on tokens.
- `page_table_adapter.py`: collapse to a thin wrapper around NSA's existing `transform_index_page_table_decode` (the same helper NSA uses). Most of the 404 LOC deletes.
- `selector.py::retrieve_topk` return type: `(selected_token_indices: int32[bs, max_top_k_tokens], valid_lengths: int32[bs])`, sequence-ascending with `-1` padding.
- `cuda_graph.py::DSGraphState`: static `selected_token_indices: int32[max_bs, max_top_k_tokens]`. Shape change only; same machinery.
- `config.py`: `top_k` semantics now max **tokens** per request (matches sglang-last's `--ds-heavy-token-num`; same name kept). Default 2048.
- `validator.py`: assert at boot that `DoubleSparsityConfig.top_k == get_dsa_index_topk(hf_config)` (the model's intrinsic top-k used by NSA's lightning indexer). Operator can override the assertion with `SGLANG_DS_ALLOW_TOPK_MISMATCH=1` for ablation runs, but the default refuses to start when DS would pick a different number of tokens than DSA does — this is what keeps the Phase B comparison apples-to-apples.

Note: page_size=64 is **unchanged**. It is FlashMLA's KV layout requirement; selection granularity is orthogonal.

### M1 — Live token-label cache population from the KV-write path

Hook the per-token write at `set_mla_kv_buffer` call sites in `python/sglang/srt/layers/attention/nsa_backend.py` (Codex flagged ~L1383, ~L1583, ~L2108 in Loop 2). The hook writes one label row per token slot, indexed by `out_cache_loc`.

**Chunked prefill is expected to work implicitly** because the write fires once per chunk and writes the chunk's own slots — same way sglang-last got it for free with its slot-indexed `label_buffer`. Phase A includes a probe (AC-1b) that asserts this implicit support. **No explicit chunked-prefill code is written in Loop 4**; if the probe fails, the Phase B baseline disables chunked prefill on both sides (see AC-8 / AC-9) and explicit support becomes Loop 5.

The token-label cache shares the KV pool's allocator lifetime — when a slot is freed/evicted/reused on the KV side, the label slot follows automatically. No explicit retract code.

### M2 — Per-request token range mask attached to ForwardBatch

Build a per-request token range `(req_start[r], req_end[r])` from `req_to_token_pool.req_to_token`, `req_pool_indices`, `seq_lens`. The selector's score reduction masks tokens outside the request's own range to `-inf` before top-K, so picks never escape the request's KV range. This is the token-level equivalent of the page-level `sparse_mask` from earlier scope — cheaper to express because it's a range, not a bitmask over pages.

### V3.2 calibration code path

Adapt `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` to walk DSv3.2 MLA layers: read `self_attn.kv_b_proj` (MLA's K↑proj) on the K side; channel axis = 512 (post-RoPE nope dimension). Add `--model-arch deepseek_v3` flag. Keep the dataset + scoring identical to all three reference impls: **Pile validation, seed=42, 256 × 512 tokens, Method 1 (`mean(abs(Q·K))` per channel)**. Run the script against real V3.2 weights on the H200 cluster to **generate** `dsv32-fp8-channel-mask.safetensors`. **The mask file itself is NOT committed to git** — it lives at `/models/dsv32-fp8-channel-mask.safetensors` on the cluster. The CI tiny-fixture path must remain green.

### Multi-process two-rank TP harness

Loop 2 closed a single-process synthetic two-rank fixture; the real multi-process harness was deferred. With TP=8 in production, rank-divergent `selected_token_indices` produces silently-wrong attention. Add `test/registered/integration/test_double_sparsity_tp_multiprocess.py` that spawns 2 processes via `torch.multiprocessing`, initializes a process group, runs the DEC-9 `all_reduce(SUM)` over `[max_tokens]` scores, and asserts bit-equal `selected_token_indices` across ranks. Includes a negative test that deliberately perturbs one rank's pre-reduce scores and confirms post-reduce equality (catches a no-op all-reduce).

### Hardware CUDA-graph capture at conc=64

`cuda_graph.py::capture_decode_step` (full-graph capture) is exercised only on synthetic shapes. Capture against a real V3.2 conc=64 decode batch at the Option B operating point (piecewise off). `assert_no_alloc_in_region` does not trip. Replay 100 steps without `CUDA error: launch failed`. Eager and graph outputs match on a deterministic fixture (`max_abs_diff <= 1e-6`).

### Short-sequence MHA bypass for DS

DSA falls back to MHA for prefills below `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` (model's `index_topk`, ≈ 2048 for V3.2). DS mirrors this: for prefills below the threshold, the DS selector **does not run at prefill** — the path goes straight to MHA. Decode after a short prefill still runs DS (the dense prefill path has already written per-token labels via the M1 hook, so the first decode step sees a populated buffer).

This matches all three reference impls' "prefill dense, decode sparse" pattern. **Prefill always writes labels; selection only runs at decode.**

### M3 — End-to-end DeepSeek-V3.2 (FP8) bench_serving with DS enabled

The **done criterion for Phase A.** Boot DSv3.2 FP8 on 8×H200 at the Option B operating point with `--enable-double-sparsity --double-sparsity-config '{"top_k":2048,"page_size":64,"channel_mask_path":"/models/dsv32-fp8-channel-mask.safetensors","device_buffer_size":4096}'`. Run `bench_serving` against it: ≥ 64 requests, ISL ≈ 4096, mixed lengths, conc 16/32/64. Then run the lightweight quality smoke (see AC-8).

## Stretch scope — Phase B (4 ACs: AC-9 through AC-12, attempted if Phase A closes early)

The "MVP done if possible" arc — match or beat DSA at the Option B operating point with quality deltas inside budget.

### B1 — DSA baseline run at the Option B operating point
DSv3.2 default, TP=8, FP8 explicit (`--kv-cache-dtype fp8_e4m3 --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv`), `--disable-overlap-schedule --disable-piecewise-cuda-graph`, radix cache ON, full-graph CUDA graphs ON. Save TPS / TTFT / TPOT / goodput at conc 16/32/64 to `development/results/native_dsa_<timestamp>.json`.

### B2 — Radix cache ON under DS
Run the M3-B hardware fixture (`token_label_write --m3b-fixture-hardware-run` against real V3.2 weights + the generated mask). Cold-prefix vs warm-prefix labels must be bit-stable. On pass, flip `_double_sparsity_radix_fixture_passed = True` in the operator config (RUNBOOK Phase 5; DEC-2). Remove `--disable-radix-cache` from `serve_double_sparsity.sh`. (The cold/warm equality is even more robust at token level than page level — each token's label is a pure function of its K bytes; no within-page-mean to disturb.)

### B3 — DS bench_serving with CUDA graphs + radix cache ON
Re-run `bench_serving` at the same Option B operating point but with radix cache ON and CUDA graphs ON, conc 16/32/64. JSON to `development/results/double_sparsity_<timestamp>.json`.

### B4 — Comparator row
`python development/benchmark_compare.py --ds-results … --baseline-results …`. **Gate:** DS-on TPS ≥ DSA-on TPS at conc=64; P99 TTFT not worse than +10 % vs DSA. The comparator already enforces that only `--enable-double-sparsity` + `--double-sparsity-config` differ between baseline and DS columns.

---

## BitLesson Selection (REQUIRED FOR EACH TASK)

Before executing each task or sub-task, you MUST:

1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task to select relevant lesson IDs
3. Follow the selected lesson IDs (or `NONE`) during implementation

Include a `## BitLesson Delta` section in your summary with:
- Action: none|add|update
- Lesson ID(s): NONE or comma-separated IDs
- Notes: what changed and why (required if action is add or update)

Reference: @/sgl-workspace/sglang/.humanize/bitlesson.md

---

## Goal Tracker Rules

Throughout your work, you MUST maintain the Goal Tracker:

1. **Before starting a round**: Re-anchor on the original plan and current round contract
2. **Before starting a task**: Mark the relevant mainline task as "in_progress" in Active Tasks
   - Confirm Tag/Owner routing is correct before execution
3. **Active Tasks** are MAINLINE tasks only - side issues do not belong there
4. **Blocking Side Issues** are reserved for issues that truly stop mainline progress
5. **Queued Side Issues** are non-blocking and must not take over the round
6. **After completing a mainline task**: Move it to "Completed and Verified" with evidence (but mark as "pending verification")
7. **If you discover the plan has errors**:
   - Do NOT silently change direction
   - Add entry to "Plan Evolution Log" with justification
   - Explain how the change still serves the Ultimate Goal
8. **If you need to defer a task**:
   - Move it to "Explicitly Deferred" section
   - Provide strong justification
   - Explain impact on Acceptance Criteria
9. **If you discover new issues**:
   - Add to "Blocking Side Issues" only if mainline progress is blocked
   - Otherwise add to "Queued Side Issues" or keep them as `[queued]` tasks/backlog

---

Note: You MUST NOT try to exit `start-rlcr-loop` loop by lying or edit loop state file or try to execute `cancel-rlcr-loop`

After completing the work, please:
0. If you have access to the `code-simplifier` agent, use it to review and optimize the code you just wrote
1. Finalize @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md (this is Round 0, so you are initializing it - see "Goal Tracker Setup" above)
2. Write your round contract into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-0-contract.md
3. Commit your changes with a descriptive commit message
4. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-0-summary.md
