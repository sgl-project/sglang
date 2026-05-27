# Loop 4 Plan — Double Sparsity MVP on DeepSeek-V3.2 (FP8): Token-Level Rotation + Option B Benchmark

## Goal Description

Deliver two coupled changes on `dev/double-sparsity-standalone`, anchored at the §13-committed head (base tag: `loop4-base`).

**First:** Rotate the Double Sparsity (DS) implementation from page-level to token-level label storage. Labels are stored per physical KV slot (slot-indexed by `out_cache_loc`), written by hooking the projected 128-d nope K (`qk_nope_head_dim=128`) at the three `set_mla_kv_buffer` call sites in `dsa_backend.py`. The custom page lifecycle, within-page averaging quality delta, and most of the adapter indirection are deleted in a single rename + reshape operation. The selector emits logical sequence positions; the adapter converts them to physical token indices via `req_to_token` for FlashMLA's sparse input.

**Second:** Prove the Option B MVP — at the locked operating point (FP8, `flashmla_kv` backends, overlap off, piecewise off), DS-on must pass a full quality gate (AC-12: NIAH-Δ ≤ 5 pp at 4K/16K/64K, MMLU-Δ ≤ 1 pp), survive end-to-end `bench_serving` without crash (AC-8), and leave a Phase B stretch comparator row (AC-11) open for TPS parity measurement.

**Why token-level is the right shape:** K labels written per-slot at the `set_mla_kv_buffer` hook site are automatically slot-consistent with K and V. The KV allocator's lifecycle applies without any separate page hooks; chunked prefill writes chunk-by-chunk into the same slots; radix cache slot reuse is transparent. The adapter collapses to a `req_to_token` gather plus -1 padding.

**Why 128-d nope K:** Paper-faithful. Channel selection operates in the same space as the actual attention computation (`Q_nope · K_nope^T`). Channel importance measured in nope space directly predicts attention score contribution, unlike 512-d latent K which is an indirect proxy through the fixed `kv_b_proj` projection. The existing `calibrate.py` already uses `k_head_dim = qk_nope_head_dim = 128` and is correct. The write kernel reads `k` (k_nope, bf16/fp16) directly from the `set_mla_kv_buffer` call site — no FP8 dequantization required.

<comment>CRITIQUE [CRITICAL-1] The `k` argument at `set_mla_kv_buffer` is the MLA latent key (512-d, `kv_lora_rank=512`), not the 128-d projected nope K. The per-head projection `Q·K_nope^T` only materializes inside the attention kernel when `kv_b_proj` is applied — it is never written to the hook site. `memory_pool.py` confirms: `cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512)|scales(16)]`. Writing 128-d labels at this hook site requires applying `kv_b_proj` at write time (expensive matmul on the prefill-critical path) or finding a different hook point after projection. The entire "no FP8 dequant required" rationale is wrong: reading 128-d at this site IS NOT possible without the projection matmul. Existing `page_signature_write.py` correctly FP8-dequants the 512-d latent. This is a P0 blocker — resolve before writing any label-write code.</comment>

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
    - Token label table allocates with shape `[L_local, max_tokens, H_local, label_dim]` where `max_tokens = KV pool slot count`.
    - `DoubleSparsityConfig.top_k` defaults to 2048 (max tokens, not pages). Validator boots without error when `top_k == get_dsa_index_topk(hf_config)`.
    - `DSGraphState.selected_indices` shape is `int32[max_bs, max_top_k]`. `__init__.py` re-exports use the new names.
    - A non-contiguous physical-slot fixture (e.g., `out_cache_loc = [7, 64, 200, 512]`) is correctly scored and the adapter maps the logical top-K result to those physical slots via `req_to_token`.
  - Negative Tests (expected to FAIL):
    - Importing `PageSignatureTable` or calling `page_signature_write` raises `ImportError` or `AttributeError`.
    - Validator refuses to start when `top_k != get_dsa_index_topk(hf_config)` and `SGLANG_DS_ALLOW_TOPK_MISMATCH` is unset.

  <comment>CRITIQUE [CRITICAL-3] `validator.py` reads `server_args.nsa_prefill_backend` and `server_args.nsa_decode_backend` for the backend-KV-dtype check. Those attributes don't exist — live names are `dsa_prefill_backend` and `dsa_decode_backend` (`server_args.py:L554-557`). The validator's backend pairing check silently passes (reads `None, None`) regardless of actual config. This is a live bug that AC-0 must fix before writing any new validator assertions.</comment>

  <comment>CRITIQUE [CRITICAL-4] `device_buffer_size=4096` is passed as `max_pages` / `max_tokens` to `allocate_page_signature_table` via `deepseek_v2.py:L1887`. After token rotation, `max_tokens=4096` covers only ~5% of a typical H200 KV pool (50K–100K slots). Any slot with index ≥ 4096 triggers an out-of-bounds write or silently drops labels. The plan says "repurpose as score-scratch buffer cap" but no task updates `deepseek_v2.py:L1887` to derive `max_tokens` from the runtime pool size. Add an explicit task: derive `max_tokens = req_to_token_pool.size` at bind time; keep `device_buffer_size` as the separate scratch-cap config.</comment>

  <comment>CRITIQUE [SEQ-2] `deepseek_v2.py` changes are missing from the task table entirely. Required changes: `_bind_double_sparsity_runtime_data` (import rename, max_tokens sizing), `_select_topk_indices` (adapter ABI change after rewrite), `ds_topk_indices_out` pre-allocation resize, and the error-handler dict wiring after the adapter rewrite removes `row_errors`. None of these appear as tasks.</comment>

  <comment>CRITIQUE [TEST-2] AC-13 says "regression suite green after every code change throughout the loop." The 150 existing unit tests use page-level shapes — they will all fail during AC-0 development before migration completes. The continuous-green gate is impossible to satisfy mid-AC-0. Either migrate the tests first (as the very first commit, before any code changes) or explicitly scope AC-13 as a post-AC-0 gate only.</comment>

  <comment>CRITIQUE [CODEX-AGREE: TEST-2/3/4] Confirmed. "Green after every code change" is impossible during page→token migration unless tests migrate first (make it the first commit). TP score shape is `[bs, max_tokens]`, not flat `[max_tokens]` — a test asserting the wrong shape will fail on a correct implementation. CUDA graph safety requires preallocated output tensors because the capture itself rejects allocations, not just `assert_no_alloc_in_region`; the detector is a belt-and-suspenders check, not the primary safety mechanism.</comment>

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

  <comment>CRITIQUE [LT-2] "Freed/evicted KV slots do not produce persistent label pollution: the label at a reused slot is overwritten on the next write before the selector reads it." This claim is asserted, not proved. The old `PageSignatureTable` had an explicit `on_page_freed` → `valid_mask = False` to close the window between slot reallocation and first write. Deleting that requires proving `set_mla_kv_buffer` always fires before the selector reads for any newly allocated slot. That's true for the decode path but needs verification for every code path (prefill with fused tilelang, chunked prefill second chunk, radix-cache hit). If even one path can read before write, stale labels from a prior request will corrupt selection. The AC-2 positive test should include a synthetic freed→reallocated→read fixture that fails without valid_mask protection.</comment>

  <comment>CRITIQUE [CODEX-AGREE: LT-2/TEST-5] Confirmed. Deleting lifecycle invalidation is only safe if every read-after-allocation path writes labels first. The `save_kv_cache=False` fused tilelang path is the most likely place this breaks. AC-2 needs a stale-slot negative test, and AC-7 needs to pin the actual locked backend path for FP8 V3.2 prefill to confirm the hook fires.</comment>

- **AC-3** (M2 — per-request token range ownership prevents cross-request picks):
  - Positive Tests (expected to PASS):
    - Multi-request batch with disjoint KV regions: `retrieve_topk` picks for request B are all within request B's `req_to_token[req_pool_idx_B, :seq_len_B]` slots — no slot from request A appears.
    - Kernel-level fixture with known cross-request boundaries: inspecting `selected_token_indices` directly confirms no boundary violations.
  - Negative Tests (expected to FAIL):
    - Without the ownership mask, a fixture with two requests sharing adjacent token index ranges produces cross-request picks — confirming the mask is load-bearing.

- **AC-4** (V3.2 calibration — `dsv32-fp8-channel-mask.safetensors` generated):
  - Positive Tests (expected to PASS):
    - `python -m sglang.srt.layers.attention.double_sparsity.calibrate --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2 --model-arch deepseek_v3 --output /models/dsv32-fp8-channel-mask.safetensors --dtype bfloat16 --page-size 64 --label-dim 16` runs to completion on H200 cluster.
    - Calibration uses `k_head_dim = qk_nope_head_dim = 128` (projected nope K, matching the write kernel). Dataset: Pile validation, seed=42, 256 × 512 tokens, Method 1 (`mean(abs(Q·K))` per channel).
    - Generated file passes `channel_mask.py::load_channel_mask` validation: correct shape (`[L, H, 16]` channel selections from 128-d space), content-hash, and AC-4 sanity probe against real V3.2 model.
    - CI tiny-fixture path stays green after the `--model-arch` flag addition.
  - Negative Tests (expected to FAIL):
    - `load_channel_mask` rejects a file generated with 512-d channel indices (out-of-range for 128-d nope K space).
    - Calibration with mismatched `--label-dim` vs model's `qk_nope_head_dim` rejects with a clear error.
  - **The generated mask file is NOT committed to git.** It lives at `/models/dsv32-fp8-channel-mask.safetensors` on the H200 cluster.

  <comment>CRITIQUE [MAJOR-6] The plan says `calibrate.py` implements "Method 1 (`mean(abs(Q·K))` per channel)" and "is already correct." It isn't. The actual implementation computes L2-squared K importance: `sum_over_tokens(K_channel^2)` — no Q involvement at all (`calibrate.py:L244-246`). Method 1 requires hooking both Q and K projections and computing joint attention scores. Either fix `calibrate.py` to implement the claimed Q·K method (requires adding Q hooks), or explicitly document that Loop 4 uses L2(K) importance as a pragmatic approximation. Don't assert the script is correct when it implements a different algorithm than documented.</comment>

  <comment>CRITIQUE [MAJOR-5] The `--model-arch deepseek_v3` flag doesn't exist in `calibrate.py` — it has no `add_argument` for it. The plan says the CI tiny-fixture path "stays green after the flag addition," implying the flag must be added as part of AC-4. But `task-ac4-calibrate` doesn't make this explicit. Additionally, `calibrate.py` already auto-detects architecture via `getattr(config, "qk_nope_head_dim", 0)`. Clarify whether `--model-arch` is truly needed, or whether the auto-detect already works for V3.2, and add the flag implementation to the task description explicitly if required.</comment>

  <comment>CRITIQUE [RISK-1] Pre-loop prerequisite: read `get_dsa_index_topk(V3.2_config)` on the actual H200 cluster before writing any validator code. The plan assumes this returns 2048, but it's never verified. A one-line check (`python -c "from sglang.srt.configs.model_config import get_dsa_index_topk; from transformers import AutoConfig; print(get_dsa_index_topk(AutoConfig.from_pretrained('/cluster-storage/models/deepseek-ai/DeepSeek-V3.2')))"`) blocks a potential boot failure in AC-0.</comment>

- **AC-5** (Multi-process two-rank TP harness — bit-equal `selected_token_indices`):
  - `test/registered/integration/test_double_sparsity_tp_multiprocess.py` spawns 2 processes via `torch.multiprocessing`, initializes a process group.
  - Positive Tests (expected to PASS):
    - TP=2 processes score a deterministic fixture; `all_reduce(SUM)` fires over `[max_tokens]`-shaped score tensor; both ranks produce bit-equal `selected_token_indices` (in logical position domain, before `req_to_token` conversion).
    - Physical-slot permutation case: rank-0 and rank-1 have different physical slot assignments for the same logical sequence; after `all_reduce` in logical space and `req_to_token` conversion, physical indices are rank-specific but logical positions agree — confirms domain separation is correct.
  - Negative Tests (expected to FAIL):
    - Without `all_reduce` (mock no-op): perturbed rank produces divergent `selected_token_indices` — confirms `all_reduce` is load-bearing.

  <comment>CRITIQUE [TEST-3] AC-5 says `all_reduce(SUM)` fires over `[max_tokens]`-shaped score tensor. The actual `all_reduce_page_scores` in `selection_kernel.py:L306-311` reduces a `[bs, max_pages]` tensor (batch-keyed, not flat). After token rotation this becomes `[bs, max_tokens]` — NOT a 1-D `[max_tokens]` tensor. A test asserting a 1-D shape will fail on a correct implementation. Fix the description before writing the test fixture.</comment>

  <comment>CRITIQUE [LT-3] TP=2 on a single node uses shared-memory NCCL collectives, not the actual NCCL over NVLink/IB path used in production TP=8. Different failure modes: no dropped packets, no timeout behavior, same NUMA domain. A rank-divergence bug caused by a wrong process group config would only appear at AC-6/AC-8. If TP correctness is the goal, consider at least testing with `NCCL_P2P_DISABLE=1` to force a more production-representative collective path.</comment>

- **AC-6** (CUDA graph hardware capture at conc=64):
  - Positive Tests (expected to PASS):
    - `capture_decode_step` completes against a real V3.2 conc=64 decode batch at Option B operating point (piecewise off).
    - `assert_no_alloc_in_region` does not trip during capture.
    - 100 replay steps run without `CUDA error: launch failed`.
    - Eager path output matches graph replay on a deterministic fixture (`max_abs_diff <= 1e-6`).
  - Negative Tests (expected to FAIL):
    - With a non-preallocated scoring buffer (allocation inside capture region), the alloc detector fires — confirming it is exercised and the preallocation fix is load-bearing.

  <comment>CRITIQUE [TEST-4] `assert_no_alloc_in_region` counts PyTorch caching-allocator allocations, not in-kernel allocations. The real CUDA graph capture barrier is PyTorch's graph mode itself, which errors on any `cudaMalloc` regardless of the detector. If `compute_token_scores` allocates a transient output tensor (like `out = torch.empty(...)` at `selection_kernel.py:L155`) inside the capture region, the graph capture will fail outright — not from the detector, but from PyTorch's capture mode. The AC-6 negative test is valid but the mechanism it claims to test is muddled. The preallocation fix is load-bearing for the graph capture itself, not just for the alloc detector.</comment>

  <comment>CRITIQUE [SEQ-5] `capture_decode_step` calls `selector.retrieve_topk(queries, layer_id, req_pool_indices, sparse_mask, seq_lens)`. After AC-3 (M2 range mask), `retrieve_topk` must accept a `per_request_valid` ownership mask as an additional argument. `task-ac0-cuda-graph` doesn't mention this parameter. If `task-ac0-cuda-graph` closes before `task-m2-rangemask`, the captured graph omits the ownership mask and must be recaptured. Explicitly add the M2 mask parameter to the `task-ac0-cuda-graph` task, or move `task-ac0-cuda-graph` to depend on `task-m2-rangemask`.</comment>

- **AC-7** (Short-seq MHA bypass for DS):
  - Positive Tests (expected to PASS):
    - For a prefill below `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`, `DoubleSparsitySelector.retrieve_topk` is NOT invoked (verified by forward-pass hook).
    - Token-label write hook DOES fire during the dense prefill path → labels for short-prefill tokens are populated.
    - First decode step after short prefill: `retrieve_topk` IS invoked and returns a non-empty, non-trivial selection.
  - Negative Tests (expected to FAIL):
    - Without the bypass gate, `retrieve_topk` is called at prefill below the threshold — the test catches it.

  <comment>CRITIQUE [TEST-5] The label-write hook at `dsa_backend.py:L1439` is inside `if save_kv_cache:`. The `save_kv_cache=False` path (tilelang fused prefill) writes the KV cache directly and never calls `set_mla_kv_buffer`. For FP8 prefill on V3.2, this may be the default fast path. If `save_kv_cache=False` is used for the AC-7 short-prefill test, the token-label write hook silently never fires, and the first decode step operates on uninitialized labels. AC-7 must explicitly verify whether the FP8 V3.2 prefill path sets `save_kv_cache=True` or find the alternative hook site.</comment>

- **AC-8** (M3 — end-to-end `bench_serving` + lightweight quality smoke):
  - Boot DSv3.2 FP8 on 8×H200 at Option B with `--enable-double-sparsity --double-sparsity-config '{"top_k":2048,"page_size":64,"channel_mask_path":"/models/dsv32-fp8-channel-mask.safetensors","device_buffer_size":4096}'` and `--disable-radix-cache` (until AC-10 flips it).
  - Positive Tests (expected to PASS):
    - `bench_serving` with ≥64 requests, ISL ≈ 4096, mixed lengths, conc 16/32/64: no crash for the benchmark duration.
    - `selected_tokens.shape[1] < total_seq_len` on ≥90% of decode steps (non-trivial sparsity active).
    - `dense_fallback_total` matches `error_containment` counter accounting (no silent fallback).
    - Lightweight quality smoke (20 deterministic prompts, `temperature=0`, vs DSA-on reference outputs): prefix-match ≥ 80%; mean ROUGE-L ≥ 0.85; NIAH-mini needle recall ≥ 4/5; no prompt with first-8-tokens entirely different.
  - Negative Tests (expected to FAIL):
    - DSA-only baseline run: `error_containment` counter reads 0 — confirms it is not always-firing.

  <comment>CRITIQUE [RISK-3] The quality smoke baseline ("reference outputs cached from DSA-on") has no specification for when and how it was generated or verified. `temperature=0` does not guarantee bit-identical outputs across server restarts with FP8 KV quantization — per-block scale factors can vary if block boundaries shift. If the reference is stale or generated under different chunked-prefill conditions (pre-AC-1b probe), the `prefix-match ≥ 80%` threshold may fail for a correct DS implementation. Specify: generate the reference on the same server binary used for the DS run, same random seed, same CUDA graph warm-up, immediately before the smoke test. Document the DSA server commit SHA alongside the reference file.</comment>

  <comment>CRITIQUE [SMELL-2] The `error_containment` counter check ("dense_fallback_total matches error_containment counter") requires the adapter's `row_errors` side-channel dict to survive the < 150 LOC rewrite. Currently `expand_ds_selection_to_topk_indices` communicates per-row errors via a mutable dict pre-allocated by the caller at `deepseek_v2.py:L2077`. If the rewrite drops this dict, AC-8's `error_containment` assertion becomes untestable. Explicitly decide: keep the `row_errors` dict pattern, replace with structured exceptions and a catch-at-callsite, or remove per-row error tracking. Document the choice in `task-ac0-adapter`.</comment>

- **AC-9** (STRETCH — DSA baseline at Option B):
  - DSv3.2 default, TP=8, FP8 explicit, `flashmla_kv` both sides, overlap off, piecewise off, radix ON, CUDA graphs ON. TPS/TTFT/TPOT/goodput at conc 16/32/64 saved to `development/results/native_dsa_<timestamp>.json`.

- **AC-10** (STRETCH — radix cache ON under DS):
  - M3-B hardware fixture (`token_label_write --m3b-fixture-hardware-run`) passes against real V3.2 + generated mask. Cold-prefix vs warm-prefix labels are bit-stable. Operator config: `_double_sparsity_radix_fixture_passed = True`. `serve_double_sparsity.sh` removes `--disable-radix-cache`.

  <comment>CRITIQUE [RISK-4] "Cold-prefix vs warm-prefix labels are bit-stable" assumes FP8 quantization scale factors are identical for a token written in isolation vs. within a fully packed KV page. This is not guaranteed — FP8 block quantization assigns per-block scale factors, and a token at the tail of a partially-filled page may get a different scale than the same token in a full page. The "pure function of K bytes" claim is only true if KV bytes are bit-identical across write contexts, which requires verifying FP8 quantization is applied the same way regardless of block-fill level. If scale factors differ, warm-prefix hits will use stale, misscaled labels. Linus would note: you're calling an assumption a guarantee.</comment>

- **AC-11** (STRETCH — DS vs DSA comparator row):
  - `python development/benchmark_compare.py --ds-results ... --baseline-results ...`. Gate: **directional with ≥5% tolerance** — DS-on TPS within 5% of DSA-on TPS passes; regression ≥5% triggers a profiling obligation (not an automatic loop failure). P99 TTFT ≤ DSA-on P99 TTFT × 1.10. Only `--enable-double-sparsity` and `--double-sparsity-config` differ between columns.

  <comment>CRITIQUE [LT-4] "DS-on TPS within 5% of DSA-on TPS" is not falsifiable as written. A single `bench_serving` run at conc=64 with ISL=4096 has P99 TTFT variance that can swing 5%+ from measurement noise alone. The gate needs: (1) fixed random seed for the request arrival process, (2) minimum run duration (e.g., 600s), (3) warm-up period before measurement window, (4) number of independent runs and aggregation method (median? min?). Without these, whether the comparator row says "pass" or "fail" depends on which noisy sample happens to be measured on that cluster run.</comment>

  <comment>CRITIQUE [CODEX-AGREE: RISK-3/LT-4] Confirmed. Both quality smoke and performance gates need reproducibility mechanics: generate DSA references immediately before the DS comparison run (same server restart), record commit SHA + full server args + chunking config alongside each result, use fixed request arrival seeds, define warmup duration and measurement window, and aggregate at least 3 runs before reporting a pass/fail. Without this the gates are noise-sensitive — a run that passes one hour may fail the next.</comment>

- **AC-12** (HARD — full quality gate; moved from stretch):
  - `test/manual/test_double_sparsity_v32.py`. Runs both DS and DSA in quality-test mode (no separate AC-9 JSON required for quality comparison). NIAH @ 4K/16K/64K: DS-on delta ≤ 5 pp at each length vs DSA baseline. MMLU 5-shot: DS-on delta ≤ 1.0 pp vs DSA baseline. **Loop does not close without AC-12 passing.**

- **AC-13** (Regression — 150 Loop-2 unit tests pass after shape migration):
  - Positive Tests (expected to PASS):
    - `pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py` — all 150 tests pass, same count, after AC-0 shape updates.
  - Negative Tests (expected to FAIL):
    - Running the suite against the pre-AC-0 codebase (page-shaped) fails on shape assertions — confirms tests actually check the shapes being migrated.

---

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The implementation delivers AC-0 through AC-8 (architecture rotation + end-to-end bench_serving), AC-12 (full quality gate), AC-13 (regression), plus all Phase B stretch ACs (AC-9 through AC-11) if Phase A + AC-12 close before round 8. The AC-0 rotation covers all six affected files in the `double_sparsity/` package, the adapter rewrite to < 150 LOC, all three `dsa_backend.py` hook sites, and the 150-test migration. Phase B stretch is fully attempted: DSA baseline JSON, radix cache flip, and comparator row.

### Lower Bound (Minimum Acceptable Scope)

AC-0 through AC-8, AC-12, and AC-13 are closed. Phase B stretch ACs (AC-9 through AC-11) are deferred to a future loop if Phase A + AC-12 exhaust the round budget. The AC-1b probe result is recorded regardless of outcome (chunked-prefill behavior documented). AC-12 is hard: the loop is not declared done without it, even if it requires extending past AC-8.

### Allowed Choices

- **Hook implementation:** Python-level hook at `set_mla_kv_buffer` call sites, reading `k` (k_nope) and `cache_loc` directly. Triton kernel for `token_label_write` acceptable if eager is a bottleneck.
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

At each `set_mla_kv_buffer(layer, cache_loc, k, k_rope)` call in `dsa_backend.py`, `k` is the 128-d projected nope K in bf16/fp16 with shape `[num_tokens, H_local, qk_nope_head_dim]`. The label write projects it to `label_dim=16` selected channels:

<comment>CRITIQUE [CRITICAL-1 repeated in code path] This `k` is NOT 128-d projected nope K. It is the MLA latent key, `kv_lora_rank=512` shaped `[num_tokens, 1, 512]` (or with FP8 scales: `[num_tokens, 1, 528]`). `forward_mla.py` shows `k_nope = latent_cache[..., :kv_lora_rank]` where `kv_lora_rank=512`. The per-head 128-d projection only materializes inside the attention computation via `kv_b_proj`. The shape claim `[num_tokens, H_local, qk_nope_head_dim]` is wrong for this hook site — it would be `[num_tokens, H_local, 128]` only AFTER `kv_b_proj` is applied. You cannot read this from `set_mla_kv_buffer` without adding the projection. Fix this before any implementation starts.</comment>

```python
# Conceptual token_label_write — reads k_nope directly, no FP8 dequant needed
labels = k_nope[:, :, channel_selection[layer_id]] * channel_weights[layer_id]
# labels: [num_tokens, H_local, label_dim]
table.signatures[layer_id, cache_loc, :, :] = labels
table.valid_mask[layer_id, cache_loc] = True
```

**Selector flow with logical → physical conversion (page_table_adapter, AC-0):**

The scorer operates in physical-slot space (iterating `req_to_token[req_pool_idx, :seq_len]` slots), reduces scores per request, and runs top-K. The selector returns **logical positions** (0-indexed within the request's sequence). The adapter then resolves to physical token indices:

```python
# retrieve_topk returns logical positions [bs, K], sequence-ascending, -1 padded
logical_topk, valid_lens = selector.retrieve_topk(...)

# Adapter: logical → physical via req_to_token
physical_slots = req_to_token[req_pool_indices[:, None], logical_topk]  # [bs, K]
physical_slots = torch.where(logical_topk == -1, torch.tensor(-1), physical_slots)

# FlashMLA sparse path (Option B, flashmla_kv) consumes physical_slots as
# int32[bs, get_dsa_index_topk] flattened physical token indices, -1 padded
```

<comment>CRITIQUE [RISK-2] The plan describes physical_slots as "flattened physical token indices" but FlashMLA at `dsa_backend.py:L1864-L1874` expects indices into `kv_cache.view(-1, 64, ...)` — each index selects a 64-token block, not a single token slot. Slot ID 128 means block 2 (slots 128-191), but passing `indices[i]=128` to FlashMLA would select block 128 (slots 8192-8255). The adapter must either output `selected_slots // page_size` (page-block indices) or FlashMLA's interface handles slot→block conversion internally. Verify which form `dsa_indexer.py` actually passes to the FlashMLA kernel before writing the adapter.</comment>

**Calibration (AC-4) — channel axis is 128:**

The `--model-arch deepseek_v3` flag in `calibrate.py` reads `self_attn.kv_b_proj` on the K side and hooks the projected nope K output (shape `[-1, num_heads, qk_nope_head_dim=128]`). Importance is `mean(abs(Q_nope · K_nope))` per channel dimension, exactly Method 1 from `07-mvp-proposed-architecture.md §10`. No changes to the importance metric — only the model-arch traversal and the confirmed `k_head_dim=128` (already correct in existing `calibrate.py`).

<comment>CRITIQUE [MAJOR-6] "No changes to the importance metric" is incorrect. `calibrate.py` computes L2-squared K importance (`sum_over_tokens(K_channel^2)`, lines L244-246) — not Q·K. Claiming it "is correct" and "already implements Method 1" is false. If Method 1 (Q·K joint scoring) is the required calibration contract, `calibrate.py` needs significant changes: hook the Q projection output in addition to K, compute the joint per-channel score, and aggregate. This is not a trivial addition and is not reflected in `task-ac4-calibrate`'s scope.</comment>

### Relevant References

- `python/sglang/srt/layers/attention/double_sparsity/` — AC-0 rename targets: `page_signature_table.py`, `page_signature_write.py`, `page_table_adapter.py`, `selection_kernel.py`, `selector.py`, `cuda_graph.py`, `config.py`, `validator.py`
- `python/sglang/srt/layers/attention/dsa_backend.py` L1439, L1637, L2162 — three live `set_mla_kv_buffer` hook sites for AC-1
- `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` — reference for FlashMLA sparse token index input format (Option B adapter target)
- `python/sglang/srt/models/deepseek_v2.py` L1832 (`_bind_double_sparsity_runtime_data`), L2018 (`_select_topk_indices`) — selector dispatch wiring
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:245` (`forward_absorb_prepare`) — short-seq MHA bypass gate (AC-7)
- `python/sglang/srt/model_executor/forward_batch_info.py` — `ForwardBatch`, M2 `req_to_token` attachment (AC-3)
- `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` — update for `--model-arch deepseek_v3`, `kv_b_proj` K-side traversal (AC-4)
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — 150-test regression suite (AC-13)
- `test/manual/test_double_sparsity_v32.py` — full quality suite (AC-12)
- `test/manual/test_dsv32_quality_smoke.py` — lightweight quality smoke fixture for AC-8 (new file, 20 prompts)
- `development/serve_double_sparsity.sh`, `development/benchmark.sh`, `development/benchmark_compare.py` — bench harness (AC-8, AC-9, AC-11)
- `development/past_implementations/study/07-mvp-proposed-architecture.md` — design intent, §9.4 quality smoke spec, §10 calibration contract, §12 non-goals, §13 token-level rotation spec
- `development/CLIENT_SLOS.md` — client SLOs

---

## Dependencies and Sequence

### Milestones

1. **Architecture Rotation (Phase A1 — prerequisite for everything)**
   - AC-0: file renames, shape updates, adapter rewrite, validator, config, 150-test migration
   - AC-13: regression suite green after every code change throughout the loop

2. **Data Flow and Selection (Phase A2 — parallel after A1)**
   - AC-1 + AC-1b: M1 hook at dsa_backend.py (3 sites), hardware population test, chunked-prefill probe
   - AC-2: token-label cache lifetime, HBM boot log
   - AC-3: M2 per-request token range ownership
   - AC-7: short-seq MHA bypass

3. **Calibration + TP (Phase A3 — requires A2)**
   - AC-4: update calibrate.py, hardware run on H200, generate mask file
   - AC-5: multi-process TP=2 test (requires M1 + M2 wired)

4. **Graph Capture + End-to-End (Phase A4 — requires A3)**
   - AC-6: CUDA graph hardware capture at conc=64 (requires full decode path: M1, M2, AC-7)
   - AC-8: end-to-end bench_serving + lightweight quality smoke (requires AC-6 + calibrated mask)

5. **Quality Gate (Phase A5 — requires A4)**
   - AC-12: full NIAH/MMLU quality gate (hard; requires bench infra from AC-8 and DSA quality baseline)

6. **Phase B Stretch (requires Phase A5)**
   - AC-9: DSA benchmark JSON (can run in parallel with AC-12 quality gate)
   - AC-10: radix cache ON (requires AC-8 passing)
   - AC-11: comparator row (requires AC-9 + AC-10)

Dependencies: AC-0 → everything. AC-1 → AC-2, AC-3, AC-7, AC-5, AC-6. AC-4 → AC-6, AC-8. AC-6 → AC-8. AC-8 → AC-1b, AC-12, AC-9, AC-10. AC-10 → AC-11. AC-9 → AC-11.

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
| task-ac0-kernel | selection_kernel.py: compute_page_scores→compute_token_scores; valid_mask [L,P]→[L,T]; select_topk_sequence_order operates on tokens | AC-0 | coding | task-ac0-rename |
| task-ac0-adapter | Rewrite page_table_adapter.py to < 150 LOC: retrieve_topk emits logical positions; adapter converts via `req_to_token` gather to `int32[bs, get_dsa_index_topk]` physical token indices for FlashMLA sparse input | AC-0 | coding | task-ac0-kernel |
| task-ac0-config | DoubleSparsityConfig: top_k = max tokens (not pages), default 2048; device_buffer_size repurposed as score-scratch buffer cap; update parse_double_sparsity_config | AC-0 | coding | task-ac0-rename |
| task-ac0-validator | Boot assert: `top_k == get_dsa_index_topk(hf_config)`; respect `SGLANG_DS_ALLOW_TOPK_MISMATCH=1` override | AC-0 | coding | task-ac0-config |

<!-- CRITIQUE [SEQ-2 + SMELL-1] Missing task: update `deepseek_v2.py` for the AC-0 rotation. Required changes: (1) `_bind_double_sparsity_runtime_data`: rename `page_signature_table` import, derive `max_tokens` from `req_to_token_pool.size` instead of `device_buffer_size=4096`, pass to new `TokenLabelTable`. (2) `_select_topk_indices`: update call to new adapter ABI after rewrite. (3) `ds_topk_indices_out` pre-allocation: resize to match token-level `max_top_k`. (4) `row_errors` dict usage: update or remove depending on adapter error-handling decision. Without this task, the AC-0 rotation is incomplete and the server won't boot. Add: task-ac0-deepseek-v2 | coding | task-ac0-adapter -->
| task-ac0-cuda-graph | DSGraphState: selected_indices shape `[max_bs, max_top_k]`; update capture_decode_step to new selector ABI (logical positions, req_to_token adapter) | AC-0 | coding | task-ac0-kernel |
| task-ac0-tests | Migrate 150 Loop-2 unit tests for page→token shape changes; verify count preserved (AC-13 gate) | AC-0, AC-13 | coding | task-ac0-rename |
| task-m1-hook | Wire token_label_write at dsa_backend.py L1439 (forward_extend), L1637 (forward_decode), L2162 (TRT-LLM MLA); read `k` (k_nope, 128-d bf16) and `cache_loc` directly from each call site | AC-1 | coding | task-ac0-rename |
| task-ac1-hwtest | Hardware test: real forward_extend pass on H200; assert `token_label_table.signatures[layer_id, out_cache_loc]` non-zero for each written token slot | AC-1 | coding | task-m1-hook |
| task-ac2-lifetime | Add boot-time GB/rank log; 2× slot-budget test; verify slot count bounded by KV pool slot count; confirm reused-slot overwrite semantics | AC-2 | coding | task-m1-hook |
| task-m2-rangemask | ForwardBatch: attach per-request token range from req_to_token_pool; scorer masks out-of-range slots to −inf before top-K | AC-3 | coding | task-ac0-kernel |
| task-ac3-test | Multi-request boundary kernel test: confirm picks never cross request KV range; negative fixture confirms mask is load-bearing | AC-3 | coding | task-m2-rangemask |
| task-ac7-bypass | Short-seq MHA bypass: DS selector not invoked for prefill below SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD; token-label write hook still fires; first decode runs DS | AC-7 | coding | task-m1-hook |
| task-ac4-calibrate | Update calibrate.py: add `--model-arch deepseek_v3`; kv_b_proj K-side traversal; confirm k_head_dim = qk_nope_head_dim = 128; keep Pile-val-256×512-Method-1 scoring; CI tiny-fixture green | AC-4 | coding | task-ac0-rename |
| task-ac4-hwrun | Hardware run on H200 cluster: generate `/models/dsv32-fp8-channel-mask.safetensors`; validate with load_channel_mask; NOT committed to git | AC-4 | analyze | task-ac4-calibrate |
| task-ac5-tp | Add test/registered/integration/test_double_sparsity_tp_multiprocess.py: TP=2 multiprocess, all_reduce(SUM) on [max_tokens] scores in logical-position space, bit-equal logical selected_token_indices; physical-slot permutation negative case | AC-5 | coding | task-m2-rangemask |
| task-ac6-cuda-graph | Decode-path full-graph capture with preallocated buffers; assert_no_alloc_in_region does not trip; 100-step replay; eager==graph deterministic; alloc-detector negative test | AC-6 | coding | task-m1-hook, task-m2-rangemask |
| task-ac6-hwrun | Hardware run: execute full-graph capture at conc=64 against real V3.2, Option B | AC-6 | analyze | task-ac6-cuda-graph, task-ac4-hwrun |
| task-ac8-server | Boot DSv3.2 FP8 TP=8 8×H200 at Option B; bench_serving conc 16/32/64, ≥64 requests, ISL ≈ 4096 | AC-8 | analyze | task-ac6-hwrun |
| task-ac1b-probe | Chunked-prefill probe: run AC-8 server with chunked_prefill_size=4096; compare token labels 0..4095 vs non-chunked baseline | AC-1b | analyze | task-ac8-server |
<!-- CRITIQUE [SEQ-4] AC-1b is scheduled after task-ac8-server, but its outcome retroactively changes AC-8's operating point. If the probe fails, `--chunked-prefill-size -1` must be added to both launch commands — meaning AC-8's bench_serving run was done with the wrong config and must be re-run. This doubles H200 cluster time for AC-8. Consider running task-ac1b-probe BEFORE the full bench_serving run in AC-8: the probe only needs the server booted and a single chunked vs. non-chunked label comparison, not the full 64-request workload. Move task-ac1b-probe to depend on task-ac6-hwrun (same server config), not task-ac8-server. -->
| task-ac8-quality | Lightweight quality smoke: 20 deterministic prompts, temperature=0, vs DSA-on reference; assert 4 quality gates | AC-8 | analyze | task-ac8-server |
| task-ac12-quality | Full quality gate: run DS and DSA in quality-test mode (NIAH @ 4K/16K/64K + MMLU 5-shot); assert deltas ≤ 5 pp / 1.0 pp | AC-12 | analyze | task-ac8-server |
| task-ac9-baseline | DSA baseline: boot DSA at Option B; bench_serving conc 16/32/64; save JSON to development/results/ | AC-9 | analyze | task-ac8-server |
| task-ac10-radix | M3-B hardware fixture; flip `_double_sparsity_radix_fixture_passed`; update serve_double_sparsity.sh | AC-10 | coding | task-ac9-baseline |
| task-ac11-compare | Run benchmark_compare.py; emit comparator row; check TPS directional gate (≥5% tolerance), P99 TTFT ≤ 1.10× | AC-11 | analyze | task-ac9-baseline, task-ac10-radix |

---

## Claude-Codex Deliberation

### Agreements

- Slot-indexed label rotation (slot = `out_cache_loc`) is the correct architecture for token-level DS.
- `dsa_backend.py` (not deprecated `nsa_backend.py`) contains the three live `set_mla_kv_buffer` hook sites at L1439, L1637, L2162.
- `nsa_backend.py`, `nsa/nsa_indexer.py`, and `nsa/transform_index.py` are deprecated re-export shims; no new code may reference them.
- `dsa/transform_index.py::transform_index_page_table_decode` asserts `page_size==1` and hard-codes `TOPK=2048`; it is unsuitable for the adapter path.
- `selected_token_indices` must be **logical sequence positions** (not physical KV slots) to ensure consistent range-mask semantics, TP sync domain, and sequence-ascending ordering.
- The adapter converts logical positions to physical token indices via `req_to_token` gather; the Option B `flashmla_kv` sparse path consumes `int32[bs, get_dsa_index_topk]` physical token indices, not a block_table.
- An HBM budget gate (boot-time GB/rank log + fail-fast) is required for the `[L,T,H,D]` token label table.
- The `top_k == get_dsa_index_topk(hf_config)` boot assert is correct and keeps Phase B comparison apples-to-apples.

### Resolved Disagreements

- **Calibration/write-space dimension (512-d vs 128-d):** Codex v1 and v2 both flagged a mismatch between `page_signature_write.py`'s `_NOPE_DIM=512` (MLA latent K, what `set_mla_kv_buffer` stores in the KV pool buffer) and `calibrate.py`'s `k_head_dim = qk_nope_head_dim = 128` (projected nope K). Both Codex passes required resolution. **User chose 128-d nope K**: paper-faithful (channel selection in attention-score space), theoretically superior (direct importance signal), and the existing `calibrate.py` is already correct. The write kernel is updated to read `k` (k_nope, bf16) directly from the `set_mla_kv_buffer` call site arguments — avoiding the FP8 dequantization detour entirely. No re-read from the KV pool buffer.

<comment>CRITIQUE [CRITICAL-1 in deliberation] This resolution rests on a false factual premise: "The write kernel is updated to read `k` (k_nope, bf16) directly from the `set_mla_kv_buffer` call site." The `k` argument at that call site is 512-d MLA latent K, not 128-d projected nope K. Reading 128-d per-head K_nope from this site requires applying `kv_b_proj` (a projection matmul) at write time — that is not "reading directly," it is adding a matmul on the write path. The decision is theoretically correct (128-d is superior) but the implementation path described here is physically impossible without the projection. This needs to be re-opened as an implementation question: (a) add `kv_b_proj` projection at write time and accept the cost, or (b) find a hook site downstream of the projection where 128-d K_nope is available. The "converged" status is premature on this specific point.</comment>
- **AC-11/AC-12 scope (stretch vs hard):** Codex recommended both hard; draft had both stretch. **User resolved: AC-11 stays stretch; AC-12 moves to hard.** Rationale: quality correctness (NIAH/MMLU deltas) is a hard MVP requirement — shipping DS with unknown quality degradation is not acceptable. TPS parity (AC-11) is a performance target with directional tolerance.
- **AC-11 TPS gate type:** Codex and Claude both treated it as binary in first pass. **User confirmed directional with ≥5% tolerance** — regression ≥5% triggers profiling obligation, not automatic loop failure.
- **`device_buffer_size` semantics:** After token rotation, the original "page buffer count" semantics are obsolete. **Resolved: repurpose as score-scratch buffer cap** (maximum concurrently live tokens for the decode scoring scratch tensor). The config value `4096` is retained as default.

### Convergence Status

- Final Status: `converged`
- Rounds: 2 Codex passes
- No REQUIRED_CHANGES remain unaddressed.

---

## Pending User Decisions

All decisions resolved during deliberation. No PENDING items.

- DEC-1: Label dimension source (512-d latent K vs 128-d nope K)
  - Claude Position: 512-d pragmatic (existing path); 128-d more correct
  - Codex Position: Required resolution; flagged both as viable with different tradeoffs
  - Decision Status: **RESOLVED — 128-d nope K (user choice; paper-faithful, theoretically superior)**

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
- Round budget: ≤ 14 rounds. Phase A (AC-0 through AC-8 + AC-12): 9+1 hard ACs; Phase B: 3 stretch.
- Plan budget cap (advisory): if a round closes < 2 ACs AND opens > 2 new gaps, escalate immediately.
- Cancel signal: if Day 3 of the loop hasn't closed AC-0, the rename + reshape estimate is wrong; stop and re-scope.

<comment>CRITIQUE [LT-5] The 14-round budget for what is primarily a rename + reshape + three hook-site writes signals a lack of confidence in the implementation estimate. Net-new code for AC-0 through AC-3 is under 200 lines. If 14 rounds are genuinely needed, the plan has significantly underspecified what is being built (likely because CRITICAL-1 and the missing deepseek_v2.py tasks are unresolved). Resolve the CRITICAL-1 question (128-d requires adding `kv_b_proj` projection vs. accepting 512-d latent) before the loop starts — if 128-d is confirmed but requires adding a projection matmul on the write path, that is a substantial new workload that should appear explicitly as a task with its own round budget.</comment>

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

### B5 — Full quality gate
`test/manual/test_double_sparsity_v32.py`: NIAH @ 4K / 16K / 64K must stay within **5 pp** of the DSA baseline at each length; MMLU 5-shot within **1.0 pp**.

## Acceptance Criteria

- **AC-0 (architecture rotation):** Token-level signatures land. `python -c "from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, retrieve_topk; ..."` works. The selector's `retrieve_topk` returns `(selected_token_indices: int32[bs, max_top_k_tokens], valid_lengths: int32[bs])`, sequence-ascending. `page_table_adapter.py` is < 150 LOC (down from 404). The renamed files preserve `__init__.py` re-exports. **Validator boot-time assert:** `DoubleSparsityConfig.top_k == get_dsa_index_topk(hf_config)` (refuses to start on mismatch unless `SGLANG_DS_ALLOW_TOPK_MISMATCH=1`). Default config carries `top_k=2048`; expected to match V3.2's intrinsic top-k.

- **AC-1 (M1):** Token-label cache slots transition from default to populated for newly written tokens within the same forward step. Verified by (a) a hardware-level test that runs a real forward pass and inspects the slots at the request's `out_cache_loc`, and (b) the M3 benchmark not crashing on selector reads.

- **AC-1b (chunked-prefill probe — NOT a code AC):** Phase A includes a one-time probe: run M3 (AC-8) once with `chunked_prefill_size=4096` (forces a ≥ 4096-token prompt across two chunks). Assert the per-token label for tokens 0..4095 written by chunk 1 is byte-equal to what it would be in a non-chunked baseline. **Pass:** chunked prefill works implicitly; AC-8 / AC-9 run with the H200 default (`chunked_prefill_size=8192`). **Fail:** AC-8 / AC-9 disable chunked prefill on both DSA baseline and DS (`--chunked-prefill-size -1`), and explicit chunked-prefill support is moved to Loop 5's scope. **No code is written in Loop 4 to fix the probe failure.**

- **AC-2 (M1 cleanup):** Token-label cache shares the KV pool's allocator lifetime. Running 2× the slot budget of requests through the same server doesn't leak labels — the slot count consumed never exceeds the KV pool's slot count.

- **AC-3 (M2):** Per-request token range mask correctly excludes tokens outside the request's seq range. Verified with a multi-request batch test where requests have disjoint KV regions. `retrieve_topk` picks never land outside the request's KV range — verified with a kernel-level test that submits a fixture with known cross-request boundaries.

- **AC-4 (calibration):** `python -m sglang.srt.layers.attention.double_sparsity.calibrate --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2 --model-arch deepseek_v3 --output /models/dsv32-fp8-channel-mask.safetensors --dtype fp8_e4m3 --page-size 64 --label-dim 16` runs to completion on the H200 cluster and writes a file that `channel_mask.py::load_channel_mask` accepts (passes shape, content-hash, and AC-4 sanity probe checks against the real V3.2 model). The tiny-CI-fixture path stays green. **The generated mask file is NOT committed to git.**

- **AC-5 (TP rank sync):** Multi-process two-rank harness spawns TP=2 processes, runs the DS path through the DEC-9 `all_reduce(SUM)` on `[max_tokens]`-shaped scores on a deterministic fixture, asserts bit-equal `selected_token_indices` on both ranks. Includes a negative test that deliberately perturbs one rank's pre-reduce scores and confirms post-reduce equality.

- **AC-6 (CUDA graph hardware):** Decode-path full-graph CUDA capture at conc=64 against a real V3.2 batch (Option B operating point, piecewise off). Captures without `CUDA error: launch failed` and replays for at least 100 steps on a fixed batch. `assert_no_alloc_in_region` does not trip. Eager path produces identical output to graph replay on a deterministic fixture.

- **AC-7 (short-seq MHA bypass):** For a prefill of length below `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`, DS does not call the selector during prefill — the path goes straight to MHA. Verified by a forward-pass hook test that asserts `DoubleSparsitySelector.retrieve_topk` is not invoked at prefill below the threshold. **However**, the token-label write hook (M1) still fires at the dense prefill path, so the labels for those tokens are populated. Decode after the short prefill **does** run DS — asserted by the same test on the first decode step.

- **AC-8 (M3 end-to-end):** `bench_serving` against the booted V3.2 FP8 server at the Option B operating point, conc=16/32/64, ≥ 64 requests, ISL ≈ 4096, mixed lengths. **Must hold:**
  - DS-on does not crash for the duration of the benchmark.
  - `selected_tokens.shape[1] < total_seq_len` on at least 90 % of decode steps (non-trivial selection).
  - `dense_fallback_total` matches the `error_containment` counter accounting (no silent fallback).
  - **Lightweight quality smoke** (per `07-mvp-proposed-architecture.md` §9.4): ~20 deterministic prompts (5 short QA, 5 code completion, 5 summarization, 5 NIAH-mini), `temperature=0`, reference outputs cached from DSA-on. DS-on candidates must satisfy: prefix-match ≥ 80 %, mean ROUGE-L ≥ 0.85, NIAH-mini needle recall ≥ 4/5, no first-8-tokens-entirely-different prompt.

- **AC-9 (Phase B baseline) — STRETCH:** DSA baseline run committed at the Option B operating point, conc=16/32/64; comparator JSON produced.

- **AC-10 (DEC-2 flip) — STRETCH:** M3-B hardware fixture passes against real V3.2 + generated mask; operator config has `_double_sparsity_radix_fixture_passed = True`; `serve_double_sparsity.sh` no longer sets `--disable-radix-cache`.

- **AC-11 (Phase B comparator) — STRETCH:** Comparator emits a green row at conc=64: DS-on TPS ≥ DSA-on TPS, P99 TTFT ≤ DSA-on P99 TTFT × 1.10. The only flag differences between baseline and DS columns are `--enable-double-sparsity` and `--double-sparsity-config`.

- **AC-12 (Phase B quality) — STRETCH:** NIAH @ 4K / 16K / 64K within 5 pp of DSA baseline at each length; MMLU 5-shot within 1.0 pp.

- **AC-13 (regression):** All 150 Loop-2 unit tests continue to pass (with shape updates for the page → token rename — same test count after migration).

## Explicit non-goals

These are deferred per `07-mvp-proposed-architecture.md` §12 and the cookbook-scoping decisions in this loop's planning conversation:

- **Default cookbook bf16 path** (`flashmla_sparse` prefill + `fa3` decode). Stays at Option A scope, future loop.
- **Piecewise CUDA graphs and overlap scheduler** under DS. Both default-on in V3.2 but require AC-8 multi-step backend metadata fixup work that Loop 2 deferred. Phase B explicitly disables both.
- **MTP / EAGLE speculative decoding** under DS. Out of scope.
- **GLM-5.1, 128K ISL, FP4 weights, DP Attention.** Loop 1 client deferrals.
- **Twilight top-p selection, Extensions, PD-Disagg, HiCache, CPU offload.** Downstream features.
- **Phase C kernel ports** (Triton ports of `compute_token_scores` / `token_label_write`; fused FP8-dequant + projection; `raft_topk` adoption). Gated on Phase B5 profile evidence — only fires if Phase B shows DS-on < DSA-on TPS by > 5 % AND profile names a specific DS kernel as bottleneck.
- **AC-8 device-side value-domain assertion kernel; M3-B perturbation negative.** Test gaps, not runtime gaps.
- **Page-level signature design at 1M context.** Documented in §3.3 and recoverable from git history; not in scope until a client asks for 256K+ context.
- **`SGLANG_DS_RADIX_OVERRIDE` env var.** Intentionally kept (Loop 2 task 11 deliberation).

- **Explicit chunked-prefill support code.** Loop 4 only *probes* (AC-1b). If the probe fails, the baseline disables chunked prefill and explicit code becomes **Loop 5 scope**. The DSA cookbook default at the Option B operating point uses `chunked_prefill_size=8192` (auto on H200 per `_handle_gpu_memory_settings`); if Loop 4's baseline has to set this to `-1` because of the probe, that's a known operating-point asymmetry — to be closed in Loop 5, not Loop 4.

## "Done" definition for the loop (single sentence)

Phase A: a committed round summary showing successful end-to-end DSv3.2 FP8 `bench_serving` with DS-on at the Option B operating point on 8×H200, AC-0 through AC-8 closed; **plus**, if Phase A closes by round 8, a committed comparator row from AC-11 (positive or negative) so we know whether MVP is hit.

## Carry-forward lessons from Loops 1–3

- **BL-20260520-read-fields-before-abort-mutation:** capture batch-wide cursor spans BEFORE invoking abort helpers (`set_finish_with_abort` rewrites `req.origin_input_ids = [0]`). Relevant to AC-8 if any prompt triggers per-request abort via the `error_containment` path.
- **BL-20260520-symbol-vs-test-fixture-drift:** test fixtures must reference live dataclass field names (use `forward_batch.rids` not `req_ids`; verify with `dataclasses.fields(ForwardBatch)`). Relevant to AC-3 and AC-8 fixtures.
- **Loop 2 2-vs-6 budget mismatch:** if 2 consecutive rounds open more gaps than they close, **stop the loop manually** with `/humanize:cancel-rlcr-loop` and reassess scope. Phase A is 9 ACs (including AC-0) across ~5 bench-days; if Day 3 closes < 2 ACs with > 2 new gaps open, that's the cancel signal.
- **Loop 3 lesson:** Loop 3 was budgeted at 3 ACs and never ran. The diagnosis was lack of hardware urgency. Loop 4 makes hardware verification a first-class AC criterion (AC-1 hw test, AC-4 real calibration, AC-5 multi-process, AC-6 conc=64 capture, AC-8 bench_serving); unit tests are necessary but not sufficient.
- **AC-0 risk:** the architecture rotation is the first AC because it's load-bearing for all the others. If AC-0 takes more than 2 rounds to close, that's a signal the rotation is harder than estimated and Phase B should be cut from scope.

## Hardware available

- **Pod 1 (this node, rank-0):** 8× H200, `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2` available.
- **Pod 2 (rank-1, reachable via `ssh double-sparsity` or `rx devbox run --rank 1`):** another 8× H200, same model storage.
- **Both pods in sync on `dev/double-sparsity-standalone @ <head with §13>`** at loop start.
- **For Phase A:** single 8×H200 node is sufficient. The multi-process TP harness (AC-5) uses TP=2 within a single node.
- **For Phase B:** still single-node; Option B is pure TP=8. Cross-node 16-way TP stays deferred per DEC-9.

## Files of interest (so plan generation doesn't re-derive them)

- **DS package (subject to AC-0 rename):** `python/sglang/srt/layers/attention/double_sparsity/`
  - `page_signature_table.py` → `token_label_table.py`
  - `page_signature_write.py` → `token_label_write.py`
  - `page_table_adapter.py` → mostly deleted; thin wrapper around NSA's `transform_index_page_table_decode`
  - `selection_kernel.py`, `selector.py`, `cuda_graph.py`, `config.py` → in-place shape updates
- **NSA helper to reuse (AC-0):** `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` and the `transform_index_page_table_decode` helper (search for the symbol; it converts token indices to page-aligned block tables).
- **KV-write sites for AC-1 hook:** `python/sglang/srt/layers/attention/nsa_backend.py` — search for `set_mla_kv_buffer` (Codex flagged ~L1383, ~L1583, ~L2108).
- **DS attention hook (already wired):** `python/sglang/srt/models/deepseek_v2.py::DeepseekV2AttentionMLA._select_topk_indices` (line 2060) + `_bind_double_sparsity_runtime_data` (line 1832).
- **`forward_absorb_prepare` (DS path skip-topk gate, Loop 2 R0):** `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:245-277`.
- **Short-seq MHA threshold (AC-7 reference):** `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` env (set in `server_args.py:1808`); model's `index_topk` from `configs/model_config.py::get_dsa_index_topk`.
- **ForwardBatch (M2 attachment):** `python/sglang/srt/model_executor/forward_batch_info.py`.
- **Existing 150-test suite (subject to AC-0 shape updates):** `test/registered/unit/layers/attention/test_double_sparsity_unit.py`.
- **Bench harness:** `development/serve_double_sparsity.sh`, `development/serve_native_nsa.sh`, `development/benchmark.sh`, `development/benchmark_baseline.sh`, `development/benchmark_compare.py`.
- **Calibration recipe (V3.2-only adaptation):** `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` + the Pile-val-256x512-Method-1 contract from `07-mvp-proposed-architecture.md` §10.
- **Quality smoke fixture (new, AC-8):** `test/manual/test_dsv32_quality_smoke.py` + the 20-prompt deterministic fixture inline in the test.
- **Full quality suite (existing, AC-12):** `test/manual/test_double_sparsity_v32.py`.
- **M3-B hardware fixture (AC-10):** `python -m sglang.srt.layers.attention.double_sparsity.token_label_write --m3b-fixture-hardware-run` (renamed from `page_signature_write`).
- **Validator (DEC-2 gate to flip):** `python/sglang/srt/layers/attention/double_sparsity/validator.py` + `_double_sparsity_radix_fixture_passed` server-args attribute.
- **Design intent + §13 rotation note:** `development/past_implementations/study/07-mvp-proposed-architecture.md`.
- **Client SLOs:** `development/CLIENT_SLOS.md`.

## RLCR loop configuration

- **Anchor base branch:** `loop4-base` (create at the §13-committed head at loop start).
- **Working branch:** `dev/double-sparsity-standalone` (continues).
- **Plan budget cap (advisory):** if a round closes < 2 ACs AND opens > 2 new gaps, escalate immediately.
- **Round budget:** ≤ 14 rounds. Phase A is 9 ACs (including AC-0); Phase B is 5 stretch.
- **Cancel signal:** if Day 3 of the loop hasn't closed AC-0 (the rotation), the rename + reshape estimate is wrong; stop and re-scope before continuing.

## Operating-point cheatsheet (Option B, locked)

For all of Phase A + Phase B, both DSA baseline and DS runs use this operating point. Any deviation breaks the apples-to-apples requirement.

```bash
# DSA baseline (Phase B AC-9):
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3.2 \
  --tp 8 \
  --kv-cache-dtype fp8_e4m3 \
  --dsa-prefill-backend flashmla_kv \
  --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule \
  --disable-piecewise-cuda-graph \
  --page-size 64 \
  --trust-remote-code

# DS (Phase A AC-8 / Phase B AC-11):
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3.2 \
  --tp 8 \
  --kv-cache-dtype fp8_e4m3 \
  --dsa-prefill-backend flashmla_kv \
  --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule \
  --disable-piecewise-cuda-graph \
  --page-size 64 \
  --trust-remote-code \
  --enable-double-sparsity \
  --double-sparsity-config '{"top_k":2048,"page_size":64,"channel_mask_path":"/models/dsv32-fp8-channel-mask.safetensors","device_buffer_size":4096}'
  # Phase A: also pass --disable-radix-cache (until AC-10 flips it).
  # Phase B (after AC-10): remove --disable-radix-cache; radix cache ON.
```

CUDA graphs are ON by default in both (full-graph; piecewise off per `--disable-piecewise-cuda-graph`). Overlap scheduler OFF in both. Radix cache ON in DSA baseline; ON in DS after AC-10.

**Chunked prefill (conditional on AC-1b outcome):**
- If AC-1b probe **passes** (chunked prefill works implicitly under token-level DS): leave the H200 auto-default (`chunked_prefill_size=8192`) on both DSA and DS.
- If AC-1b probe **fails**: append `--chunked-prefill-size -1` to **both** launch commands. The asymmetry vs the cookbook default is intentional and recorded in the Loop 4 round summary. Explicit chunked-prefill support is then Loop 5 scope.

Note: `top_k=2048` in the DS config is now **max tokens** (matches sglang-last's `--ds-heavy-token-num` semantics post-AC-0), not max pages. At page_size=64, that's at most 32 pages of FlashMLA `block_table` after `transform_index_page_table_decode` (in the worst case where each of the 2048 tokens lands in a distinct page — typically fewer pages because tokens cluster).

The validator asserts `top_k == get_dsa_index_topk(hf_config)` at boot. For V3.2 this is expected to be 2048; if a future model has a different intrinsic top-k, the operator must update the config or set `SGLANG_DS_ALLOW_TOPK_MISMATCH=1` to acknowledge the asymmetry. This keeps the Phase B comparison apples-to-apples on selection size.

--- Original Design Draft End ---
