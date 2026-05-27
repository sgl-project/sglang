# Loop 4 Draft — Reach the Double Sparsity MVP on DeepSeek-V3.2 (FP8)

## Why this loop exists

Loop 3 set the scope at 3 items (M1/M2/M3) and never ran. The structural plumbing landed in Loops 1–2 (3,887 LOC, 150 unit tests, ABI locked, `bind_runtime_data` wired at `deepseek_v2.py:1541`), but the **data isn't flowing through it on real V3.2**:

- `page_signature_write` is never called from a live KV-write site (`grep` in `nsa_backend.py` returns zero hits).
- `sparse_mask` is not attached to `ForwardBatch`.
- No calibrated channel mask exists for DSv3.2 — `calibrate.py` works on tiny CI fixtures only.
- No multi-process TP harness — single-rank simulations only.
- No hardware-level CUDA-graph capture against a real V3.2 batch.
- Zero `bench_serving` runs against a real V3.2 model.

This loop closes those gaps and **reaches the MVP defined in [`development/past_implementations/study/06-proposed-architecture-v2.md`](../past_implementations/study/06-proposed-architecture-v2.md) §9.2**: DS-on matches or beats DSA-on at the **Option B operating point** (FP8 + flashmla_kv + overlap off + piecewise off + radix cache, both runs), with NIAH-Δ ≤ 5 pp and MMLU-Δ ≤ 1.0 pp.

The default cookbook bf16 / flashmla_sparse / fa3 / overlap / piecewise variant is **deferred** to a later loop — that's Option A scope, separately budgeted.

**Anchor:** start from `dev/double-sparsity-standalone` at the post-`06-v2` head (currently `627e7be2b`).

## Hard scope — Phase A (7 ACs, must close)

These are the items §12.6 of the v2 design doc classifies as 🛑 must-address, plus the cookbook-derived items confirmed in this loop's scoping (chunked prefill, short-seq MHA bypass).

### M1 — Live PageSignatureTable population from the KV-write path

`page_signature_write` API exists (498 LOC torch reference, FP8-scale-aware) but is never called from the live KV-write sites. Hook it at `set_mla_kv_buffer` call sites in `python/sglang/srt/layers/attention/nsa_backend.py` (Codex flagged ~L1383, ~L1583, ~L2108 in Loop 2). Retract entries on KV-free in `req_to_token_pool` deallocation.

**Chunked prefill awareness:** the same logical page is written across multiple chunks when `chunked_prefill_size < page_size * tokens_per_chunk`. The write hook must accumulate into the same signature slot (re-project on each chunk, write to the same `signature_table[L, p, h, d]` row), not overwrite.

### M2 — Per-request page ownership mask attached to ForwardBatch

Build `sparse_mask: [bs, max_pages]` from `req_to_token_pool.req_to_token`, `req_pool_indices`, `seq_lens`. Attach to `ForwardBatch.sparse_mask`. `retrieve_topk` consumes the mask before argmax so picks never escape the request's own KV range.

### V3.2 calibration code path

Adapt `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` to walk DSv3.2 MLA layers: read `self_attn.kv_b_proj` (MLA's K↑proj) on the K side; channel axis = 512 (post-RoPE nope dimension). Add `--model-arch deepseek_v3` flag. Keep the dataset + scoring identical to all three reference impls: **Pile validation, seed=42, 256 × 512 tokens, Method 1 (`mean(abs(Q·K))` per channel)**. Run the script against real V3.2 weights on the H200 cluster to **generate** `dsv32-fp8-channel-mask.safetensors`. **The mask file itself is NOT committed to git** — it lives at `/models/dsv32-fp8-channel-mask.safetensors` on the cluster. The CI tiny-fixture path must remain green.

### Multi-process two-rank TP harness

Loop 2 closed a single-process synthetic two-rank fixture; the real multi-process harness was deferred. With TP=8 in production, rank-divergent `selected_indices` produces silently-wrong attention output. Add `test/registered/integration/test_double_sparsity_tp_multiprocess.py` that spawns 2 processes via `torch.multiprocessing`, initializes a process group, runs the DEC-9 `all_reduce(SUM)` path, and asserts bit-equal `selected_indices` across ranks. Also detects a no-op all-reduce by deliberately perturbing one rank's local scores pre-reduce.

### Hardware CUDA-graph capture at conc=64

`cuda_graph.py::capture_decode_step` (full-graph capture path) is exercised only on synthetic shapes. Capture against a real V3.2 conc=64 decode batch (Option B operating point, piecewise off per scope decision). `assert_no_alloc_in_region` does not trip. Replay 100 steps without `CUDA error: launch failed`. Eager and graph outputs match on a deterministic fixture.

### Short-sequence MHA bypass for DS

DSA falls back to MHA for prefills below `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` (model's `index_topk`, ≈ 2048 for V3.2). DS must mirror this: for prefills below the threshold, the DS selector **does not run at prefill** — the path goes straight to MHA. Decode after a short prefill still runs DS (the dense prefill path has already written page signatures via the M1 hook, so the first decode step sees a populated table). This matches all three reference impls' "prefill dense, decode sparse" pattern.

### M3 — End-to-end DeepSeek-V3.2 (FP8) bench_serving with DS enabled

The **done criterion for Phase A.** Boot DSv3.2 FP8 on 8×H200 at the Option B operating point with `--enable-double-sparsity --double-sparsity-config '{"top_k":2048,"page_size":64,"channel_mask_path":"/models/dsv32-fp8-channel-mask.safetensors","device_buffer_size":4096}'`. Run `bench_serving` against it: ≥ 64 requests, ISL ≈ 4096, mixed lengths, conc 16/32/64. Then run the lightweight quality smoke (see Acceptance Criteria AC-7).

## Stretch scope — Phase B (5 ACs, attempted if Phase A closes early)

Phase B is the "MVP done if possible" arc — match or beat DSA at the Option B operating point with quality deltas inside budget.

### B1 — DSA baseline run at the Option B operating point
DSv3.2 default, TP=8, FP8 explicit (`--kv-cache-dtype fp8_e4m3 --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv`), `--disable-overlap-schedule --disable-piecewise-cuda-graph`, radix cache ON, full-graph CUDA graphs ON. Save TPS / TTFT / TPOT / goodput at conc 16/32/64 to `development/results/native_dsa_<timestamp>.json`.

### B2 — Radix cache ON under DS
Run the M3-B hardware fixture (`page_signature_write --m3b-fixture-hardware-run` against real V3.2 weights + the generated mask). Cold-prefix vs warm-prefix signatures must be bit-stable. On pass, flip `_double_sparsity_radix_fixture_passed = True` in the operator config (RUNBOOK Phase 5; DEC-2). Remove `--disable-radix-cache` from `serve_double_sparsity.sh`.

### B3 — DS bench_serving with CUDA graphs + radix cache ON
Re-run `bench_serving` at the same Option B operating point but with radix cache ON and CUDA graphs ON, conc 16/32/64. JSON to `development/results/double_sparsity_<timestamp>.json`.

### B4 — Comparator row
`python development/benchmark_compare.py --ds-results … --baseline-results …`. **Gate:** DS-on TPS ≥ DSA-on TPS at conc=64; P99 TTFT not worse than +10 % vs DSA. The comparator already enforces that only `--enable-double-sparsity` + `--double-sparsity-config` differ between baseline and DS columns.

### B5 — Full quality gate
`test/manual/test_double_sparsity_v32.py`: NIAH @ 4K / 16K / 64K must stay within **5 pp** of the DSA baseline at each length; MMLU 5-shot within **1.0 pp**.

## Acceptance Criteria

- **AC-1 (M1):** `PageSignatureTable.valid_mask` transitions False→True for newly written pages within the same forward step. Verified by both (a) a hardware-level test in `test/registered/integration/...` that runs a real forward pass and inspects the mask, and (b) the M3 benchmark not crashing on selector reads. **Chunked-prefill positive test:** a prefill of 8192 tokens at `chunked_prefill_size=4096` writes the same logical page twice (across the chunk boundary); the test asserts the signature is the cumulative-mean projection over both chunks (not the second chunk only).

- **AC-2 (M1 retract):** Running 2× the page budget of requests through the same server doesn't leak signatures — `valid_mask.sum()` stays bounded.

- **AC-3 (M2):** `sparse_mask` correctly excludes pages outside the per-request seq range. Verified with a multi-request batch test where requests have disjoint KV regions. `retrieve_topk` picks never land outside the request's KV range — verified with a kernel-level test.

- **AC-4 (calibration):** `python -m sglang.srt.layers.attention.double_sparsity.calibrate --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2 --model-arch deepseek_v3 --output /models/dsv32-fp8-channel-mask.safetensors --dtype fp8_e4m3 --page-size 64 --label-dim 16` runs to completion on the H200 cluster and writes a file that `channel_mask.py::load_channel_mask` accepts (passes shape, content-hash, and AC-4 sanity probe checks against the real V3.2 model). The tiny-CI-fixture path stays green. **The generated mask file is NOT committed to git.**

- **AC-5 (TP rank sync):** Multi-process two-rank harness spawns TP=2 processes, runs the DS path through the DEC-9 `all_reduce(SUM)` path on a deterministic fixture, asserts bit-equal `selected_indices` on both ranks. Includes a negative test that deliberately perturbs one rank's pre-reduce scores and confirms post-reduce equality (catches a no-op all-reduce).

- **AC-6 (CUDA graph hardware):** Decode-path full-graph CUDA capture at conc=64 against a real V3.2 batch (Option B operating point, piecewise off). Captures without `CUDA error: launch failed` and replays for at least 100 steps on a fixed batch. `assert_no_alloc_in_region` does not trip. Eager path produces identical output to graph replay on a deterministic fixture (`max_abs_diff <= 1e-6`).

- **AC-7 (short-seq MHA bypass):** For a prefill of length below `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`, DS does not call the selector during prefill — the path goes straight to MHA. Verified by a forward-pass hook test that asserts `DoubleSparsitySelector.retrieve_topk` is not invoked at prefill below the threshold. Decode after the short prefill **does** run DS (asserted by the same test on the first decode step).

- **AC-8 (M3 end-to-end):** `bench_serving` against the booted V3.2 FP8 server at the Option B operating point, conc=16/32/64, ≥ 64 requests, ISL ≈ 4096, mixed lengths. **Must hold:**
  - DS-on does not crash for the duration of the benchmark.
  - `selected_pages < total_pages` on at least 90 % of decode steps (non-trivial selection).
  - `dense_fallback_total` matches the `error_containment` counter accounting (no silent fallback).
  - **Lightweight quality smoke** (per `06-proposed-architecture-v2.md` §9.4): ~20 deterministic prompts (5 short QA, 5 code completion, 5 summarization, 5 NIAH-mini), `temperature=0`, reference outputs cached from DSA-on. DS-on candidates must satisfy: prefix-match ≥ 80 %, mean ROUGE-L ≥ 0.85, NIAH-mini needle recall ≥ 4/5, no first-8-tokens-entirely-different prompt.

- **AC-9 (Phase B baseline) — STRETCH:** DSA baseline run committed at the Option B operating point, conc=16/32/64; comparator JSON produced.

- **AC-10 (DEC-2 flip) — STRETCH:** M3-B hardware fixture passes against real V3.2 + generated mask; operator config has `_double_sparsity_radix_fixture_passed = True`; `serve_double_sparsity.sh` no longer sets `--disable-radix-cache`.

- **AC-11 (Phase B comparator) — STRETCH:** Comparator emits a green row at conc=64: DS-on TPS ≥ DSA-on TPS, P99 TTFT ≤ DSA-on P99 TTFT × 1.10. The only flag differences between baseline and DS columns are `--enable-double-sparsity` and `--double-sparsity-config`.

- **AC-12 (Phase B quality) — STRETCH:** NIAH @ 4K / 16K / 64K within 5 pp of DSA baseline at each length; MMLU 5-shot within 1.0 pp.

- **AC-13 (regression):** All 150 Loop-2 unit tests continue to pass.

## Explicit non-goals

These are deferred per `06-proposed-architecture-v2.md` §12 and the cookbook-scoping decisions in this loop's planning conversation:

- **Default cookbook bf16 path** (`flashmla_sparse` prefill + `fa3` decode). Stays at the Option A scope, future loop. DS validator's fp8_e4m3-only check is fine for this loop.
- **Piecewise CUDA graphs and overlap scheduler** under DS. Both default-on in V3.2 but require AC-8 multi-step backend metadata fixup work that Loop 2 deferred. Phase B explicitly disables both.
- **MTP / EAGLE speculative decoding** under DS. Out of scope — accepted that DS-on Phase B numbers will be at a lower-TPS operating point than the cookbook's MTP-on flavor.
- **GLM-5.1, 128K ISL, FP4 weights, DP Attention.** Loop 1 client deferrals.
- **Twilight top-p selection, Extensions, PD-Disagg, HiCache, CPU offload.** Downstream features.
- **Phase C kernel ports** (Triton ports of `compute_page_scores` / `page_signature_write`; fused FP8-dequant + projection; `raft_topk` adoption). Gated on Phase B5 profile evidence — only fires if Phase B shows DS-on < DSA-on TPS by > 5 % AND profile names a specific DS kernel as bottleneck.
- **AC-8 device-side value-domain assertion kernel; M3-B perturbation negative.** Test gaps, not runtime gaps.
- **`transform_index_page_table_decode_fast` 2048 hard-assert.** Should-address-in-MVP per §12.6 #9, cheap. Pulled in as a sub-task under AC-3 implementation if convenient; otherwise lands in a follow-on.
- **`SGLANG_DS_RADIX_OVERRIDE` env var.** Intentionally kept (Loop 2 task 11 deliberation).

## "Done" definition for the loop (single sentence)

Phase A: a committed round summary showing successful end-to-end DSv3.2 FP8 `bench_serving` with DS-on at the Option B operating point on 8×H200, AC-1 through AC-8 closed; **plus**, if Phase A closes by round 7, a committed comparator row from AC-11 (positive or negative) so we know whether MVP is hit.

## Carry-forward lessons from Loop 2 and Loop 3

- **BL-20260520-read-fields-before-abort-mutation:** capture batch-wide cursor spans BEFORE invoking abort helpers (`set_finish_with_abort` rewrites `req.origin_input_ids = [0]`). Relevant to AC-8 if any prompt triggers per-request abort via the `error_containment` path.
- **BL-20260520-symbol-vs-test-fixture-drift:** test fixtures must reference live dataclass field names (use `forward_batch.rids` not `req_ids`; verify with `dataclasses.fields(ForwardBatch)`). Relevant to AC-3 and AC-8 fixtures.
- **Loop 2 2-vs-6 budget mismatch:** if 2 consecutive rounds open more gaps than they close, **stop the loop manually** with `/humanize:cancel-rlcr-loop` and reassess scope. Don't wait for the circuit breaker. Phase A is 7 ACs across ~5 bench-days; if Day 3 closes < 2 ACs with > 2 new gaps open, that's the cancel signal.
- **Loop 3 lesson:** Loop 3 was budgeted at 3 ACs and never ran. The diagnosis was lack of hardware urgency. Loop 4 makes hardware verification a first-class AC criterion (AC-1b, AC-5, AC-6, AC-8); unit tests are necessary but not sufficient.

## Hardware available

- **Pod 1 (this node, rank-0):** 8× H200, `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2` available.
- **Pod 2 (rank-1, reachable via `ssh double-sparsity` or `rx devbox run --rank 1`):** another 8× H200, same model storage.
- **Both pods in sync on `dev/double-sparsity-standalone @ 627e7be2b`** at loop start.
- **For Phase A:** single 8×H200 node is sufficient. The multi-process TP harness (AC-5) uses TP=2 within a single node, not cross-node TP.
- **For Phase B:** still single-node; the Option B operating point is pure TP=8. Cross-node 16-way TP stays deferred per DEC-9 cost analysis.

## Files of interest (so plan generation doesn't re-derive them)

- **DS package:** `python/sglang/srt/layers/attention/double_sparsity/` (13 files, 3,887 LOC).
- **Page signature API (already exists, just unused at serve time):** `python/sglang/srt/layers/attention/double_sparsity/page_signature_write.py`, `page_signature_table.py`.
- **KV-write sites for AC-1 hook:** `python/sglang/srt/layers/attention/nsa_backend.py` — search for `set_mla_kv_buffer` (Codex flagged ~L1383, ~L1583, ~L2108).
- **DS selector + adapter:** `python/sglang/srt/layers/attention/double_sparsity/selector.py`, `page_table_adapter.py`, `cuda_graph.py`.
- **DSv3.2 attention hook (already wired):** `python/sglang/srt/models/deepseek_v2.py::DeepseekV2AttentionMLA._select_topk_indices` (line 2060) + `_bind_double_sparsity_runtime_data` (line 1832).
- **`forward_absorb_prepare` (DS path skip-topk gate, Loop 2 R0):** `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:245-277`.
- **Short-seq MHA threshold (AC-7 reference):** `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` env (set in `server_args.py:1808`); model's `index_topk` from `configs/model_config.py::get_dsa_index_topk`.
- **Chunked prefill (AC-1):** `server_args.py:1467-1468` (default 8192 on H200); chunk write happens via the existing prefill path that already calls `set_mla_kv_buffer`.
- **ForwardBatch (M2 attachment):** `python/sglang/srt/model_executor/forward_batch_info.py`.
- **Existing 150-test suite:** `test/registered/unit/layers/attention/test_double_sparsity_unit.py`.
- **Bench harness:** `development/serve_double_sparsity.sh`, `development/serve_native_nsa.sh`, `development/benchmark.sh`, `development/benchmark_baseline.sh`, `development/benchmark_compare.py`.
- **Calibration recipe (V3.2-only fixture):** `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` + the Pile-val-256x512-Method-1 contract from `06-proposed-architecture-v2.md` §10.
- **Quality smoke fixture (new, AC-8):** `test/manual/test_dsv32_quality_smoke.py` + the 20-prompt deterministic fixture inline in the test.
- **Full quality suite (existing, AC-12):** `test/manual/test_double_sparsity_v32.py`.
- **M3-B hardware fixture (AC-10):** `python -m sglang.srt.layers.attention.double_sparsity.page_signature_write --m3b-fixture-hardware-run`.
- **Validator (DEC-2 gate to flip):** `python/sglang/srt/layers/attention/double_sparsity/validator.py` + `_double_sparsity_radix_fixture_passed` server-args attribute.
- **Design intent and §12 deferred inventory:** `development/past_implementations/study/06-proposed-architecture-v2.md`.
- **Client SLOs:** `development/CLIENT_SLOS.md`.

## RLCR loop configuration

- **Anchor base branch:** `loop4-base` (create at `627e7be2b` at loop start).
- **Working branch:** `dev/double-sparsity-standalone` (continues).
- **Plan budget cap (advisory):** if a round closes < 2 ACs AND opens > 2 new gaps, escalate immediately. The Loop-2 R9 stagnation pattern was a slow-burn version of this.
- **Round budget:** ≤ 14 rounds. Phase A is 7 ACs; Phase B is 5 stretch. With strict per-AC hardware verification, 14 is the realistic budget — anything less is the Loop 2 mistake repeated.
- **Cancel signal:** if Day 5 of the loop hasn't closed AC-1 (M1), the design assumption about the KV-write hook sites is wrong; stop and re-survey `nsa_backend.py` before continuing.

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

CUDA graphs are ON by default in both (full-graph; piecewise off per `--disable-piecewise-cuda-graph`). Overlap scheduler is OFF in both. Radix cache is ON in DSA baseline; ON in DS after AC-10 lands.
