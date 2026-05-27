# Loop 4 Draft — Reach the Double Sparsity MVP on DeepSeek-V3.2 (FP8)

## Why this loop exists

**Framing:** the implementation we are landing is **sglang-last's DS algorithm (per-token sparsity, dense prefill, sparse decode, per-token K_label cache slot-indexed by `out_cache_loc`) — but performant**. The performance knobs sglang-last couldn't support (FlashMLA backend, FP8 KV cache, MLA model, CUDA graphs, radix cache, page sizes including 64, multi-process TP rank sync, mixed batches via per-request range masks) come "for free" from inheriting the rest of sglang's modern infrastructure once the selection granularity is right. The win vs sglang-last is not a different algorithm — it's the same algorithm at the modern operating point.

Loop 3 set the scope at 3 items (M1/M2/M3) and never ran. The structural plumbing landed in Loops 1–2 (3,887 LOC, 150 unit tests, ABI locked, `bind_runtime_data` wired at `deepseek_v2.py:1541`), but the **data isn't flowing through it on real V3.2** and the **selection granularity was page-level, paying complexity for memory savings that only matter at 1M context** (no client ask).

Loop 4 makes two changes at once:

1. **Rotates the architecture to token-level signatures at page_size=64** (per `06-proposed-architecture-v2.md` §13). Selection is per-token, storage is per-token (slot-indexed by `out_cache_loc` exactly like K and V), FlashMLA still reads at page granularity via NSA's existing `transform_index_page_table_decode`. This deletes the custom page adapter, the page lifecycle hooks, and the within-page averaging quality delta — all in one move, by becoming the same shape sglang-last had.
2. **Reaches the MVP at the Option B operating point** (FP8 + `flashmla_kv` + overlap off + piecewise off, both DSA baseline and DS at the same operating point): DS-on matches or beats DSA-on TPS at conc=64 with NIAH-Δ ≤ 5 pp and MMLU-Δ ≤ 1 pp.

**Chunked prefill is *probed*, not actively supported.** sglang-last got chunked prefill for free because its K_label cache was slot-indexed by `out_cache_loc`. The token-level rotation preserves that property, so chunked prefill *should* work implicitly. **Phase A includes a probe test** that asserts implicit support. **If the probe fails**, the Phase B comparison disables chunked prefill on both DSA and DS (`--chunked-prefill-size -1`) and chunked-prefill explicit support becomes Loop 5 scope. **No explicit chunked-prefill code lands in Loop 4 regardless of the probe outcome** — either it works implicitly and we leave it alone, or it doesn't and the baseline gets adjusted.

**Anchor:** start from `dev/double-sparsity-standalone` at the head with §13 committed.

## Hard scope — Phase A (8 ACs, must close)

### AC-0 — Architecture rotation: token-level signatures, page_size=64 stays

The first round closes the rotation. Subsequent ACs assume token-level.

- Rename `page_signature_table.py` → `token_label_table.py`. Shape changes from `[L_local, max_pages, H_local, label_dim]` to `[L_local, max_tokens, H_local, label_dim]`. The "max_tokens" is the KV pool's slot count — no separate lifecycle (the KV allocator already manages it).
- Rename `page_signature_write.py` → `token_label_write.py`. Per-token FP8 dequant + channel projection — no page-mean reduction. Slot-indexed by `out_cache_loc`, exactly parallel to K/V writes.
- `selection_kernel.py`: `compute_page_scores` → `compute_token_scores` (BGEMV `Q_label · K_label_bufferᵀ`). `select_topk_sequence_order` operates on tokens.
- `page_table_adapter.py`: collapse to a thin wrapper around NSA's existing `transform_index_page_table_decode` (the same helper NSA uses). Most of the 404 LOC deletes.
- `selector.py::retrieve_topk` return type: `(selected_token_indices: int32[bs, max_top_k_tokens], valid_lengths: int32[bs])`, sequence-ascending with `-1` padding.
- `cuda_graph.py::DSGraphState`: static `selected_token_indices: int32[max_bs, max_top_k_tokens]`. Shape change only; same machinery.
- `config.py`: `top_k` semantics now max **tokens** per request (matches sglang-last's `--ds-heavy-token-num`; same name kept). Default 2048.

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

## Stretch scope — Phase B (5 ACs, attempted if Phase A closes early)

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

- **AC-0 (architecture rotation):** Token-level signatures land. `python -c "from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, retrieve_topk; ..."` works. The selector's `retrieve_topk` returns `(selected_token_indices: int32[bs, max_top_k_tokens], valid_lengths: int32[bs])`, sequence-ascending. `page_table_adapter.py` is < 150 LOC (down from 404). The renamed files preserve `__init__.py` re-exports.

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
  - **Lightweight quality smoke** (per `06-proposed-architecture-v2.md` §9.4): ~20 deterministic prompts (5 short QA, 5 code completion, 5 summarization, 5 NIAH-mini), `temperature=0`, reference outputs cached from DSA-on. DS-on candidates must satisfy: prefix-match ≥ 80 %, mean ROUGE-L ≥ 0.85, NIAH-mini needle recall ≥ 4/5, no first-8-tokens-entirely-different prompt.

- **AC-9 (Phase B baseline) — STRETCH:** DSA baseline run committed at the Option B operating point, conc=16/32/64; comparator JSON produced.

- **AC-10 (DEC-2 flip) — STRETCH:** M3-B hardware fixture passes against real V3.2 + generated mask; operator config has `_double_sparsity_radix_fixture_passed = True`; `serve_double_sparsity.sh` no longer sets `--disable-radix-cache`.

- **AC-11 (Phase B comparator) — STRETCH:** Comparator emits a green row at conc=64: DS-on TPS ≥ DSA-on TPS, P99 TTFT ≤ DSA-on P99 TTFT × 1.10. The only flag differences between baseline and DS columns are `--enable-double-sparsity` and `--double-sparsity-config`.

- **AC-12 (Phase B quality) — STRETCH:** NIAH @ 4K / 16K / 64K within 5 pp of DSA baseline at each length; MMLU 5-shot within 1.0 pp.

- **AC-13 (regression):** All 150 Loop-2 unit tests continue to pass (with shape updates for the page → token rename — same test count after migration).

## Explicit non-goals

These are deferred per `06-proposed-architecture-v2.md` §12 and the cookbook-scoping decisions in this loop's planning conversation:

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
- **Calibration recipe (V3.2-only adaptation):** `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` + the Pile-val-256x512-Method-1 contract from `06-proposed-architecture-v2.md` §10.
- **Quality smoke fixture (new, AC-8):** `test/manual/test_dsv32_quality_smoke.py` + the 20-prompt deterministic fixture inline in the test.
- **Full quality suite (existing, AC-12):** `test/manual/test_double_sparsity_v32.py`.
- **M3-B hardware fixture (AC-10):** `python -m sglang.srt.layers.attention.double_sparsity.token_label_write --m3b-fixture-hardware-run` (renamed from `page_signature_write`).
- **Validator (DEC-2 gate to flip):** `python/sglang/srt/layers/attention/double_sparsity/validator.py` + `_double_sparsity_radix_fixture_passed` server-args attribute.
- **Design intent + §13 rotation note:** `development/past_implementations/study/06-proposed-architecture-v2.md`.
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
