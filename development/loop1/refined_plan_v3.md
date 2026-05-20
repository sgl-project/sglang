# Plan: Deliver Standalone Double Sparsity for DeepSeek-V3.2 (FP8)

## Goal Description

### Design at a glance

Three paragraphs of data flow, before anything else. Skip this only if you already know.

- **What DS does.** On DeepSeek-V3.2 (FP8), replace the per-step token-selection role of NSA's `Indexer` (`python/sglang/srt/layers/attention/nsa/nsa_indexer.py`) with an offline-calibrated channel-importance projection plus a runtime page-signature top-K. Everything else stays: paged FlashMLA, FP8 KV cache, NSA's quant / dequant kernels, the model code outside the indexer call site.
- **Two artifacts.** Offline: a **channel mask file** (`safetensors`, ships once per model revision; contains channel selection + weights + content hash). Online: a **page signature table** (GPU tensor, allocator-owned metadata next to the KV page table, populated by a scale-aware FP8 dequant + channel projection on every page assignment, invalidated on free / eviction / abort).
- **Single edit site.** `DeepseekV2AttentionMLA.forward_core` in `python/sglang/srt/models/deepseek_v2.py` (line 1704). One config-gated branch: when `self.use_double_sparsity` is true, call `self.double_sparsity_selector.retrieve_topk(...)`; otherwise call the existing `self.indexer.forward(...)`. No monkey-patching, no model fork, no new attention backend. The DS path is opt-in per layer object.

Deliver this in a shape that meets the immediate client SLO on DeepSeek-V3.2 (FP8), is forward-compatible with deferred client requests (GLM-5, 128K ISL, FP4 weights), keeps the door open for downstream work (Twilight top-p selection, "Extensions" engine knob, eventual PD-Disagg / HiSparse coexistence), and ships in a shape that is upstream-reviewable in `sgl-project/sglang`. The plan also resolves the open question "resume vs restart" by recommending a path and stating the explicit cost of the alternatives.

### Standalone, Not a HiSparse Algorithm

Double Sparsity must work **standalone**, without requiring `--enable-hisparse`, without requiring PD-disaggregation, and without depending on the HiSparse memory tiering / coordinator stack. Today's HiSparse in `sgl-project/sglang` is wired only to PD-disaggregation decode instances (verified in `python/sglang/srt/managers/scheduler.py` — `_build_hisparse_decode_batch`, `set_decode_producer_stream`, and the `disaggregation/decode.py` admission path), so building DS on top of it would inherit that PD-only constraint and contradict the client requirement that users must be able to run DS on a single-instance server. HiSparse is treated as **inspiration only**: the plan permits factoring shared helpers into a neutral location and using them from both packages (see Implementation Notes), but it does not register DS as a HiSparse algorithm or invoke `SparseCoordinator`. HiSparse + DS coexistence is a real decision (DEC-8), not a future-work bullet; HiCache integration follows HiSparse decoupling per the draft's downstream ordering.

### Resume-vs-Restart Recommendation

**Restart from a fresh branch off current `main`. Implement Double Sparsity as a standalone feature under a new module (`python/sglang/srt/layers/attention/double_sparsity/`) that adds a config-gated branch to `DeepseekV2AttentionMLA.forward_core`, gated by `--enable-double-sparsity` plus a `--double-sparsity-config` JSON. Reimplement the selection / page-signature / Triton kernel work for the MLA + FP8 path, using PR #25304 as reading material — not cherry-pick targets. Close PR #22992. Mark PR #25304 reference-only.**

Rationale:
- PR #22992 (`dev/double-sparsity-reintro`, +1873 / 12 files): restores the legacy Llama-only, page=1, Triton-attention DS backend. The PR body documents a 3–12 % throughput regression on H100 and states "Performance optimization is planned for follow-up work." No MLA, no FP8, no page=64, predates DSv3.2 NSA. **Not a viable base.**
- PR #25304 (`dev/double-sparsity-v2`, +22552 / 90 files): valuable selection / calibration scaffolding (M1 skeleton; M2 K_label storage + write kernels; M3 selection pipeline; M4 FA3 adaptor; v1.1 stage-1 / stage-2 Triton block-topk + score-aware union + CUDA-graph capture; v2 native sparse-decode kernels) reaching a "BOTH GATES PASS" point at conc=32 / 128K / tb=8192. **But** FA3 / Llama backbone (not MLA), custom coordinator, 90 files including `HANDOFF_NATIVE.md` / `SESSION_REPORT_*.md` / pensieve installs / ad-hoc bench harnesses, no PR description, no CI. **Reading material**, not cherry-pick targets — the kernels need to be rewritten for the MLA `K_label` layout and FP8 quantization.
- Restart on `main` lets the work land as a focused, standalone module that can be reviewed independently and is not blocked on the HiSparse + PD-disagg coupling.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Double Sparsity is exposed via a new top-level pair of server args: `--enable-double-sparsity` (boolean) plus `--double-sparsity-config` (JSON, parsed by an independent validator). DS does **not** require `--enable-hisparse`, does **not** require `disaggregation_mode != "null"`, and does **not** register itself as a HiSparse algorithm. The runtime guard against simultaneously enabling both DS and HiSparse is governed by DEC-8.
  - Positive Tests:
    - Starting an SGLang server with `--enable-double-sparsity --double-sparsity-config '{"top_k":2048,"page_size":64,"channel_mask_path":"<...>"}'` on DeepSeek-V3.2 (FP8) on a single-instance server (no `--disaggregation-mode`) initialises the DS module and routes V3.2 attention through `DoubleSparsitySelector.retrieve_topk` instead of `Indexer.forward`.
    - Running with `--enable-double-sparsity` while `--enable-hisparse` is **not** set succeeds; the two features are independent on the standalone path.
  - Negative Tests:
    - `--enable-double-sparsity` together with `--enable-hisparse` fails at startup with the message "Double Sparsity and HiSparse are mutually exclusive; there are no plans to integrate them" (per DEC-8 resolution). DS and HiSparse may both ship in the codebase but cannot be enabled together at runtime.
    - `--enable-double-sparsity` without `--double-sparsity-config` fails at startup with a clear "channel_mask_path required" message.

- AC-2: The DS selector hooks into the DeepSeek-V3.2 NSA attention path at one named edit site and produces a page-table contract that FlashMLA accepts without translation tricks. NSA's quant / dequant / cache plumbing (`python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `dequant_k_cache.py`, the Triton + tilelang kernels, MTP precompute / verification) remains authoritative and untouched. The hook lives in the V3.2 attention path itself; it does not modify `mem_cache/sparsity/` or `managers/hisparse_coordinator.py`.
  - Hook site (one sentence, per CMT-4): `DeepseekV2AttentionMLA.forward_core` in `python/sglang/srt/models/deepseek_v2.py` (line 1704) gains a single config-gated branch — `if self.use_double_sparsity: selected = self.double_sparsity_selector.retrieve_topk(...) else: selected = self.indexer.forward(...)` — and `DeepseekV2AttentionMLA.__init__` constructs `self.double_sparsity_selector` when DS is enabled.
  - Selector contract (per CMT-5): `retrieve_topk` returns `selected_indices: int32 [bs, max_top_k]` of **logical page IDs sorted ascending in sequence order, with `-1` padding to `max_top_k`**, plus `valid_lengths: int32 [bs]`. An adapter step inside `forward_core` maps logical → physical via the existing `req_to_token` lookup and emits the FlashMLA `block_table` slot in the order the kernel expects (sequence order, not score order). Score order is consumed internally to choose the top-K set; the order in the returned `selected_indices` is sequence ascending.
  - Positive Tests:
    - Unit test calls `DoubleSparsitySelector.retrieve_topk(queries, layer_id, req_pool_indices, sparse_mask)` and receives `(selected_indices, valid_lengths)` with the documented shapes and `selected_indices` is verified monotonically ascending per row (modulo `-1` padding).
    - Integration test asserts that the FlashMLA `block_table` reaching the kernel during a DS-enabled decode step contains only the page IDs returned by `retrieve_topk` (via a metadata-assertion fixture; no production code mutation).
    - **Hot-page test** (per CMT-14): in a deterministic mid-page decode step where the current page is partially filled, the test verifies the active (in-fill) page is present in `selected_indices` regardless of its score — i.e. the local-window override fires.
    - **FP8 tolerance test** (replaces "bit-for-bit" per CMT-4): single-decode-step attention output against a Python-FP32 reference restricted to the same selected pages satisfies `max_abs_diff(out_fp8, out_ref) <= 5e-3` and `cosine_similarity(out_fp8, out_ref) >= 0.9999` averaged over 16 deterministic prompts. The tolerance is a documented invariant, not a wish.
  - Negative Tests:
    - Calling the selector on a request with `sparse_mask=False` returns the documented "unselected (dense)" sentinel and the integration path takes the native NSA branch.
    - Submitting a request before the page signature table is populated (forced via fault-injection switch) fails the request with a documented error rather than producing garbage output.
    - The diff against `main` adds zero files under `python/sglang/srt/mem_cache/sparsity/algorithms/` and zero registrations in `python/sglang/srt/mem_cache/sparsity/factory.py::_ALGORITHM_REGISTRY` (verified by `git diff --name-only`).

- AC-3: Page size 64 works on DeepSeek-V3.2 (FP8) and at least one alternate page size is exercised in tests. The implementation does not hard-code page=64. Backend / dtype pairing follows the V3.2 MLA backend rule: `fp8_e4m3` ↔ FlashMLA FP8 path, `bfloat16` ↔ FlashMLA BF16 path.
  - Positive Tests:
    - End-to-end smoke run with `kv_cache_dtype=fp8_e4m3`, `page_size=64`, V3.2 FP8 MLA backend succeeds for one warm prefill + one decode batch.
    - End-to-end smoke run with `kv_cache_dtype=bfloat16`, `page_size=64`, V3.2 BF16 MLA backend succeeds for one warm prefill + one decode batch.
    - Unit tests for the selection kernel run with `page_size in {32, 64}` and produce identical logical top-K rankings on a deterministic fixture (modulo page granularity).
  - Negative Tests:
    - Configuring `kv_cache_dtype=fp8_e4m3` with a BF16-only MLA backend (mismatched pair) fails at startup with an explicit backend / dtype mismatch error emitted by the new DS validator.
    - Configuring an unsupported page size (e.g. `page_size=7`) for DS fails at startup with an explicit page-size validation error, not silently.

- AC-4: A persisted **channel mask file** is required to enable Double Sparsity. The file is validated against the loaded model and configuration before serving begins, and an end-to-end sanity probe runs once at startup. Missing, structurally invalid, or semantically inconsistent files cause fail-fast at startup; there is no silent dense fallback in production. Per CMT-6, the validator focuses on **shape correctness, content identity, and behavioural sanity** — not on bookkeeping fields that don't catch the actual bug class.
  - Validator fields kept (shape / dtype safety):
    - `dtype` (must match `kv_cache_dtype`)
    - `head_dim` (sanity check against the model)
    - `page_size` (must match the running server's page size)
    - `label_dim` (must match the selector buffer allocation)
  - Validator fields **added** (per Codex C-NEW-5):
    - `content_sha256` — SHA-256 over the `channel_selection` and `channel_weights` tensors concatenated, recomputed at load and compared to the metadata.
    - `startup_sanity_probe` — at startup, run a single deterministic prompt through DS and verify NIAH-min (one needle, 512-token haystack) retrieves the needle. Failure aborts the server with a "channel mask sanity check failed" error and the probe's score.
  - Validator fields **dropped** (didn't catch real bugs):
    - `model_revision_sha` — passes for LoRA fine-tunes that miscalibrate; superseded by the sanity probe.
    - `tp_world_size` — derivable from the running config; carrying it in the artifact invites the rank-divergence bug DEC-9 has to settle anyway.
  - Positive Tests:
    - Loading a valid file (matching `dtype`, `head_dim`, `page_size`, `label_dim` and whose `content_sha256` matches the payload) succeeds and the sanity probe passes.
    - The loader emits a startup log line listing `content_sha256[:12]`, `dtype`, `head_dim`, `page_size`, `label_dim`, `created_at`, plus the sanity-probe score.
  - Negative Tests:
    - Missing file or unreadable path fails the server before the engine starts, with a non-zero exit code and a "channel mask file required" message.
    - File whose `content_sha256` does not match its payload fails the server with a "content hash mismatch" error.
    - File whose `dtype`, `head_dim`, `page_size`, or `label_dim` mismatches the running configuration fails the server with a message naming each mismatched field.
    - Sanity-probe failure (e.g. corrupted channel weights that pass shape checks but produce noise) aborts the server with the probe score in the error.

- AC-5: A calibration script produces a channel mask file for DeepSeek-V3.2 (FP8) and is invocable as a standalone command. CI uses a tiny NSA-shaped fixture; the production recipe runs on the agreed hardware (DEC-1) and is documented but not committed.
  - Positive Tests:
    - `python -m sglang.srt.layers.attention.double_sparsity.calibrate --model <tiny-NSA-fixture> --dtype fp8_e4m3 --tp 1 --output /tmp/channel_mask.safetensors` runs to completion in CI under a minute and writes a non-empty file with the documented schema (including `content_sha256`).
    - The CI artifact loads successfully via AC-4's loader against the same fixture and passes the sanity probe.
    - The production recipe is documented in `docs/advanced_features/double_sparsity_calibration.md` with the exact CLI invocation, expected dataset, agreed hardware, and expected wall-clock.
  - Negative Tests:
    - Running calibration with `--model` pointing to an unsupported architecture fails with a clear "DoubleSparsity calibration is only supported for ..." message.
    - Running calibration with `--tp 8` on a 1-GPU box fails before allocation with a config error, not at first NCCL call.

- AC-6: CUDA-graph piecewise capture / replay works for the DS decode path at the target concurrencies (16, 32, 64). Per CMT-8 (Codex C-NEW-3), static output buffers alone are not sufficient: grid sizes, scratch buffers, page-table writes, valid page counts, dense-sentinel branching, and radix-cache hit shapes must all be replay-stable.
  - Positive Tests:
    - Decode-path piecewise CUDA graph captures at conc=64 without `CUDA error: launch failed` and replays for at least 100 steps on a fixed batch.
    - The selector's output buffers are allocated with static shape `[bs, max_top_k]` for `selected_indices` (padded with `-1`) and `[bs]` for `valid_lengths`; the same buffers are reused across capture and replay.
    - The selection-kernel scratch buffer (stage-1 partial top-K, stage-2 merge scratch) is preallocated to a worst-case size before capture; no CUDA allocation occurs inside the captured region.
    - All branching inside the captured region is device-side (`tl.where`, mask multiplies, or kernel-internal predication). The captured Python region contains zero `if` statements that read CUDA tensor values to the host.
  - Negative Tests:
    - Setting `max_top_k` smaller than `top_k` fails at startup, not at capture.
    - Removing CUDA-graph capture (per-step eager) does not regress correctness — golden output is unchanged.
    - A regression test that intentionally calls `torch.empty(...)` inside the captured region demonstrates that capture fails (proves the rule is enforced, not just declared).

- AC-7: A **native_nsa** baseline (DeepSeek-V3.2 running with its built-in NSA selection on a single-instance server) is recorded on the same hardware (per DEC-1: 2-node H200 cluster; 8-way TP on `h200-10-220-51-6` or `-51-8`, optionally 16-way cross-node, or 2× 8-way replicas behind SMG), same model revision, same workload, same radix-cache setting, and same concurrency as the **double_sparsity** run. **No HiSparse baseline is required at any point** (DEC-6 + DEC-8 resolutions).
  - Positive Tests:
    - `development/benchmark.sh` plus a sibling `development/benchmark_baseline.sh` produce a side-by-side report with two columns: `native_nsa`, `double_sparsity`. Per-column rows: per-request output tok/s P50 / P99, TTFT P50 / P99, TPOT P50 / P99, and goodput-under-SLO.
  - Negative Tests:
    - The report fails to publish if any of {GPU id, TP size, page size, radix-cache setting, concurrency} differs between baseline and DS rows.

- AC-8: The DS run meets or beats the immediate SLO under the clarified throughput definition (DEC-1, RESOLVED): **per-request output throughput P50 ≥ 30 tok/s** (equivalently `1000 / TPOT_ms` ≥ 30, or `output_tok_per_sec / concurrency` ≥ 30) and **P99 TTFT ≤ 22 s, including scheduler-queue wait**, at `max-concurrency=64` (also at `min-concurrency=16`) on the workload defined in `development/benchmark.sh` (ISL ≈ 4096, OSL = 512, ~55 % prefix-cache hit) on **2-node H200** hardware (8-way TP default; 16-way cross-node TP also acceptable but expected slower per DEC-9's per-step all-reduce cost).
  - Positive Tests:
    - `bench_serving` over `gsp_isl4096_osl512_c64.jsonl` reports per-request output tok/s P50 ≥ 30 and P99 TTFT ≤ 22 s.
    - The same benchmark at conc=16 reports per-request output tok/s P50 ≥ 30 and P99 TTFT ≤ 22 s.
    - `sglang_double_sparsity_dense_fallback_total` is zero across the run.
  - Negative Tests:
    - The "no-op detector" gate fires if any of: `selected_pages == total_pages` (no selection happened), `dense_fallback_total != 0`, or the FlashMLA metadata-assertion fixture from AC-2 fails to confirm a restricted page table reached the kernel.

- AC-9: Quality gates pass against the native_nsa baseline. Agreed thresholds (DEC-3): NIAH retrieval @ 4K / 16K / 64K within 5 percentage points of native_nsa; MMLU within 1.0 percentage point of native_nsa.
  - Positive Tests:
    - `test/manual/test_double_sparsity_v32.py` runs NIAH at 4K / 16K / 64K and reports DS scores within 5 pp of native_nsa on each length.
    - MMLU 5-shot on DeepSeek-V3.2 (FP8) is within 1.0 pp of native_nsa.
  - Negative Tests:
    - A run with a deliberately corrupted channel mask file (random-permuted channel selection) makes NIAH @ 64K drop more than 20 pp below native_nsa, confirming the test is sensitive to file content.
    - A run with an empty / zero page signature table (fault-injected) makes NIAH @ 16K drop more than 30 pp, confirming the test is sensitive to runtime signatures.

- AC-10: Observability surfaces are exposed per step and aggregated per request. Healthy DS runs report `dense_fallback_total == 0` and `channel_mask_valid == 1` on every TP rank. Per-request `meta_info` carries `sparsity_rate` (selected / total page count), `selected_pages` (count), `dense_fallback` (0/1). Prometheus exposes `sglang_double_sparsity_*` gauges and counters (the namespace is `sglang_double_sparsity_*`, not `sglang_hisparse_double_sparsity_*`, because DS is standalone).
  - Positive Tests:
    - A scrape of `/metrics` after a healthy 64-concurrency run shows `sglang_double_sparsity_channel_mask_valid = 1`, `..._dense_fallback_total = 0`, non-zero `..._selected_pages_sum` and `..._selected_pages_count`.
    - Per-request `meta_info` carries `sparsity_rate`, `selected_pages`, `dense_fallback` and they aggregate to the Prometheus values modulo sampling.
  - Negative Tests:
    - A fault-injected run with an unwritable page signature table (forced by test flag) increments `..._dense_fallback_total` and the test asserts this — confirming the metric is wired, without requiring a fallback in production.
    - With `--disable-metrics` the server still serves and produces correct outputs; the per-request `meta_info` fields are unaffected.

- AC-11: The selection ABI is stable for the initial deliverable: `DoubleSparsitySelector.retrieve_topk(queries, layer_id, req_pool_indices, sparse_mask) -> (selected_indices, valid_lengths)`. Per CMT-10, **no `selection_mode` parameter is added in the initial scope**; top-p / Twilight enablement is its own follow-on with its own ABI design. The `DoubleSparsityConfig` JSON does not carry `selection_mode`, `top_p`, `min_top_k`, or `max_top_k` beyond what the top-K kernel needs (`top_k`, `device_buffer_size`).
  - Positive Tests:
    - The selector signature is documented in `selector.py` exactly as written above; a unit test inspects the public API surface.
    - `DoubleSparsityConfig` exposes only `top_k`, `page_size`, `channel_mask_path`, `device_buffer_size`, and optional algorithm-specific `extra` (a free dict). Other fields fail JSON parsing.
  - Negative Tests:
    - A `--double-sparsity-config '{"selection_mode":"TOPP"}'` fails parsing with a documented "unknown field" error. The error message links to the Twilight follow-on tracking issue (once filed).
    - A `--double-sparsity-config '{"top_p":0.9}'` fails parsing with the same error class.

- AC-12: The shipping branch is upstream-shaped. No `HANDOFF*.md`, `SESSION_REPORT*.md`, pensieve installs, ad-hoc bench harnesses, or workspace notes are committed.
  - Positive Tests:
    - `git log --name-only origin/main..HEAD` shows only files under: `python/sglang/srt/{layers/attention/double_sparsity, server_args.py, model_executor, models/deepseek_v2.py, metrics, managers}`, `sgl-kernel/`, `test/`, `docs/`, `development/benchmark*.sh`, plus optionally `python/sglang/srt/utils/sparse_helpers.py` if a shared helper is hoisted per Implementation Notes. Notably absent: `python/sglang/srt/mem_cache/sparsity/algorithms/double_sparsity/` (DS is not a HiSparse algorithm) and `python/sglang/srt/arg_groups/hisparse_hook.py` (DS does not extend HiSparse validation).
    - `git diff --stat origin/main..HEAD` is bounded by an agreed budget (DEC-4) and individual commits are < ~1500 lines each except for the Triton kernel commit (allowed to be a single logical unit).
  - Negative Tests:
    - A pre-commit hook (added in `task20`) blocks any session-artifact filename pattern (`HANDOFF*.md`, `SESSION_REPORT*.md`, `*.HANDOFF.md`, top-level pensieve dirs added to git).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The implementation delivers a standalone `python/sglang/srt/layers/attention/double_sparsity/` module with its own selector hooked into the DeepSeek-V3.2 NSA attention path at `DeepseekV2AttentionMLA.forward_core`, an FP8-aware channel-mask loader with content-hash + startup sanity probe, a page-signature table that is allocator-owned (lifecycle: write on page assign, invalidate on free / evict / retract) with a documented worst-case HBM budget, a page_signature_write kernel that consumes the same per-tile FP8 scales as the existing NSA quant / dequant path, a calibration script that defaults to NIAH-shaped synthetic data plus an opt-in dataset hook, Triton selection kernels (stage-1 block-topk + stage-2 merge, score-aware union, capture-safe — reimplemented for MLA using PR #25304 as reading material), TP-rank-synchronized selection per DEC-9, CUDA-graph piecewise capture with replay-stable grid / scratch / branching, per-request and Prometheus observability under `sglang_double_sparsity_*`, an MLA + FP8 quality regression suite (NIAH at 4K / 16K / 64K plus MMLU), and a benchmark harness that publishes side-by-side `native_nsa` / `double_sparsity` results on a single-instance server.

### Lower Bound (Minimum Acceptable Scope)
The implementation adds `--enable-double-sparsity` and `--double-sparsity-config` server args with an independent validator, lands the DS module hooked into `DeepseekV2AttentionMLA.forward_core` per AC-2's hook-site spec, ships a calibration script that produces a valid channel mask file for one fixed NSA-shaped fixture (CI) and documents the production recipe, validates the file at load time (shape + content hash + sanity probe), populates the page signature table on prefill with FP8-scale-aware projection and invalidates on free, meets AC-8 SLO at `max-concurrency=64` and at `min-concurrency=16` on a single-instance server, passes AC-9 quality gates with the agreed deltas, exposes the AC-10 metrics minimally (`selected_pages`, `dense_fallback_total`, `channel_mask_valid`), and produces an upstream-shippable branch (AC-12). Top-p / Twilight runtime behavior, GLM-5, 128K ISL, FP4 weights, and any HiSparse / PD-Disagg integration are deferred. The channel-mask schema (AC-4 fields) is shaped to admit GLM-5 / 128K / FP4 without rewrite per task 6's gap analysis.

### Allowed Choices
- Can use: a new top-level CLI surface (`--enable-double-sparsity`, `--double-sparsity-config`), Triton selection kernels (reimplemented for MLA using PR #25304 as reading material), CUDA-graph piecewise capture with preallocated scratch, `safetensors` for the channel mask file format with `content_sha256`, the existing FlashMLA backends (`flashmla_kv` for FP8, `flashmla_sparse` for BF16) as the underlying dense kernel, and shared utilities factored into a neutral helper module (per Implementation Notes) when patterns overlap with HiSparse code.
- Cannot use: registering DS in `python/sglang/srt/mem_cache/sparsity/factory.py::_ALGORITHM_REGISTRY`; calling `SparseCoordinator`; calling `HiSparseCoordinator`; extending `python/sglang/srt/arg_groups/hisparse_hook.py`; gating DS behind `--enable-hisparse`; requiring `disaggregation_mode != "null"`; the legacy `double_sparsity_backend.py` re-imported from PR #22992; `HANDOFF*.md` / `SESSION_REPORT*.md` files committed to the upstream branch; FA3-only kernels on the V3.2 path; a `selection_mode` parameter or any top-p plumbing in the initial scope.

> **Note on Deterministic Designs**: The user redirect ("Absolutely not, ...") narrows the architectural choice: DS is standalone. The remaining flexibility is in kernel implementation strategy. Per CMT-17, "cherry-pick from PR #25304" is honestly called what it is: reimplementing for MLA + FP8 with PR #25304 open as reading material. Budget accordingly.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

1. Add `python/sglang/srt/layers/attention/double_sparsity/` as a **package**, peer to `nsa/` and `dsv4/`:
   - `__init__.py` → re-exports `DoubleSparsitySelector`, `DoubleSparsityConfig`, `validate_double_sparsity`.
   - `selector.py` → `class DoubleSparsitySelector` exposing `retrieve_topk(queries, layer_id, req_pool_indices, sparse_mask) -> (selected_indices, valid_lengths)`. Returns logical page IDs in **sequence-order ascending** with `-1` padding (per AC-2 contract). The class does NOT inherit from `BaseSparseAlgorithm` and is NOT registered in `_ALGORITHM_REGISTRY`.
   - `config.py` → `@dataclass DoubleSparsityConfig` with fields `top_k`, `page_size`, `channel_mask_path`, `device_buffer_size`, `extra` (free dict). No `selection_mode`, no `top_p`, no `min_top_k`, no `max_top_k` (per CMT-10).
   - `channel_mask.py` → channel mask file `safetensors` loader, validator, schema definition, `content_sha256` recomputation, and the startup sanity probe.
   - `calibrate.py` → standalone CLI entry point for offline calibration (`python -m sglang.srt.layers.attention.double_sparsity.calibrate`).
   - `page_signature_table.py` → allocator-owned table allocator + lifecycle (write on assign, invalidate on free / evict / retract) + FP8-scale-aware page_signature_write Triton kernel wrapper.
   - `selection_kernel.py` → wrapper around the (reimplemented) stage-1 / stage-2 Triton kernels. Single-mode top-K only.
   - `validator.py` → `validate_double_sparsity(server_args)` parallel to `validate_hisparse(server_args)` but independent. Initially V3.2-specific per DEC-10; the capability check can generalize later.

2. **Hook site (concrete, per CMT-4).** File `python/sglang/srt/models/deepseek_v2.py`, class `DeepseekV2AttentionMLA` (line 1309), method `forward_core` (line 1704). The current code path inside `forward_core` calls `self.indexer.forward(...)` (where `self.indexer: Indexer` from `nsa_indexer.py:178`) to pick a token subset, then FlashMLA on the subset. The DS edit adds:
   - In `__init__`: assign `self.double_sparsity_selector = DoubleSparsitySelector(...) if model_config.use_double_sparsity else None`; assign `self.use_double_sparsity = bool(self.double_sparsity_selector)`.
   - In `forward_core`: one branch — `if self.use_double_sparsity: selected = self.double_sparsity_selector.retrieve_topk(...)` else: `selected = self.indexer.forward(...)`. The downstream FlashMLA call is unchanged; the adapter step that maps `selected_indices` (sequence-order logical page IDs) to FlashMLA `block_table` (sequence-order physical page IDs) reuses the existing `req_to_token` logic that NSA already uses.
   - No monkey-patching, no model class fork, no new attention backend.

3. New server args in `python/sglang/srt/server_args.py`:
   ```python
   enable_double_sparsity: bool = False
   double_sparsity_config: Optional[str] = None  # JSON
   ```
   Plus argparse entries with `--enable-double-sparsity` and `--double-sparsity-config`. The mutual-exclusion behavior with `--enable-hisparse` is governed by DEC-8 and applied in `_handle_double_sparsity` (a new private method paralleling `_handle_hisparse` / `_handle_pd_disaggregation`).

4. **Channel mask file format** (`safetensors`):
   - Tensors: `channel_selection[L, H, label_dim]` (int32 indices) and `channel_weights[L, H, label_dim]` (fp32) where `L = num_layers`, `H = num_heads`.
   - Metadata: `dtype`, `head_dim`, `page_size`, `label_dim`, `created_at`, `schema_version`, `content_sha256` (per CMT-6).
   - Dropped from earlier draft: `model_revision_sha`, `tp_world_size` (per CMT-6 — replaced by content hash and runtime sanity probe).
   - Forward-compatible against GLM-5 (per-rank metadata), 128K ISL (no length-dependent fields), FP4 weights (dtype-agnostic) — see task 6's memo.

5. **Page signature table** — lifecycle, memory, hot page, FP8 path:
   - **Allocation and ownership** (per CMT-11). The table is allocator-owned metadata sitting next to the KV page table. Shape per TP rank: `[num_layers_local, max_pages, num_heads_local, label_dim]` (head- and layer-shared across ranks as appropriate). The KV page allocator gains a "page assigned" / "page freed" / "page evicted" / "page reused" hook; the page signature table writes / invalidates entries in lockstep. A `valid_mask[L, max_pages]` bool tensor backs the invalidation; selection ignores invalid pages.
   - **Memory budget** (per CMT-12). Worst case for V3.2 with 60 layers × 128 heads × `label_dim=16` × fp16 × 15,625 max pages (1 M context / page=64) is ≈ 3.8 GB per rank if not sharded. With TP=8 and head-sharded `H_local=16`, it drops to ≈ 480 MB / rank. With `label_dim=16` and fp16, this is the operating point. The plan picks: `dtype=fp16`, `label_dim=16`, TP-head-sharded allocation, owned by the KV-page allocator's lifetime. If `label_dim` needs to grow, the plan revisits this number; for now the budget is documented.
   - **Hot page** (per CMT-14). During decode the active KV page is being filled token-by-token; if the signature is only refreshed at page-boundary, the freshest up-to-63 tokens are invisible to selection. Mitigation: (a) update the active page's signature every decode step (cheap — one page-worth of K vectors), and (b) force the active page (and a configurable local window of N most-recent pages, default 1) into the selected set unconditionally regardless of score. The local-window override is an explicit AC-2 positive test.
   - **FP8 scale-aware projection** (per CMT-16). The K cache stores per-tile FP8 nope with inline scales: byte layout `[nope_fp8(512) | scales(16)]` per token per page (from `nsa/quant_k_cache.py`'s `quantize_k_cache_separate`). The page_signature_write kernel must either (a) read the inline scales and dequantize the nope FP8 to BF16 before projecting through `channel_selection`/`channel_weights`, or (b) integrate the per-tile scale into the projection arithmetic. Option (a) is the default — simpler, easier to validate against an FP32 reference, and reuses the dequant logic already in `nsa/dequant_k_cache.py`. The plan picks (a). The rope BF16 half does not require dequant.
   - **TP rank synchronization** (per CMT-7 / DEC-9 resolution). Page signatures stay TP/head-sharded; they are never all-gathered across ranks. Each rank computes a **scalar page score** per page from its local head shard. Those scalar scores are then **all-reduced (SUM) across the attention TP group** — a `[max_pages]`-shaped tensor, bandwidth-small but latency-sensitive (one collective per layer per decode step). Every rank runs deterministic top-K independently from the same reduced scores, producing bit-equal `selected_indices` across ranks by construction. A rank-agreement test (task16) constructs a setup where per-rank local top-K would diverge and asserts that the all-reduced result agrees. Validation order: 8-way intra-node H200 TP / 2× replicas behind SMG first (NVLink-only); cross-node 16-way TP second. If the cross-node per-step all-reduce dominates latency, the plan may pivot back to per-rank selection with a documented perf-vs-quality benchmark (a future amendment, not the default).

6. **Radix cache.** DS does not inherit HiSparse's `assert server_args.disable_radix_cache`. The DS validator gates radix cache on the M3-B page-stability fixture: a deterministic prefix is run cold (cache miss) and warm (cache hit) and the test asserts identical `retrieve_topk` output across both runs. Passing the fixture grants permission; failing it makes the DS validator require `--disable-radix-cache` until resolved. The 55 % prefix-cache-hit workload from `development/benchmark.sh` is preserved.

7. **Observability hooks.** Add Prometheus gauges / counters under `sglang_double_sparsity_*`; thread per-request fields `sparsity_rate`, `selected_pages`, `dense_fallback` through `ScheduleBatch` → `meta_info`.

8. **Tests.** `test/manual/test_double_sparsity_v32.py` (NIAH 4K / 16K / 64K + MMLU 5-shot on V3.2-FP8); `test/srt/test_double_sparsity_unit.py` (selector kernel, channel-mask loader with content hash + sanity probe, page signature lifecycle + invalidation, FP8 scale-aware projection equivalence to BF16 reference, ABI shape including AC-11 `selection_mode` absence, fault-injection, CUDA-graph capture rule enforcement). CI smoke under `test/run_suite.py` using the tiny NSA fixture.

### Future-Work Notes (out of initial scope; for the PR description / docs)

Per DEC-6 + DEC-8 resolutions, **HiSparse integration is out of all scope — initial and downstream alike. There is no HiSparse adapter on the roadmap.** HiSparse is in flux; if a redesign to integrate ever becomes justified, it will be a separate plan after the current changes settle.

- **GLM-5 (specifically GLM-5.1)** — deferred-but-hard requirement (DEC-6 split). The selector ABI, channel-mask schema, and validator (capability-check per DEC-10) are designed so GLM-5.1 falls in for free once it exposes the same `nsa.Indexer` hook. Task 6's schema-compatibility memo treats GLM-5.1 as the load-bearing forward-compat target.
- **PD-Disagg integration** — kept in mind for downstream; no customer has asked for it on top of DS yet. The page signature table crosses the prefill / decode boundary; if and when this lands, either recompute on the decode worker (cheap if the channel mask is loaded there) or transport alongside KV.
- **HiCache integration** — kept in mind for downstream; same reasoning as PD-Disagg.
- **128K ISL** — deferred; schema admits it (no length-dependent fields in the channel mask file).
- **FP4 (nvfp4 / mxfp4) weights** — deferred; schema is dtype-agnostic.
- **Twilight / top-p selection** — deferred; the selector ABI ships single-mode top-K (AC-11) and adding top-p later is a separate plan with its own ABI design.

### Relevant References

- `python/sglang/srt/models/deepseek_v2.py` — `DeepseekV2AttentionMLA` (line 1309), `forward_core` (line 1704), `DeepseekV32ForCausalLM` (line 2573). The single edit site lives here.
- `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` — `Indexer` (line 178). DS replaces its selection role on the V3.2 path; nothing else.
- `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` — FP8 scale layout (nope: `[nope_fp8(512) | scales(16)]` per token, rope: BF16). The page_signature_write kernel consumes the inline scales.
- `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` — reference dequant path; reuse for option (a) of the FP8 projection.
- `python/sglang/srt/layers/attention/dsv4/` — DeepSeek-V4 DSA internals; out of scope for initial deliverable.
- `python/sglang/srt/layers/attention/flashmla_backend.py`, `flashmla_kv` — underlying dense MLA kernel.
- `python/sglang/srt/server_args.py` — where the new server args are added; `_handle_double_sparsity` parallels the existing `_handle_hisparse`.
- `python/sglang/srt/arg_groups/hisparse_hook.py` — **inspiration only**; do not import or extend.
- `python/sglang/srt/managers/hisparse_coordinator.py` — **inspiration only**; do not import or invoke.
- `python/sglang/srt/jit_kernel/hisparse.py` — Triton helpers; useful patterns that, if reused, must be hoisted to `python/sglang/srt/utils/sparse_helpers.py` (a new neutral helper) and imported from both — per CMT-19.
- `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py` — closest existing non-MLA analogue for representation-pool patterns. Read-only reference.
- `development/benchmark.sh` — workload definition; basis for `benchmark_baseline.sh`.
- PR #25304 commits (reading material only — the kernels there target FA3 / Llama and must be reimplemented for MLA + FP8):
  - `a8efc6068` M1 skeleton + calibration schema
  - `567eff67b` M2 K_label storage + write kernels → page_signature_write (rewrite for FP8 + MLA)
  - `e3570f2fb` M3 selection pipeline
  - `0b776ca05` v1.1-4 stage-1 block-topk Triton kernel
  - `1b5e52863` v1.1-5 stage-2 merge Triton kernel
  - `7fe8002a3` v1.1-6 score-aware union + CUDA-graph capture/replay
  - `30ba60dae` v2 pivot native sparse-decode (reference only)
  - `dc3dcf13f` pluggable selector backends (pattern reference)
  - `e8824f86a` M7 calibration script
  - `3dca4be73` NIAH synthetic prompt generator

## Dependencies and Sequence

### Milestones

1. **Milestone 0 — Decision artifact & branch setup**
   - Phase A: Land this refined plan. Close PR #22992. Mark PR #25304 reference-only.
   - Phase B: Cut a feature branch off current `main`: `dev/double-sparsity-standalone`.
   - Phase C: 8 of 10 `## Pending User Decisions` are RESOLVED (DEC-1, DEC-3, DEC-4, DEC-5, DEC-6, DEC-8, DEC-9, DEC-10). Two carry forward: DEC-2 (radix cache; revisit at Milestone-1-land checkpoint) and DEC-7 (Extensions interpretation; Claude default holds, low-stakes). The original CMT-18 execution gate on Milestone 5 reduces to DEC-2 only.

2. **Milestone 1 — Server args + validator + V3.2 attention-path seam** (targets AC-1, AC-2 backbone)
   - Phase A: Add `enable_double_sparsity` and `double_sparsity_config` to `server_args.py`; argparse entries; `_handle_double_sparsity` private method; mutual-exclusion behavior per DEC-8.
   - Phase B: Land `layers/attention/double_sparsity/validator.py::validate_double_sparsity(server_args)`. Per DEC-10 default, V3.2-specific (model check matches `Indexer` presence on the attention layer; dtype check; page-size check; `channel_mask_path` required).
   - Phase C: Add the one-line branch in `DeepseekV2AttentionMLA.forward_core` (line 1704) and the `self.double_sparsity_selector` field in `__init__`. The selector returns a deterministic placeholder until Milestone 3. Synthetic-fixture unit tests assert the FlashMLA `block_table` receives the selected sequence-order indices.
   - Phase D: Land `development/benchmark_baseline.sh` (native_nsa column on a single-instance server) so DEC-1 / DEC-2 / DEC-3 have data to anchor on.

3. **Milestone 2 — DS package skeleton + channel mask file + page signature table** (targets AC-1, AC-4)
   - Phase A: Land the `double_sparsity` package (`__init__.py`, `selector.py`, `config.py`) with a placeholder `retrieve_topk` that returns deterministic top-K based on simple key heuristics. **Guard**: a server-side check refuses to serve real traffic while the selector is a placeholder (hard error if a placeholder-built binary is asked to handle anything beyond unit / smoke tests).
   - Phase B: Channel mask file `safetensors` loader + validator (`channel_mask.py`) including `content_sha256` recomputation and the startup sanity probe. Schema reviewed by task 6 (gap analysis for GLM-5 / 128K / FP4) before merging.
   - Phase C: Page signature table allocator (`page_signature_table.py`); allocation hooked into the KV page allocator's assign / free / evict / retract callbacks. The table is allocated and lifecycle-managed; the population kernel comes in Milestone 3.
   - Phase D: Validator (`validator.py` from M1-B) enforces channel-mask requirement, page-size pairing, and the radix-cache decision from DEC-2.

4. **Milestone 3 — Selection kernels (real DS math), page-signature write, and ABI lock-in** (targets AC-2 real, AC-6, DEC-2 page-stability)
   - Phase A: Reimplement the stage-1 block-topk + stage-2 merge + score-aware union Triton kernels for MLA `K_label` layout. Use PR #25304's commits as reading material; budget the rewrite, do not call this a port. Output: `selected_indices` in sequence order, capture-safe.
   - Phase B: Land the page_signature_write Triton kernel that reads the inline FP8 scales from `quant_k_cache`'s `nope_part` (bytes 512–528) and projects through the channel mask after dequant. Wire to the page signature table from M2-C; populate on prefill, update the active page every decode step, force the active page into the selected set. **Land the DEC-2 page-stability fixture here**: deterministic prefix run cold (cache miss) and warm (cache hit), assert identical `retrieve_topk` output. The DS validator's radix-cache permission is gated on this fixture passing. **Land the TP rank-synchronization all-reduce per DEC-9 here**: per-rank partial scores → all-reduce → top-K.
   - Phase C: CUDA-graph piecewise capture for the DS decode path at conc 16 / 32 / 64. Scratch buffers preallocated, branching device-side, page-table writes use `index_put_` with static shape. Includes the AC-6 regression test that an in-region `torch.empty(...)` fails capture.

5. **Milestone 4 — Calibration tooling** (targets AC-5)
   - Phase A: Port the calibration script (commits `a8efc6068`, `e8824f86a`) to `calibrate.py`. Default dataset: NIAH-shaped synthetic (`3dca4be73`); `--dataset` accepts external corpora. Emits `content_sha256` into the file metadata.
   - Phase B: Produce + version an external channel mask file for `deepseek-ai/DeepSeek-V3.2` (FP8) **outside** the repo; commit only the documented recipe under `docs/advanced_features/double_sparsity_calibration.md`. **Blocked on DEC-1 (agreed hardware) and DEC-4 (artifact storage location).**

6. **Milestone 5 — Quality & SLO gates** (targets AC-7, AC-8, AC-9, AC-10). DEC-1 (hardware = 2-node H200; 8-way TP / 2× SMG replicas default, 16-way cross-node optional) and DEC-3 (NIAH ≤ 5 pp, MMLU ≤ 1.0 pp) are RESOLVED. The remaining CMT-18 dependency is DEC-2 (radix cache), whose resolution checkpoint is at Milestone-1 land — well before this milestone.
   - Phase A: Update `development/benchmark.sh` to consume the baseline harness from M1-D and emit the two-column report (`native_nsa` / `double_sparsity`).
   - Phase B: `test/manual/test_double_sparsity_v32.py` for NIAH (4K / 16K / 64K) and MMLU. CI smoke `test/srt/test_double_sparsity_unit.py`.
   - Phase C: Prometheus metrics + per-request `meta_info` fields.

7. **Milestone 6 — Ship-gate** (targets AC-12)
   - Phase A: Branch hygiene: rewrite history if needed, write a single PR description, add the pre-commit hook that blocks session-artifact filename patterns, run CI green, request review.
   - The Twilight / top-p ABI work that previously lived in this milestone is **dropped per CMT-10**; it is its own follow-on with its own design.

> Dependencies:
> - M0-C unblocks M1 (server args presume DEC-8 / DEC-9 / DEC-10 resolutions).
> - M1-A unblocks every other task (server args must exist before validator can read them).
> - M1-B unblocks M1-C (validator decides whether DS is engaged).
> - M1-C unblocks M2 (the attention-path seam must accept the selector ABI).
> - M1-D unblocks the DEC-1 / DEC-2 / DEC-3 conversations and AC-7.
> - M2-A unblocks M2-B, C, D (skeleton first).
> - M2-B's schema review (task 6) unblocks M2-D (no validator hard-coding without a frozen schema).
> - M3-B page-stability fixture unblocks the DS validator's radix-cache permission (DEC-2).
> - M3-B TP all-reduce wiring is required before the M5 benchmark runs are meaningful (rank divergence breaks reproducibility).
> - M4 unblocks AC-8 (no SLO test without a real channel mask file).
> - M5 unblocks M6.

## Task Breakdown

> **Note (per CMT-17):** Tasks form a near-linear critical path, expected for a single-implementer scope. The task IDs serve as anchors for reviewer comments and cross-references, not as a parallelizable work breakdown. Task 18 from the prior plan is dropped per CMT-10 (Twilight ABI deferred); task 20 dependencies are updated accordingly.

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|-------------------------|------------|
| task1 | Confirm the "restart + standalone + reimplement" recommendation with the user; close PR #22992; relabel PR #25304 as reference-only | (decision narrative) | analyze | - |
| task2 | Resolve `## Pending User Decisions` DEC-1..DEC-10 (SLO + hardware; radix cache; quality thresholds; calibration ownership; V3.2 semantic; deferred-req scope; "Extensions" interpretation; HiSparse + DS composition; TP rank sync; V3.2-scope vs generic) | AC-1, AC-7, AC-8, AC-9, AC-4 | analyze | task1 |
| task3 | Add `enable_double_sparsity` and `double_sparsity_config` server args (`python/sglang/srt/server_args.py`); argparse entries; `_handle_double_sparsity` method; mutual-exclusion behavior per DEC-8 | AC-1 | coding | task2 |
| task4 | Add the V3.2 attention-path branch in `DeepseekV2AttentionMLA.forward_core` (line 1704) and the `self.double_sparsity_selector` field in `__init__`; selector returns a deterministic placeholder; synthetic-fixture unit tests assert the FlashMLA `block_table` receives the selected sequence-order indices | AC-1, AC-2 | coding | task3 |
| task5 | Land `development/benchmark_baseline.sh` (native_nsa, single-instance, agreed hardware); refactor `benchmark.sh` to emit the two-column report skeleton (DS column empty until M3 lands) | AC-7 | coding | task4 |
| task6 | Gap analysis for GLM-5, 128K ISL, and FP4-weights compatibility of the **channel-mask file schema**; produce a one-page memo identifying any schema fields required to keep these deferred reqs cheap | AC-4 (schema) | analyze | task2 |
| task7 | Land `double_sparsity` package skeleton (`__init__.py`, `selector.py`, `config.py`) with placeholder `retrieve_topk` (sequence-order ascending output, `-1` padding); add the **server-side placeholder-guard** that refuses real traffic when the placeholder is built | AC-1, AC-2 | coding | task4 |
| task8 | Land channel mask file `safetensors` loader + validator (`channel_mask.py`) with schema from task 6 frozen in; include `content_sha256` recomputation and the startup NIAH-min sanity probe; add unit tests for happy-path load, content-hash mismatch, all shape-mismatch negative tests, and sanity-probe failure | AC-4 | coding | task6, task7 |
| task9 | Land page signature table allocator (`page_signature_table.py`); allocate when DS is enabled with `dtype=fp16, label_dim=16`, TP-head-sharded; wire allocator hooks (assign / free / evict / retract) with `valid_mask` invalidation; do not yet populate (kernel comes in task 12) | AC-2 | coding | task8 |
| task10 | Land `validator.py::validate_double_sparsity(server_args)`; enforce channel-mask path required, page-size pairing, V3.2-specific model check per DEC-10 default, backend / dtype pairing, and the DEC-2 radix-cache permission gated by the M3-B fixture | AC-1, AC-3, AC-4 | coding | task9 |
| task11 | Reimplement DS Triton selection kernels (stage-1 block-topk, stage-2 merge, score-aware union) for MLA `K_label` layout using PR #25304 commits as reading material; output `selected_indices` in **sequence-order ascending** with `-1` padding; capture-safe with preallocated scratch | AC-2, AC-6 | coding | task10 |
| task12 | Land the page_signature_write Triton kernel that consumes FP8 inline scales from `quant_k_cache`'s nope_part (bytes 512–528) and projects through channel mask after dequant to BF16; wire to page signature table from task 9; populate on prefill, update active page every decode step, force active page into selected set (hot-page rule); land the M3-B page-stability fixture (cold vs warm prefix) and the TP all-reduce wiring per DEC-9 | AC-2 | coding | task11 |
| task13 | Enable CUDA-graph piecewise capture / replay for the DS decode path at conc 16 / 32 / 64; preallocate scratch buffers; device-side branching only; regression test that in-region allocation fails capture | AC-6 | coding | task12 |
| task14 | Port calibration script from PR #25304 commits `a8efc6068`, `e8824f86a`, `3dca4be73` to `calibrate.py`; default to NIAH-shaped synthetic data; emit `content_sha256` into file metadata; document the production recipe in `docs/advanced_features/double_sparsity_calibration.md` (no file committed) | AC-5 | coding | task8 |
| task15 | Extend `development/benchmark.sh` (already split by task 5) to populate the DS column; add side-by-side `native_nsa` / `double_sparsity` rows; enforce match on {GPU id, TP size, page size, radix-cache setting, concurrency}. Cannot run until a real channel mask file exists (task 14) | AC-7, AC-8 | coding | task5, task13, task14 |
| task16 | Add `test/manual/test_double_sparsity_v32.py` (NIAH @ 4K / 16K / 64K + MMLU 5-shot) and `test/srt/test_double_sparsity_unit.py` (selector kernel, channel-mask loader, content-hash, sanity probe, page-signature lifecycle + invalidation, FP8 dequant equivalence to BF16 reference, ABI shape including `selection_mode` absence, fault injection, CUDA-graph capture rule enforcement) | AC-9, AC-3, AC-4, AC-11 | coding | task14 |
| task17 | Add Prometheus metrics under `sglang_double_sparsity_*` and per-request `meta_info` fields (`sparsity_rate`, `selected_pages`, `dense_fallback`); test with fault injection | AC-10 | coding | task13 |
| task19 | Independent reasonability audit of the reimplemented selection kernels vs the Double Sparsity paper and Twilight repo; verify channel-sparsity math matches the published algorithm; document deltas | AC-9 | analyze | task11 |
| task20 | Branch hygiene + ship-gate: rewrite history if needed, write the PR description (must explicitly call out "standalone, no HiSparse"), add the pre-commit hook that blocks `HANDOFF*.md` / `SESSION_REPORT*.md` / pensieve installs, run CI green, prepare reviewer guide. Include a "Future-Work" section that scopes the HiSparse / PD-disagg decoupling per DEC-8 | AC-12 | coding | task15, task16, task17 |

> Task 18 (Twilight / TOPP unit-test + feature flag) is removed from the initial plan per CMT-10. The ID is intentionally skipped to preserve the original cross-references; a future Twilight plan can reuse it or pick a new ID.

## Claude-Codex Deliberation

### Agreements (after Codex Round 1 + 2 + user CMT-1 redirect + Linus + Codex critique round)
- The restart recommendation stands: PR #22992 is not a viable base; PR #25304 is reading material, not cherry-pick targets.
- Hook site is named explicitly: `DeepseekV2AttentionMLA.forward_core` in `python/sglang/srt/models/deepseek_v2.py` (line 1704), with one config-gated branch.
- The selector contract returns logical page IDs in sequence-order ascending; FlashMLA receives a sequence-order block table, not score-order.
- The page signature table is allocator-owned metadata with explicit lifecycle (assign / free / evict / retract) and a documented HBM budget.
- FP8 path consumes the existing per-tile scales from `quant_k_cache`'s `nope_part`; dequant to BF16 before projection.
- TP rank synchronization happens via a per-layer per-step all-reduce of partial head scores (DEC-9 default).
- CUDA-graph capture safety requires preallocated scratch, device-side branching, and an enforcement test — not just static output buffers.
- The Twilight `selection_mode` ABI is deferred entirely from initial scope; AC-11 ships a single-mode top-K interface.
- HiSparse coexistence is a real decision (DEC-8), not a future-work bullet; the HiSparse adapter story is gated on a separate HiSparse / PD-disagg decoupling effort.
- Channel mask file naming and page signature table naming replace the prior "calibration artifact" / "runtime label cache" pair.
- Symbol-name boundary: shared helpers go to a neutral `python/sglang/srt/utils/sparse_helpers.py`, not "copy don't import".

### Resolved Disagreements (this round)
- **CMT-1 (plan bureaucracy)** — Added a "Design at a glance" three-paragraph summary at the top of Goal Description per Linus. Codex partially agreed; the data-flow summary front-loads the architecture without removing the detail reviewers need.
- **CMT-2 (label naming)** — Renamed "calibration artifact" → "channel mask file" and "runtime label cache" → "page signature table" throughout. Deleted the "Two Different Labels" disambiguation section. Codex agreed.
- **CMT-3 (HiSparse mutual exclusion)** — Reframed AC-1's negative test to bind to DEC-8 (composition is a real decision, not a "not yet integrated" punt). Codex partially agreed and asked the plan to commit to one direction; DEC-8 forces that.
- **CMT-4 (AC-2 hook site + FP8)** — Named the hook site (`DeepseekV2AttentionMLA.forward_core` line 1704) and replaced "bit-for-bit identical" with a documented tolerance (`max_abs_diff <= 5e-3`, `cosine_similarity >= 0.9999`).
- **CMT-5 (selected_indices contract)** — Selector returns sequence-order-ascending logical page IDs with `-1` padding; score order is internal only. AC-2 gains a sequence-order monotonicity test.
- **CMT-6 (AC-4 validator theater)** — Reworked validator: kept shape fields (`dtype`, `head_dim`, `page_size`, `label_dim`); added `content_sha256` and a startup NIAH-min sanity probe; dropped `model_revision_sha` and `tp_world_size` (one is bypassable by LoRA, the other is derivable and feeds the rank-divergence bug DEC-9 addresses).
- **CMT-7 (TP rank correctness)** — New DEC-9 with all-reduce as the Claude default. Documented the per-layer per-step cost.
- **CMT-8 (CUDA graph capture safety)** — Strengthened AC-6 with scratch / branching / allocation rules plus a regression test that intentionally allocates inside the captured region.
- **CMT-9 (DS off vs DS on naming)** — Renamed AC-7 columns to `native_nsa` and `double_sparsity`.
- **CMT-10 (selection_mode YAGNI)** — Dropped `selection_mode`, `top_p`, `min_top_k`, `max_top_k` from `DoubleSparsityConfig`. AC-11 ships a single-mode top-K ABI; task 18 is removed; Twilight enablement is its own follow-on plan.
- **CMT-11 (page signature lifecycle)** — Made the page signature table allocator-owned with explicit assign / free / evict / retract callbacks; added `valid_mask` invalidation.
- **CMT-12 (memory budget)** — Documented the worst-case bytes: `dtype=fp16, label_dim=16, TP=8 head-sharded` ≈ 480 MB / rank at 1 M context, page=64.
- **CMT-13 (HiSparse decouple investigation)** — Reframed Future-Work Notes: HiSparse + PD-disagg decoupling is a separate plan; the HiSparse adapter story for DS is gated on that decoupling, not promised. DEC-8 records the immediate composition decision.
- **CMT-14 (hot page during decode)** — Added active-page-every-step update + force-active-page-into-selected-set rule; AC-2 gains a hot-page test.
- **CMT-15 (model-class special-case)** — DEC-10 records the V3.2-specific-vs-generic decision; Claude default is V3.2-specific via capability presence, not name string match.
- **CMT-16 (FP8 quant scales)** — page_signature_write kernel consumes the per-tile scales from `quant_k_cache`'s `nope_part` bytes 512–528; default is dequant-to-BF16 before projection.
- **CMT-17 (cherry-pick fiction + linear theatre)** — Replaced "cherry-pick from PR #25304" with "reimplement using PR #25304 as reading material" everywhere; added a note above the task table acknowledging the linear single-implementer scope.
- **CMT-18 (pending DECs = no falsifiable success)** — Added an explicit execution gate: Milestone 5 cannot publish a passing SLO claim until DEC-1, DEC-2, DEC-3 are resolved. Recorded as a Dependencies note and in Milestone 5's preamble.
- **CMT-19 (copy don't import)** — Replaced the symbol-name boundary rule with "extract shared helpers to a neutral module and import from both". Documented under Implementation Notes.

### User DEC-resolution round (resolutions encoded into the DEC bodies above)
- **DEC-1 RESOLVED** — User confirmed hardware = 2-node H200 (`h200-10-220-51-6` and `h200-10-220-51-8`, 8 GPUs each). Default operating shape is 8-way TP / 2× replicas behind SMG; cross-node 16-way TP is acceptable but expected slower. Per-request TPS = `1000 / TPOT_ms` = `output_tok_per_sec / concurrency`. Cascaded into AC-7 / AC-8.
- **DEC-2 deferred** — User chose to revisit after Milestone 1 lands. Claude default (radix cache enabled by default, M3-B page-stability fixture as gate) remains in force. Does not block any pre-M3 work.
- **DEC-3 RESOLVED** — User accepted Claude defaults (NIAH ≤ 5 pp delta, MMLU ≤ 1.0 pp delta) with the caveat that thresholds can be loosened if they prove too restrictive at M5.
- **DEC-4 RESOLVED** — User confirmed: calibration script in repo, channel mask file NOT tracked by repo (deploying team owns artifact storage; CI uses tiny fixture).
- **DEC-5 RESOLVED** — User confirmed: leave NSA alone; DS is an alternative selector path. Do not double-filter an already sparse set.
- **DEC-6 RESOLVED with nuance** — HiSparse is OUT of all scope (initial AND downstream). HiCache and PD-Disagg remain "kept in mind" future-work. GLM-5.1 is deferred-but-hard; schema and validator (capability-check per DEC-10) must be designed to admit GLM-5.1 at land time. 128K ISL and FP4 weights are deferred without special handling.
- **DEC-8 RESOLVED** — Adopt the mutual-exclusion startup guard at runtime; the two packages may both ship in the codebase. Remove all future-work claims about integrating DS as a HiSparse algorithm. AC-1 negative test cites this directly.
- **DEC-9 RESOLVED with implementation contract** — Do NOT all-gather page signatures (they stay TP/head-sharded). Compute scalar page scores per rank, all-reduce the SCORES (not signatures) across the attention TP group, then run deterministic top-K independently per rank. Rank-agreement test lives in task16. Validate on 8-way intra-node TP first; cross-node 16-way second. Pivot to per-rank with perf-vs-quality benchmark is an allowed future amendment if the per-step all-reduce dominates latency.
- **DEC-10 RESOLVED** — Capability-check validator (presence of `nsa.Indexer` on the attention layer), not a model-name string match. The capability check generalizes to GLM-5.1 (DEC-6's deferred-but-hard requirement) without name-string special cases.

### Convergence Status
- Final Status: `partially_converged`. After the user-DEC-resolution round: 8 of 10 DECs are RESOLVED (DEC-1, DEC-3, DEC-4, DEC-5, DEC-6, DEC-8, DEC-9, DEC-10). Two remain PENDING: DEC-2 (radix cache; deferred by user to revisit after Milestone 1 lands; Claude default holds meanwhile) and DEC-7 (Extensions interpretation; low-stakes; Claude default holds). Per CMT-18, the original Milestone-5 execution gate was on DEC-1, DEC-2, DEC-3; DEC-1 and DEC-3 are now resolved, so the gate reduces to DEC-2 — which itself will be revisited at M1-land time, well before M5.

## Pending User Decisions
- DEC-1: SLO definition + hardware
  - Claude Position: 30 tok/s is per-request P50 output throughput. P99 TTFT < 22 s includes scheduler-queue wait, prefix-cache lookup, and prefill. Both `max-concurrency=64` and `min-concurrency=16` must hit the per-request SLO.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: Per-request 30 tok/s at conc=64 implies aggregate ≈ 1920 tok/s decode. Aggregate 30 tok/s would be trivial.
  - Decision Status: **RESOLVED**. Hardware is **2 nodes of H200**, specifically `h200-10-220-51-6` (8 GPUs) and `h200-10-220-51-8` (8 GPUs). Two deployment shapes are allowed: cross-node **16-way TP** or **2× replicas of 8-way TP behind SMG**. Client explicitly requested H200s. The 8-way TP / 2× replica shape is the default operating point for SLO claims (lower latency on the per-layer all-reduce that DEC-9 introduces); cross-node 16-way TP is supported but expected to be slower. "30 tok/s per request" is equivalent to `1000 / TPOT_ms` and to `output_tok_per_sec / concurrency` (consistency check). AC-7 and AC-8 cite this hardware exactly.

<comment>
I think claude's position is fine.
</comment>
- DEC-2: Radix cache reconciliation
  - Claude Position: DS keeps radix cache enabled by default. The DS validator gates radix cache on the M3-B page-stability fixture; failure makes the validator require `--disable-radix-cache` until resolved. The 55 % prefix-cache-hit workload is preserved.
  - Codex Position: Apples-to-apples concern resolved with the two-column comparison on the same single-instance server.
  - Tradeoff Summary: The page-stability fixture is the engineering check; the validator default is local to DS and not entangled with HiSparse.
  - Decision Status: `PENDING` (revisit after Milestone 1 lands). User explicitly chose to defer until M1's validator scaffolding is in place. Claude default remains in force in the meantime; M3-B's page-stability fixture is the engineering gate. Does not block any pre-M3 work.

- DEC-3: Quality threshold deltas vs native_nsa
  - Claude Position: NIAH @ {4K, 16K, 64K} within 5 pp of native_nsa; MMLU within 1.0 pp of native_nsa.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: Tighter thresholds fail for cosmetic reasons; looser thresholds miss silent regressions.
  - Decision Status: **RESOLVED**. Adopt Claude's defaults as written (NIAH ≤ 5 pp delta, MMLU ≤ 1.0 pp delta). If the thresholds prove too restrictive during M5 quality work, loosening is allowed via a follow-on amendment to AC-9 — but do not pre-loosen in anticipation. AC-9 cites these numbers as the contract.

- DEC-4: Calibration ownership and artifact distribution
  - Claude Position: SGLang ships the calibration **script** under `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` and documents the production recipe under `docs/advanced_features/double_sparsity_calibration.md`. The **channel mask file itself** is **external** to the repo: produced by the deploying team, stored in their model registry / object store.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: Shipping the file in-repo bloats the wheel and pins model revisions. A tiny NSA-shaped CI fixture is the compromise.
  - Decision Status: **RESOLVED**. Adopt Claude's position: the calibration script ships in the repo at the named path; the production channel mask file is not tracked by the repo. CI uses a tiny NSA-shaped fixture; production builds reference the externally-stored file via `--double-sparsity-config '{"channel_mask_path":"<...>"}'`. The deploying team owns artifact storage.

- DEC-5: Semantic relationship of DS to DeepSeek-V3.2 NSA
  - Claude Position: On DeepSeek-V3.2, the DS selector **replaces the NSA `Indexer.forward()` selection role** (per `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` line 178). NSA quant / dequant / cache plumbing remains authoritative and is unchanged. DSv4 is out of scope.
  - Codex Position: "DS as an alternative selector / label-cache path for DeepSeek-V3.2 sparse attention."
  - Tradeoff Summary: Stacking DS on top of NSA would double-filter an already sparse set. Replacing the selector keeps the same level of sparsity with a different mechanism.
  - Decision Status: **RESOLVED**. Adopt Claude + Codex direction: DS is an alternative to NSA's selection role. Do not stack DS after NSA (no double-filtering of an already sparse set). Leave the rest of NSA (quant / dequant / cache plumbing) untouched.

- DEC-6: Scope of deferred-requirements coverage
  - Claude Position: GLM-5, 128K ISL, FP4 weights, **HiSparse integration, PD-Disagg integration, HiCache integration** are explicitly OUT of the initial scope. The selector ABI (AC-11), the channel-mask schema (AC-4), and the validator pattern are shaped to admit them without rewrite; task 6 produces a one-page schema-compatibility memo before the loader merges.
  - Codex Position: Round 1 surfaced the question; CMT-1 (user) confirmed HiSparse / HiCache integration is downstream.
  - Tradeoff Summary: Including any deferred requirement in initial scope blows the milestone budget; excluding them entirely risks a redesign later. The shape-now / behavior-later compromise is encoded in AC-11 (ABI minimal but stable) and task 6 (schema memo).
  - Decision Status: **RESOLVED** with the following nuanced split:
    - **HiSparse integration: OUT of initial scope AND out of downstream scope.** HiSparse is in flux right now; if a redesign to integrate is ever justified, it will be a separate plan after current changes settle. No future-work bullet promises HiSparse adapter coverage. PD-Disagg and HiCache integration also remain out of initial scope but stay in the "kept in mind" future-work category (they are commonly-asked features; no customer has asked for them on top of DS yet).
    - **GLM-5 (specifically GLM-5.1) is a deferred-but-hard requirement.** The selector ABI, channel-mask schema, and validator must be designed to admit GLM-5.1 when it lands — see DEC-10's capability-check approach. Task 6's schema memo treats GLM-5.1 as the load-bearing forward-compat target, with 128K ISL and FP4 weights as secondary targets.
    - **128K ISL and FP4 weights**: deferred. Schema admits them but no implementation work in initial scope.

- DEC-7: "Extensions as a general knob for the sglang engine" interpretation
  - Claude Position: Interpreted as "expose DS runtime knobs through `--double-sparsity-config`'s JSON blob (`top_k`, `page_size`, `channel_mask_path`, `device_buffer_size`, free `extra`); no new top-level CLI flags per knob; no plugin system."
  - Codex Position: "Vague; needs user confirmation."
  - Tradeoff Summary: A plugin system is a separate design effort; a config blob fits the existing surface and is reversible.
  - Decision Status: `PENDING` (low-stakes; default ships as Claude position).

- DEC-8: HiSparse + DS coexistence (from CMT-3)
  - Claude Position: For the initial deliverable, ship a startup error if both `--enable-double-sparsity` and `--enable-hisparse` are set, with the message "Coexistence of DS and HiSparse is deliberately undesigned for the standalone v1 deliverable. The HiSparse-PD-disagg decoupling required to compose them is its own plan." Do **not** describe this as "not yet integrated" — that wording implies design intent that does not exist.
  - Codex Position: "Startup guard is acceptable for a first standalone deliverable, but the plan must either delete future coexistence claims or define the eventual composition model now."
  - Tradeoff Summary: Guard is cheap and honest. Defining composition now blows scope and depends on a HiSparse decoupling that nobody has scoped. Removing future-work claims removes optionality.
  - Decision Status: **RESOLVED**. Adopt Option 1: ship a startup error if both `--enable-double-sparsity` and `--enable-hisparse` are set. The two packages may coexist **in the codebase** (both can live in the source tree) but are **mutually exclusive at runtime**. Remove all future-work claims that DS will be wrapped as a HiSparse algorithm; the client has no expectation of using HiSparse with DS. AC-1's negative test cites this resolution directly with a "mutually exclusive; no plans to integrate" error message.

- DEC-9: TP rank synchronization of page selection (from CMT-7 / Codex C-NEW-2)
  - Claude Position: **Globally synchronized via per-layer per-step all-reduce of partial head scores.** Per-rank top-K is then computed from the all-reduced scores, guaranteeing every TP rank computes the same `selected_indices` and `block_table`.
  - Codex Position: "Specify whether page selection is per-rank by design or globally synchronized; if global, define the reduction/all-gather path and test rank agreement."
  - Tradeoff Summary: Per-rank selection avoids the all-reduce but breaks the backend metadata invariant that all ranks see the same page table (silent wrong attention). Global selection incurs measurable but bounded all-reduce cost.
  - Decision Status: **RESOLVED** with the following implementation contract (per the user's CMT-8 spec):
    1. **Do not all-gather page signatures.** Page signatures stay TP/head-sharded; signatures are never moved across ranks.
    2. Each rank computes a **scalar page score** per page from its local head shard (locally-reduced over head_dim and the head shard).
    3. **All-reduce the scalar scores** across the attention TP group (`SUM`). The reduction is on `[max_pages]`-shaped fp16/fp32 scalar tensors, bandwidth-small but latency-sensitive (one collective per layer per decode step).
    4. Every rank runs deterministic top-K independently from the same reduced scores; the result is bit-equal across ranks by construction.
    5. **Rank-agreement test** lives in `task16`: construct a setup where per-rank local top-K would diverge (heads with imbalanced local scores) and assert that the all-reduced result agrees on all ranks.
    6. **Validation order**: prefer 8-way H200 TP / 2× replicas behind SMG first (intra-node NVLink-only all-reduce). Cross-node 16-way TP is tested second; expect higher latency.
    7. **Fallback**: if per-step score all-reduce is too costly on 16-way cross-node TP, the plan may pivot back to per-rank selection (with a benchmark proving the rank-divergence does not introduce a serious perf-vs-quality degradation). The pivot is recorded as a future amendment, not a default.

  Precedent note: Legacy SGL and Twilight both appear to do per-head local selection without a TP rank-agreement path; that precedent does not carry over here because the V3.2 / FlashMLA page-table metadata contract is shared across ranks.


- DEC-10: MLA-capability validator (from CMT-15)
  - Claude Position: The validator checks for the presence of a `nsa.Indexer` on the attention layer (the hook-site precondition) rather than a model-name string match like `is_deepseek_nsa`. The capability check generalizes naturally to any model that exposes the same `Indexer` interface (e.g. GLM-5.1 once it lands), without name-string special cases.
  - Codex Position: Agreed with Linus on CMT-15: either V3.2-specific or capability-checked, not both.
  - Tradeoff Summary: Capability check is honest about scope and avoids the maxim violation of name-string special cases. A pure model-name check is simpler to write today but bakes a V3.2 assumption into the validator that has to be undone for GLM-5.1.
  - Decision Status: **RESOLVED**. Adopt the capability-check approach. The user explicitly cited GLM-5.1 as a hard deferred requirement (DEC-6 split) — therefore the validator must be designed so GLM-5.1 falls in for free as long as it exposes the same `nsa.Indexer` hook point. The validator code is V3.2-tested at land time; the capability gate is the generalization seam.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. "channel mask loader", "page signature table", "page_signature_write", "selector kernel").
- Follow `.claude/rules/speculative-naming.md` for any identifier that crosses into the speculative-decoding namespace (this plan does not, but kernels reused from PR #25304 should be re-named if they leaked speculative terms).
- Tensors are plural (`selected_indices`, `valid_lengths`, `page_signatures`); scalars are singular.
- Count prefix `num_` for counts, `_ct` for accumulators, `_total` for monotonic counters in Prometheus, `_rate` for ratios, no prefix for content arrays.
- Shared-helper rule (replaces "copy don't import" per CMT-19): if a pattern from `python/sglang/srt/mem_cache/sparsity/`, `python/sglang/srt/jit_kernel/hisparse.py`, or `python/sglang/srt/managers/hisparse_coordinator.py` is useful inside the new `double_sparsity/` package, factor it into `python/sglang/srt/utils/sparse_helpers.py` (a new neutral module) and import from both. Do **not** copy code with adapted imports — that creates drift with no clear owner. Do **not** reach across the boundary directly into HiSparse internals. The neutral helper is the only sanctioned bridge.

--- Original Design Draft Start ---

I am in the middle of delivering a double sparsity implementation into SGLang.

Here are the immediate client requirements:
- Model: deepseek-ai/DeepSeek-V3.2 (FP8)
- Inference SLOs: 30 tokens/s with a P99 TTFT of < 22s
- Workload: 4096 ISL, 512 OSL, max-concurrency: 64, minimum concurrency: 16, Cache hit: ~55% (benchmark found in development/benchmark.sh) 
- Page size: 64 (technically not explicitly listed as a hard requirement, but significantly preferred and implementation should support different page sizes)

Deferred Client requirements ordered from most important to least:
1. zai-org/GLM-5.
2. 128k ISL, 1024 OSL.
3. nvfp4 and mxfp4 quantizated weight support.

Downstream requirements after client deliverables:
1. Twilight (top-p selection instead of top-k)
2. Extensions as a general knob for the sglang engine
3. Integration into all other sglang features, like PD-Disagg and HiSparse.

Double Sparsity Implementation Sources (listed from most recent to least recent)
1. Twilight: https://github.com/tsinghua-ideal/Twilight, with https://github.com/tsinghua-ideal/flash-topk-attention/tree/d8803b29961c44d77a747636ad4282bd7a9094af
2. Legacy SGL implementation (parent commit of commit that removed double sparsity): https://github.com/sgl-project/sglang/tree/29f56cb2304bf6699da78e4e5a738fb794babcfd/python/sglang/srt/layers/attention 
3. Original Author Implementation: https://github.com/andy-yang-1/DoubleSparse
4. Paper: https://arxiv.org/pdf/2408.07092

Hi-Sparse Implementation (Quite irrelevant, but an example of succesfull ship of performant sglang sparsity feature, design should ideally be inspired by this)
Guide: https://docs.sglang.io/docs/advanced_features/hisparse_guide
PRS: https://github.com/sgl-project/sglang/pull/20343, https://github.com/sgl-project/sglang/pull/23013, https://github.com/sgl-project/sglang/pull/21591


I am deciding between whether I should
1. Resume from the legacy sglang restoration: https://github.com/sgl-project/sglang/pull/22992
2. Resume from the current rewrite session: https://github.com/sgl-project/sglang/pull/25304/commits
3. Restart from scratch on a new branch of SGLang as this will be a huge downstream decision. I am leaning towards this as neither of the previous options were designed/created before the clients gave us these requirements

Help me first decide whether I should resume or restart from scratch on Sglang.
--- Original Design Draft End ---
