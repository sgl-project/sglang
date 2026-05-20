# Plan: Deliver Standalone Double Sparsity for DeepSeek-V3.2 (FP8)

## Goal Description

Deliver a production-quality Double Sparsity (DS) implementation in SGLang that meets the immediate client SLO on DeepSeek-V3.2 (FP8), is forward-compatible with deferred client requests (GLM-5, 128K ISL, FP4 weights), unlocks downstream work (Twilight top-p selection, "Extensions" engine knob, eventual PD-Disagg and HiSparse integration), and ships in a shape that is upstream-reviewable in `sgl-project/sglang`. The plan also resolves the open question "resume vs restart" by recommending a path and stating the explicit cost of the alternatives.

<comment>
[Linus] This is a plan document for a SELECTOR FUNCTION. Twelve ACs, twenty tasks, seven PENDING decisions, seven milestones, a deliberation log, a QA ledger — and the actual code is what, two thousand lines including tests? You have not proposed an architecture, you have proposed a project-management timesheet. Strip this down: what is the selector signature, what page table does it emit, where does it hook in? Three pages, not ten. Until the data flow fits in two paragraphs nobody will read past AC-3, and AC-3 is where the real questions start.
</comment>

### Standalone, Not a HiSparse Algorithm

This refined plan reflects an explicit user redirect (recorded in `## Claude-Codex Deliberation`): Double Sparsity must work **standalone**, without requiring `--enable-hisparse`, without requiring PD-disaggregation, and without depending on the HiSparse memory tiering / coordinator stack. Today's HiSparse in `sgl-project/sglang` is wired only to PD-disaggregation decode instances (verified in `python/sglang/srt/managers/scheduler.py` — `_build_hisparse_decode_batch`, `set_decode_producer_stream`, and the `disaggregation/decode.py` admission path), so building DS on top of it would inherit that PD-only constraint and contradict the client requirement that users must be able to run DS on a single-instance server. HiSparse is treated as **inspiration only**: the plan permits borrowing Triton kernels, calibration patterns, and configuration shapes from HiSparse code, but it does not register DS as a HiSparse algorithm or invoke `SparseCoordinator`. Integration with HiSparse and HiCache is downstream of the initial client deliverable, per the draft's "Downstream requirements after client deliverables (3) Integration into all other sglang features, like PD-Disagg and HiSparse."

### Resume-vs-Restart Recommendation

**Restart from a fresh branch off current `main`. Implement Double Sparsity as a standalone feature under a new module (`python/sglang/srt/layers/attention/double_sparsity/`) that hooks into the DeepSeek-V3.2 NSA attention path via its own selector, gated by a new `--enable-double-sparsity` CLI plus a `--double-sparsity-config` JSON. Cherry-pick selection / `K_label` / Triton kernel work from PR #25304 where it directly supports the DeepSeek-V3.2 MLA + FP8 path. Close PR #22992. Keep PR #25304 open as a reference link until the new branch reaches kernel parity.**

Rationale:
- PR #22992 (`dev/double-sparsity-reintro`, +1873 / 12 files): restores the legacy Llama-only, page=1, Triton-attention DS backend. The PR body itself documents a 3–12% throughput regression on H100 and states "Performance optimization is planned for follow-up work." It has no MLA path, no FP8 path, no page=64 support, and predates DSv3.2 NSA. **Not a viable base.**
- PR #25304 (`dev/double-sparsity-v2`, +22552 / 90 files): contains valuable selection kernels and calibration scaffolding (M1 skeleton; M2 K_label storage + write kernels; M3 selection pipeline; M4 FA3 adaptor; v1.1 stage-1 / stage-2 Triton block-topk + score-aware union + CUDA-graph capture; v2 native sparse-decode kernels) and reaches a "BOTH GATES PASS" point at conc=32 / 128K / tb=8192. **But** the backbone is FA3 (Llama-style dense attention), not MLA / DeepSeek-V3.2; it uses a custom coordinator instead of any framework; the PR carries `HANDOFF_NATIVE.md`, `SESSION_REPORT_*.md`, pensieve installs and ad-hoc bench harnesses; it has no PR description and no CI run. **Not upstream-shippable as-is** but is a valuable kernel source.
- Restart on `main` lets the work land as a focused, standalone module that can be reviewed independently and is not blocked on the HiSparse + PD-disagg coupling.

### Two Different "Labels"

Two terms must not be confused; the plan separates them throughout:
- **Calibration artifact** (offline, file on disk, produced by a calibration script): channel-importance schema for the model. Per-layer / per-head selection of which channels matter (typical `label_dim` 16–32), plus model-revision metadata. Static for a given model revision; ships separately from the SGLang wheel.
- **Runtime label cache** (online, GPU tensor, populated during serving): per-served-KV-page label tensor produced by projecting K-cache pages through the channels named in the calibration artifact. Allocated when DS is enabled, written on prefill (and incrementally on decode) by a `K_label` write kernel cherry-picked from PR #25304 commit `567eff67b`. Affected by prefix-cache hits (already-cached pages can reuse labels).

<comment>
[Linus] If you need a seventy-word section in your goal description to disambiguate two concepts called "calibration artifact" and "runtime label cache" — and BOTH names contain the word "label" — your names are wrong. The offline thing is a CHANNEL MASK or CHANNEL SELECTION. The online thing is a PAGE SIGNATURE TABLE. Pick names that make confusion impossible, then delete this section because nobody will need it.
</comment>

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Double Sparsity is exposed via a new top-level pair of server args: `--enable-double-sparsity` (boolean) plus `--double-sparsity-config` (JSON, modelled on `--hisparse-config` but parsed by an independent validator). DS does **not** require `--enable-hisparse`, does **not** require `disaggregation_mode != "null"`, and does **not** register itself as a HiSparse algorithm.
  - Positive Tests:
    - Starting an SGLang server with `--enable-double-sparsity --double-sparsity-config '{"top_k":2048,"page_size":64,"calibration_artifact_path":"<...>"}'` on DeepSeek-V3.2 (FP8) on a single-instance server (no `--disaggregation-mode`) initialises the DS module and routes V3.2 attention through the DS selector.
    - Running with `--enable-double-sparsity` while `--enable-hisparse` is **not** set succeeds; the two features are independent.
  - Negative Tests:
    - `--enable-double-sparsity` together with `--enable-hisparse` fails at startup with a documented "Double Sparsity and HiSparse are not yet integrated; enable one or the other for the initial deliverable. See downstream-integration roadmap." message.
    - `--enable-double-sparsity` without `--double-sparsity-config` fails at startup with a clear "calibration_artifact_path required" message (the config blob must at minimum carry the artifact path).

<comment>
[Linus] AC-1's "--enable-double-sparsity together with --enable-hisparse fails at startup with 'not yet integrated'" is admitting upfront that two sparsity features in the same engine cannot compose. That is not orthogonality — that is a bug guarded by an error message. Pick: (a) figure out composition now, or (b) accept HiSparse + DS will never coexist and drop the future-work bullet. The middle ground — ship a runtime guard that says "we'll fix this later" — is the worst option, because the code knows it should compose and you've punted.
</comment>

- AC-2: The DS selector hooks into the DeepSeek-V3.2 NSA attention path under `python/sglang/srt/layers/attention/double_sparsity/` and replaces the NSA indexer's token-selection role while leaving NSA's quant / dequant / cache plumbing (`python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `dequant_k_cache.py`, the Triton + tilelang kernels, MTP precompute / verification) authoritative and untouched. The hook lives in the V3.2 attention path itself; it does not modify `mem_cache/sparsity/` or `managers/hisparse_coordinator.py`.
  - Positive Tests:
    - Unit test calls `DoubleSparsitySelector.retrieve_topk(queries, layer_id, req_pool_indices, sparse_mask)` and receives `(selected_indices, valid_lengths)` with shapes `[bs, max_top_k]` / `[bs]` as documented by the contract.
    - Integration test asserts that the page table reaching the FlashMLA backend during a DS-enabled decode step contains only the page IDs returned by `retrieve_topk` (covered by a metadata-assertion fixture, not by mutating production code).
    - Single-decode-step golden test on DeepSeek-V3.2 (FP8) produces attention output identical bit-for-bit to a reference computation restricted to the same selected pages.
  - Negative Tests:
    - Calling the selector on a request with `sparse_mask=False` returns the documented "unselected (dense)" sentinel and the integration path takes the native NSA branch.
    - Submitting a request before the runtime label cache is populated (forced via fault-injection switch) fails the request with a documented error rather than producing garbage output.
    - The diff against `main` adds zero files under `python/sglang/srt/mem_cache/sparsity/algorithms/` and zero registrations in `python/sglang/srt/mem_cache/sparsity/factory.py::_ALGORITHM_REGISTRY` (verified by `git diff --name-only`).

<comment>
[Linus] "DS replaces the NSA indexer's token-selection role" — replaces HOW. Monkey-patch `DeepSeekV32Attention.forward`? Add a `if config.enable_double_sparsity` branch inside `nsa_indexer.py`? Fork the model class? This is the load-bearing architectural question of the whole plan and you have left it as a verb. Until you can sketch the actual edit-site in one sentence, AC-2 is a wish, not a contract.

Also: "bit-for-bit identical attention output" on FP8 is a fantasy. FP8 GEMM is not deterministic across launch orderings. State a tolerance or pick a different correctness check.
</comment>

<comment>
[Codex] `selected_indices` is not a contract. FlashMLA needs physical block-table entries with causal lengths, not a bag of ranked logical page ids. Decide now: selector returns logical positions and an adapter maps/sorts them into the backend block table, or selector returns physical page ids already ordered for the kernel. Passing top-k score order into a paged attention backend that expects sequence order is how you get silent wrong attention.
</comment>

- AC-3: Page size 64 works on DeepSeek-V3.2 (FP8) and at least one alternate page size is exercised in tests. The implementation does not hard-code page=64. Backend / dtype pairing follows the V3.2 MLA backend rule: `fp8_e4m3` ↔ FlashMLA FP8 path, `bfloat16` ↔ FlashMLA BF16 path.
  - Positive Tests:
    - End-to-end smoke run with `kv_cache_dtype=fp8_e4m3`, `page_size=64`, V3.2 FP8 MLA backend succeeds for one warm prefill + one decode batch.
    - End-to-end smoke run with `kv_cache_dtype=bfloat16`, `page_size=64`, V3.2 BF16 MLA backend succeeds for one warm prefill + one decode batch.
    - Unit tests for the selection kernel run with `page_size in {32, 64}` and produce identical logical top-K rankings on a deterministic fixture (modulo page granularity).
  - Negative Tests:
    - Configuring `kv_cache_dtype=fp8_e4m3` with a BF16-only MLA backend (mismatched pair) fails at startup with an explicit backend / dtype mismatch error emitted by the new DS validator.
    - Configuring an unsupported page size (e.g. `page_size=7`) for DS fails at startup with an explicit page-size validation error, not silently.

- AC-4: A persisted **calibration artifact** is required to enable Double Sparsity. The artifact is validated against the loaded model and configuration before serving begins. Missing or mismatched artifacts cause fail-fast at startup; there is no silent dense fallback in production.
  - Positive Tests:
    - Loading a valid artifact (matching `model_revision_sha`, `head_dim`, `tp_world_size`, `dtype`, `page_size`, `label_dim`) succeeds and the DS selector reports `calibration_loaded=true` on every TP rank.
    - The loader emits a startup log line listing artifact `model_revision_sha`, `head_dim`, `tp_world_size`, `dtype`, `page_size`, `label_dim`, `created_at`.
  - Negative Tests:
    - Missing artifact path or unreadable file fails the server before the engine starts, with a non-zero exit code and a "calibration artifact required" message.
    - Artifact whose `model_revision_sha`, `head_dim`, `dtype`, `tp_world_size`, or `page_size` mismatches the running configuration fails the server with a message naming each mismatched field.

<comment>
[Linus] Validating `model_revision_sha + head_dim + tp_world_size + dtype + page_size + label_dim` is paranoid bookkeeping that will not catch the actual bug class. Calibration done on a different LoRA fine-tune of the same revision passes every one of those checks and produces garbage. Either ship a content hash of the channel mask plus an end-to-end NIAH sanity probe on startup (which catches the real bugs), or trust the user and delete the validator. `head_dim` and `tp_world_size` are derivable from the running config; embedding them in the artifact metadata is an excuse to fail-fast on a non-bug.
</comment>

<comment>
[Codex] `tp_world_size` in the artifact is not enough for tensor parallel correctness. If each TP rank computes page scores from its local shard, ranks can choose different page tables for the same request, which breaks backend metadata assumptions and makes output rank-dependent. Specify whether page selection is per-rank by design or globally synchronized; if it is global, define the reduction/all-gather path and test rank agreement.
</comment>

- AC-5: A calibration script produces a calibration artifact for DeepSeek-V3.2 (FP8) and is invocable as a standalone command. CI uses a tiny NSA-shaped fixture; the production recipe runs on the agreed hardware (DEC-1) and is documented but not committed.
  - Positive Tests:
    - `python -m sglang.srt.layers.attention.double_sparsity.calibrate --model <tiny-NSA-fixture> --dtype fp8_e4m3 --tp 1 --output /tmp/labels.safetensors` runs to completion in CI under a minute and writes a non-empty artifact with the documented schema.
    - The CI artifact loads successfully via AC-4's loader against the same fixture.
    - The production recipe is documented in `docs/advanced_features/double_sparsity_calibration.md` with the exact CLI invocation, expected dataset, agreed hardware, and expected wall-clock.
  - Negative Tests:
    - Running calibration with `--model` pointing to an unsupported architecture fails with a clear "DoubleSparsity calibration is only supported for ..." message.
    - Running calibration with `--tp 8` on a 1-GPU box fails before allocation with a config error, not at first NCCL call.

- AC-6: CUDA-graph piecewise capture / replay works for the decode path with `--enable-double-sparsity` at the target concurrencies (16, 32, 64). The selection ABI uses static / max-bounded output buffers so capture is safe.
  - Positive Tests:
    - Decode-path piecewise CUDA graph captures at conc=64 without `CUDA error: launch failed` and replays for at least 100 steps on a fixed batch.
    - The selector's output buffers are allocated with static shape `[bs, max_top_k]` for `selected_indices` (padded with `-1`) and `[bs]` for `valid_lengths`; the same buffers are reused across capture and replay.
  - Negative Tests:
    - Setting `max_top_k` smaller than `top_k` fails at startup, not at capture.
    - Removing CUDA-graph capture (per-step eager) does not regress correctness — golden output is unchanged.

<comment>
[Codex] Static output buffers do not make CUDA graph capture safe. Grid sizes, scratch buffers, page-table writes, valid page counts, dense sentinel branching, and radix-cache hit shapes also have to be replay-stable. Define the captured decode ABI as fixed input/output tensors plus device-side masking, and ban CPU-side allocation or metadata rebuilding inside the captured region.
</comment>

- AC-7: A native-NSA baseline (DS off — DeepSeek-V3.2 running with its built-in NSA selection on a single-instance server) is recorded on the same hardware, same model revision, same workload, same radix-cache setting, and same concurrency as the DS run. **No HiSparse baseline is required for the initial deliverable**; it is documented as an optional follow-up benchmark under the future-work section.
  - Positive Tests:
    - `development/benchmark.sh` plus a sibling `development/benchmark_baseline.sh` produce a side-by-side report with two columns: `double_sparsity_off (native NSA)`, `double_sparsity_on`. Per-column rows: per-request output tok/s P50 / P99, TTFT P50 / P99, TPOT P50 / P99, and goodput-under-SLO.
  - Negative Tests:
    - The report fails to publish if any of {model_revision_sha, GPU id, TP size, page size, radix-cache setting, concurrency} differs between baseline and DS rows.

<comment>
[Linus] "DS off (native NSA)" vs "DS on" is incoherent. Native NSA is not a flag, it is the model's default forward pass. If DS REPLACES the NSA selector (per AC-2 and DEC-5), then "DS off" runs no DS pipeline at all — it runs the unmodified model. Call the columns `native_nsa` and `double_sparsity` (two different selectors, A/B'd), or admit there is one running mode and the baseline is "without DS patch". The current naming betrays a mental model where DS is a stacked feature; everywhere else the plan says it is a replacement.
</comment>

- AC-8: The DS run meets or beats the immediate SLO under the clarified throughput definition (see DEC-1): **per-request output throughput P50 ≥ 30 tok/s** and **P99 TTFT ≤ 22 s, including scheduler-queue wait**, at `max-concurrency=64` (also at `min-concurrency=16`) on the workload defined in `development/benchmark.sh` (ISL≈4096, OSL=512, ~55 % prefix-cache hit on the agreed hardware (DEC-1)).
  - Positive Tests:
    - `bench_serving` over `gsp_isl4096_osl512_c64.jsonl` reports per-request output tok/s P50 ≥ 30 and P99 TTFT ≤ 22 s.
    - The same benchmark at conc=16 reports per-request output tok/s P50 ≥ 30 and P99 TTFT ≤ 22 s.
    - The reported `dense_fallback_count` per request is zero (AC-10 metric); if non-zero, the SLO claim is invalid.
  - Negative Tests:
    - A "no-op detector" gate fires if any of the following hold for the DS run: `selected_pages == total_pages` (every page chosen, i.e. no selection happened); `dense_fallback_count != 0` (selection failed and dense took over silently); or the FlashMLA metadata-assertion fixture from AC-2 fails to confirm a restricted page table reached the kernel. Any one of these invalidates the SLO claim.

- AC-9: Quality gates pass against the native-NSA baseline. Agreed thresholds (DEC-3): NIAH retrieval @ 4K / 16K / 64K within 5 percentage points of native-NSA score; MMLU within 1.0 percentage point of native-NSA score.
  - Positive Tests:
    - `test/manual/test_double_sparsity_v32.py` runs NIAH at 4K / 16K / 64K and reports DS scores within 5 pp of native NSA on each length.
    - MMLU 5-shot on DeepSeek-V3.2 (FP8) is within 1.0 pp of native NSA.
  - Negative Tests:
    - A run with a deliberately corrupted calibration artifact (random-permuted channel selection) makes NIAH @ 64K drop more than 20 pp below native-NSA baseline, confirming the test is sensitive to artifact quality.
    - A run with an empty / zero runtime label cache (fault-injected) makes NIAH @ 16K drop more than 30 pp, confirming the test is sensitive to the runtime label cache.

- AC-10: Observability surfaces are exposed per step and aggregated per request. Healthy DS runs report `dense_fallback_count == 0` and `calibration_artifact_valid == 1` on every TP rank. Per-request `meta_info` carries `sparsity_rate` (selected / total page count), `selected_pages` (count), and `dense_fallback` (0/1). Prometheus exposes `sglang_double_sparsity_*` gauges and counters (note the namespace is `sglang_double_sparsity_*`, **not** `sglang_hisparse_double_sparsity_*`, because DS is standalone).
  - Positive Tests:
    - A scrape of `/metrics` after a healthy 64-concurrency run shows `sglang_double_sparsity_calibration_artifact_valid = 1`, `..._dense_fallback_total = 0`, non-zero `..._selected_pages_sum` and `..._selected_pages_count`.
    - Per-request `meta_info` carries `sparsity_rate`, `selected_pages`, `dense_fallback` and they aggregate to the Prometheus values modulo sampling.
  - Negative Tests:
    - A fault-injected run with an unwritable runtime label cache (forced by test flag) increments `..._dense_fallback_total` and the test asserts this — confirming the metric is wired, without requiring a fallback in production.
    - With `--disable-metrics` the server still serves and produces correct outputs; the per-request `meta_info` fields are unaffected.

- AC-11: The selection ABI accepts both **fixed top-k** (initial scope) and a future **bounded top-p** (Twilight, deferred). The ABI is shape-locked from Milestone 3; top-p **behavior** is deferred behind a `selection_mode` parameter that defaults to `TOPK`. `--double-sparsity-config` carries algorithm-specific knobs (`min_top_k`, `max_top_k`, `top_p`).
  - Positive Tests:
    - Triton selection kernel signature accepts a `selection_mode` argument (`TOPK`, `TOPP`); the `TOPK` path is exercised end-to-end; the `TOPP` path passes a unit test on synthetic scores producing `valid_lengths` clipped to `[min_top_k, max_top_k]`.
    - `selected_indices.shape == (bs, max_top_k)` invariant holds under both modes; no shape change between calls.
  - Negative Tests:
    - Requesting `selection_mode=TOPP` end-to-end on the server (not the unit test) fails with a documented "not yet enabled" error until Twilight ships.
    - Requesting an unknown `selection_mode` fails with an enum error.
    - Requesting `top_p` with `max_top_k > device_buffer_size` fails at startup, not at first decode.

<comment>
[Linus] `selection_mode=TOPK / TOPP`, TOPK shipped, TOPP unit-test only — exactly the YAGNI flavor where you bake a future API in early "for forward compatibility" and then discover the buffer shape was wrong when the future arrives. If Twilight is deferred, defer the ABI. Add the parameter when Twilight ships; the renaming pain is real but smaller than the pain of carrying an unused enum through every kernel signature for months. Ship `retrieve_topk(queries, ...) -> (indices[bs, K], lens[bs])` and stop designing for hypothetical futures.
</comment>

- AC-12: The shipping branch is upstream-shaped. No `HANDOFF*.md`, `SESSION_REPORT*.md`, pensieve installs, ad-hoc bench harnesses, or workspace notes are committed. The branch may modify files under these paths, but each commit is reviewable.
  - Positive Tests:
    - `git log --name-only origin/main..HEAD` shows only files under: `python/sglang/srt/{layers/attention/double_sparsity, server_args.py, model_executor, metrics, managers}`, `sgl-kernel/`, `test/`, `docs/`, `development/benchmark*.sh`. Notably absent: `python/sglang/srt/mem_cache/sparsity/algorithms/double_sparsity/` (DS is not a HiSparse algorithm) and `python/sglang/srt/arg_groups/hisparse_hook.py` (DS does not extend HiSparse validation).
    - `git diff --stat origin/main..HEAD` is bounded by an agreed budget (DEC-4) and individual commits are < ~1500 lines each except for the Triton kernel commit (allowed to be a single logical unit).
  - Negative Tests:
    - A pre-commit hook (added in `task20`) blocks any session-artifact filename pattern (`HANDOFF*.md`, `SESSION_REPORT*.md`, `*.HANDOFF.md`, top-level pensieve dirs added to git).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The implementation delivers a standalone `python/sglang/srt/layers/attention/double_sparsity/` module with its own selector hooked into the DeepSeek-V3.2 NSA attention path, an FP8-aware calibration-artifact loader and validator, a runtime label-cache populator using a cherry-picked `K_label` write kernel from PR #25304, a calibration script that defaults to NIAH-shaped synthetic data plus an opt-in dataset hook, Triton selection kernels cherry-picked and re-MLA-ified from PR #25304 (stage-1 block-topk + stage-2 merge, score-aware union, capture-safe), top-k baseline with a top-p-ready selector ABI (`selection_mode=TOPK` shipped, `TOPP` unit-tested), CUDA-graph piecewise capture, per-request and Prometheus observability under `sglang_double_sparsity_*`, an MLA + FP8 quality regression suite (NIAH at 4K / 16K / 64K plus MMLU), a benchmark harness that publishes side-by-side `DS off (native NSA)` / `DS on` results on a single-instance server, and a written future-work section that describes (without implementing) the path to HiSparse + PD-Disagg integration.

### Lower Bound (Minimum Acceptable Scope)
The implementation adds `--enable-double-sparsity` and `--double-sparsity-config` server args with an independent validator, lands the DS selector module hooked into the V3.2 NSA path, ships a calibration script that produces a valid artifact for one fixed NSA-shaped fixture (CI) and documents the production recipe, validates the artifact at load time, populates the runtime label cache on prefill, meets AC-8 SLO at `max-concurrency=64` and at `min-concurrency=16` on a single-instance server, passes AC-9 quality gates with the agreed deltas, exposes the AC-10 metrics minimally (`selected_pages`, `dense_fallback_count`, `calibration_artifact_valid`), and produces an upstream-shippable branch (AC-12). Top-p selection runtime behavior, GLM-5, 128K ISL, FP4 weights, and any HiSparse / PD-Disagg integration are deferred but the selector ABI (AC-11) and the calibration-artifact schema (AC-4 fields) are shaped to admit them without rewrite.

### Allowed Choices
- Can use: a new top-level CLI surface (`--enable-double-sparsity`, `--double-sparsity-config`), Triton selection kernels (cherry-picked from PR #25304 and adapted), CUDA-graph piecewise capture, `safetensors` for the calibration-artifact format, the existing FlashMLA backends (`flashmla_kv` for FP8, `flashmla_sparse` for BF16) as the underlying dense kernel, **code patterns** borrowed from HiSparse (e.g. `BaseSparseAlgorithm`-style abstract signatures, `K_label` write kernel from `jit_kernel/hisparse.py` or its PR-#25304 sibling, `SparseConfig` JSON shape) without taking a runtime dependency on HiSparse infrastructure.
- Cannot use: registering DS in `python/sglang/srt/mem_cache/sparsity/factory.py::_ALGORITHM_REGISTRY`; calling `SparseCoordinator`; calling `HiSparseCoordinator`; extending `python/sglang/srt/arg_groups/hisparse_hook.py`; gating DS behind `--enable-hisparse`; requiring `disaggregation_mode != "null"`; the legacy `double_sparsity_backend.py` re-imported from PR #22992; `HANDOFF*.md` / `SESSION_REPORT*.md` files committed to the upstream branch; FA3-only kernels on the V3.2 path.

> **Note on Deterministic Designs**: The user redirect ("Absolutely not, ...") narrows the architectural choice: DS is standalone. Within that constraint, the kernel-cherry-pick vs. reimplement choice is the main flexibility; both options are acceptable as long as the result hooks into the V3.2 NSA path without touching HiSparse infrastructure.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

1. Add `python/sglang/srt/layers/attention/double_sparsity/` as a **package**, peer to `nsa/` and `dsv4/`:
   - `__init__.py` → re-exports `DoubleSparsitySelector`, `DoubleSparsityConfig`, `validate_double_sparsity`.
   - `selector.py` → `class DoubleSparsitySelector` exposing `retrieve_topk(queries, layer_id, req_pool_indices, sparse_mask) -> (selected_indices, valid_lengths)`. ABI matches the `BaseSparseAlgorithm.retrieve_topk` signature for forward compatibility, but the class does NOT inherit from `BaseSparseAlgorithm` and is NOT registered in `_ALGORITHM_REGISTRY`.
   - `config.py` → `@dataclass DoubleSparsityConfig` (parsed from `--double-sparsity-config` JSON; fields `top_k`, `max_top_k`, `min_top_k`, `top_p`, `selection_mode`, `page_size`, `calibration_artifact_path`, `device_buffer_size`, `extra`).
   - `calibration.py` → calibration-artifact `safetensors` loader + validator + schema definition.
   - `calibrate.py` → standalone CLI entry point for offline calibration (`python -m sglang.srt.layers.attention.double_sparsity.calibrate`).
   - `runtime_label_cache.py` → per-served-KV-page label allocator and the `K_label` write kernel wrapper.
   - `selection_kernel.py` → wrapper around the cherry-picked stage-1 / stage-2 Triton kernels with `selection_mode` parameter (`TOPK`, `TOPP`).
   - `validator.py` → `validate_double_sparsity(server_args)` parallel to `validate_hisparse(server_args)` but independent.
2. Hook into V3.2 attention path. The current path runs `nsa_indexer` to pick a token subset, then FlashMLA on the subset. When `--enable-double-sparsity` is set, a branch at the V3.2 attention `forward()` calls `DoubleSparsitySelector.retrieve_topk(...)` in place of the NSA indexer's selection step, then proceeds into the same FlashMLA call with the DS-selected page table. The NSA quant_k_cache, dequant_k_cache, and Triton / tilelang kernels are reused unchanged.
3. New server args in `python/sglang/srt/server_args.py`:
   ```python
   enable_double_sparsity: bool = False
   double_sparsity_config: Optional[str] = None  # JSON
   ```
   Plus argparse entries with `--enable-double-sparsity` and `--double-sparsity-config`. The mutual-exclusion check with `--enable-hisparse` (AC-1 negative test) lives near the existing `_handle_hisparse` / `_handle_pd_disaggregation` code in `server_args.py`.
4. Calibration artifact format (`safetensors`): top-level tensors `channel_selection[L, H, label_dim]` (int32 indices) and `channel_weights[L, H, label_dim]` (fp32) where `L = num_layers`, `H = num_heads`; metadata block with `model_revision_sha`, `head_dim`, `tp_world_size`, `dtype`, `page_size`, `label_dim`, `created_at`, `schema_version`. Schema reviewed against GLM-5 (per-rank metadata), 128K ISL (no length-dependent fields), and FP4 weights (dtype-agnostic) before merging — see task 6.
5. Runtime label cache: allocated when DS is enabled with shape `[num_layers, max_pages, num_heads, label_dim]`; populated by an MLA-adapted port of PR #25304 commit `567eff67b`'s `K_label` write kernel during prefill, and incrementally during decode for new pages. Cache-hit pages reuse prior labels; a page-stability fixture (M3-B) verifies this.

<comment>
[Codex] The page signature table is keyed by `max_pages`, but the plan never ties entries to the KV page allocator lifecycle. Paged KV blocks are freed, reused, evicted from radix cache, and invalidated by aborted requests; without an owner/epoch check, stale signatures will select pages from a previous request. Make page signatures allocator-owned metadata, overwrite them on every page assignment, and invalidate them on free/eviction/retract.
</comment>

<comment>
[Codex] `[num_layers, max_pages, num_heads, label_dim]` has no memory budget. On V3.2 with long contexts and high concurrency, this can consume enough HBM to reduce KV capacity and invalidate the benchmark before any kernel runs. Pick the dtype, TP shard dimensions, allocation owner, and worst-case bytes in the plan; if the table is too large, compress it or allocate only for resident pages in the existing KV pool.
</comment>

6. Radix cache: DS does not inherit HiSparse's `assert server_args.disable_radix_cache`. The DS validator must instead validate (via M3-B) that the runtime label cache is page-stable under cache hits, then permit radix cache by default for DS. If the fixture fails, the validator refuses radix cache for DS until the failure is resolved.
7. Observability hooks: add Prometheus gauges / counters under `sglang_double_sparsity_*`; thread per-request fields `sparsity_rate`, `selected_pages`, `dense_fallback` through `ScheduleBatch` → `meta_info`.
8. Tests: `test/manual/test_double_sparsity_v32.py` (NIAH 4K / 16K / 64K + MMLU 5-shot on V3.2-FP8); `test/srt/test_double_sparsity_unit.py` (selector kernel, calibration loader, runtime label cache, ABI shape, fault-injection). CI smoke under `test/run_suite.py` using the tiny NSA fixture.

### Future-Work Notes (out of initial scope; for the PR description / docs)
- **HiSparse integration**: a follow-on can wrap `DoubleSparsitySelector` behind the HiSparse `BaseSparseAlgorithm` interface and register it in `_ALGORITHM_REGISTRY["double_sparsity"]` so that PD-disaggregation decode instances can also use DS. This is purely additive over the standalone path.
- **PD-Disagg integration**: separate from HiSparse. The label cache crosses the prefill / decode boundary; either recompute on the decode worker or transport alongside KV. Resolved later.
- **HiCache integration**: per the draft's downstream item, layered after PD-Disagg.

<comment>
[Linus] The future-work plan says "wrap DoubleSparsitySelector behind the HiSparse BaseSparseAlgorithm interface and register it in _ALGORITHM_REGISTRY['double_sparsity']" — i.e. the FINAL state is the architecture the user rejected at CMT-1. You are proposing a two-step migration: build standalone now, integrate with HiSparse later. That is twice the work and twice the test surface. The right question is "why does HiSparse require PD-disaggregation?" and "can that be decoupled?" If yes, do the decoupling first and build DS as a HiSparse algorithm in one move. If no, say DS will NEVER be a HiSparse algorithm and delete this bullet.
</comment>

### Relevant References
- `python/sglang/srt/layers/attention/nsa/` — DeepSeek-V3.2 NSA internals (indexer, quant_k_cache, dequant_k_cache, kernels). DS replaces NSA's selection role on V3.2, not its quant / dequant plumbing.
- `python/sglang/srt/layers/attention/dsv4/` — DeepSeek-V4 DSA internals; out of scope for the initial deliverable but the directory structure is a useful precedent for `double_sparsity/`.
- `python/sglang/srt/layers/attention/flashmla_backend.py`, `flashmla_kv` — the underlying dense MLA kernel that DS feeds with a reduced page table.
- `python/sglang/srt/server_args.py` — where `--enable-double-sparsity` and `--double-sparsity-config` are added; mutual-exclusion check with `enable_hisparse` lives here.
- `python/sglang/srt/arg_groups/hisparse_hook.py` — **inspiration only** for validator pattern; do NOT import or extend.
- `python/sglang/srt/managers/hisparse_coordinator.py` — **inspiration only** for coordinator pattern; do NOT import or invoke.
- `python/sglang/srt/jit_kernel/hisparse.py` (e.g. `load_cache_to_device_buffer_mla`) — Triton helpers that may be cherry-picked / forked into the DS module.
- `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py` — closest existing non-MLA analogue for representation-pool patterns.
- `development/benchmark.sh` — workload definition; basis for `benchmark_baseline.sh`.
- PR #25304 commits (reference / cherry-pick targets only):
  - `a8efc6068` M1 skeleton + calibration schema
  - `567eff67b` M2 K_label storage + write kernels  → runtime label cache
  - `e3570f2fb` M3 selection pipeline (torch ref + Triton)
  - `0b776ca05` v1.1-4 stage-1 block-topk Triton kernel  → selection kernel
  - `1b5e52863` v1.1-5 stage-2 merge Triton kernel  → selection kernel
  - `7fe8002a3` v1.1-6 score-aware union + CUDA-graph capture/replay  → selection kernel
  - `30ba60dae` v2 pivot native sparse-decode (reference only; not used on MLA path)
  - `dc3dcf13f` pluggable selector backends (pattern reference for `selection_mode` ABI)
  - `e8824f86a` M7 calibration script (reference for the standalone calibrate CLI)
  - `3dca4be73` NIAH synthetic prompt generator (reference for calibration default dataset)

## Dependencies and Sequence

### Milestones

1. **Milestone 0 — Decision artifact & branch setup**
   - Phase A: Land this refined plan. Close PR #22992. Mark PR #25304 as reference.
   - Phase B: Cut a feature branch off current `main`: `dev/double-sparsity-standalone`.
   - Phase C: Resolve `## Pending User Decisions` (DEC-1..DEC-7). Encode the resolutions into the test plan.

2. **Milestone 1 — Server args + validator + V3.2 attention-path seam** (targets AC-1, AC-2 backbone)
   - Phase A: Add `enable_double_sparsity` and `double_sparsity_config` to `server_args.py`; argparse entries; mutual-exclusion check with `enable_hisparse`.
   - Phase B: Land `layers/attention/double_sparsity/validator.py::validate_double_sparsity(server_args)` (model check for `is_deepseek_nsa(hf_config)`, dtype check, page-size check, artifact-path required).
   - Phase C: Add the V3.2 attention-path branch that, when DS is enabled, calls `DoubleSparsitySelector.retrieve_topk(...)` instead of the NSA indexer's selection step. Selector returns a deterministic placeholder until M3. Adaptor / page-table threading is exercised by a synthetic-fixture unit test (no live model required).
   - Phase D: Land `development/benchmark_baseline.sh` (DS off only, single-instance, on agreed hardware) so DEC-1 / DEC-2 / DEC-3 conversations have data to anchor on.

3. **Milestone 2 — DS module skeleton + calibration artifact + runtime label cache** (targets AC-1, AC-4)
   - Phase A: Land the `double_sparsity` package (`__init__.py`, `selector.py`, `config.py`) with a placeholder `retrieve_topk`. **Guard**: a server-side check refuses to serve real traffic while the selector is a placeholder (hard error if a placeholder-built binary is asked to handle anything beyond unit / smoke tests).
   - Phase B: Calibration-artifact `safetensors` loader + validator (`calibration.py`). Schema reviewed by task 6 (gap analysis for GLM-5 / 128K / FP4) before merging.
   - Phase C: Runtime label cache allocator (`runtime_label_cache.py`).
   - Phase D: Validator (`validator.py` from M1-B) enforces artifact-path requirement, page-size pairing, and the radix-cache decision from DEC-2.

4. **Milestone 3 — Selection kernels (real DS math) and ABI shape lock-in** (targets AC-2 real, AC-6, AC-11, DEC-2 page-stability)
   - Phase A: Port the M3 / v1.1-4 / v1.1-5 / v1.1-6 selection kernels from PR #25304 to MLA-shaped `K_label` / page layout. Cover stage-1 block-topk, stage-2 merge, score-aware union, capture-safe dispatch. Ship `selection_mode` parameter from this milestone (the ABI shape) with `TOPK` enabled.
   - Phase B: Wire the `K_label` write kernel from commit `567eff67b` to the runtime label cache built in M2-C; populate on prefill, incrementally extend on decode. **Land the DEC-2 page-stability fixture here**: a deterministic prefix is run cold (cache miss) and warm (cache hit) and the test asserts identical `retrieve_topk` output across both runs. Passing this fixture is the precondition for the DS validator to permit radix cache.

<comment>
[Codex] "Incrementally extend on decode for new pages" misses the hot page. During decode, the current KV page changes every token until it fills; if the signature is only written when a new page appears, the freshest tokens are invisible to selection for up to 63 steps at page size 64. Either update the active page signature every decode step or force the active/local window into the selected page table unconditionally.
</comment>

   - Phase C: CUDA-graph piecewise capture for the DS decode path at conc 16 / 32 / 64.

5. **Milestone 4 — Calibration tooling** (targets AC-5)
   - Phase A: Port the calibration script (commits `a8efc6068`, `e8824f86a`) to `calibrate.py`. Default dataset: NIAH-shaped synthetic (`3dca4be73`); `--dataset` accepts external corpora.
   - Phase B: Produce + version an external label artifact for `deepseek-ai/DeepSeek-V3.2` (FP8) **outside** the repo; commit only the documented recipe under `docs/advanced_features/double_sparsity_calibration.md`.

6. **Milestone 5 — Quality & SLO gates** (targets AC-7, AC-8, AC-9, AC-10)
   - Phase A: Update `development/benchmark.sh` to consume the baseline harness from M1-D and emit the two-column report (`DS off` / `DS on`).
   - Phase B: `test/manual/test_double_sparsity_v32.py` for NIAH (4K / 16K / 64K) and MMLU. CI smoke `test/srt/test_double_sparsity_unit.py`.
   - Phase C: Prometheus metrics + per-request `meta_info` fields.

7. **Milestone 6 — Twilight-ABI runtime enablement + ship-gate** (targets AC-11 runtime, AC-12)
   - Phase A: Add `selection_mode=TOPP` unit-test path; gate the end-to-end server path behind a "Twilight" feature flag (default off) so AC-11's negative test passes ("not yet enabled").
   - Phase B: Branch hygiene: rewrite history if needed, write a single PR description, add the pre-commit hook that blocks session-artifact filename patterns, run CI green, request review.

<comment>
[Linus] Hard-coding `is_deepseek_nsa(hf_config)` in the validator (M1-B, task10) is the special-case branch the project's own maxim — `eliminate-special-cases-by-redesigning-data-flow` — tells you to refactor away. DS is sold as a generic selector but pinned to V3.2 via a model-class check. Be honest: either DS is V3.2-specific (then drop GLM-5 from DEC-6 and stop pretending the ABI is portable), or DS works on any MLA + page-table attention model (then the validator checks for that capability, not a model name). The middle ground is YAGNI in validator form.
</comment>

> Dependencies:
> - M1-A unblocks every other task (server args must exist before validator can read them).
> - M1-B unblocks M1-C (validator decides whether DS is engaged).
> - M1-C unblocks M2 (the attention-path seam must accept the selector ABI).
> - M1-D unblocks the DEC-1 / DEC-2 / DEC-3 discussions and AC-7.
> - M2-A unblocks M2-B, C, D (skeleton first).
> - M2-B's schema review (task 6) unblocks M2-D (no validator hard-coding without a frozen schema).
> - M3 unblocks AC-6 and AC-11; ABI shape decided here.
> - M3-B page-stability fixture unblocks the DS validator's radix-cache permission (DEC-2).
> - M4 unblocks AC-8 (no SLO test without a real artifact).
> - M5 unblocks M6 (baseline ships before futures).

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|-------------------------|------------|
| task1 | Confirm the "restart + standalone + cherry-pick" recommendation with the user; close PR #22992; relabel PR #25304 as reference-only | (decision narrative) | analyze | - |
| task2 | Resolve `## Pending User Decisions` DEC-1..DEC-7 (SLO definition + hardware, radix-cache reconciliation, quality thresholds, calibration ownership, V3.2 semantic, deferred-req scope, "Extensions" interpretation) | AC-7, AC-8, AC-9, AC-4 | analyze | task1 |
| task3 | Add `enable_double_sparsity` and `double_sparsity_config` server args (`python/sglang/srt/server_args.py`); argparse entries; mutual-exclusion check with `enable_hisparse`; document the "DS+HiSparse not yet integrated" error message | AC-1 | coding | task2 |
| task4 | Land the V3.2 attention-path branch that calls `DoubleSparsitySelector.retrieve_topk(...)` when DS is enabled; selector returns a deterministic placeholder; synthetic-fixture unit tests assert the FlashMLA page table receives the selected indices | AC-1, AC-2 | coding | task3 |
| task5 | Land `development/benchmark_baseline.sh` (DS off only, single-instance, agreed hardware); refactor `benchmark.sh` to emit the two-column report skeleton (the DS column may be empty until M3 lands) | AC-7 | coding | task4 |
| task6 | Gap analysis for GLM-5, 128K ISL, and FP4-weights compatibility of the **calibration-artifact schema** (not full feature implementation); produce a one-page memo identifying any schema fields required to keep these deferred reqs cheap | AC-4 (schema) | analyze | task2 |
| task7 | Land `double_sparsity` package skeleton (`__init__.py`, `selector.py`, `config.py`) with placeholder `retrieve_topk`; add the **server-side placeholder-guard** that refuses real traffic when the placeholder is built | AC-1 | coding | task4 |
| task8 | Land calibration-artifact `safetensors` loader + validator (`calibration.py`) with schema from task 6 frozen in; add unit tests for happy-path load, all mismatched-field negative tests | AC-4 | coding | task6, task7 |
| task9 | Land runtime label cache allocator (`runtime_label_cache.py`); allocate when DS is enabled; do not yet populate (kernel comes in task 12) | AC-2 | coding | task8 |
| task10 | Land `validator.py::validate_double_sparsity(server_args)`; enforce artifact-path requirement, page-size pairing, model-class check (`is_deepseek_nsa`), backend / dtype pairing, and the DEC-2 radix-cache permission gated by the M3-B fixture | AC-1, AC-3, AC-4 | coding | task9 |
| task11 | Port DS Triton selection kernels (stage-1 block-topk, stage-2 merge, score-aware union) from PR #25304 commits `0b776ca05`, `1b5e52863`, `7fe8002a3` and adapt for MLA `K_label` layout; ship `selection_mode` parameter (`TOPK` runtime, `TOPP` unit-test) | AC-2, AC-6, AC-11 | coding | task10 |
| task12 | Port `K_label` write kernel from PR #25304 commit `567eff67b`; wire to runtime label cache from task 9; populate on prefill, incrementally extend on decode; land the M3-B page-stability fixture (cold vs warm prefix) for DS | AC-2 | coding | task11 |
| task13 | Enable CUDA-graph piecewise capture / replay for the DS decode path at conc 16 / 32 / 64; verify stable max-K buffer ABI from task 11 | AC-6 | coding | task12 |
| task14 | Port calibration script from PR #25304 commits `a8efc6068`, `e8824f86a`, `3dca4be73` to `calibrate.py`; default to NIAH-shaped synthetic data; document the production recipe in `docs/advanced_features/double_sparsity_calibration.md` (no artifact committed) | AC-5 | coding | task8 |
| task15 | Extend `development/benchmark.sh` (already split by task 5) to populate the DS column; add side-by-side `DS off (native NSA)` / `DS on` rows; enforce match on {model_revision_sha, GPU id, TP size, page size, radix-cache setting, concurrency}. Cannot run until a real calibration artifact exists (task 14). | AC-7, AC-8 | coding | task5, task13, task14 |
| task16 | Add `test/manual/test_double_sparsity_v32.py` (NIAH @ 4K / 16K / 64K + MMLU 5-shot) and `test/srt/test_double_sparsity_unit.py` (selector kernel, calibration loader, runtime label cache, ABI shape, fault injection) | AC-9, AC-3, AC-4 | coding | task14 |
| task17 | Add Prometheus metrics under `sglang_double_sparsity_*` and per-request `meta_info` fields (`sparsity_rate`, `selected_pages`, `dense_fallback`); test with fault injection | AC-10 | coding | task13 |
| task18 | Add `selection_mode=TOPP` unit-test path; gate the end-to-end server path behind a "Twilight" feature flag (default off); AC-11 negative test passes | AC-11 | coding | task11 |
| task19 | Independent reasonability audit of the cherry-picked selection kernels vs the Double Sparsity paper and Twilight repo; verify channel-sparsity math matches the published algorithm; document deltas | AC-9 | analyze | task11 |
| task20 | Branch hygiene + ship-gate: rewrite history if needed, write the PR description (must explicitly call out "standalone, no HiSparse"), add the pre-commit hook that blocks `HANDOFF*.md` / `SESSION_REPORT*.md` / pensieve installs, run CI green, prepare reviewer guide. Include a "Future-Work" section in the PR description that scopes the downstream HiSparse + PD-Disagg integration path. | AC-12 | coding | task15, task16, task17, task18 |

<comment>
[Codex] The FP8 path is underspecified (task 12). A `K_label` kernel reading FP8 cache bytes without the exact quantization scales used by `quant_k_cache` is computing scores in the wrong numeric space. The page signature writer must consume the same scale metadata as the NSA dequant path, or it must explicitly dequantize before projection; otherwise the offline channel weights are calibrated against values the runtime never reconstructs.
</comment>

<comment>
[Linus] PR #25304 commits `0b776ca05`, `1b5e52863`, `7fe8002a3` are cited ten times as "cherry-pick targets". They were on a FA3 + Llama path. "Adapt for MLA `K_label` layout" is not cherry-pick — it is rewrite the kernel with the old one open in a side window. Call it what it is: "study PR #25304, reimplement for MLA". The cherry-pick framing makes the work sound smaller than it is and will bite during execution when M3 kernels take three times the budgeted effort.

Separately: tasks 1-20 form a near-linear critical path. Twenty task IDs for one programmer's sequential work is theatre. Either parallelize explicitly (and name who owns what), or drop the table and write an ordered milestone list.
</comment>

## Claude-Codex Deliberation

### Agreements (after Codex Round 1 + 2 + user CMT-1 redirect)
- The restart recommendation stands: PR #22992 is not a viable base; PR #25304 is not upstream-shippable as-is but its selection kernels and calibration scaffolding are valuable.
- CUDA-graph capture safety requires a static / max-bounded selector ABI; Twilight (top-p) requires the ABI now but not the runtime behavior.
- A native-NSA baseline must be reported alongside DS; the SLO claim is meaningless without it.
- Quality gates need NIAH (multi-length) plus a general-knowledge benchmark (MMLU).
- "Calibration artifact" (offline schema / channel-selection) and "runtime label cache" (per-served-page tensor) are distinct concepts and the plan must keep them separate.
- The `double_sparsity.py` vs `double_sparsity/calibrate.py` collision is resolved by using a **package directory**, not a module file. The package lives at `python/sglang/srt/layers/attention/double_sparsity/` per CMT-1 (not under `mem_cache/sparsity/algorithms/`).

### Resolved Disagreements
- **AC-3 BF16 negative test** — Claude originally tested `bfloat16` + `flashmla_kv` as invalid; Codex correctly pointed out that today's HiSparse validator pairs `bfloat16` with `flashmla_sparse`. AC-3's negative test now asserts an explicit `fp8_e4m3` + BF16-only MLA backend mismatch via the new DS validator.
- **AC-10 dense-fallback semantics** — Claude originally allowed non-zero `_dense_fallback_total` in a normal run; Codex argued production DS should have `dense_fallback == 0`. Resolved: healthy runs have zero; the metric is exercised via a fault-injection test.
- **AC-4 vs AC-10 fail-fast vs fallback** — Resolved: AC-4 is fail-fast at startup; AC-10's fault-injection test is the only way to see non-zero fallback.
- **AC-11 top-p scope** — Resolved: ABI now (lower bound), runtime behavior deferred behind a "Twilight" feature flag.
- **Quest-on-MLA smoke test** — Codex argued Quest's representation pool assumes Llama K_buffer shape, not MLA latent; replaced with a synthetic-fixture unit test (no model required). With the standalone refactor this is moot — Quest is no longer in the integration path.
- **DEC-5 path confusion** — Codex caught Claude conflating `nsa/` (V3.2) with `dsv4/` (V4). DEC-5 uses repo terms: DS replaces NSA's **selection** role on V3.2; the NSA quant / dequant / cache components remain authoritative; DSv4 is out of scope.
- **DEC-2 / AC-7 radix-cache scope** — In the prior (HiSparse-based) plan, Codex pushed back that lifting `disable_radix_cache` only for `double_sparsity` would block `deepseek_nsa` from running with radix cache on. With the standalone refactor this becomes simpler: DS does not flow through `hisparse_hook.py`, so the existing `assert disable_radix_cache` doesn't apply. The DS validator carries its own radix-cache permission gated by an M3-B page-stability fixture; the native-NSA baseline (DS off) is the model's default behavior and inherits no HiSparse assertion.
- **AC-8 no-op detector** — Replaced the flaky "DS must not match dense within 1%" assertion with three deterministic invariants: `selected_pages < total_pages`, `dense_fallback_count == 0`, and the FlashMLA metadata-assertion fixture from AC-2 confirms a restricted page table reached the kernel.
- **CMT-1 user redirect (architectural pivot)** — Claude's earlier draft integrated DS as a HiSparse algorithm under `mem_cache/sparsity/`. The user rejected this with "Absolutely not," citing two facts: HiSparse today requires PD-disaggregation and runs decode-instance-only, and users must be able to use DS on a single-instance server. Verified in the codebase via `python/sglang/srt/managers/scheduler.py` (`_build_hisparse_decode_batch`, `set_decode_producer_stream`) and `python/sglang/srt/disaggregation/decode.py`. Resolution: DS is **standalone**, lives under `python/sglang/srt/layers/attention/double_sparsity/`, has its own CLI (`--enable-double-sparsity`), validator (`validate_double_sparsity`), and metric namespace (`sglang_double_sparsity_*`); HiSparse code is reference / borrow-only. The HiSparse + PD-Disagg + HiCache integration is documented as a future-work item per the draft's downstream-requirements ordering.

### Convergence Status
- Final Status: `partially_converged` (after CMT-1 refinement). The architectural pivot is fully applied. DEC-1..DEC-7 remain `PENDING` user decisions, which is the normal end-state for a discussion-mode plan.

<comment>
[Linus] "partially_converged" with seven PENDING decisions — including hardware (DEC-1), the radix-cache mechanism (DEC-2), and the quality thresholds (DEC-3) — means AC-7, AC-8, AC-9 are unmeasurable as written. The plan ships with no falsifiable success condition. DEC-1 and DEC-3 are not architecture decisions, they are number selections. "What hardware" and "what threshold" should not block on a refine-plan loop; they block on one Slack message. Resolve them before this plan is anything but a draft.
</comment>

## Pending User Decisions

- DEC-1: SLO definition + hardware
  - Claude Position: 30 tok/s is **per-request P50 output throughput**. P99 TTFT < 22 s **includes scheduler-queue wait, prefix-cache lookup, and prefill**. Hardware is **H200 8-way TP**, matching the typical DeepSeek-V3.2 FP8 deployment pattern. Both `max-concurrency=64` and `min-concurrency=16` must hit the per-request SLO; the benchmark file lists both.
  - Codex Position: N/A - open question (Codex first-pass and Round 1 both demanded explicit hardware / metric definition)
  - Tradeoff Summary: Per-request 30 tok/s at conc=64 implies aggregate ~1920 tok/s decode; aggregate 30 tok/s would be trivial and not match `bench_serving`'s per-request reporting convention. Hardware assumption can be revised down (H100) but the plan's kernel choices change accordingly.
  - Decision Status: `PENDING`

- DEC-2: Radix cache reconciliation
  - Claude Position: With the standalone refactor, DS no longer inherits HiSparse's `assert disable_radix_cache`. DS keeps radix cache enabled by default. The DS validator gates radix cache on an M3-B page-stability fixture: a deterministic prefix is run cold (cache miss) and warm (cache hit) and the test asserts identical `retrieve_topk` output across both runs. Passing the fixture grants permission; failing it makes the DS validator require `--disable-radix-cache` until resolved. The 55 % prefix-cache-hit workload from `development/benchmark.sh` is preserved.
  - Codex Position: "Fix DEC-2 so the common radix-cache setting applies to all three benchmark columns." With the standalone refactor and a two-column comparison (DS off vs DS on, both single-instance), the apples-to-apples concern is satisfied automatically: both columns run on the same server with the same radix-cache setting.
  - Tradeoff Summary: A standalone DS path means the radix-cache decision is local to DS and not entangled with HiSparse. The page-stability fixture remains the engineering check.
  - Decision Status: `PENDING` (page-stability fixture is the precondition for the validator's default of radix-cache-on)

- DEC-3: Quality threshold deltas vs native NSA
  - Claude Position: NIAH @ {4K, 16K, 64K} score within 5 percentage points of native NSA; MMLU within 1.0 pp of native NSA. Tight enough to detect a corrupted calibration artifact (proved by AC-9 negative test); loose enough that a well-calibrated DS run passes.
  - Codex Position: N/A - open question (Codex first-pass and Round 1 both flagged that the thresholds need user agreement)
  - Tradeoff Summary: Tighter thresholds risk failing for cosmetic reasons; looser thresholds fail to catch silent regressions.
  - Decision Status: `PENDING`

- DEC-4: Calibration ownership and artifact distribution
  - Claude Position: SGLang ships the calibration **script** under `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` and documents the production recipe under `docs/advanced_features/double_sparsity_calibration.md`. The **artifact itself** (the DeepSeek-V3.2 FP8 calibration `safetensors`) is **external** to the repo: it is produced by the deploying team (the user / client) and stored in their model registry / object store, not in the wheel.
  - Codex Position: N/A - open question (Codex Round 1: artifacts are not shipped in repo; M4 must mean external deployment artifact plus documented recipe)
  - Tradeoff Summary: Shipping artifacts in-repo bloats the wheel, pins model revisions, and conflicts with the model-license boundary. Shipping only the script keeps the repo small but requires deployment-side calibration. A tiny NSA-shaped CI fixture is the compromise.
  - Decision Status: `PENDING`

- DEC-5: Semantic relationship of DS to DeepSeek-V3.2 NSA
  - Claude Position: On DeepSeek-V3.2, the DS selector **replaces the NSA indexer's token-selection role** (the part of `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` that decides which pages contribute to attention). The NSA quant / dequant / cache plumbing (`quant_k_cache.py`, `dequant_k_cache.py`, the Triton + tilelang kernels, MTP precompute / verification) **remains authoritative** and is unchanged. DeepSeek-V4's DSA (`python/sglang/srt/layers/attention/dsv4/`) is **out of scope** for the initial deliverable.
  - Codex Position: "DS should not be blindly stacked after DSA. Likely intent is `DS as an alternative selector / label-cache path for DeepSeek-V3.2 sparse attention`." Codex Round 1 also caught Claude conflating `nsa/` with `dsv4/`; the position above incorporates that correction.
  - Tradeoff Summary: Stacking DS on top of NSA's existing selection would double-filter an already sparse set (quality regression). Replacing NSA's selector with DS keeps the same level of sparsity with a different (offline-calibrated) mechanism, enabling A/B comparison and Twilight follow-on. Augmenting NSA internals (DS for channel-level sparsity inside the indexer) is plausible but architecturally larger and out of scope.
  - Decision Status: `PENDING` (Claude and Codex agree on direction; user confirmation required)

- DEC-6: Scope of deferred-requirements coverage in this plan
  - Claude Position: GLM-5, 128K ISL, FP4 weights, **HiSparse integration, PD-Disagg integration, and HiCache integration** are explicitly OUT of the initial scope. The selector ABI (AC-11), the calibration-artifact schema (AC-4), and the validator pattern (modelled on but not invoking `validate_hisparse`) are shaped to admit them without rewrite, and task 6 produces a one-page schema-compatibility memo before M2's loader merge. A "Future-Work" section in the PR description records the downstream integration paths.
  - Codex Position: "Which of these constrain the *initial* design? E.g. if Twilight (top-p) is on the roadmap, the selection kernel should be top-p-shaped from day one." Round 1: "Move a small version of task15 before task5" — applied as task 6 (schema memo) before task 8 (loader merge). User CMT-1: HiSparse and HiCache integration are downstream-only.
  - Tradeoff Summary: Including any deferred requirement in initial scope blows the milestone budget. Excluding them entirely risks a redesign later. The shape-now / behavior-later compromise is encoded in AC-11 and task 6. The HiSparse path is described in the future-work notes but not implemented.
  - Decision Status: `PENDING`

- DEC-7: "Extensions as a general knob for the sglang engine" interpretation
  - Claude Position: Interpreted as "expose Double Sparsity's runtime knobs (`top_k` / `top_p` / `selection_mode` / `calibration_artifact_path`) through `--double-sparsity-config`'s JSON blob, not through new top-level CLI flags per knob." No new plugin system is introduced in this plan.
  - Codex Position: "Vague. What does this mean concretely?" Round 1: "still needs user confirmation."
  - Tradeoff Summary: Treating it as a plugin system would require a separate design doc; treating it as a config blob keeps it within the existing surface and is reversible if the user wants more later.
  - Decision Status: `PENDING` (low-stakes; default ships as Claude position)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. "label cache loader", "selector kernel", "calibration script").
- Follow `.claude/rules/speculative-naming.md` for any identifier that crosses into the speculative-decoding namespace (this plan does not, but kernels reused from PR #25304 should be re-named if they leaked speculative terms).
- Tensors are plural (`selected_indices`, `valid_lengths`, `label_cache`); scalars are singular.
- Count prefix `num_` for counts, `_ct` for accumulators, `_rate` for ratios, no prefix for content arrays.
- Symbol-name boundary: do NOT import from `python/sglang/srt/mem_cache/sparsity/` or `python/sglang/srt/managers/hisparse_coordinator.py` or `python/sglang/srt/arg_groups/hisparse_hook.py` from inside the new `double_sparsity/` package. Borrow patterns by *copying code* with adapted imports, or by extracting a shared helper into a neutral location and importing from both — never reach across the boundary.

<comment>
[Linus] "do NOT import from mem_cache/sparsity/ ... Borrow patterns by *copying code*" is a code-duplication mandate dressed up as a discipline rule. It gives you the worst of both worlds: the patterns drift, no clear owner, no shared bugfix path. If the patterns are useful enough to copy, hoist them into a neutral helper module and import from both packages. If they are not useful enough, write your own without looking at the original. "Copy don't import" is a workaround for a layering problem you have not solved.
</comment>

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
