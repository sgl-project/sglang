# Ask Codex Input

## Question

# Convergence Review — Round 2

You are running the third Codex pass in a Claude–Codex convergence loop. Claude revised the candidate plan in response to your Round 1 critique. Your job now: verify that the revisions are complete and correct, surface any **new** REQUIRED_CHANGES, and decide whether the plan has converged.

## Your Round 1 Critique (Recap)

You raised these REQUIRED_CHANGES in Round 1:
1. Add coordinator-integration task before AC-2/task3 (HiSparseCoordinator vs SparseCoordinator wiring)
2. Rewrite DEC-5 using repo terms (V3.2 = `nsa/`; V4 = `dsv4/`)
3. Fix AC-3 BF16 negative test (BF16 + flashmla_kv is the wrong assertion; today's validator pairs BF16 with flashmla_sparse)
4. Resolve Python packaging conflict (`double_sparsity.py` vs `double_sparsity/calibrate.py`)
5. Split "calibration artifact" from "runtime label cache"
6. Make DEC-2 a hard blocker for AC-7/AC-8
7. Fix `algorithm=none` references
8. Clarify AC-8 metric semantics (hardware, TP/EP, per-request vs aggregate)
9. Fix AC-9 negative test ("disabling sparsity doesn't change MMLU" is useless)
10. Move ABI shape work into M3/task6 (before CUDA graph capture)
11. Move a small version of task15 before task5 (deferred-req schema memo)
12. Fix AC-12 allowed paths (must include managers/, scheduler/, metrics/, benchmark)
13. Fix DEC-4 vs M4 (artifact is external, not repo-committed)
14. Adjust AC-5 positive test (tiny NSA fixture for CI, production recipe for hardware)

Plus DISAGREE: AC-3 BF16, AC-10 fallback semantics, AC-4 vs AC-10 fail-fast, AC-11 scope, M1B Quest-on-MLA, DEC-5 paths.

## Repository Facts (Unchanged)

- main = `2a35707`. `python/sglang/srt/mem_cache/sparsity/` houses HiSparse framework.
- `NSABackendAdaptor.adapt_for_attn_metadata` is still a TODO stub.
- `arg_groups/hisparse_hook.py` enforces: model is `is_deepseek_nsa(hf_config)` or `is_deepseek_v4(hf_config)`; `server_args.disable_radix_cache`; backend pairing fp8_e4m3↔flashmla_kv, bfloat16↔flashmla_sparse.
- Two coordinators coexist: `managers/hisparse_coordinator.py` (DSA-aware tiering + PD) and `mem_cache/sparsity/core/sparse_coordinator.py` (generic algorithm driver). Not wired together yet.
- DeepSeek-V3.2 sparse path is `layers/attention/nsa/`; DeepSeek-V4 sparse path is `layers/attention/dsv4/`.
- PR #25304 commit hashes preserved (a8efc6068, 567eff67b, e3570f2fb, 0b776ca05, 1b5e52863, 7fe8002a3, 30ba60dae, dc3dcf13f, e8824f86a, 3dca4be73).
- Benchmark: ISL≈4096, OSL=512, conc 16/32/64, ~55% prefix-cache hit; current script only exercises conc=64.

## Candidate Plan v2 (full text after applying Round 1 changes)

```
# Plan: Deliver Double Sparsity for DeepSeek-V3.2 (FP8) via the HiSparse Framework

## Goal Description

Deliver a production-quality Double Sparsity (DS) implementation in SGLang that meets the immediate client SLO on DeepSeek-V3.2 (FP8), is forward-compatible with deferred client requests (GLM-5, 128K ISL, FP4 weights), unlocks downstream work (Twilight top-p selection, "Extensions" engine knob, PD-Disagg + HiSparse integration), and ships in a shape that is upstream-reviewable in `sgl-project/sglang`. The plan also resolves the open question "resume vs restart" by recommending a path and stating the explicit cost of the alternatives.

### Resume-vs-Restart Recommendation

**Restart from a fresh branch off current `main`. Integrate Double Sparsity as a new algorithm under the existing HiSparse framework (`python/sglang/srt/mem_cache/sparsity/algorithms/`). Cherry-pick the proven selection / label-cache / Triton kernel work from PR #25304 only where it directly supports the DeepSeek-V3.2 MLA + FP8 path. Close PR #22992. Keep PR #25304 open as a reference link until the new branch reaches kernel parity.**

Rationale:
- PR #22992 (`dev/double-sparsity-reintro`, +1873 / 12 files): restores the legacy Llama-only, page=1, Triton-attention DS backend. The PR body itself documents a 3–12% throughput regression on H100 and states "Performance optimization is planned for follow-up work." It has no MLA path, no FP8 path, no page=64 support, and predates the HiSparse framework. **Not a viable base.**
- PR #25304 (`dev/double-sparsity-v2`, +22552 / 90 files): contains valuable selection kernels and calibration scaffolding (M1 skeleton; M2 K_label storage + write kernels; M3 selection pipeline; M4 FA3 adaptor; v1.1 stage-1/stage-2 Triton block-topk + score-aware union + CUDA-graph capture; v2 native sparse-decode kernels) and reaches a "BOTH GATES PASS" point at conc=32 / 128K / tb=8192. **But** the backbone is FA3 (Llama-style dense attention), not MLA / DeepSeek-V3.2; it uses a custom coordinator instead of `SparseCoordinator`; the PR carries `HANDOFF_NATIVE.md`, `SESSION_REPORT_*.md`, pensieve installs and bench harnesses; it has no PR description and no CI run. **Not upstream-shippable as-is** but is a valuable kernel source.
- Restart on `main` lets the work inherit the existing HiSparse plumbing (`SparseConfig`, `_ALGORITHM_REGISTRY`, `BaseSparseAlgorithm`, `SparseCoordinator`, `HiSparseCoordinator`, PD-disagg hooks, `--enable-hisparse`/`--hisparse-config` CLI). The current `NSABackendAdaptor.adapt_for_attn_metadata` is a TODO stub — completing it is the central enabling lift, and is reusable by both `quest` and future `double_sparsity`.

### Two Coordinators, One Plan

A repo precondition the rest of the plan depends on: today `python/sglang/srt/managers/hisparse_coordinator.py` (`HiSparseCoordinator`, DSA-aware host/device tiering + PD-disagg) and `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py` (`SparseCoordinator`, generic algorithm driver) coexist but are **not wired into a single execution path**. DS execution must traverse both: tiering / PD-disagg via `HiSparseCoordinator`, algorithm lifecycle (`construct_representations`, `update_representations`, `retrieve_topk`) via `SparseCoordinator`. Task 3 (below) defines this seam before any algorithm code lands.

### Two Different "Labels"

Two terms must not be confused; the plan separates them throughout:
- **Calibration artifact** (offline, file on disk, produced by a calibration script): channel-importance schema for the model. Per-layer / per-head selection of which channels matter (typical `label_dim` 16–32), plus model-revision metadata. Static for a given model revision; ships separately from the SGLang wheel.
- **Runtime label cache** (online, GPU tensor, populated during serving): per-served-KV-page label tensor produced by projecting K-cache pages through the channels named in the calibration artifact. Allocated by `initialize_representation_pool`, written on prefill (and incrementally on decode) by the existing-style `K_label` write kernel cherry-picked from PR #25304 commit `567eff67b`. Affected by prefix-cache hits (already-cached pages can reuse labels).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Double Sparsity is exposed only via `--enable-hisparse --hisparse-config '{"algorithm":"double_sparsity", ...}'`. No standalone `--enable-double-sparsity` CLI is added; no parallel sparse coordinator is introduced.
  - Positive Tests:
    - Starting an SGLang server with `--enable-hisparse --hisparse-config '{"algorithm":"double_sparsity","top_k":2048,"page_size":64,"calibration_artifact_path":"<...>"}'` on DeepSeek-V3.2 (FP8) initialises a `DoubleSparsityAlgorithm` and routes selections through `SparseCoordinator`.
    - `_ALGORITHM_REGISTRY["double_sparsity"]` constructs `DoubleSparsityAlgorithm` from a `SparseConfig`.
  - Negative Tests:
    - `--enable-double-sparsity` is not present in `--help`; argparse rejects it as unknown.
    - Server fails to start if `algorithm` is unset while `--enable-hisparse` is set, with a message naming available algorithms (`quest`, `deepseek_nsa`, `double_sparsity`).

- AC-2: `DoubleSparsityAlgorithm` implements `BaseSparseAlgorithm` end-to-end on the MLA path. The previously-stubbed `NSABackendAdaptor.adapt_for_attn_metadata` is completed and routes `selected_indices` / `valid_lengths` from the algorithm to the FlashMLA backend; metadata reaches the kernel in shape.
  - Positive Tests:
    - Unit test calls `DoubleSparsityAlgorithm.retrieve_topk(queries, layer_id, req_pool_indices, sparse_mask)` and receives `(selected_indices, valid_lengths)` with shapes `[bs, max_top_k]` / `[bs]` as documented by the contract.
    - Integration test asserts that the adapted attention metadata reaching the FlashMLA backend contains only the page IDs returned by `retrieve_topk` (no shadow dense path); covered by a metadata-assertion fixture, not by mutating production code.
    - Single-decode-step golden test on DeepSeek-V3.2 (FP8) produces attention output identical bit-for-bit to a reference computation restricted to the same selected pages.
  - Negative Tests:
    - Calling the algorithm on a request with `sparse_mask=False` returns the documented "unselected (dense)" sentinel and the integration path takes the dense MLA branch.
    - Submitting a request before the runtime label cache is populated (forced via fault-injection switch) fails the request with a documented error rather than producing garbage output.

- AC-3: Page size 64 works on DeepSeek-V3.2 (FP8) and at least one alternate page size is exercised in tests. The implementation does not hard-code page=64. Backend / dtype pairing follows the existing HiSparse rule: `fp8_e4m3` ↔ `flashmla_kv`, `bfloat16` ↔ `flashmla_sparse`.
  - Positive Tests:
    - End-to-end smoke run with `kv_cache_dtype=fp8_e4m3`, `page_size=64`, `nsa_*_backend=flashmla_kv` succeeds for one warm prefill + one decode batch.
    - End-to-end smoke run with `kv_cache_dtype=bfloat16`, `page_size=64`, `nsa_*_backend=flashmla_sparse` succeeds for one warm prefill + one decode batch.
    - Unit tests for the selection kernel run with `page_size in {32, 64}` and produce identical logical top-K rankings on a deterministic fixture (modulo page granularity).
  - Negative Tests:
    - Configuring `kv_cache_dtype=fp8_e4m3` with `nsa_decode_backend=flashmla_sparse` (mismatched pair) fails at startup with the existing validator's backend/dtype-mismatch error.
    - Configuring an unsupported page size (e.g. `page_size=7`) for `double_sparsity` fails at startup with an explicit page-size validation error, not silently.

- AC-4: A persisted **calibration artifact** is required to enable `double_sparsity`. The artifact is validated against the loaded model and configuration before serving begins. Missing or mismatched artifacts cause fail-fast at startup; there is no silent dense fallback in production.
  - Positive Tests:
    - Loading a valid artifact (matching `model_revision_sha`, `head_dim`, `tp_world_size`, `dtype`, `page_size`, `label_dim`) succeeds and `DoubleSparsityAlgorithm._calibration_loaded` is true on every TP rank.
    - The loader emits a startup log line listing artifact `model_revision_sha`, `head_dim`, `tp_world_size`, `dtype`, `page_size`, `label_dim`, `created_at`.
  - Negative Tests:
    - Missing artifact path or unreadable file fails the server before the engine starts, with a non-zero exit code and a "calibration artifact required" message.
    - Artifact whose `model_revision_sha`, `head_dim`, `dtype`, `tp_world_size`, or `page_size` mismatches the running configuration fails the server with a message naming each mismatched field.

- AC-5: A calibration script produces a calibration artifact for DeepSeek-V3.2 (FP8) and is invocable as a standalone command. CI uses a tiny NSA-shaped fixture; the production recipe runs on the agreed hardware (DEC-1) and is documented but not committed.
  - Positive Tests:
    - `python -m sglang.srt.mem_cache.sparsity.algorithms.double_sparsity.calibrate --model <tiny-NSA-fixture> --dtype fp8_e4m3 --tp 1 --output /tmp/labels.safetensors` runs to completion in CI under a minute and writes a non-empty artifact with the documented schema.
    - The CI artifact loads successfully via AC-4's loader against the same fixture.
    - The production recipe is documented in `docs/advanced_features/double_sparsity_calibration.md` with the exact CLI invocation, expected dataset, agreed hardware, and expected wall-clock.
  - Negative Tests:
    - Running calibration with `--model` pointing to an unsupported architecture fails with a clear "DoubleSparsity calibration is only supported for ..." message.
    - Running calibration with `--tp 8` on a 1-GPU box fails before allocation with a config error, not at first NCCL call.

- AC-6: CUDA-graph piecewise capture/replay works for the decode path with `double_sparsity` at the target concurrencies (16, 32, 64). The selection ABI uses static / max-bounded output buffers so capture is safe.
  - Positive Tests:
    - Decode-path piecewise CUDA graph captures at conc=64 without `CUDA error: launch failed` and replays for at least 100 steps on a fixed batch.
    - The selector's output buffers are allocated with static shape `[bs, max_top_k]` for `selected_indices` (padded with `-1`) and `[bs]` for `valid_lengths`; the same buffers are reused across capture and replay.
  - Negative Tests:
    - Setting `max_top_k` smaller than `top_k` fails at startup, not at capture.
    - Removing CUDA-graph capture (per-step eager) does not regress correctness — golden output is unchanged.

- AC-7: A dense baseline (HiSparse disabled) and a native-NSA baseline (`--enable-hisparse --hisparse-config '{"algorithm":"deepseek_nsa", ...}'`) are recorded on the same hardware, same model revision, same workload, same radix-cache setting, and same concurrency as the DS run. **Blocks if DEC-2 is not resolved** because the radix-cache setting must be common to all three columns.
  - Positive Tests:
    - `development/benchmark.sh` plus a sibling `development/benchmark_baseline.sh` produce a side-by-side report with three columns: `hisparse_off (dense)`, `hisparse_on / algorithm=deepseek_nsa`, `hisparse_on / algorithm=double_sparsity`. Per-column rows: per-request output tok/s P50 / P99, TTFT P50 / P99, TPOT P50 / P99, and goodput-under-SLO.
  - Negative Tests:
    - The report fails to publish if any of {model_revision_sha, GPU id, TP size, page size, radix-cache setting, concurrency} differs between baseline and DS rows.

- AC-8: The DS run meets or beats the immediate SLO under the clarified throughput definition (see DEC-1): **per-request output throughput P50 ≥ 30 tok/s** and **P99 TTFT ≤ 22 s, including scheduler-queue wait**, at `max-concurrency=64` (also at `min-concurrency=16`) on the workload defined in `development/benchmark.sh` (ISL≈4096, OSL=512, ~55 % prefix-cache hit on the agreed hardware (DEC-1)).
  - Positive Tests:
    - `bench_serving` over `gsp_isl4096_osl512_c64.jsonl` reports per-request output tok/s P50 ≥ 30 and P99 TTFT ≤ 22 s.
    - The same benchmark at conc=16 reports per-request output tok/s P50 ≥ 30 and P99 TTFT ≤ 22 s.
    - The reported `dense_fallback_count` per request is zero (AC-10 metric); if non-zero, the SLO claim is invalid.
  - Negative Tests:
    - A `hisparse_off` (dense) run on the same hardware and workload is published in the same report; the DS column must not match the dense column within a 1 % tolerance, because that would indicate the algorithm is no-op'ing.

- AC-9: Quality gates pass against the native NSA baseline. Agreed thresholds (DEC-3): NIAH retrieval @ 4K/16K/64K within 5 percentage points of native-NSA score; MMLU within 1.0 percentage point of native-NSA score.
  - Positive Tests:
    - `test/manual/test_double_sparsity_v32.py` runs NIAH at 4K/16K/64K and reports DS scores within 5 pp of native NSA on each length.
    - MMLU 5-shot on DeepSeek-V3.2 (FP8) is within 1.0 pp of native NSA.
  - Negative Tests:
    - A run with a deliberately corrupted calibration artifact (random-permuted channel selection) makes NIAH @ 64K drop more than 20 pp below native-NSA baseline, confirming the test is sensitive to artifact quality.
    - A run with an empty / zero runtime label cache (fault-injected) makes NIAH @ 16K drop more than 30 pp, confirming the test is sensitive to the runtime label cache.

- AC-10: Observability surfaces are exposed per step and aggregated per request. Healthy DS runs report `dense_fallback_count == 0` and `calibration_artifact_valid == 1` on every TP rank. Per-request `meta_info` carries `sparsity_rate` (selected / total page count), `selected_pages` (count), and `dense_fallback` (0/1). Prometheus exposes `sglang_hisparse_double_sparsity_*` gauges and counters.
  - Positive Tests:
    - A scrape of `/metrics` after a healthy 64-concurrency run shows `sglang_hisparse_double_sparsity_calibration_artifact_valid = 1`, `..._dense_fallback_total = 0`, non-zero `..._selected_pages_sum` and `..._selected_pages_count`.
    - Per-request `meta_info` carries `sparsity_rate`, `selected_pages`, `dense_fallback` and they aggregate to the Prometheus values modulo sampling.
  - Negative Tests:
    - A fault-injected run with an unwritable runtime label cache (forced by test flag) increments `..._dense_fallback_total` and the test asserts this — confirming the metric is wired, without requiring a fallback in production.
    - With `--disable-metrics` the server still serves and produces correct outputs; the per-request `meta_info` fields are unaffected.

- AC-11: The selection ABI accepts both **fixed top-k** (initial scope) and a future **bounded top-p** (Twilight, deferred). The ABI is shape-locked from Milestone 3; top-p **behavior** is deferred behind a `selection_mode` parameter that defaults to `TOPK`. `SparseConfig.sparse_extra_config` carries algorithm-specific knobs (`min_top_k`, `max_top_k`, `top_p`).
  - Positive Tests:
    - Triton selection kernel signature accepts a `selection_mode` argument (`TOPK`, `TOPP`); the `TOPK` path is exercised end-to-end; the `TOPP` path passes a unit test on synthetic scores producing `valid_lengths` clipped to `[min_top_k, max_top_k]`.
    - `selected_indices.shape == (bs, max_top_k)` invariant holds under both modes; no shape change between calls.
  - Negative Tests:
    - Requesting `selection_mode=TOPP` end-to-end on the server (not the unit test) fails with a documented "not yet enabled" error until Twilight ships.
    - Requesting an unknown `selection_mode` fails with an enum error.
    - Requesting `top_p` with `max_top_k > device_buffer_size` fails at startup, not at first decode.

- AC-12: The shipping branch is upstream-shaped. No `HANDOFF*.md`, `SESSION_REPORT*.md`, pensieve installs, ad-hoc bench harnesses, or workspace notes are committed. The branch may modify files under these paths, but each commit is reviewable.
  - Positive Tests:
    - `git log --name-only origin/main..HEAD` shows only files under: `python/sglang/srt/{mem_cache/sparsity, layers/attention, arg_groups, managers, model_executor, metrics}`, `sgl-kernel/`, `test/`, `docs/`, `development/benchmark*.sh`.
    - `git diff --stat origin/main..HEAD` is bounded by an agreed budget (DEC-4) and individual commits are < ~1500 lines each except for the Triton kernel commit (allowed to be a single logical unit).
  - Negative Tests:
    - A pre-commit hook (added in `task14`) blocks any session-artifact filename pattern (`HANDOFF*.md`, `SESSION_REPORT*.md`, `*.HANDOFF.md`, top-level pensieve dirs added to git).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The implementation delivers a full HiSparse-native `DoubleSparsityAlgorithm` with completed `NSABackendAdaptor.adapt_for_attn_metadata`, an FP8-aware calibration-artifact loader and validator, a runtime label-cache populator using a cherry-picked `K_label` write kernel from PR #25304, a calibration script that defaults to NIAH-shaped synthetic data plus an opt-in dataset hook, Triton selection kernels cherry-picked and re-MLA-ified from PR #25304 (stage-1 block-topk + stage-2 merge, score-aware union, capture-safe), top-k baseline with a top-p-ready selector ABI (`selection_mode=TOPK` shipped, `TOPP` unit-tested), CUDA-graph piecewise capture, per-request and Prometheus observability under `sglang_hisparse_double_sparsity_*`, an MLA + FP8 quality regression suite (NIAH at 4K/16K/64K plus MMLU), a benchmark harness that publishes side-by-side `hisparse_off` / `algorithm=deepseek_nsa` / `algorithm=double_sparsity` results, and a smoke test that DS+PD-Disagg coexists.

### Lower Bound (Minimum Acceptable Scope)
The implementation registers `double_sparsity` in `_ALGORITHM_REGISTRY`, completes the `NSABackendAdaptor` MLA path enough that DS runs end-to-end on DeepSeek-V3.2 (FP8) at `page_size=64`, ships a calibration script that produces a valid artifact for one fixed NSA-shaped fixture (CI) and documents the production recipe, validates the artifact at load time, populates the runtime label cache on prefill, meets AC-8 SLO at `max-concurrency=64` and at `min-concurrency=16`, passes AC-9 quality gates with the agreed deltas, exposes the AC-10 metrics minimally (selected pages, dense-fallback count, calibration-artifact validity), and produces an upstream-shippable branch (AC-12). Top-p selection runtime behavior, GLM-5, 128K ISL, and FP4 weights are deferred but the selector ABI (AC-11) and the calibration-artifact schema (AC-4 fields) are shaped to admit them without rewrite.

### Allowed Choices
- Can use: `BaseSparseAlgorithmImpl`, `SparseCoordinator`, `HiSparseCoordinator`, `NSABackendAdaptor`, FlashMLA backends (`flashmla_kv` for FP8, `flashmla_sparse` for BF16), Triton selection kernels (cherry-picked from PR #25304 and adapted), CUDA-graph piecewise capture, `--hisparse-config` JSON for all DS-specific knobs, `safetensors` for the calibration-artifact format.
- Cannot use: a separate `--enable-double-sparsity` CLI, a parallel sparse coordinator, `page_size=1` only, FA3-only kernels on the V3.2 path, the legacy `double_sparsity_backend.py` re-imported from PR #22992, `HANDOFF*.md` / `SESSION_REPORT*.md` files committed to the upstream branch.

> **Note on Deterministic Designs**: The HiSparse integration path is the fixed architectural choice; the boundaries above reflect that. Kernel selection (cherry-pick from PR #25304 vs reimplement) is the main choice point within the bounds; both options are acceptable.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

1. Add `python/sglang/srt/mem_cache/sparsity/algorithms/double_sparsity/` as a **package** (avoiding the file-vs-directory collision Codex flagged):
   - `__init__.py` → re-exports `DoubleSparsityAlgorithm`.
   - `algorithm.py` → `class DoubleSparsityAlgorithm(BaseSparseAlgorithmImpl)`.
   - `calibration.py` → calibration-artifact `safetensors` loader + validator + schema definition.
   - `calibrate.py` → standalone CLI entry point for offline calibration (`python -m sglang.srt.mem_cache.sparsity.algorithms.double_sparsity.calibrate`).
   - `runtime_label_cache.py` → per-served-KV-page label allocator and the `K_label` write kernel wrapper.
   - `selection_kernel.py` → wrapper around the cherry-picked stage-1 / stage-2 Triton kernels with `selection_mode` parameter (`TOPK`, `TOPP`).
2. Register in `python/sglang/srt/mem_cache/sparsity/factory.py`:
   ```python
   _ALGORITHM_REGISTRY = {
       "quest": ...,
       "deepseek_nsa": ...,
       "double_sparsity": lambda c, d, **kw: DoubleSparsityAlgorithm(c, d, **kw),
   }
   ```
   In `_create_backend_adaptor`, route `DoubleSparsityAlgorithm` to `NSABackendAdaptor` for MLA models and `FlashAttentionAdaptor` for any future Llama-style enablement.
3. Complete `NSABackendAdaptor.adapt_for_attn_metadata` in `python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py`. Translate `selected_indices` (logical page IDs, padded with `-1`) into the FlashMLA backend's `page_table` / `block_table` slots; mask invalid entries; preserve metadata invariants required by `flashmla_kv` (FP8) and `flashmla_sparse` (BF16).
4. Define the coordinator seam (per "Two Coordinators, One Plan" above): `HiSparseCoordinator` retains host/device tiering and PD-disagg ownership; `SparseCoordinator` drives the algorithm lifecycle (`initialize_representation_pool`, `construct_representations`, `update_representations`, `retrieve_topk`). The seam is a single explicit handoff per layer per forward batch; document it in a module-level docstring rather than a comment heap.
5. Extend `arg_groups/hisparse_hook.py`:
   - Add `"double_sparsity"` to the allowed `algorithm` set.
   - For `double_sparsity`, validate that `calibration_artifact_path` is present, that the artifact loads, and that `page_size` matches the artifact's recorded value.
   - Reconcile the existing `--disable-radix-cache` assertion with the client's 55 % prefix-cache hit workload. See DEC-2.
6. Calibration artifact format (`safetensors`): top-level tensors `channel_selection[L, H, label_dim]` (int32 indices) and `channel_weights[L, H, label_dim]` (fp32) where `L = num_layers`, `H = num_heads`; metadata block with `model_revision_sha`, `head_dim`, `tp_world_size`, `dtype`, `page_size`, `label_dim`, `created_at`, `schema_version`. The schema must be designed for forward compatibility with GLM-5 (per-rank metadata), 128K ISL (no length-dependent fields), and FP4 weights (dtype-agnostic) — see task 6 below.
7. Runtime label cache: allocated by `initialize_representation_pool` with shape `[num_layers, max_pages, num_heads, label_dim]`; populated by an MLA-adapted port of PR #25304 commit `567eff67b`'s `K_label` write kernel during prefill, and incrementally during decode for new pages. Cache-hit pages reuse prior labels (this is the DEC-2 mechanism that lets the radix-cache assertion lift safely for DS).
8. Observability hooks: add Prometheus gauges/counters under `sglang_hisparse_double_sparsity_*`; thread per-request fields `sparsity_rate`, `selected_pages`, `dense_fallback` through `ScheduleBatch` → `meta_info`.
9. Tests: `test/manual/test_double_sparsity_v32.py` (NIAH 4K/16K/64K + MMLU 5-shot on V3.2-FP8); `test/srt/test_double_sparsity_unit.py` (selector kernel, calibration loader, runtime label cache, ABI shape, fault-injection). CI smoke under `test/run_suite.py` using the tiny NSA fixture.

### Relevant References
- `python/sglang/srt/mem_cache/sparsity/factory.py` — `_ALGORITHM_REGISTRY`, `_create_backend_adaptor`, `_parse_sparse_config`.
- `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py` — `BaseSparseAlgorithm` ABC and `BaseSparseAlgorithmImpl`.
- `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py` — closest non-MLA analogue.
- `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py` — MLA-path algorithm peer.
- `python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py` — `NSABackendAdaptor.adapt_for_attn_metadata` TODO stub.
- `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py` — generic algorithm driver.
- `python/sglang/srt/arg_groups/hisparse_hook.py` — CLI validation and backend defaults.
- `python/sglang/srt/managers/hisparse_coordinator.py` — DSA-aware host/device tiering, PD-disagg, page-table accounting.
- `python/sglang/srt/layers/attention/nsa/` — DeepSeek-V3.2 NSA internals (indexer, quant_k_cache, dequant_k_cache, kernels). DS replaces NSA's selection role on V3.2, not its quant/dequant plumbing.
- `python/sglang/srt/layers/attention/dsv4/` — DeepSeek-V4 DSA internals (compressor, indexer); out of scope for the initial deliverable.
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

1. **Milestone 0 — Decision artifact $(CANDIDATE_PLAN_V2) branch setup**
   - Phase A: Land this plan. Close PR #22992. Mark PR #25304 as reference.
   - Phase B: Cut a feature branch off current `main`: `dev/double-sparsity-hisparse`.
   - Phase C: Resolve `## Pending User Decisions` (DEC-1..DEC-7). Encode the resolutions into the test plan.

2. **Milestone 1 — Coordinator seam + NSA adaptor (enabling)** (targets AC-2 backbone, DEC blockers)
   - Phase A: Document and implement the `HiSparseCoordinator` ↔ `SparseCoordinator` handoff (Hint #4 above). Add a single explicit seam, not a comment heap.
   - Phase B: Complete `NSABackendAdaptor.adapt_for_attn_metadata` for `flashmla_kv` (FP8) and `flashmla_sparse` (BF16). Add unit tests with synthetic `selected_indices` / `valid_lengths` that assert the resulting metadata equals an expected fixture (no live model required).
   - Phase C: Land `development/benchmark_baseline.sh` early (independent of DS kernels) so DEC-1 / DEC-2 / DEC-3 conversations have data to anchor on.

3. **Milestone 2 — `DoubleSparsityAlgorithm` skeleton + calibration artifact + runtime label cache** (targets AC-1, AC-4, plus the deferred-req future-proofing)
   - Phase A: Land the `double_sparsity` package (`__init__.py`, `algorithm.py`) implementing `BaseSparseAlgorithmImpl` with `retrieve_topk` returning a deterministic placeholder top-k so plumbing can be exercised. **Guard**: a server-side check refuses to serve real traffic while the selector is a placeholder (a hard error if a placeholder-built binary is asked to handle anything beyond unit/smoke tests).
   - Phase B: Calibration-artifact `safetensors` loader + validator (`calibration.py`). Schema reviewed by task 6 (gap analysis for GLM-5 / 128K / FP4) before merging.
   - Phase C: Runtime label cache allocator (`runtime_label_cache.py`). Wire `initialize_representation_pool`.
   - Phase D: Register in `_ALGORITHM_REGISTRY`. Extend `hisparse_hook.py` with `algorithm="double_sparsity"` validation (including artifact-path requirement).

4. **Milestone 3 — Selection kernels (real DS math) and ABI shape lock-in** (targets AC-2 real, AC-6, AC-11)
   - Phase A: Port the M3 / v1.1-4 / v1.1-5 / v1.1-6 selection kernels from PR #25304 to MLA-shaped `K_label` / page layout. Cover stage-1 block-topk, stage-2 merge, score-aware union, capture-safe dispatch. Ship `selection_mode` parameter from this milestone (the ABI shape) with `TOPK` enabled.
   - Phase B: Wire the `K_label` write kernel from commit `567eff67b` to the runtime label cache built in M2-C; populate on prefill, incrementally extend on decode.
   - Phase C: CUDA-graph piecewise capture for the DS decode path at conc 16 / 32 / 64.

5. **Milestone 4 — Calibration tooling** (targets AC-5)
   - Phase A: Port the calibration script (commits `a8efc6068`, `e8824f86a`) to `calibrate.py`. Default dataset: NIAH-shaped synthetic (`3dca4be73`); `--dataset` accepts external corpora.
   - Phase B: Produce + version an external label artifact for `deepseek-ai/DeepSeek-V3.2` (FP8) **outside** the repo; commit only the documented recipe under `docs/advanced_features/double_sparsity_calibration.md`.

6. **Milestone 5 — Quality $(CANDIDATE_PLAN_V2) SLO gates** (targets AC-7, AC-8, AC-9, AC-10)
   - Phase A: Update `development/benchmark.sh` to consume the baseline harness from M1-C and emit the three-column report. Run dense + native-NSA + DS columns.
   - Phase B: `test/manual/test_double_sparsity_v32.py` for NIAH (4K / 16K / 64K) and MMLU. CI smoke `test/srt/test_double_sparsity_unit.py`.
   - Phase C: Prometheus metrics + per-request `meta_info` fields.

7. **Milestone 6 — Twilight-ABI runtime enablement + ship-gate** (targets AC-11 runtime, AC-12)
   - Phase A: Add `selection_mode=TOPP` unit-test path; gate the end-to-end server path behind a "Twilight" feature flag (default off) so AC-11's negative test passes ("not yet enabled").
   - Phase B: Branch hygiene: rewrite history if needed, write a single PR description, add the pre-commit hook that blocks session-artifact filename patterns, run CI green, request review.

> Dependencies:
> - M1-A unblocks every algorithm task (coordinator seam must exist before the algorithm is wired).
> - M1-B unblocks M2 (the adaptor must accept the algorithm's output shape).
> - M1-C unblocks the DEC-1 / DEC-2 / DEC-3 discussions and AC-7.
> - M2-A unblocks M2-B, C, D (skeleton first).
> - M2-B's schema review (task 6 below) unblocks M2-D (no registry without a frozen schema).
> - M3 unblocks AC-6 and AC-11; ABI shape decided here.
> - M4 unblocks AC-8 (no SLO test without a real artifact).
> - M5 unblocks M6 (baseline ships before futures).

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|-------------------------|------------|
| task1 | Confirm the "restart + HiSparse + cherry-pick" recommendation with the user; close PR #22992; relabel PR #25304 as reference-only | (decision narrative) | analyze | - |
| task2 | Resolve `## Pending User Decisions` DEC-1..DEC-7 (SLO definition + hardware, radix-cache reconciliation, quality thresholds, calibration ownership, V3.2 semantic, deferred-req scope, "Extensions" interpretation) | AC-7, AC-8, AC-9, AC-4 | analyze | task1 |
| task3 | Document and implement the `HiSparseCoordinator` ↔ `SparseCoordinator` seam; specify the per-layer handoff; add an integration test that proves the seam without an algorithm under it | AC-1, AC-2 | coding | task2 |
| task4 | Complete `NSABackendAdaptor.adapt_for_attn_metadata` for `flashmla_kv` (FP8) and `flashmla_sparse` (BF16); unit tests assert the adapted metadata equals an expected fixture for synthetic `selected_indices` / `valid_lengths` | AC-2, AC-3 | coding | task3 |
| task5 | Land `development/benchmark_baseline.sh`; refactor `benchmark.sh` to emit the three-column report skeleton (the DS column may be empty until M3 lands) | AC-7 | coding | task4 |
| task6 | Gap analysis for GLM-5, 128K ISL, and FP4-weights compatibility of the **calibration-artifact schema** (not full feature implementation); produce a one-page memo identifying any schema fields required to keep these deferred reqs cheap | AC-4 (schema) | analyze | task2 |
| task7 | Land `double_sparsity` package skeleton (`__init__.py`, `algorithm.py`) with placeholder `retrieve_topk`; add the **server-side placeholder-guard** that refuses real traffic when the placeholder is built | AC-1 | coding | task3 |
| task8 | Land calibration-artifact `safetensors` loader + validator (`calibration.py`) with schema from task 6 frozen in; add unit tests for happy-path load, all mismatched-field negative tests | AC-4 | coding | task6, task7 |
| task9 | Land runtime label cache allocator (`runtime_label_cache.py`); wire `initialize_representation_pool`; do not yet populate (kernel comes in task 11) | AC-2 | coding | task8 |
| task10 | Register `double_sparsity` in `_ALGORITHM_REGISTRY`; extend `arg_groups/hisparse_hook.py` with `algorithm="double_sparsity"` validation (artifact-path requirement, page-size pairing, DEC-2 radix-cache resolution applied) | AC-1, AC-3, AC-4 | coding | task9 |
| task11 | Port DS Triton selection kernels (stage-1 block-topk, stage-2 merge, score-aware union) from PR #25304 commits `0b776ca05`, `1b5e52863`, `7fe8002a3` and adapt for MLA `K_label` layout; ship `selection_mode` parameter (`TOPK` runtime, `TOPP` unit-test) | AC-2, AC-6, AC-11 | coding | task10 |
| task12 | Port `K_label` write kernel from PR #25304 commit `567eff67b`; wire to runtime label cache from task 9; populate on prefill, incrementally extend on decode | AC-2 | coding | task11 |
| task13 | Enable CUDA-graph piecewise capture/replay for the DS decode path at conc 16/32/64; verify stable max-K buffer ABI from task 11 | AC-6 | coding | task12 |
| task14 | Port calibration script from PR #25304 commits `a8efc6068`, `e8824f86a`, `3dca4be73` to `calibrate.py`; default to NIAH-shaped synthetic data; document the production recipe in `docs/advanced_features/double_sparsity_calibration.md` (no artifact committed) | AC-5 | coding | task8 |
| task15 | Extend `development/benchmark.sh` (already split by task 5) to populate the DS column; add side-by-side `hisparse_off` / `algorithm=deepseek_nsa` / `algorithm=double_sparsity` rows; enforce match on {model_revision_sha, GPU id, TP size, page size, radix-cache setting, concurrency} | AC-7, AC-8 | coding | task13 |
| task16 | Add `test/manual/test_double_sparsity_v32.py` (NIAH @ 4K/16K/64K + MMLU 5-shot) and `test/srt/test_double_sparsity_unit.py` (selector kernel, calibration loader, runtime label cache, ABI shape, fault injection) | AC-9, AC-3, AC-4 | coding | task14 |
| task17 | Add Prometheus metrics under `sglang_hisparse_double_sparsity_*` and per-request `meta_info` fields (`sparsity_rate`, `selected_pages`, `dense_fallback`); test with fault injection | AC-10 | coding | task13 |
| task18 | Add `selection_mode=TOPP` unit-test path; gate the end-to-end server path behind a "Twilight" feature flag (default off); AC-11 negative test passes | AC-11 | coding | task11 |
| task19 | Independent reasonability audit of the cherry-picked selection kernels vs the Double Sparsity paper and Twilight repo; verify channel-sparsity math matches the published algorithm; document deltas | AC-9 | analyze | task11 |
| task20 | Branch hygiene + ship-gate: rewrite history if needed, write the PR description, add the pre-commit hook that blocks `HANDOFF*.md` / `SESSION_REPORT*.md` / pensieve installs, run CI green, prepare reviewer guide | AC-12 | coding | task15, task16, task17, task18 |

## Claude-Codex Deliberation

### Agreements (after Codex Round 1)
- The HiSparse framework is the correct integration point; do not introduce a parallel `--enable-double-sparsity` CLI.
- PR #22992 is not a viable base; PR #25304 is not upstream-shippable as-is but its selection kernels and calibration scaffolding are valuable.
- The `NSABackendAdaptor.adapt_for_attn_metadata` TODO stub plus the `HiSparseCoordinator` ↔ `SparseCoordinator` seam are the central enabling lifts.
- CUDA-graph capture safety requires a static / max-bounded selector ABI; Twilight (top-p) requires the ABI now but not the runtime behavior.
- A dense + native-NSA baseline must be reported alongside DS; the SLO claim is meaningless without it.
- Quality gates need NIAH (multi-length) plus a general-knowledge benchmark (MMLU).
- "Calibration artifact" (offline schema/channel-selection) and "runtime label cache" (per-served-page tensor) are distinct concepts and the plan must keep them separate.
- The `algorithms/double_sparsity.py` vs `algorithms/double_sparsity/calibrate.py` collision is resolved by using a **package directory**, not a module file.

### Resolved Disagreements
- **AC-3 BF16 negative test** — Claude originally tested `bfloat16` + `flashmla_kv` as invalid; Codex correctly pointed out that today's validator pairs `bfloat16` with `flashmla_sparse`. AC-3's negative test now asserts an explicit `fp8_e4m3` + `flashmla_sparse` mismatch error.
- **AC-10 dense-fallback semantics** — Claude originally allowed non-zero `_dense_fallback_total` in a normal run; Codex argued production DS should have `dense_fallback == 0`. Resolved: healthy runs have zero; the metric is exercised via a fault-injection test.
- **AC-4 vs AC-10 fail-fast vs fallback** — Resolved: AC-4 is fail-fast at startup; AC-10's fault-injection test is the only way to see non-zero fallback.
- **AC-11 top-p scope** — Resolved: ABI now (lower bound), runtime behavior deferred behind a "Twilight" feature flag.
- **Milestone 1 Phase B "Quest on MLA as smoke test"** — Codex argued Quest's representation pool assumes Llama K_buffer shape, not MLA latent; replaced with a synthetic-fixture unit test (no model required).
- **DEC-5 path confusion** — Codex caught Claude conflating `nsa/` (V3.2) with `dsv4/` (V4). DEC-5 is rewritten to use repo terms: DS replaces NSA's **selection** role on V3.2; the NSA quant/dequant/cache components remain authoritative; DSv4 is out of scope for the initial deliverable.

### Convergence Status
- Final Status: TBD (this is the v2 plan after Round 1; will be set after Round 2 / max rounds)

## Pending User Decisions

- DEC-1: SLO definition + hardware
  - Claude Position: 30 tok/s is **per-request P50 output throughput**. P99 TTFT < 22 s **includes scheduler-queue wait, prefix-cache lookup, and prefill**. Hardware is **H200 8-way TP**, matching the typical DeepSeek-V3.2 FP8 deployment pattern. Both `max-concurrency=64` and `min-concurrency=16` must hit the per-request SLO; the benchmark file lists both.
  - Codex Position: N/A - open question (Codex first-pass and Round 1 both demanded explicit hardware/metric definition)
  - Tradeoff Summary: Per-request 30 tok/s at conc=64 implies aggregate ~1920 tok/s decode; aggregate 30 tok/s would be trivial and not match `bench_serving`'s per-request reporting convention. Hardware assumption can be revised down (H100) but the plan's kernel choices change accordingly.
  - Decision Status: `PENDING`

- DEC-2: Radix cache reconciliation
  - Claude Position: The workload requires ~55 % prefix-cache hit, so radix cache must remain enabled. Lift the current `assert disable_radix_cache` for `algorithm="double_sparsity"` once the runtime label cache is verified to be page-stable under cache hits (M3-B test fixture). If verification fails, redefine the benchmark with `--disable-radix-cache` and renegotiate the cache-hit assumption with the client.
  - Codex Position: N/A - open question (Codex Round 1: "Make DEC-2 a hard blocker for AC-7/AC-8")
  - Tradeoff Summary: Existing `hisparse_hook.py` enforces `--disable-radix-cache`. Without resolution, AC-7 and AC-8 cannot be met as currently written. Lifting the assertion only for DS algorithms preserves safety for the rest of HiSparse.
  - Decision Status: `PENDING` (BLOCKS AC-7, AC-8)

- DEC-3: Quality threshold deltas vs native NSA
  - Claude Position: NIAH @ {4K, 16K, 64K} score within 5 percentage points of native NSA; MMLU within 1.0 pp of native NSA. Tight enough to detect a corrupted calibration artifact (proved by AC-9 negative test); loose enough that a well-calibrated DS run passes.
  - Codex Position: N/A - open question (Codex first-pass and Round 1 both flagged that the thresholds need user agreement)
  - Tradeoff Summary: Tighter thresholds risk failing for cosmetic reasons; looser thresholds fail to catch silent regressions.
  - Decision Status: `PENDING`

- DEC-4: Calibration ownership and artifact distribution
  - Claude Position: SGLang ships the calibration **script** under `python/sglang/srt/mem_cache/sparsity/algorithms/double_sparsity/calibrate.py` and documents the production recipe under `docs/advanced_features/double_sparsity_calibration.md`. The **artifact itself** (the DeepSeek-V3.2 FP8 calibration `safetensors`) is **external** to the repo: it is produced by the deploying team (the user / client) and stored in their model registry / object store, not in the wheel.
  - Codex Position: N/A - open question (Codex Round 1: "REQUIRED_CHANGE 13: artifacts are not shipped in repo; M4 must mean external deployment artifact plus documented recipe")
  - Tradeoff Summary: Shipping artifacts in-repo bloats the wheel, pins model revisions, and conflicts with the model-license boundary. Shipping only the script keeps the repo small but requires deployment-side calibration. A tiny NSA-shaped CI fixture is the compromise.
  - Decision Status: `PENDING`

- DEC-5: Semantic relationship of DS to DeepSeek-V3.2 NSA
  - Claude Position: On DeepSeek-V3.2, the `double_sparsity` algorithm **replaces the NSA indexer's token-selection role** (the part of `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` that decides which pages contribute to attention). The NSA quant/dequant/cache plumbing (`quant_k_cache.py`, `dequant_k_cache.py`, the Triton + tilelang kernels, MTP precompute/verification) **remains authoritative** and is unchanged. DeepSeek-V4's DSA (`python/sglang/srt/layers/attention/dsv4/`) is **out of scope** for the initial deliverable.
  - Codex Position: "DS should not be blindly stacked after DSA. Likely intent is `DS as an alternative selector/label-cache path for DeepSeek-V3.2 sparse attention`." Codex Round 1 also caught Claude conflating `nsa/` with `dsv4/`; the position above incorporates that correction.
  - Tradeoff Summary: Stacking DS on top of NSA's existing selection would double-filter an already sparse set (quality regression). Replacing NSA's selector with DS keeps the same level of sparsity with a different (offline-calibrated) mechanism, enabling A/B comparison and Twilight follow-on. Augmenting NSA internals (DS for channel-level sparsity inside the indexer) is plausible but architecturally larger and out of scope.
  - Decision Status: `PENDING` (Claude and Codex agree on direction; user confirmation required)

- DEC-6: Scope of deferred-requirements coverage in this plan
  - Claude Position: GLM-5, 128K ISL, and FP4 weights are explicitly OUT of the initial scope. The selector ABI (AC-11) and the calibration-artifact schema (AC-4) are shaped to admit them without rewrite, and task 6 produces a one-page schema-compatibility memo before M2's registry merge.
  - Codex Position: "Which of these constrain the *initial* design? E.g. if Twilight (top-p) is on the roadmap, the selection kernel should be top-p-shaped from day one." Codex Round 1: "Move a small version of task15 before task5" — applied as task 6 (schema memo) before task 8 (loader merge).
  - Tradeoff Summary: Including any deferred requirement in initial scope blows the milestone budget. Excluding them entirely risks a redesign later. The shape-now / behavior-later compromise is encoded in AC-11 and task 6.
  - Decision Status: `PENDING`

- DEC-7: "Extensions as a general knob for the sglang engine" interpretation
  - Claude Position: Interpreted as "expose Double Sparsity's runtime knobs (top_k / top_p / selection mode / artifact path) through `--hisparse-config`'s `sparse_extra_config` blob, not through new top-level CLI flags." No new plugin system is introduced in this plan.
  - Codex Position: "Vague. What does this mean concretely?" Codex Round 1: "still needs user confirmation."
  - Tradeoff Summary: Treating it as a plugin system would require a separate design doc; treating it as `sparse_extra_config` keeps it within the existing surface and is reversible if the user wants more later.
  - Decision Status: `PENDING` (low-stakes; default ships as Claude position)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. "label cache loader", "selector kernel", "calibration script").
- Follow `.claude/rules/speculative-naming.md` for any identifier that crosses into the speculative-decoding namespace (this plan does not, but kernels reused from PR #25304 should be re-named if they leaked speculative terms).
- Tensors are plural (`selected_indices`, `valid_lengths`, `label_cache`); scalars are singular.
- Count prefix `num_` for counts, `_ct` for accumulators, `_rate` for ratios, no prefix for content arrays.
```

## Output Format

```
AGREE:
- <Round 1 items confirmed resolved>
- <new agreements>

DISAGREE:
- <new disagreements or unresolved Round 1 items>

REQUIRED_CHANGES:
- <only NEW must-fix items; "none" is an acceptable answer if v2 fully addresses Round 1>

OPTIONAL_IMPROVEMENTS:
- <non-blocking suggestions>

UNRESOLVED:
- <DECs that genuinely require human decision and cannot be settled by the plan itself>

CONVERGENCE_VERDICT:
- CONVERGED   (no REQUIRED_CHANGES; remaining items are user decisions or optional)
  or
- NEEDS_ROUND_3   (list the single most important reason)
```

Rules:
- If you say `CONVERGED`, the convergence loop terminates and the plan goes to user review with the UNRESOLVED items as Pending User Decisions.
- If you say `NEEDS_ROUND_3`, you must give one concrete blocker, not a vague concern.
- Do not re-raise issues from Round 1 that were addressed in v2.
- Do not invent new scope.
- Prefer "CONVERGED" if the plan is shippable as a planning artifact, even if some DECs are PENDING — those exist for the user-decision phase, not the convergence loop.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-05-19_09-16-57
- Tool: codex
