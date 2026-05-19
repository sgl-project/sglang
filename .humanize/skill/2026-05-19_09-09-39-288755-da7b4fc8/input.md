# Ask Codex Input

## Question

# Convergence Review — Round 1

You are running the second Codex pass in a Claude–Codex convergence loop. Your job: review the candidate plan below for **reasonability**, flag anything unsupported by the repository's actual state, identify required changes, and surface unresolved opposite opinions. **Do not just rephrase.** If the plan is sound, say so explicitly and list optional improvements.

## Prior Codex Findings (Round 0 / first-pass)

The first-pass Codex flagged the following (summary, not your job to re-output):
- DS-vs-DSA semantic ambiguity on V3.2 (Codex preferred "independent algorithm replacing DSA's token selector")
- PR #22992 not viable (Llama-only, page=1, no FP8, regression)
- PR #25304 not upstream-shippable but kernels valuable
- HiSparse framework is correct integration point
- `NSABackendAdaptor.adapt_for_attn_metadata` is a TODO stub — central enabling lift
- SLO underspecified; needs hardware, aggregate-vs-per-request, TP/EP
- Radix cache validator currently requires `--disable-radix-cache`, conflicts with ~55% cache-hit workload
- Quality gates: NIAH + MMLU
- Selector ABI must be top-p compatible from day one (Twilight)
- 12 candidate ACs proposed

Claude has incorporated these into the candidate plan below.

## Candidate Plan v1 (full text)

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

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Double Sparsity is exposed only via `--enable-hisparse --hisparse-config '{"algorithm":"double_sparsity", ...}'`. No standalone `--enable-double-sparsity` CLI is added; no parallel sparse framework is introduced.
  - Positive Tests:
    - Starting an SGLang server with `--enable-hisparse --hisparse-config '{"algorithm":"double_sparsity","top_k":2048,"page_size":64,"label_path":"<...>"}'` on DeepSeek-V3.2 (FP8) initialises a `DoubleSparsityAlgorithm` and routes selections through `SparseCoordinator`.
    - `_ALGORITHM_REGISTRY["double_sparsity"]` constructs `DoubleSparsityAlgorithm` from a `SparseConfig`.
  - Negative Tests:
    - `--enable-double-sparsity` does not exist as a CLI flag (argparse rejects it).
    - Server fails to start if `algorithm` is unset while `--enable-hisparse` is set, with a message naming the available algorithms (`quest`, `deepseek_nsa`, `double_sparsity`).

- AC-2: `DoubleSparsityAlgorithm` implements the `BaseSparseAlgorithm` ABC end-to-end on the MLA path. The previously-stubbed `NSABackendAdaptor.adapt_for_attn_metadata` is completed and routes `selected_indices` / `valid_lengths` from the algorithm to the FlashMLA backend.
  - Positive Tests:
    - Unit test calls `DoubleSparsityAlgorithm.retrieve_topk(queries, layer_id, req_pool_indices, sparse_mask)` and gets back `(selected_indices, valid_lengths)` with shapes `[bs, max_selected]` / `[bs]` matching the documented contract.
    - Integration test runs a single decode step on DeepSeek-V3.2 (FP8) and confirms the attention output uses only the selected pages (top-K verification via stubbed/mocked attention kernel).
  - Negative Tests:
    - Test that mutating `NSABackendAdaptor.adapt_for_attn_metadata` to `pass` (stub state) causes the integration test to fail (regression guard).
    - Test that calling the algorithm on a request with `sparse_mask=False` returns an unselected (dense) result.

- AC-3: Page size 64 works on DeepSeek-V3.2 (FP8) and at least one alternate page size is exercised in tests. The implementation does not hard-code page=64.
  - Positive Tests:
    - End-to-end smoke run with `page_size=64` and `flashmla_kv` backend succeeds for one warm prefill + one decode batch.
    - Unit tests for the selection kernel run with `page_size in {32, 64}` and produce identical logical top-K rankings on a deterministic fixture (modulo page granularity).
  - Negative Tests:
    - Configuring `page_size=64` with `bfloat16` KV and `flashmla_kv` backend fails at startup with the existing HiSparse validator error.
    - Configuring an unsupported page size (e.g. `page_size=7`) fails at startup with an explicit page-size validation error, not silently.

- AC-4: A persisted **label-cache artifact** is required to enable `double_sparsity`. The artifact is validated against the loaded model and configuration before serving begins.
  - Positive Tests:
    - Loading a valid artifact (matching model revision, TP size, head dim, dtype, page size) succeeds and `DoubleSparsityAlgorithm._labels_loaded` is true on every TP rank.
    - The label-cache loader exposes the artifact's model-revision SHA, dtype, head_dim, TP world size, and creation timestamp in server-startup logs.
  - Negative Tests:
    - Missing artifact path or unreadable file fails the server before the engine starts, with an exit code and a "calibration artifact required" message.
    - Artifact whose model-revision SHA, head dim, dtype, or TP size mismatches the running configuration fails the server with a message naming the mismatched fields.

- AC-5: A calibration script produces a label-cache artifact for DeepSeek-V3.2 (FP8) and is invocable as a standalone command.
  - Positive Tests:
    - `python -m sglang.srt.mem_cache.sparsity.algorithms.double_sparsity.calibrate --model deepseek-ai/DeepSeek-V3.2 --dtype fp8_e4m3 --tp 1 --output /tmp/labels.safetensors [--dataset <path|"niah">]` runs to completion on a small fixture and writes a non-empty artifact.
    - The artifact loads successfully via AC-4's loader.
  - Negative Tests:
    - Running calibration with `--model` pointing to an unsupported architecture fails with a clear "DoubleSparsity calibration is only supported for ..." message.
    - Running calibration with `--tp 8` on a 1-GPU box fails before allocation with a config error, not at first NCCL call.

- AC-6: CUDA-graph capture/replay works for the decode path with `double_sparsity` at the target concurrencies (16, 32, 64). The selection ABI uses static / max-bounded output buffers so capture is safe.
  - Positive Tests:
    - Decode-path piecewise CUDA graph captures at conc=64 without `CUDA error: launch failed` and replays for at least 100 steps on a fixed batch.
    - The selector's output `selected_indices` is allocated with a static `[bs, max_top_k]` shape padded with `-1`; `valid_lengths` is `[bs]` int32. Same ABI accepts top-p selection in a future change without a shape change.
  - Negative Tests:
    - Removing CUDA-graph capture in tests does not regress correctness (golden output unchanged).
    - Setting `max_top_k` smaller than `top_k` fails at startup, not at capture.

- AC-7: A dense / native-NSA baseline benchmark is recorded on the same hardware, same model revision, same workload, and same `radix_cache` setting as the DS run.
  - Positive Tests:
    - `development/benchmark.sh` plus a sibling `development/benchmark_baseline.sh` produce a side-by-side report with: tokens/s (per-request + aggregate), TTFT P50 / P99, TPOT P50 / P99, and goodput-under-SLO. Baseline rows include `algorithm=none (dense)` and `algorithm=deepseek_nsa`.
  - Negative Tests:
    - The report fails to publish if any of {model revision, GPU id, TP size, page size, radix setting, concurrency} differs between baseline and DS rows.

- AC-8: The DS run meets or beats the immediate SLO under the clarified throughput definition: **30 tok/s per request output throughput and P99 TTFT < 22 s** at `max-concurrency=64` (and ≥30 tok/s at `min-concurrency=16`) on the workload defined in `development/benchmark.sh` (ISL≈4096, OSL=512, ~55 % prefix-cache hit).
  - Positive Tests:
    - `bench_serving` over `gsp_isl4096_osl512_c64.jsonl` reports per-request output tok/s P50 ≥ 30 and P99 TTFT ≤ 22 s on the agreed hardware.
    - The same benchmark at conc=16 reports per-request output tok/s P50 ≥ 30 and P99 TTFT ≤ 22 s.
  - Negative Tests:
    - A run with the calibration artifact loaded but `algorithm=none` (dense) on the same hardware and workload is reported alongside; the implementation must not pass the gate by accidentally falling back to dense. (Detected via the observability counters in AC-10.)

- AC-9: Quality gates pass against the native NSA baseline. The agreed thresholds (Decision DEC-3) are: NIAH retrieval @ 4K/16K/64K ≥ native-NSA score - 5 percentage points, and MMLU ≥ native-NSA score - 1.0 percentage point.
  - Positive Tests:
    - `test/manual/test_double_sparsity_v32.py` runs NIAH at 4K/16K/64K and reports DS scores within the agreed deltas of the native NSA baseline.
    - MMLU 5-shot on DeepSeek-V3.2 (FP8) is within 1.0 pp of the native NSA score.
  - Negative Tests:
    - Loading a deliberately corrupted label cache (random-permuted labels) makes NIAH @ 64K drop more than 20 pp below baseline (proving the test is sensitive to label-cache quality).
    - Disabling sparsity at runtime (algorithm=none) does not change MMLU.

- AC-10: Observability metrics are exposed for each step and aggregated per request: selected-pages count, effective sparsity, label-cache hit/miss / validity, top-k vs top-p mode, dense-fallback count, CUDA-graph status. Metrics are reachable via the existing Prometheus / `meta_info` channel.
  - Positive Tests:
    - A scrape of `/metrics` after a 64-concurrency run shows non-zero `sglang_hisparse_double_sparsity_selected_pages_sum`, `sglang_hisparse_double_sparsity_dense_fallback_total`, and `sglang_hisparse_double_sparsity_label_cache_valid` gauges.
    - Per-request `meta_info` carries `accept_rate`-shaped sparsity stats (e.g. `sparsity_rate`) following Rule 4 naming conventions.
  - Negative Tests:
    - With `--disable-metrics` the server still runs; with metrics enabled, missing labels produce a non-zero `_dense_fallback_total` and `_label_cache_valid=0`.

- AC-11: The selection ABI accepts both **fixed top-k** and a future **bounded top-p** policy without changing attention metadata layout. The `SparseConfig.sparse_extra_config` carries algorithm-specific knobs (e.g. `min_top_k`, `max_top_k`, `top_p`); the `selected_indices` shape is `[bs, max_top_k]`.
  - Positive Tests:
    - Switching `sparse_extra_config={"selection":"top_p","top_p":0.9,"min_top_k":512,"max_top_k":4096}` runs end-to-end and produces variable per-request `valid_lengths` clipped to `[min_top_k, max_top_k]`.
    - Triton selection kernel exposes a `selection_mode` argument (`TOPK`, `TOPP`) without re-allocating buffers.
  - Negative Tests:
    - Requesting `top_p` with `max_top_k > device_buffer_size` fails at startup, not at first decode.
    - Requesting an unknown `selection` mode fails with a documented enum error.

- AC-12: The shipping branch is upstream-shaped. No `HANDOFF*.md`, `SESSION_REPORT*.md`, pensieve installs, ad-hoc bench harnesses, or workspace notes are committed. Each commit is reviewable; tests, calibration, kernels, framework wiring, and docs are separated.
  - Positive Tests:
    - `git log --name-only origin/main..HEAD` shows only files under `python/sglang/srt/{mem_cache/sparsity,layers/attention,arg_groups}`, `sgl-kernel/`, `test/`, `docs/`, and `development/benchmark*.sh`.
    - `git diff --stat origin/main..HEAD` is bounded by an agreed budget (Decision DEC-4) and individual commits are < 1500 lines each except for the kernel commit (allowed to be larger if a single logical unit).
  - Negative Tests:
    - CI lint (pre-commit) and the existing `validate-gen-plan-io.sh`-style validators block any session artifact filename pattern (`HANDOFF*.md`, `SESSION_REPORT*.md`).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The implementation delivers a full HiSparse-native `DoubleSparsityAlgorithm` with completed `NSABackendAdaptor.adapt_for_attn_metadata`, an FP8-aware label-cache loader and validator, a calibration script with NIAH-shaped synthetic data plus an opt-in dataset hook, Triton selection kernels cherry-picked and re-MLA-ified from PR #25304 (stage-1 block-topk + stage-2 merge, score-aware union, capture-safe), top-k baseline with a top-p-ready selector ABI, CUDA-graph piecewise capture, per-request and Prometheus observability, an MLA + FP8 quality regression suite (NIAH at 4 K/16 K/64 K plus MMLU), and a benchmark harness that publishes side-by-side dense / native-NSA / DS results. PD-disagg compatibility is verified at smoke level (labels stay on the decode worker; KV pages move via existing HiSparse PD path).

### Lower Bound (Minimum Acceptable Scope)
The implementation registers `double_sparsity` in `_ALGORITHM_REGISTRY`, completes the `NSABackendAdaptor` MLA path enough that DS runs end-to-end on DeepSeek-V3.2 (FP8) at `page_size=64`, ships a calibration script that produces a valid artifact for one fixed dataset (NIAH synthetic), validates the artifact at load time, meets AC-8 SLO at `max-concurrency=64` and at `min-concurrency=16`, passes AC-9 quality gates with the agreed deltas, exposes the AC-10 metrics minimally (selected pages, dense fallback, label-cache validity), and produces an upstream-shippable branch (AC-12). Top-p selection, GLM-5, 128K ISL, and FP4 weights are deferred but the selector ABI (AC-11) is shaped to admit them without rewrite.

### Allowed Choices
- Can use: `BaseSparseAlgorithmImpl`, `SparseCoordinator`, `HiSparseCoordinator`, `NSABackendAdaptor`, FlashMLA backends (`flashmla_kv` for FP8, `flashmla_sparse` for BF16), Triton selection kernels (cherry-picked from PR #25304 and adapted), CUDA-graph piecewise capture, `--hisparse-config` JSON for all DS-specific knobs, `safetensors` for the label-cache artifact format.
- Cannot use: a separate `--enable-double-sparsity` CLI, a parallel sparse coordinator, `page_size=1` only, FA3-only kernels, the legacy `double_sparsity_backend.py` re-imported from PR #22992, `HANDOFF*.md` / `SESSION_REPORT*.md` files committed to the upstream branch.

> **Note on Deterministic Designs**: The HiSparse integration path is the fixed architectural choice; the boundaries above reflect that. Kernel selection (cherry-pick from PR #25304 vs reimplement) is the main choice point within the bounds; both options are acceptable.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

1. Add `python/sglang/srt/mem_cache/sparsity/algorithms/double_sparsity.py` implementing `BaseSparseAlgorithmImpl`. Key methods:
   - `initialize_representation_pool`: allocate the **label cache** (per-layer, per-head, per-page channel-importance signatures). Shape ≈ `[num_layers, num_pages, num_heads, label_dim]` with `label_dim ≪ head_dim` (typical 16 or 32). Memory budget verified at startup.
   - `construct_representations`: load offline-calibrated labels from the artifact (no on-prefill computation in the baseline). On-prefill recomputation is an optional fallback.
   - `update_representations`: incremental write for new pages emitted during decode (use existing `K_label` write kernel from PR #25304 M2 / commit `567eff67b`).
   - `retrieve_topk`: compute `query · label_cache` approximate scores, run two-stage block-topk Triton kernel (cherry-pick from PR #25304 v1.1-4 / v1.1-5), pad to `[bs, max_top_k]`.

2. Register in `python/sglang/srt/mem_cache/sparsity/factory.py`:
   ```python
   _ALGORITHM_REGISTRY = {
       "quest": ...,
       "deepseek_nsa": ...,
       "double_sparsity": lambda c, d, **kw: DoubleSparsityAlgorithm(c, d, **kw),
   }
   ```
   In `_create_backend_adaptor`, route `DoubleSparsityAlgorithm` to `NSABackendAdaptor` for MLA models and `FlashAttentionAdaptor` for future Llama-style models.

3. Complete `NSABackendAdaptor.adapt_for_attn_metadata` in `python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py`. Translate `selected_indices` (logical page IDs) into the FlashMLA backend's `page_table` / `block_table` slots, masking invalid (`-1`) entries.

4. Extend `arg_groups/hisparse_hook.py`:
   - Allow `algorithm in {"deepseek_nsa", "quest", "double_sparsity"}` (already string-based).
   - For `double_sparsity`, validate `kv_cache_dtype` matches the artifact, `page_size` matches the artifact, and `label_path` is present.
   - Reconcile the existing `--disable-radix-cache` assertion with the client's 55 % prefix-cache hit workload. Either: (a) lift the assertion for `double_sparsity` if labels are page-stable across cache hits, or (b) document and accept that prefix cache must be disabled for DS and re-define the benchmark accordingly. **This is DEC-2.**

5. Calibration: add `python/sglang/srt/mem_cache/sparsity/algorithms/double_sparsity/calibrate.py` (or a module under the algorithm). Reuse the calibration logic from PR #25304 M1/M7. Output format: `safetensors` with metadata block: `model_revision_sha`, `dtype`, `head_dim`, `tp_world_size`, `page_size`, `label_dim`, `created_at`.

6. Observability hooks: add metrics under `sglang_hisparse_double_sparsity_*` namespace; thread per-request stats through `meta_info` via `ScheduleBatch`.

7. Tests: `test/manual/test_double_sparsity_v32.py` (NIAH + MMLU on V3.2-FP8); `test/srt/test_double_sparsity_unit.py` (selector kernel, label-cache loader, ABI shape); CI smoke under `test/run_suite.py` if the V3.2 fixture is small enough.

### Relevant References
- `python/sglang/srt/mem_cache/sparsity/factory.py` — `_ALGORITHM_REGISTRY`, `_create_backend_adaptor`, `_parse_sparse_config`.
- `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py` — `BaseSparseAlgorithm` ABC and `BaseSparseAlgorithmImpl`.
- `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py` — closest existing analogue, peer algorithm.
- `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py` — MLA-path algorithm peer.
- `python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py` — `NSABackendAdaptor.adapt_for_attn_metadata` TODO stub to complete.
- `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py` — coordinator that drives the algorithm; understand its forward-pass lifecycle.
- `python/sglang/srt/arg_groups/hisparse_hook.py` — CLI validation and backend defaults.
- `python/sglang/srt/managers/hisparse_coordinator.py` — host/device memory tiering and PD-disagg interaction.
- `python/sglang/srt/layers/attention/nsa/` — DeepSeek V3.2 native sparse attention internals (do not duplicate).
- `development/benchmark.sh` — workload definition; will become baseline + DS sweep harness.
- PR #25304 commits (reference only):
  - `a8efc6068` M1 skeleton + calibration schema
  - `567eff67b` M2 K_label storage + write kernels
  - `e3570f2fb` M3 selection pipeline (torch ref + Triton)
  - `0b776ca05` v1.1-4 stage-1 block-topk Triton kernel
  - `1b5e52863` v1.1-5 stage-2 merge Triton kernel
  - `7fe8002a3` v1.1-6 score-aware union + CUDA-graph capture/replay
  - `30ba60dae` v2 pivot native sparse-decode (reference; not used on MLA path)
  - `dc3dcf13f` pluggable selector backends + FlashInfer top_k_page_table (selector backend pattern)

## Dependencies and Sequence

### Milestones

1. **Milestone 0 — Decision artifact $(CANDIDATE_PLAN_V1) branch setup**
   - Phase A: Land this plan (the document). Close PR #22992. Mark PR #25304 as reference.
   - Phase B: Cut a new feature branch off current `main`: `dev/double-sparsity-hisparse`.
   - Phase C: Resolve `## Pending User Decisions` DEC-1..DEC-N. Encode DEC-2 (radix-cache reconciliation) and DEC-3 (quality thresholds) into the test plan.

2. **Milestone 1 — HiSparse framework completion (enabling)** (targets AC-2 backbone)
   - Phase A: Complete `NSABackendAdaptor.adapt_for_attn_metadata` for the FlashMLA backends (`flashmla_kv` FP8 and `flashmla_sparse` BF16). Add adaptor unit tests with synthetic `selected_indices`/`valid_lengths`.
   - Phase B: Wire `quest` algorithm onto MLA path as a smoke test of the adaptor (proves the adaptor without needing DS yet). Mark `quest` on MLA as experimental in docs.

3. **Milestone 2 — `DoubleSparsityAlgorithm` skeleton + label cache** (targets AC-1, AC-4)
   - Phase A: Add `algorithms/double_sparsity.py` implementing `BaseSparseAlgorithmImpl` with `retrieve_topk` returning a deterministic top-k by `len`-prefix (placeholder) so plumbing can be tested.
   - Phase B: Add label-cache `safetensors` loader + validator (model revision, dtype, head dim, TP, page size). Register in `_ALGORITHM_REGISTRY`. Extend `hisparse_hook.py` with `double_sparsity` validation.

4. **Milestone 3 — Selection kernels (real DS math)** (targets AC-2 real + AC-6)
   - Phase A: Port the M3 / v1.1-4 / v1.1-5 / v1.1-6 selection kernels from PR #25304 onto the MLA-shaped `K_label` / page layout. Cover stage-1 block-topk, stage-2 merge, score-aware union, capture-safe dispatch.
   - Phase B: CUDA-graph piecewise capture for the DS decode path at conc 16 / 32 / 64.

5. **Milestone 4 — Calibration tooling** (targets AC-5)
   - Phase A: Port the M1/M7 calibration script (commits `a8efc6068`, `e8824f86a`) to the new module. Default dataset: NIAH-shaped synthetic prompts (commit `3dca4be73`).
   - Phase B: Produce + version a label-cache artifact for `deepseek-ai/DeepSeek-V3.2` (FP8) and document the calibration recipe.

6. **Milestone 5 — Quality $(CANDIDATE_PLAN_V1) SLO gates** (targets AC-7, AC-8, AC-9, AC-10)
   - Phase A: Add `development/benchmark_baseline.sh` and update `development/benchmark.sh` to publish a unified report. Run dense + native-NSA + DS columns.
   - Phase B: Add `test/manual/test_double_sparsity_v32.py` for NIAH (4K / 16K / 64K) and MMLU.
   - Phase C: Add observability metrics under `sglang_hisparse_double_sparsity_*` + per-request `meta_info` fields.

7. **Milestone 6 — ABI futures $(CANDIDATE_PLAN_V1) ship-gate** (targets AC-11, AC-12)
   - Phase A: Extend selection kernel signature with `selection_mode` (`TOPK`, `TOPP`) but ship behind a feature flag (default TOPK).
   - Phase B: Branch hygiene: rewrite history to drop session noise, write a single PR description, run CI green, request review.

> Dependencies: M1 unblocks M2 and M3 (the adaptor is required before any algorithm hits the MLA path). M2 unblocks M3 (skeleton before real math). M3 unblocks M5 (kernels must work for the benchmark). M4 unblocks AC-8 (no SLO test without a real artifact). M5 unblocks M6 (ABI futures only after baseline ships).

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|-------------------------|------------|
| task1 | Confirm the "restart + HiSparse + cherry-pick" recommendation with the user and close/relabel PR #22992 and PR #25304 accordingly | (decision narrative) | analyze | - |
| task2 | Resolve `## Pending User Decisions` (DEC-1..DEC-N): SLO definition, radix-cache reconciliation, quality thresholds, calibration ownership, deferred-req scope | AC-7, AC-8, AC-9 | analyze | task1 |
| task3 | Complete `NSABackendAdaptor.adapt_for_attn_metadata` for `flashmla_kv` (FP8) and `flashmla_sparse` (BF16), with adaptor unit tests using synthetic selections | AC-2 | coding | task2 |
| task4 | Land `algorithms/double_sparsity.py` skeleton implementing `BaseSparseAlgorithmImpl`; register in `_ALGORITHM_REGISTRY`; route via `NSABackendAdaptor` for MLA models | AC-1 | coding | task3 |
| task5 | Implement label-cache `safetensors` loader + validator (model_revision_sha, dtype, head_dim, tp_world_size, page_size, label_dim); add `hisparse_hook.py` validations for `algorithm="double_sparsity"` | AC-4, AC-3 | coding | task4 |
| task6 | Port DS Triton selection kernels (stage-1 block-topk, stage-2 merge, score-aware union) from PR #25304 commits `0b776ca05`, `1b5e52863`, `7fe8002a3` and adapt for MLA `K_label` layout | AC-2, AC-6 | coding | task5 |
| task7 | Enable CUDA-graph piecewise capture/replay for DS decode path at conc 16/32/64; verify stable max-K buffer ABI | AC-6, AC-11 | coding | task6 |
| task8 | Port calibration script from PR #25304 commits `a8efc6068`, `e8824f86a`, `3dca4be73`; default to NIAH-shaped synthetic data; produce a versioned label-cache artifact for DeepSeek-V3.2 (FP8) | AC-5 | coding | task5 |
| task9 | Add `development/benchmark_baseline.sh`; extend `development/benchmark.sh` to emit dense / native-NSA / DS side-by-side report with strict matching of model revision, GPU id, TP, page size, radix, concurrency | AC-7 | coding | task6 |
| task10 | Add `test/manual/test_double_sparsity_v32.py` (NIAH @ 4K / 16K / 64K + MMLU 5-shot) and unit tests for selector kernel, label loader, ABI shape | AC-9, AC-3 | coding | task6, task8 |
| task11 | Add Prometheus metrics under `sglang_hisparse_double_sparsity_*` and per-request `meta_info` sparsity fields | AC-10 | coding | task6 |
| task12 | Extend selector kernel signature with `selection_mode` (`TOPK`, `TOPP`) and validate buffer-stable behavior; ship default TOPK | AC-11 | coding | task7 |
| task13 | Branch hygiene + ship-gate: rewrite history if needed, write PR description, run CI green, prepare reviewer guide | AC-12 | coding | task7, task9, task10, task11 |
| task14 | Independent reasonability audit of the kernels vs the paper / Twilight implementation; verify channel-sparsity math matches the published algorithm | AC-9 | analyze | task6 |
| task15 | Investigate compatibility plan for GLM-5, 128K ISL, and FP4 weights — produce a written gap analysis pointing to the smallest set of changes needed beyond Milestone 6 | (deferred AC) | analyze | task13 |

## Claude-Codex Deliberation

### Agreements (after Codex first-pass)
- The HiSparse framework is the correct integration point; do not introduce a parallel `--enable-double-sparsity` CLI.
- PR #22992 is not a viable base; PR #25304 is not upstream-shippable as-is but its selection kernels and calibration scaffolding are valuable.
- The `NSABackendAdaptor.adapt_for_attn_metadata` TODO stub is the central enabling lift.
- CUDA-graph capture safety requires a static / max-bounded selector ABI; Twilight (top-p) is a near-future requirement and must not force an ABI change.
- A dense + native-NSA baseline must be reported alongside DS; the SLO claim is meaningless without it.
- Quality gates need NIAH (multi-length) plus a general-knowledge benchmark (MMLU). MMLU alone is insufficient.

### Resolved Disagreements
*(none yet — first Codex pass produced findings, not disagreements with Claude. The convergence loop will exercise this.)*

### Convergence Status
- Final Status: TBD (will be set after Phase 5)

## Pending User Decisions

- DEC-1: SLO definition
  - Claude Position: 30 tok/s is **per-request** output throughput at max-concurrency=64. P99 TTFT < 22 s **includes** queueing + scheduler wait. Hardware is **assumed H100/H200 8-way TP** (matching the V3.2 FP8 deployment pattern). Min-concurrency=16 must also hit 30 tok/s per request.
  - Codex Position: N/A - open question
  - Tradeoff Summary: Per-request 30 tok/s at conc=64 implies aggregate ~1920 tok/s decode; aggregate 30 tok/s would be trivial. The benchmark file uses `--max-concurrency` and `bench_serving` reports per-request throughput, supporting the per-request reading. Hardware is undefined in the draft.
  - Decision Status: `PENDING`

- DEC-2: Radix cache reconciliation
  - Claude Position: For DEC-2, the workload requires ~55 % prefix-cache hit, so radix cache must remain enabled. Lift the current `assert disable_radix_cache` for `algorithm in {"double_sparsity"}` once label-cache page-stability under cache hits is verified (Milestone 3 test). If that verification fails, redefine the benchmark with `--disable-radix-cache` and renegotiate the cache-hit assumption with the client.
  - Codex Position: N/A - open question
  - Tradeoff Summary: Existing `hisparse_hook.py` enforces `--disable-radix-cache`. Without resolution, AC-5 (cache+sparsity coexistence) cannot be met. Lifting the assertion only for DS algorithms preserves safety for the rest of HiSparse.
  - Decision Status: `PENDING`

- DEC-3: Quality threshold deltas vs native NSA
  - Claude Position: NIAH @ {4K, 16K, 64K} score ≥ native NSA - 5 percentage points; MMLU ≥ native NSA - 1.0 pp. These are tight enough that a corrupted label cache is detectable, loose enough that a well-calibrated DS run passes.
  - Codex Position: N/A - open question (CANDIDATE_CRITERIA AC-9 in Codex output asks for "agreed thresholds")
  - Tradeoff Summary: Tighter thresholds risk failing for cosmetic reasons; looser thresholds fail to catch silent regressions.
  - Decision Status: `PENDING`

- DEC-4: Calibration ownership
  - Claude Position: SGLang ships the calibration **script** under `python/sglang/srt/mem_cache/sparsity/algorithms/double_sparsity/calibrate.py`; the **artifact** is produced by the deploying team (the user / client) and not shipped in the repo. The script defaults to NIAH-shaped synthetic data; a `--dataset` flag accepts user-provided calibration corpora.
  - Codex Position: N/A - open question
  - Tradeoff Summary: Shipping artifacts in-repo bloats the wheel and pins model revisions; shipping only the script keeps the repo small but requires deployment-side calibration.
  - Decision Status: `PENDING`

- DEC-5: Semantic relationship of DS to DeepSeek-V3.2 NSA / DSA
  - Claude Position: DS is offered as an **independently selectable sparse algorithm** under HiSparse. On DeepSeek-V3.2, `algorithm="double_sparsity"` **replaces** the DSA learned-indexer's token-selection role with the offline-calibrated label-cache approximation. DSA's compressor / dequant_k_cache plumbing remains in use for MLA correctness; only the indexing decision is replaced.
  - Codex Position: "DS should not be blindly stacked after DSA. Likely intent is `DS as an alternative selector/label-cache path for DeepSeek-V3.2 sparse attention`."
  - Tradeoff Summary: Stacking DS on top of DSA would double-filter an already sparse set (likely quality regression). Replacing DSA's indexer with DS keeps the same level of selection with a different mechanism, enabling A/B comparison and Twilight follow-on. Augmenting DSA internals (DS for channel-level sparsity inside NSA's indexer) is plausible but architecturally larger and not in scope.
  - Decision Status: `PENDING` (Claude and Codex agree on direction; user confirmation required)

- DEC-6: Scope of deferred-requirements coverage in this plan
  - Claude Position: GLM-5, 128K ISL, and FP4 weights are explicitly OUT of the initial scope. The selector ABI (AC-11) and the label-cache schema (AC-4) are shaped to admit them without rewrite, but they are tracked as task15 (gap analysis only).
  - Codex Position: "Which of these constrain the *initial* design? E.g. if Twilight (top-p) is on the roadmap, the selection kernel should be top-p-shaped from day one."
  - Tradeoff Summary: Including any deferred requirement in the initial scope blows the milestone budget. Excluding them entirely risks a redesign later. The AC-11 shape-now / behavior-later trade-off is a low-cost middle ground.
  - Decision Status: `PENDING`

- DEC-7: "Extensions as a general knob for the sglang engine" interpretation
  - Claude Position: This is interpreted as "expose Double Sparsity's runtime knobs (top_k / top_p / selection mode / label path) through `--hisparse-config`'s `sparse_extra_config` blob, not through new top-level CLI flags." No new plugin system is introduced.
  - Codex Position: "Vague. What does this mean concretely?"
  - Tradeoff Summary: Treating it as a plugin system would require a separate design doc; treating it as `sparse_extra_config` keeps it within the existing surface.
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

## Repository Facts to Anchor Your Review

- main = `2a35707`; `git grep -r "double_sparsity"` matches only `development/{draft,plan}.md`.
- `python/sglang/srt/mem_cache/sparsity/factory.py` registers only `quest` and `deepseek_nsa`.
- `python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py`: `NSABackendAdaptor.adapt_for_attn_metadata` is `# TODO: Implement NSA backend adaptor logic\npass`.
- `python/sglang/srt/arg_groups/hisparse_hook.py`: `validate_hisparse` asserts `is_deepseek_nsa(hf_config) or is_deepseek_v4(hf_config)` and asserts `server_args.disable_radix_cache`. Backend pairing: `flashmla_kv` for `fp8_e4m3`, `flashmla_sparse` for `bfloat16`.
- `python/sglang/srt/managers/hisparse_coordinator.py`: existing DSA-shaped coordinator already manages MLA host/device tiering and PD-disagg interactions; `HiSparseCoordinator` is not yet wired to `SparseCoordinator` (they coexist).
- PR #22992 commits: `05ac9d250 Revert "Remove deprecated double sparsity feature (#23009)"`; `ca5ef1668 Restore Double Sparsity attention backend on latest main`. Confirmed Llama / Triton / page=1 / no MLA / no FP8 / 3-12% throughput regression.
- PR #25304 commits include M1..M9 milestones, v1.1 fix train, v2 native sparse-decode pivot, FA3 adaptor, FlashInfer top_k_page_table selector backend. 90 files; contains `HANDOFF_NATIVE.md`, `SESSION_REPORT_2026-05-14.md`, pensieve install, ad-hoc bench harnesses.
- DeepSeek-V3.2 NSA path: `python/sglang/srt/layers/attention/nsa/` with `nsa_indexer.py`, `dequant_k_cache.py`, `quant_k_cache.py`, Triton + tilelang kernels, MTP precompute/verification.
- DeepSeek-V4 DSA path: `python/sglang/srt/layers/attention/dsv4/` with `compressor.py`, `compressor_v2.py`, `indexer.py`, `metadata.py`, `metadata_kernel.py`.
- Benchmark `development/benchmark.sh`: GSP dataset, SYS_LEN=2253, Q_LEN=1843 (ISL≈4096), OSL=512, NUM_PROMPTS=320, concurrencies 16/32/64 (only 64 active), PORT=30000.
- CLAUDE.md / `.pensieve/maxims/`: `eliminate-special-cases-by-redesigning-data-flow`, `prefer-pragmatic-solutions-over-theoretical-completeness`, `preserve-user-visible-behavior-as-a-hard-rule`, `reduce-complexity-before-adding-branches`.

## Output Format

Use EXACTLY these section headers:

```
AGREE:
- ...

DISAGREE:
- <topic>: Claude says X; you say Y; why your position is better
- ...

REQUIRED_CHANGES:
- <numbered, must-fix items the plan must adopt before convergence>
- ...

OPTIONAL_IMPROVEMENTS:
- <non-blocking suggestions>
- ...

UNRESOLVED:
- <opposite opinions needing a human decision>
- ...
```

Be specific. Quote AC/DEC IDs and section names from the plan when you push back. If you think an AC is wrong (e.g. wrong shape, wrong threshold, wrong test), say WHICH AC and what it should be. If you think a Milestone ordering is wrong, name the milestones. If a Pending User Decision is mis-framed, say so. If a Task is missing or redundant, name it. Examples of high-signal pushback:

- "AC-3's page-size negative test is wrong — `page_size=64` with `bfloat16` does not fail today; the validator routes it to `flashmla_sparse`."
- "Milestone 1 Phase B (wire Quest to MLA as a smoke test) is misordered — Quest's representation pool assumes Llama K_buffer shape, not MLA latent."
- "task15 is too late — GLM-5 compatibility constrains the label artifact schema; move it before task5."

The plan is allowed to have unresolved items — your job is to make sure the right items remain unresolved and that resolved items are actually resolved.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-05-19_09-09-39
- Tool: codex
