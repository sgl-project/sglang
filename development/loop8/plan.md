# Loop 8 (roadmap Loop 10) — GLM-5.1 DS bring-up on the existing DSA backend

## Goal Description

Bring up the **opt-in Double-Sparsity (DS) path on GLM-5.1-FP8**, single-node 8×H200 TP=8 (FP8 e4m3, page 64, fp8 KV; weights at `/cluster-storage/models/models--zai-org--GLM-5.1-FP8/`), and record the accuracy + client-SLO gates **DS-vs-DSA-native on the same node** — **without** a GLM-specific standalone DS backend and **without** regressing the model's native DSA default. This is a **model bring-up, not recall R&D** (that was Loop 7).

GLM-5.1 ships a **native trained DSA indexer**, and `is_deepseek_dsa()` already returns True for `GlmMoeDsaForCausalLM` (`configs/model_config.py`, with `index_topk: 2048`, `index_head_dim: 128`, `index_n_heads: 32`). So GLM-5.1 already routes through the **same `dsa_backend.py`** as DeepSeek-V3.2. **DSA-native is the default; DS is the reversible opt-in fallback (default-off)**, valuable where the trained indexer underperforms (e.g. long-context recall).

**Reframing of draft Scope-IN#1 (intent preserved, mechanism clarified by code evidence):** `glm4_moe.py` defines `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`, so GLM **already inherits** every DS model-forward hook (`_select_topk_indices`, `forward_absorb_prepare`, `finalize_double_sparsity_bind`, `_bind_double_sparsity_runtime_data`) and the `dsa_backend._write_token_labels` hook, and the MLA shape params (`qk_nope_head_dim`, `kv_lora_rank`, `qk_rope_head_dim`) are read from config — not hardcoded to DeepSeek values. The draft's intent ("reuse the existing wiring, no standalone GLM DS backend, no duplicated hooks") is therefore **already satisfied by inheritance**. The literal "generalize the DeepSeek-specific hooks" is delivered as a concrete deliverable: **for each inherited DS hook, either a config-driven patch removing a DeepSeek-only assumption, or documented source evidence that it is already GLM-safe** — plus the narrowest shape specialization only where GLM's wider shapes (`qk_nope=192`, `v_head=256`, indexer `128×32`) actually break a kernel or reshape. No broad model-forward abstraction.

### Client SLOs (canonical bar — `development/CLIENT_SLOS.md`, rebased 2026-06-07)

- **Model:** `zai-org/GLM-5.1 (FP8)`.
- **TPS (new distinct definition):** `30 TPS = (Total Latency − TTFT) / total tokens` per request (decode-only throughput; **distinct** from the old `output_tokens / e2e`).
- **Tail latency:** `P99 TTFT < 22 s`.
- **Workload:** 4096 ISL, 512 OSL, max-concurrency 64, minimum concurrency 16, prefix cache hit ≈ 55% — driven by `development/benchmark.sh`.
- **Knob support required:** TP, CUDA graphs, radix cache.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. (The `AC-*` labels are plan-documentation identifiers only; see Implementation Notes — they must not leak into code or comments.)

- AC-1: **DSA-native default is byte-identical with DS off (no regression to the shipped model).** With DS disabled, GLM-5.1 native DSA decode produces identical output token IDs before vs after all Loop-8 changes, under a fixed seed / request / runtime tuple (defined below).
  - Positive Tests (expected to PASS):
    - Same prompt set + fixed seed + identical server args (TP=8, FP8 e4m3, page 64, fp8 KV, CUDA graph on), DS off: generated token-ID sequences are **equal** to the pre-change baseline captured on the same commit's parent.
    - `is_deepseek_dsa(GLM-5.1)` remains True and the DSA path is selected unchanged when `--enable-double-sparsity` is absent.
  - Negative Tests (expected to FAIL):
    - A test that asserts the DS-off path diverges from baseline must FAIL (any divergence is a regression).
    - A test asserting DS code runs while `--enable-double-sparsity` is absent must FAIL (DS must be fully inert when not requested).
  - AC-1.1: Fixed verification tuple is recorded: model path, seed, prompt set, max_new_tokens, sampling=greedy (temperature 0), TP=8, page 64, fp8 KV, CUDA-graph state.
    - Positive: the result record cites the exact tuple and the parent commit SHA.
    - Negative: a run with an unrecorded/ambiguous tuple does not satisfy AC-1.

- AC-2: **DS attaches to GLM via the inherited `dsa_backend.py` wiring — no standalone GLM DS backend, no duplicated hooks — and is shape-correct on GLM's MLA/indexer dims.**
  - Deliverable: **per-hook generalization evidence** — for each inherited DS hook (`_select_topk_indices`, `_write_token_labels`, `forward_absorb_prepare`, the channel-mask bind), either a config-driven patch removing a DeepSeek-only assumption, or a documented source-evidence note that it is already GLM-safe.
  - Bind-time shape verification covers: `layers=78`, local/global head counts, `qk_nope=192`, `qk_rope=64`, `q_lora=2048`, `kv_lora=512`, `v_head=256`, `index_n_heads=32`, `index_head_dim=128`, mask `label_dim`, `page=64`, `TP=8`, FP8 e4m3, fp8 KV.
  - **Failure policy:** if DS is **explicitly requested** (`--enable-double-sparsity`) and any shape/mask is unsupported, the server **hard-errors with a diagnostic naming the offending field** — it does **not** silently fall back to DSA (silent fallback corrupts gate interpretation). When DS is **not** requested, the DSA path is fully unaffected.
  - Positive Tests (expected to PASS):
    - DS-enabled GLM-5.1 startup binds the inherited hooks and logs the full shape set matching the config.
    - `grep` of the diff shows no new `*ForCausalLM` DS backend class and no duplicated `_select_topk_indices`/`_write_token_labels` body for GLM.
    - A deliberately corrupted mask (`label_dim` mismatch) on a DS-requested launch raises a hard error citing `label_dim`.
  - Negative Tests (expected to FAIL):
    - A test asserting a separate `GlmDsBackend` / standalone GLM DS path exists must FAIL.
    - A test asserting DS silently degrades to DSA on an unsupported shape (instead of hard-erroring) must FAIL.
  - AC-2.1: Any kernel/reshape change is the **narrowest** shape specialization for `head_dim=192`; no broad model-forward abstraction is introduced.
    - Positive: changed lines trace to a concrete GLM-shape break found by the audit (task t1).
    - Negative: a refactor extracting a model-neutral helper without an audit-proven break must FAIL review (per DEC-1).

- AC-3: **GLM-5.1 channel mask calibrated, loaded, runtime-validated; DS decode coherent.**
  - Calibration uses the existing `calibrate.py` `q_b_proj` / `kv_b_proj` hooks, **verified to collect non-empty tensors at GLM's `qk_nope=192` no-PE slice offsets** (GLM's per-head projection output is wider than DeepSeek's because `qk_nope=192`, `v_head=256`).
  - Mask **`label_dim` is a GLM-native value** (per DEC-3 — chosen from GLM's wider MLA shapes, e.g. 24 or 32; **not** the DeepSeek-V3.2 value 16), justified in the calibration record.
  - **Artifact contract** (recorded in the mask metadata + a calibration provenance note): `label_dim`, `page_size=64`, TP layout (TP=8), `layers=78`, `q_lora=2048`, `kv_lora=512`, `qk_rope=64`, `qk_nope=192`, `v_head=256`, index dims (`index_topk=2048`, `index_head_dim=128`, `index_n_heads=32`), mask tensor shape, `content_sha256`, output path, and validation output.
  - Positive Tests (expected to PASS):
    - Calibration produces a safetensors mask whose metadata matches the contract and whose `content_sha256` re-verifies on load.
    - `load_channel_mask` + `validate_against_runtime` pass for GLM (head_dim/page/label_dim agree).
    - The within-budget / dense-DS startup sanity probe (`startup_sanity_probe`) passes (diagnostic, not a gate).
  - Negative Tests (expected to FAIL):
    - A calibration run that collects **empty** or wrong-width `q_b_proj`/`kv_b_proj` tensors must FAIL (silent wrong-channel collection is the primary calibration risk).
    - Loading a DeepSeek-V3.2 mask (`head_dim=128`) against GLM must FAIL runtime validation.

- AC-4: **Gates recorded DS-vs-DSA-native on the same node; landing policy applied (per DEC-2/DEC-4).**
  - Gates: (i) **MMLU** within tolerance of GLM DSA-native; (ii) **NIAH within-budget** characterization vs DSA (within-budget tolerance unless changed); (iii) **decode TPS ≥ 30** under the *new* definition `(Total Latency − TTFT) / total tokens`; (iv) **P99 TTFT < 22 s**.
  - **Landing policy (DEC-2):** **MANDATORY-to-land** = AC-1 byte-identical DS-off, MMLU within tolerance of DSA, DS-vs-DSA non-regression, and the SLO gates (iii)+(iv). **CHARACTERIZATION-only** = NIAH / long-context recall uplift-or-gap (the draft says characterize, not close).
  - **Workload (DEC-4):** the rebased `development/CLIENT_SLOS.md` GLM client workload — 4096 ISL, 512 OSL, conc {16…64}, ≈55% prefix cache hit — via `development/benchmark.sh`; MMLU/NIAH via `test/manual/test_double_sparsity_v32.py`; TTFT/decode-TPS via `development/loop7/perf_closed_batch.py` and/or `python/sglang/bench_serving.py`.
  - Positive Tests (expected to PASS):
    - A result record reports DS and DSA on the same node/op-point with absolute numbers for all four gates, plus a V3.2-vs-GLM shape matrix and exact repro commands.
    - The reported TPS is computed as `(Total Latency − TTFT) / total tokens` (the new definition), not `output_tokens / e2e`.
    - Mandatory gates meet their bars; recall is characterized (uplift / parity / documented gap).
  - Negative Tests (expected to FAIL):
    - A result that reports TPS via the old `output_tokens / e2e` definition must FAIL the SLO check.
    - A record that omits the DSA-native comparison column on the same node must FAIL.
    - A "pass" claimed without absolute P99 TTFT vs the 22 s bar must FAIL.
  - AC-4.1: **Harness model + SLO support.** `development/benchmark.sh`, `bench_serving.py`, and the MMLU/NIAH harness run against `zai-org/GLM-5.1 (FP8)` and the SLO computation emits the new TPS definition.
    - Positive: a smoke run of each harness against GLM-5.1 completes and the TPS field is the new definition.
    - Negative: a harness that only supports the prior model or only emits the old TPS definition does not satisfy AC-4.1.

- AC-5: **The already-committed DeepSeek-V3.2 DS path stays working after any shared-hook/kernel change (do-not-break-userspace).**
  - Positive Tests (expected to PASS):
    - Parameterized synthetic shape tests pass for **both** `qk_nope/v_head = 128/128` (V3.2) and `192/256` (GLM), plus the indexer dims `128×32`.
    - A DeepSeek-V3.2 DS smoke regression (bind + short decode) still passes.
  - Negative Tests (expected to FAIL):
    - A change that makes any V3.2 synthetic shape test or the V3.2 smoke regress must FAIL CI.
    - A specialization that only covers `192` and drops `128` support must FAIL.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

Shape-correct inherited DS on GLM-5.1 with **per-hook generalization evidence**; a GLM-native-`label_dim` calibrated channel mask carrying the full artifact contract; the complete gate record (MMLU + NIAH + new-definition TPS + P99 TTFT) DS-vs-DSA on the same node with a V3.2-vs-GLM shape matrix and repro commands; a **hard-error** bind-time shape guard for DS-requested-on-unsupported-shape; parameterized dual-shape (`128/128` and `192/256`) synthetic tests; and a verified DeepSeek-V3.2 non-regression. Benchmark + eval harnesses verified to run GLM-5.1 and emit the new TPS definition.

### Lower Bound (Minimum Acceptable Scope)

DS serves coherently on GLM-5.1 with a calibrated GLM-native-`label_dim` mask; the DSA default is byte-identical with DS off (AC-1); all four gates are **recorded** with absolute numbers DS-vs-DSA on the same node under the rebased client workload and the new TPS definition; the mandatory gates (DS-off identity, MMLU within tolerance, SLO TPS+TTFT, DS-vs-DSA non-regression) pass; recall is characterized even if it is only parity-or-documented-gap; and the committed DeepSeek-V3.2 DS path remains unbroken.

### Allowed Choices

- Can use: the inherited `DeepseekV2ForCausalLM` DS hooks as the intentional shared boundary; config-driven MLA shape params; narrow per-shape kernel/reshape specialization for `head_dim=192`; reuse of the existing gate harnesses (`test_double_sparsity_v32.py`, `perf_closed_batch.py`, `bench_serving.py`, `benchmark.sh`) extended for GLM-5.1; a GLM-native `label_dim` for calibration.
- Cannot use: a standalone GLM DS backend; duplicated model-forward hook bodies for GLM; a broad model-neutral model-forward abstraction introduced without an audit-proven shape break (per DEC-1); silent DSA fallback when DS is explicitly requested; the old `output_tokens / e2e` TPS definition; chasing parity-beating long-context recall in this bring-up loop.

> **Note on Deterministic Designs**: The draft fixes several choices (existing `dsa_backend.py` wiring, no standalone backend, the rebased client workload + new TPS definition, the calibration hook path). For those, upper and lower bounds converge — the only genuine latitude is (a) how much per-hook change the t1 audit forces and (b) the GLM-native `label_dim` value, both bounded by the decisions below.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

```
1. Smoke (zero/low cost): launch GLM-5.1 DS-off → confirm is_deepseek_dsa True, DSA path selected.
   Launch DS-on → confirm inherited hooks bind, log the full shape set, compare to config.
2. Audit (t1): read _write_token_labels (dsa_backend.py), _select_topk_indices
   (deepseek_v2.py), token_label_write.py, selection_kernel.py for any implicit
   qk_nope=128 / v_head=128 / power-of-two assumption. Produce per-hook evidence.
3. Shape guard (t2) + narrow specialization (t3) only where the audit finds a real break at 192.
   DS-requested + unsupported shape → hard error naming the field.
4. Calibration (t4 verify hooks → t5 produce mask): confirm q_b_proj/kv_b_proj collect non-empty
   tensors at GLM's qk_nope=192 no-PE offsets; calibrate with GLM-native label_dim; write the
   full artifact contract + content_sha256.
5. Dual-shape synthetic tests + V3.2 smoke (t6, AC-5) so shared changes can't break the
   committed DeepSeek-V3.2 DS path.
6. AC-1 byte-identical DS-off check (t7) under the fixed tuple.
7. Gates (t8): run MMLU + NIAH + closed-batch TPS (NEW def = (latency-TTFT)/tokens) + P99 TTFT
   DS-vs-DSA on the same node under the rebased CLIENT_SLOS workload; write the characterized
   record + shape matrix + repro commands.
```

### Relevant References

- `python/sglang/srt/layers/attention/dsa_backend.py` — DSA backend; `_write_token_labels` hook, MLA shape params read from config, DS metadata alloc.
- `python/sglang/srt/configs/model_config.py` — `is_deepseek_dsa` (GLM listed in the recognized arch set); `get_dsa_index_topk` / `get_dsa_index_head_dim` / `get_dsa_index_n_heads`.
- `python/sglang/srt/models/deepseek_v2.py` — `_select_topk_indices`, channel-mask bind, `finalize_double_sparsity_bind`, `_bind_double_sparsity_runtime_data`.
- `python/sglang/srt/models/glm4_moe.py` — `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`; shares `deepseek_common` weight loader.
- `python/sglang/srt/layers/attention/double_sparsity/` — `calibrate.py`, `channel_mask.py` (loader/validator/`startup_sanity_probe`), `token_label_write.py`, `selection_kernel.py`, `token_label_table.py`, `selector.py`.
- `development/CLIENT_SLOS.md` — rebased GLM-5.1 client bar (new TPS definition + workload).
- `development/benchmark.sh` — 4096 ISL / 512 OSL / ~55% cache-hit sweep (conc 16/32/64).
- `test/manual/test_double_sparsity_v32.py` — MMLU + NIAH gate harness; `development/loop7/perf_closed_batch.py` — closed-batch decode TPS + TTFT streaming probe; `python/sglang/bench_serving.py` — online SLO benchmark.
- `development/loop7/m12_final_decision.md` — Loop-7 decision + DS↔V3.2 bring-up record (prior art).

## Dependencies and Sequence

### Milestones

1. **Inherited-path validation + shape hardening** (AC-1, AC-2, AC-5)
   - Phase A: smoke — GLM enters DSA, DS bind runs, shape logs match config.
   - Phase B: audit inherited hooks/kernels for DeepSeek-only shape assumptions → per-hook evidence.
   - Phase C: bind-time shape verification + hard-error-on-unsupported-when-DS-requested; narrowest `head_dim=192` specialization where the audit forces it.
   - Phase D: dual-shape synthetic tests (`128/128` + `192/256`) + V3.2 smoke regression; AC-1 byte-identical DS-off check.

2. **GLM channel-mask calibration** (AC-3) — depends on Milestone 1 hook validation
   - Step 1: verify `q_b_proj`/`kv_b_proj` hook names + GLM no-PE slice offsets collect non-empty, correct-width channels.
   - Step 2: calibrate with GLM-native `label_dim`; write artifact + full contract metadata + `content_sha256`; runtime-validate + sanity probe.

3. **Gates + characterization** (AC-4) — depends on Milestones 1 and 2
   - Step 1: ensure `benchmark.sh` / `bench_serving.py` / MMLU+NIAH harness run GLM-5.1 and emit the new TPS definition (AC-4.1).
   - Step 2: run MMLU + NIAH + decode-TPS (new def) + P99 TTFT DS-vs-DSA on the same node under the rebased workload; write characterized record + shape matrix + repro commands; apply landing policy.

Dependencies: calibration (M2) cannot start until the inherited path is shape-validated (M1 Phase B/C), because the mask collection depends on the verified projection offsets. The gates (M3) require a loaded, validated mask (M2) and the unbroken DS-off baseline (M1 Phase D / AC-1).

## Task Breakdown

Each task includes exactly one routing tag: `coding` (Claude) or `analyze` (Codex, via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Audit inherited DS hooks + kernels (`_write_token_labels`, `_select_topk_indices`, `token_label_write.py`, `selection_kernel.py`) for hidden `qk_nope=128` / `v_head=128` / power-of-two / DeepSeek-only assumptions vs GLM `192/256/128×32`; produce per-hook generalization evidence (patch-or-justify). | AC-2 | analyze | - |
| task2 | Add bind-time shape verification (full GLM shape set) + hard-error-with-diagnostic when DS is explicitly requested on an unsupported shape/mask; ensure DSA path untouched when DS off. | AC-2 | coding | task1 |
| task3 | Narrowest shape specialization in any kernel/reshape task1 finds broken for `head_dim=192`. | AC-2 | coding | task1 |
| task4 | Verify `calibrate.py` `q_b_proj`/`kv_b_proj` hook names + GLM no-PE slice offsets collect non-empty, correct-width channels at `qk_nope=192`/`v_head=256`. | AC-3 | analyze | task1 |
| task5 | Produce GLM-5.1 channel-mask calibration recipe + safetensors artifact with GLM-native `label_dim`, full contract metadata, `content_sha256`, runtime validation + sanity probe. | AC-3 | coding | task4 |
| task6 | Low-cost smoke (GLM enters DSA, DS bind runs, shape logs match) + parameterized synthetic shape tests for `128/128` and `192/256` + DeepSeek-V3.2 DS smoke regression. | AC-2, AC-3, AC-5 | coding | task2, task3 |
| task7 | AC-1 byte-identical DS-off before/after token-ID comparison under the fixed seed/request/runtime tuple. | AC-1 | coding | task2 |
| task8 | Ensure `benchmark.sh` / `bench_serving.py` / MMLU+NIAH harness support GLM-5.1 and compute the new TPS definition `(Total Latency − TTFT)/total tokens`. | AC-4 | coding | - |
| task9 | Run accuracy (MMLU + NIAH) + SLO (new-def decode TPS + P99 TTFT) gates DS-vs-DSA on the same node under the rebased client workload; write characterized result record + V3.2-vs-GLM shape matrix + repro commands; apply landing policy. | AC-4 | coding | task5, task6, task7, task8 |

## Claude-Codex Deliberation

### Agreements

- This is a GLM model bring-up, not recall R&D; DSA-native stays the default, DS is the reversible opt-in (default-off).
- `GlmMoeDsaForCausalLM` already inherits the DeepSeek DS hooks; the real work is **validation + shape hardening + calibration + gates**, not hook plumbing or a new backend.
- `qk_nope=192`, `v_head=256`, and the indexer `128×32` are the concrete shape-risk points; the DS submodules are otherwise parametric/config-driven.
- DS-off byte-identity (AC-1), GLM-specific calibration with verified hook collection (AC-3), and same-node DS-vs-DSA gating (AC-4) are the right gate shapes.
- The committed DeepSeek-V3.2 DS path must not regress (AC-5).

### Resolved Disagreements

- **"Generalize hooks" vs "inheritance is enough":** Resolved — inheritance satisfies the draft's *intent*, but acceptance requires explicit **per-hook evidence** (patch-or-justify), not wording alone (AC-2 deliverable).
- **Unsupported-shape behavior:** Resolved — when DS is explicitly requested, the server **hard-errors with diagnostics**; it does not silently fall back to DSA (silent fallback would corrupt gate interpretation). DSA is untouched only when DS is not requested.
- **NIAH / long-context gate:** Resolved — the draft says *characterize*, not close, the recall gap; NIAH within-budget is **restored as a required (characterization) gate** (AC-4 ii), distinct from the mandatory MMLU/SLO bars.
- **"Recorded" vs "passed":** Resolved by DEC-2 — mandatory-to-land = DS-off identity + MMLU within tolerance + DS-vs-DSA non-regression + SLO (TPS+TTFT); recall is characterization-only.
- **DeepSeek-V3.2 non-regression:** Resolved — added AC-5 with dual-shape synthetic tests + a V3.2 smoke regression.
- **Hook boundary (shared vs extracted helper):** Resolved by DEC-1 — keep the inherited boundary; extract a model-neutral helper only if the task1 audit forces it.

### Convergence Status

- Final Status: `converged` (2 convergence rounds; round 2 returned no `DISAGREE` and no `REQUIRED_CHANGES`).

## Pending User Decisions

All carried decisions were resolved with the user during gen-plan discussion; none remain `PENDING`.

- DEC-1: DS model-forward hook boundary on GLM (shared inherited class vs extracted model-neutral MLA-DSA helper).
  - Claude Position: keep the inherited `DeepseekV2ForCausalLM` boundary; extract only if the audit proves a real DeepSeek-only assumption.
  - Codex Position: human decision — shared boundary vs proactive extraction.
  - Tradeoff Summary: extraction is cleaner but adds blast radius and risk to the committed V3.2 path; inheritance avoids speculative abstraction.
  - Decision Status: **Keep inherited boundary; extract only if the task1 audit forces it.**

- DEC-2: Landing policy — must accuracy/SLO thresholds pass to land, or is "characterize + default-off opt-in" enough?
  - Claude Position: parity (DS-off identity + MMLU within tolerance + DS-vs-DSA non-regression) + SLO mandatory; recall characterized.
  - Codex Position: human decision — experimental opt-in vs threshold-gated landing.
  - Tradeoff Summary: strictest (all four mandatory) risks blocking a useful default-off opt-in on a recall gap the draft says to characterize; loosest weakens the SLO numbers.
  - Decision Status: **Parity + SLO mandatory; NIAH/long-context recall characterization-only.**

- DEC-3: GLM first-calibration `label_dim` — reuse DeepSeek-V3.2 value (16) vs GLM-native.
  - Claude Position: reuse 16 for direct comparability (simplest baseline).
  - Codex Position: human decision — reuse vs GLM-native choice.
  - Tradeoff Summary: reuse is directly comparable; GLM-native may improve recall but adds an un-baselined variable.
  - Decision Status: **Pick a GLM-native `label_dim`** (chosen from GLM's wider MLA shapes, e.g. 24/32; justified in the calibration record) — **not** the DeepSeek value 16.

- DEC-4: SLO workload + TPS definition for the gates.
  - Claude Position: reuse the Loop-7 closed-batch op-point for comparability.
  - Codex Position: human decision — reuse vs new GLM workload.
  - Tradeoff Summary: reuse is comparable to V3.2 records; a new workload is more representative of the GLM client.
  - Decision Status: **Define a new GLM client workload** per the rebased `development/CLIENT_SLOS.md` — 4096 ISL / 512 OSL / conc 16…64 / ~55% cache hit via `development/benchmark.sh`, with the **new TPS definition `(Total Latency − TTFT)/total tokens`**, model `zai-org/GLM-5.1 (FP8)`; benchmark + eval scripts must support the new model and verify these SLOs (AC-4.1, task8).

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.
- Per the project Torvalds-doctrine guidelines: no standalone GLM DS backend, no duplicated hook bodies, no broad model-forward abstraction without an audit-proven break; fix shape handling in config-driven data, not with sprayed conditionals; every changed line must trace to a GLM shape break or a listed AC.

--- Original Design Draft Start ---

# Loop 8 Draft — GLM-5.1 DS bring-up (this is roadmap **Loop 10**, pulled forward)

> Written 2026-06-07, after **Loop 7 closed** (`.humanize/rlcr/2026-06-01_09-27-07/`, decision
> `development/loop7/m12_final_decision.md`). The client re-prioritized to **GLM-5.1 (deferred client #1)**,
> so roadmap **Loop 10** is pulled ahead of Loops 8/9 and executed in the next on-disk loop dir
> (`development/loop8/` = roadmap Loop 10). Feed this through `gen-plan` once scope is confirmed.

---

## Objective

Bring up the **opt-in Double-Sparsity (DS) path on GLM-5.1-FP8**, single-node TP=8, and re-run the accuracy +
SLO gates — **without** building a GLM-specific standalone DS path and **without** regressing the model's
native default. This is a model bring-up, not recall R&D (that was Loop 7).

## Model + weights (already staged)

- **Weights are already in `/cluster-storage/models/models--zai-org--GLM-5.1-FP8/`** (HF cache layout; 142
  FP8 safetensor shards under `snapshots/<hash>/`). No download needed.
- Architecture: **`GlmMoeDsaForCausalLM` / `model_type: glm_moe_dsa`** — MLA attention + a **native trained
  DSA indexer** (`self_attn.indexer` / `indexers_proj`) + 256-expert MoE; FP8 e4m3 block-quant (128×128);
  78 layers, hidden 6144, kv_lora 512 / q_lora 2048 / qk_rope 64 / qk_nope 192 / v_head 256; ~198k max ctx.

## Key finding — §4.0 question is ANSWERED for GLM-5.1

GLM-5.1 **ships a native trained DSA indexer**, and **`is_deepseek_dsa()` returns True for it** (confirmed:
config exposes `index_topk: 2048`, `index_head_dim: 128`, `index_n_heads: 32`;
`python/sglang/srt/configs/model_config.py:111`). So GLM-5.1 already routes through the **same
`dsa_backend.py`** as DeepSeek-V3.2 (and at the same 2048 budget). ⇒ **Same posture as V3.2: DSA-native is the default; DS is the
opt-in fallback**, valuable only where the trained indexer underperforms (e.g. long-context recall — the
Loop-7 regime). DS is *not* the primary path here; it is the reversible opt-in knob, default-off.

## Scope — IN

1. **Compatibility — wire DS into the preexisting backend (the load-bearing requirement).** GLM-5.1 uses the
   existing DSA backend (`dsa_backend.py`); the DS solution must **reuse our current wiring pattern** — bind
   DS into that preexisting backend (the bind site + `TokenLabelTable` + the selection/label-write hooks),
   **not** a separate GLM DS backend. Concretely: generalize the DS model-forward hooks that are today
   DeepSeek-specific (`deepseek_v2.py`: `_select_topk_indices` / `forward_absorb_prepare` /
   `_write_token_labels` / the channel-mask bind) onto the **GLM model forward**, so DS attaches to GLM the
   same way it attaches to V3.2.
2. **Calibrate a GLM-5.1 channel mask** (the offline importance projection) for the GLM MLA shapes, and bring
   up the DS serving path (TP=8, FP8, page 64).
3. **Re-run the gates on GLM-5.1**: accuracy (MMLU within tolerance of the DSA-native default) + the client
   SLOs (≥30 TPS/req, P99 TTFT < 22 s at the client workload) + DS-vs-DSA-native non-regression.

## Scope — OUT

- **Re-litigating V3.2** or the Loop-7 recall R&D (the learned-selector follow-on is its own loop, roadmap 11).
- **nvfp4/mxfp4, multi-node TP, the knob-compat matrix** — their own roadmap loops (8/9 deferred behind this).
- Closing any GLM long-context recall gap — first **bring it up and characterize**; recall R&D is downstream.

## Acceptance criteria (draft — `gen-plan` formalizes)

1. GLM-5.1 **serves** with DS opt-in on TP=8 FP8; the DSA-native default path is **byte-identical when DS is
   off** (no regression to the shipped model).
2. DS attaches via the **existing `dsa_backend.py` wiring** (no standalone GLM DS backend); the DeepSeek-only
   model-forward hooks are generalized, not duplicated.
3. GLM-5.1 channel mask calibrated + loaded; DS decode is coherent (dense-DS / within-budget sanity).
4. Accuracy + SLO gates recorded DS-vs-DSA-native on the same node; result characterized (uplift or parity or
   documented gap).

## Hardware / inputs to read first

- **Hardware:** single node 8×H200 (TP=8), FP8 e4m3, page 64, fp8 KV.
- **The preexisting backend DS wires into:** `python/sglang/srt/layers/attention/dsa_backend.py`
  (`is_deepseek_dsa` → `use_dsa`); the DSA recognition at `configs/model_config.py:102-114`.
- **The DS wiring to generalize:** `deepseek_v2.py` (DS hooks) + `double_sparsity/` (selection_kernel,
  token_label_write, channel-mask calibration).
- **GLM model forward (where the DS hooks land):** `python/sglang/srt/models/glm4_moe.py` registers
  `GlmMoeDsaForCausalLM` and already shares `models/deepseek_common/deepseek_weight_loader.py` — so the
  DeepSeek-common infra is partly reusable and the generalized DS hooks have a natural home here.
- **Prior art:** the Loop-7 decision (`development/loop7/m12_final_decision.md`) + the DS↔V3.2 bring-up record.

## Pending decisions (resolve in `gen-plan` discussion)

- **How much of the DeepSeek DS model-forward hook can be shared vs GLM-specialized.** `glm4_moe.py` handles
  `GlmMoeDsaForCausalLM` and already shares `deepseek_common`, so part is reusable — but the MLA dims differ
  (v_head 256, qk_nope 192) and the indexer is wider (`index_head_dim` 128 × 32 heads). Where is the clean
  shared-vs-specialized boundary, and does the DS channel-mask calibration assume DeepSeek-only shapes?
- **Is DS even worth landing on GLM-5.1 given its native indexer?** Default expectation (per §4.0): land it as
  the reversible opt-in fallback + characterize; do not chase parity-beating recall in this bring-up loop.

--- Original Design Draft End ---
