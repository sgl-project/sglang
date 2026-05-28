# Loop 5: Double Sparsity MVP on H200 — Smoke Milestone + Loop4-Compatible MVP

## Goal Description

Ship a demonstrable Double Sparsity (DS) MVP on DeepSeek-V3.2 (FP8) running end-to-end on the H200 cluster, in two explicit tiers, without conflating a hardware smoke milestone with the loop4-complete MVP.

- **TIER 1 — Smoke MVP:** DS-on V3.2 FP8 serves real requests on H200, produces *genuinely sparse* (non-trivial) DS selection, yields one DS benchmark artifact + one matching DSA benchmark artifact at a clearly-labeled smoke shape, and passes the paired quality smoke. Narrative: "DS-on V3.2 FP8 serves at the locked Option B operating point. Side-by-side with DSA at conc 16/32/64. Quality smoke passes on 20 paired prompts. Here's the bench JSON and the comparator report."
- **TIER 2 — Loop4-compatible MVP:** the smoke milestone plus the loop4 requirements needed to claim comparable default-cookbook behavior: radix cache enabled for the final serving run (AC-10), the AC-11 directional comparator, the AC-12 full quality gate, CUDA-graph status recorded (AC-6), and the chunked-prefill probe run and recorded (AC-1b). Narrative adds: "The final run used matching production knobs, including radix cache enabled; CUDA graph and chunked-prefill status are recorded; AC-11 comparator and AC-12 quality gates are complete."

If AC-10 radix, AC-11 comparator, or AC-12 full quality are missing, the result is a useful smoke milestone, not the minimal viable working version requested by loop4.

**Root blocker (loop 4 never executed against hardware):** the calibration artifact `/models/dsv32-fp8-channel-mask.safetensors` does not exist on disk, and generating it unblocks every DS-on criterion. A second prerequisite is the Round-38 AC-10 producer bug, which must be patched before any radix-on or default-cookbook parity claim.
<comment>The plan correctly names the channel mask as the root blocker, then immediately introduces a second prerequisite (the AC-0 producer fix) that is gated on radix-on claims only. But task1 (AC-0) and task3 (calibration load change) are shown as independent, meaning a round could open with the AC-0 fix and spend the entire round on code changes before any hardware is touched. The draft's explicit instruction is "the existing loop-4 code stays as-is unless a specific bench failure mode requires patching it" and "the Round 38 AC-10 producer bug is the one known exception." The plan should make explicit that AC-0 is a prerequisite for radix-on claims only, not for the mask generation or the boot smoke. If a round starts by working on the producer fix, it must also produce a hardware artifact that round — the fix alone is not a complete round. The current framing doesn't enforce this; "coding" tag is not the same as "artifact produced."</comment>
<comment>Codex — Agree. Code confirms the AC-0 producer bug is capture-specific: `dsa_backend._write_token_labels` writes the token-label table first, then references undefined `forward_batch` only inside the `SGLANG_DS_RADIX_FIXTURE_CAPTURE` branch. Mask generation and a radix-off boot do not consume that capture path, so AC-0 should gate radix evidence/default parity, not calibration; any AC-0 round still needs a hardware artifact to avoid repeating Loop 4 drift.</comment>

**Key feasibility correction (resolved with the user):** the documented single-GPU bf16 calibration command cannot run as written. V3.2 is FP8-quantized on disk (~671GB), which fits across one node's 8 H200s (1.14TB), but `calibrate.py` currently upcasts the load to bf16 (~1.3TB, will not fit) and pins the whole model to a single CUDA device. The agreed approach is to change the calibration load path to a native-FP8, device-sharded load across the node's 8 GPUs (single node, no second node required).
<comment>Codex — Additional critique: AC-4 says native FP8 sharded load, but the current calibration forward loop still calls `model(input_ids=block.to(model.device))`, which is a single-device assumption that can break once HF dispatch shards modules. Task3 should produce a tiny one-block calibration log that records parameter dtypes and device placement before spending a full 256-block run.</comment>

## Acceptance Criteria

Acceptance criteria preserve the project's existing domain AC numbering (referenced by in-code markers such as `AC-10-FIXTURE-MARKER` and by AC-11 artifact naming) so plan and codebase stay traceable. Following TDD philosophy, each criterion includes positive and negative tests.

- **AC-0: Round-38 AC-10 producer-bug fix.** `_write_token_labels` accepts a `forward_batch` argument, and every production call site threads the live `forward_batch`: the extend, decode, and TRT-LLM call sites in `dsa_backend.py`, and the MHA_ONE_SHOT call site in `forward_mha.py` (where `forward_batch` is already in scope). Token-label writes stay first; the radix capture publishes the extend snapshot only when `forward_batch` is present and the mode is extend. A producer-side regression is added.
  - Positive Tests (expected to PASS):
    - With `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, a `/generate` request returns non-empty `meta_info["double_sparsity_radix_capture"]` with `per_token_slot_sha` populated and `per_layer_written_all_true=True`, and no capture error key.
    - The new producer-side pytest regression passes.
  - Negative Tests (expected to FAIL / be rejected):
    - With capture disabled, the response carries no `double_sparsity_radix_capture` key.
    - A decode-only forward does not publish or overwrite the extend snapshot.
    - A short dense prefill routed through the MHA_ONE_SHOT path still writes token labels (no silent skip).

- **AC-4: Channel mask generated and validated.** The calibration load path is changed to load native FP8 weights sharded across the node's GPUs (no bf16 upcast, no single-device pin). Output `/models/dsv32-fp8-channel-mask.safetensors` is readable by every DS process on the node, and its content SHA is recorded.
  - Positive Tests (expected to PASS):
    - `load_channel_mask()` succeeds; metadata `dtype="fp8_e4m3"`, `page_size=64`, `label_dim=16`, `head_dim=128`; `channel_selection` is `int32 [L, H, 16]`; content SHA recorded.
    - Calibration completes on a single node using the sharded FP8 load (does not OOM).
  - Negative Tests (loader-enforceable, expected to be rejected):
    - A mask with missing tensors, wrong dimensionality, `dtype`/`label_dim`/`page_size` mismatch, a channel index outside `[0, head_dim)`, or a content-SHA mismatch is rejected by `load_channel_mask`.
  - Artifact-review note (NOT loader-enforceable): the mask file carries no calibration-provenance field, so a random mask with a valid hash will load successfully. Provenance is therefore established by (a) `calibrate.log` present in the run directory, (b) the recorded content SHA, and (c) AC-1.1 demonstrating genuinely non-trivial selection. A degenerate/synthetic mask is caught indirectly via AC-1.1, not by the loader.

- **AC-1: DS boot smoke.** `serve_double_sparsity.sh` boots on a single node at TP=8 with `MODEL_PATH` pinned to the cluster weights and the newly generated mask; one `/generate` request returns text; the token-label table populates from the production `_write_token_labels` hook.
<comment>Codex — Additional critique: this is not just a run step yet, because `development/serve_double_sparsity.sh` still defaults `MODEL_PATH` to `deepseek-ai/DeepSeek-V3.2`, while DEC-6 requires `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`. Task5 must edit or override that before first boot, otherwise `/get_server_info` can describe a different revision/download path than the artifact bundle claims.</comment>
  - Positive Tests (expected to PASS):
    - `/get_server_info` shows DS enabled, TP=8, `kv_cache_dtype=fp8_e4m3`, `page_size=64`, and the expected radix setting; `/generate` returns non-empty text.
  - Negative Tests (expected to FAIL / be rejected):
    - A missing or invalid mask causes the validator to reject boot with a verbatim error rather than silently falling back to dense attention.
  - **AC-1.1: Non-trivial DS selection (TIER 1).** A prompt longer than `top_k` proves the selection is genuinely sparse, using the real DS meta fields `sparsity_rate` (float in `[0,1]`), `selected_tokens`, and `dense_fallback`.
    - Positive: `meta_info["double_sparsity"]` shows `0 < sparsity_rate < 1` (selected tokens fewer than the sequence length) and `dense_fallback == 0` on a long prompt.
    - Negative: `sparsity_rate == 1` (all tokens selected) on a long prompt fails — DS would be effectively dense; `dense_fallback == 1` also fails.

- **AC-1b: Chunked-prefill probe (TIER 2).** The probe is run and recorded. If it passes, the default chunked-prefill setting is kept; if it fails, chunked-prefill is disabled on BOTH DS and DSA for apples-to-apples evidence and a follow-up is filed. Sequencing: AC-1b belongs to the loop4 milestone and must run before the AC-11 sweep so the sweep collects artifacts at the final operating point. Deferred for the TIER 1 smoke.
  - Positive Tests (expected to PASS):
    - The probe result is recorded in the run directory; if disabled, both DS and DSA sidecars show `chunked_prefill_size=-1`.
  - Negative Tests (expected to FAIL / be rejected):
    - An AC-11 benchmark set collected under mismatched chunked-prefill settings between DS and DSA is invalid.

- **AC-6: CUDA-graph status recorded (TIER 2).** The final bundle records whether CUDA graphs are enabled, with capture/replay success OR a clearly-recorded exception explaining why capture cannot be used under Option B.
<comment>Codex — Additional critique: AC-6 should distinguish regular CUDA graph from piecewise CUDA graph. Option B passes `--disable-piecewise-cuda-graph`, but regular `disable_cuda_graph` remains false by default and `ModelRunner.init_device_graphs()` is the boot-time capture path; evidence that only says piecewise is disabled does not prove regular capture/replay status.</comment>
  - Positive Tests (expected to PASS):
    - The run directory contains an explicit CUDA-graph status (capture/replay outcome).
  - Negative Tests (expected to FAIL / be rejected):
    - Recording only `disable_cuda_graph=False` with no capture/replay evidence or documented exception is insufficient.

- **AC-8 / AC-9: DS and DSA benchmark artifacts.** AC-8 covers the DS run via `benchmark.sh`; AC-9 covers the DSA baseline via `benchmark_baseline.sh` at the matching operating point. For the TIER 1 smoke, both run with explicit `TRIALS=1` and a shortened `MEASUREMENT_WINDOW_S`, clearly labeled as non-AC-11, with radix cache disabled on BOTH sides (matching the DS launcher's `--disable-radix-cache`).
  - Positive Tests (expected to PASS):
    - The configured number of JSONL artifacts is produced; each artifact's measured duration meets its configured window; `.meta.json` sidecars are present and valid; smoke runs are labeled.
  - Negative Tests (expected to FAIL / be rejected):
    - `benchmark.sh`'s hard guard refuses to publish any JSONL whose observed duration is below `MEASUREMENT_WINDOW_S`.
    - A smoke-labeled JSONL presented as AC-11 evidence is rejected.

- **AC-10: Radix-cache flip (TIER 2).** Both radix fixtures pass (the label-capture fixture and the FP8 scale-stability fixture), the validator guard is flipped, `--disable-radix-cache` is removed from the final DS launch, and the DS server boots radix-on WITHOUT relying on an environment override.
<comment>The Feasibility Hint (step 5) says "design and implement the AC-10 radix-flip mechanism." That phrasing — design, then implement — is the exact pattern that caused Loop 4 to stall. The AC text itself is correctly outcome-focused ("boots radix-on WITHOUT relying on an environment override"), but combining it with a dedicated analyze task (task10) creates a scaffolding trap. The validator guard flip is a few lines in `validator.py` and a launcher flag. There is no mechanism here worth a separate design round — the design is: pass a CLI flag, the validator reads it, done. Maxim: eliminate-special-cases — the "design" step is a special case that should not exist; just implement it directly in task11.</comment>
<comment>Codex — Partially agree. I agree the analyze wording invites drift, but the "few lines" premise is too glib: `validate_double_sparsity()` reads a transient `server_args._double_sparsity_radix_fixture_passed` before boot, `record_radix_fixture_passed()` only sets that in-process attribute and logs an optional artifact SHA, and `serve_double_sparsity.sh` has no CLI/state hook today. That is real ServerArgs/launcher plumbing, not just deleting `--disable-radix-cache`.</comment>
  - Positive Tests (expected to PASS):
    - The DS server boots radix-on; both fixtures pass; the final comparator runs with radix cache on for both sides.
  - Negative Tests (expected to FAIL / be rejected):
    - A radix-on boot that still requires an environment override (rather than a launcher/CLI mechanism) fails AC-10.

- **AC-11: Directional comparator (TIER 2).** A 3-trial DSA + DS sweep at conc 16/32/64, 120s warmup, 600s measurement window, median comparison. DS TPS within 5% of DSA and DS P99 TTFT no worse than 1.10x DSA are **directional targets**, not hard build-breaks: a miss is recorded as an AC-11 failure requiring follow-up tuning. Radix settings MUST match between sides (the comparator refuses a mismatch).
  - Positive Tests (expected to PASS):
    - The expected DSA and DS JSONL counts are produced, each with duration ≥ 600s and valid sidecars; radix settings match; the comparator exits 0 and emits a TPS/TTFT pass-or-fail summary.
  - Negative Tests (expected to FAIL / be rejected):
    - The comparator refuses to publish when `disable_radix_cache` differs between the two sides.
    - A directional miss (TPS gap > 5% or P99 TTFT > 1.10x) is published as an explicit AC-11 failure with a follow-up, not silently absorbed or hidden.

- **AC-12: Full quality gate (TIER 2).** NIAH 4K/16K/64K + MMLU 5-shot via `test_double_sparsity_v32.py`.
  - Positive Tests (expected to PASS):
    - All NIAH and MMLU gates pass at their thresholds.
  - Negative Tests (expected to FAIL / be rejected):
    - Any gate below threshold fails the loop4 MVP claim.
    - Optional negative-sensitivity runs: corrupt-mask and zero-signature servers fail loudly (demonstrating the gate has teeth).

- **AC-Q: Paired quality smoke (TIER 1).** `test_dsv32_quality_smoke.py` compares DS-on vs DSA outputs on 20 deterministic prompts. Because two TP=8 servers cannot co-reside on one 8-GPU node, the smoke runs single-node sequentially: capture DSA reference outputs first, then run DS against them. Four gates: `prefix_match_rate >= 0.80`, `mean_rouge_l >= 0.85`, `niah_mini_recall >= 4/5`, AND `first_8_tokens_divergence == 0`.
  - Positive Tests (expected to PASS):
    - All four gates pass on the 20 paired prompts.
  - Negative Tests (expected to FAIL / be rejected):
    - Any single gate below threshold fails the quality smoke (e.g., `first_8_tokens_divergence > 0` fails even if the other three pass).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
Both tiers complete: AC-0 producer fix; the calibration FP8-sharded load change; AC-4 real calibrated mask; AC-1 + AC-1.1 boot and genuine sparsity; AC-8/AC-9 benchmarks; AC-Q quality smoke; plus the loop4 tier — AC-10 radix-on final serving, AC-11 3-trial comparator, AC-6 CUDA-graph evidence, AC-1b chunked-prefill probe, and AC-12 full quality gate — with the complete evidence bundle assembled in `runs/<date>_dsv32_mvp/`.

### Lower Bound (Minimum Acceptable Scope)
TIER 1 smoke only: AC-0 producer fix, the calibration load change sufficient to produce AC-4's real calibrated mask, AC-1 + AC-1.1 (boot and non-trivial selection), one smoke-labeled DS benchmark, one matching radix-off DSA benchmark, the radix-off-both-sides smoke comparator, and one AC-Q quality-smoke artifact — assembled side-by-side and explicitly labeled "smoke milestone, not loop4 MVP."

### Allowed Choices
- Can use: single-node sequential serving (DS then DSA, with DSA reference outputs captured first for the quality smoke); native-FP8 device-sharded model loading in the calibration path; an explicit smoke benchmark shape (`TRIALS=1`, shortened `MEASUREMENT_WINDOW_S`) provided artifacts are labeled non-AC-11; a radix-off-both-sides smoke comparator.
- Cannot use: a synthetic/placeholder mask presented as calibrated; a comparator run across mismatched radix-cache settings; a bf16 upcast / single-device model load for calibration (it cannot fit); a smoke artifact presented as AC-11 evidence; a radix-on final boot that depends on an environment override.

> Per the draft, several knobs are fixed and the boundaries are intentionally narrow: TP=8, `kv_cache_dtype=fp8_e4m3`, `page_size=64`, the locked Option B operating point (overlap scheduling and piecewise CUDA graph disabled), conc 16/32/64. These are deterministic constraints, not free choices.

## Feasibility Hints and Suggestions

> Reference only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
1. Patch the producer bug first (AC-0): add `forward_batch` to `_write_token_labels`, thread it from all four call sites, gate the extend-snapshot publish on `forward_batch is not None and forward_mode.is_extend()`, add the regression, then verify the capture probe over `/generate`.
2. Change the calibration load path (AC-4 prerequisite) to a native-FP8, device-sharded load across the node's 8 GPUs (e.g., `device_map="auto"` without forcing a bf16/fp16 `torch_dtype` upcast), then run calibration on a single node and validate the artifact via `load_channel_mask`.
3. Pin `MODEL_PATH` to the cluster weights for both serve scripts, boot DS at TP=8, run the boot smoke and the long-prompt sparsity check (AC-1, AC-1.1).
4. Run the smoke benchmarks with explicit `TRIALS=1` + shortened window, radix-off on both sides, then the smoke comparator; run the sequential quality smoke (AC-8/AC-9/AC-Q).
5. For the loop4 tier: design and implement the AC-10 radix-flip mechanism (no env override for the final boot — see DEC-5), run the chunked-prefill probe (AC-1b), then the AC-11 3-trial sweep radix-on, record CUDA-graph status (AC-6), and run the AC-12 full quality gate. Assemble the evidence bundle.

### Relevant References
- `python/sglang/srt/layers/attention/dsa_backend.py` — `_write_token_labels` and its extend/decode/TRT-LLM call sites; the env-gated radix capture branch.
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` — the MHA_ONE_SHOT `_write_token_labels` call site (fourth site).
- `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` — calibration entrypoint; the model-load path that needs the FP8-sharded change.
- `python/sglang/srt/layers/attention/double_sparsity/channel_mask.py` — `load_channel_mask`, metadata schema, content-SHA validation.
- `python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py` — M3-B capture primitive.
- `python/sglang/srt/layers/attention/double_sparsity/validator.py` — DEC-2 guard and the radix-fixture-passed recording used by the AC-10 flip.
- `python/sglang/srt/layers/attention/double_sparsity/metrics.py` — DS meta field names (`sparsity_rate`, `selected_tokens`, `dense_fallback`).
- `development/serve_double_sparsity.sh`, `development/serve_native_nsa.sh` — single-node TP=8 launchers; `MODEL_PATH` default and the `--disable-radix-cache` marker.
- `development/benchmark.sh`, `development/benchmark_baseline.sh` — `MODE`/`CONCURRENCIES`/`TRIALS`/`WARMUP_SECONDS`/`MEASUREMENT_WINDOW_S` env knobs and the hard duration guard.
- `development/benchmark_compare.py` — the radix-parity-enforcing comparator (`--baseline`/`--ds`/`--output`).
- `test/manual/test_dsv32_quality_smoke.py`, `test/manual/test_double_sparsity_v32.py` — quality smoke (4 gates) and full quality gate.

## Dependencies and Sequence

### Milestones
1. **M1 — Unblock.**
   - Phase A: AC-0 producer-bug fix + regression.
   - Phase B: calibration FP8-sharded load change; generate and validate the mask (AC-4).
2. **M2 — Smoke (TIER 1).**
   - Phase A: DS boot smoke (AC-1) and non-trivial selection (AC-1.1).
   - Phase B: smoke DS + DSA benchmarks, radix-off both sides, `TRIALS=1` + shortened window (AC-8/AC-9).
   - Phase C: smoke comparator (radix-off both sides) and sequential paired quality smoke (AC-Q).
3. **M3 — Loop4-compatible (TIER 2).**
   - Phase A: AC-10 radix-flip mechanism + both fixtures; remove `--disable-radix-cache`.
   - Phase B: AC-1b chunked-prefill probe (must precede the sweep).
   - Phase C: AC-11 3-trial radix-on sweep + comparator; AC-6 CUDA-graph evidence.
   - Phase D: AC-12 full quality gate; assemble the `runs/<date>_dsv32_mvp/` evidence bundle.

Dependencies: M1 Phase B depends on the calibration load change; M2 depends on M1 (mask + producer fix); M3 depends on M2 and on the AC-10 flip (Phase A) before the radix-on sweep; AC-1b (M3 Phase B) precedes AC-11 (M3 Phase C). Each round must produce an artifact under `runs/<date>_dsv32_mvp/`, not just code changes; a round that produces no artifact stalled.

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Add `forward_batch` to `_write_token_labels`; thread it from extend/decode/TRT-LLM (dsa_backend.py) and MHA_ONE_SHOT (forward_mha.py); gate extend-snapshot publish; add producer regression | AC-0 | coding | - |
| task2 | Verify capture probe: `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` `/generate` returns non-empty radix-capture meta_info | AC-0 | coding | task1 |
<comment>task2 is tagged `coding` but it's a hardware run that produces an artifact, not a code change. If the probe fails here, task3 is unblocked and task4 depends on task3, so you'd start calibration before you've confirmed the producer fix holds. The dependency graph should show task4 (and therefore task5) depending on a passing task2, not just task1. Right now the plan lets a broken AC-0 silently coexist with a running calibration.</comment>
<comment>Codex — Partially agree. The graph really does allow task3/task4 before task2 because task3 has no dependency and task4 depends only on task3, and task2 is an artifact-producing hardware run mislabeled as coding. I would not make calibration depend on AC-0, though: `calibrate.py` and `load_channel_mask()` do not consume the radix capture path, and task5 already gates first DS boot on both task2 and task4. The cleaner fix is to let mask generation proceed in parallel while making task2 an explicit artifact gate for M2.</comment>
| task3 | Change calibration load path to native-FP8 device-sharded load (no bf16 upcast, no single-device pin) | AC-4 | coding | - |
<comment>The `device_map="auto"` approach mentioned in the Feasibility Hints is speculative — `calibrate.py` likely uses its own model-loading routine tied to SGLang's engine, not a bare HuggingFace `from_pretrained` call. If the calibration entrypoint doesn't go through the standard `AutoModel` path, `device_map="auto"` does nothing and you're back to OOM on load. task3 needs to verify the actual load call site in `calibrate.py` before proposing a fix, or you'll code-change the wrong thing and not discover it until task4 blows up on hardware. Maxim: prefer-pragmatic-solutions — read the code before prescribing the patch.</comment>
<comment>Codex — Disagree with the factual premise. The actual load is a bare `AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map={"": "cuda" if torch.cuda.is_available() else "cpu"}, trust_remote_code=True)` after `AutoConfig`, so `device_map="auto"` would hit HF/Accelerate rather than an SGLang engine loader. The real risk is different: task3 must prove HF can shard-load the DeepSeek FP8 block-quantized checkpoint without upcasting, and must fix the later `model.device` forward-loop assumption if modules are dispatched across GPUs.</comment>
| task4 | Run single-node calibration; produce and validate the mask via `load_channel_mask`; record content SHA | AC-4 | coding | task3 |
| task5 | Pin `MODEL_PATH`; boot DS at TP=8 with the mask; confirm `/get_server_info` knobs and `/generate` text | AC-1 | coding | task2, task4 |
| task6 | Long-prompt sparsity check: assert `0 < sparsity_rate < 1` and `dense_fallback == 0` from meta_info | AC-1.1 | coding | task5 |
| task7 | Run smoke DS + DSA benchmarks (`TRIALS=1`, shortened window, radix-off both sides, labeled) | AC-8, AC-9 | coding | task5 |
| task8 | Run smoke comparator radix-off both sides; emit `mvp_compare.md` | AC-8 | coding | task7 |
<comment>task8 is labeled AC-8 but it's the comparator output, which is distinct from the DS benchmark artifact that AC-8 actually requires. This is a wrong AC assignment — the comparator is the AC-8/AC-9 *consumer*, not its target. Minor, but in a plan where artifact labeling is the whole point of the smoke-vs-loop4 distinction, mislabeling here will cause confusion during the evidence-bundle assembly in task16.</comment>
<comment>Codex — Agree. `development/benchmark.sh` and `development/benchmark_baseline.sh` produce the benchmark JSONL plus `.meta.json` sidecars, while `development/benchmark_compare.py` consumes `--baseline` and `--ds` JSONLs and emits a report. task8 should target comparator/report evidence, not AC-8 alone; AC-8/AC-9 are task7's producer artifacts.</comment>
| task9 | Sequential paired quality smoke (capture DSA refs, then DS); assert all four gates | AC-Q | coding | task5 |
| task10 | Design the AC-10 radix-flip mechanism (CLI/launcher vs env override; final boot must not need an env override) | AC-10 | analyze | task2 |
<comment>This is the exact drift the draft warned about. Loop 4 stalled on "analyze" work that never produced hardware artifacts. task10 is a pure design task with no artifact output — it doesn't even produce a `runs/` entry. The draft says "single mainline objective per round = the next concrete command." The AC-10 radix-flip mechanism is not architecturally ambiguous: the validator reads a flag or a state file, the launcher sets it, done. You don't need a design round for this; you need a commit and a boot. Collapse task10 into task11 or, at minimum, require task10 to produce a concrete proposal artifact (a written decision, not just "we thought about it") so it doesn't become a multi-round analysis spiral. Maxim: prefer-pragmatic-solutions — don't design what you can just implement.</comment>
<comment>Codex — Partially agree. Collapse the analyze-only task or require a written decision artifact, yes. But current code is not just a flag flip: `ServerArgs` has no fixture-passed CLI field, `validate_double_sparsity()` runs during `check_server_args()`, and the helper's state is not persisted across processes. A clean no-env path needs a specific ServerArgs/launcher/artifact contract.</comment>
| task11 | Implement the radix-flip mechanism; pass both fixtures; remove `--disable-radix-cache` from the final DS launch | AC-10 | coding | task10 |
| task12 | Run and record the chunked-prefill probe; if it fails, disable on both DS and DSA | AC-1b | coding | task11 |
| task13 | AC-11 3-trial radix-on sweep (conc 16/32/64, 120s/600s) + comparator; emit directional pass/fail summary | AC-11 | coding | task11, task12 |
| task14 | Record CUDA-graph capture/replay status or documented exception | AC-6 | coding | task11 |
<comment>task14 depends on task11 (radix flip), but CUDA-graph status is independent of whether radix cache is on or off. You can record this status immediately after the first DS boot in task5 — that's when the server either captures graphs or logs why it can't. Deferring it to M3 means you have no CUDA-graph evidence until after the AC-10 flip, which adds an unnecessary wait and a round of work that could have been a one-liner artifact in task5. This is scope-creep-by-dependency-ordering. Move task14 to depend on task5.</comment>
<comment>Codex — Agree. `ModelRunner.init_device_graphs()` captures regular CUDA graphs during model-runner initialization whenever `disable_cuda_graph` is false, before any request and independent of radix cache; the DS launcher disables piecewise CUDA graph, not regular CUDA graph. Replay/use is then visible through scheduler logs and metrics via `can_run_cuda_graph`, so the first task5 boot plus smoke request can already record capture and replay status.</comment>
| task15 | AC-12 full quality gate (NIAH 4K/16K/64K + MMLU 5-shot) | AC-12 | coding | task11 |
| task16 | Assemble the `runs/<date>_dsv32_mvp/` evidence bundle (logs, server args, knob evidence, JSONLs + sidecars, comparator, quality artifacts) | AC-8, AC-9, AC-11, AC-12 | coding | task8, task9, task13, task14, task15 |

## Claude-Codex Deliberation

### Agreements
- AC-0 is the correct first blocker: `_write_token_labels` references `forward_batch` without accepting it, and the failure is swallowed by a `try/except` so the extend snapshot never publishes.
- The two-tier split (smoke milestone, then loop4-compatible MVP gated on AC-10/11/12 plus AC-6/AC-1b) is sound.
- AC-11 must enforce radix parity; `benchmark_compare.py` refuses a `disable_radix_cache` mismatch.
- AC-10 requires both the label-capture and FP8 scale-stability fixtures before removing `--disable-radix-cache`.
- The calibration feasibility issue, topology, smoke shape, and AC-10 flip mechanism are genuine scope/deployment decisions, not resolvable from code alone.

### Resolved Disagreements
- **Calibration feasibility:** Claude (with Codex's first-pass finding) flagged that `calibrate.py` loads the full model on one device in bf16, which cannot fit V3.2. User clarified V3.2 is FP8 on disk and chose to make the calibration load FP8-sharded across one node — resolving the blocker without a second node. Rationale: native FP8 (~671GB) fits in 8×143GB.
- **AC-0 call-site coverage:** Codex (round 1) noted the MHA_ONE_SHOT call site in `forward_mha.py` was missing; verified and added as the fourth site. Rationale: short dense prefills route through MHA_ONE_SHOT and would otherwise never write labels.
- **AC-4 negative tests:** Codex noted the loader cannot reject a random-but-hash-valid mask (no provenance field). Resolved by splitting AC-4 into loader-enforceable negatives plus an artifact-review provenance note backed by AC-1.1. Rationale: verified `channel_mask.py` metadata keys carry no provenance.
- **AC-1.1 field names:** Codex noted `total_tokens` does not exist; switched to `sparsity_rate`/`selected_tokens`/`dense_fallback`. Rationale: verified against `metrics.py`.
- **Quality smoke gates:** Codex noted the fourth gate `first_8_tokens_divergence == 0` was omitted; added. Rationale: verified four gates in `test_dsv32_quality_smoke.py`.
- **AC-1b tiering:** reconciled to TIER 2, sequenced before the AC-11 sweep. Rationale: the probe sets the operating point the sweep must match.
- **Smoke vs AC-11 confusion:** smoke benchmarks must be explicitly labeled non-AC-11. Rationale: `benchmark.sh` defaults to the full AC-11 shape with a hard publish guard.

### Convergence Status
- Final Status: `converged` (2 convergence rounds; round 1 raised 5 required changes, all incorporated and verified; round 2 returned no required changes, no disagreements, no unresolved items).

## Pending User Decisions

- **DEC-1: Calibration approach.** RESOLVED — Modify `calibrate.py` to load native FP8 weights sharded across one node's 8 GPUs (no bf16 upcast, no single-device pin). Calibration code changes are in scope; single node is sufficient.
  - Decision Status: `RESOLVED — modify calibrate.py for FP8 + sharded single-node load`
- **DEC-2: Serving topology.** RESOLVED — Single-node, sequential: run DS then DSA on node 0's 8 GPUs; the quality smoke captures DSA reference outputs first, then runs DS against them.
  - Decision Status: `RESOLVED — single-node sequential`
- **DEC-3: Smoke benchmark shape.** RESOLVED — Explicit `TRIALS=1` + shortened `MEASUREMENT_WINDOW_S`, clearly labeled non-AC-11.
  - Decision Status: `RESOLVED — TRIALS=1, shortened window, labeled non-AC-11`
- **DEC-4: Smoke comparator.** RESOLVED — Run the smoke comparator now with radix cache OFF on BOTH DS and DSA (radix parity satisfied). The comparator always requires radix parity; deferring was the alternative.
  - Decision Status: `RESOLVED — radix-off both sides, comparator now`
<comment>
claude and codex seem to agree so go with claude's positions of not using environemtn override.
</comment>
<comment>Codex — Additional critique: agree final evidence should not use `SGLANG_DS_RADIX_OVERRIDE`; the pending choice should be narrowed to a CLI/artifact-path or state-file contract that sets `_double_sparsity_radix_fixture_passed` before validation. Leaving the env override as an option conflicts with AC-10's negative test and this DEC-5 user note.</comment>
- **DEC-5: AC-10 radix-flip mechanism.** PENDING — How should `record_radix_fixture_passed` be wired before `validate_double_sparsity`: a CLI flag, a launcher init module, or an environment override?
  - Claude Position: Prefer a launcher/CLI mechanism so the final radix-on boot does not depend on an environment override.
  - Codex Position: Agrees the final radix-on evidence should boot without an environment override; exact mechanism left open.
  - Tradeoff Summary: A CLI/launcher flag is cleaner and auditable but needs launcher plumbing; an env override is fastest but is explicitly disallowed for the final AC-10 boot. Decide during M3 Phase A (task10).
  - Decision Status: `PENDING`
- **DEC-6: MODEL_PATH pinning.** RESOLVED — Pin `MODEL_PATH` to `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2` for both serve scripts (the default HF id would trigger a download / wrong revision).
  - Decision Status: `RESOLVED — pin MODEL_PATH to the cluster weights`
- **DEC-7: AC-11 performance metrics.** RESOLVED — DS TPS within 5% of DSA and DS P99 TTFT ≤ 1.10x DSA are directional targets; a miss is recorded as an AC-11 failure requiring follow-up tuning, not an outright build-break. (Quality-smoke and AC-12 thresholds remain hard pass/fail gates.)
  - Decision Status: `RESOLVED — directional targets`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Phase", "Step", "Tier", "DEC-", or similar workflow markers. These belong in this plan document, not in the resulting codebase.
- Use descriptive, domain-appropriate naming in code (e.g., refer to "radix capture extend snapshot", "FP8 sharded calibration load", "chunked-prefill probe" by their behavior, not by AC numbers).
- Note: the project's existing in-code markers (e.g., `AC-10-FIXTURE-MARKER`) and AC-named artifacts predate this plan and are retained only where they already exist for traceability; do not introduce new AC-named identifiers in code.

--- Original Design Draft Start ---

# Loop 5 Draft — Double Sparsity MVP on H200

## Objective

Get a **demonstrable Double Sparsity (DS) MVP** running end-to-end on
the 2-node H200 cluster as fast as possible, without confusing a
hardware smoke milestone with the loop4-complete MVP.

There are two deliverables:

1. **Smoke MVP:** DS-on DeepSeek-V3.2 (FP8) serves real requests on
   H200, produces non-trivial DS selection, has one DS benchmark JSON
   + one DSA benchmark JSON, and passes the paired quality smoke.
2. **Loop4-compatible MVP:** the smoke milestone plus the loop4
   requirements needed to claim comparable default-cookbook behavior:
   TP=8, FP8 KV, page size 64, CUDA graphs represented, chunked
   prefill probed, radix cache enabled for the final run, DSA baseline
   captured with matching knobs, AC-11 comparator run, and AC-12 full
   quality gate run.

If AC-10 radix, AC-11 comparator, or AC-12 full quality are missing,
the result is a useful smoke milestone, not the minimal viable working
version requested by loop4.

## Why a new loop

Loop 4 built deep code-tier scaffolding (comparator validation
gauntlet, bench_serving timing path, M3-B fixture infrastructure,
AC-12 harness, validator helpers) but never executed against
hardware — even though `CLUSTER.md` advertised an 8× H200 local +
8× H200 remote setup the entire time. The CPU-only loop drifted
because the remaining ACs were all hardware-gated and I kept
adding fixture code instead of running the existing code.

The critical artifact `/models/dsv32-fp8-channel-mask.safetensors`
**does not exist on disk**. Generating it unblocks every DS-on AC.
That single missing file is the actual root blocker.

## Hardware (per `CLUSTER.md` + auto-memory)

- Node 0 (local): 8× H200, hostname `h200-10-220-51-16`. Verified
  via `nvidia-smi`: 8 GPUs × 143 GB free.
- Node 1 (remote): 8× H200, hostname `h200-10-220-51-5`. Access via
  `rx devbox run double-sparsity --rank 1 -- <cmd>`.
- DSv3.2 FP8 weights: `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`.
- Default ports: workers 30001, router 30000, prometheus 29000.
- Logs: node 0 `/sgl-workspace/sglang/development/logs/`;
  node 1 `/tmp/sgl_logs/`.

## MVP scope — IN

0. **Close the Round 38 AC-10 producer bug before claiming radix-on.**
   The current capture producer path is unreachable: `_write_token_labels`
   does not accept `forward_batch`, but the capture branch references it
   and then hides the failure. Fix this first:
   - update `_write_token_labels(..., forward_batch: Optional[ForwardBatch] = None)`;
   - pass the live `forward_batch` at the extend, decode, and TRT-LLM
     call sites;
   - keep token-label writes first and publish radix capture only when
     `forward_batch` is present and the mode is extend;
   - add the producer-side regression required by Round 38;
   - verify `/generate` exposes non-empty
     `meta_info["double_sparsity_radix_capture"]` when
     `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`.

1. **Generate the channel mask** (`task-ac4-hwrun`). Single GPU,
   `--tp 1`, ~15–30 min wall-clock. Unblocks every DS-on AC.
   Output: `/models/dsv32-fp8-channel-mask.safetensors`. Validate
   `shape=[L, H, 16]`, `dtype=fp8_e4m3`, `head_dim=128`,
   `page_size=64`, `label_dim=16`.

2. **DS boot smoke** (`task-ac1-hwtest`). Launch
   `serve_double_sparsity.sh` on local 8× H200 TP=8 with the new
   mask; issue one `/generate` request; confirm the server returns
   text and the token-label table populates from the production
   `_write_token_labels` hook (the env-gated capture log built in
   Round 36–38 is the easiest probe: set
   `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, send the request, read
   `meta_info["double_sparsity_radix_capture"]` for non-empty
   `per_token_slot_sha` and `per_layer_written_all_true=True`).

3. **DSA + DS benchmark pair** (`task-ac8-server` + `task-ac9-baseline`).
   - DSA baseline: boot `serve_native_nsa.sh`, run
     `development/benchmark_baseline.sh` with the locked Option B
     flags at conc 16 / 32 / 64. ~30 min total.
   - DS run: boot `serve_double_sparsity.sh`, run
     `development/benchmark.sh` with the same operating point.
     ~30 min total.
   - A radix-off DS run is allowed only as a smoke/debug run. The
     final loop4-compatible MVP run must close AC-10 and run DS and
     DSA with radix cache enabled.
   - A single trial is allowed only for the smoke milestone. The final
     comparable-performance run uses the AC-11 shape: conc 16 / 32 /
     64, 3 trials, 120s warmup, 600s measurement window, median
     comparison.

4. **Quality smoke** (`task-ac8-quality`). Boot both servers
   simultaneously on different ports; run
   `test/manual/test_dsv32_quality_smoke.py` to compare DS-on vs
   DSA outputs on 20 deterministic prompts. Gates: prefix-match
   ≥ 0.80, ROUGE-L ≥ 0.85, NIAH-mini 4/5. ~5 min.

That is the smoke MVP. Demonstrable: one DS benchmark JSON, one DSA
benchmark JSON, one quality smoke artifact, side-by-side. Enough
to say "DS works on V3.2 FP8 with comparable quality at the
locked Option B operating point."

The loop4-compatible MVP additionally requires radix-on final serving,
AC-11 comparator evidence, and AC-12 full quality evidence.

## Smoke-only items that are NOT enough for loop4 MVP

These are allowed to defer only for the smoke milestone. They are not
allowed to remain deferred when claiming the loop4-compatible MVP:

- **AC-10 radix-cache flip.** Radix-off is acceptable for first boot
  and bench smoke only. To claim default-cookbook comparable behavior,
  run the M3-B fixtures, prove producer capture works, flip the guard,
  remove `--disable-radix-cache` from the final DS launch, and run the
  final comparator with radix cache on.
- **AC-11 directional comparator.** Single-trial bench_serving runs
  size the gap but do not prove comparable speed/performance. The final
  result needs the 3-trial DSA + DS sweep at conc 16 / 32 / 64 with
  120s warmup, 600s measurement, medians, DS TPS within 5% of DSA, and
  DS P99 TTFT no worse than 1.10x DSA.
- **AC-6 CUDA-graph capture validation.** Eager-mode DS is useful for
  diagnosis, but the client asks for performant knobs. The final bundle
  must record whether CUDA graphs are enabled and must include a clear
  exception if capture cannot be used.
- **AC-1b chunked-prefill probe.** Run and record the probe. If it
  passes, keep the default chunked-prefill setting. If it fails, disable
  it on both DS and DSA for apples-to-apples evidence and file the
  follow-up.
- **AC-12 full NIAH 4K/16K/64K + MMLU 5-shot.** The 5-minute quality
  smoke gates whether to continue, but the full gate is required before
  declaring loop4 MVP complete.

## Critical path (concrete commands)

```bash
# 0. Sanity
nvidia-smi --query-gpu=index,name,memory.free --format=csv

# 0a. Before radix-on claims
# Patch the Round 38 AC-10 producer bug and verify:
#   SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 /generate returns
#   meta_info["double_sparsity_radix_capture"] with non-empty
#   per-token and per-layer evidence.

# 1. Channel mask (~15-30 min, single GPU)
mkdir -p /models
python -m sglang.srt.layers.attention.double_sparsity.calibrate \
    --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2 \
    --dtype bfloat16 \
    --kv-cache-dtype fp8_e4m3 \
    --tp 1 \
    --output /models/dsv32-fp8-channel-mask.safetensors \
    --label-dim 16 \
    --page-size 64 \
    --num-samples 256 \
    --block-size 512 \
    --seed 42 \
    -v 2>&1 | tee /sgl-workspace/sglang/development/logs/calibrate_$(date +%Y%m%d-%H%M%S).log

# 2. Validate mask artifact
python -c "
from sglang.srt.layers.attention.double_sparsity.channel_mask import load_channel_mask
m = load_channel_mask('/models/dsv32-fp8-channel-mask.safetensors')
print(f'dtype={m.dtype} head_dim={m.head_dim} page_size={m.page_size} label_dim={m.label_dim}')
print(f'channel_selection.shape={tuple(m.channel_selection.shape)}')
print(f'content_sha256[:12]={m.content_sha256[:12]}')
"

# 3. DS boot smoke (~5 min, separate terminal)
SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 \
  bash development/serve_double_sparsity.sh &
# Wait for /health on :30000, then:
curl -s -X POST http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello from DS", "sampling_params": {"temperature": 0.0, "max_new_tokens": 32}}' \
  | python -c "import sys,json; r=json.load(sys.stdin); print(r['text'][:200]); print('capture:', bool(r.get('meta_info',{}).get('double_sparsity_radix_capture')))"

# 4. DSA baseline bench (~10-30 min)
MODE=native_nsa CONCURRENCIES="16 32 64" \
  bash development/benchmark_baseline.sh
# (After it finishes, kill the DSA server. Boot DS server.)

# 5. DS bench
MODE=double_sparsity CONCURRENCIES="16 32 64" \
  bash development/benchmark.sh

# 6. Two-column comparator (single trial, --baseline / --ds form)
python development/benchmark_compare.py \
  --baseline development/results/native_nsa_gsp_isl4096_osl512_c64_t1.jsonl \
  --ds       development/results/double_sparsity_gsp_isl4096_osl512_c64_t1.jsonl \
  --output development/results/mvp_compare.md

# 7. Quality smoke (both servers up simultaneously on different ports)
DS_BASE_URL=http://127.0.0.1:30000 \
DSA_BASE_URL=http://127.0.0.1:30001 \
  pytest test/manual/test_dsv32_quality_smoke.py -v

# 8. Final loop4-compatible comparator, after AC-10 radix flip
# Ensure the DS launcher no longer passes --disable-radix-cache.
TRIALS=3 WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 \
MODE=native_nsa CONCURRENCIES="16 32 64" \
  bash development/benchmark_baseline.sh

TRIALS=3 WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 \
MODE=double_sparsity CONCURRENCIES="16 32 64" \
  bash development/benchmark.sh

# 9. Full quality gate
DS_BASE_URL=http://127.0.0.1:30000 \
DSA_BASE_URL=http://127.0.0.1:30001 \
  pytest test/manual/test_double_sparsity_v32.py -v
```

## Acceptance evidence — what "MVP done" looks like

A single directory `/sgl-workspace/sglang/runs/<date>_dsv32_mvp/`
containing:

- `calibrate.log` + `dsv32-fp8-channel-mask.safetensors` validation
  output.
- `serve_*.log` for both DS and DSA boots (showing no crashes,
  validator accepted, all 8 GPUs visible).
- Branch and commit SHA.
- Full server args from `/get_server_info`.
- Knob evidence: TP value, `kv_cache_dtype=fp8_e4m3`, `page_size=64`,
  CUDA graph status, radix cache status, chunked-prefill setting,
  DS config path, mask content hash, and whether overlap scheduling /
  piecewise CUDA graph remain disabled under Option B.
- Six bench JSONLs: `native_nsa_*c{16,32,64}_t1.jsonl` and
  `double_sparsity_*c{16,32,64}_t1.jsonl` plus matching `.meta.json`
  sidecars.
- `mvp_compare.md` from `benchmark_compare.py` (single-trial
  AC-7/AC-8 report — TPS, TTFT, no-op detector).
- `dsv32_quality_smoke_*.json` with prefix-match / ROUGE-L /
  NIAH-mini numbers for the paired DS/DSA run.
- Final loop4-compatible evidence, when claiming MVP complete:
  - radix-on DS and DSA launch evidence;
  - AC-11 3-trial comparator artifacts and pass/fail summary;
  - AC-12 NIAH 4K/16K/64K + MMLU 5-shot artifacts and pass/fail
    summary.

The smoke narrative: "DS-on V3.2 FP8 serves at the locked Option B
operating point. Side-by-side with DSA at conc 16/32/64. Quality
smoke passes on 20 paired prompts. Here's the bench JSON and the
comparator report."

The loop4 MVP narrative adds: "The final run used matching production
knobs, including radix cache enabled; CUDA graph and chunked-prefill
status are recorded; AC-11 comparator and AC-12 quality gates are
complete."

## Risks + likely failure modes

1. **Calibrate OOMs at TP=1.** Mitigation: bump to TP=2 with
   `--tp 2 --gpus 0,1`. The calibrate module accepts both.
2. **DS server fails the validator's DEC-2 guard.** Expected:
   the launcher already passes `--disable-radix-cache`, so the
   guard accepts. If it fails on something else (mask hash, page
   size pairing), read the validator error verbatim.
3. **bench_serving crashes on DS selection.** This would mean
   the production `_write_token_labels` hook is buggy on hardware
   (despite passing CPU unit tests). Round 18–20 work claims it's
   wired through `ForwardContext`; the boot smoke (step 3) catches
   this before the bench.
4. **Quality smoke prefix-match ≪ 0.80.** Could indicate either
   (a) DS labels are bad → re-check calibrate output, or
   (b) prompts are too short / sensitive. Investigate by running
   ROUGE-L only on long-output prompts first.
5. **TPS gap > 5%.** Acceptable for the smoke milestone. Not
   acceptable for the loop4-compatible MVP unless the artifact is
   explicitly reported as an AC-11 failure requiring follow-up tuning.

## Loop-runner notes

- Single mainline objective per round: the *next concrete
  command from the critical path above*. No more multi-day fixture
  refactors.
- Each round produces ARTIFACTS in `runs/<date>_dsv32_mvp/`, not
  just code changes. If a round did not produce an artifact, it
  stalled.
- The existing loop-4 code stays as-is unless a specific bench
  failure mode requires patching it. The Round 38 AC-10 producer
  bug is the one known exception that must be patched before a
  radix-on or default-cookbook parity claim.

--- Original Design Draft End ---
