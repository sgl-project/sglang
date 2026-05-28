# Refine Plan QA

## Summary

This refinement processed **19 comment blocks** added to `development/loop5/plan.md` during a two-stage annotated review (a pensieve/Linus-style pass and a Codex adjudication pass, plus one user note). Classification: **0 questions, 18 change requests, 1 research request**. All 19 were resolved — the research request was answered by reading the actual code, every change request was applied to the refined plan, and the user's note resolved the one outstanding decision (DEC-5). The most consequential edits were: collapsing the analyze-only "design the AC-10 flip" task into direct implementation (removing the Loop-4-style drift trap), correcting the calibration framing (the load IS a bare HF `from_pretrained`, so `device_map="auto"` is the right hook; the real risk is FP8 block-quant shard-loading), moving the CUDA-graph evidence to the first boot, relabeling the smoke-comparator task as a consumer of AC-8/AC-9, and pinning down `MODEL_PATH` override. No pending decisions remain; convergence status is `converged`.

## Comment Ledger

| CMT-ID | Classification | Location | Original Text (excerpt) | Disposition |
|--------|----------------|----------|-------------------------|-------------|
| CMT-1 | change_request | Goal Description | "The plan correctly names the channel mask as the root blocker, then immediately introduces a second prerequisite..." | applied |
| CMT-2 | change_request | Goal Description | "Codex — Agree. Code confirms the AC-0 producer bug is capture-specific..." | applied |
| CMT-3 | change_request | Goal Description | "Codex — Additional critique: AC-4 says native FP8 sharded load, but the current calibration forward loop still calls model(input_ids=block.to(model.device))..." | applied |
| CMT-4 | change_request | Acceptance Criteria | "Codex — Additional critique: this is not just a run step yet, because serve_double_sparsity.sh still defaults MODEL_PATH..." | applied |
| CMT-5 | change_request | Acceptance Criteria | "Codex — Additional critique: AC-6 should distinguish regular CUDA graph from piecewise CUDA graph..." | applied |
| CMT-6 | change_request | Acceptance Criteria | "The Feasibility Hint (step 5) says 'design and implement the AC-10 radix-flip mechanism.' That phrasing — design, then implement — is the exact pattern that caused Loop 4 to stall..." | applied |
| CMT-7 | change_request | Acceptance Criteria | "Codex — Partially agree. I agree the analyze wording invites drift, but the 'few lines' premise is too glib..." | applied |
| CMT-8 | change_request | Task Breakdown | "task2 is tagged coding but it's a hardware run that produces an artifact, not a code change..." | applied |
| CMT-9 | change_request | Task Breakdown | "Codex — Partially agree. The graph really does allow task3/task4 before task2... I would not make calibration depend on AC-0..." | applied |
| CMT-10 | research_request | Task Breakdown | "The device_map=\"auto\" approach mentioned in the Feasibility Hints is speculative — calibrate.py likely uses its own model-loading routine..." | researched |
| CMT-11 | change_request | Task Breakdown | "Codex — Disagree with the factual premise. The actual load is a bare AutoModelForCausalLM.from_pretrained(...device_map={\"\": \"cuda\"...})..." | applied |
| CMT-12 | change_request | Task Breakdown | "task8 is labeled AC-8 but it's the comparator output, which is distinct from the DS benchmark artifact that AC-8 actually requires..." | applied |
| CMT-13 | change_request | Task Breakdown | "Codex — Agree. benchmark.sh and benchmark_baseline.sh produce the benchmark JSONL plus .meta.json sidecars, while benchmark_compare.py consumes..." | applied |
| CMT-14 | change_request | Task Breakdown | "This is the exact drift the draft warned about. Loop 4 stalled on 'analyze' work that never produced hardware artifacts. task10 is a pure design task..." | applied |
| CMT-15 | change_request | Task Breakdown | "Codex — Partially agree. Collapse the analyze-only task or require a written decision artifact, yes. But current code is not just a flag flip..." | applied |
| CMT-16 | change_request | Task Breakdown | "task14 depends on task11 (radix flip), but CUDA-graph status is independent of whether radix cache is on or off..." | applied |
| CMT-17 | change_request | Task Breakdown | "Codex — Agree. ModelRunner.init_device_graphs() captures regular CUDA graphs during model-runner initialization whenever disable_cuda_graph is false..." | applied |
| CMT-18 | change_request | Pending User Decisions | "claude and codex seem to agree so go with claude's positions of not using environemtn override." | resolved |
| CMT-19 | change_request | Pending User Decisions | "Codex — Additional critique: agree final evidence should not use SGLANG_DS_RADIX_OVERRIDE; the pending choice should be narrowed..." | applied |

## Answers

No `question`-type comments were present. All comments were directive (change requests) or investigative (one research request). See **Research Findings** and **Plan Changes Applied**.

## Research Findings

### CMT-10: Verify the actual calibration model-load site before prescribing `device_map`

**Original Comment:**
```
The `device_map="auto"` approach mentioned in the Feasibility Hints is speculative — `calibrate.py` likely uses its own model-loading routine tied to SGLang's engine, not a bare HuggingFace `from_pretrained` call. If the calibration entrypoint doesn't go through the standard `AutoModel` path, `device_map="auto"` does nothing and you're back to OOM on load. task3 needs to verify the actual load call site in `calibrate.py` before proposing a fix, or you'll code-change the wrong thing and not discover it until task4 blows up on hardware. Maxim: prefer-pragmatic-solutions — read the code before prescribing the patch.
```

**Research Scope:**
Read the model-load site in `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` (the `AutoConfig`/`AutoModelForCausalLM` block and the calibration forward loop). Cross-checked against the Codex finding in CMT-11.

**Findings:**
The premise is incorrect. The load is a **bare** `AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map={"": "cuda" if torch.cuda.is_available() else "cpu"}, trust_remote_code=True)` following an `AutoConfig.from_pretrained(...)` — it does NOT go through an SGLang engine loader. Therefore `device_map="auto"` IS the correct hook (it routes through HF/Accelerate dispatch). The real risks are different: (a) whether HF can shard-load the DeepSeek FP8 block-quantized checkpoint without upcasting to bf16, and (b) the calibration forward loop's single-device assumption (`model(input_ids=block.to(model.device))`), which must be fixed once modules are dispatched across multiple GPUs.

**Impact on Plan:**
The calibration framing in `AC-4`, `task3`, and Feasibility Hint step 2 was rewritten to state the load path explicitly, add a one-block dry-run that logs parameter dtypes and device placement (proving no upcast) before the full run, and flag the `model.device` forward-loop fix. The original speculative wording was removed.

## Plan Changes Applied

### CHANGE-A (CMT-1, CMT-2): AC-0 scope clarification + artifact-per-round

**Original Comments:**
```
CMT-1: The plan correctly names the channel mask as the root blocker, then immediately introduces a second prerequisite (the AC-0 producer fix) that is gated on radix-on claims only. But task1 (AC-0) and task3 (calibration load change) are shown as independent, meaning a round could open with the AC-0 fix and spend the entire round on code changes before any hardware is touched. The draft's explicit instruction is "the existing loop-4 code stays as-is unless a specific bench failure mode requires patching it" and "the Round 38 AC-10 producer bug is the one known exception." The plan should make explicit that AC-0 is a prerequisite for radix-on claims only, not for the mask generation or the boot smoke. If a round starts by working on the producer fix, it must also produce a hardware artifact that round — the fix alone is not a complete round. The current framing doesn't enforce this; "coding" tag is not the same as "artifact produced."
```
```
CMT-2: Codex — Agree. Code confirms the AC-0 producer bug is capture-specific: `dsa_backend._write_token_labels` writes the token-label table first, then references undefined `forward_batch` only inside the `SGLANG_DS_RADIX_FIXTURE_CAPTURE` branch. Mask generation and a radix-off boot do not consume that capture path, so AC-0 should gate radix evidence/default parity, not calibration; any AC-0 round still needs a hardware artifact to avoid repeating Loop 4 drift.
```

**Changes Made:**
Added a scope sentence to AC-0 stating it gates radix-on evidence and default-cookbook parity claims ONLY — mask generation (AC-4) and the radix-off boot smoke (AC-1) do not consume the capture path and may proceed in parallel — plus the rule that an AC-0 round must still produce a hardware artifact (the `coding` tag alone does not satisfy artifact-per-round).

**Affected Sections:**
- Acceptance Criteria: AC-0 scope note added.
- Dependencies and Sequence: dependency note now states calibration runs in parallel and does not depend on AC-0.

**Cross-Reference Updates:** none (no ID changes).

### CHANGE-B (CMT-3, CMT-10, CMT-11): Calibration load framing, dry-run, and FP8 shard-load risk

**Original Comments:**
```
CMT-3: Codex — Additional critique: AC-4 says native FP8 sharded load, but the current calibration forward loop still calls `model(input_ids=block.to(model.device))`, which is a single-device assumption that can break once HF dispatch shards modules. Task3 should produce a tiny one-block calibration log that records parameter dtypes and device placement before spending a full 256-block run.
```
```
CMT-11: Codex — Disagree with the factual premise. The actual load is a bare `AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map={"": "cuda" if torch.cuda.is_available() else "cpu"}, trust_remote_code=True)` after `AutoConfig`, so `device_map="auto"` would hit HF/Accelerate rather than an SGLang engine loader. The real risk is different: task3 must prove HF can shard-load the DeepSeek FP8 block-quantized checkpoint without upcasting, and must fix the later `model.device` forward-loop assumption if modules are dispatched across GPUs.
```
(CMT-10 verbatim text is in **Research Findings** above.)

**Changes Made:**
Rewrote AC-4 to name the actual load call, the `device_map="auto"` HF/Accelerate path, the two real risks (FP8 block-quant shard-load without upcast; `model.device` forward-loop assumption), and added a positive test requiring a one-block dry-run that logs dtypes/device placement before the full run. Updated Feasibility Hint step 2 and the task3 description to match.

**Affected Sections:**
- Acceptance Criteria: AC-4 description + new positive test.
- Feasibility Hints and Suggestions: step 2 rewritten.
- Task Breakdown: task3 description updated.

**Cross-Reference Updates:** none.

### CHANGE-C (CMT-4): MODEL_PATH override before first boot

**Original Comment:**
```
Codex — Additional critique: this is not just a run step yet, because `development/serve_double_sparsity.sh` still defaults `MODEL_PATH` to `deepseek-ai/DeepSeek-V3.2`, while DEC-6 requires `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`. Task5 must edit or override that before first boot, otherwise `/get_server_info` can describe a different revision/download path than the artifact bundle claims.
```

**Changes Made:**
AC-1 now states the serve script defaults `MODEL_PATH` to the HF id and MUST be overridden to the cluster path before first boot (per DEC-6), with a positive test that `/get_server_info` reports the cluster path (not the HF-id default). task5 description updated to "Override the serve script's MODEL_PATH default...".

**Affected Sections:**
- Acceptance Criteria: AC-1 description + positive test.
- Task Breakdown: task5 description.

**Cross-Reference Updates:** none.

### CHANGE-D (CMT-5, CMT-16, CMT-17): AC-6 regular-vs-piecewise CUDA graph + capture at first boot

**Original Comments:**
```
CMT-5: Codex — Additional critique: AC-6 should distinguish regular CUDA graph from piecewise CUDA graph. Option B passes `--disable-piecewise-cuda-graph`, but regular `disable_cuda_graph` remains false by default and `ModelRunner.init_device_graphs()` is the boot-time capture path; evidence that only says piecewise is disabled does not prove regular capture/replay status.
```
```
CMT-16: task14 depends on task11 (radix flip), but CUDA-graph status is independent of whether radix cache is on or off. You can record this status immediately after the first DS boot in task5 — that's when the server either captures graphs or logs why it can't. Deferring it to M3 means you have no CUDA-graph evidence until after the AC-10 flip, which adds an unnecessary wait and a round of work that could have been a one-liner artifact in task5. This is scope-creep-by-dependency-ordering. Move task14 to depend on task5.
```
```
CMT-17: Codex — Agree. `ModelRunner.init_device_graphs()` captures regular CUDA graphs during model-runner initialization whenever `disable_cuda_graph` is false, before any request and independent of radix cache; the DS launcher disables piecewise CUDA graph, not regular CUDA graph. Replay/use is then visible through scheduler logs and metrics via `can_run_cuda_graph`, so the first task5 boot plus smoke request can already record capture and replay status.
```

**Changes Made:**
AC-6 rewritten to require the REGULAR CUDA-graph capture/replay status (distinct from disabled piecewise), observable at first boot via `init_device_graphs()` and `can_run_cuda_graph` metrics. The CUDA-graph evidence task now depends on the first boot (task5), not the radix flip, and was moved into M2. Milestones updated.

**Affected Sections:**
- Acceptance Criteria: AC-6 rewritten (positive/negative tests).
- Dependencies and Sequence: AC-6 evidence moved to M2 Phase A; removed from M3 Phase C.
- Task Breakdown: CUDA-graph task now `task10`, depends on `task5`.

**Cross-Reference Updates:** the CUDA-graph task moved from old `task14` to new `task10`; the evidence-bundle task (now `task15`) updated to depend on `task10` for the AC-6 artifact.

### CHANGE-E (CMT-12, CMT-13): Relabel the smoke-comparator task as an AC-8/AC-9 consumer

**Original Comments:**
```
CMT-12: task8 is labeled AC-8 but it's the comparator output, which is distinct from the DS benchmark artifact that AC-8 actually requires. This is a wrong AC assignment — the comparator is the AC-8/AC-9 *consumer*, not its target. Minor, but in a plan where artifact labeling is the whole point of the smoke-vs-loop4 distinction, mislabeling here will cause confusion during the evidence-bundle assembly in task16.
```
```
CMT-13: Codex — Agree. `development/benchmark.sh` and `development/benchmark_baseline.sh` produce the benchmark JSONL plus `.meta.json` sidecars, while `development/benchmark_compare.py` consumes `--baseline` and `--ds` JSONLs and emits a report. task8 should target comparator/report evidence, not AC-8 alone; AC-8/AC-9 are task7's producer artifacts.
```

**Changes Made:**
task8 retargeted from `AC-8` to `AC-8, AC-9` and its description rewritten to "Smoke comparator report (`mvp_compare.md`) CONSUMING task7's DS+DSA JSONLs" — clarifying it is the consumer, while task7 remains the AC-8/AC-9 producer.

**Affected Sections:**
- Task Breakdown: task8 description and Target AC.

**Cross-Reference Updates:** none (task7 remains the producer).

### CHANGE-F (CMT-6, CMT-7, CMT-14, CMT-15): Collapse the AC-10 "design" task into direct implementation

**Original Comments:**
```
CMT-6: The Feasibility Hint (step 5) says "design and implement the AC-10 radix-flip mechanism." That phrasing — design, then implement — is the exact pattern that caused Loop 4 to stall. The AC text itself is correctly outcome-focused ("boots radix-on WITHOUT relying on an environment override"), but combining it with a dedicated analyze task (task10) creates a scaffolding trap. The validator guard flip is a few lines in `validator.py` and a launcher flag. There is no mechanism here worth a separate design round — the design is: pass a CLI flag, the validator reads it, done. Maxim: eliminate-special-cases — the "design" step is a special case that should not exist; just implement it directly in task11.
```
```
CMT-7: Codex — Partially agree. I agree the analyze wording invites drift, but the "few lines" premise is too glib: `validate_double_sparsity()` reads a transient `server_args._double_sparsity_radix_fixture_passed` before boot, `record_radix_fixture_passed()` only sets that in-process attribute and logs an optional artifact SHA, and `serve_double_sparsity.sh` has no CLI/state hook today. That is real ServerArgs/launcher plumbing, not just deleting `--disable-radix-cache`.
```
```
CMT-14: This is the exact drift the draft warned about. Loop 4 stalled on "analyze" work that never produced hardware artifacts. task10 is a pure design task with no artifact output — it doesn't even produce a `runs/` entry. The draft says "single mainline objective per round = the next concrete command." The AC-10 radix-flip mechanism is not architecturally ambiguous: the validator reads a flag or a state file, the launcher sets it, done. You don't need a design round for this; you need a commit and a boot. Collapse task10 into task11 or, at minimum, require task10 to produce a concrete proposal artifact (a written decision, not just "we thought about it") so it doesn't become a multi-round analysis spiral. Maxim: prefer-pragmatic-solutions — don't design what you can just implement.
```
```
CMT-15: Codex — Partially agree. Collapse the analyze-only task or require a written decision artifact, yes. But current code is not just a flag flip: `ServerArgs` has no fixture-passed CLI field, `validate_double_sparsity()` runs during `check_server_args()`, and the helper's state is not persisted across processes. A clean no-env path needs a specific ServerArgs/launcher/artifact contract.
```

**Changes Made:**
Removed the analyze-only "design the AC-10 flip" task entirely and folded its scope into a single direct implementation task. The implementation task now specifies the concrete contract Codex identified: add a ServerArgs/launcher field (or state-file/artifact-path) that sets `_double_sparsity_radix_fixture_passed` before `validate_double_sparsity` runs in `check_server_args()`; pass both fixtures; remove `--disable-radix-cache`. Feasibility Hint step 5 changed from "design and implement" to "implement ... directly (no separate design round)". After collapsing, the entire Task Breakdown was renumbered to stay contiguous (task1–task15). The plan now has no `analyze` tasks, which is intentional (the only analyze task was the drift trap being removed).

**Affected Sections:**
- Feasibility Hints and Suggestions: step 5.
- Dependencies and Sequence: M3 Phase A wording.
- Task Breakdown: old `task10` removed; old `task11`→`task11` (implementation, now depends on `task2`); all subsequent tasks renumbered.
- Pending User Decisions: DEC-5 implementation note (see CHANGE-H).

**Cross-Reference Updates:** task IDs renumbered after the collapse — old task11→task11 (impl, dep task2), old task12→task12, old task13→task13, old task14→task10 (CUDA graph), old task15→task14, old task16→task15. All `Depends On` references and milestone text were updated to the new IDs.

### CHANGE-G (CMT-8, CMT-9): task2 as the explicit M2 artifact gate; calibration stays parallel

**Original Comments:**
```
CMT-8: task2 is tagged `coding` but it's a hardware run that produces an artifact, not a code change. If the probe fails here, task3 is unblocked and task4 depends on task3, so you'd start calibration before you've confirmed the producer fix holds. The dependency graph should show task4 (and therefore task5) depending on a passing task2, not just task1. Right now the plan lets a broken AC-0 silently coexist with a running calibration.
```
```
CMT-9: Codex — Partially agree. The graph really does allow task3/task4 before task2 because task3 has no dependency and task4 depends only on task3, and task2 is an artifact-producing hardware run mislabeled as coding. I would not make calibration depend on AC-0, though: `calibrate.py` and `load_channel_mask()` do not consume the radix capture path, and task5 already gates first DS boot on both task2 and task4. The cleaner fix is to let mask generation proceed in parallel while making task2 an explicit artifact gate for M2.
```

**Resolution / Changes Made:**
Codex's resolution was adopted over the stricter CMT-8 proposal: calibration (task3/task4) is NOT made dependent on AC-0, because the calibration and `load_channel_mask` paths do not consume the radix capture. Instead, task2's description now marks it as "the explicit artifact gate into M2," and first DS boot (task5) already gates on both task2 (passing capture probe) and task4 (validated mask). The Dependencies note states this explicitly. The routing tag stays `coding` because the schema permits only `coding`/`analyze`; the "hardware-run produces an artifact" nuance is captured in the description and the artifact-per-round rule rather than a new tag.

**Affected Sections:**
- Task Breakdown: task2 description.
- Dependencies and Sequence: dependency note.

**Cross-Reference Updates:** none.

### CHANGE-H (CMT-18, CMT-19): Resolve DEC-5 — no environment override

**Original Comments:**
```
CMT-18: claude and codex seem to agree so go with claude's positions of not using environemtn override.
```
```
CMT-19: Codex — Additional critique: agree final evidence should not use `SGLANG_DS_RADIX_OVERRIDE`; the pending choice should be narrowed to a CLI/artifact-path or state-file contract that sets `_double_sparsity_radix_fixture_passed` before validation. Leaving the env override as an option conflicts with AC-10's negative test and this DEC-5 user note.
```

**Changes Made:**
DEC-5 flipped from PENDING to RESOLVED (user decision): no environment override; the flip is wired via a ServerArgs/launcher field or a state-file/artifact-path contract that sets `_double_sparsity_radix_fixture_passed` before `validate_double_sparsity` runs in `check_server_args()`. Recorded the implementation-plumbing note and that the work happens directly in the implementation task (no separate design round).

**Affected Sections:**
- Pending User Decisions: DEC-5 status and body.
- Claude-Codex Deliberation: Convergence Status updated to note DEC-5 resolution; a refinement bullet added to Resolved Disagreements.

**Cross-Reference Updates:** none.

## Remaining Decisions

None. All decisions DEC-1 through DEC-7 are RESOLVED (DEC-5 resolved by the user during this refinement). No items require further user input.

## Refinement Metadata

- **Input Plan:** development/loop5/plan.md
- **Output Plan:** development/loop5/refined_plan_v1.md
- **QA Document:** .humanize/plan_qa/plan-qa.md
- **Total Comments Processed:** 19
  - Questions: 0
  - Change Requests: 18
  - Research Requests: 1
- **Plan Sections Modified:** Goal Description, Acceptance Criteria, Feasibility Hints and Suggestions, Dependencies and Sequence, Task Breakdown, Claude-Codex Deliberation, Pending User Decisions
- **Convergence Status:** converged
- **Refinement Date:** 2026-05-28
