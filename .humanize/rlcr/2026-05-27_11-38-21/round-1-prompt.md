Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop4/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop4/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-1-contract.md

Your round contract must contain:
- Exactly one **mainline objective**
- The 1-2 target ACs for this round
- Which issues are truly **blocking** that mainline objective
- Which issues are **queued** and explicitly out of scope
- Concrete success criteria for this round

Do not start implementation until the round contract exists.

## Task Lane Rules

Use the Task system (TaskCreate, TaskUpdate, TaskList) with one required tag per task:
- `[mainline]` for plan-derived work that directly advances this round's objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` work is the round's primary success condition
- `[blocking]` work is allowed only when it truly blocks the mainline objective
- `[queued]` work must be documented but must NOT replace the round objective
- If a new bug does not block the current objective, tag it `[queued]` and keep moving on mainline work

Before executing each task in this round:
1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task
3. Follow selected lesson IDs (or `NONE`) during implementation

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
# Round 0 Review Result

Mainline Progress Verdict: REGRESSED

Round 0 does not close AC-0 or AC-13. The implementation moved files toward token-level naming, but the acceptance-gated state is red: the required package import fails, the 150-test regression suite fails, the selector/adapter domains are inconsistent, and production binding sizes the label table before the KV pool exists.

## Mainline Gaps

1. **AC-13 is false: the 150-test suite does not pass.**

   Evidence: `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` produced `147 passed, 3 failed`.

   Failing tests:
   - `TestSelectTopkIndicesHookBranch::test_ds_branch_returns_topk_indices_via_adapter`: DS branch returns `-1` where the test expects the first mapped token.
   - `TestSanityProbeRealSelector::test_probe_finds_planted_needle`: `startup_sanity_probe` writes token 256 into a 16-slot table.
   - `TestR6Coverage::test_forward_decode_dispatches_to_flashmla_kv`: test still instantiates `NSAMetadata` with `nsa_*` fields; current dataclass is DSA-shaped.

   Required fix: repair these tests and rerun the exact full command above. Do not claim AC-13 until all 150 pass.

2. **AC-0 required public import fails.**

   Plan requires:

   ```python
   from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, token_label_write, retrieve_topk
   ```

   Actual result:

   ```text
   ImportError: cannot import name 'retrieve_topk' from 'sglang.srt.layers.attention.double_sparsity'
   ```

   Evidence: `python/sglang/srt/layers/attention/double_sparsity/__init__.py:21-51` exports `TokenLabelTable` and `token_label_write`, but no `retrieve_topk`.

   Required fix: define or re-export the public `retrieve_topk` API and add a test for the exact import tuple from the plan.

3. **TokenLabelTable is not actually sized from `req_to_token_pool.size` in the normal server construction order.**

   Evidence:
   - `python/sglang/srt/models/deepseek_v2.py:1541-1547` calls `_bind_double_sparsity_runtime_data` during attention construction.
   - `python/sglang/srt/models/deepseek_v2.py:1889-1892` falls back to `ds_parsed.device_buffer_size` when `_ds_req_to_token_pool` is absent.
   - `python/sglang/srt/model_executor/model_runner.py:648-653` only publishes `_ds_req_to_token_pool` before `load_model()` if `self.req_to_token_pool` already exists.
   - `python/sglang/srt/model_executor/model_runner.py:751-752` initializes the memory pool after model loading, so the normal target-worker path reaches attention construction before the pool exists.

   This violates the AC-0 requirement that `max_tokens = req_to_token_pool.size` at bind time, not `device_buffer_size`. Once AC-1 writes real `out_cache_loc` values, any slot above 4095 can write out of range if the fallback table was allocated.

   Required fix: do not allocate or bind the token label table in `DeepseekV2AttentionMLA.__init__`. Construct the selector there, then bind all DS attention modules after `ModelRunner.init_memory_pool()` using the actual `self.req_to_token_pool.size`. Delete the `device_buffer_size` fallback and fail fast if the pool is unavailable.

4. **Selector output is in physical-slot domain, but the adapter treats it as logical sequence positions.**

   Evidence:
   - `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:203-251` computes scores over `token_signatures[layer_id]` with columns `[0..max_tokens)`, i.e. physical KV slots.
   - `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:301-315` returns top-k column indices from that physical slot axis.
   - `python/sglang/srt/layers/attention/double_sparsity/page_table_adapter.py:64-66` interprets those indices as logical positions and gathers `req_to_token[pool, selected]`.

   Reproducer with required non-contiguous slots `[7, 64, 200, 512]`:

   ```text
   selected_from_selector= [[7, 64, 200, 512]] lengths= [4]
   IndexError index 7 is out of bounds for dimension 1 with size 4
   ```

   The selector must return logical positions `[0, 1, 2, 3]`, not physical slots. This is the exact non-contiguous fixture required by AC-0, and it currently fails.

   Required fix: make production selection gather physical labels through `req_to_token[req_pool_idx, logical_pos]`, score a `[bs, max_seq_len]` logical-position score tensor, top-k over logical columns, return logical positions, and only then call `logical_to_physical`.

5. **The live DS path passes latent `q_lora`, but the new scorer requires projected Q-noPE.**

   Evidence:
   - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:226-230` stores `q_lora = q` before `q_b_proj`.
   - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:269-281` computes projected `q` but still passes `q_lora` into `_select_topk_indices`.
   - `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:183-186` rejects anything except `[bs, H, head_dim]`.

   Minimal failure:

   ```text
   ValueError queries must be 3-D [bs, H, head_dim], got shape (2, 1536).
   ```

   Required fix: keep the existing DSA indexer input unchanged, but for DS pass projected Q-noPE `[bs, H_local, 128]` into the selector. In both forward_mla branches, derive `q_nope_for_ds = q[..., :qk_nope_head_dim]` after `q_b_proj` and pass that to `_select_topk_indices` only when `self.use_double_sparsity` is true.

6. **AC-1 through AC-12 remain pending and cannot be treated as complete or deferred.**

   Claude’s summary lists AC-1 onward as remaining. That is accurate as unfinished work, but not a reason to output `COMPLETE`. The original plan’s lower bound still requires AC-0 through AC-8, AC-12, and AC-13; AC-9 through AC-11 are stretch but still tracked.

## Blocking Side Issues

1. **Migrated tests still use stale NSA names and shims.**

   Evidence:
   - `test/registered/unit/layers/attention/test_double_sparsity_unit.py:240-241` and `:283-284` still set `nsa_prefill_backend` / `nsa_decode_backend`, so the validator happy path does not prove `dsa_prefill_backend` / `dsa_decode_backend`.
   - `test/registered/unit/layers/attention/test_double_sparsity_unit.py:3727` and `:3747` import from deprecated `nsa_backend`.

   This directly blocks AC-13 and weakens AC-0 validator coverage.

2. **`dsa_backend.py` hook sites are still unmodified.**

   Evidence: `python/sglang/srt/layers/attention/dsa_backend.py:1439`, `:1637`, and `:2162` still only call `set_mla_kv_buffer`; there is no `token_label_write` hook or `kv_b_proj` K-side projection. This is expected if Round 0 only targeted AC-0, but it remains a blocking prerequisite for AC-1 and every hardware/quality AC.

## Queued Side Issues

1. Code and tests still contain plan-era names/comments such as `AC-*`, `NSA`, `hot_pages`, and `selected_pages` in places that now refer to token-level DS. This is not the next blocker while AC-0/AC-13 are red, but it should be cleaned when the failing tests are repaired so future reviews do not confuse page and token semantics.

2. `token_label_write.py` is a torch gather/write helper, not the Triton-backed write path claimed in the summary. This is acceptable for initial correctness because the plan allows a Python-level hook, but the summary should stop describing it as Triton-backed unless a Triton path lands.

## Required Implementation Plan

Execute this in order; do not start AC-1 hardware or calibration work until the AC-0/AC-13 repair passes.

1. **Repair the AC-0 public/API surface.**
   - Add the plan-required `retrieve_topk` export to `double_sparsity/__init__.py`.
   - Add a unit test for `from ...double_sparsity import TokenLabelTable, token_label_write, retrieve_topk`.
   - Replace stale `nsa_*` backend attributes and deprecated `nsa_backend` imports in the migrated tests with DSA names.

2. **Move DS runtime binding after KV pool initialization.**
   - In `DeepseekV2AttentionMLA.__init__`, instantiate `DoubleSparsitySelector` and record DS config, but do not allocate `TokenLabelTable` or call `bind_runtime_data`.
   - Add a model traversal helper invoked immediately after `ModelRunner.init_memory_pool(pre_model_load_memory)` that binds every DS attention module with the real `req_to_token_pool`.
   - Allocate exactly one table per model/rank with `max_tokens = self.req_to_token_pool.size`.
   - Remove the `device_buffer_size` fallback in `_bind_double_sparsity_runtime_data`; absence of the pool must raise.

3. **Fix the selector domain.**
   - Change the real selector path to accept `req_to_token`, `req_pool_indices`, and `seq_lens`.
   - Gather table rows for each request’s logical sequence positions: `physical = req_to_token[req_pool_idx, logical_pos]`.
   - Score those gathered labels into `[bs, max_seq_len]`; mask positions `>= seq_len`.
   - Run top-k over that logical axis and return ascending logical positions with `-1` padding.
   - Keep `logical_to_physical` as the single final conversion to FlashMLA physical slots.
   - Add the AC-0 fixture with `req_to_token = [7, 64, 200, 512]` and assert selector output `[0, 1, 2, 3]` maps to physical `[7, 64, 200, 512]`.

4. **Fix the query-space input.**
   - In `forward_mla.py`, after `q_b_proj`, derive projected Q-noPE for DS from the projected `q`, not from latent `q_lora`.
   - Pass projected Q-noPE into `_select_topk_indices` for DS; keep latent `q_lora` for the regular DSA indexer path.
   - Add a unit test that the real DS selector path receives a 3-D `[bs,H,128]` query tensor.

5. **Repair and rerun AC-13.**
   - Fix the three failing tests listed above.
   - Run `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`.
   - Only after all 150 pass, move AC-0/AC-13 tasks back to Completed and Verified in the goal tracker.

6. **Then implement AC-1.**
   - At `dsa_backend.py:1439`, `:1637`, and `:2162`, after `set_mla_kv_buffer`, project the latent K through `layer.kv_b_proj`, slice the K-noPE prefix, reshape to `[T,H_local,128]`, and call `token_label_write` with `out_cache_loc`.
   - Attach the bound token label table and per-rank channel selection to the attention layer during DS bind so the backend hook does not search global state.
   - Add the negative test that no hook leaves signatures zero, then the positive forward-extend/decode tests that written slots become non-zero.

7. **Continue the original active plan in dependency order.**
   - AC-2 lifetime and stale-slot protection.
   - AC-3 per-request ownership mask, now built on the logical-domain selector.
   - AC-7 short-seq bypass with label writes still firing.
   - AC-4 Method 1 Q+K calibration, then H200 mask generation.
   - AC-5 TP multiprocess logical-index equality.
   - AC-6 graph capture after M2/ownership mask lands.
   - AC-1b probe, AC-8 bench/quality smoke, AC-12 full quality gate.
   - AC-9 through AC-11 remain stretch but tracked.

## Goal Alignment Summary

```text
ACs: 2/14 addressed, 0/14 closed | Forgotten items: 0 after tracker update | Unjustified deferrals: 0
```

AC-0 and AC-13 were attempted but failed verification. AC-1, AC-1b, and AC-2 through AC-12 are still active pending work. There are no formal deferrals in the tracker, but the loop is incomplete until the original lower-bound ACs close.

## Tracker Update

I updated `goal-tracker.md` mutable sections:
- Reopened AC-0 and AC-13.
- Moved original `task-ac0-*` tasks back to Active with failed/pending verification status.
- Added the verified blocking issues from this review.
- Left Completed and Verified empty because no task has passed Codex verification in this round.

NOT COMPLETE
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-1-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-1-summary.md

## Task Tag Routing Reminder

Follow the plan's per-task routing tags strictly:
- `coding` task -> Claude executes directly
- `analyze` task -> execute via `/humanize:ask-codex`, then integrate the result
- Keep Goal Tracker Active Tasks columns `Tag` and `Owner` aligned with execution

**Optional fallback**: if you could not safely update the mutable section of `goal-tracker.md` directly, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Blocking Side Issues: bug Y blocks AC-2"]
- [E.g., "Add to Queued Side Issues: cleanup Z is non-blocking"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and reconcile the Goal Tracker if justified.
