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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-2-contract.md

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
# Round 1 Review Result

Mainline Progress Verdict: ADVANCED

Round 1 materially advanced the AC-0/AC-13 repair: the required public import succeeds, the unit regression suite now passes 150/150, the selector has a logical-domain path, bind is deferred until after `init_memory_pool()`, and the normal DS path passes projected Q-noPE.

However, AC-0 is still not safe to close. The implementation binds `TokenLabelTable.max_tokens` from `req_to_token_pool.size`, but in this codebase that field is the request-row capacity, not the physical KV slot capacity addressed by `out_cache_loc`. Once AC-1 wires the write hook, normal cache locations beyond the request count will write out of bounds.

## Mainline Gaps

1. **AC-0 still uses the wrong data authority for token-label table size.**

   Evidence:
   - `python/sglang/srt/models/deepseek_v2.py:1906-1915` reads `server_args._ds_req_to_token_pool` and sets `max_tokens = ds_pool.size`.
   - `python/sglang/srt/mem_cache/memory_pool.py:138-160` defines `ReqToTokenPool.size` as the number of request rows; its tensor is shaped `[size + 1, max_context_len]`.
   - The physical slots written by `out_cache_loc` come from the token-to-KV allocator. For paged allocation, `python/sglang/srt/mem_cache/allocator.py:402-405` returns `page * page_size + offset`, and `:516-518` makes pages range from `1..num_pages`.
   - The DSA KV buffer is sized to that physical address space, e.g. `python/sglang/srt/mem_cache/memory_pool.py:1686-1693` allocates `size + page_size` rows, and `python/sglang/srt/layers/attention/double_sparsity/token_label_write.py:53-55` writes labels at `cache_loc` directly.

   Reproducer:

   ```text
   TokenLabelTable(max_tokens=2); token_label_write(cache_loc=[64])
   -> IndexError: index 64 is out of bounds for dimension 0 with size 2
   ```

   Required fix: after `ModelRunner.init_memory_pool()`, publish the physical token-to-KV slot capacity, not the request-pool row count. Use the same address space that `set_mla_kv_buffer(..., cache_loc, ...)` accepts, preferably the first dimension of the actual KV buffer for the target DSA pool or an equivalent `token_to_kv_pool.size + token_to_kv_pool.page_size` value. Bind `TokenLabelTable(max_tokens=that_capacity)`. Add a regression where `req_to_token_pool.size` is small, `req_to_token` maps logical positions to physical slots like `[7, 64, 200, 512]`, and both `retrieve_topk` and `token_label_write` succeed for those slots.

2. **Claude’s claimed non-contiguous fixture is not actually in the committed unit path.**

   `test_ds_branch_returns_topk_indices_via_adapter` uses an identity `req_to_token` mapping and a selector with `IS_PLACEHOLDER = False` but no bound table/mask, so it exercises the placeholder ascending path, not logical-domain real scoring (`test/registered/unit/layers/attention/test_double_sparsity_unit.py:363-381`). I manually verified the function-level logical path with `[7,64,200,512]`, but AC-0 needs this as a committed regression because this exact gap escaped Round 0.

3. **AC-1 through AC-12 remain unfinished mainline work.**

   This round was scoped to AC-0/AC-13, but the original lower bound is still AC-0 through AC-8, AC-12, and AC-13. In particular, the three `dsa_backend.py` hook sites still only call `set_mla_kv_buffer` and do not project `kv_b_proj` K-side labels or call `token_label_write` (`python/sglang/srt/layers/attention/dsa_backend.py:1439`, `:1637`, `:2162`).

## Blocking Side Issues

1. **Do not start AC-1 hook wiring until the slot-count authority is fixed.**

   If the hook is added now, real `out_cache_loc` values will index the undersized label table. The next round must first make the table cover the physical token-to-KV address space, then add the hook.

## Queued Side Issues

1. **AC-6 graph helper is stale relative to logical-domain selection.**

   `capture_decode_step` still calls `selector.retrieve_topk(..., seq_lens=seq_lens)` without passing `req_to_token` (`python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py:135-142`, `:157-178`). That forces the physical-domain fallback during graph capture, contrary to the AC-6 dependency on the M2 logical ownership path. This does not need to take over AC-1, but it must be fixed before `task-ac6-cuda-graph`.

2. **AC-8 observability still mixes token counts with page-named metrics.**

   `_publish_ds_request_summary` uses `selected = valid_lengths[b]` from token top-K but divides it by `total_pages` and publishes `selected_pages` (`python/sglang/srt/models/deepseek_v2.py:1997-2004`). With token-level `top_k=2048` and a 4096-token sequence, this can produce nonsensical negative sparsity. Queue this for AC-8 metrics cleanup.

## Required Implementation Plan

1. Add a physical slot-capacity bind source in `ModelRunner` immediately after `init_memory_pool()`. Publish both `req_to_token_pool` for logical gathers and `token_label_max_tokens` for label allocation.
2. Change `_bind_double_sparsity_runtime_data` to allocate `TokenLabelTable` from that physical slot capacity. Keep failing fast if either required runtime object is absent.
3. Add tests proving `ReqToTokenPool.size` is not used as label capacity: a small request pool with physical slots `[7,64,200,512]` must write labels, retrieve logical `[0,1,2,3]`, and adapt back to those physical slots.
4. Then implement AC-1 at all three `dsa_backend.py` sites: after each KV write, project latent K through `layer.kv_b_proj`, slice/reshape projected 128-d K-noPE per local head, and call `token_label_write` with `out_cache_loc`.
5. Continue in dependency order: AC-2 stale-slot/lifetime, AC-3 ownership tests, AC-7 bypass, AC-4 calibration, AC-5 TP, AC-6 graph capture, AC-1b probe, AC-8 smoke, and AC-12 quality.

## Goal Alignment Summary

```text
ACs: 2/15 addressed, 1/15 closed | Forgotten items: 0 after tracker update | Unjustified deferrals: 0
```

AC-13 is verified. AC-0 is reopened for the physical slot-capacity bug. AC-1, AC-1b, AC-2 through AC-12, and AC-9 through AC-11 remain tracked as active work, not accepted deferrals.

## Tracker Update

I updated `goal-tracker.md` mutable sections:
- Added `task-ac0-slot-authority` as active blocking work.
- Added the request-pool-vs-KV-slot sizing issue to Blocking Side Issues.
- Moved AC-13 to Completed and Verified.
- Kept AC-0 out of Completed and Verified.
- Added queued AC-6 graph-helper and AC-8 observability issues.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-2-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-2-summary.md

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
