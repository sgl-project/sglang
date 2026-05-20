# Loop 3 — New Session Runbook

Run these in order in a **fresh session** (not the compacted one).

## Phase 0 — Pre-session sanity

```bash
# Verify you're at the right anchor before opening a new session
cd /sgl-workspace/sglang
git status                       # should be clean except for development/loop3/
git log --oneline -1             # should show ba7d55d64 (R9)
git branch --show-current        # should be dev/double-sparsity-standalone

# Create the loop-3 anchor at the current HEAD (only if it doesn't exist)
git rev-parse --verify loop3-base 2>/dev/null || git branch loop3-base ba7d55d64

# Commit the loop3 draft + this runbook so the new session sees them
git add development/loop3/draft.md development/loop3/RUNBOOK.md
git commit -m "[Sparsity] Loop-3: scaffold draft and runbook"
```

If you want rank-1 to have these too:
```bash
git push jiminator dev/double-sparsity-standalone loop3-base
ssh double-sparsity 'cd /sgl-workspace/sglang && git fetch jiminator && git checkout dev/double-sparsity-standalone && git reset --hard jiminator/dev/double-sparsity-standalone && git branch -f loop3-base jiminator/loop3-base'
```

## Phase 1 — Open new Claude Code session in this directory

```bash
cd /sgl-workspace/sglang
# Then start a fresh `claude` session
```

## Phase 2 — Inside the new session, run plan → refine → loop

### Step 1: Generate the plan from the draft
```
/humanize:gen-plan --input development/loop3/draft.md --output development/loop3/plan.md --discussion
```

This produces `development/loop3/plan.md` with structured ACs, task breakdown, and routing tags. Codex will do its first-pass analysis; you'll be asked to resolve disagreements interactively.

### Step 2 (optional but recommended): add critique comments before refining

If you want the same Linus-voiced + Codex critique passes that Loop 2 used:

```
Ask penseive to review @development/loop3/plan.md and check for any code smell, software architecture issues. How would Linus Torvalds react to this plan? Structure your critiques by adding comments to the file with <comment>CRITIQUE</comment>
```

Then:

```
/humanize:ask-codex Do you agree with these comments from Linus in @development/loop3/plan.md If you have any additional comments/critiques to add and check for any issues about not limited to but including code smell, software architecture issues. Structure your critiques by adding comments to the file with <comment>CRITIQUE</comment>
```

Skip this step if you want to go faster — the draft is already tighter than Loop 2's, so there's less to critique.

### Step 3: Refine the plan (bake comments in, or just clean up)
```
/humanize:refine-plan --input development/loop3/plan.md --output development/loop3/refined_plan.md --discussion
```

### Step 4: Start the RLCR loop
```
/humanize:start-rlcr-loop --plan-file development/loop3/refined_plan.md --yolo --base-branch loop3-base
```

Notes:
- `--base-branch loop3-base` makes the code review compare only Loop 3's diff (not all of Loop 2's R0–R9 commits).
- `--yolo` skips the plan-understanding quiz (you wrote this draft yourself, so you know what's in it).
- Default budget is enough for the 3-AC scope; don't extend it. If you hit round 6 with <2 ACs closed, stop and reassess — that's the Loop 2 lesson.

## Phase 3 — During the loop: hardware verification rules

Unit tests are necessary but **not sufficient** for any AC in this loop. Every AC ends with at least one of:
- A forward pass run against a real model (smoke test on H200), OR
- A `bench_serving` run with real numbers committed to the round summary.

If a round closes an AC using only unit tests, the next round must add the hardware step before moving on. Reviewer should flag this.

## Phase 4 — Done criterion

The loop is COMPLETE when the round summary contains a `bench_serving` table like:

```
| config       | TPS   | accept_rate | accept_length | latency_p50 |
|--------------|-------|-------------|---------------|-------------|
| DS-off       | ...   | n/a         | n/a           | ...         |
| DS-on        | ...   | ...         | ...           | ...         |
```

…with DS-on not crashing, accept_rate > 0, and a quality smoke check passing.

## Phase 5 — If the loop stagnates again

Apply the Loop 2 lesson: **if 2 consecutive rounds open more gaps than they close, stop the loop manually** with `/humanize:cancel-rlcr-loop` and reassess scope. Don't wait for the circuit breaker.

## Cleanup if you abort

If you need to bail out and try again:
```bash
git checkout dev/double-sparsity-standalone
git reset --hard loop3-base       # discards loop-3 commits, keeps draft+runbook if committed before reset
rm -rf .humanize/rlcr/<loop3-timestamp>
```
