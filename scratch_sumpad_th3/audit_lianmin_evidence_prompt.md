# Adversarial audit: evidence report for the Lianmin message

You are an adversarial reviewer. Your job is to REFUTE the report below. Assume it is wrong until proven otherwise. Direction-level errors (a claim whose conclusion is inverted or whose evidence does not support it) matter far more than typos.

## Target

`/Users/tom/domains/human/context/notes/projects/sglang/2026-07-29-dp-prefill-sum-padding/agent-drafts/2026-07-31-report-th3-lianmin-message-evidence.md`

The report defends 4 claims, each with source-code evidence and experimental evidence:

1. When a batch activates the breakable prefill CUDA graph (bcg), MAX_LEN padding is forced (so reverting PR #10414 has no effect on that path).
2. Under bcg, padding wastes attention-compute tokens (mainly for uneven / has-idle batches).
3. Without bcg, SUM_LEN padding does not waste attention-compute tokens (attention always runs on local rows; the sum-len buffer only exists between attention and MoE).
4. Without bcg, MAX_LEN padding does waste attention-compute tokens (same padding code path; heuristic entry).

## What to verify

- Every source citation is against **upstream/main commit `a1c30701aa`**. The local worktree is on an experiment branch with debug hooks, so read upstream code via `git show upstream/main:<path>` from the repo root (`/Users/tom/domains/agent-default/workspaces/ws-main/worktrees/sglang-dev-a`). Check every file:line range quoted in the report: does the code at those lines actually say what the report claims? Are there other code paths that invalidate the claim (e.g. paths where attention DOES run on the gathered buffer, or where SUM pads local rows, or where the force-MAX does not apply)?
- Every number is re-derivable from the artifacts under `/Users/tom/domains/human/others/artifacts/sglang/2026-07-30-dp-padding-th3/`. The per-experiment index with extraction commands is `/Users/tom/domains/human/context/notes/projects/sglang/2026-07-29-dp-prefill-sum-padding/agent-drafts/2026-07-30-index-th3-all-experiments.md`. Spot-check at least: one [DPPAD] waste number, one skew gemm number pair (6.52x), one e2e throughput pair, the forced_max_steps counts.
- Logic: does each piece of evidence actually support the chapter's claim, or only something weaker? Call out any claim that is overstated, under-qualified, or generalizes beyond what was measured (single model Qwen3-8B, tp8dp8, one skew shape).
- The report will be used to message an sglang maintainer, so any statement he could falsify with his own repro is a critical finding.

## Output

Write a markdown verdict with one section per chapter of the report: verdict CONFIRMED / OVERSTATED / REFUTED, plus the exact evidence for your verdict (commands you ran, code you read, numbers you recomputed). End with a list titled "Direction-level risks" — anything that could make the overall message wrong in front of the maintainer. Do not fix the report; only judge it.
