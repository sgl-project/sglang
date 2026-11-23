---
name: code-cleanup
description: Audit and clean up SGLang production code -- dead or unreachable code, duplicate implementations, redundant flags and env vars, compatibility shims, misplaced modules, wrong hardware gates -- using a catalog of patterns from this repo's refactor and cleanup PRs, RFCs and reverts, and the evidence bar those PRs were held to. Use when asked to clean up or simplify an area, remove dead code, dedupe kernels or helpers, retire a flag, env var, backend or feature, or review a cleanup or refactor PR.
---

# Code Cleanup

Keeps production code under `python/sglang/`, `sgl-kernel/` and `rust/`, and the docs that
describe it, free of code nothing runs, code that exists twice, and knobs that restate what
the code can decide. Neighbors:

- Tests and CI: [ci-test-audit](../ci-test-audit/SKILL.md). A cleanup keeps finding tests
  that never run; hand them over with the evidence.
- Proving a move or split is a pure relocation:
  [mechanical-refactor-verify](../mechanical-refactor-verify/SKILL.md).
- Where a kernel belongs, and which of several implementations of one kernel contract
  survives: [kernel-organization](../kernel-organization/SKILL.md). Env vars:
  [env-var-conventions](../env-var-conventions/SKILL.md). The frozen large classes:
  [large-class-style](../large-class-style/SKILL.md). Comments:
  `.claude/rules/comment-style.md` (a comment-only diff never touches one-line comments).

## How to use it

Read [patterns.md](patterns.md) first. It is the substance of this skill: each pattern is a
rule, how to spot it, the cases where it is wrong, and the PRs that applied or reverted it.
Section G is the set of guardrails that reverted cleanups taught; the in-flight list at the
end names migrations a cleanup must not fight.

The user names the scope: a directory, a subsystem, a kind of debt ("env vars that restate
the GPU"), or a PR to review. Every finding is specific: the file and symbol, the pattern
it matches, the evidence, and the action. "This module could be simpler" is not a finding.

## Workflow

1. **Fix the base.** Work from the latest main and record the SHA every comparison uses.
   Merge main again before opening the PR and re-run the leftover greps; main may have
   added a reader of something you deleted while the PR was open.
2. **Inventory with scripts before reading files.**
   - Identifier frequency over the whole repo, strings included: top-level defs whose name
     occurs only at their definition. Then drop the dynamic ones (A1, "Not this pattern").
   - Env vars: reads (`envs.X.get()`, raw `os.environ`, `get_bool_env_var`) vs declarations
     in `srt/environ.py` and `multimodal_gen/envs.py` vs doc rows; flag names read under a
     different name than documented.
   - Architecture gates: `get_device_capability`, `is_sm_at_least`, `is_blackwell`, `major
     >=`, gfx substring tests, next to the kernels they guard.
   - Dead locals: `ruff check --select F841` (the repo hook does not select it).
   - Baseline: import every module under the packages in scope and run `pytest
     --collect-only` on the test trees, on the base SHA.
3. **Verify every candidate at every site.** Read the code, not the grep line. Check the
   other platforms (ROCm, NPU, XPU, MUSA, CPU), out-of-tree users (the `sgl-kernel` wheel
   API, plugin pins, `from sglang import ...`), open PRs that are about to use it (`gh pr
   list --state all --search <symbol>`), trackers closed by the inactivity bot (stalled, not
   finished), and its age: code merged in the last few weeks is usually waiting to be
   wired, not dead.
4. **Decide per finding:** delete, dedupe, fix, keep, or report to the owner. Default to
   keep and report when the decision is a product call (removing a user-facing feature,
   a backend that is someone's only path), when an owner reserved the directory, or when
   an in-flight migration covers it.
5. **Implement.**
   - Delete the whole footprint: definition, every reader, docs and cookbook rows, tests,
     benchmarks, CI and scripts that set it. Grep the bare name, the dotted access and the
     string across `python/ test/ docs/ scripts/ docker/ .github/ benchmark/ rust/
     sgl-kernel/ .agents/` until nothing is left.
   - No forwarding shims for internal moves, no comments narrating the change, no
     compatibility code for names that were never public.
   - Keep relocation commits apart from semantic ones, and bug fixes apart from both.
   - A test that toggled a deleted switch patches the gate instead, so the comparison
     survives.
   - Format with the pinned hooks: `pre-commit run --files <changed files>`.
6. **Validate** (see "Evidence by change type"): base vs branch on the same machine, on
   every architecture the changed code runs on.
7. **Land** with the PR structure below; split only into steps that are each green.

## Evidence by change type

Base vs branch on the same machine, the same command, failing test ids compared. Re-run
every branch-only failure serially on both trees before calling it a regression; parallel
runs produce port collisions (`EADDRINUSE`) that look like failures.

- **Dead code deletion:** whole-repo grep including strings and configs; the import walk and
  `--collect-only` show no new failures; the tests that cover the neighborhood pass. A
  deleted AOT op needs a clean rebuild of sgl-kernel.
- **Refactor that claims no behavior change:** greedy tokens and logprobs compared bitwise,
  base vs branch. A small accuracy eval cannot resolve it: identical code has moved GSM8K by
  0.03 on 200 questions.
- **Dedup of two implementations:** outputs bit-identical (`torch.equal`, or identical
  generated code) over the shapes and dtypes every caller uses, on each architecture
  claimed, including edge inputs (zero tokens, fully masked rows, ties, all-underflow). Force
  the merged path when the default config skips it, and log or count that it ran. If not
  bit-exact, keep both and say why in one line.
- **Kernel rewrite or move between packages:** the above, plus CUDA graph / piecewise graph
  / `torch.compile` paths, the custom-op registration, and the nightly accuracy test of an
  affected model; JIT lands with a bitwise JIT-vs-AOT test before the AOT copy goes.
- **Backend or feature removal:** a fallback story, parity numbers for the replacement, every
  caller ported, platform owners' runs, and a deprecation issue when users may depend on it.
- **Flag, env var or endpoint removal:** the deprecation shape used (alias, `Deprecated*Action`,
  `_DEPRECATED_ENVS`), YAML configs still parse, the old defaults preserved, docs and
  cookbook updated, the result checked against the real parser.
- **Default change:** end-to-end accuracy with sgl-eval on the target hardware, a log line or
  trace proving the new default path ran, and perf where the path is a fast path.
- **Architecture gate change:** the gate written as a pure function of the capability tuple
  and unit-tested on CPU across families (or `current_arch` patched), a probe on each
  architecture it touches, and the kernel's tests on the hardware of the lane they are
  registered on.
- **Diffusion fast path:** the tier it runs at, scored with
  `multimodal_gen/test/quality_tier_admission.py`, and `torch.equal` at `exact`.

## PR structure

Title `[Refactor] <area>: <what>`. In the body, in this order:

1. Motivation: what drifted and how much, in one paragraph.
2. **Bugs fixed (review these first)**: area, problem, fix, and the regression test that
   fails on main.
3. **Behavior changes**: before / after table. A cleanup with no behavior change says so.
4. What was deleted or merged, grouped by pattern, with the proof for each group.
5. **Kept on purpose**: things that look removable and are not, with the reason.
6. **For the owners (not changed here)**: findings that need a product or owner decision.
7. Validation per hardware with the base and head SHA each run used, and **not validated**:
   platforms and paths you could not run.

Do not add a line-count target or a deletion quota; the value of a cleanup is in the silent
failures it finds, not its size.

## Reviewing a cleanup or refactor PR

- Does each deletion have a proof that covers string lookups and other platforms?
- Is every merged pair shown bit-identical, or kept with a reason?
- Does the "mechanical" part reproduce exactly (mechanical-refactor-verify), with the
  behavior changes listed separately?
- Are user-facing names retired with an alias and warning (C4, C5), and internal ones moved
  without shims (E1)?
- Were the hardware lanes the change touches actually run, including nightly-only tests of
  affected models?
- Does it fight an in-flight migration from the list in patterns.md?
- Is each problem found in review classified as introduced by the PR or pre-existing on the
  base SHA, and does every result name the SHA it ran on?

## Traps

- macOS `git grep` does not support `\|`; use `-E` with `|`. Paths with non-ASCII
  characters need `LC_ALL=en_US.UTF-8`.
- zsh does not word-split `$VAR` and treats `$VAR:t` as a modifier: use `${=VAR}` and
  `"${VAR}:path"`, or write the loop in Python. A keep-list that silently fails to split
  deletes what it was meant to keep.
- `CUDA_VISIBLE_DEVICES=` (empty) crashes collection in some test utilities; hide GPUs with
  a valid index set instead.
- A test that passes on sm_103 has not been validated on sm_100; validate on the lane's own
  architecture.
- If main changes a file you deleted while the PR is open: a change that extends a guard
  or a test means the file is alive, restore it; a mechanical sync keeps it deleted.
- `from sgl_kernel import X` under an XPU branch resolves to `sgl-kernel-xpu`; it is not a
  reader of the CUDA AOT op.
- `DeprecationWarning` raised from library code is hidden by default; a deprecation users
  must see needs `logger.warning`, `UserWarning` or `FutureWarning`.
- argparse prefix matching: a deleted flag that prefixes one live flag silently binds to it.
  Check removals against the real parser, not grep.
- Tests that fake `server_args` with `MagicMock` never notice a read of a deleted field;
  build one real `ServerArgs`.
- `/rerun-test` on a unittest file takes `file.py::Class.method`; `Class::method` fails with
  `AttributeError` before any test runs.
- Subagents fanning out over files: give them a fixed brief, have them prove each verdict,
  and re-verify every proposed deletion yourself; "covered by X" claims from a quick pass
  are leads, not findings.

## Keeping the catalog current

When a newer PR applies a pattern, or a new pattern recurs across PRs, update
`patterns.md`: keep the rule general and treat the PR numbers as replaceable evidence. When
a tracker in the in-flight list finishes, remove its entry.
