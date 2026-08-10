# MegaMoE Runner Alias Implementation Plan

> **For Codex:** Execute this plan task-by-task with test-driven development.

**Goal:** Accept `--moe-runner-backend megamoe` as a compatibility spelling
for `--moe-a2a-backend megamoe`, raise the default MegaMoE per-rank token cap
to 8192, and exercise the alias in the DeepSeek V4 Pro GB300 nightly.

**Architecture:** Keep MegaMoE as an A2A backend, not a kernel runner. Add a
target-model-only CLI choice and normalize it at the start of `ServerArgs`
construction to runner `auto` plus A2A `megamoe`, before model-specific
resolution. Keep the speculative runner's real-backend choices unchanged.

**Tech Stack:** Python dataclasses, argparse, unittest/pytest, SGLang's
`CustomTestCase` and CPU CI registration.

---

### Task 1: Specify the CLI alias behavior

**Files:**
- Modify: `test/registered/unit/test_server_args_migration.py`
- Modify: `python/sglang/srt/server_args.py`

1. Add a test that parses `--moe-runner-backend megamoe` for a dummy model and
   asserts the constructed `ServerArgs` has runner `auto` and A2A `megamoe`.
2. Add a test that combines the alias with a conflicting A2A backend and
   asserts MegaMoE wins.
3. Add a parser test that `--speculative-moe-runner-backend megamoe` remains
   rejected.
4. Run the focused tests and confirm they fail because `megamoe` is not an
   accepted runner choice.
5. Add a separate target-model runner choice list that includes `megamoe`,
   while preserving out-of-tree runner choice extensions and leaving the
   speculative choices unchanged.
6. Add an early `ServerArgs` input-normalization helper. Convert runner
   `megamoe` to runner `auto` and A2A `megamoe`; warn when replacing a
   conflicting A2A backend.
7. Re-run the focused tests and confirm they pass.

### Task 2: Raise and verify the environment default

**Files:**
- Create: `test/registered/unit/test_environ.py`
- Modify: `python/sglang/srt/environ.py`

1. Add CPU tests that clear the MegaMoE cap variable, assert the default is
   8192, then set an explicit value and assert it remains authoritative.
2. Run the test and confirm the default assertion fails with 1024.
3. Change the `EnvInt` default from 1024 to 8192.
4. Re-run the environment test and confirm it passes.

### Task 3: Exercise the alias in the GB300 nightly

**Files:**
- Modify: `test/registered/gb300/test_deepseek_v4_pro_fp4.py`

1. Replace the high-throughput variant's canonical A2A selector with
   `--moe-runner-backend megamoe`.
2. Keep
   `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320` unchanged.
3. Add a static unit assertion only if the existing source-level configuration
   is not adequately covered by the reviewed diff; do not duplicate the GPU
   nightly locally.

### Task 4: Verify and publish

**Files:** All changed files.

1. Run the focused unit tests for CLI normalization and environment behavior.
2. Run the complete affected unit modules.
3. Run formatter/linter checks and `git diff --check`.
4. Review the diff for accidental speculative behavior or unrelated changes.
5. Commit the implementation with an intentional message.
6. Push `codex/megamoe-runner-alias` and open a draft PR describing the
   nightly failure mode, behavior change, memory impact, and checks run.
