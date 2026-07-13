# PD Flip Source Terminal-State Monotonicity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent migration status polling from regressing a released or aborted source session to a transfer intermediate state.

**Architecture:** Preserve all transfer polling and accounting, but gate the delta pump's final state assignment when the source session is already terminal. Verify the behavior at the scheduler method boundary and on the real four-node chain.

**Tech Stack:** Python, pytest, SGLang scheduler, Docker, Mooncake, existing trace40 experiment runner.

## Global Constraints

- `source_released` and `source_aborted` must never be overwritten by transfer polling.
- Do not force a role switch.
- Do not relax rollover or idle safety predicates.
- Preserve non-terminal success and failure transitions.
- Execute in the existing workspace; do not create a worktree.

---

### Task 1: Terminal-state guard

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py:3860-3945`
- Test: `test/srt/test_pd_flip_migration_accounting.py`

**Interfaces:**
- Preserves: `Scheduler._pd_flip_source_pump_delta_transfer(session) -> None`
- Changes only the allowed final assignment of `session["state"]`.

- [ ] **Step 1: Write failing terminal-state tests**

Construct a scheduler/session with a completed non-noop delta and assert that calling the pump preserves `source_released`. Add the same assertion for `source_aborted`. Include a non-terminal control that still transitions to `source_delta_transferred`.

- [ ] **Step 2: Verify RED**

Run the focused tests and confirm the released/aborted cases fail because the state becomes `source_delta_transferred`.

- [ ] **Step 3: Implement the minimal guard**

Compute `terminal = session.get("state") in {"source_released", "source_aborted"}` before final state mutation. Update `source_failed` or `source_delta_transferred` only when `not terminal`.

- [ ] **Step 4: Verify GREEN and regression coverage**

Run the focused tests, complete migration-accounting tests, rollover-blocker tests, progressive-controller tests, scheduler py_compile, and `git diff --check`.

### Task 2: Real-chain acceptance

**Files:**
- Sync: `python/sglang/srt/managers/scheduler.py` to cloud-099 through cloud-102.
- Reuse: `experiments/pd_flip_trace40_full_chain.sh`.
- Produce: a new raw/log archive under `/home/tiancij/pd-artifacts`.

- [ ] **Step 1: Verify identical source hashes and compile on all nodes**

- [ ] **Step 2: Restart only node2/node3 to clear old in-memory sessions**

- [ ] **Step 3: Verify P,D,D,D, admission open, migration state none**

- [ ] **Step 4: Run the 40-request trace with 0.03-second TTFT trigger, ratio 0.5, and 10-second observation**

- [ ] **Step 5: Require all acceptance conditions**

Acceptance requires controller success, node2 Prefill, router role agreement, 40 completed requests, zero request errors, and all four workers healthy after the workload. Any later target pool leak is a failed acceptance and starts a new root-cause/TDD cycle.

### Task 3: Target terminal-state guard

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py:4405-4485`
- Modify: `python/sglang/srt/managers/scheduler.py:4490-4610`
- Test: `test/srt/test_pd_flip_migration_accounting.py`

- [ ] **Step 1: Write failing tests**

Create active and target-aborted sessions whose initial and delta entries are already transferred. Calling each pump must preserve the terminal state. Add non-terminal controls that retain the existing target transfer transitions.

- [ ] **Step 2: Verify RED**

Confirm old code regresses `active` to `target_transferred`/`target_delta_transferred` and regresses `target_aborted` in the same way.

- [ ] **Step 3: Implement minimal guards**

In both target pump methods, protect `active` and `target_aborted` from final state assignment while preserving counter refresh and all non-terminal transitions.

- [ ] **Step 4: Verify GREEN**

Run focused target-pump tests, source terminal tests, rollover blockers, progressive controller, py_compile, and diff-check.
