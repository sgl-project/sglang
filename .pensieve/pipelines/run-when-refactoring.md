---
id: run-when-refactoring
type: pipeline
title: Refactor Pipeline
status: active
created: 2026-04-27
updated: 2026-04-27
tags: [pensieve, pipeline, refactor]
name: run-when-refactoring
description: Mandatory refactor workflow: trust upstream data, proceed in 2-3 user-visible steps, delete old code once new code works, forbid compatibility/fallback/temporary branches, and preserve current user-visible behavior. Trigger words: refactor, large refactor, split code.

stages: [tasks]
gate: auto
---

# Refactor Pipeline

This pipeline only handles real refactors. The goal is not to rewrite code for style; it is to remove wrong data boundaries, special cases, and old code residue without changing user-visible behavior.

**Context Links (at least one)**:
- Related: [[knowledge/taste-review/content]]

---

## Hard Rules

1. Trust upstream data fully. If data is missing, provide it upstream instead of patching downstream.
2. Very large code changes must proceed in 2-3 steps; each step must end with user-visible behavior, not isolated code. When the new code is active, the old code must be deleted.
3. Do not write compatibility, fallback, temporary, backup, or mode-specific branches. Eliminate special cases instead of wrapping them.
4. Preserve current user-visible behavior.

---

## Task Blueprint (execute in order)

### Task 1: Confirm this is a real problem

**Goal**: Reject refactors done only because the code "looks better."

**Execution steps**:
1. State the real current problem: wrong data ownership, duplicated logic, leaked boundaries, old-path residue, or complexity blocking future work.
2. State the practical cost of not refactoring.
3. If the problem cannot be explained in one sentence, stop and do not implement.

**Completion criteria**: The refactor purpose is stated in one sentence and maps to real code paths.

---

### Task 2: Pin down data structure and upstream authority

**Goal**: Fix data boundaries before changing control flow.

**Execution steps**:
1. Identify the core data: what it is, who creates it, who owns it, and who consumes it.
2. If downstream code lacks fields or works by guessing, add the fields to the upstream authority path.
3. Delete downstream patch branches created for missing data.

**Completion criteria**: Data has a single source; downstream code consumes an explicit contract and does not guess or fall back.

---

### Task 3: Split large changes into 2-3 user-visible steps

**Goal**: Each step runs, shows usable behavior, and can be regression-tested.

**Execution steps**:
1. Split the refactor into at most 3 steps.
2. For each step, state:
   - What user-visible behavior still works
   - Which old code path will be deleted when the new code is active
   - Which command or manual path verifies it
3. Do not pile up unused new code first.

**Completion criteria**: No step ends with isolated code, and old paths shrink as the refactor progresses.

---

### Task 4: Eliminate special cases during implementation

**Goal**: Make the normal path cover all normal inputs.

**Execution steps**:
1. Prefer changing data structure or call boundaries over adding another `if`.
2. When a compatibility, fallback, temporary, backup, or mode-specific branch appears, first ask why the normal path cannot handle it.
3. If an old path is no longer authoritative, delete it.

**Completion criteria**: Special branches decrease, and old code no longer remains on runtime paths.

---

### Task 5: Verify user-visible behavior is unchanged

**Goal**: Prove the refactor did not break existing user workflows.

**Execution steps**:
1. Run lint/typecheck/test/e2e checks that match the changed scope.
2. Manually or automatically verify user-visible paths.
3. If behavior must change, stop and explain why; this is no longer a pure refactor.

**Completion criteria**: Current user-visible behavior remains unchanged, and the verification result can be repeated.

---

## Failure Fallback

1. No real problem found: do not refactor.
2. Missing data can only be guessed downstream: fix upstream first; do not patch.
3. A large change cannot be split into user-visible steps: shrink the scope and land one workable step first.
4. Two runtime paths must be kept: redesign the boundary until the old path can be deleted.
5. User-visible behavior changes: stop, reclassify the task as a feature change or breaking migration, and do not pretend it is a refactor.
