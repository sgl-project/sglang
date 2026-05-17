---
id: run-when-committing
type: pipeline
title: Commit Pipeline
status: active
created: 2026-02-28
updated: 2026-02-28
tags: [pensieve, pipeline, commit, self-improve]
name: run-when-committing
description: Mandatory commit-stage pipeline. First determine whether there are insights worth capturing; if so, run self-improve to capture them, then perform atomic commits. Trigger words: commit, git commit.

stages: [tasks]
gate: auto
---

# Commit Pipeline

Before committing, automatically extract insights from the session context + diff and capture them, then perform atomic commits. No user confirmation is requested at any point.

**Self-improve reference**: `.src/tools/self-improve.md`

**Context Links (at least one)**:
- Based on: [[knowledge/taste-review/content]]
- Related: none

---

## Signal Judgment Rules

The value of capturing insights lies in reuse next time; unsubstantiated guesses will mislead future decisions.

- Only capture insights that are reusable and evidence-backed; unverifiable guesses must not be stored.
- Classify according to semantic layers: IS -> `knowledge`, WANT -> `decision`, MUST -> `maxim`.
- Assign by semantics, not by "default to knowledge", because misclassification causes a mismatch in binding strength (something that should be MUST but is written as knowledge will easily be ignored later).

---

## Task Blueprint (create tasks in order)

### Task 1: Decide whether to capture -- determine if there are insights worth capturing

**Goal**: Quickly determine whether this commit contains experience worth capturing; skip to Task 3 if not

**Read inputs**:
1. `git diff --cached` (staged changes)
2. Current session context

**Steps**:
1. Run `git diff --cached --stat` to understand the scope of changes
2. Review the current session, checking for any of the following signals (any match triggers capture):
   - Identified a bug root cause (debugging session)
   - Made an architectural or design decision (considered multiple approaches)
   - Discovered a new pattern or anti-pattern
   - Exploration produced a "symptom -> root cause -> location" mapping
   - Clarified boundaries, ownership, or constraints
   - Discovered a capability that does not exist / has been deprecated in the system
3. If none of the above signals are present (purely mechanical changes: formatting, renaming, dependency upgrades, simple fixes), mark "skip capture" and jump directly to Task 3

**Completion criteria**: Clear determination of "capture needed" or "skip capture", with a one-line rationale

---

### Task 2: Auto-capture -- extract insights and write them

**Goal**: Extract insights from session context + diff, write to user data, without asking the user

**Read inputs**:
1. Task 1 determination result (if "skip", skip this Task)
2. `git diff --cached`
3. Current session context
4. `.src/tools/self-improve.md`

**Steps**:
1. Read `self-improve.md` and execute its Phase 1 (extract and classify) + Phase 2 (read spec + write)
2. Extract core insights from the session (may be multiple)
3. For each insight, first determine the semantic layer and classify (IS->knowledge, WANT->decision, MUST->maxim; may land in multiple layers simultaneously if needed)
4. Read the spec for the target type from `.src/references/`, and generate content according to the spec
5. Type-specific requirements:
   - `decision`: include the "three exploration-reduction items" (fewer questions next time / fewer lookups / invalidation conditions)
   - Exploration-type `knowledge`: include (state transitions / symptom->root cause->location / boundaries and ownership / anti-patterns / verification signals)
   - `pipeline`: must meet conditions (recurring + non-interchangeable steps + verifiable)
6. Write to target path, add association links
7. Refresh Pensieve project state:
   ```
   bash "$PENSIEVE_SKILL_ROOT/.src/scripts/maintain-project-state.sh" --event self-improve --note "auto-improve: {files}"
   ```
8. Output a brief summary (write path + capture type)

**DO NOT**: Do not ask user for confirmation, do not show drafts awaiting approval, write directly

**Completion criteria**: Insights have been written to user data (or explicitly determined as not worth capturing), `state.md` and `.state/pensieve-user-data-graph.md` have been refreshed

---

### Task 3: Atomic commits

**Goal**: Perform atomic git commits

**Read inputs**:
1. `git diff --cached`
2. User's commit intent (commit message or context)

**Steps**:
1. Analyze staged changes, cluster by reason for change
2. If multiple independent change groups exist, commit each separately (one atomic commit per group)
3. Commit message conventions:
   - Title: imperative mood, <50 characters, specific
   - Body: explain "why" not "what"
4. Execute `git commit`

**Completion criteria**: All staged changes have been committed, each commit is independent and revertible

---

## Short-Term Memory Prompt

After committing, if there are expired entries in `short-term/`, append a one-line reminder:

> Short-term memory has N entries pending triage. Run pensieve refine to process them. Tool spec: `.src/tools/refine.md`.

Do not perform triage during the commit flow; only remind.

## Failure Fallback

1. `git diff --cached` is empty: skip Task 2/Task 3, output "no staged changes, nothing to commit".
2. Capture step fails: log the blocking reason and skip capture, continue to Task 3; append "suggest running `doctor`" at the end.
3. `state.md` maintenance fails: keep already-captured content, report the failed command and retry suggestion, do not roll back already-written files.
