---
id: run-when-reviewing-code
type: pipeline
title: Code Review Pipeline
status: active
created: 2026-02-11
updated: 2026-02-28
tags: [pensieve, pipeline, review]
name: run-when-reviewing-code
description: Code review stage pipeline. First explore commit history and code hotspots, extract capture candidates, then produce high-signal taste review conclusions following the fixed Task Blueprint. Trigger words: review, code review, inspect code.

stages: [tasks]
gate: auto
---

# Code Review Pipeline

This pipeline handles task orchestration only. Review standards and their underlying rationale are kept in Knowledge to avoid duplication in this file.

**Knowledge reference**: `knowledge/taste-review/content.md`

**Context Links (at least one)**:
- Based on: [[knowledge/taste-review/content]]
- Leads to: none
- Related: none

---

## Signal Judgment Rules

The value of a review report depends on its signal-to-noise ratio -- too many low-signal issues drown out the truly important ones.

- Only report high-signal issues: reproducible, locatable, and affecting correctness/stability/user-visible behavior.
- Candidate issues must be verified before entering the final report, because unverified speculation wastes fix time.
- Default confidence threshold: `>= 80` to enter the final report.
- Do not report pure style suggestions, subjective preferences, or risk items based on speculation.

---

## Task Blueprint (create tasks in order)

### Task 1: Baseline Exploration (commit history + actual code)

**Goal**: Identify hotspots and capture candidates first to avoid blind review

**Read inputs**:
1. `git log` (default: last 30 commits, can be overridden by user-specified range)
2. Actual code (prioritize recently and frequently changed files)
3. `knowledge/taste-review/content.md`

**Steps**:
1. Summarize high-frequency files/modules and main change types from recent commits
2. Read the corresponding code, identify complexity hotspots and areas with unclear boundaries
3. Output two lists:
   - Files to review (by priority)
   - Capture candidates (annotated with suggested type: `knowledge/decision/maxim/pipeline`, each with evidence)

**Completion criteria**: Actionable review scope + capture candidate list (both with evidence)

---

### Task 2: Prepare Review Context

**Goal**: Clarify review boundaries to avoid missed coverage

**Read inputs**:
1. User-specified files / commits / PR range (if any)
2. Task 1 output: files to review list and candidate information
3. `knowledge/taste-review/content.md`

**Steps**:
1. Merge user-specified range with Task 1 findings to determine the final review scope
2. Identify technical language, business constraints, and risk points
3. Finalize the files-to-review list (by priority)

**Completion criteria**: Scope is clear, with a finalized file list

---

### Task 3: Per-file Review with Evidence Recording

**Goal**: Produce a candidate issue list (with evidence and confidence scores)

**Read inputs**:
1. Task 2 output: finalized files-to-review list
2. `knowledge/taste-review/content.md`

**Steps**:
1. Run the review checklist on each file (theory and rationale are in Knowledge, not duplicated here)
2. Record only "possibly real" candidate issues, with confidence scores (0-100)
3. Annotate each candidate issue with precise code location and evidence
4. Record user-visible behavior change risks (if any)

**Completion criteria**: Candidate issue list (with confidence, evidence, location)

---

### Task 4: Verify Candidate Issues and Filter False Positives

**Goal**: Retain only high-signal, verifiable issues

**Read inputs**:
1. Task 3 candidate issue list
2. Corresponding code context and rule rationale

**Steps**:
1. Verify each candidate issue for real reproducibility
2. Update final confidence for each issue and remove items `<80`
3. Remove issues with insufficient evidence, unclear scope, or reliance on speculation
4. Produce the "verified issue list"

**Completion criteria**: High-signal issue list (each locatable, explainable, confidence >= 80)

---

### Task 5: Generate Actionable Review Report

**Goal**: Output directly actionable fix suggestions with priorities

**Read inputs**:
1. Task 4 verified issue list

**Steps**:
1. Summarize key issues by severity (CRITICAL -> WARNING)
2. Provide specific fix suggestions or rewrite direction for each issue
3. Clarify user-visible behavior changes and regression risks
4. If no issues found, explicitly output "no high-signal issues"

**Completion criteria**: Report contains only verified issues with a clear fix order

---

### Task 6: Capture Reusable Conclusions (optional)

**Goal**: Capture reusable conclusions into the existing four categories

**Read inputs**:
1. Task 5 review report

**Steps**:
1. If the conclusion is a project-specific choice, capture to `decision`
2. If the conclusion is a general external method, capture to `knowledge`
3. Add `Based on / Leads to / Related` links in the captured entry (at least one, if it is a decision)
4. If no reusable conclusions exist, explicitly record "no new captures"

**Completion criteria**: Capture result is clear (written or explicitly skipped)

---

## Failure Fallback

Each exception scenario has a corresponding handling approach, preventing misleading conclusions when information is insufficient.

1. Cannot obtain commit history (not a Git project or no history): Mark `SKIPPED` and continue to Task 2 (review based on existing code only).
2. Review scope is unclear: Return missing information and stop; do not enter Task 3 -- reviewing with unclear scope tends to drift off focus.
3. Cannot verify candidate issues: Mark "unverifiable" and filter out; do not include in the final report.
4. If all candidates are filtered out: Output "no high-signal issues". Padding the report with low-quality suggestions damages report credibility.
