# PD Flip Runtime Full Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When target HiCache prefix restore fails after a non-zero prefix hit, retry only the failed RIDs by transferring their complete `[0, C0)` KV from the source decode node.

**Architecture:** Extend the existing migration protocol with a target-reported fallback-required RID set and a source full-copy retry. The controller coordinates the retry between initial prepare and terminal failure, while scheduler helpers clean partial target resources and rebuild sender/receiver state without changing ownership before commit.

**Tech Stack:** Python, SGLang scheduler IPC/HTTP control plane, Mooncake KV transfer, pytest.

## Global Constraints

- Initial selection remains fixed-ratio first-N; capacity fallback remains repeated halving.
- Only RIDs whose target HiCache restore fails use full fallback.
- Full fallback sends complete `[0, C0)` KV and sets `stitch_mode=source_decode_full_fallback`.
- No ownership change occurs until target prepare and commit succeed.
- Any fallback failure safely aborts and returns ownership to source decode.
- Status and measurement output expose fallback reason, attempted flag, source bytes, and duration.

---

### Task 1: Scheduler fallback state and full-copy retry

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: existing PD flip scheduler test file(s) under `test/srt/`

**Interfaces:**
- Produces target status fields `fallback_required_rids`, `fallback_reason`, and per-entry fallback measurements.
- Produces source operation that rebuilds a sender for complete `[0, C0)` coverage.

- [ ] Add failing unit tests for successful partial restore, restore-failure fallback request, selective multi-RID fallback, full-copy success, fallback failure rollback, and resource cleanup.
- [ ] Run the focused tests and confirm the new cases fail before implementation.
- [ ] Change target restore failure handling from immediate terminal failure to fallback-required for non-zero-hit stitch entries.
- [ ] Add target cleanup/reprepare logic that releases partial restore/preallocation state before full receiver setup.
- [ ] Add source full-copy resend logic using page range `[0, C0)`, preserving output boundary and ownership.
- [ ] Record `fallback_reason`, `fallback_attempted`, `fallback_source_bytes`, `fallback_duration_seconds`, and final stitch mode.
- [ ] Run focused tests and `python -m py_compile python/sglang/srt/managers/scheduler.py`.

### Task 2: Controller fallback handshake

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py`
- Modify: existing controller tests under `test/srt/`

**Interfaces:**
- Consumes target `fallback_required_rids`.
- Invokes source full-copy retry and target fallback prepare before resuming normal wait/commit.

- [ ] Add failing tests for no-fallback success, selective fallback handshake, fallback error abort, and journal/action records.
- [ ] Run focused controller tests and confirm failure.
- [ ] Insert fallback handshake after initial target prepare/status and before terminal failure.
- [ ] Ensure retries are limited to one full fallback attempt per RID/session.
- [ ] Preserve existing observation, delta, finish, commit, activation, and role-switch ordering.
- [ ] Run focused tests.

### Task 3: Measurement, regression, and live validation

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_migration_measure.py`
- Modify: relevant measurement tests under `test/srt/`

**Interfaces:**
- Emits fallback fields in raw JSONL and derived CSV.

- [ ] Add failing measurement tests for the four fallback fields and fallback stage timing.
- [ ] Add fields to request samples, status samples, timeline, and summary outputs.
- [ ] Run all PD flip focused tests and diff checks.
- [ ] Sync changed files to cloud-099 through cloud-102.
- [ ] Re-run the saved 40-request interleaved trace and capture raw data/logs/report.
- [ ] Verify successful fallback reaches target activation and node2 runtime/router prefill role; otherwise preserve failure evidence and report the precise remaining blocker.
