# PD Flip Rollover Blocker Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the exact internal predicate and RID that blocks a second PD migration session without changing migration behavior.

**Architecture:** Extract the existing rollover predicate into a pure structured-diagnostics helper and keep the boolean helper as a compatibility wrapper. Surface the bounded blocker list in conflict responses and migration status, then reproduce with the existing trace40 runner.

**Tech Stack:** Python, pytest/unittest, SGLang scheduler and existing four-node Docker experiment.

## Global Constraints

- Do not relax, reorder, or delete any migration safety condition.
- Do not automatically clear, archive, or replace a migration session.
- Do not log prompts, outputs, credentials, or KV contents.
- Preserve `_pd_flip_can_rollover_session(session, next_role) -> bool`.
- Blocker records contain only stable codes, optional RID, and scalar observed values.
- Execute in the existing workspace; do not create a git worktree.

---

### Task 1: Structured rollover blockers

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py:1780-1835`
- Modify: `python/sglang/srt/managers/scheduler.py:1860-1880`
- Modify: `python/sglang/srt/managers/scheduler.py:2025-2045`
- Modify: `python/sglang/srt/managers/scheduler.py:4880-4970`
- Test: `test/srt/test_pd_flip_reconciliation.py`

**Interfaces:**
- Produces: `Scheduler._pd_flip_rollover_blockers(session, next_role) -> List[Dict[str, Any]]`
- Preserves: `Scheduler._pd_flip_can_rollover_session(session, next_role) -> bool`
- Adds status field: `rollover_blockers`

- [ ] **Step 1: Write failing tests**

Add tests constructing source-released sessions with explicit entries:

```python
def test_worker_rollover_blockers_identify_source_entry_flag():
    session = worker_with_session("source", "source_released").pd_flip_migration_session
    session["source_entries"] = {
        "r0": {
            "metadata_freed": True,
            "transferred": False,
            "final_owner": "target",
        }
    }
    assert Scheduler._pd_flip_rollover_blockers(session, "source") == [
        {"code": "source_entry_not_transferred", "rid": "r0"}
    ]


def test_worker_rollover_blockers_empty_for_safe_source_entry():
    session = worker_with_session("source", "source_released").pd_flip_migration_session
    session["source_entries"] = {
        "r0": {
            "metadata_freed": True,
            "transferred": True,
            "final_owner": "target",
            "delta": {"noop": False, "transferred": True, "metadata_freed": True},
        }
    }
    assert Scheduler._pd_flip_rollover_blockers(session, "source") == []
    assert Scheduler._pd_flip_can_rollover_session(session, "source")
```

Add a conflict-response test asserting `source_entry_not_transferred` appears in both the response message and `output.status["rollover_blockers"]`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
wsl -e bash -lc "cd /mnt/c/Users/Tianci\ J/Desktop/sglang && python3 -m pytest -q test/srt/test_pd_flip_reconciliation.py -k rollover_blockers"
```

Expected: failure because `_pd_flip_rollover_blockers` and the status field do not exist.

- [ ] **Step 3: Implement the pure helper**

Implement stable blocker codes for every condition currently present in `_pd_flip_can_rollover_session`. Make the boolean helper return `not Scheduler._pd_flip_rollover_blockers(...)`. Keep entry iteration deterministic by sorted RID.

- [ ] **Step 4: Surface blockers**

On source/target conflict, compute blockers once and append their compact JSON representation to the existing message. Add `rollover_blockers` to both empty and active migration status dictionaries.

- [ ] **Step 5: Verify GREEN**

Run the focused tests and the complete reconciliation test module. Expected: all pass.

- [ ] **Step 6: Run related regression tests**

Run:

```powershell
wsl -e bash -lc "cd /mnt/c/Users/Tianci\ J/Desktop/sglang && python3 -m pytest -q test/srt/test_pd_flip_reconciliation.py test/srt/test_pd_flip_atomic_batch.py test/srt/test_pd_flip_migration_accounting.py"
```

Expected: all selected tests pass.

### Task 2: Four-node reproduction

**Files:**
- Sync: `python/sglang/srt/managers/scheduler.py` to cloud-099 through cloud-102.
- Reuse: `experiments/pd_flip_trace40_full_chain.sh`
- Produce: a new directory under `/home/tiancij/pd-artifacts/`

**Interfaces:**
- Consumes the existing trace40 environment with TTFT override 0.03 seconds.
- Produces controller logs containing `rollover_blockers`.

- [ ] **Step 1: Compile on all nodes**

Run `python3 -m py_compile` inside each node's SGLang container against the synchronized scheduler.

- [ ] **Step 2: Establish clean initial state**

Confirm no experiment sidecars are active. Restart only the user-owned node2/node3 containers if an old source/target session remains. Verify roles P,D,D,D and migration state `none`.

- [ ] **Step 3: Run the existing 40-request experiment**

Start the runner with `TRACE_TTFT_SLO_OVERRIDE_SECONDS=0.03`, ratio 0.5, observation 10 seconds, and raw/log collection enabled.

- [ ] **Step 4: Classify the second-session outcome**

Record the remaining RID list immediately before second start. If empty, verify the controller skips migration and switches role. If nonempty and rejected, extract the exact blocker code/RID from the response and status.

- [ ] **Step 5: Preserve evidence**

Download the new tarball, validate hashes and raw counts, and append the diagnosed root cause to the experiment report. Do not claim the identity switch succeeded unless node2 reports Prefill and all workers remain healthy after workload completion.
