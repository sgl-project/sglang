# Controller Monitor FSM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pd_flip_controller.py monitor` expose and drive an explicit controller-owned SAFE -> PREPARING -> FLIPPING -> SAFE state machine without changing worker-side flip APIs.

**Architecture:** Keep the state machine local to the controller monitor loop. Reuse the existing D->P two-phase HTTP actions and P->D drain/role-switch actions, but record each phase as structured monitor state in the returned `MonitorLoopResult`. Do not change worker modules, KV migration endpoints, source selection, or PD-ratio policy.

**Tech Stack:** Python dataclasses, existing unittest suite, existing fake HTTP clients in `test/srt/test_pd_flip_controller.py`.

---

### Task 1: Add Monitor State Result Coverage

**Files:**
- Modify: `test/srt/test_pd_flip_controller.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert monitor results include a per-state trace:

```python
def test_monitor_committed_d_to_p_reports_state_trace(self):
    client = ExecutingFakeClient()
    controller = self.script.PDFlipController(self.config, client)
    slo_monitor = SequenceSLOMonitor(self.script, [0.80, 0.80, 0.80])

    result = controller.monitor(
        slo_monitor=slo_monitor,
        enter_threshold=0.90,
        exit_threshold=0.95,
        commit_threshold=0.90,
        iterations=1,
        poll_interval_seconds=0.0,
    )

    states = [entry["state"] for entry in result.state_trace]
    self.assertEqual(
        states,
        ["safe", "preparing_kv_transfer", "flipping_role", "safe"],
    )
    self.assertEqual(result.state_trace[1]["direction"], "d_to_p")
    self.assertEqual(result.state_trace[-1]["role_after"], "prefill")
```

Add a second test for SLO recovery:

```python
def test_monitor_recovered_d_to_p_reports_abort_state_trace(self):
    client = ExecutingFakeClient()
    controller = self.script.PDFlipController(self.config, client)
    slo_monitor = SequenceSLOMonitor(self.script, [0.80, 0.96])

    result = controller.monitor(
        slo_monitor=slo_monitor,
        enter_threshold=0.90,
        exit_threshold=0.95,
        commit_threshold=0.90,
        iterations=1,
        poll_interval_seconds=0.0,
    )

    states = [entry["state"] for entry in result.state_trace]
    self.assertEqual(states, ["safe", "preparing_kv_transfer", "safe"])
    self.assertEqual(result.state_trace[-1]["reason"], "slo_recovered")
```

- [ ] **Step 2: Verify red**

Run:

```bash
python3 -m unittest test/srt/test_pd_flip_controller.py
```

Expected: the new tests fail with `AttributeError` or missing `state_trace`.

### Task 2: Add Controller Monitor State Data Structures

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py`

- [ ] **Step 1: Implement minimal dataclasses and result field**

Add a `MonitorState` string namespace and `MonitorStateRecord` dataclass. Extend `MonitorLoopResult` with `state_trace: List[JsonDict] = field(default_factory=list)`.

- [ ] **Step 2: Verify existing monitor tests still serialize**

Run:

```bash
python3 -m unittest test/srt/test_pd_flip_controller.py
```

Expected: new tests may still fail until Task 3; existing tests keep passing.

### Task 3: Route D->P Through Explicit Monitor States

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py`
- Modify: `test/srt/test_pd_flip_controller.py`

- [ ] **Step 1: Thread state trace through monitor D->P path**

Initialize trace with `safe` before a decision. When prefill SLO is risky, append `preparing_kv_transfer` before drain/pause/start/prepare. If SLO recovers during preparing, append `safe` with `reason=slo_recovered`. If SLO remains risky, append `flipping_role` before commit/finish/role switch, then append final `safe` with `role_after=prefill`.

- [ ] **Step 2: Verify green**

Run:

```bash
python3 -m unittest test/srt/test_pd_flip_controller.py
```

Expected: all controller tests pass.

### Task 4: Add P->D State Trace Without Changing Behavior

**Files:**
- Modify: `test/srt/test_pd_flip_controller.py`
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py`

- [ ] **Step 1: Write failing P->D monitor trace test**

Use a fake SLO monitor with decode attainment below threshold and assert:

```python
["safe", "preparing_drain", "flipping_role", "safe"]
```

- [ ] **Step 2: Verify red**

Run the controller test file and confirm only the new P->D state trace test fails.

- [ ] **Step 3: Add trace around existing `execute(direction="p_to_d")` monitor path**

Append `preparing_drain`, `flipping_role`, and final `safe` records around the existing P->D execution call. Keep HTTP action ordering unchanged.

- [ ] **Step 4: Verify green**

Run:

```bash
python3 -m unittest test/srt/test_pd_flip_controller.py
```

Expected: all controller tests pass.

### Task 5: Regression Verification

**Files:**
- No production changes.

- [ ] **Step 1: Run PD flip test group**

Run:

```bash
python3 -m unittest discover -s test/srt -p 'test_pd_flip*.py'
```

Expected: all PD flip tests pass.

- [ ] **Step 2: Confirm no worker module changes**

Run:

```bash
git diff --stat
```

Expected: this task adds tests and modifies `pd_flip_controller.py`; no worker-side FSM or scheduler files are changed by this monitor FSM work.
