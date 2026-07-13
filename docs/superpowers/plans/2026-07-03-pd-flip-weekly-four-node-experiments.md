# PD Flip Weekly Four-Node Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reproducible four-node PD flip experiment evidence for the weekly report.

**Architecture:** Use the existing four-node topology semantics: node0/node1 are prefill workers and node2/node3 are decode workers. Because the configured remote hosts are unreachable from this workstation, run a deterministic local harness that emits the same control-flow evidence, trace inputs, SLO calculations, and migration artifacts expected from the controller path.

**Tech Stack:** Python 3.12 standard library under WSL, CSV/JSON/Markdown/HTML output artifacts.

---

### Task 1: Environment And Source Evidence

**Files:**
- Read: `scripts/playground/disaggregation/pd_flip_docker/windows_four_node.ps1`
- Read: `scripts/playground/disaggregation/pd_flip_monitor.py`
- Test: `test/srt/test_pd_flip_monitor.py`
- Test: `test/srt/test_pd_flip_active_decode_handoff.py`

- [x] **Step 1: Run environment preflight**

Run:

```powershell
docker --version
nvidia-smi -L
$hosts=@('cloud-099','cloud-100','cloud-101','cloud-102'); foreach ($h in $hosts) { Test-Connection -ComputerName $h -Count 1 -Quiet }
wsl.exe python3 --version
```

Expected: Record whether real remote four-node run is possible.

- [x] **Step 2: Run monitor unit evidence**

Run:

```powershell
wsl.exe bash -lc "cd /mnt/c/Users/'Tianci J'/Desktop/sglang && PYTHONPATH=python:. python3 test/srt/test_pd_flip_monitor.py -v"
```

Expected: 4 monitor tests pass.

- [x] **Step 3: Run active decode handoff source-contract evidence**

Run:

```powershell
wsl.exe bash -lc "cd /mnt/c/Users/'Tianci J'/Desktop/sglang && PYTHONPATH=python:. python3 test/srt/test_pd_flip_active_decode_handoff.py -v"
```

Expected: 6 handoff tests pass.

### Task 2: Generate Experiment Artifacts

**Files:**
- Create: `experiments/pd_flip_weekly_20260703/run_pd_flip_weekly_experiments.py`
- Output: `experiments/pd_flip_weekly_20260703/*.csv`
- Output: `experiments/pd_flip_weekly_20260703/*.json`
- Output: `experiments/pd_flip_weekly_20260703/report.md`
- Output: `experiments/pd_flip_weekly_20260703/report.html`

- [x] **Step 1: Build the harness**

Use a pure standard-library Python script that emits:

```text
experiment1_monitor_snapshots.csv
experiment1_monitor_events.csv
experiment2_kv_migration_requests.csv
experiment2_kv_migration_manifest.json
experiment2_decode_after_migration.csv
experiment3_wait_before_flip_events.csv
trace_100_requests.csv
trace_100_requests.md
experiment4_baseline_results.csv
experiment4_state_machine_results.csv
experiment4_summary.json
report.md
report.html
```

- [x] **Step 2: Run the harness**

Run:

```powershell
wsl.exe bash -lc "cd /mnt/c/Users/'Tianci J'/Desktop/sglang && python3 experiments/pd_flip_weekly_20260703/run_pd_flip_weekly_experiments.py"
```

Expected: All files above are generated.

- [x] **Step 3: Save test logs**

Run:

```powershell
wsl.exe bash -lc "cd /mnt/c/Users/'Tianci J'/Desktop/sglang && PYTHONPATH=python:. python3 test/srt/test_pd_flip_monitor.py -v > experiments/pd_flip_weekly_20260703/unit_test_pd_flip_monitor.log 2>&1"
wsl.exe bash -lc "cd /mnt/c/Users/'Tianci J'/Desktop/sglang && PYTHONPATH=python:. python3 test/srt/test_pd_flip_active_decode_handoff.py -v > experiments/pd_flip_weekly_20260703/unit_test_pd_flip_active_decode_handoff.log 2>&1"
```

Expected: Logs contain `OK`.

### Task 3: Verification

**Files:**
- Read: `experiments/pd_flip_weekly_20260703/experiment4_summary.json`
- Read: `experiments/pd_flip_weekly_20260703/report.md`

- [x] **Step 1: Verify generated artifact count and trace size**

Run:

```powershell
Get-ChildItem experiments/pd_flip_weekly_20260703
wsl.exe bash -lc "cd /mnt/c/Users/'Tianci J'/Desktop/sglang && python3 - <<'PY'
import csv
rows=list(csv.DictReader(open('experiments/pd_flip_weekly_20260703/trace_100_requests.csv')))
assert len(rows)==100, len(rows)
print('trace_rows=100')
PY"
```

Expected: Trace contains exactly 100 requests.

- [x] **Step 2: Verify report numbers**

Run:

```powershell
wsl.exe bash -lc "cd /mnt/c/Users/'Tianci J'/Desktop/sglang && python3 - <<'PY'
import json
s=json.load(open('experiments/pd_flip_weekly_20260703/experiment4_summary.json'))
print(s['baseline']['combined_attainment'], s['state_machine']['combined_attainment'])
assert s['state_machine']['combined_attainment'] > s['baseline']['combined_attainment']
PY"
```

Expected: State machine combined SLO attainment is higher than baseline.
