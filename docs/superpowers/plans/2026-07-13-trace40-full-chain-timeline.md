# Trace40 SLO-Driven PD Full-Chain Timeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cloud-099 single-entry runner for the real request-level SLO progressive `node2 -> node3` D-to-P chain, with dual-source stitching enabled and stitch failure/full-source fallback measured separately.

**Architecture:** Extend the append-only trace SLO monitor with logical resets, expose the existing progressive controller through a deterministic CLI, enrich existing Worker/controller measurements, and orchestrate four hosts from one scoped Bash runner. Local tests exercise behavior without SSH, Docker, GPUs, or cluster writes.

**Tech Stack:** Python 3, pytest/unittest, Bash, Docker CLI, SSH, SGLang PD APIs, Mooncake, JSONL/CSV/Markdown.

## Global Constraints

- Keep dual-source HiCache stitching enabled.
- Record stitch restore failure and exactly one full-source fallback per RID as separate phases.
- Use 40 interleaved requests: 20 long 10,000-character and 20 short 1,000-character prompts, with per-request SLOs.
- Fix migration to node2 source and node3 target.
- Use first-N ratio 0.5 and repeatedly halve N on target shortage.
- Observation uses only events after logical reset.
- Never write credentials into artifacts or command logs.
- Local verification must not contact or start the cluster.
- Preserve pre-existing dirty-worktree changes.

---

### Task 1: Trace SLO logical window reset

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_trace_slo.py`
- Modify: `test/srt/test_pd_flip_trace_slo_monitor.py`

**Interfaces:**
- Consumes: append-only JSONL records with `request_id` and monotonic `event_time`.
- Produces: `TraceSLOMonitor.reset_window() -> None`; later collection ignores old events without truncating the ledger.

- [ ] **Step 1: Add a failing reset test**

Write an old record, reset, append a new record, then assert only the new record contributes and both lines remain:

```python
monitor = TraceSLOMonitor(
    ledger_path=str(ledger), window_seconds=30, client=client, time_fn=clock
)
append_record(ledger, request_id="old", event_time=100.0, ttft_met=False)
clock.value = 101.0
monitor.reset_window()
append_record(ledger, request_id="new", event_time=102.0, ttft_met=True)
clock.value = 103.0
snapshot = monitor.collect_cluster(nodes)
assert snapshot.prefill_counts.total == 1
assert snapshot.prefill_counts.good == 1
assert len(ledger.read_text().splitlines()) == 2
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest test/srt/test_pd_flip_trace_slo_monitor.py -q`

Expected: FAIL because `TraceSLOMonitor` lacks `reset_window`.

- [ ] **Step 3: Implement the cutoff**

```python
def reset_window(self) -> None:
    self.window_start_time = self.time_fn()

rolling = now - self.window_seconds if self.window_seconds > 0 else None
cutoffs = [x for x in (rolling, self.window_start_time) if x is not None]
cutoff = max(cutoffs) if cutoffs else None
```

Initialize `window_start_time` to `None`; retain the existing latest-per-request logic.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest test/srt/test_pd_flip_trace_slo_monitor.py -q`

Expected: all tests pass.

---

### Task 2: Deterministic progressive-monitor CLI

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_controller.sh`
- Modify: `test/srt/test_pd_flip_progressive_controller.py`
- Create: `test/srt/test_pd_flip_progressive_cli.py`

**Interfaces:**
- Consumes: trace ledger, source/target names, iteration count, poll interval.
- Produces: `monitor-progressive` calling `monitor_progressive` with fixed selection.

- [ ] **Step 1: Add failing parser/dispatch tests**

```python
args = build_arg_parser().parse_args([
    "--router-url", "http://router", "--node", NODE0,
    "monitor-progressive", "--trace-slo-ledger", "/raw/ledger.jsonl",
    "--source-name", "node2", "--migration-target-name", "node3",
    "--iterations", "120", "--poll-interval", "0.25",
])
assert args.command == "monitor-progressive"
assert args.source_name == "node2"
assert args.migration_target_name == "node3"
```

Patch only the CLI boundary and assert it calls `monitor_progressive`, never legacy `monitor`.

- [ ] **Step 2: Add failing fixed-selection tests**

Create a scenario where node1 is more loaded, pass node2/node3 explicitly, and assert the state trace still selects node2/node3. Assert unknown names fail before any POST.

- [ ] **Step 3: Verify RED**

Run: `python -m pytest test/srt/test_pd_flip_progressive_cli.py test/srt/test_pd_flip_progressive_controller.py -q`

Expected: FAIL because the command and selection parameters do not exist.

- [ ] **Step 4: Implement parser, dispatch, and selection**

```python
def monitor_progressive(
    self,
    slo_monitor,
    *,
    iterations,
    poll_interval_seconds=None,
    source_name=None,
    migration_target_name=None,
):
    # resolve explicit names with _find_metric; retain automatic defaults otherwise
```

Add `monitor-progressive` CLI arguments and construct `TraceSLOMonitor` in `main`. Add a matching `run_controller.sh` action that forwards `PD_FLIP_TRACE_SLO_LEDGER`, `SOURCE_NAME`, and `MIGRATION_TARGET_NAME`.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
python -m pytest test/srt/test_pd_flip_progressive_cli.py \
  test/srt/test_pd_flip_progressive_controller.py \
  test/srt/test_pd_flip_controller.py \
  test/srt/test_pd_flip_controller_quiesce.py -q
```

Expected: all tests pass.

---

### Task 3: Stitch/fallback phase timing

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `scripts/playground/disaggregation/pd_flip_migration_measure.py`
- Modify: `test/srt/test_pd_flip_hicache_stitch.py`
- Modify: `test/srt/test_pd_flip_migration_measure.py`

**Interfaces:**
- Consumes: existing timing helper, stitch metadata, fallback status, sampled events.
- Produces: per-RID exact timing fields and separate stitch/fallback report rows.

- [ ] **Step 1: Add failing Worker measurement tests**

Build a synthetic entry and assert these fields are exposed:

```python
assert measurement["target_hicache_prefix_match_s"] == 0.01
assert measurement["target_hicache_restore_duration_s"] == 0.02
assert measurement["stitch_failure_detected_at_s"] == 100.03
assert measurement["fallback_requested_at_s"] == 100.04
assert measurement["fallback_transfer_duration_s"] == 0.05
assert measurement["stitch_failure_to_fallback_complete_s"] == 0.06
assert measurement["failed_stitch_added_cost_s"] == 0.03
```

Use injected numeric timestamps; never sleep.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest test/srt/test_pd_flip_hicache_stitch.py -q`

Expected: FAIL because the phase fields are absent.

- [ ] **Step 3: Instrument exact boundaries**

Use `_pd_flip_note_timing` for:

```text
target_prefix_query_started / target_prefix_query_completed
target_hicache_restore_started / target_hicache_restore_failed
target_fallback_required
source_fallback_command_received / source_fallback_transfer_started
source_fallback_transfer_completed
target_fallback_prepare_received / target_fallback_receive_completed
```

Preserve existing timing keys. Include L1/L2/L3, H/P/C0, fallback reason, and token ranges.

- [ ] **Step 4: Add failing summarizer tests**

Feed source/target measurements into the summarizer and assert distinct `stitch_attempt`, `stitch_failure`, `full_source_fallback`, and combined rows. Missing byte metrics must remain null with `bytes_available=false`.

- [ ] **Step 5: Verify summarizer RED**

Run: `python -m pytest test/srt/test_pd_flip_migration_measure.py -q`

Expected: FAIL because derived phases are absent.

- [ ] **Step 6: Extend summaries**

Add:

```text
stitch_attempt_s
stitch_failure_detection_s
fallback_control_s
fallback_transfer_s
failed_stitch_added_cost_s
total_initial_migration_s
measurement_kind
```

Use `exact_process`, `wall_clock`, or `observed_poll_bound` for `measurement_kind`.

- [ ] **Step 7: Verify GREEN**

Run:

```bash
python -m pytest test/srt/test_pd_flip_hicache_stitch.py \
  test/srt/test_pd_flip_migration_measure.py \
  test/srt/test_pd_flip_migration_accounting.py -q
```

Expected: all tests pass.

---

### Task 4: Four-node trace40 coordinator

**Files:**
- Create: `experiments/pd_flip_trace40_full_chain.sh`
- Create: `test/srt/test_pd_flip_trace40_full_chain_runner.py`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/README.md`

**Interfaces:**
- Consumes: private env file and `preflight|run|status|collect|stop`.
- Produces: run-scoped artifacts; `DRY_RUN=1` renders redacted commands without external effects.

- [ ] **Step 1: Add failing static/dry-run tests**

Use fake `ssh`, `docker`, and `curl` binaries. Assert all hosts, fixed pair, ledger forwarding, 50 ms sampler, clock capture, run-scoped names, and no secret output. Assert `preflight` never starts/stops anything.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest test/srt/test_pd_flip_trace40_full_chain_runner.py -q`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement configuration and safety**

Use `set -euo pipefail`; validate host/node/trace/image/repository variables; generate a unique `RUN_ID`; reject existing directories; tag ephemeral containers `tiancij-pd-${RUN_ID}-<component>`; redact secrets from `commands.log`.

- [ ] **Step 4: Implement subcommands**

`preflight` checks SSH, image, repository, trace, clocks, ports, roles, idle migration, and stitch-enabled arguments. `run` starts measurement, workload, then progressive controller. `status` is read-only. `collect` gathers all raw data and invokes the summarizer. `stop` targets only exact run-owned names.

- [ ] **Step 5: Implement EXIT handling**

Always record exit codes and collect artifacts; do not stop shared Worker/router/Mooncake containers; do not mask Controller/workload failure.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
python -m pytest test/srt/test_pd_flip_trace40_full_chain_runner.py -q
bash -n experiments/pd_flip_trace40_full_chain.sh
```

Expected: tests pass and shell syntax is valid.

---

### Task 5: Full local acceptance

**Files:**
- Modify only Task 1-4 files if verification exposes defects.

**Interfaces:**
- Consumes: completed implementation.
- Produces: locally verified afternoon commands; no cluster action.

- [ ] **Step 1: Run focused suite**

```bash
python -m pytest \
  test/srt/test_pd_flip_trace_slo_monitor.py \
  test/srt/test_pd_flip_progressive_cli.py \
  test/srt/test_pd_flip_progressive_controller.py \
  test/srt/test_pd_flip_hicache_stitch.py \
  test/srt/test_pd_flip_migration_measure.py \
  test/srt/test_pd_flip_trace40_full_chain_runner.py -q
```

- [ ] **Step 2: Run adjacent regressions**

```bash
python -m pytest \
  test/srt/test_pd_flip_controller.py \
  test/srt/test_pd_flip_controller_quiesce.py \
  test/srt/test_pd_flip_reconciliation.py \
  test/srt/test_pd_flip_atomic_batch.py \
  test/srt/test_pd_flip_migration_accounting.py -q
```

- [ ] **Step 3: Verify syntax and diffs**

```bash
python -m compileall -q scripts/playground/disaggregation
bash -n experiments/pd_flip_trace40_full_chain.sh
git diff --check
git status --short
```

- [ ] **Step 4: Dry-run without effects**

Run the coordinator with test-owned fake commands. Verify enabled stitching, ledger, fixed pair, ratio, observation, raw collection, and no literal credential.

- [ ] **Step 5: Prepare the afternoon commands**

```bash
cd /home/tiancij/sglang-pd-e9c4472c3
ENV_FILE=/home/tiancij/trace40-full-chain.env \
  bash experiments/pd_flip_trace40_full_chain.sh preflight
ENV_FILE=/home/tiancij/trace40-full-chain.env \
  bash experiments/pd_flip_trace40_full_chain.sh run
```

Do not execute these commands during local implementation.

