# Monitor-Driven PD Flip State Machine Four-Node Docker Plan

**Goal:** Complete the PD flip state machine so a four-node Docker experiment can run end-to-end: a monitor computes per-node SLO attainment, enters `preparing` when the P/D ratio policy crosses a threshold, starts decode-to-decode KV pre-transfer without changing role, returns to `safe` if SLO recovers, or enters `flipping` and finishes in the new role's `safe` state if SLO remains risky.

**Architecture:** Keep SGLang workers as role/migration executors and make the monitor/controller the cluster-level decision owner. The monitor reads TTFT/TPOT and queue/load signals from all workers, updates the state machine with cluster snapshots, drives KV pre-transfer during `preparing`, and commits or aborts based on continued SLO observations. The router must route away from workers in `preparing` or `flipping`, and the Docker harness must make this reproducible on four physical nodes.

**Tech Stack:** Python state machine and controller, SGLang worker admin endpoints, existing decode-to-decode KV migration endpoints, Rust/SMG router PD admin APIs, Prometheus or `/metrics` scraping, `/v1/loads`, Docker host networking, SSH aliases `cloud-099` through `cloud-102`.

---

## Target Four-Node Layout

Use the SSH host aliases already configured locally:

```text
cloud-099 -> node0 -> initial prefill
cloud-100 -> node1 -> initial prefill
cloud-101 -> node2 -> initial decode, primary D->P flip candidate
cloud-102 -> node3 -> initial decode, KV migration target
```

Run one 8-GPU SGLang worker container per physical node. Run the router and controller on `cloud-099` unless network checks show a better placement.

Initial serving topology:

```text
node0: prefill
node1: prefill
node2: decode
node3: decode
```

Expected D->P experiment topology after a successful flip:

```text
node0: prefill
node1: prefill
node2: prefill
node3: decode
```

The reverse P->D case should also be supported by the state machine, but the first full Docker acceptance run focuses on D->P because it exercises active decode KV pre-transfer.

---

## Final State Machine Contract

### States

- `safe`: worker is serving its current role normally.
- `preparing`: selected source worker stops new admission, keeps its current role, and starts KV pre-transfer to a target decode worker.
- `flipping`: migration is committed and the source worker is switching identity to the requested role.
- `safe(new_role)`: source worker has completed identity switch and router metadata matches the new role.

### Transitions

```text
safe
  -> preparing
     when monitor sees P/D SLO policy cross the enter threshold

preparing
  -> safe
     when SLO recovers below the exit threshold before commit

preparing
  -> flipping
     when KV pre-transfer is complete and SLO remains above the commit threshold

flipping
  -> safe(new_role)
     when runtime role switch or restart-based role switch succeeds

preparing/flipping
  -> safe(current_role)
     on timeout, migration failure before commit, controller abort, or role switch failure
```

### Monitor Inputs

- Prefill SLO attainment: fraction of prefill requests whose TTFT is under the configured TTFT SLO.
- Decode SLO attainment: fraction of decode token intervals whose TPOT is under the configured TPOT SLO.
- Queue/load signals: `num_running_reqs`, `num_waiting_reqs`, token usage, decode transfer queue depth, prefill inflight queue depth.
- Node state: current role, FSM state, migration state, admission paused, router draining flag.

### Policy

- Maintain a sliding window per node and role.
- Aggregate by role to get cluster prefill and decode attainment.
- Evaluate nearby P/D ratios and choose the smallest role change that improves the risky SLO.
- Use hysteresis:
  - enter threshold: start preparing only after the risky score is above threshold for N windows.
  - exit threshold: abort preparing only after recovery is stable for M windows.
  - commit threshold: commit only if risk remains after KV pre-transfer completes.

---

## Acceptance Criteria

- Unit tests cover all state transitions, including `preparing -> safe` recovery and `preparing -> flipping` commit.
- Controller tests cover monitor-driven source/target selection, migration start, migration abort, migration commit, role switch, router refresh, and cleanup.
- Router tests prove workers in `preparing` and `flipping` are excluded from new traffic.
- Worker tests prove active decode migration can copy KV without immediately adopting target requests, and can later commit or abort.
- Docker scripts can launch four workers, one router, one monitor/controller, and one load generator using the `cloud-099..102` SSH aliases.
- Four-node D->P experiment demonstrates both branches:
  - Recovery branch: trigger `preparing`, complete or partially complete KV pre-transfer, improve SLO, abort migration, return to `safe(decode)`.
  - Commit branch: trigger `preparing`, keep SLO risky, complete KV pre-transfer, enter `flipping`, switch node2 to prefill, return to `safe(prefill)`.
- Experiment artifacts include command logs, monitor decisions, `/server_info` snapshots, migration timings, router worker states, TTFT/TPOT attainment windows, and final topology.

---

## Implementation Plan

### Current Local Execution Status

Completed in the local checkout:

- State machine supports `preparing -> safe` recovery, `preparing` wait reasons,
  and `preparing -> flipping` only after KV pre-transfer and commit decision.
- Monitor model parses worker TTFT/TPOT histograms, keeps sliding-window
  attainment, and emits serializable cluster snapshots.
- P/D policy supports enter/exit/commit thresholds and hysteresis.
- Decode KV migration target supports two-phase `prepare_only`, target
  `commit`, and target `abort`.
- Controller has a `monitor` subcommand covering safe/no-op, recovery abort, and
  risky commit branches.
- Docker harness has four-node `cloud-099..102` defaults, monitor runner, and
  SSH/tmux launch notes.
- Windows-side SSH orchestration is available via
  `scripts/playground/disaggregation/pd_flip_docker/windows_four_node.ps1` for
  environments where Windows can reach the cloud aliases but WSL cannot.

Still requires the real four-node environment:

- Router cargo tests or router-side PD-state exclusion checks.
- SSH/GPU/Docker/RDMA/model-path preflight on `cloud-099..102`.
- Real Docker acceptance runs for baseline, recovery, commit, and failure
  branches with saved artifacts.

Local verification used direct `unittest` and module-level checks because this
machine does not currently have `pytest` or `numpy` installed.

Read-only SSH preflight was attempted from WSL and timed out during banner
exchange for all four aliases, resolving through `198.18.0.23..26`. The user's
Windows host can connect to `cloud-099`, so cluster validation should be run from
Windows with `windows_four_node.ps1` after the Windows checkout and remote cloud
checkouts contain this same code.

### Task 1: Lock Current Behavior With Tests

**Files:**
- Modify: `test/srt/test_pd_flip_state_machine.py`
- Modify: `test/srt/test_pd_flip_controller.py`
- Modify: `test/srt/test_pd_flip_experiment_script.py`

- [ ] Add tests documenting the current `safe -> preparing -> flipping -> safe` happy path.
- [ ] Add failing tests for `preparing -> safe` when SLO recovers.
- [ ] Add failing tests for `preparing` staying in current role until commit.
- [ ] Add failing tests for aborting migration before target adoption.
- [ ] Run:

```bash
PYTHONPATH=python python -m pytest \
  test/srt/test_pd_flip_state_machine.py \
  test/srt/test_pd_flip_controller.py \
  test/srt/test_pd_flip_experiment_script.py -q
```

### Task 2: Extend State Machine Semantics

**Files:**
- Modify: `python/sglang/srt/disaggregation/flip_state_machine.py`
- Modify: `python/sglang/srt/managers/scheduler.py`
- Test: `test/srt/test_pd_flip_state_machine.py`

- [ ] Add explicit transition reason metadata for `slo_recovered`, `kv_ready`, `slo_still_risky`, `migration_failed`, and `flip_failed`.
- [ ] Re-evaluate SLO while in `preparing`.
- [ ] Allow `preparing -> safe` without entering `flipping`.
- [ ] Require both `kv_pretransfer_complete` and `commit_decision=True` for `preparing -> flipping`.
- [ ] Keep `flipping -> safe` dependent on role switch completion.
- [ ] Expose status fields: `prepare_started_at`, `prepare_elapsed_seconds`, `commit_ready`, `abort_ready`, `migration_commit_mode`, and `last_recovery_reason`.

Verification:

```bash
PYTHONPATH=python python -m pytest test/srt/test_pd_flip_state_machine.py -q
```

### Task 3: Build Monitor SLO Attainment Model

**Files:**
- Add: `scripts/playground/disaggregation/pd_flip_monitor.py`
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py`
- Test: `test/srt/test_pd_flip_monitor.py`
- Test: `test/srt/test_pd_flip_controller.py`

- [ ] Implement a sliding-window data model for per-node TTFT and TPOT samples.
- [ ] Parse worker `/metrics` counters or histograms for `sglang:time_to_first_token_seconds` and `sglang:inter_token_latency_seconds`.
- [ ] Fall back to router metrics `smg_router_ttft_seconds` and `smg_router_tpot_seconds` when worker metrics are unavailable.
- [ ] Fetch `/v1/loads?include=all` for queue/load signals.
- [ ] Compute:
  - `prefill_slo_attainment = ttft_good / ttft_total`
  - `decode_slo_attainment = tpot_good / tpot_total`
  - per-node queue pressure and cluster P/D counts
- [ ] Emit a `ClusterSLOSnapshot` object that can be serialized into controller logs and worker `/set_internal_state`.

Verification:

```bash
PYTHONPATH=python python -m pytest \
  test/srt/test_pd_flip_monitor.py \
  test/srt/test_pd_flip_controller.py -q
```

### Task 4: Replace Threshold-Only Evaluator With P/D Ratio Policy

**Files:**
- Modify: `python/sglang/srt/disaggregation/flip_state_machine.py`
- Modify: `scripts/playground/disaggregation/pd_flip_monitor.py`
- Test: `test/srt/test_pd_flip_state_machine.py`
- Test: `test/srt/test_pd_flip_monitor.py`

- [ ] Add a policy object that evaluates current P/D ratio and nearby ratios.
- [ ] Support configurable thresholds:
  - `enter_threshold`
  - `exit_threshold`
  - `commit_threshold`
  - `min_enter_windows`
  - `min_exit_windows`
- [ ] Choose D->P when prefill TTFT attainment is risky and decode capacity can donate a node.
- [ ] Choose P->D when decode TPOT attainment is risky and prefill capacity can donate a node.
- [ ] Keep the existing simple threshold evaluator as a test/helper path.

Verification:

```bash
PYTHONPATH=python python -m pytest \
  test/srt/test_pd_flip_state_machine.py \
  test/srt/test_pd_flip_monitor.py -q
```

### Task 5: Make KV Pre-Transfer Two-Phase

**Files:**
- Modify: `python/sglang/srt/managers/io_struct.py`
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `python/sglang/srt/managers/tokenizer_control_mixin.py`
- Modify: `python/sglang/srt/entrypoints/http_server.py`
- Test: `test/srt/test_pd_flip_active_decode_handoff.py`

- [ ] Split target migration behavior into `prepare_only` and `adopt_on_commit`.
- [ ] Ensure target can receive KV and hold reconstructed requests without scheduling them.
- [ ] Add `POST /pd_flip/migration/target/commit` to adopt held requests after monitor commits the flip.
- [ ] Add `POST /pd_flip/migration/target/abort` to release target-side held KV and metadata.
- [ ] Ensure source requests are not released until target commit succeeds.
- [ ] Keep existing `adopt_on_success=True` path for old controller tests, but mark it legacy/demo.

Verification:

```bash
PYTHONPATH=python python -m pytest test/srt/test_pd_flip_active_decode_handoff.py -q
```

### Task 6: Add Monitor-Driven Controller Loop

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py`
- Add/modify: `scripts/playground/disaggregation/pd_flip_monitor.py`
- Test: `test/srt/test_pd_flip_controller.py`
- Test: `test/srt/test_pd_flip_monitor.py`

- [ ] Add `monitor` subcommand that runs continuously.
- [ ] Let monitor select source and migration target based on current roles, SLO risk, and queue pressure.
- [ ] On enter decision:
  - drain source in router
  - pause source admission
  - start source migration
  - prepare target migration with `prepare_only`
- [ ] During `preparing`:
  - continue SLO sampling
  - abort if SLO recovery is stable
  - commit if KV transfer completed and risk remains
- [ ] During `flipping`:
  - switch source runtime role or invoke restart command
  - refresh router role
  - resume admission
  - undrain source
- [ ] On any failure:
  - abort source and target migration sessions
  - resume source admission
  - undrain source
  - keep source in original role

Verification:

```bash
PYTHONPATH=python python -m pytest \
  test/srt/test_pd_flip_controller.py \
  test/srt/test_pd_flip_monitor.py -q
```

### Task 7: Router And Worker State Integration

**Files:**
- Modify: `sgl-model-gateway/src/routers/http/pd_router.rs`
- Modify: router admin files under `sgl-model-gateway/src`
- Test: router tests under `sgl-model-gateway/tests` or existing proxy/router test paths

- [ ] Ensure router excludes workers with `pd_flip.state in {preparing, flipping}`.
- [ ] Ensure router admin worker list exposes role, URL, drain status, active load, and bootstrap port.
- [ ] Add or update endpoint to refresh worker role after flip.
- [ ] Add test for stale worker `/server_info` query failure: router should fail open only when no PD flip label says draining.
- [ ] Add test for explicit `pd_flip_state=preparing` label.

Verification:

```bash
cd sgl-model-gateway
cargo test pd_flip --all-targets
```

### Task 8: Docker Harness For Four Nodes

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_docker/env.example`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_worker.sh`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_router.sh`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_controller.sh`
- Add: `scripts/playground/disaggregation/pd_flip_docker/run_monitor.sh`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/README.md`
- Test: `test/srt/test_pd_flip_experiment_script.py`

- [ ] Add a host mapping section for `cloud-099`, `cloud-100`, `cloud-101`, `cloud-102`.
- [ ] Keep one physical node in one PD role with all local GPUs.
- [ ] Add environment variables:
  - `NODE0_HOST=cloud-099`
  - `NODE1_HOST=cloud-100`
  - `NODE2_HOST=cloud-101`
  - `NODE3_HOST=cloud-102`
  - `NODE0_ROLE=prefill`
  - `NODE1_ROLE=prefill`
  - `NODE2_ROLE=decode`
  - `NODE3_ROLE=decode`
  - `TTFT_SLO_SECONDS`
  - `TPOT_SLO_SECONDS`
  - `PD_FLIP_ENTER_THRESHOLD`
  - `PD_FLIP_EXIT_THRESHOLD`
  - `PD_FLIP_COMMIT_THRESHOLD`
- [ ] Add `run_monitor.sh` that starts the monitor/controller against the four worker URLs and router URL.
- [ ] Document how to run each script over SSH and tmux.
- [ ] Add smoke test that checks all expected env keys and scripts exist.

Verification:

```bash
PYTHONPATH=python python -m pytest test/srt/test_pd_flip_experiment_script.py -q
```

### Task 9: Four-Node Preflight

**Execution environment:** local machine plus SSH aliases.

- [ ] Verify SSH access:

```bash
for h in cloud-099 cloud-100 cloud-101 cloud-102; do
  ssh "$h" 'hostname; nvidia-smi -L | wc -l; docker --version'
done
```

- [ ] Verify image availability on all nodes.
- [ ] Verify all nodes can reach each worker serve port, bootstrap port, and router port.
- [ ] Verify RDMA or configured transfer backend path.
- [ ] Verify shared model path or image-contained model path exists on every node.
- [ ] Record:
  - GPU count
  - Docker image tag
  - SGLang git commit
  - transfer backend
  - model path
  - network interface/IP selected for bootstrap

### Task 10: Four-Node Docker Acceptance Runs

**Run A: Baseline safe serving**

- [ ] Start node0/node1 as prefill.
- [ ] Start node2/node3 as decode.
- [ ] Start router.
- [ ] Send steady traffic through router.
- [ ] Confirm TTFT/TPOT metrics and `/v1/loads` are visible for all nodes.

**Run B: Recovery branch**

- [ ] Inject or generate prefill SLO risk so node2 enters `preparing`.
- [ ] Start KV pre-transfer from node2 to node3.
- [ ] Reduce pressure so SLO recovers before commit.
- [ ] Confirm monitor aborts migration.
- [ ] Confirm node2 returns to `safe(decode)`.
- [ ] Confirm router undrains node2 as decode.

**Run C: Commit branch**

- [ ] Generate sustained prefill SLO risk.
- [ ] Confirm node2 enters `preparing`.
- [ ] Confirm KV pre-transfer completes to node3.
- [ ] Confirm monitor keeps risk above commit threshold.
- [ ] Confirm node2 enters `flipping`.
- [ ] Confirm node2 reaches `safe(prefill)`.
- [ ] Confirm router sees node2 as prefill.
- [ ] Confirm final topology is 3 prefill / 1 decode.

**Run D: Failure branch**

- [ ] Force target migration failure or timeout.
- [ ] Confirm source remains original decode role.
- [ ] Confirm source admission and router drain are cleaned up.
- [ ] Confirm target held KV/resources are released.

Artifacts to save:

```text
/tmp/pd-flip-monitor-four-node/
  preflight/
  baseline-safe/
  recovery-branch/
  commit-branch/
  failure-branch/
```

Each run directory should include:

- controller/monitor JSONL log
- router worker state snapshots
- worker `/server_info` snapshots
- worker `/v1/loads` snapshots
- TTFT/TPOT SLO window summaries
- migration status snapshots
- Docker logs
- final result summary markdown

---

## Completion Definition

This plan is complete when a fresh checkout can:

1. Build or pull the required Docker image.
2. Start the four-node PD topology on `cloud-099..102`.
3. Run the monitor/controller loop without manual `/set_internal_state` injection for normal decisions.
4. Demonstrate both `preparing -> safe` recovery and `preparing -> flipping -> safe(new_role)` commit.
5. Preserve serving correctness during the experiment, with no unexpected 5xx from the router beyond intentionally aborted failure tests.
6. Produce a reproducible report under `docs/superpowers/reports/` with command transcript, metrics, and state timelines.
