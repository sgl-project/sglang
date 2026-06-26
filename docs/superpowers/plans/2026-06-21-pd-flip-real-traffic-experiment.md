# PD Flip Real Traffic Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the experimental PD flip path capable of moving an active decode request from one decode worker to another and running a real traffic D->P identity-switch experiment with observable request continuity.

**Architecture:** Implement this in four gates. First, make target-side migration resume a reconstructed request through the normal decode scheduler. Second, expose a clear continuation result channel so the experiment can observe target output without pretending client stream handoff is done. Third, add a controller that safely drains/migrates/restarts/commits a worker. Fourth, add router-facing hooks for real client stream handoff as a follow-up gate. This avoids mixing the server-side migration proof with the harder client-stream ownership problem.

**Tech Stack:** Python scheduler/runtime, FastAPI control endpoints, existing SGLang PD KV sender/receiver abstractions, pytest/unittest, sgl-model-gateway or experimental sgl-router for routing handoff.

---

## File Structure

- `python/sglang/srt/managers/scheduler.py`
  - Owns PD flip state, migration source/target sessions, target-side resume, and internal status.
- `python/sglang/srt/managers/io_struct.py`
  - Adds request/response structures for target continuation inspection if needed.
- `python/sglang/srt/entrypoints/http_server.py`
  - Exposes new experimental control endpoint(s), behind the same admin optional auth as existing PD flip migration APIs.
- `python/sglang/srt/managers/tokenizer_control_mixin.py`
  - Fans new control requests out to DP schedulers.
- `scripts/playground/disaggregation/pd_flip_experiment.py`
  - Becomes the first real experiment controller: observe, trigger, migrate, wait target resume, finish source, restart, commit.
- `test/srt/test_pd_flip_internal_state_update.py`
  - Scheduler-level unit tests for target resume, state/status, and prepare gating.
- `test/srt/test_disaggregation_fake_decode.py`
  - Fake-transfer tests for target migration pumping into the decode path.
- `test/srt/test_pd_flip_experiment_script.py`
  - Controller flow tests.
- `sgl-model-gateway/src/routers/http/pd_router.rs` or `experimental/sgl-router/src/server/routes/chat.rs`
  - Later gate: router stream handoff / retry behavior. Do not touch in the first server-only gate.

---

## Milestone 1: Server-Side Active Request Migration Resume

This milestone answers: after target receives KV, can the target decode worker continue the migrated request internally?

### Task 1: Add Target Resume Test

**Files:**
- Modify: `test/srt/test_disaggregation_fake_decode.py`
- Modify: `python/sglang/srt/managers/scheduler.py`

- [ ] **Step 1: Add a failing test that target migration enqueues the reconstructed request**

Append this test to `test/srt/test_disaggregation_fake_decode.py` near the existing `TestFakePDFlipTargetMigration` tests:

```python
    def test_pd_flip_target_transfer_enqueues_resumed_request(self):
        class Receiver:
            def __init__(self):
                self.has_metadata = False
                self.cleared = False

            def poll(self):
                return KVPoll.Success if self.has_metadata else KVPoll.WaitingForInput

            def clear(self):
                self.cleared = True

            def abort(self):
                raise AssertionError("abort should not be called")

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting_queue = []
        released = []
        freed_metadata = []

        def prealloc(entry):
            entry["decode_req"].kv_receiver.has_metadata = True

        scheduler._pd_flip_target_init_receiver = lambda decode_req: True
        scheduler._pd_flip_target_prealloc_and_send_metadata = prealloc
        scheduler._pd_flip_target_metadata_ready = lambda entry: True
        scheduler._pd_flip_release_target_request = lambda entry: released.append(
            entry["decode_req"].req.rid
        )
        scheduler._pd_flip_free_target_metadata = lambda entry: freed_metadata.append(
            entry["decode_req"].req.rid
        )

        receiver = Receiver()
        req = types.SimpleNamespace(
            rid="req",
            finished=lambda: False,
            time_stats=types.SimpleNamespace(set_wait_queue_entry_time=lambda: None),
        )
        session = {
            "manifests": [{"rid": "req"}],
            "target_entries": {
                "req": {
                    "decode_req": types.SimpleNamespace(
                        req=req, kv_receiver=receiver
                    ),
                    "phase": "new",
                }
            },
        }

        Scheduler._pd_flip_target_pump_transfer(scheduler, session)

        self.assertEqual([queued.rid for queued in scheduler.waiting_queue], ["req"])
        self.assertEqual(session["state"], "target_transferred")
        self.assertTrue(receiver.cleared)
        self.assertEqual(freed_metadata, ["req"])
        self.assertEqual(released, [])
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python -m pytest test/srt/test_disaggregation_fake_decode.py::TestFakePDFlipTargetMigration::test_pd_flip_target_transfer_enqueues_resumed_request -q
```

Expected: FAIL because `_pd_flip_target_pump_transfer` currently releases the target request instead of enqueuing it.

- [ ] **Step 3: Implement target resume helper**

In `python/sglang/srt/managers/scheduler.py`, add this method near `_pd_flip_release_target_request`:

```python
    def _pd_flip_enqueue_target_request(self, entry: Dict[str, Any]) -> None:
        if entry.get("target_enqueued"):
            return
        decode_req = entry.get("decode_req")
        req = getattr(decode_req, "req", None)
        if req is None:
            return
        if getattr(req, "finished", lambda: False)():
            return
        if hasattr(req, "time_stats"):
            req.time_stats.set_wait_queue_entry_time()
        self.waiting_queue.append(req)
        entry["target_enqueued"] = True
```

- [ ] **Step 4: Use the helper on target transfer success**

In `_pd_flip_target_pump_transfer`, replace the success block:

```python
                        transferred.add(rid)
                        entry["phase"] = "transferred"
                        decode_req.kv_receiver.clear()
                        self._pd_flip_release_target_request(entry)
                        self._pd_flip_free_target_metadata(entry)
```

with:

```python
                        transferred.add(rid)
                        entry["phase"] = "transferred"
                        decode_req.kv_receiver.clear()
                        self._pd_flip_enqueue_target_request(entry)
                        self._pd_flip_free_target_metadata(entry)
```

- [ ] **Step 5: Run targeted tests**

Run:

```bash
python -m pytest test/srt/test_disaggregation_fake_decode.py::TestFakePDFlipTargetMigration -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/managers/scheduler.py test/srt/test_disaggregation_fake_decode.py
git commit -m "feat: resume migrated pd flip requests on target"
```

### Task 2: Preserve Request State Required For Decode Resume

**Files:**
- Modify: `test/srt/test_pd_flip_internal_state_update.py`
- Modify: `python/sglang/srt/managers/scheduler.py`

- [ ] **Step 1: Add a failing manifest round-trip test**

Add this test to `test/srt/test_pd_flip_internal_state_update.py`:

```python
    def test_pd_flip_manifest_to_req_preserves_decode_resume_fields(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model_config = types.SimpleNamespace(
            vocab_size=32000,
            hf_eos_token_id=[2],
        )
        scheduler.tokenizer = None
        scheduler.server_args = types.SimpleNamespace(disaggregation_bootstrap_port=8998)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.metrics_reporter = types.SimpleNamespace(enable_metrics=False)
        scheduler.metrics_collector = None

        manifest = {
            "rid": "rid-1",
            "origin_input_ids": [10, 11, 12],
            "output_ids": [100, 101],
            "kv_committed_len": 4,
            "migration_bootstrap_room": 1234,
            "source_bootstrap_port": 8999,
            "return_logprob": False,
            "priority": 7,
            "routing_key": "route-a",
            "extra_key": "extra-a",
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 8},
        }

        req = Scheduler._pd_flip_manifest_to_req(
            scheduler, manifest, source_host="127.0.0.1"
        )

        self.assertEqual(req.rid, "rid-1")
        self.assertEqual(list(req.origin_input_ids), [10, 11, 12])
        self.assertEqual(list(req.output_ids), [100, 101])
        self.assertEqual(req.kv_committed_len, 4)
        self.assertEqual(req.bootstrap_host, "127.0.0.1")
        self.assertEqual(req.bootstrap_port, 8999)
        self.assertEqual(req.bootstrap_room, 1234)
        self.assertEqual(req.priority, 7)
        self.assertEqual(req.routing_key, "route-a")
        self.assertEqual(req.extra_key, "extra-a")
```

- [ ] **Step 2: Run the test**

Run:

```bash
python -m pytest test/srt/test_pd_flip_internal_state_update.py::TestPDFlipInternalStateUpdate::test_pd_flip_manifest_to_req_preserves_decode_resume_fields -q
```

Expected: PASS if current manifest reconstruction is sufficient. If it fails, update `_pd_flip_manifest_to_req` only for the missing fields asserted above.

- [ ] **Step 3: Add explicit migrated-request marker**

In `_pd_flip_manifest_to_req`, after `req.kv_committed_len = ...`, add:

```python
        req.pd_flip_migrated = True
        req.pd_flip_source_rid = str(manifest.get("rid", ""))
```

- [ ] **Step 4: Assert the marker**

Extend the test with:

```python
        self.assertTrue(req.pd_flip_migrated)
        self.assertEqual(req.pd_flip_source_rid, "rid-1")
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest test/srt/test_pd_flip_internal_state_update.py::TestPDFlipInternalStateUpdate::test_pd_flip_manifest_to_req_preserves_decode_resume_fields -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/managers/scheduler.py test/srt/test_pd_flip_internal_state_update.py
git commit -m "feat: mark pd flip migrated requests"
```

### Task 3: Expose Target Resume Status

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `test/srt/test_pd_flip_internal_state_update.py`

- [ ] **Step 1: Add status assertions**

Add this test:

```python
    def test_pd_flip_migration_status_reports_resumed_reqs(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.pd_flip_migration_session = {
            "role": "target",
            "state": "target_transferred",
            "session_id": "session-1",
            "pending_reqs": 0,
            "transferred_reqs": 1,
            "released_reqs": 0,
            "failed_reqs": 0,
            "last_error": "",
            "dry_run": False,
            "resumed_rids": {"rid-1"},
        }

        status = Scheduler._pd_flip_migration_status_dict(scheduler)

        self.assertEqual(status["resumed_reqs"], 1)
```

- [ ] **Step 2: Run and confirm failure**

```bash
python -m pytest test/srt/test_pd_flip_internal_state_update.py::TestPDFlipInternalStateUpdate::test_pd_flip_migration_status_reports_resumed_reqs -q
```

Expected: FAIL because `resumed_reqs` is not returned yet.

- [ ] **Step 3: Update `_pd_flip_migration_status_dict`**

In the empty-session return dict, add:

```python
                "resumed_reqs": 0,
```

In the non-empty return dict, add:

```python
            "resumed_reqs": len(session.get("resumed_rids", set())),
```

- [ ] **Step 4: Track resumed rids in `_pd_flip_target_pump_transfer`**

Inside `_pd_flip_target_pump_transfer`, initialize:

```python
        resumed = set(session.get("resumed_rids", set()))
```

When success enqueues the target request, add:

```python
                        resumed.add(rid)
```

Before returning, store:

```python
        session["resumed_rids"] = resumed
```

- [ ] **Step 5: Expose through `get_pd_flip_internal_state`**

After `status["migration_transferred_reqs"] = ...`, add:

```python
        status["migration_resumed_reqs"] = migration_status["resumed_reqs"]
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest test/srt/test_pd_flip_internal_state_update.py test/srt/test_disaggregation_fake_decode.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/managers/scheduler.py test/srt/test_pd_flip_internal_state_update.py
git commit -m "feat: report pd flip target resume status"
```

---

## Milestone 2: Experiment Controller For Drain/Migrate/Restart/Commit

This milestone answers: can one script drive a complete D->P experiment safely and repeatedly?

### Task 4: Wait For Target Resume Before Source Finish

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_experiment.py`
- Modify: `test/srt/test_pd_flip_experiment_script.py`

- [ ] **Step 1: Add failing controller test**

Add a test to `test/srt/test_pd_flip_experiment_script.py`:

```python
    def test_migration_waits_for_target_resumed_reqs_before_source_finish(self):
        source_url = "http://127.0.0.1:30000"
        target_url = "http://127.0.0.1:30001"
        client = FakeClient([])
        client.migration_statuses = {
            source_url: [
                {"success": True, "status": {"pending_reqs": 0, "failed_reqs": 0}},
            ],
            target_url: [
                {
                    "success": True,
                    "status": {
                        "pending_reqs": 0,
                        "failed_reqs": 0,
                        "resumed_reqs": 1,
                    },
                },
            ],
        }

        result = self.script.wait_migration_completion(
            client=client,
            source_url=source_url,
            target_url=target_url,
            timeout_seconds=1.0,
            poll_interval_seconds=0.0,
            sleep_fn=lambda _: None,
            log_fn=lambda _: None,
        )

        self.assertEqual(result["status"], "migration_transferred")
```

- [ ] **Step 2: Update fake client if needed**

If `FakeClient` does not support per-worker status sequences, add this to its `get_json` handling for `/pd_flip/migration/status`:

```python
        if path == "/pd_flip/migration/status":
            statuses = getattr(self, "migration_statuses", {})
            values = statuses.get(base_url)
            if values:
                return values.pop(0)
            return {
                "success": True,
                "status": {
                    "pending_reqs": 0,
                    "failed_reqs": 0,
                    "resumed_reqs": 0,
                },
            }
```

- [ ] **Step 3: Update completion predicate**

In `scripts/playground/disaggregation/pd_flip_experiment.py`, change `_migration_status_done` to:

```python
def _migration_status_done(status: JsonDict, require_resumed: bool = False) -> bool:
    inner = status.get("status") or {}
    if not bool(status.get("success")):
        return False
    if int(inner.get("pending_reqs") or 0) != 0:
        return False
    if int(inner.get("failed_reqs") or 0) != 0:
        return False
    if require_resumed and int(inner.get("resumed_reqs") or 0) <= 0:
        return False
    return True
```

Then in `wait_migration_completion`, change the done check to:

```python
        if _migration_status_done(last_source) and _migration_status_done(
            last_target, require_resumed=True
        ):
            return {
                "status": "migration_transferred",
                "source": last_source,
                "target": last_target,
            }
```

- [ ] **Step 4: Run controller tests**

```bash
python -m pytest test/srt/test_pd_flip_experiment_script.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/playground/disaggregation/pd_flip_experiment.py test/srt/test_pd_flip_experiment_script.py
git commit -m "feat: wait for pd flip target resume"
```

### Task 5: Add Idempotent Controller State Checks

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_experiment.py`
- Modify: `test/srt/test_pd_flip_experiment_script.py`

- [ ] **Step 1: Add helper that validates a worker is safe to flip**

Add to the script:

```python
def validate_flip_candidate(pd_flip: JsonDict, direction: str) -> Optional[str]:
    if not pd_flip.get("enabled"):
        return "pd_flip state machine is not enabled"
    if pd_flip.get("state") not in ("safe", "preparing", "flipping"):
        return f"unexpected pd_flip state: {pd_flip.get('state')}"
    if direction == "d_to_p" and pd_flip.get("current_role") != "decode":
        return f"D->P requires current_role=decode, got {pd_flip.get('current_role')}"
    if direction == "p_to_d" and pd_flip.get("current_role") != "prefill":
        return f"P->D requires current_role=prefill, got {pd_flip.get('current_role')}"
    return None
```

- [ ] **Step 2: Add tests**

```python
    def test_validate_flip_candidate_rejects_wrong_role(self):
        reason = self.script.validate_flip_candidate(
            {"enabled": True, "state": "safe", "current_role": "prefill"},
            "d_to_p",
        )
        self.assertIn("current_role=decode", reason)

    def test_validate_flip_candidate_accepts_decode_to_prefill(self):
        reason = self.script.validate_flip_candidate(
            {"enabled": True, "state": "safe", "current_role": "decode"},
            "d_to_p",
        )
        self.assertIsNone(reason)
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest test/srt/test_pd_flip_experiment_script.py::TestPDFlipExperimentScript::test_validate_flip_candidate_rejects_wrong_role test/srt/test_pd_flip_experiment_script.py::TestPDFlipExperimentScript::test_validate_flip_candidate_accepts_decode_to_prefill -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/playground/disaggregation/pd_flip_experiment.py test/srt/test_pd_flip_experiment_script.py
git commit -m "feat: validate pd flip experiment candidates"
```

---

## Milestone 3: Minimal Real-Traffic Experiment Without Stream Handoff

This milestone answers: can we run real background traffic while flipping one worker, accepting that in-flight migrated client streams are not yet transparent?

### Task 6: Add Explicit Experiment Mode Names

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_experiment.py`

- [ ] **Step 1: Add CLI argument**

In `build_parser()`, add to `run_parser`:

```python
    run_parser.add_argument(
        "--continuity-mode",
        choices=("drain_only", "server_resume", "stream_handoff"),
        default="server_resume",
        help=(
            "drain_only waits for source to empty naturally; "
            "server_resume migrates KV and resumes on target without client stream handoff; "
            "stream_handoff is reserved for router-integrated continuation."
        ),
    )
```

- [ ] **Step 2: Thread the argument into `run_once`**

Change `run_once` signature:

```python
    continuity_mode: str = "server_resume",
```

Pass it from `main()`:

```python
            continuity_mode=args.continuity_mode,
```

- [ ] **Step 3: Guard unsupported stream handoff**

At the start of `run_once`, add:

```python
    if continuity_mode == "stream_handoff":
        return {
            "status": "unsupported",
            "reason": "stream_handoff requires router integration and is not implemented yet",
        }
```

- [ ] **Step 4: Run script unit tests**

```bash
python -m pytest test/srt/test_pd_flip_experiment_script.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/playground/disaggregation/pd_flip_experiment.py
git commit -m "feat: name pd flip experiment continuity modes"
```

### Task 7: Manual Experiment Runbook

**Files:**
- Create: `docs/superpowers/specs/2026-06-21-pd-flip-real-traffic-runbook.md`

- [ ] **Step 1: Create the runbook**

Create the file with:

```markdown
# PD Flip Real Traffic Experiment Runbook

## Scope

This runbook validates `server_resume` mode. It proves active decode request state and KV can move to another decode worker and resume server-side. It does not claim transparent client stream handoff.

## Prerequisites

- One prefill worker.
- At least two decode workers.
- All workers launched with `--enable-pd-flip-state-machine`.
- Decode workers use the same model, tokenizer, TP/DP-compatible layout, and disaggregation transfer backend.
- Router/gateway is configured to skip workers with `pd_flip_state=preparing` or `flipping`.

## Commands

Observe:

```bash
python scripts/playground/disaggregation/pd_flip_experiment.py observe \
  --worker-url http://127.0.0.1:30000 \
  --worker-url http://127.0.0.1:30001 \
  --watch
```

Trigger D->P on the source decode:

```bash
python scripts/playground/disaggregation/pd_flip_experiment.py trigger \
  --worker-url http://127.0.0.1:30000 \
  --direction d_to_p \
  --prefill-nodes 1 \
  --decode-nodes 2 \
  --threshold 0.9 \
  --window-seconds 0.0
```

Drive one flip:

```bash
python scripts/playground/disaggregation/pd_flip_experiment.py run-once \
  --worker-url http://127.0.0.1:30000 \
  --migration-target-url http://127.0.0.1:30001 \
  --continuity-mode server_resume \
  --restart-command 'docker compose restart decode0'
```

## Success Criteria

- Source enters `preparing`.
- Router stops assigning new work to source.
- Source migration status reaches `source_released`.
- Target migration status reaches `target_transferred`.
- Target reports `migration_resumed_reqs > 0`.
- Source reaches `flipping`.
- Restart command brings the worker back as prefill.
- `/server_info` shows the replacement worker as prefill.
- Controller returns `completed` or `completed_by_restart`.

## Failure Criteria

- Any worker reports `migration_failed_reqs > 0`.
- Source remains in `preparing` after timeout.
- Target never reports `migration_resumed_reqs > 0`.
- Restart succeeds but router still classifies the worker as decode.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-21-pd-flip-real-traffic-runbook.md
git commit -m "docs: add pd flip real traffic runbook"
```

---

## Milestone 4: Router Stream Handoff Gate

This milestone is required before claiming transparent real-client request migration.

### Task 8: Add Router-Level Design Spec Before Coding

**Files:**
- Create: `docs/superpowers/specs/2026-06-21-pd-flip-router-stream-handoff-design.md`

- [ ] **Step 1: Write the design skeleton**

Create:

```markdown
# PD Flip Router Stream Handoff Design

## Problem

Server-side KV migration can move request compute state, but the original client stream still belongs to the router/source path. To claim transparent request migration, the router must either hand the stream to the target worker or replay/resume the request and proxy target output to the same client connection.

## Required Behavior

- Router detects source worker `pd_flip_state=preparing`.
- Router stops new routing to source.
- For each migratable request, router associates `rid` with target worker.
- Router keeps the original client response stream open.
- After target resumes, router proxies target output to the original client.
- If migration fails before source release, router keeps source stream.
- If migration fails after source release, router sends a terminal error event.

## First Implementation Choice

Use explicit continuation polling or an internal stream endpoint from target. Do not attempt to move raw HTTP sockets between workers.

## Open Interfaces

- Source exposes migrated `rid` list.
- Target exposes a continuation stream by `rid`.
- Router maps original request stream to target continuation stream.

## Out Of Scope

- Multi-target fanout.
- Cross-router handoff.
- Transparent migration for multimodal, LoRA, constrained decoding, and speculative decoding until server-side state coverage is verified.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-21-pd-flip-router-stream-handoff-design.md
git commit -m "docs: design pd flip router stream handoff"
```

---

## Verification Checklist

- [ ] Run server-side unit tests:

```bash
python -m pytest test/srt/test_disaggregation_fake_decode.py test/srt/test_pd_flip_internal_state_update.py -q
```

- [ ] Run controller tests:

```bash
python -m pytest test/srt/test_pd_flip_experiment_script.py -q
```

- [ ] Run all PD flip tests:

```bash
python -m pytest test/srt/test_pd_flip_state_machine.py test/srt/test_pd_flip_internal_state_update.py test/srt/test_pd_flip_experiment_script.py test/srt/test_disaggregation_fake_decode.py -q
```

- [ ] Run manual `server_resume` experiment from the runbook.
- [ ] Record source and target `/server_info` before, during, and after flip.
- [ ] Confirm `migration_resumed_reqs > 0` on target before source finish.
- [ ] Confirm no claim of transparent client stream handoff until Milestone 4 is implemented.

---

## Done Definition

This plan is complete when:

- A migrated active decode request can be enqueued and resumed on the target decode worker.
- The experiment controller waits for target resume before releasing source requests.
- D->P identity switch can be driven through prepare, restart, and commit while real background traffic is running.
- Documentation clearly labels the mode as `server_resume`, not transparent stream handoff.
- Router stream handoff has a separate accepted design before implementation begins.
