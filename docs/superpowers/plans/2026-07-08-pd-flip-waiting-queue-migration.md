# PD Flip Waiting Queue Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate eligible decode `waiting_queue` requests during PD flip KV migration and produce a full-link latency experiment diagram with waiting-queue stages.

**Architecture:** Source migration will scan running and waiting requests, build manifests for eligible waiting requests, freeze source waiting requests after entry initialization, restore them on abort, and release them differently from running requests on finish. Target migration keeps using the existing `transferred_held` and adoption path.

**Tech Stack:** Python scheduler code in `python/sglang/srt/managers/scheduler.py`, unit tests under `test/srt`, existing PD flip experiment scripts under `scripts/playground/disaggregation` and `experiments`.

## Global Constraints

- Only migrate waiting requests with `req_pool_idx`, positive `kv_committed_len`, and non-empty `output_ids`.
- Do not migrate no-KV, prealloc, transfer, offload, or grammar queue requests in this patch.
- Do not remove waiting requests from source `waiting_queue` until source entries are initialized successfully.
- Restore frozen waiting requests on source abort.
- Do not use `FINISH_MIGRATED()` for source-side waiting entries on finish.
- Git index is currently corrupt on this checkout, so implementation verification must not depend on `git status` or commits until the index is repaired.

---

### Task 1: Scheduler Unit Tests

**Files:**
- Modify: `test/srt/test_pd_flip_migration_accounting.py`

**Interfaces:**
- Consumes: `Scheduler.__new__(Scheduler)` and private PD flip helper methods.
- Produces: regression tests for waiting request classification, freeze, abort restore, and release.

- [ ] **Step 1: Add fake request helper**

```python
def _waiting_req(rid="waiting-1", req_pool_idx=7, output_ids=None, kv_committed_len=4):
    req = types.SimpleNamespace(
        rid=rid,
        req_pool_idx=req_pool_idx,
        output_ids=[1] if output_ids is None else output_ids,
        origin_input_ids=[11, 12, 13],
        kv_committed_len=kv_committed_len,
        pd_flip_defer_kv_release=False,
        pd_flip_force_kv_release=False,
        pd_flip_kv_release_deferred=False,
        pd_flip_deferred_kv_release_is_insert=False,
        finished=lambda: False,
    )
    return req
```

- [ ] **Step 2: Test eligible/ineligible waiting classification**

Run: `python -m pytest test/srt/test_pd_flip_migration_accounting.py -k waiting_queue -q`

Expected before implementation: tests fail because helper methods or fields are missing.

- [ ] **Step 3: Test freeze/abort/release behavior**

Expected before implementation: tests fail because waiting entries are treated like running entries.

### Task 2: Source Waiting Queue Classification

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`

**Interfaces:**
- Produces: `_pd_flip_classify_waiting_reqs(waiting_reqs)` returning `(eligible, skipped)`.
- Produces manifest fields: `pd_flip_source_queue`, `pd_flip_waiting_queue_index`.

- [ ] **Step 1: Implement `_pd_flip_waiting_req_skip_reason`**

The method returns:

```python
"missing_req_pool_idx"
"missing_committed_kv"
"missing_output_token"
"finished"
""
```

- [ ] **Step 2: Include eligible waiting manifests in `start_pd_flip_migration_source`**

Running manifests keep `pd_flip_source_queue="running"`. Waiting manifests use
`pd_flip_source_queue="waiting"` and preserve the original queue index.

- [ ] **Step 3: Expose timing/debug fields**

Add `scan_waiting_reqs_s`, `waiting_reqs`, `waiting_manifest_count`,
`waiting_skipped_count`, and `waiting_skipped`.

### Task 3: Source Freeze, Abort Restore, and Finish Release

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`

**Interfaces:**
- Produces: `_pd_flip_freeze_waiting_source_requests(session)`.
- Produces: `_pd_flip_restore_waiting_source_requests(session)`.
- Updates: `_pd_flip_abort_source_session`, `_pd_flip_release_source_requests`.

- [ ] **Step 1: Freeze waiting requests only after source entries succeed**

Remove eligible waiting objects from `self.waiting_queue` once entries are ready.
Store frozen request references and original indices in `session`.

- [ ] **Step 2: Restore waiting requests on abort**

Abort must call `_pd_flip_restore_waiting_source_requests` after aborting
senders and clearing defer flags.

- [ ] **Step 3: Release waiting requests on finish without `FINISH_MIGRATED()`**

For entries where `manifest["pd_flip_source_queue"] == "waiting"`, clear defer
flags, release deferred KV if needed, free metadata, and mark the entry released.
Do not set `req.to_finish`.

### Task 4: Observability and Diagram Data

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `scripts/playground/disaggregation/pd_flip_migration_measure.py`
- Modify: `experiments/make_pd_state_machine_latency_diagram.py`

**Interfaces:**
- Consumes: `status["timing_debug"]`, `status["index_debug"]`.
- Produces: CSV rows and SVG labels for waiting scan, waiting manifest, waiting freeze, target held, target adopt, and source waiting release.

- [ ] **Step 1: Add source_queue to migration timing entries**

Each entry in `_pd_flip_migration_timing_debug` includes `source_queue`.

- [ ] **Step 2: Preserve new timing fields in measurement CSV**

`migration_status_samples.csv` keeps the JSON timing payload with waiting fields.

- [ ] **Step 3: Update diagram builder labels**

The generated diagram shows waiting-queue stages beside the existing running
request stages.

### Task 5: Verification and Remote Experiment

**Files:**
- No code files beyond previous tasks.
- Outputs under `experiments/pd_flip_waiting_queue_<timestamp>/`.

**Interfaces:**
- Consumes: modified scheduler and experiment scripts.
- Produces: raw tarball, CSV timing data, request trace, TTFT/TPOT/SLO data, and latency diagram.

- [ ] **Step 1: Run local unit tests**

Run:

```powershell
python -m pytest test/srt/test_pd_flip_migration_accounting.py -q
python -m pytest test/srt/test_pd_flip_migration_measure.py test/srt/test_pd_flip_trace_replay.py -q
```

- [ ] **Step 2: Sync changed files to cloud099-cloud102**

Use targeted copy of modified files only.

- [ ] **Step 3: Run a full-link observation workload**

Use the existing four nodes. Force a D->P migration while decode has running
requests and, if needed, constrain decode concurrency to create eligible waiting
requests.

- [ ] **Step 4: Build raw package**

Package the request trace, TTFT raw, TPOT raw, SLO summary, migration raw events,
status samples, stage durations, and the labeled full-link latency diagram.
