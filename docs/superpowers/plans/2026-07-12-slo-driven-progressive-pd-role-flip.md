# SLO-Driven Progressive PD Role Flip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a controller-owned D-to-P state machine that first migrates a capacity-safe fraction of a decode worker's running requests, observes fresh TTFT/TPOT SLO samples, and either restores decode admission or atomically migrates the remainder and hot-switches the worker to prefill.

**Architecture:** Keep cluster policy in the Python controller and expose narrow, idempotent worker primitives for capacity reporting, selected-request migration, Mooncake/HiCache stitching, ownership transfer, and runtime-role switching. Reconstruct each migrated request from a continuous Mooncake prefix plus the source-decode suffix, with source-full fallback on a miss. Make both migration batches atomic and redispatch the scheduler event loop only after the source is fully idle.

**Tech Stack:** Python 3 dataclasses and unittest/pytest, SGLang scheduler and PD-disaggregation queues, Mooncake Transfer Engine, Mooncake Store/HiCache, FastAPI control endpoints, Rust SGLang router control endpoints.

## Global Constraints

- The implementation target is the current working-tree PD flip code; preserve all unrelated modified and untracked files.
- The controller is the only cluster state-machine owner. Worker-local `FlipStateMachine` remains observability/safety only.
- `pd_flip_slo_threshold=0.9`, `pd_flip_first_migration_ratio=0.5`, `pd_flip_observation_seconds=10`, and both minimum SLO sample counts default to `20`.
- First-batch selection is always the first `N = max(1, ceil(running_count * effective_ratio))` running requests.
- If target capacity is insufficient, repeatedly divide the ratio by two. If one request cannot fit, do not migrate.
- Capacity preflight reserves the zero-Mooncake-hit worst case plus `num_reserved_decode_tokens` per selected request.
- The source accepts no new requests from first migration start until recovery or final role commit.
- Both first and second migration batches are all-or-nothing.
- Mooncake supplies `[0,H)` and source decode supplies `[H,C0)` plus delta `[C0,C1)`. `H` is continuous, page-aligned, and capped at the prompt boundary.
- A Mooncake miss is not an error; it sets `H=0` and transfers the full range from source decode.
- No target request may execute or emit output before source ownership is relinquished.
- Runtime role, active event-loop role, and router role must agree before worker admission resumes.
- Every task follows red-green-refactor and commits only its listed files.

---

## File Structure

- Create `scripts/playground/disaggregation/pd_flip_progressive_policy.py`: pure SLO decision and ratio/capacity selection logic with no SGLang imports.
- Modify `scripts/playground/disaggregation/pd_flip_monitor.py`: fresh observation-window reset and count-bearing snapshots.
- Modify `scripts/playground/disaggregation/pd_flip_controller.py`: progressive controller FSM and HTTP orchestration.
- Modify `python/sglang/srt/managers/io_struct.py`: selected-rid, activate, and capacity request/response contracts.
- Modify `python/sglang/srt/entrypoints/http_server.py`: activate endpoint.
- Modify `python/sglang/srt/managers/tokenizer_control_mixin.py`: broadcast activate and selected migration controls.
- Modify `python/sglang/srt/managers/scheduler.py`: capacity snapshots, selected batches, stitching, freeze/delta, target transactions, and active-loop status.
- Modify `python/sglang/srt/disaggregation/prefill.py` and `python/sglang/srt/disaggregation/decode.py`: safe loop exit on runtime-role change.
- Modify `python/sglang/srt/server_args.py`: explicit stitch feature flag.
- Modify `scripts/playground/disaggregation/pd_flip_docker/run_worker.sh`, `run_controller.sh`, and `env.example`: deployment knobs.
- Create focused tests under `test/srt/` rather than adding more source-string assertions to legacy tests.

---

### Task 1: Pure Progressive Flip Policy

**Files:**
- Create: `scripts/playground/disaggregation/pd_flip_progressive_policy.py`
- Create: `test/srt/test_pd_flip_progressive_policy.py`

**Interfaces:**
- Consumes: role-level good/total SLO counts and ordered per-request committed-token estimates.
- Produces: `ProgressiveDecision`, `RequestCapacity`, `RatioSelection`, `evaluate_slo_decision(...)`, and `select_first_batch(...)`.

- [ ] **Step 1: Write failing SLO-decision tests**

```python
def test_slo_decision_requires_prefill_risk_and_decode_headroom():
    m = load_policy_module()
    assert m.evaluate_slo_decision(14, 20, 19, 20, 0.9, 20, 20) is m.ProgressiveDecision.START
    assert m.evaluate_slo_decision(18, 20, 19, 20, 0.9, 20, 20) is m.ProgressiveDecision.RECOVER
    assert m.evaluate_slo_decision(14, 20, 17, 20, 0.9, 20, 20) is m.ProgressiveDecision.RECOVER
    assert m.evaluate_slo_decision(7, 10, 19, 20, 0.9, 20, 20) is m.ProgressiveDecision.INSUFFICIENT_SAMPLES
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `python -m pytest test/srt/test_pd_flip_progressive_policy.py -k slo_decision -q`

Expected: FAIL because `pd_flip_progressive_policy.py` does not exist.

- [ ] **Step 3: Implement the decision enum and evaluator**

```python
from dataclasses import dataclass
from enum import Enum
from math import ceil
from typing import Optional, Sequence

class ProgressiveDecision(Enum):
    START = "start"
    COMMIT = "commit"
    RECOVER = "recover"
    INSUFFICIENT_SAMPLES = "insufficient_samples"

def evaluate_slo_decision(
    prefill_good: int,
    prefill_total: int,
    decode_good: int,
    decode_total: int,
    threshold: float,
    min_prefill_samples: int,
    min_decode_samples: int,
    *,
    observing: bool = False,
) -> ProgressiveDecision:
    if prefill_total < min_prefill_samples or decode_total < min_decode_samples:
        return ProgressiveDecision.INSUFFICIENT_SAMPLES
    p = prefill_good / prefill_total
    d = decode_good / decode_total
    if p < threshold and d >= threshold:
        return ProgressiveDecision.COMMIT if observing else ProgressiveDecision.START
    return ProgressiveDecision.RECOVER
```

- [ ] **Step 4: Write failing repeated-halving tests**

```python
def test_ratio_halves_until_first_n_requests_fit():
    m = load_policy_module()
    reqs = [m.RequestCapacity(str(i), 100) for i in range(8)]
    out = m.select_first_batch(reqs, 0.75, target_req_slots=3, target_kv_tokens=450, reserve_tokens_per_req=25)
    assert out.configured_ratio == 0.75
    assert out.effective_ratio == 0.375
    assert out.selected_rids == ("0", "1", "2")
    assert out.fallback_count == 1

def test_ratio_returns_none_when_one_request_cannot_fit():
    m = load_policy_module()
    reqs = [m.RequestCapacity("r0", 100)]
    assert m.select_first_batch(reqs, 0.5, 1, 99, 0) is None
```

- [ ] **Step 5: Implement deterministic first-N selection**

```python
@dataclass(frozen=True)
class RequestCapacity:
    rid: str
    committed_tokens: int

@dataclass(frozen=True)
class RatioSelection:
    configured_ratio: float
    effective_ratio: float
    selected_rids: tuple[str, ...]
    required_kv_tokens: int
    fallback_count: int

def select_first_batch(
    requests: Sequence[RequestCapacity],
    configured_ratio: float,
    target_req_slots: int,
    target_kv_tokens: int,
    reserve_tokens_per_req: int,
) -> Optional[RatioSelection]:
    if not 0 < configured_ratio < 1:
        raise ValueError("configured_ratio must be between 0 and 1")
    ratio = configured_ratio
    fallback_count = 0
    while requests:
        count = max(1, ceil(len(requests) * ratio))
        selected = tuple(requests[:count])
        required = sum(r.committed_tokens + reserve_tokens_per_req for r in selected)
        if count <= target_req_slots and required <= target_kv_tokens:
            return RatioSelection(configured_ratio, ratio, tuple(r.rid for r in selected), required, fallback_count)
        if count == 1:
            return None
        ratio /= 2
        fallback_count += 1
    return None
```

- [ ] **Step 6: Run policy tests**

Run: `python -m pytest test/srt/test_pd_flip_progressive_policy.py -q`

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/playground/disaggregation/pd_flip_progressive_policy.py test/srt/test_pd_flip_progressive_policy.py
git commit -m "feat(pd-flip): add progressive SLO and ratio policy"
```

---

### Task 2: Fresh Observation-Window SLO Counts

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_monitor.py:27-119,148-194`
- Create: `test/srt/test_pd_flip_observation_window.py`

**Interfaces:**
- Consumes: existing `NodeSLOSample`, `SampleCounts`, and cumulative Prometheus histograms.
- Produces: `ClusterSLOSnapshot.prefill_counts`, `decode_counts`, and `PDFlipSLOMonitor.reset_window()`.

- [ ] **Step 1: Write a failing reset-window test**

```python
def test_reset_window_excludes_triggering_samples():
    m = load_monitor_module()
    monitor = m.PDFlipSLOMonitor.__new__(m.PDFlipSLOMonitor)
    monitor.window = m.SLOWindow(10.0)
    monitor.window.add(m.NodeSLOSample(1.0, "p0", "prefill", ttft=m.SampleCounts(5, 10)))
    monitor.reset_window()
    snapshot = monitor.window.snapshot(timestamp=2.0)
    assert snapshot.prefill_counts == m.SampleCounts()
    assert snapshot.decode_counts == m.SampleCounts()
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest test/srt/test_pd_flip_observation_window.py -q`

Expected: FAIL because the snapshot count fields and `reset_window` do not exist.

- [ ] **Step 3: Add count-bearing snapshots and reset**

```python
@dataclass(frozen=True)
class ClusterSLOSnapshot:
    timestamp: float
    prefill_nodes: int
    decode_nodes: int
    prefill_slo_attainment: Optional[float]
    decode_slo_attainment: Optional[float]
    nodes: List[NodeSLOSample]
    prefill_counts: SampleCounts
    decode_counts: SampleCounts

def reset_window(self) -> None:
    self.window = SLOWindow(self.window.window_seconds)
```

Update `SLOWindow.snapshot()` to populate `_sum_counts(prefill.ttft)` and `_sum_counts(decode.tpot)` rather than averaging node percentages.

- [ ] **Step 4: Run monitor regression tests**

Run: `python -m pytest test/srt/test_pd_flip_observation_window.py test/srt/test_pd_flip_monitor.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/playground/disaggregation/pd_flip_monitor.py test/srt/test_pd_flip_observation_window.py
git commit -m "feat(pd-flip): reset observation SLO windows"
```

---

### Task 3: Worker Capacity and Ordered Running-Request Status

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py:1267-1301,3745-3771`
- Create: `test/srt/test_pd_flip_capacity_status.py`

**Interfaces:**
- Consumes: scheduler running batch, request-to-token pool, and token allocator.
- Produces: `_pd_flip_capacity_status() -> Dict[str, Any]` and ordered `running_requests` entries with `rid` and `kv_committed_len`.

- [ ] **Step 1: Write failing capacity-status tests**

```python
def test_capacity_status_reports_worst_case_inputs_in_running_order():
    scheduler = make_scheduler(running=[make_req("r0", 100), make_req("r1", 250)])
    scheduler.max_running_requests = 8
    scheduler.req_to_token_pool.available_size = lambda: 5
    scheduler.token_to_kv_pool_allocator.available_size = lambda: 1000
    out = Scheduler._pd_flip_capacity_status(scheduler)
    assert out["free_request_slots"] == 3
    assert out["available_kv_tokens"] == 1000
    assert out["running_requests"] == [
        {"rid": "r0", "kv_committed_len": 100},
        {"rid": "r1", "kv_committed_len": 250},
    ]
```

- [ ] **Step 2: Run and verify missing helper**

Run: `python -m pytest test/srt/test_pd_flip_capacity_status.py -q`

Expected: FAIL with `AttributeError: _pd_flip_capacity_status`.

- [ ] **Step 3: Implement capacity status**

```python
def _pd_flip_capacity_status(self) -> Dict[str, Any]:
    running = list(getattr(getattr(self, "running_batch", None), "reqs", []))
    free_slots = min(
        max(0, self.max_running_requests - len(running)),
        int(self.req_to_token_pool.available_size()),
    )
    return {
        "free_request_slots": free_slots,
        "available_kv_tokens": int(self.token_to_kv_pool_allocator.available_size()),
        "max_running_requests_per_dp": int(self.max_running_requests),
        "reserved_decode_tokens_per_req": int(self.server_args.num_reserved_decode_tokens),
        "running_requests": [
            {"rid": str(req.rid), "kv_committed_len": int(req.kv_committed_len)}
            for req in running
            if not req.finished()
        ],
    }
```

Merge these fields into `_pd_runtime_role_status_dict()`.

- [ ] **Step 4: Run status tests**

Run: `python -m pytest test/srt/test_pd_flip_capacity_status.py test/srt/test_pd_runtime_role_switch.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/managers/scheduler.py test/srt/test_pd_flip_capacity_status.py
git commit -m "feat(pd-flip): expose migration capacity status"
```

---

### Task 4: Exact Selected-RID Source Migration

**Files:**
- Modify: `python/sglang/srt/managers/io_struct.py:1738-1743`
- Modify: `python/sglang/srt/managers/scheduler.py:1680-1740,2400-2470`
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py:48-56,277-370`
- Create: `test/srt/test_pd_flip_selected_batch.py`

**Interfaces:**
- Consumes: `RatioSelection.selected_rids` from Task 1.
- Produces: `PDFlipMigrationSourceStartReq.rids`, `include_waiting`, `_pd_flip_select_source_batch(...)`, and exact-rid manifests.

- [ ] **Step 1: Write failing request-contract and selection tests**

```python
def test_source_start_selects_exact_running_prefix():
    req = PDFlipMigrationSourceStartReq(rids=["r0", "r1"], include_waiting=False)
    scheduler = make_scheduler(running=[make_req("r0"), make_req("r1"), make_req("r2")])
    selected = Scheduler._pd_flip_select_source_batch(scheduler, req)
    assert [r.rid for r in selected] == ["r0", "r1"]

def test_source_start_rejects_non_prefix_selection():
    req = PDFlipMigrationSourceStartReq(rids=["r1"], include_waiting=False)
    scheduler = make_scheduler(running=[make_req("r0"), make_req("r1")])
    with pytest.raises(ValueError, match="running-batch prefix"):
        Scheduler._pd_flip_select_source_batch(scheduler, req)
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest test/srt/test_pd_flip_selected_batch.py -q`

Expected: FAIL because the request fields/helper do not exist.

- [ ] **Step 3: Extend the request and implement selection**

```python
@dataclass
class PDFlipMigrationSourceStartReq(BaseReq):
    session_id: Optional[str] = None
    target_url: Optional[str] = None
    rids: Optional[List[str]] = None
    include_waiting: bool = False

def _pd_flip_select_source_batch(self, recv_req):
    running = [r for r in self.running_batch.reqs if not r.finished()]
    if recv_req.rids is None:
        selected = running
    else:
        count = len(recv_req.rids)
        selected = running[:count]
        if [str(r.rid) for r in selected] != [str(r) for r in recv_req.rids]:
            raise ValueError("selected rids must be a running-batch prefix")
    if recv_req.include_waiting:
        waiting, skipped = self._pd_flip_classify_waiting_reqs(self.waiting_queue)
        if [x for x in skipped if x.get("reason") != "finished"]:
            raise ValueError("remaining waiting requests are not migratable")
        selected.extend(waiting)
    return selected
```

Replace the current `max_reqs` all-or-nothing rejection with this exact selection.

- [ ] **Step 4: Send exact rids from the controller**

```python
def _migration_source_start_payload(session_id, target_url, rids, include_waiting=False):
    return {
        "session_id": session_id,
        "target_url": target_url,
        "rids": list(rids),
        "include_waiting": include_waiting,
    }
```

- [ ] **Step 5: Run migration accounting tests**

Run: `python -m pytest test/srt/test_pd_flip_selected_batch.py test/srt/test_pd_flip_migration_accounting.py -q`

Expected: all tests PASS after updating obsolete tests that expected `refusing partial migration` to instead assert exact-prefix validation.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/managers/io_struct.py python/sglang/srt/managers/scheduler.py scripts/playground/disaggregation/pd_flip_controller.py test/srt/test_pd_flip_selected_batch.py test/srt/test_pd_flip_migration_accounting.py
git commit -m "feat(pd-flip): migrate an exact running-request prefix"
```

---

### Task 5: Capacity-Aware First-Batch Planning in Controller

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py:115-170,277-370,1747-1778,2259-2314`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/env.example`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_controller.sh`
- Create: `test/srt/test_pd_flip_progressive_controller.py`

**Interfaces:**
- Consumes: Task 1 `select_first_batch(...)` and Task 3 runtime status fields.
- Produces: `PDClusterConfig.first_migration_ratio`, sample/observation settings, and `_select_progressive_first_batch(...)`.

- [ ] **Step 1: Write failing controller planning tests**

```python
def test_controller_uses_status_capacity_and_halves_ratio():
    controller = make_controller(first_migration_ratio=0.75)
    source = metric("d0", running_requests=[("r0", 100), ("r1", 100), ("r2", 100), ("r3", 100)])
    target = metric("d1", free_request_slots=1, available_kv_tokens=150)
    selection = controller._select_progressive_first_batch(source, target)
    assert selection.selected_rids == ("r0",)
    assert selection.effective_ratio == 0.1875
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest test/srt/test_pd_flip_progressive_controller.py -k capacity -q`

Expected: FAIL because progressive config and selector do not exist.

- [ ] **Step 3: Add exact controller config**

```python
@dataclass(frozen=True)
class PDClusterConfig:
    router_url: str
    nodes: List[PDNode]
    request_timeout_seconds: float = 10.0
    migration_timeout_seconds: float = 120.0
    migration_poll_interval_seconds: float = 0.5
    observation_quiesce_seconds: float = 0.0
    post_migration_idle_timeout_seconds: float = 2.0
    first_migration_ratio: float = 0.5
    observation_seconds: float = 10.0
    slo_threshold: float = 0.9
    min_prefill_slo_samples: int = 20
    min_decode_slo_samples: int = 20
    session_journal_path: str = "pd_flip_session.json"
```

Extend `from_dict()` and `config_from_args()` for every new field. Import `RequestCapacity`, `ProgressiveDecision`, `evaluate_slo_decision`, and `select_first_batch` with the same script-local fallback pattern used for `pd_flip_monitor.py`. Implement `_select_progressive_first_batch()` by mapping ordered status entries to `RequestCapacity` and calling `select_first_batch()` with target free slots, target available KV, and target `reserved_decode_tokens_per_req`.

- [ ] **Step 4: Wire environment variables**

```text
PD_FLIP_FIRST_MIGRATION_RATIO=0.5
PD_FLIP_OBSERVATION_SECONDS=10
PD_FLIP_SLO_THRESHOLD=0.9
PD_FLIP_MIN_PREFILL_SLO_SAMPLES=20
PD_FLIP_MIN_DECODE_SLO_SAMPLES=20
```

Parse these in `run_controller.sh` and pass matching CLI arguments.

- [ ] **Step 5: Run controller tests**

Run: `python -m pytest test/srt/test_pd_flip_progressive_controller.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/playground/disaggregation/pd_flip_controller.py scripts/playground/disaggregation/pd_flip_docker/env.example scripts/playground/disaggregation/pd_flip_docker/run_controller.sh test/srt/test_pd_flip_progressive_controller.py
git commit -m "feat(pd-flip): plan capacity-safe first migrations"
```

---

### Task 6: Mooncake Prefix Stitch and Source-Full Fallback

**Files:**
- Modify: `python/sglang/srt/server_args.py:867-885,7440-7470`
- Modify: `python/sglang/srt/managers/scheduler.py:2400-2470,2659-2694,2795-2870,3504-3640`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_worker.sh`
- Modify: `test/srt/test_disaggregation_fake_decode.py`
- Create: `test/srt/test_pd_flip_hicache_stitch.py`

**Interfaces:**
- Consumes: target `decode_prefix_len` from existing Mooncake receiver metadata.
- Produces: `enable_pd_flip_hicache_stitch`, `_pd_flip_source_page_indices_range(...)`, and per-request H/P/C0 stitch metadata.

- [ ] **Step 1: Write failing full/partial/miss tests**

```python
@pytest.mark.parametrize(
    "storage_hit,prompt_len,page_size,expected_h,expected_mode",
    [(1024, 1024, 16, 1024, "full_prefix_stitch"),
     (768, 1024, 16, 768, "partial_prefix_stitch"),
     (0, 1024, 16, 0, "source_decode_full_fallback"),
     (1024, 1027, 16, 1024, "full_prefix_stitch")],
)
def test_stitch_boundary(storage_hit, prompt_len, page_size, expected_h, expected_mode):
    h, mode = Scheduler._pd_flip_stitch_boundary(storage_hit, prompt_len, page_size)
    assert (h, mode) == (expected_h, expected_mode)
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest test/srt/test_pd_flip_hicache_stitch.py -q`

Expected: FAIL because `_pd_flip_stitch_boundary` does not exist.

- [ ] **Step 3: Add explicit feature flag and boundary helper**

```python
enable_pd_flip_hicache_stitch: bool = False

@staticmethod
def _pd_flip_stitch_boundary(storage_hit: int, prompt_len: int, page_size: int):
    h = min(storage_hit, (prompt_len // page_size) * page_size)
    if h == 0:
        return 0, "source_decode_full_fallback"
    if h == (prompt_len // page_size) * page_size:
        return h, "full_prefix_stitch"
    return h, "partial_prefix_stitch"
```

Replace the environment-only target gate with this server argument and enable it in `run_worker.sh`.

Validate at server-argument finalization that `--enable-pd-flip-hicache-stitch` requires `--enable-pd-runtime-role-switch`, `--disaggregation-decode-enable-radix-cache`, and a configured HiCache storage backend. Raise `ValueError` with all three required flags when the combination is invalid.

- [ ] **Step 4: Write a failing sender-slicing test**

```python
def test_source_consumes_decode_prefix_before_sender_init():
    sender = FakeSender(prefix_len=32)
    scheduler = make_source_scheduler(page_size=1, kv_indices=list(range(100)))
    entry = make_source_entry(sender=sender, committed_len=100)
    Scheduler._pd_flip_source_send_initial(scheduler, entry)
    assert sender.init_args[0] == 68
    assert sender.sent_indices.tolist() == list(range(32, 100))
    assert entry["mooncake_hit_len"] == 32
```

- [ ] **Step 5: Defer sender init until target metadata is ready**

```python
def _pd_flip_source_send_initial(self, entry):
    sender = entry["sender"]
    req = entry["req"]
    h = int(sender.pop_decode_prefix_len())
    c0 = int(entry["committed_len"])
    page_indices = self._pd_flip_source_page_indices_range(req, h, c0)
    sender.init(len(page_indices), entry["metadata_index"])
    sender.send(page_indices, self._pd_flip_source_state_indices(req, c0, sender.kv_mgr))
    entry.update(
        mooncake_hit_len=h,
        source_transfer_start=h,
        source_transfer_end=c0,
        stitch_mode=("source_decode_full_fallback" if h == 0 else "prefix_stitch"),
    )
```

Call this helper only after `sender.poll() == KVPoll.WaitingForInput`. Remove the earlier full-range `sender.init(...)` from source-entry creation.

- [ ] **Step 6: Gate target completion on HiCache restore and coverage**

Store P/H/C0 in the target entry and reject completion unless `H <= P <= C0`, HiCache restore is ready, and the received suffix begins at H.

- [ ] **Step 7: Run focused and existing fake-backend tests**

Run: `python -m pytest test/srt/test_pd_flip_hicache_stitch.py test/srt/test_disaggregation_fake_decode.py -q`

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add python/sglang/srt/server_args.py python/sglang/srt/managers/scheduler.py scripts/playground/disaggregation/pd_flip_docker/run_worker.sh test/srt/test_pd_flip_hicache_stitch.py test/srt/test_disaggregation_fake_decode.py
git commit -m "feat(pd-flip): stitch Mooncake prefix with decode suffix"
```

---

### Task 7: Atomic Batch Quiesce, Commit, Finish, and Activate

**Files:**
- Modify: `python/sglang/srt/managers/io_struct.py:1779-1807`
- Modify: `python/sglang/srt/entrypoints/http_server.py:725-788`
- Modify: `python/sglang/srt/managers/tokenizer_control_mixin.py:874-938`
- Modify: `python/sglang/srt/managers/tokenizer_manager.py: communicator declarations and output relay state`
- Modify: `python/sglang/srt/managers/scheduler.py:1941-2200,2894-2969,3150-3743`
- Modify: `python/sglang/srt/disaggregation/decode.py:1721-1789`
- Create: `test/srt/test_pd_flip_atomic_batch.py`

**Interfaces:**
- Consumes: prepared/held target entries and source delta support.
- Produces: `PDFlipMigrationTargetActivateReq`, `/pd_flip/migration/target/activate`, a short source batch-quiesce barrier, and target transaction states `PREPARED_HELD`, `READY_TO_ACTIVATE`, `ACTIVE`.

- [ ] **Step 1: Write failing atomic-commit tests**

```python
def test_target_commit_does_not_schedule_before_activate():
    scheduler = make_target(entries=[held("r0"), held("r1")])
    out = Scheduler.commit_pd_flip_migration_target(
        scheduler, PDFlipMigrationTargetCommitReq(session_id="s", rids=["r0", "r1"])
    )
    assert out.success
    assert scheduler.waiting_queue == []
    assert scheduler.pd_flip_migration_session["state"] == "ready_to_activate"

def test_one_failed_entry_aborts_entire_batch():
    scheduler = make_target(entries=[held("r0"), failed("r1")])
    out = Scheduler.commit_pd_flip_migration_target(
        scheduler, PDFlipMigrationTargetCommitReq(session_id="s", rids=["r0", "r1"])
    )
    assert not out.success
    assert all(entry["phase"] == "aborted" for entry in scheduler.pd_flip_migration_session["target_entries"].values())
```

- [ ] **Step 2: Run and verify current eager adoption fails the test**

Run: `python -m pytest test/srt/test_pd_flip_atomic_batch.py -k target -q`

Expected: FAIL because current commit adopts requests immediately.

- [ ] **Step 3: Add activate request and control path**

```python
@dataclass
class PDFlipMigrationTargetActivateReq(BaseReq):
    session_id: Optional[str] = None
    rids: Optional[List[str]] = None
```

Add `POST /pd_flip/migration/target/activate`, broadcast it through `TokenizerControlMixin`, and dispatch it to `Scheduler.activate_pd_flip_migration_target`.

Register `PDFlipMigrationTargetActivateReq` in TokenizerManager's migration communicator and Scheduler's request dispatcher. Add it to `io_struct._check_all_req_types()` coverage.

- [ ] **Step 4: Split target commit from activation**

```python
def commit_pd_flip_migration_target(self, recv_req):
    session = self._pd_flip_require_target_session(recv_req.session_id)
    entries = self._pd_flip_target_batch_entries(session, recv_req.rids)
    if not entries or any(e.get("phase") != "transferred_held" for e in entries):
        self._pd_flip_abort_target_session(session, "batch is not atomically ready")
        return self._pd_flip_migration_output(False, "batch is not atomically ready")
    for entry in entries:
        self._pd_flip_target_commit_hicache_restore(entry["decode_req"])
        entry["phase"] = "ready_to_activate"
    session["state"] = "ready_to_activate"
    return self._pd_flip_migration_output(True, "target batch ready to activate")

def activate_pd_flip_migration_target(self, recv_req):
    session = self._pd_flip_require_target_session(recv_req.session_id)
    entries = self._pd_flip_target_batch_entries(session, recv_req.rids)
    if any(e.get("phase") != "ready_to_activate" for e in entries):
        return self._pd_flip_migration_output(False, "target batch is not committed")
    requests = [entry["decode_req"].req for entry in entries]
    for req in requests:
        req.init_next_round_input(self.tree_cache)
        req.time_stats.set_wait_queue_entry_time()
    self.waiting_queue.extend(requests)
    for entry in entries:
        entry["request_adopted"] = True
        entry["phase"] = "active"
    session["state"] = "active"
    return self._pd_flip_migration_output(True, "target batch activated")
```

- [ ] **Step 5: Add a short whole-source quiesce barrier for delta cutover**

Do not mutate `running_batch.reqs` while an overlap result is pending. `start_pd_flip_migration_source_delta` sets a quiesce request. The decode event loop finishes the current result, stops launching new batches, and reports `pd_flip_batch_quiesced=true`. Only then capture C1 and start delta transfer. Unselected requests pause only for this short delta/ownership interval.

```python
def _pd_flip_request_batch_quiesce(self, rids):
    self.pd_flip_quiesce_rids = tuple(str(r) for r in rids)
    self.pd_flip_quiesce_requested = True

def _pd_flip_maybe_enter_batch_quiesce(self):
    if not self.pd_flip_quiesce_requested:
        return False
    if getattr(self, "result_queue", None):
        return False
    self.pd_flip_batch_quiesced = True
    return True

def _pd_flip_resume_batch_after_cutover(self):
    self.pd_flip_quiesce_requested = False
    self.pd_flip_batch_quiesced = False
    self.pd_flip_quiesce_rids = ()
```

In both decode loops, call `_pd_flip_maybe_enter_batch_quiesce()` after processing the prior result and before `get_next_disagg_decode_batch_to_run()`. While it returns true, continue polling control requests but do not schedule a batch.

On source finish, mark selected requests with the migration finish reason and call `running_batch.filter_batch()` to remove finished entries, then resume unselected requests. On abort before source finish, do not remove any request; just clear the quiesce state so all source requests continue.

- [ ] **Step 6: Add monotonic output-sequence handoff**

Add `last_emitted_output_seq` to each manifest. The source increments it before every relayed output. TokenizerManager stores `pd_flip_last_relay_seq_by_rid`; `relay_pd_flip_migration_output` drops any output with `output_seq <= last_seen` and updates the map before forwarding a newer output.

- [ ] **Step 7: Run atomic transaction tests**

Run: `python -m pytest test/srt/test_pd_flip_atomic_batch.py test/srt/test_pd_flip_active_decode_handoff.py test/srt/test_pd_flip_migration_accounting.py -q`

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add python/sglang/srt/managers/io_struct.py python/sglang/srt/entrypoints/http_server.py python/sglang/srt/managers/tokenizer_control_mixin.py python/sglang/srt/managers/tokenizer_manager.py python/sglang/srt/managers/scheduler.py python/sglang/srt/disaggregation/decode.py test/srt/test_pd_flip_atomic_batch.py test/srt/test_pd_flip_active_decode_handoff.py test/srt/test_pd_flip_migration_accounting.py
git commit -m "feat(pd-flip): make migration ownership transfer atomic"
```

---

### Task 8: Controller Progressive FSM and Observation Branches

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py:252-567,663-965,1125-1221`
- Modify: `test/srt/test_pd_flip_progressive_controller.py`

**Interfaces:**
- Consumes: Tasks 1-7 policy, SLO window, selected-rid, transaction, and activate APIs.
- Produces: `ProgressiveMonitorState`, `_execute_progressive_d_to_p(...)`, and the exact SAFE/recover/commit flow.

- [ ] **Step 1: Write failing recovery and commit traces**

```python
def test_progressive_flow_recovers_after_first_batch():
    controller, client, monitor = scenario(observation=(18, 20, 19, 20))
    result = controller.monitor_progressive(monitor, iterations=1)
    assert result.success
    assert [x["state"] for x in result.state_trace] == [
        "safe", "selecting", "first_migrating", "observing", "recovering", "safe"
    ]
    assert client.steps.count("set_source_runtime_role") == 0

def test_progressive_flow_commits_when_prefill_stays_risky():
    controller, client, monitor = scenario(observation=(14, 20, 19, 20))
    result = controller.monitor_progressive(monitor, iterations=1)
    assert result.success
    assert "second_migrating" in [x["state"] for x in result.state_trace]
    assert client.steps[-4:] == [
        "set_source_runtime_role", "wait_source_prefill_loop", "refresh_router_source_role", "router_undrain_source"
    ]
```

- [ ] **Step 2: Run and verify missing FSM**

Run: `python -m pytest test/srt/test_pd_flip_progressive_controller.py -k progressive_flow -q`

Expected: FAIL because `monitor_progressive` does not exist.

- [ ] **Step 3: Define controller states**

```python
class ProgressiveMonitorState:
    SAFE = "safe"
    SELECTING = "selecting"
    FIRST_MIGRATING = "first_migrating"
    OBSERVING = "observing"
    RECOVERING = "recovering"
    SECOND_MIGRATING = "second_migrating"
    FLIPPING_ROLE = "flipping_role"
```

- [ ] **Step 4: Implement the first migration and fresh observation**

Drain/pause source, send selected rids with `include_waiting=false`, prepare target, wait initial transfer, quiesce at a batch boundary for delta, target commit, source finish, and target activate. Then call `slo_monitor.reset_window()`, poll for exactly `observation_seconds`, and evaluate fresh counts with `observing=True`.

```python
def _execute_atomic_batch(self, source, target, session_id, rids, include_waiting):
    start = self._post_worker(
        [], "source_start", source, "/pd_flip/migration/source/start",
        _migration_source_start_payload(session_id, target.worker_url, rids, include_waiting),
    )
    manifests = _response_manifests(start)
    self._post_worker([], "target_prepare", target, "/pd_flip/migration/target/prepare", {
        "session_id": session_id,
        "source_url": source.worker_url,
        "manifests": manifests,
        "prepare_only": True,
        "adopt_on_commit": False,
    })
    self._wait_migration([], "source_initial", source)
    self._wait_migration([], "target_initial", target)
    delta = self._post_worker([], "source_delta", source, "/pd_flip/migration/source/delta", {
        "session_id": session_id,
        "rids": list(rids),
    })
    self._post_worker([], "target_delta", target, "/pd_flip/migration/target/delta/prepare", {
        "session_id": session_id,
        "source_url": source.worker_url,
        "manifests": _response_manifests(delta),
    })
    self._wait_migration([], "source_delta_wait", source)
    self._wait_migration([], "target_delta_wait", target)
    self._post_worker([], "target_commit", target, "/pd_flip/migration/target/commit", {
        "session_id": session_id, "rids": list(rids),
    })
    self._post_worker([], "source_finish", source, "/pd_flip/migration/source/finish", {
        "session_id": session_id, "released_rids": list(rids),
    })
    self._post_worker([], "target_activate", target, "/pd_flip/migration/target/activate", {
        "session_id": session_id, "rids": list(rids),
    })
```

The production method passes one shared `records` list to every helper and wraps the sequence in `try/except`; before source finish, any exception calls target abort and source abort, then clears source quiesce.

- [ ] **Step 5: Implement both observation branches**

For `RECOVER` or `INSUFFICIENT_SAMPLES`, resume source admission and router service without moving first-batch requests back. For `COMMIT`, run a second atomic migration with all remaining running rids and `include_waiting=true`, assert source idle, then enter role flip.

```python
decision = evaluate_slo_decision(
    snapshot.prefill_counts.good,
    snapshot.prefill_counts.total,
    snapshot.decode_counts.good,
    snapshot.decode_counts.total,
    self.config.slo_threshold,
    self.config.min_prefill_slo_samples,
    self.config.min_decode_slo_samples,
    observing=True,
)
if decision in (ProgressiveDecision.RECOVER, ProgressiveDecision.INSUFFICIENT_SAMPLES):
    self._resume_decode_source(source, records)
    return self._progressive_result(True, "source remains decode", state_trace, records)
if decision is ProgressiveDecision.COMMIT:
    remaining = tuple(item["rid"] for item in self._source_running_requests(source))
    self._execute_atomic_batch(source, target, session_id + "-final", remaining, True)
    self._assert_source_idle_after_migration(records, source)
    self._flip_idle_source_to_prefill(source, records)
    return self._progressive_result(True, "source switched to prefill", state_trace, records)
raise RuntimeError(f"unexpected observation decision: {decision}")
```

- [ ] **Step 6: Enforce ownership order in controller calls**

The exact order for each atomic batch is target prepare, initial wait, delta prepare/wait, target commit-to-ready, source finish, and target activate. Abort target and source before source finish on any earlier error.

- [ ] **Step 7: Run controller tests**

Run: `python -m pytest test/srt/test_pd_flip_progressive_controller.py test/srt/test_pd_flip_controller_quiesce.py -q`

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/playground/disaggregation/pd_flip_controller.py test/srt/test_pd_flip_progressive_controller.py test/srt/test_pd_flip_controller_quiesce.py
git commit -m "feat(pd-flip): orchestrate progressive D-to-P transitions"
```

---

### Task 9: Dynamic PD Event-Loop Redispatch

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py:1267-1362,4232-4255,6784-6812`
- Modify: `python/sglang/srt/disaggregation/prefill.py:424-504`
- Modify: `python/sglang/srt/disaggregation/decode.py:1721-1789`
- Modify: `python/sglang/srt/managers/scheduler_pp_mixin.py:170-430`
- Create: `test/srt/test_pd_runtime_event_loop_switch.py`

**Interfaces:**
- Consumes: idle-only `set_pd_runtime_role`.
- Produces: `active_pd_event_loop_role`, role-loop exit checks, and outer redispatch.

- [ ] **Step 1: Write a failing redispatch test**

```python
def test_dispatch_loop_redispatches_after_decode_loop_changes_role():
    scheduler = fake_scheduler(initial="decode")
    calls = []
    scheduler.event_loop_normal_disagg_decode = lambda: (calls.append("decode"), setattr(scheduler, "disaggregation_mode", DisaggregationMode.PREFILL))
    scheduler.event_loop_normal_disagg_prefill = lambda: (calls.append("prefill"), setattr(scheduler, "_shutdown_requested", True))
    Scheduler._run_pd_dispatch_loop(scheduler)
    assert calls == ["decode", "prefill"]
```

- [ ] **Step 2: Run and verify only one dispatch occurs**

Run: `python -m pytest test/srt/test_pd_runtime_event_loop_switch.py -q`

Expected: FAIL because `run_event_loop` currently dispatches once.

- [ ] **Step 3: Add outer redispatch and active-loop status**

```python
def _run_pd_dispatch_loop(self):
    while not self._shutdown_requested:
        dispatch_event_loop(self)

def run_event_loop(self):
    self._shutdown_requested = False
    with self.device_module.StreamContext(self.schedule_stream):
        self._run_pd_dispatch_loop()

def _pd_role_loop_should_exit(self, expected: DisaggregationMode) -> bool:
    return self.disaggregation_mode != expected or self._shutdown_requested
```

Set `active_pd_event_loop_role` on loop entry and clear it on return. Add it to runtime-role status.

- [ ] **Step 4: Add safe checks to every PD loop variant**

At the top of each normal, overlap, and PP disaggregated loop iteration, return when `_pd_role_loop_should_exit(expected_role)` is true. For overlap loops, assert `result_queue` is empty before return; runtime role mutation remains idle-only.

- [ ] **Step 5: Make controller wait for active-loop agreement**

After `/runtime_role/set`, poll status until `role == active_event_loop_role == "prefill"` before updating the router.

- [ ] **Step 6: Run scheduler tests**

Run: `python -m pytest test/srt/test_pd_runtime_event_loop_switch.py test/srt/test_pd_runtime_role_switch.py -q`

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/managers/scheduler.py python/sglang/srt/disaggregation/prefill.py python/sglang/srt/disaggregation/decode.py python/sglang/srt/managers/scheduler_pp_mixin.py test/srt/test_pd_runtime_event_loop_switch.py
git commit -m "feat(pd-flip): redispatch scheduler loops after role change"
```

---

### Task 10: Idempotency, Reconciliation, and Observability

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py:1315-1707,2167-2209`
- Modify: `python/sglang/srt/managers/scheduler.py:1267-1301,3801-3920`
- Modify: `scripts/playground/disaggregation/pd_flip_migration_measure.py`
- Create: `test/srt/test_pd_flip_reconciliation.py`

**Interfaces:**
- Consumes: session states from Tasks 6-8.
- Produces: `PDFlipSessionJournal`, `reconcile_session(session_id)`, idempotent action responses, and raw per-request stitch/ownership timing fields.

- [ ] **Step 1: Write failing reconciliation tests**

```python
def test_reconcile_activates_ready_target_after_source_finished():
    controller, client = interrupted_scenario(source_state="released", target_state="ready_to_activate")
    result = controller.reconcile_session("s")
    assert result.success
    assert client.steps == ["activate_target"]

def test_reconcile_does_not_repeat_active_session():
    controller, client = interrupted_scenario(source_state="released", target_state="active")
    assert controller.reconcile_session("s").success
    assert client.steps == []
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest test/srt/test_pd_flip_reconciliation.py -q`

Expected: FAIL because reconciliation does not exist.

- [ ] **Step 3: Make worker actions idempotent**

For matching session IDs, repeat prepare/status/commit/finish/activate/abort calls return the existing terminal state. A conflicting session ID returns `success=false` without mutating the active session.

- [ ] **Step 4: Add an atomic controller session journal**

```python
class PDFlipSessionJournal:
    def __init__(self, path: Path):
        self.path = path

    def write(self, record: JsonDict) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def read(self) -> Optional[JsonDict]:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
```

Add `session_journal_path` to `PDClusterConfig`, defaulting to `pd_flip_session.json` beside the controller config. Persist source name/URL, target name/URL, session ID, batch rids, phase, and whether source finish completed before every ownership-changing action.

- [ ] **Step 5: Implement controller reconciliation matrix**

```python
def reconcile_session(self, session_id):
    record = self.session_journal.read()
    if record is None or record["session_id"] != session_id:
        return FlipExecutionResult(False, "session journal not found", "d_to_p", None, None, None)
    source = self._migration_status(record["source_url"], session_id)
    target = self._migration_status(record["target_url"], session_id)
    if source["state"] == "released" and target["state"] == "ready_to_activate":
        return self._activate_target(session_id)
    if target["state"] == "active":
        return FlipExecutionResult(True, "session already active", "d_to_p", record["source_name"], "decode", record["target_name"])
    if source["state"] != "released" and target["state"] in {"prepared", "transferred_held", "ready_to_activate"}:
        return self._abort_session_before_source_finish(session_id)
    return FlipExecutionResult(False, "session state requires operator recovery", "d_to_p", record["source_name"], None, record["target_name"])
```

- [ ] **Step 6: Export exact stitch and ownership fields**

Add P/H/C0/C1, mode, Mooncake/source/delta bytes and durations, held/freeze/commit/activation times, source queue, final owner, output boundary, configured/effective ratio, capacity fallback count, and SLO counts to status and measurement CSV/JSON.

- [ ] **Step 7: Run reconciliation and measurement tests**

Run: `python -m pytest test/srt/test_pd_flip_reconciliation.py test/srt/test_pd_flip_migration_measure.py -q`

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/playground/disaggregation/pd_flip_controller.py python/sglang/srt/managers/scheduler.py scripts/playground/disaggregation/pd_flip_migration_measure.py test/srt/test_pd_flip_reconciliation.py test/srt/test_pd_flip_migration_measure.py
git commit -m "feat(pd-flip): reconcile and measure progressive sessions"
```

---

### Task 11: Four-Node Deployment and End-to-End Verification

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_docker/README.md`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/env.example`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_worker.sh`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_controller.sh`
- Create: `test/srt/test_pd_flip_progressive_contract.py`

**Interfaces:**
- Consumes: completed controller and worker implementation.
- Produces: documented `1P3D -> 2P2D` runbook and an automated contract test.

- [ ] **Step 1: Add a failing contract test**

```python
def test_progressive_contract_exposes_required_endpoints_and_knobs():
    assert route_exists("/pd_flip/migration/target/activate")
    assert worker_arg_exists("--enable-pd-flip-hicache-stitch")
    assert env_value("PD_FLIP_FIRST_MIGRATION_RATIO") == "0.5"
    assert env_value("PD_FLIP_OBSERVATION_SECONDS") == "10"
```

- [ ] **Step 2: Run and verify missing deployment contract**

Run: `python -m pytest test/srt/test_pd_flip_progressive_contract.py -q`

Expected: FAIL until scripts and routes are fully wired.

- [ ] **Step 3: Document the exact four-node topology**

```text
node0: prefill
node1: decode
node2: decode source selected for D-to-P
node3: decode target
final: node0/node2 prefill, node1/node3 decode
```

Document required Mooncake metadata/master/store services, HiCache write policy, worker flags, controller variables, status checks, abort/recovery commands, and expected artifacts.

- [ ] **Step 4: Run the local regression suite**

Run:

```bash
python -m pytest \
  test/srt/test_pd_flip_progressive_policy.py \
  test/srt/test_pd_flip_observation_window.py \
  test/srt/test_pd_flip_capacity_status.py \
  test/srt/test_pd_flip_selected_batch.py \
  test/srt/test_pd_flip_progressive_controller.py \
  test/srt/test_pd_flip_hicache_stitch.py \
  test/srt/test_pd_flip_atomic_batch.py \
  test/srt/test_pd_runtime_event_loop_switch.py \
  test/srt/test_pd_flip_reconciliation.py \
  test/srt/test_pd_flip_progressive_contract.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Run existing PD regression tests**

Run:

```bash
python -m pytest \
  test/srt/test_pd_runtime_role_switch.py \
  test/srt/test_pd_flip_state_machine.py \
  test/srt/test_disaggregation_fake_decode.py \
  test/srt/test_pd_flip_migration_accounting.py \
  test/srt/test_pd_flip_active_decode_handoff.py \
  test/srt/test_pd_flip_controller_quiesce.py \
  test/srt/test_pd_flip_migration_measure.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Run four-node Mooncake experiment**

Use the documented `1P3D` topology and mixed short/long trace. Verify one run for each mode:

```text
full_prefix_stitch
partial_prefix_stitch
source_decode_full_fallback
SLO recovery without role flip
persistent prefill risk with successful D-to-P
```

Expected: no missing/duplicate output; final worker/router/event-loop roles agree; raw artifacts contain all SLO, ratio, capacity, H/P/C0/C1, transfer, ownership, and latency fields.

- [ ] **Step 7: Commit**

```bash
git add scripts/playground/disaggregation/pd_flip_docker/README.md scripts/playground/disaggregation/pd_flip_docker/env.example scripts/playground/disaggregation/pd_flip_docker/run_worker.sh scripts/playground/disaggregation/pd_flip_docker/run_controller.sh test/srt/test_pd_flip_progressive_contract.py
git commit -m "docs(pd-flip): add progressive four-node runbook"
```

---

## Final Verification

- [ ] Run `git diff --check` and ensure no whitespace errors.
- [ ] Confirm every new control endpoint is admin-authenticated.
- [ ] Confirm no source or target session can activate a strict subset of a batch.
- [ ] Confirm Mooncake zero-hit fallback reserves enough target KV before migration starts.
- [ ] Confirm observation decisions use only post-first-migration samples.
- [ ] Confirm recovery leaves first-batch requests on the target and resumes source decode admission.
- [ ] Confirm final D-to-P updates worker event loop before router role.
- [ ] Confirm output relay discards sequences at or below the last source-emitted sequence.
- [ ] Package the four-node raw artifacts and attach the exact configuration used.
