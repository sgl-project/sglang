# PD Runtime Role Flip Four-Node Docker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify SGLang so four Dockerized 8-GPU nodes can switch between prefill and decode roles in process, with D->P switching time dominated by active decode KV/request migration rather than process restart.

**Architecture:** Keep model weights, TP/DP process groups, and KV pools resident. Add a runtime PD role layer on top of existing `--disaggregation-mode`: the flag remains the initial role, while `--enable-pd-runtime-role-switch` eagerly initializes both prefill and decode PD queues and exposes controlled role mutation. A Python controller drives router drain, SGLang PREPARING/FLIPPING, active decode migration, router refresh, and four-node Docker experiment validation.

**Tech Stack:** Python dataclasses/FastAPI/FanOutCommunicator, SGLang scheduler/tokenizer manager, existing Mooncake/NIXL/fake transfer abstractions, Rust `experimental/sgl-router`, pytest, cargo tests, Docker host networking.

---

## Execution Rules

- Execute tasks in order.
- Commit after each task that passes its verification command.
- Do not start the four-node Docker experiment until unit tests for scheduler role switch, migration state, router drain, and controller dry-run pass.
- Keep the old `docs/superpowers/plans/2026-06-18-pd-flip-active-decode-migration.md` as historical context. This plan supersedes it for the production experiment.

## File Structure

- `python/sglang/srt/disaggregation/flip_state_machine.py`: role flip state transitions and status contract.
- `python/sglang/srt/server_args.py`: new runtime role switch flags.
- `python/sglang/srt/managers/scheduler.py`: hybrid prefill/decode queue initialization, active role admission, migration source/target hooks, role mutation.
- `python/sglang/srt/managers/tokenizer_manager.py`: active role stored in tokenizer manager and bootstrap service startup for hot-switch workers.
- `python/sglang/srt/managers/tokenizer_control_mixin.py`: scheduler control-plane fanout for role status/switch and migration endpoints.
- `python/sglang/srt/entrypoints/http_server.py`: HTTP admin endpoints and `/server_info` active role reporting.
- `python/sglang/srt/managers/io_struct.py`: request/response dataclasses for runtime role control and migration handoff.
- `experimental/sgl-router/src/workers/worker.rs`: per-worker draining state.
- `experimental/sgl-router/src/workers/registry.rs`: worker lookup and mode refresh helpers.
- `experimental/sgl-router/src/server/routes/pd_flip.rs`: router admin endpoints for drain, undrain, role refresh, and pool status.
- `experimental/sgl-router/src/server/app.rs`: mount router admin endpoints.
- `scripts/playground/disaggregation/pd_flip_controller.py`: production-shaped four-node experiment controller.
- `scripts/playground/disaggregation/pd_flip_docker/`: Docker compose/env templates for four physical nodes.
- `test/srt/test_pd_runtime_role_switch.py`: scheduler and HTTP control tests.
- `test/srt/test_pd_flip_active_decode_handoff.py`: migration handoff unit tests.
- `test/srt/test_pd_flip_controller.py`: controller unit tests with fake HTTP workers/router.
- `experimental/sgl-router/tests/component/pd_flip_admin.rs`: router drain/refresh component tests.

---

### Task 1: Runtime Role Flags And Status Contract

**Files:**
- Modify: `python/sglang/srt/server_args.py`
- Modify: `python/sglang/srt/disaggregation/flip_state_machine.py`
- Modify: `python/sglang/srt/managers/io_struct.py`
- Test: `test/srt/test_pd_runtime_role_switch.py`

- [ ] **Step 1: Add failing tests for CLI defaults and state-machine status**

Add these tests to `test/srt/test_pd_runtime_role_switch.py`:

```python
from sglang.srt.disaggregation.flip_state_machine import (
    ClusterSnapshot,
    FlipDirection,
    FlipState,
    FlipStateMachine,
    SLOThresholdFlipEvaluator,
)
from sglang.srt.server_args import ServerArgs


def test_runtime_role_switch_flags_default_off():
    args = ServerArgs(model_path="dummy")
    assert args.enable_pd_runtime_role_switch is False
    assert args.pd_runtime_initial_role is None


def test_d_to_p_status_declares_hot_switch_and_migration():
    machine = FlipStateMachine(
        SLOThresholdFlipEvaluator(slo_threshold=0.9),
        min_window_seconds=0,
    )
    event = machine.tick(
        ClusterSnapshot(
            timestamp=1.0,
            role="decode",
            prefill_nodes=1,
            decode_nodes=3,
            prefill_slo_attainment=0.5,
            decode_slo_attainment=1.0,
        )
    )
    assert event.to_state == FlipState.PREPARING
    status = machine.status()
    assert status["direction"] == FlipDirection.D_TO_P.value
    assert status["requires_active_request_migration"] is True
    assert status["can_hot_switch_in_process"] is True
    assert status["requires_process_restart"] is False
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
PYTHONPATH=python python -m pytest test/srt/test_pd_runtime_role_switch.py -q
```

Expected: FAIL because `enable_pd_runtime_role_switch`, `pd_runtime_initial_role`, and hot-switch status are not implemented.

- [ ] **Step 3: Add server args**

Add dataclass fields near existing PD flip fields in `python/sglang/srt/server_args.py`:

```python
enable_pd_runtime_role_switch: bool = False
pd_runtime_initial_role: Optional[Literal["prefill", "decode"]] = None
```

Add CLI args near `--enable-pd-flip-state-machine`:

```python
parser.add_argument(
    "--enable-pd-runtime-role-switch",
    action="store_true",
    default=ServerArgs.enable_pd_runtime_role_switch,
    help="Enable in-process PD role switching. The initial role is taken from --disaggregation-mode unless --pd-runtime-initial-role is set.",
)
parser.add_argument(
    "--pd-runtime-initial-role",
    type=str,
    choices=["prefill", "decode"],
    default=ServerArgs.pd_runtime_initial_role,
    help="Initial active PD role for runtime role switch workers.",
)
```

In validation, reject incompatible startup:

```python
if self.enable_pd_runtime_role_switch:
    initial = self.pd_runtime_initial_role or self.disaggregation_mode
    if initial not in ("prefill", "decode"):
        raise ValueError(
            "--enable-pd-runtime-role-switch requires --disaggregation-mode prefill/decode or --pd-runtime-initial-role"
        )
    self.disaggregation_mode = initial
```

- [ ] **Step 4: Update flip status contract**

In `flip_state_machine.py`, change `status()` so active D->P/P->D decisions report hot-switch capability:

```python
"requires_external_orchestrator": self.direction != FlipDirection.NONE,
"can_hot_switch_in_process": True,
"requires_process_restart": False,
```

Keep `requires_active_request_migration` and `requires_kv_migration` true only for D->P.

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=python python -m pytest test/srt/test_pd_runtime_role_switch.py test/srt/test_pd_flip_state_machine.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/server_args.py python/sglang/srt/disaggregation/flip_state_machine.py test/srt/test_pd_runtime_role_switch.py
git commit -m "feat(pd-flip): add runtime role switch flags"
```

---

### Task 2: Hybrid Scheduler Queue Initialization

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `python/sglang/srt/managers/scheduler_components/load_inquirer.py`
- Modify: `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`
- Test: `test/srt/test_pd_runtime_role_switch.py`

- [ ] **Step 1: Add failing unit tests for active-role admission**

Add lightweight scheduler fixture tests using a minimal object with the same attributes read by the new helpers:

```python
from sglang.srt.disaggregation.utils import DisaggregationMode


class DummyRuntimeRoleMixin:
    def __init__(self):
        self.disaggregation_mode = DisaggregationMode.DECODE
        self.server_args = type("Args", (), {"enable_pd_runtime_role_switch": True})()


def test_runtime_role_helper_uses_active_mode():
    obj = DummyRuntimeRoleMixin()
    assert obj.disaggregation_mode == DisaggregationMode.DECODE
    obj.disaggregation_mode = DisaggregationMode.PREFILL
    assert obj.disaggregation_mode == DisaggregationMode.PREFILL
```

Replace this dummy with direct imports once helper functions are created in Step 3.

- [ ] **Step 2: Split current `init_disaggregation` into role-specific helpers**

Refactor `Scheduler.init_disaggregation()` into:

```python
def init_disaggregation(self):
    self.disaggregation_mode = DisaggregationMode(self.server_args.disaggregation_mode)
    self.transfer_backend = TransferBackend(self.server_args.disaggregation_transfer_backend)
    self._init_disagg_common()
    if self.server_args.enable_pd_runtime_role_switch:
        self._init_decode_disaggregation()
        self._init_prefill_disaggregation()
    elif self.disaggregation_mode == DisaggregationMode.DECODE:
        self._init_decode_disaggregation()
    elif self.disaggregation_mode == DisaggregationMode.PREFILL:
        self._init_prefill_disaggregation()
```

Move the existing decode branch into `_init_decode_disaggregation()` and the existing prefill branch into `_init_prefill_disaggregation()`. Preserve the current object names:

```python
self.disagg_decode_transfer_queue
self.disagg_decode_prealloc_queue
self.disagg_prefill_bootstrap_queue
self.disagg_prefill_inflight_queue
```

- [ ] **Step 3: Keep active role as the single admission switch**

Add helpers in `Scheduler`:

```python
def pd_runtime_role(self) -> str:
    return DisaggregationMode.to_engine_type(self.disaggregation_mode.value)

def pd_runtime_role_switch_enabled(self) -> bool:
    return bool(getattr(self.server_args, "enable_pd_runtime_role_switch", False))
```

Leave `_add_request_to_queue()` using `self.disaggregation_mode`; this is the desired active-role gate.

- [ ] **Step 4: Guard metrics against inactive queues**

Update `load_inquirer.py` and `metrics_reporter.py` so they count only queues for `self.disaggregation_mode`, but tolerate both queues being initialized. Do not count inactive-role queues in load balancing.

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=python python -m pytest test/srt/test_pd_runtime_role_switch.py test/srt/test_pd_flip_internal_state_update.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/scheduler_components/load_inquirer.py python/sglang/srt/managers/scheduler_components/metrics_reporter.py test/srt/test_pd_runtime_role_switch.py
git commit -m "feat(pd-flip): initialize hybrid PD scheduler queues"
```

---

### Task 3: Worker Runtime Role Control Endpoints

**Files:**
- Modify: `python/sglang/srt/managers/io_struct.py`
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `python/sglang/srt/managers/tokenizer_control_mixin.py`
- Modify: `python/sglang/srt/entrypoints/http_server.py`
- Modify: `python/sglang/srt/managers/tokenizer_manager.py`
- Modify: `python/sglang/srt/managers/disagg_service.py`
- Test: `test/srt/test_pd_runtime_role_switch.py`

- [ ] **Step 1: Add failing tests for role mutation**

Add tests that call scheduler role control directly:

```python
def test_apply_runtime_role_rejects_without_hot_switch(fake_scheduler):
    fake_scheduler.server_args.enable_pd_runtime_role_switch = False
    out = fake_scheduler.set_pd_runtime_role("prefill")
    assert out.success is False
    assert "requires --enable-pd-runtime-role-switch" in out.message


def test_apply_runtime_role_updates_active_mode_and_server_args(fake_scheduler):
    fake_scheduler.server_args.enable_pd_runtime_role_switch = True
    fake_scheduler.disaggregation_mode = DisaggregationMode.DECODE
    fake_scheduler.server_args.disaggregation_mode = "decode"
    out = fake_scheduler.set_pd_runtime_role("prefill")
    assert out.success is True
    assert fake_scheduler.disaggregation_mode == DisaggregationMode.PREFILL
    assert fake_scheduler.server_args.disaggregation_mode == "prefill"
```

- [ ] **Step 2: Add IO dataclasses**

Add to `io_struct.py`:

```python
@dataclass
class PDSetRuntimeRoleReq(BaseReq):
    role: str
    reason: str = ""


@dataclass
class PDSetRuntimeRoleReqOutput(BaseReq):
    success: bool
    message: str
    role: str
```

- [ ] **Step 3: Add scheduler setter**

Add to `Scheduler`:

```python
def set_pd_runtime_role(self, role: str, reason: str = "") -> PDSetRuntimeRoleReqOutput:
    if not self.pd_runtime_role_switch_enabled():
        return PDSetRuntimeRoleReqOutput(False, "runtime role switch requires --enable-pd-runtime-role-switch", self.pd_runtime_role())
    if role not in ("prefill", "decode"):
        return PDSetRuntimeRoleReqOutput(False, f"invalid role {role}", self.pd_runtime_role())
    next_mode = DisaggregationMode(role)
    if next_mode == self.disaggregation_mode:
        return PDSetRuntimeRoleReqOutput(True, "role unchanged", role)
    if not self.pd_flip_is_idle_for_commit(self.build_pd_flip_snapshot()):
        return PDSetRuntimeRoleReqOutput(False, "worker is not idle for role switch", self.pd_runtime_role())
    self.disaggregation_mode = next_mode
    self.server_args.disaggregation_mode = role
    return PDSetRuntimeRoleReqOutput(True, reason or "role updated", role)
```

- [ ] **Step 4: Wire communicator and HTTP**

Add a communicator entry in `tokenizer_control_mixin.py`:

```python
("set_pd_runtime_role", PDSetRuntimeRoleReqOutput),
```

Add method:

```python
async def set_pd_runtime_role(self: TokenizerManager, obj: PDSetRuntimeRoleReq) -> List[PDSetRuntimeRoleReqOutput]:
    self.auto_create_handle_loop()
    responses = await self.set_pd_runtime_role_communicator(obj)
    if responses and all(r.success for r in responses):
        self.disaggregation_mode = DisaggregationMode(obj.role)
        self.server_args.disaggregation_mode = obj.role
    return responses
```

Add scheduler dispatch entry where `Scheduler` builds its `TypeBasedDispatcher`:

```python
(PDSetRuntimeRoleReq, self.set_pd_runtime_role),
```

Add endpoint in `http_server.py`:

```python
@app.post("/pd_flip/runtime_role")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def set_pd_runtime_role(obj: PDSetRuntimeRoleReq, request: Request):
    return await _global_state.tokenizer_manager.set_pd_runtime_role(obj)
```

- [ ] **Step 5: Start bootstrap service for hot-switch decode workers**

Modify `start_disagg_service(server_args)` so a hot-switch worker starts the bootstrap server even when the initial role is decode:

```python
starts_prefill_bootstrap = (
    DisaggregationMode(server_args.disaggregation_mode) == DisaggregationMode.PREFILL
    or getattr(server_args, "enable_pd_runtime_role_switch", False)
)
```

Use `starts_prefill_bootstrap` in the existing prefill branch.

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONPATH=python python -m pytest test/srt/test_pd_runtime_role_switch.py test/srt/test_pd_flip_internal_state_update.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/managers/io_struct.py python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/tokenizer_control_mixin.py python/sglang/srt/entrypoints/http_server.py python/sglang/srt/managers/tokenizer_manager.py python/sglang/srt/managers/disagg_service.py test/srt/test_pd_runtime_role_switch.py
git commit -m "feat(pd-flip): expose runtime PD role control"
```

---

### Task 4: Router Drain And Role Refresh

**Files:**
- Modify: `experimental/sgl-router/src/workers/worker.rs`
- Modify: `experimental/sgl-router/src/workers/registry.rs`
- Create: `experimental/sgl-router/src/server/routes/pd_flip.rs`
- Modify: `experimental/sgl-router/src/server/routes/mod.rs`
- Modify: `experimental/sgl-router/src/server/app.rs`
- Test: `experimental/sgl-router/tests/component/pd_flip_admin.rs`

- [ ] **Step 1: Add failing component tests**

Create `experimental/sgl-router/tests/component/pd_flip_admin.rs` with tests for:

```rust
#[tokio::test]
async fn drained_worker_is_not_returned_as_healthy() {
    let w = worker("http://d0:30000", WorkerMode::Decode);
    assert!(w.is_routable());
    w.set_draining(true);
    assert!(!w.is_routable());
}

#[tokio::test]
async fn mode_refresh_moves_worker_between_pd_pools() {
    let registry = WorkerRegistry::default();
    registry.add(spec("n0", WorkerMode::Decode)).unwrap();
    registry.set_worker_mode(&WorkerId("n0".into()), WorkerMode::Prefill, Some(8998)).unwrap();
    assert_eq!(registry.workers_for_mode(&ModelId("m".into()), WorkerMode::Decode).len(), 0);
    assert_eq!(registry.workers_for_mode(&ModelId("m".into()), WorkerMode::Prefill).len(), 1);
}
```

- [ ] **Step 2: Add draining state**

In `worker.rs`, add:

```rust
draining: AtomicBool,
```

Initialize it to `false`. Add:

```rust
pub fn set_draining(&self, draining: bool) {
    self.draining.store(draining, Ordering::Relaxed);
}

pub fn is_draining(&self) -> bool {
    self.draining.load(Ordering::Relaxed)
}

pub fn is_routable(&self) -> bool {
    !self.is_draining() && self.breaker.would_allow()
}
```

- [ ] **Step 3: Use routable workers in registry**

Change `healthy_workers_for()` to filter with `w.is_routable()`.

Add registry helper:

```rust
pub fn set_worker_mode(&self, id: &WorkerId, mode: WorkerMode, bootstrap_port: Option<u16>) -> Result<(), String> {
    let Some(w) = self.get(id) else {
        return Err(format!("unknown worker {}", id.0));
    };
    w.set_mode(mode);
    w.set_bootstrap_port(bootstrap_port);
    Ok(())
}
```

Add `Worker::set_bootstrap_port()` using an interior mutable field. Use `Mutex<Option<u16>>` for bootstrap port if the current immutable field blocks updates.

- [ ] **Step 4: Add router admin routes**

Create `routes/pd_flip.rs` with:

```rust
#[derive(Deserialize)]
pub struct WorkerAdminReq {
    pub worker_id: String,
}

#[derive(Serialize)]
pub struct WorkerAdminResp {
    pub success: bool,
    pub message: String,
}
```

Implement:

- `POST /pd_flip/router/drain`
- `POST /pd_flip/router/undrain`
- `POST /pd_flip/router/refresh_roles`
- `GET /pd_flip/router/pools`

`refresh_roles` calls `/server_info` on each worker, maps `disaggregation_mode` to `WorkerMode`, and updates the in-memory worker.

- [ ] **Step 5: Mount routes**

In `server/routes/mod.rs` add:

```rust
pub mod pd_flip;
```

In `server/app.rs` add:

```rust
.route("/pd_flip/router/drain", post(crate::server::routes::pd_flip::drain_worker))
.route("/pd_flip/router/undrain", post(crate::server::routes::pd_flip::undrain_worker))
.route("/pd_flip/router/refresh_roles", post(crate::server::routes::pd_flip::refresh_roles))
.route("/pd_flip/router/pools", get(crate::server::routes::pd_flip::pools))
```

- [ ] **Step 6: Run router tests**

Run:

```bash
cd experimental/sgl-router
cargo test pd_flip_admin --tests
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add experimental/sgl-router/src/workers/worker.rs experimental/sgl-router/src/workers/registry.rs experimental/sgl-router/src/server/routes/pd_flip.rs experimental/sgl-router/src/server/routes/mod.rs experimental/sgl-router/src/server/app.rs experimental/sgl-router/tests/component/pd_flip_admin.rs
git commit -m "feat(router): add PD flip drain and role refresh admin APIs"
```

---

### Task 5: Controller Skeleton, Metrics, And Dry-Run Flip

**Files:**
- Create: `scripts/playground/disaggregation/pd_flip_controller.py`
- Test: `test/srt/test_pd_flip_controller.py`

- [ ] **Step 1: Add controller tests with fake HTTP**

Create tests for inventory, decision, and dry-run ordering:

```python
def test_controller_selects_decode_for_d_to_p_when_prefill_slo_low(fake_cluster):
    ctl = fake_cluster.controller(prefill_slo=0.70, decode_slo=0.99)
    decision = ctl.decide()
    assert decision.direction == "d_to_p"
    assert decision.source.role == "decode"
    assert decision.target.role == "decode"


def test_dry_run_d_to_p_orders_router_worker_and_migration_calls(fake_cluster):
    ctl = fake_cluster.controller(prefill_slo=0.70, decode_slo=0.99, dry_run=True)
    ctl.run_once()
    assert fake_cluster.calls == [
        ("router.drain", "node2"),
        ("worker.source_start", "node2"),
        ("worker.target_prepare", "node3"),
        ("poll_migration", "node2", "node3"),
        ("worker.source_finish", "node2"),
        ("worker.runtime_role", "node2", "prefill"),
        ("router.refresh_roles", None),
        ("router.undrain", "node2"),
    ]
```

- [ ] **Step 2: Implement data models**

In `pd_flip_controller.py`:

```python
@dataclass
class WorkerNode:
    name: str
    url: str
    role: str
    state: str
    running_reqs: int
    waiting_reqs: int
    token_usage: float


@dataclass
class FlipDecision:
    direction: str
    source: WorkerNode
    target: Optional[WorkerNode]
    reason: str
```

- [ ] **Step 3: Implement HTTP client**

Methods:

```python
get_server_info(url)
get_loads(url)
router_drain(worker_id)
router_undrain(worker_id)
router_refresh_roles()
migration_source_start(source_url, target_url, session_id)
migration_target_prepare(target_url, source_url, manifests, session_id)
migration_status(url)
migration_source_finish(source_url, session_id, released_rids)
set_runtime_role(url, role, reason)
```

- [ ] **Step 4: Implement decision policy**

Use conservative thresholds:

```python
if prefill_slo < slo_threshold and decode_count >= 2:
    direction = "d_to_p"
if decode_slo < slo_threshold and prefill_count >= 2:
    direction = "p_to_d"
```

Candidate rules:

- D->P source: decode node with lowest `running_reqs`, then lowest `token_usage`.
- D->P target: decode node not equal to source with lowest `token_usage`.
- P->D source: prefill node with lowest `prefill_bootstrap + prefill_inflight`.

- [ ] **Step 5: Implement D->P dry-run flow**

The controller flow must match the test order in Step 1. `dry_run=True` records calls and validates prerequisites without changing workers.

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONPATH=python python -m pytest test/srt/test_pd_flip_controller.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/playground/disaggregation/pd_flip_controller.py test/srt/test_pd_flip_controller.py
git commit -m "feat(pd-flip): add four-node controller skeleton"
```

---

### Task 6: Make D->P Migration Adopt Requests On Target Decode

**Files:**
- Modify: `python/sglang/srt/managers/io_struct.py`
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `python/sglang/srt/disaggregation/decode.py`
- Test: `test/srt/test_pd_flip_active_decode_handoff.py`

- [ ] **Step 1: Add failing tests for target adoption**

Add tests that assert migrated requests are inserted into target decode scheduling after KV arrives:

```python
def test_target_migration_adopts_request_after_kv_success(fake_decode_scheduler):
    manifest = {
        "rid": "rid-1",
        "origin_input_ids": [1, 2, 3],
        "output_ids": [4, 5],
        "kv_committed_len": 4,
        "sampling_params": {"max_new_tokens": 8},
        "http_worker_ipc": "ipc-source",
    }
    out = fake_decode_scheduler.prepare_pd_flip_migration_target(
        PDFlipMigrationTargetPrepareReq(
            session_id="s1",
            source_url="http://node2:30000",
            manifests=[manifest],
            adopt_on_success=True,
        )
    )
    assert out.success is True
    fake_decode_scheduler.force_migration_success("rid-1")
    fake_decode_scheduler.get_pd_flip_migration_status(PDFlipMigrationStatusReq())
    assert fake_decode_scheduler.has_waiting_or_running_req("rid-1")
```

- [ ] **Step 2: Extend migration dataclasses**

Add fields:

```python
adopt_on_success: bool = False
```

to `PDFlipMigrationTargetPrepareReq`, and include `http_worker_ipc`, `stream`, `return_logprob`, `logprob_start_len`, and `time_stats` in migration manifests.

- [ ] **Step 3: Preserve output routing in manifests**

In `_pd_flip_build_migration_manifest()`, include:

```python
"http_worker_ipc": getattr(req, "http_worker_ipc", None),
"stream": bool(getattr(req, "stream", False)),
"logprob_start_len": getattr(req, "logprob_start_len", -1),
```

In `_pd_flip_manifest_to_req()`, pass:

```python
http_worker_ipc=manifest.get("http_worker_ipc"),
```

and restore `req.stream` and `req.logprob_start_len`.

- [ ] **Step 4: Adopt request after target transfer succeeds**

In `_pd_flip_target_pump_transfer()`, replace release-only success handling with:

```python
if session.get("adopt_on_success", False):
    self._pd_flip_adopt_target_request(entry)
else:
    self._pd_flip_release_target_request(entry)
```

Implement:

```python
def _pd_flip_adopt_target_request(self, entry: Dict[str, Any]) -> None:
    if entry.get("request_adopted"):
        return
    decode_req = entry["decode_req"]
    req = decode_req.req
    req.init_next_round_input(self.tree_cache)
    self.waiting_queue.append(req)
    entry["request_adopted"] = True
```

If direct `waiting_queue.append()` is insufficient for decode-mode prebuilt batches, use the same path `DecodePreallocQueue.pop_preallocated()` uses when moving a completed transfer into waiting. The test must verify the request appears in the scheduler’s normal decode path.

- [ ] **Step 5: Source release should not emit client-facing failure after adoption**

Change `_pd_flip_release_source_requests()` so source requests are stopped with an internal migrated finish reason that does not stream an HTTP 503 after target adoption. Add a local finish reason string:

```python
"Request migrated during PD role flip; output continues on target decode."
```

The source must free KV and stop scheduling the request.

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONPATH=python python -m pytest test/srt/test_pd_flip_active_decode_handoff.py test/srt/test_pd_flip_internal_state_update.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/managers/io_struct.py python/sglang/srt/managers/scheduler.py python/sglang/srt/disaggregation/decode.py test/srt/test_pd_flip_active_decode_handoff.py
git commit -m "feat(pd-flip): adopt migrated decode requests on target"
```

---

### Task 7: State Machine Commit Performs Runtime Role Mutation

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `python/sglang/srt/disaggregation/flip_state_machine.py`
- Test: `test/srt/test_pd_runtime_role_switch.py`
- Test: `test/srt/test_pd_flip_internal_state_update.py`

- [ ] **Step 1: Add failing tests for D->P commit**

```python
def test_d_to_p_commit_changes_decode_to_prefill_after_migration_release(fake_scheduler):
    fake_scheduler.server_args.enable_pd_runtime_role_switch = True
    fake_scheduler.disaggregation_mode = DisaggregationMode.DECODE
    fake_scheduler.mark_idle()
    fake_scheduler.mark_migration_released()
    decision = FlipDecision(should_flip=True, direction=FlipDirection.D_TO_P)
    assert fake_scheduler.commit_pd_flip(fake_scheduler.build_pd_flip_snapshot(), decision)
    assert fake_scheduler.disaggregation_mode == DisaggregationMode.PREFILL
```

- [ ] **Step 2: Update prepare gate**

`prepare_pd_flip()` must return true when:

- direction is D->P and migration is released,
- or direction is P->D and prefill queues are empty,
- and the worker is idle for role switch.

Remove dependency on external `pd_flip_prepare_ack` for the hot-switch path. Keep ack support for `enable_pd_runtime_role_switch=False`.

- [ ] **Step 3: Update commit callback**

In `commit_pd_flip()`:

```python
if self.pd_runtime_role_switch_enabled():
    target = "prefill" if decision.direction == FlipDirection.D_TO_P else "decode"
    out = self.set_pd_runtime_role(target, reason=decision.reason)
    return out.success
```

Keep old external commit ack path for non-hot-switch experiments.

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=python python -m pytest test/srt/test_pd_runtime_role_switch.py test/srt/test_pd_flip_internal_state_update.py test/srt/test_pd_flip_state_machine.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/managers/scheduler.py python/sglang/srt/disaggregation/flip_state_machine.py test/srt/test_pd_runtime_role_switch.py test/srt/test_pd_flip_internal_state_update.py
git commit -m "feat(pd-flip): commit runtime PD role mutation"
```

---

### Task 8: Four-Node Docker Experiment Harness

**Files:**
- Create: `scripts/playground/disaggregation/pd_flip_docker/env.example`
- Create: `scripts/playground/disaggregation/pd_flip_docker/run_worker.sh`
- Create: `scripts/playground/disaggregation/pd_flip_docker/run_router.sh`
- Create: `scripts/playground/disaggregation/pd_flip_docker/run_controller.sh`
- Create: `scripts/playground/disaggregation/pd_flip_docker/README.md`

- [ ] **Step 1: Create `env.example`**

```bash
MODEL_PATH=/models/your-model
SGLANG_REPO=/home/tiancij/sglang
IMAGE=lmsysorg/sglang:latest
TRANSFER_BACKEND=mooncake
IB_DEVICE=mlx5_0
PORT=30000
BOOTSTRAP_PORT=8998
ROUTER_PORT=8000
NODE0=http://10.0.0.10:30000
NODE1=http://10.0.0.11:30000
NODE2=http://10.0.0.12:30000
NODE3=http://10.0.0.13:30000
```

- [ ] **Step 2: Create worker runner**

`run_worker.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source "${ENV_FILE:-./env.example}"
ROLE="${1:?usage: run_worker.sh prefill|decode}"
LOCAL_IP="${2:?usage: run_worker.sh prefill|decode <local-ip>}"
docker run --rm --gpus all --network host --ipc host --privileged \
  -v "${SGLANG_REPO}:/sgl-workspace/sglang" \
  -v "${MODEL_PATH}:${MODEL_PATH}" \
  -v /dev/infiniband:/dev/infiniband \
  "${IMAGE}" \
  bash -lc "cd /sgl-workspace/sglang && PYTHONPATH=python python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --host ${LOCAL_IP} \
    --port ${PORT} \
    --tp 8 \
    --dp 8 \
    --enable-dp-attention \
    --disaggregation-mode ${ROLE} \
    --disaggregation-transfer-backend ${TRANSFER_BACKEND} \
    --disaggregation-bootstrap-port ${BOOTSTRAP_PORT} \
    --disaggregation-ib-device ${IB_DEVICE} \
    --enable-pd-flip-state-machine \
    --enable-pd-runtime-role-switch \
    --mem-fraction-static 0.9"
```

- [ ] **Step 3: Create router runner**

`run_router.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source "${ENV_FILE:-./env.example}"
docker run --rm --network host \
  -v "${SGLANG_REPO}:/sgl-workspace/sglang" \
  "${IMAGE}" \
  bash -lc "cd /sgl-workspace/sglang/experimental/sgl-router && cargo run --release -- \
    --host 0.0.0.0 \
    --port ${ROUTER_PORT} \
    --model-path ${MODEL_PATH} \
    --worker-urls ${NODE0} ${NODE1} ${NODE2} ${NODE3}"
```

- [ ] **Step 4: Create controller runner**

`run_controller.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source "${ENV_FILE:-./env.example}"
python scripts/playground/disaggregation/pd_flip_controller.py run-once \
  --router-url "http://127.0.0.1:${ROUTER_PORT}" \
  --worker node0="${NODE0}" \
  --worker node1="${NODE1}" \
  --worker node2="${NODE2}" \
  --worker node3="${NODE3}" \
  --slo-threshold 0.90
```

- [ ] **Step 5: Create README with exact experiment order**

Document:

```text
node0: ./run_worker.sh prefill <node0-ip>
node1: ./run_worker.sh prefill <node1-ip>
node2: ./run_worker.sh decode <node2-ip>
node3: ./run_worker.sh decode <node3-ip>
router host: ./run_router.sh
controller host: ./run_controller.sh
```

Add curl checks:

```bash
curl http://node2:30000/server_info | jq '.disaggregation_mode'
curl http://router:8000/pd_flip/router/pools | jq
```

- [ ] **Step 6: Commit**

```bash
git add scripts/playground/disaggregation/pd_flip_docker
git commit -m "docs(pd-flip): add four-node Docker experiment harness"
```

---

### Task 9: End-To-End Verification On Fake Transfer

**Files:**
- Modify only files with failing tests.

- [ ] **Step 1: Run Python unit tests**

```bash
PYTHONPATH=python python -m pytest \
  test/srt/test_pd_flip_state_machine.py \
  test/srt/test_pd_flip_internal_state_update.py \
  test/srt/test_pd_runtime_role_switch.py \
  test/srt/test_pd_flip_active_decode_handoff.py \
  test/srt/test_pd_flip_controller.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run router tests**

```bash
cd experimental/sgl-router
cargo test pd_flip_admin --tests
```

Expected: PASS.

- [ ] **Step 3: Run local fake-transfer smoke**

Use four local ports if GPUs are not available:

```bash
python scripts/playground/disaggregation/pd_flip_controller.py observe \
  --router-url http://127.0.0.1:8000 \
  --worker node0=http://127.0.0.1:30000 \
  --worker node1=http://127.0.0.1:30001 \
  --worker node2=http://127.0.0.1:30002 \
  --worker node3=http://127.0.0.1:30003
```

Expected: controller prints two prefill nodes, two decode nodes, all with `enable_pd_runtime_role_switch=true`.

- [ ] **Step 4: Measure flip latency in Docker**

Record these timestamps in controller logs:

```text
t0 router_drain_start
t1 source_migration_start
t2 target_migration_complete
t3 source_release_complete
t4 runtime_role_updated
t5 router_refresh_complete
```

Acceptance:

```text
(t5 - t0) <= 1.25 * (t2 - t1) + 2.0 seconds
```

This matches the user goal that switching time is close to KV transfer time.

- [ ] **Step 5: Commit verification notes**

Add `docs/superpowers/plans/2026-06-22-pd-runtime-role-flip-four-node-docker-results.md` with:

```text
cluster nodes:
initial roles:
model:
transfer backend:
D->P source:
D->P target:
kv transfer seconds:
total flip seconds:
router unavailable errors:
request success count:
request failure count:
```

Commit:

```bash
git add docs/superpowers/plans/2026-06-22-pd-runtime-role-flip-four-node-docker-results.md
git commit -m "test(pd-flip): record four-node Docker verification"
```
