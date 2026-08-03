# MinWM Async VAE Multi-User Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持 MinWM T2V/I2V、动态 Action/Prompt、Trace 和 Dump 行为的前提下，将 TAEHV 解码拆到独立低成本 GPU Worker，并让 Chunk N+1 的 H100 Denoising 与 Chunk N 的 VAE Decode 重叠，同时开放受严格配额保护的多用户 Session。

**Architecture:** 保留现有单体路径作为同步基线，通过 `--realtime-vae-worker-url` 启用异步路径。H100 Scheduler 在 Denoising 后返回连续 BF16 latent，Realtime Gateway 使用每 Session 一条持久 WebSocket 将 latent 发送给独立 TAEHV Worker；每 Session 最多一个 Decode、一个等待 latent，下一 Chunk 只在得到本地 credit 后进入 H100。Gateway 使用可替换 Lease Store 执行每用户单 Session 与全局容量限制，初始部署使用单副本内存 Lease，DynamoDB 实现供多 Gateway 扩容时启用。

**Tech Stack:** Python 3.12、asyncio、FastAPI WebSocket、msgspec MessagePack、PyTorch BF16、TAEHV/StreamingTAEHV、Prometheus、OpenTelemetry、pytest、Node.js WebUI tests、Kubernetes/Karpenter、AWS H100 Spot 与 L4/L40S Spot。

---

## File Map

- `python/sglang/multimodal_gen/runtime/realtime/async_vae_protocol.py`: wire identity、消息编码、shape/checksum/deadline 校验。
- `python/sglang/multimodal_gen/runtime/realtime/admission.py`: Session Lease 接口、内存实现、可选 DynamoDB 实现、TTL/配额。
- `python/sglang/multimodal_gen/runtime/realtime/async_vae_client.py`: Gateway 到 VAE Worker 的单 Session WebSocket、credit 与超时。
- `python/sglang/multimodal_gen/runtime/realtime/async_vae_worker.py`: TAEHV 权重预加载、per-Session StreamingTAEHV state、有界公平队列、FrameBatch 输出。
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/realtime/latent_handoff.py`: Denoiser 输出 latent，不执行本地 VAE。
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/generate_session.py`: generation identity、多 in-flight Chunk 生命周期、活动时间。
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_video_api.py`: 多用户准入、异步 Denoise/Decode 流水、idle/max lifetime cleanup。
- `python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py`: 类型化 latent handoff 字段。
- `python/sglang/multimodal_gen/runtime/pipelines/minwm_causal_dmd_pipeline.py`: feature flag 选择本地 VAE 或 latent handoff。
- `python/sglang/multimodal_gen/runtime/realtime/session.py`: 活动 Session 容量拒绝，禁止 LRU 淘汰。
- `python/sglang/multimodal_gen/runtime/managers/gpu_worker.py`: 可配置的每 Worker Session 上限。
- `python/sglang/multimodal_gen/runtime/server_args.py`: 异步 VAE、Session 配额、deadline 与 worker 参数。
- `python/sglang/multimodal_gen/runtime/utils/realtime_trace.py`: 异步阶段 Trace 字段脱敏与关联。
- `benchmark/minwm_realtime_async_vae/`: 并发压测、A/B 汇总和报告生成。
- `benchmark/minwm_realtime_async_vae/k8s/`: H100 Denoiser、L4/L40S VAE、Gateway/NLB 与纯 Spot NodePool。

### Task 1: Wire Protocol And Generation Identity

**Files:**
- Create: `python/sglang/multimodal_gen/runtime/realtime/async_vae_protocol.py`
- Modify: `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/generate_session.py`
- Modify: `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_adapter.py`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_protocol.py`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_realtime_runtime.py`

- [ ] **Step 1: Write failing protocol and lifecycle tests**

```python
def test_latent_header_rejects_stale_generation():
    tracker = ChunkSequenceTracker("s1", "g2")
    with pytest.raises(ProtocolViolation, match="stale generation"):
        tracker.accept(LatentChunkHeader(session_id="s1", generation_id="g1", chunk_index=0, request_id="r0", dtype="bfloat16", shape=(1, 48, 1, 30, 52), byte_length=149760, checksum="x"))

def test_generate_session_allows_two_active_chunks_and_completes_in_order():
    session = GenerateSession(max_inflight_chunks=2)
    first = session.new_chunk()
    second = session.new_chunk()
    with pytest.raises(RuntimeError, match="in-flight limit"):
        session.new_chunk()
    session.generate_chunk_completed(first)
    session.generate_chunk_completed(second)
    assert session.generate_chunk_cnt == 2
```

- [ ] **Step 2: Run tests and verify failure**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_protocol.py python/sglang/multimodal_gen/test/unit/realtime/test_realtime_runtime.py -q`

Expected: FAIL because protocol types and multi-inflight lifecycle do not exist.

- [ ] **Step 3: Implement immutable identity and monotonic sequence validation**

```python
@dataclass(frozen=True, slots=True)
class LatentChunkHeader:
    session_id: str
    generation_id: str
    request_id: str
    chunk_index: int
    dtype: str
    shape: tuple[int, ...]
    byte_length: int
    checksum: str
    event_id: int | None = None
    action_version: int = 0
    prompt_version: int = 0
    deadline_epoch_ms: int = 0

class ChunkSequenceTracker:
    def accept(self, header: LatentChunkHeader) -> AcceptDisposition:
        if header.generation_id != self.generation_id:
            raise ProtocolViolation("stale generation")
        if header.chunk_index == self.next_chunk_index:
            self.next_chunk_index += 1
            return AcceptDisposition.ACCEPT
        if header.chunk_index < self.next_chunk_index:
            return AcceptDisposition.DUPLICATE
        raise ProtocolViolation("out-of-order chunk")
```

`GenerateSession` 生成独立 `generation_id`，维护 `next_chunk_index` 与 `active_chunks`，`RealtimeChunkContext` 同时携带四元 identity；Adapter 把 generation、action version、prompt version 写入 Req。

- [ ] **Step 4: Run tests and commit**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_protocol.py python/sglang/multimodal_gen/test/unit/realtime/test_realtime_runtime.py -q`

Expected: PASS.

Commit: `git commit -am "feat(realtime): add generation-aware chunk protocol"`

### Task 2: Strict Multi-User Admission And Session Capacity

**Files:**
- Create: `python/sglang/multimodal_gen/runtime/realtime/admission.py`
- Modify: `python/sglang/multimodal_gen/runtime/realtime/session.py`
- Modify: `python/sglang/multimodal_gen/runtime/managers/gpu_worker.py`
- Modify: `python/sglang/multimodal_gen/runtime/server_args.py`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_realtime_admission.py`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_realtime_runtime.py`

- [ ] **Step 1: Write failing quota, TTL, and no-eviction tests**

```python
@pytest.mark.asyncio
async def test_one_active_session_per_user_and_idempotent_release():
    store = InMemorySessionLeaseStore(max_active_sessions=2, ttl_s=60)
    lease = await store.acquire("u1", "s1", "g1")
    with pytest.raises(AdmissionRejected, match="USER_SESSION_LIMIT"):
        await store.acquire("u1", "s2", "g2")
    await store.release(lease)
    await store.release(lease)

def test_active_realtime_sessions_are_never_lru_evicted():
    cache = RealtimeSessionCache(max_sessions=1)
    cache.attach(_Req(realtime_session_id="s1", realtime_generation_id="g1", block_idx=0))
    with pytest.raises(RealtimeSessionCapacityError):
        cache.attach(_Req(realtime_session_id="s2", realtime_generation_id="g2", block_idx=0))
```

- [ ] **Step 2: Run tests and verify failure**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_realtime_admission.py python/sglang/multimodal_gen/test/unit/realtime/test_realtime_runtime.py -q`

Expected: FAIL because admission and capacity errors do not exist.

- [ ] **Step 3: Implement lease stores and explicit capacity**

```python
class SessionLeaseStore(Protocol):
    async def acquire(self, user_id: str, session_id: str, generation_id: str) -> SessionLease: ...
    async def renew(self, lease: SessionLease) -> None: ...
    async def release(self, lease: SessionLease) -> None: ...

class InMemorySessionLeaseStore:
    async def acquire(self, user_id, session_id, generation_id):
        async with self._condition:
            self._expire_locked(time.monotonic())
            if user_id in self._by_user:
                raise AdmissionRejected("USER_SESSION_LIMIT")
            if len(self._by_user) >= self.max_active_sessions:
                raise AdmissionRejected("CAPACITY_EXHAUSTED", retry_after_s=2)
            lease = SessionLease(user_id, session_id, generation_id, uuid4().hex, time.monotonic() + self.ttl_s)
            self._by_user[user_id] = lease
            return lease
```

增加惰性导入 boto3 的 `DynamoDBSessionLeaseStore`，用 `attribute_not_exists(user_id)` 条件写、TTL 字段、token 条件续约/释放；默认部署不配置 table 时只使用内存实现。`RealtimeSessionCache.attach()` 容量满时拒绝新 block 0，不再调用 LRU 淘汰；GPU Worker 从 `--realtime-max-sessions-per-worker` 读取上限。

- [ ] **Step 4: Run tests and commit**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_realtime_admission.py python/sglang/multimodal_gen/test/unit/realtime/test_realtime_runtime.py -q`

Expected: PASS.

Commit: `git commit -am "feat(realtime): enforce bounded multi-user admission"`

### Task 3: H100 Latent Handoff Boundary

**Files:**
- Create: `python/sglang/multimodal_gen/runtime/pipelines_core/stages/realtime/latent_handoff.py`
- Modify: `python/sglang/multimodal_gen/runtime/pipelines_core/stages/realtime/__init__.py`
- Modify: `python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py`
- Modify: `python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py`
- Modify: `python/sglang/multimodal_gen/runtime/pipelines/minwm_causal_dmd_pipeline.py`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_handoff.py`

- [ ] **Step 1: Write failing handoff tests**

```python
def test_handoff_returns_contiguous_bf16_latents_without_decoding():
    req = make_req(block_idx=3, latents=torch.randn(1, 48, 2, 30, 52, dtype=torch.bfloat16))
    out = RealtimeLatentHandoffStage().forward(req, server_args())
    assert out.realtime_latents.dtype == torch.bfloat16
    assert out.realtime_latents.is_contiguous()
    assert out.realtime_handoff["chunk_index"] == 3

def test_minwm_pipeline_keeps_local_decoder_without_remote_url():
    assert pipeline_stage_names(realtime_vae_worker_url=None)[-1] == "MinWMCausalVaeDecodingStage"
    assert pipeline_stage_names(realtime_vae_worker_url="ws://vae:18081/decode")[-1] == "RealtimeLatentHandoffStage"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_handoff.py -q`

Expected: FAIL because handoff output fields and stage do not exist.

- [ ] **Step 3: Implement feature-flagged handoff**

```python
class RealtimeLatentHandoffStage(PipelineStage):
    @property
    def role_affinity(self):
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        latents = batch.latents.to(dtype=torch.bfloat16).contiguous()
        return OutputBatch(
            realtime_latents=latents,
            realtime_handoff={
                "session_id": batch.realtime_session_id,
                "generation_id": batch.realtime_generation_id,
                "request_id": batch.request_id,
                "chunk_index": batch.block_idx,
                "event_id": batch.realtime_event_id,
                "has_reference": batch.image_latent is not None,
            },
            metrics=batch.metrics,
        )
```

`OutputBatch.drop_payload_for_warmup()` 清理新字段；`process_generation_batch()` 把 `realtime_latents` 视为有效输出但不调用 `save_outputs()`。MinWM DMD/UniPC pipeline 仅在 URL 非空时用 handoff stage 替代本地 decoder。

- [ ] **Step 4: Run tests and commit**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_handoff.py python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py -q`

Expected: PASS.

Commit: `git commit -am "feat(minwm): expose remote VAE latent handoff"`

### Task 4: Stateful TAEHV Worker With Bounded Credits

**Files:**
- Create: `python/sglang/multimodal_gen/runtime/realtime/async_vae_worker.py`
- Create: `python/sglang/multimodal_gen/runtime/entrypoints/realtime_vae_server.py`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_worker.py`

- [ ] **Step 1: Write failing state isolation and queue tests**

```python
@pytest.mark.asyncio
async def test_worker_keeps_decoder_state_per_generation(fake_engine):
    worker = AsyncVAEWorker(fake_engine, max_sessions=2, queue_depth_per_session=1)
    await worker.open(SessionOpen("s1", "g1"))
    await worker.open(SessionOpen("s2", "g2"))
    await worker.decode(chunk("s1", "g1", 0))
    await worker.decode(chunk("s2", "g2", 0))
    assert fake_engine.decoder_ids == {("s1", "g1"), ("s2", "g2")}

@pytest.mark.asyncio
async def test_worker_rejects_second_waiting_latent():
    worker = AsyncVAEWorker(blocking_engine(), max_sessions=1, queue_depth_per_session=1)
    first = asyncio.create_task(worker.decode(chunk("s", "g", 0)))
    await worker.wait_until_decoding()
    waiting = asyncio.create_task(worker.decode(chunk("s", "g", 1)))
    with pytest.raises(VAEBackpressureError):
        await worker.decode(chunk("s", "g", 2))
```

- [ ] **Step 2: Run tests and verify failure**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_worker.py -q`

Expected: FAIL because worker does not exist.

- [ ] **Step 3: Implement model preload, per-Session state, actor serialization, and immediate frame output**

```python
class TAEHVEngine:
    def __init__(self, checkpoint_path: str, device: str, dtype: torch.dtype):
        self.model = TAEHV(checkpoint_path=checkpoint_path).eval().to(device=device, dtype=dtype).requires_grad_(False)

    def create_decoder(self):
        return StreamingTAEHV(self.model).eval()

    def decode(self, decoder, latents, first_chunk):
        if first_chunk:
            decoder.reset()
        source = apply_minwm_latent_stats(latents).permute(0, 2, 1, 3, 4).contiguous()
        frames = []
        frame = decoder.decode(source)
        while frame is not None:
            frames.append(frame)
            frame = decoder.decode()
        return torch.cat(frames, dim=1) if frames else source.new_empty((1, 0, 3, 480, 832))
```

Worker 使用全局 actor lock 串行启动 Decode kernel、每 Session `asyncio.Queue(maxsize=1)`、generation-aware state、10 分钟 max lifetime、60 秒 heartbeat TTL；T2V chunk 0/1 保留并重放首 latent 后移除重复首帧，I2V chunk 0 接受已拼接 reference latent。输出按 1 至 3 帧小批立即编码 WebP/JPEG，不等待逻辑 Chunk 全部编码完成。入口提供 `/health`、`/metrics` 和 `/v1/realtime_vae/decode` WebSocket。

- [ ] **Step 4: Run tests and commit**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_worker.py -q`

Expected: PASS.

Commit: `git commit -am "feat(realtime): add bounded stateful TAEHV worker"`

### Task 5: Gateway Client And True Denoise/Decode Overlap

**Files:**
- Create: `python/sglang/multimodal_gen/runtime/realtime/async_vae_client.py`
- Modify: `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_video_api.py`
- Modify: `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_adapter.py`
- Modify: `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/adapters/sana_wm_realtime_adapter.py`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_pipeline.py`

- [ ] **Step 1: Write failing overlap and order tests**

```python
@pytest.mark.asyncio
async def test_chunk_n_plus_one_denoises_while_chunk_n_decodes():
    timeline = []
    await run_async_loop(
        denoise=fake_stage("denoise", 0.05, timeline),
        decode=fake_stage("decode", 0.10, timeline),
        chunks=3,
    )
    assert starts_before_finish(timeline, "denoise:1", "decode:0")
    assert emitted_chunks(timeline) == [0, 1, 2]
    assert max_decode_backlog(timeline) <= 1

@pytest.mark.asyncio
async def test_cancel_releases_remote_generation_and_local_chunk_tasks():
    session = await start_blocked_session()
    await cancel_session(session)
    assert session.active_chunks == {}
    assert session.vae_client.closed
```

- [ ] **Step 2: Run tests and verify failure**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_pipeline.py -q`

Expected: FAIL because the current loop waits for local decode.

- [ ] **Step 3: Implement bounded two-stage loop**

```python
async def _generate_loop_async_vae(ws, session):
    pending_output = None
    while session.can_schedule_chunk():
        await session.vae_credit.acquire(session.next_chunk_index)
        chunk, batch, denoise_result, timings = await _denoise_next_chunk(session)
        if pending_output is not None:
            await pending_output
        pending_output = asyncio.create_task(
            _decode_send_complete(ws, session, chunk, batch, denoise_result, timings)
        )
    if pending_output is not None:
        await pending_output
```

`RealtimeVAEClient` 建立持久 WebSocket，按 header+binary latent 发送，校验每个 FrameBatch identity，收到 `LatentAccepted` 后发放绑定 next chunk 的单次 credit。Decode task 完成后调用 Adapter 的 chunk-aware `on_chunk_complete`。发生 timeout、stale generation、队列满或连接断开时取消所有本地 task、向 VAE 发送 abort 并释放 Scheduler Session。

- [ ] **Step 4: Run tests and commit**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_pipeline.py python/sglang/multimodal_gen/test/unit/realtime/test_realtime_runtime.py -q`

Expected: PASS.

Commit: `git commit -am "feat(realtime): overlap MinWM denoise and remote VAE"`

### Task 6: Multi-User WebSocket Lifecycle And Action Freshness

**Files:**
- Modify: `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_video_api.py`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/app.js`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_realtime_admission.py`
- Test: `python/sglang/multimodal_gen/apps/realtime_webui/realtime_low_latency_test.js`

- [ ] **Step 1: Write failing lifecycle tests**

```python
@pytest.mark.asyncio
async def test_different_users_run_concurrently_but_same_user_is_rejected():
    controller = RealtimeAdmissionController(InMemorySessionLeaseStore(2, 60))
    one = await controller.admit("u1", "s1", "g1")
    two = await controller.admit("u2", "s2", "g2")
    with pytest.raises(AdmissionRejected):
        await controller.admit("u1", "s3", "g3")
    await controller.release(one)
    await controller.release(two)

def test_held_key_heartbeat_sends_complete_state():
    assert cameraHeartbeatPayload(new Set(["w", "a"])).actions == ["a", "w"]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_realtime_admission.py -q && node python/sglang/multimodal_gen/apps/realtime_webui/realtime_low_latency_test.js`

Expected: FAIL because global single-session gate remains.

- [ ] **Step 3: Replace global gate and add lifecycle watchdog**

Resolve `user_id` from signed auth subject, then `user_id` query/header for test deployments; WebUI stores a random stable browser ID in localStorage and adds it to the WS query. `generate()` waits at most 10 seconds for admission, starts a watchdog that terminates at 60 seconds without client heartbeat/Action/Prompt or 10 minutes total, renews Lease only on valid client activity, and always releases Lease in `finally`. 持续按键每 100 ms 发送完整 key state，GPU dispatch 前 Adapter 对最新状态再快照并记录 superseded version。

- [ ] **Step 4: Run tests and commit**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_realtime_admission.py python/sglang/multimodal_gen/test/unit/realtime/test_realtime_runtime.py -q && for f in python/sglang/multimodal_gen/apps/realtime_webui/*test.js; do node "$f"; done`

Expected: PASS.

Commit: `git commit -am "feat(realtime): enable quota-bound multi-user sessions"`

### Task 7: Trace, Metrics, And A/B Measurement

**Files:**
- Modify: `python/sglang/multimodal_gen/runtime/utils/realtime_trace.py`
- Modify: `python/sglang/multimodal_gen/runtime/realtime/async_vae_client.py`
- Modify: `python/sglang/multimodal_gen/runtime/realtime/async_vae_worker.py`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/app.js`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_trace.py`

- [ ] **Step 1: Write failing trace completeness and redaction tests**

```python
def test_async_chunk_trace_has_correlated_stage_timings_without_media():
    events = build_trace_for_chunk()
    assert required_names <= {event["name"] for event in events}
    assert all(event["generation_id"] == "g1" for event in events)
    serialized = json.dumps(events)
    assert "prompt text" not in serialized
    assert "latent_payload" not in serialized

def test_overlap_ratio_uses_union_not_sum():
    assert calculate_overlap_ratio(denoise=(100, 600), vae=(400, 700)) == pytest.approx(0.4)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_trace.py -q`

Expected: FAIL because remote stages are absent.

- [ ] **Step 3: Add spans and low-cardinality metrics**

Emit `denoiser.queue_wait`, `denoiser.compute`, `latent.serialize`, `latent.transfer`, `vae.queue_wait`, `vae.decode`, `frame.encode`, `frame.transfer`, `gateway.ws_write`, `browser.render`; each carries trace/session/generation/chunk/request IDs in Trace only, worker role/SKU in metric labels, prompt SHA-256 and length but no prompt text. Add histograms for stage latency, action-to-visible, queue depth, backpressure, active/free slots and an overlap gauge. WebUI keeps last complete timing when a partial update arrives.

- [ ] **Step 4: Run tests and commit**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_trace.py python/sglang/multimodal_gen/test/unit/realtime/test_perf_logger.py -q`

Expected: PASS.

Commit: `git commit -am "feat(realtime): trace async VAE critical path"`

### Task 8: Load Generator And Performance Report

**Files:**
- Create: `benchmark/minwm_realtime_async_vae/load_test.py`
- Create: `benchmark/minwm_realtime_async_vae/summarize.py`
- Create: `benchmark/minwm_realtime_async_vae/README.md`
- Test: `benchmark/minwm_realtime_async_vae/test_summarize.py`

- [ ] **Step 1: Write failing percentile and concurrency tests**

```python
def test_report_selects_highest_concurrency_that_meets_slo():
    report = summarize_runs([run(1, p95=700, fps=18), run(2, p95=920, fps=16.5), run(4, p95=1400, fps=12)])
    assert report["max_supported_concurrency"] == 2
    assert report["async_improvement_pct"] == pytest.approx(25.0)
```

- [ ] **Step 2: Run test and verify failure**

Run: `PYTHONPATH=python .venv/bin/python -m pytest benchmark/minwm_realtime_async_vae/test_summarize.py -q`

Expected: FAIL because benchmark code does not exist.

- [ ] **Step 3: Implement reproducible warm and load phases**

`load_test.py` opens 1/2/4/8 concurrent WebSockets, performs two unmeasured warmup chunks, then 60 seconds measured T2V sessions with deterministic action changes. It writes JSONL with hardware inventory, generated/rendered FPS, action-to-visible, chunk and stage timings, failures, queue depth and memory. `summarize.py` reports P50/P95/P99, max concurrency meeting P95 < 1 s and >=16 FPS, overlap ratio, async critical path, sync baseline and `(sync-async)/sync` improvement.

- [ ] **Step 4: Run tests and commit**

Run: `PYTHONPATH=python .venv/bin/python -m pytest benchmark/minwm_realtime_async_vae/test_summarize.py -q`

Expected: PASS.

Commit: `git commit -am "bench(minwm): add async VAE concurrency report"`

### Task 9: Low-Cost AWS Deployment Manifests

**Files:**
- Create: `benchmark/minwm_realtime_async_vae/k8s/namespace.yaml`
- Create: `benchmark/minwm_realtime_async_vae/k8s/h100-denoiser.yaml`
- Create: `benchmark/minwm_realtime_async_vae/k8s/l4-vae.yaml`
- Create: `benchmark/minwm_realtime_async_vae/k8s/l40s-vae.yaml`
- Create: `benchmark/minwm_realtime_async_vae/k8s/gateway-service.yaml`
- Create: `benchmark/minwm_realtime_async_vae/k8s/kustomization.yaml`
- Create: `benchmark/minwm_realtime_async_vae/k8s/validate_manifests.py`
- Test: `benchmark/minwm_realtime_async_vae/k8s/test_manifests.py`

- [ ] **Step 1: Write failing manifest policy tests**

```python
def test_gpu_nodepools_are_spot_only_and_bounded():
    docs = load_all_manifests()
    assert nodepool("minwm-async-denoiser").capacity_types == ["spot"]
    assert nodepool("minwm-async-vae").capacity_types == ["spot"]
    assert nodepool("minwm-async-denoiser").gpu_limit == 8
    assert all(container_has_limits(c) for c in gpu_containers(docs))
```

- [ ] **Step 2: Run test and verify failure**

Run: `PYTHONPATH=python .venv/bin/python -m pytest benchmark/minwm_realtime_async_vae/k8s/test_manifests.py -q`

Expected: FAIL because manifests do not exist.

- [ ] **Step 3: Add parameterized Spot manifests**

H100 NodePool allows only `p5.4xlarge/p5.48xlarge` Spot and requests the exact Denoiser GPU count; initial test uses the smallest fitting shape. VAE NodePool starts with L4 `g6` Spot and a one-GPU Worker; L40S `g6e` is an explicit alternate overlay, never an automatic on-demand fallback. Deployments carry git SHA, model/checkpoint fingerprint, owner/test-run labels, `ttl-after-test`, startup probes, `/metrics`, `terminationGracePeriodSeconds`, queue/session limits and topology affinity. Gateway NLB remains one replica for the benchmark so no DynamoDB write is required.

- [ ] **Step 4: Validate and commit**

Run: `PYTHONPATH=python .venv/bin/python -m pytest benchmark/minwm_realtime_async_vae/k8s/test_manifests.py -q && kubectl kustomize benchmark/minwm_realtime_async_vae/k8s >/tmp/minwm-async-rendered.yaml`

Expected: PASS and valid rendered YAML.

Commit: `git commit -am "deploy(minwm): add H100 and low-cost VAE spot topology"`

### Task 10: Full Local Verification

**Files:**
- Modify only files required by failures found in this task.

- [ ] **Step 1: Run focused Python suite**

Run: `TORCHDYNAMO_DISABLE=1 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/realtime benchmark/minwm_realtime_async_vae -q`

Expected: PASS; GPU-only tests SKIP with an explicit reason.

- [ ] **Step 2: Run WebUI tests**

Run: `for f in python/sglang/multimodal_gen/apps/realtime_webui/*test.js; do node "$f"; done`

Expected: every test exits 0.

- [ ] **Step 3: Run formatting and static validation**

Run: `.venv/bin/python -m compileall -q python/sglang/multimodal_gen/runtime/realtime benchmark/minwm_realtime_async_vae`

Run: `git diff --check`

Expected: both exit 0.

- [ ] **Step 4: Commit verification fixes**

Commit: `git commit -am "test(minwm): verify async VAE multi-user path"`

### Task 11: AWS H100 Spot Deployment, A/B Test, Load Test, And Cleanup

**Files:**
- Create during run: `benchmark/minwm_realtime_async_vae/results/<run-id>/raw.jsonl`
- Create during run: `benchmark/minwm_realtime_async_vae/results/<run-id>/report.json`
- Create during run: `benchmark/minwm_realtime_async_vae/results/<run-id>/report.md`

- [ ] **Step 1: Inspect account, cluster, Spot SKU, and existing model assets read-only**

Run: `AWS_PROFILE=default aws sts get-caller-identity`

Run: `AWS_PROFILE=default aws eks list-clusters --region us-east-2`

Run: `AWS_PROFILE=default aws ec2 describe-instance-type-offerings --region us-east-2 --location-type availability-zone --filters Name=instance-type,Values=p5.4xlarge,p5.48xlarge,g6.2xlarge,g6e.2xlarge`

Expected: exact target account, cluster and candidate Spot types are recorded in the run report. If H100 Spot remains Pending, poll Pod/NodePool events every 60 seconds without falling back to on-demand.

- [ ] **Step 2: Push the tested commit and deploy the async topology**

Run: `git push -u origin codex/minwm-async-vae-multiuser`

Run: `kubectl apply -k benchmark/minwm_realtime_async_vae/k8s`

Expected: one H100 Spot Denoiser and one L4 Spot VAE become Ready. If L4 cannot meet decode headroom, replace only the VAE overlay with L40S Spot.

- [ ] **Step 3: Execute correctness and isolation smoke tests**

Run: `PYTHONPATH=python .venv/bin/python benchmark/minwm_realtime_async_vae/load_test.py --url "$WEBUI_WS_URL" --concurrency 2 --duration 60 --warmup-chunks 2 --assert-order --assert-isolation --output "$RUN_DIR/smoke.jsonl"`

Expected: T2V/I2V complete, Action/Prompt versions are monotonic, no cross-Session frame/state leakage, no duplicate/gap/out-of-order Chunk, and Trace/Dump still work.

- [ ] **Step 4: Measure synchronous baseline after warmup**

Patch only the deployment flag to remove `--realtime-vae-worker-url`, restart the H100 service, wait for two warmup chunks, then run concurrency 1/2/4/8 for 60 seconds each. Do not compare cold-start samples.

Run: `PYTHONPATH=python .venv/bin/python benchmark/minwm_realtime_async_vae/load_test.py --url "$SYNC_WS_URL" --concurrency 1,2,4,8 --duration 60 --warmup-chunks 2 --output "$RUN_DIR/sync.jsonl"`

Expected: raw synchronous stage timings are saved.

- [ ] **Step 5: Measure async L4/L40S path and determine maximum supported concurrency**

Restore the remote URL, warm two chunks, run the same seed/profile at concurrency 1/2/4/8, and stop increasing after the first level that fails P95 action-to-visible <1 second or generated FPS >=16.

Run: `PYTHONPATH=python .venv/bin/python benchmark/minwm_realtime_async_vae/load_test.py --url "$ASYNC_WS_URL" --concurrency 1,2,4,8 --duration 60 --warmup-chunks 2 --output "$RUN_DIR/async.jsonl"`

Run: `PYTHONPATH=python .venv/bin/python benchmark/minwm_realtime_async_vae/summarize.py --sync "$RUN_DIR/sync.jsonl" --async "$RUN_DIR/async.jsonl" --output "$RUN_DIR"`

Expected: report includes exact GPU SKU/count, maximum passing concurrency, P50/P95/P99 E2E and every critical stage, generated/render FPS, overlap, error rate, GPU/host memory and async improvement.

- [ ] **Step 6: Release resources immediately and prove cleanup**

Run: `kubectl delete -k benchmark/minwm_realtime_async_vae/k8s --wait=true --timeout=20m`

Run: `kubectl get pods,nodes,nodeclaims -l seedleap.ai/test-run="$RUN_ID" -o wide`

Run: `AWS_PROFILE=default aws ec2 describe-instances --region us-east-2 --filters Name=tag:seedleap.ai/test-run,Values="$RUN_ID" Name=instance-state-name,Values=pending,running,stopping,stopped`

Expected: no test Pod, NodeClaim or EC2 instance remains. Record cleanup timestamp and any retained non-billable code/config artifacts in `report.md`.

- [ ] **Step 7: Commit reports and final verification**

Run: `git add benchmark/minwm_realtime_async_vae/results && git commit -m "bench(minwm): report async VAE H100 spot results"`

Run: `git status --short && git log -1 --oneline`

Expected: worktree clean and final report commit present.

