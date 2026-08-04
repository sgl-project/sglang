# MinWM Runtime Transport And Latency Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepackage MinWM runtime dependencies, remove trace data from the video WebSocket, and reduce browser display lag with a bounded low-latency playback profile.

**Architecture:** The API logs every trace event and mirrors a bounded recent window into an in-memory store exposed through HTTP. The WebUI batches client trace events over HTTP and polls server events only while the Trace tab is open. A single immutable ECR runtime image supplies both GPU roles, while external model artifacts remain on S3 volumes.

**Tech Stack:** Python, FastAPI, JavaScript, aiohttp, Kubernetes, Docker BuildKit, ECR, pytest, Node.js tests, Playwright.

---

### Task 1: Bounded HTTP Trace Store

**Files:**
- Create: `python/sglang/multimodal_gen/runtime/utils/realtime_trace_store.py`
- Modify: `python/sglang/multimodal_gen/runtime/utils/realtime_trace.py`
- Modify: `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_video_api.py`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_realtime_trace_store.py`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_realtime_runtime.py`

- [ ] Write failing tests for bounded retention, cursor pagination, client-event ingestion, and absence of the WebSocket trace sender.
- [ ] Run the focused pytest targets and verify failures describe missing HTTP trace behavior.
- [ ] Implement the thread-safe TTL store and FastAPI GET/POST endpoints.
- [ ] Remove trace queue installation and sender tasks from `generate()` while retaining structured server logs.
- [ ] Run the focused pytest targets and verify they pass.

### Task 2: WebUI Trace HTTP Client

**Files:**
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/app.js`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/realtime_trace_dump_integration_test.js`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/server.py`
- Test: `python/sglang/multimodal_gen/apps/realtime_webui/realtime_trace_http_test.js`

- [ ] Write failing static and behavioral tests for batched HTTP client events and tab-scoped polling.
- [ ] Verify the tests fail because trace still uses the WebSocket.
- [ ] Add HTTP URL conversion, bounded batching, incremental cursors, and start/stop polling hooks.
- [ ] Keep legacy incoming trace-message parsing only for backward compatibility.
- [ ] Run all WebUI Node tests.

### Task 3: Low-Latency Display Profile

**Files:**
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/app.js`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/playback_controller_test.js`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/realtime_low_latency_defaults_test.js`

- [ ] Write failing tests for first-frame startup, an 80-180 ms target, bounded jitter boost, and immediate stale-action cutover.
- [ ] Verify the tests fail with the old 220-420 ms smoothing profile.
- [ ] Enable the low-latency controller profile and update deployment-configurable defaults.
- [ ] Run the playback and static default tests.

### Task 4: Prebuilt Runtime Image

**Files:**
- Create: `benchmark/minwm_realtime_async_vae/docker/Dockerfile`
- Create: `benchmark/minwm_realtime_async_vae/docker/build_and_push.sh`
- Modify: `benchmark/minwm_realtime_async_vae/k8s/h100-denoiser.yaml`
- Modify: `benchmark/minwm_realtime_async_vae/k8s/l4-vae.yaml`
- Modify: `benchmark/minwm_realtime_async_vae/k8s/README.md`
- Modify: `benchmark/minwm_realtime_async_vae/k8s/test_manifests.py`

- [ ] Write failing manifest tests that reject runtime Git, pip, curl, and GitHub token usage.
- [ ] Verify the policy tests fail against the current manifests.
- [ ] Add the layered BuildKit Dockerfile and immutable ECR build helper.
- [ ] Replace runtime installation commands with direct packaged entrypoints.
- [ ] Run manifest rendering and policy tests.

### Task 5: Local And End-To-End Verification

**Files:**
- Create: `benchmark/minwm_realtime_async_vae/results/<run-id>/report.zh-CN.md`
- Modify: `benchmark/minwm_realtime_async_vae/README.md`

- [ ] Run focused Python and Node test suites and record exact pass counts.
- [ ] Build and push the immutable image to ECR.
- [ ] Deploy the disposable H100 plus L4 Spot topology.
- [ ] Measure cold start, warm generation latency, trace HTTP behavior, WebSocket message types, and display-lag percentiles with browser automation.
- [ ] Delete the Deployment, Service, NodePools, nodes, and load balancer, then verify no benchmark-labeled resources remain.
- [ ] Write the comparison report with measured results and residual limitations.
