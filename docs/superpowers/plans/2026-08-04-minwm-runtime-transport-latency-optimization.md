# MinWM Production Runtime, Trace, And Display Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved production chain with independent Gateway, Coordinator, H100 Denoiser, low-cost VAE, off-WebSocket Trace, prebuilt images, and low-latency playback.

**Architecture:** Gateway owns public sessions and VAE frame egress; Coordinator owns durable admission and worker-slot leases; Denoiser and VAE communicate over a bounded latent/control stream while VAE sends frames directly to Gateway. ADOT exports full traces to CloudWatch and a cached HTTP query plane serves the Trace UI.

**Tech Stack:** Python, FastAPI/aiohttp, msgpack, DynamoDB, OpenTelemetry/ADOT, CloudWatch Logs Insights, Docker BuildKit, ECR, Kubernetes, Karpenter, JavaScript, Playwright.

---

### Task 1: Production Coordinator Control Plane

**Files:**
- Create: `python/sglang/multimodal_gen/runtime/realtime/coordinator.py`
- Create: `python/sglang/multimodal_gen/runtime/entrypoints/realtime_coordinator_server.py`
- Create: `python/sglang/multimodal_gen/test/unit/realtime/test_realtime_coordinator.py`

- [x] Write failing tests for atomic user/worker-slot admission, compatibility filtering, renew, release, TTL fencing, and partial-reservation rollback.
- [x] Run focused tests and verify the production APIs are missing.
- [x] Implement in-memory and DynamoDB backends behind one Coordinator interface.
- [x] Implement heartbeat and Session lifecycle HTTP endpoints with bounded retries and structured errors.
- [x] Run focused tests and concurrency tests.

### Task 2: Gateway And Direct VAE Frame Egress

**Files:**
- Create: `python/sglang/multimodal_gen/runtime/realtime/gateway.py`
- Create: `python/sglang/multimodal_gen/runtime/entrypoints/realtime_gateway_server.py`
- Modify: `python/sglang/multimodal_gen/runtime/realtime/async_vae_protocol.py`
- Modify: `python/sglang/multimodal_gen/runtime/realtime/async_vae_client.py`
- Modify: `python/sglang/multimodal_gen/runtime/entrypoints/realtime_vae_server.py`
- Modify: `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/generate_session.py`
- Modify: `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_video_api.py`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_realtime_gateway.py`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_async_vae_pipeline.py`

- [x] Write failing tests for Coordinator-based routing, signed Session identity, direct VAE-to-Gateway frames, bounded egress queues, and cleanup.
- [x] Implement Gateway session proxy and per-Pod direct output endpoint.
- [x] Extend VAE SessionOpen and Denoiser runtime for per-Session worker/output routes.
- [x] Ensure Denoiser receives only credit/completion while Gateway receives FrameBatch.
- [x] Run protocol, Gateway, and async VAE tests.

### Task 3: Production Trace Data And Query Planes

**Files:**
- Create: `python/sglang/multimodal_gen/runtime/realtime/trace_query.py`
- Modify: `python/sglang/multimodal_gen/runtime/entrypoints/realtime_gateway_server.py`
- Modify: `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_video_api.py`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/trace_transport.js`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/app.js`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/index.html`
- Test: `python/sglang/multimodal_gen/test/unit/realtime/test_realtime_trace_query.py`
- Test: `python/sglang/multimodal_gen/apps/realtime_webui/realtime_trace_http_test.js`

- [x] Write failing tests that reject Trace on the video WebSocket and cover CloudWatch query caching, client metric batching, tab-scoped polling, and last-value retention.
- [x] Remove server Trace sender tasks and client Trace WebSocket writes.
- [x] Implement CloudWatch Logs Insights query service with 15-second bounded cache, request coalescing, concurrency limits, and IAM-safe errors.
- [x] Implement browser HTTP batching/polling and retain legacy message parsing only for compatibility.
- [x] Run Python and Node Trace tests.

### Task 4: Low-Latency Playback Profile

**Files:**
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/app.js`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/playback_controller.js`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/playback_controller_test.js`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/realtime_low_latency_defaults_test.js`

- [x] Write failing tests for first-frame start, 80–180ms target lead, bounded jitter boost, and immediate Action cutover.
- [x] Enable the low-latency profile and one-frame encoded batches.
- [x] Run all playback tests and browser timing tests.

### Task 5: Role Images And Production Kubernetes Topology

**Files:**
- Create: `benchmark/minwm_realtime_async_vae/docker/Dockerfile`
- Create: `benchmark/minwm_realtime_async_vae/docker/build_and_push.sh`
- Create: `benchmark/minwm_realtime_async_vae/k8s/gateway.yaml`
- Create: `benchmark/minwm_realtime_async_vae/k8s/coordinator.yaml`
- Create: `benchmark/minwm_realtime_async_vae/k8s/observability.yaml`
- Modify: `benchmark/minwm_realtime_async_vae/k8s/h100-denoiser.yaml`
- Modify: `benchmark/minwm_realtime_async_vae/k8s/l4-vae.yaml`
- Modify: `benchmark/minwm_realtime_async_vae/k8s/gateway-service.yaml`
- Modify: `benchmark/minwm_realtime_async_vae/k8s/kustomization.yaml`
- Modify: `benchmark/minwm_realtime_async_vae/k8s/test_manifests.py`

- [x] Write failing policy tests for role separation, CPU control plane replicas, digest images, Spot GPU pools, OTLP wiring, no runtime installs, and NLB selecting only Gateway.
- [x] Build four role targets and an immutable ECR push helper.
- [x] Add Gateway, Coordinator, ADOT, Worker Services, autoscaling, probes, PDBs, NetworkPolicy, bounded resources, and scheduled GPU scale-to-zero.
- [x] Add declarative AWS control plane, minimum-privilege IRSA, versioned model publisher, and explicit cleanup tooling.
- [x] Render and validate the complete topology locally.

### Task 6: End-To-End Production-Chain Verification

**Files:**
- Create: `benchmark/minwm_realtime_async_vae/e2e_production_chain.py`
- Create: `benchmark/minwm_realtime_async_vae/results/<run-id>/report.zh-CN.md`

- [ ] Run complete local Python, Node, manifest, infrastructure, browser-probe, and image smoke tests.
- [ ] Push the branch and build three immutable ECR images.
- [ ] Prepare the exact DynamoDB table operation and obtain the required immediate human confirmation before executing it.
- [ ] Deploy two Gateway Pods, two Coordinator Pods, one H100 Spot Denoiser, and one L4/L40S Spot VAE.
- [ ] Run single-session, four-concurrent-session, Action/Prompt, Trace query, disconnect, and Worker-failure tests from the NLB endpoint.
- [ ] Capture cold start, stage timings, media route evidence, Trace transport evidence, and display-lag P50/P95.
- [ ] Delete all disposable GPU, NLB, and temporary observability resources; verify zero labeled resources remain.
- [ ] Publish the measured production-chain report and residual risks.
