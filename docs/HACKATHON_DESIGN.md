## Groq Hackathon: Inference Platform (Hathora + SGLang) - Minimal Design

### Goals
- Deploy HF models to H100-only workers with SGLang runtime.
- User config: HF token, model id, precision/quantization, TP size, autoscaling hints.
- Provide logs and Prometheus metrics; surface inference URL for cURL and a mock Groq UI.
- Launch via Hathora apps; document the reference implementation end-to-end.

### Components
1) Serving Container (SGLang runtime)
   - Entrypoint: `entrypoint.sh` (Hathora compatible)
   - App: `serve_hathora.py` (FastAPI)
   - Config: `hathora_config.py` (Pydantic)
   - Exposed routes:
     - `POST /v1/chat/completions` (OpenAI-compatible)
     - `GET /health`
     - `GET /metrics` (Prometheus, if enabled)
     - `GET /logs` (tail server logs)

2) Backend Orchestrator (to be added)
   - API surface:
     - `POST /deploy` { hf_token, model_id, tp_size, precision, autoscale hints }
     - `GET /status/:id`
     - `GET /logs/:id` (proxy)
     - `GET /metrics/:id` (proxy)
   - Calls Hathora API to launch/scale workers and returns inference URL(s).

3) Mock Groq UI (to be added)
   - Form to collect HF token, model id, TP, precision, autoscale hints.
   - Button to deploy; shows URL and readiness; embeds logs and metrics panels.

4) Autoscaling Adapter (to be added)
   - Scrapes `/metrics` and DCGM exporter.
   - Simple rules: scale up when queue depth high or p95 TTFB high; scale down on low util.

### Runtime Flow
1) User submits config from UI → Backend → Hathora launches a pod with our image.
2) `entrypoint.sh` verifies H100 GPUs, optionally sets TP automatically, registers NEG.
3) `serve_hathora.py` parses `DEPLOYMENT_CONFIG_JSON` or envs, enforces H100-only if requested, defaults to FP8 on H100, starts SGLang `Engine` with mapped ServerArgs, enables `/metrics`.
4) User obtains inference URL and uses cURL or Groq-like UI to test.

### Configuration Mapping (User → SGLang)
- `hf_token` → env `HF_TOKEN` for HF API
- `model_id` → `ServerArgs.model_path`
- `revision` → `ServerArgs.revision`
- `dtype` → `ServerArgs.dtype`
- `quantization` → `ServerArgs.quantization`
- `kv_cache_dtype` → `ServerArgs.kv_cache_dtype`
- `tp_size` → `ServerArgs.tp_size`
- `max_total_tokens` → `ServerArgs.max_total_tokens`
- `mem_fraction_static` → `ServerArgs.mem_fraction_static`
- `schedule_conservativeness` → `ServerArgs.schedule_conservativeness`
- `max_queued_requests` → `ServerArgs.max_queued_requests`
- `enable_metrics` → `ServerArgs.enable_metrics` and FastAPI middleware
- `enable_p2p_check`, `enable_torch_compile` → corresponding flags
- `h100_only` → startup guard via `nvidia-smi` name match
- `auto_use_fp8_on_h100` → defaults: `quantization=fp8`, `kv_cache_dtype=fp8_e5m2` if not set

### Observability
- Prometheus metrics via `/metrics` from SGLang multiprocess exporter.
- Logs streaming via `/logs` (simple tail approach).
- Recommended: add DCGM exporter sidecar for GPU metrics.

### Interfaces
- Inference: `POST /v1/chat/completions` compatible with OpenAI Chat API; supports stream and non-stream.
- Health: `GET /health` returns readiness and model info.
- Metrics: `GET /metrics` exposes SGLang counters/gauges (token usage, queue depth, throughput, TTFB histogram).
- Logs: `GET /logs` streams server logs for debugging.

### Security
- `hf_token` is only used for HF downloads; set via env or JSON at deploy time.
- Optionally add API key middleware at the FastAPI layer if public exposure is required.

### Integration Plan
- Backend orchestrator calls Hathora to launch container with `DEPLOYMENT_CONFIG_JSON`.
- UI posts to backend; backend returns deployment id and eventual inference URL.
- Observability endpoints proxied via backend to avoid direct worker exposure.

### Minimal Test Plan
1) Config loader
   - Provide `DEPLOYMENT_CONFIG_JSON` and env fallbacks; verify parsed `DeploymentConfig` and defaulting.
2) Health endpoint
   - Start app with a tiny model or mock engine; assert `/health` OK and fields populated.
3) Chat completions
   - Mock `sglang.Engine.async_generate` to return deterministic text; test non-stream and stream paths.
4) Metrics exposure
   - Enable metrics; assert `/metrics` contains known SGLang metric names.
5) H100 guard
   - Mock `nvidia-smi` call; ensure process exits when non-H100 and `h100_only=true`.

### Out of Scope for Monday (stretch)
- Multi-node TP over multiple hosts, container upload (P1), fine-grained auth.


