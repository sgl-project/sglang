# M4 GPU E2E Results

## Run summary

- **Host**: `kangyan-zhou-h200-4gpu-m4-e2e.devbox.rdxa` — 4x NVIDIA H200 (cluster `h200-sci-k8s`, node `gpu1-10-220-51-47`)
- **CUDA**: 13.0, driver 580.105.08
- **Image**: `lmsysorg/sglang:latest` (Ubuntu 24.04, Python 3.12, SGLang editable at `/sgl-workspace/sglang/python`)
- **Date**: 2026-05-18
- **Router commit**: `f7474dfe0` on `sgl-router/m4-cache-aware-pd`
- **Devbox wall time**: ~70 min (acquire 14:58 → release 16:08)

## Patch contract verification

Patch 1 — SGLang `/server_info` exposes resolved `kv_events` block — verified
end-to-end against `Qwen/Qwen3-0.6B`:

```json
$ curl http://127.0.0.1:30001/get_server_info | jq .kv_events
{
  "publisher": "zmq",
  "endpoint_host": "*",
  "endpoint_port_base": 5557,
  "topic": "kv",
  "block_size": 1,
  "dp_size": 1
}
```

Patches 2 + 3 are exercised indirectly via the convergence test (see below).

## Test results

| Test | Result | Evidence |
|---|---|---|
| `test_cache_aware_zmq_converges_to_one_worker` | **PASS** | 10/10 same-prefix requests dispatched to a single worker; `sgl_router_requests_total{worker_url="http://127.0.0.1:41939",...}` = 10. Test threshold is >=8/10. |
| `test_pd_mode_response_has_decode_affinity_header` | **FAIL (M4 follow-up — bootstrap injection)** | `400 Bad Request` from prefill worker: `"Disaggregated request received without bootstrap room id"`. Router does not inject `bootstrap_room` / `bootstrap_host` / `bootstrap_port` on PD-mode requests and does not dual-dispatch to the decode worker. This is an M4 gap, not an M5 deferral — Task 6 of the M4 plan requires "E2E in PD mode … bonus tokens decoded correctly". Earlier in-progress framing on this branch (now corrected) had labelled this "deferred to M5" off of an agent-authored code comment that did not match the M4 plan. Fix lands in a follow-up commit on this branch: plumb `bootstrap_port` through `WorkerSpec`, inject the three fields into the JSON body, fan to both prefill and decode via `tokio::join!`. |
| `test_no_prefill_workers_available_returns_503` | **FAIL (M4 follow-up — same as above)** | `httpx.ReadTimeout` after the test froze the prefill worker. Same root cause: every PD request hits the bootstrap-room-id 400 before the breaker can observe enough timeouts. Resolved by the bootstrap-injection + dual-dispatch follow-up commit. |
| `test_stale_request_expired_returns_504` | **FAIL** | Got 502 `upstream_timeout` at 60s instead of 504 `stale_request_expired` at ~5s. Root cause: `experimental/sgl-router/src/main.rs:94` hard-codes `ActiveLoadRegistry::with_defaults()` which uses `stale_request_timeout = 5 minutes`. The test expects a `[active_load] stale_request_timeout_secs = 5` config knob; that config plumbing is not wired in M4. The router's own upstream-read timeout (~60s) fires first, returning 502 `upstream_timeout` from the proxy layer. |

Single-test rerun of the convergence test in isolation also passed:
`m4_acceptance/test_cache_aware_zmq_convergence.py::test_cache_aware_zmq_converges_to_one_worker PASSED in 78.44s`.

## Per-worker request distribution (convergence test)

```
sgl_router_requests_total{worker_url="http://127.0.0.1:41939",model_id="Qwen/Qwen3-0.6B",mode="plain",outcome="success"} 10
```

Worker `41939` received all 10 requests; worker `36451` received 0. Perfect convergence under the cache_aware_zmq policy with a shared 200-token prefix. The 8/10 threshold accounts for the timing race between request 1's response and request 2's selection — observed run had no stragglers.

## Test-infra fixes applied to make the suite runnable

Three small fixes in `experimental/sgl-router/e2e/infra/` while running the suite. These are test-infra-only changes; no router source touched.

1. **`gateway.py` `_get_open_port` / `model_pool.py` `_get_open_port`** — cap allocated ephemeral ports to `[20000, 55535]`. SGLang derives its internal gRPC port as `http_port + 10000`; the kernel's default ephemeral range goes to 60999, so ~30% of picks were overflowing 65535 and SGLang rejected the worker startup with `ValueError: SGLANG_GRPC_PORT (NNNNN) must be between 1 and 65535`.
2. **`gateway.py` `_resolve_tokenizer_path`** — when the e2e test passes a HuggingFace repo id like `Qwen/Qwen3-0.6B` as `tokenizer_path`, resolve it via `huggingface_hub.try_to_load_from_cache(..., "tokenizer.json")` before writing the TOML. sgl-router's tokenizer loader inspects the file extension and rejected `.6B` as an unsupported tokenizer file type.
3. **`gateway.py` worker TOML enum case** — the router's worker-mode enum is lowercase (`plain` / `prefill` / `decode`), the test infra was writing `Plain` / `Prefill` / `Decode`.

Also added a small `m4_acceptance/test_cache_aware_zmq_convergence.py` hook to persist the `/metrics` snapshot to `/tmp/m4-metrics.txt` (overridable via `SGL_E2E_METRICS_DUMP_PATH`) for artifact capture during CI runs.

## Outstanding wiring gaps surfaced by this run

The Phase D brief flagged that several metrics are registered but unwired at call sites. This run confirms that, plus two larger wiring gaps:

| Gap | Where | Impact |
|---|---|---|
| `ActiveLoadRegistry` `stale_request_timeout` not configurable | `main.rs:94` — `with_defaults()` (5-minute hard-coded timeout) | `test_stale_request_expired` cannot exercise the janitor within a test-reasonable time. Fix: thread a `[active_load] stale_request_timeout_secs` config key from `RouterConfig` into `main.rs`, replacing the `with_defaults()` call. |
| PD-mode bootstrap injection + dual dispatch not wired | `src/server/routes/chat.rs` (M4 follow-up — corrected scope; the previous code comment that called this "deferred to M5" was agent drift, not the M4 plan). | Every PD-disagg request fails at the prefill worker with `BadRequestError: Disaggregated request received without bootstrap room id`. Blocks the two PD acceptance tests (`test_pd_mode_response_has_decode_affinity_header`, `test_no_prefill_workers_available_returns_503`) and any real PD traffic. Fix in M4 follow-up: plumb `bootstrap_port` through `WorkerSpec`, inject the three bootstrap fields into the JSON body, dual-dispatch to prefill + decode via `tokio::join!`. |
| `sgl_router_overlap_blocks` histogram unwired | Policy reads radix tree directly; observation hook missing in `policies/cache_aware_zmq.rs` selection path | No buckets / sum / count emitted on `/metrics`. Convergence still works; observability missing. |
| `sgl_router_active_load` gauge unwired | No `set_active_load` calls from `ActiveLoadRegistry` mutations | Gauge family present but no labels. |
| `sgl_router_stale_requests_total` counter unwired | Janitor fires internally but does not call `record_stale_request` | Counter family present but no labels. |
| `sgl_router_decode_affinity_total` counter unwired | `select_decode_with_affinity` does not call `record_decode_affinity` | Counter family present but no labels. |

The convergence test still passes despite the missing histogram/gauge counters because it only depends on `sgl_router_requests_total` (which IS wired at the chat-handler success path).

## Devbox lifecycle confirmation

- Acquired: `rx devbox acquire --gpu h200 --count 4 --name kangyan-zhou-h200-4gpu-m4-e2e --ttl 3h` at ~14:58 UTC.
- Released: `rx devbox release kangyan-zhou-h200-4gpu-m4-e2e` at ~16:08 UTC. Confirmed via `rx devbox list` — only the pre-existing `kangyan-zhou-h200-1gpu` remains.
