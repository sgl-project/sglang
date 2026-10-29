# SGLang Endpoints and Signals

Use this reference when checking a live server.

## Auth

Most read endpoints are public unless the server is protected by `api_key` or
`admin_api_key`.

Use:

```bash
curl -H "Authorization: Bearer <token>" ...
```

Rules:

- normal protected endpoints require `api_key`
- admin endpoints require `admin_api_key`
- some HiCache endpoints fail if `admin_api_key` is not configured at all
- `/health` and metrics-style health checks are usually still exposed

## Core Endpoints

### `/health`

Cheap liveness check.

- `200`: process is alive enough to answer health
- `503`: starting, shutting down, or unhealthy

`/health` alone is not enough for latency or hang diagnosis.

### `/health_generate`

Active health check.

- exercises a real generate or embedding path
- catches stuck schedulers or broken worker paths that `/health` can miss

Use this when requests time out but `/health` is still green.

### `/model_info`

Use for model identity:

- `model_path`
- `tokenizer_path`
- `is_generation`
- `weight_version`
- multimodal flags
- model type or architectures

This is the first check for wrong-output or wrong-weight problems.

### `/server_info`

Use for runtime shape:

- serialized `server_args`
- scheduler info
- per-DP `internal_states`
- SGLang version

This is usually the single best live snapshot.

## Load And Capacity

### `/v1/loads?include=all`

Best structured load endpoint for a first pass.

Useful fields:

- `num_running_reqs`
- `num_waiting_reqs`
- `num_total_tokens`
- `num_used_tokens`
- `token_usage`
- `gen_throughput`
- `cache_hit_rate`
- `memory`
- `speculative`
- `disaggregation`
- `queues`

Useful queries:

```bash
curl -s http://127.0.0.1:30000/v1/loads
curl -s "http://127.0.0.1:30000/v1/loads?include=all"
curl -s "http://127.0.0.1:30000/v1/loads?include=core,queues,disagg"
curl -s "http://127.0.0.1:30000/v1/loads?format=prometheus"
```

What to look for:

- high `num_waiting_reqs` with low compute throughput usually means queueing or capacity pressure
- `token_usage` near `1.0` usually means KV or token-capacity pressure
- low `cache_hit_rate` after a deploy can explain TTFT regressions
- PD queue fields often explain transfer or prealloc bottlenecks hidden by plain queue size

### `/metrics`

Prometheus endpoint. Use it when you need trends rather than one live snapshot.

High-value metrics:

- `sglang:time_to_first_token_seconds`
- `sglang:time_per_output_token_seconds`
- `sglang:e2e_request_latency_seconds`
- `sglang:num_running_reqs`
- `sglang:num_queue_reqs`
- `sglang:num_used_tokens`
- `sglang:cache_hit_rate`
- `sglang:gen_throughput`
- `sglang:token_usage`

## Request Capture

### `/configure_logging`

Used by `python -m sglang.srt.managers.configure_logging`.

Main use:

- enable request logging
- set request logging level
- enable request dump folder
- set request dump threshold

Typical payload:

```json
{
  "log_requests": true,
  "log_requests_level": 3,
  "dump_requests_folder": "/tmp/sglang_request_dump",
  "dump_requests_threshold": 100
}
```

Use this when the problem is ongoing and you need the next failing request
without restarting the service.

## HiCache

### `GET /hicache/storage-backend`

Returns tokenizer-side HiCache storage status:

- `hicache_storage_backend`
- `hicache_storage_backend_extra_config`
- `hicache_storage_prefetch_policy`
- `hicache_write_policy`

Use this when long-context or PD problems may involve storage-backed KV reuse.

### `PUT /hicache/storage-backend`
### `DELETE /hicache/storage-backend`

Runtime attach or detach. These are operational actions, not passive checks.

## Profiling And Tracing Controls

### `/start_profile`
### `/stop_profile`

Use only after the problem is already narrowed down.

### `/set_trace_level?level=N`

Changes trace verbosity when tracing was enabled at startup.

Levels:

- `0`: disabled
- `1`: important slices
- `2`: all slices except nested ones
- `3`: all slices

## Quick Reads By Problem Type

### TTFT spike

Read:

- `/server_info`
- `/v1/loads?include=all`
- `/metrics`

Compare:

- queue size
- token usage
- cache hit rate
- PD disaggregation queues

### Hang or timeout

Read:

- `/health`
- `/health_generate`
- `/server_info`
- `/v1/loads?include=all`

If tracing is already enabled, look at trace data before heavier profiling.

### Wrong model behavior

Read:

- `/model_info`
- `/server_info`
- exact request payload and parser or template config

Do not jump to kernel profiling until config drift is ruled out.
