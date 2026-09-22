# Static multi-model gateway

This example starts isolated SGLang workers and registers them with one
Inference Gateway (IGW). Replace the model paths and GPU IDs in
`static_multi_model.json` before running it on a single NVIDIA CUDA host.

```bash
python -m sglang_router.launch_server \
  --multi-model-config examples/multi_model_router/static_multi_model.json
```

Every item in a model's `gpu_groups` starts one replica. A replica's group must
contain exactly `tp_size * pp_size` GPU IDs. GPU IDs cannot be reused anywhere
in the file by default. For deliberately oversubscribed development hosts, set
top-level `allow_gpu_sharing` to `true`; each replica remains a separate CUDA
process, so the operator must keep the combined model and KV-cache allocations
within device memory.

After the supervisor reports readiness, send normal OpenAI-compatible requests
to the router, selecting the model ID configured in the JSON file:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-chat","messages":[{"role":"user","content":"Hello"}]}'
```

This MVP supports static, single-node standard DP replicas only. It does not
perform cross-model batching or KV-cache reuse, and it rejects PD, DPA, and
multi-node model specifications rather than launching an unsafe topology.

## Resource-aware virtual models

`resource_aware_multi_model.json` adds a public Router-edge ModelResolver in
front of the Rust Router. The regular `router.port` is its private backend
port; `model_resolver.port` is the public OpenAI-compatible endpoint. The two
ports must differ.

```bash
python -m sglang_router.launch_server \
  --multi-model-config examples/multi_model_router/resource_aware_multi_model.json
```

Clients can then send a request to a virtual model such as `general-chat`:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"general-chat","max_tokens":128,"messages":[{"role":"user","content":"Hello"}]}'
```

For every new virtual-model request, the resolver polls each direct Runtime's
`/v1/loads?include=core,memory,queues` endpoint. It excludes unhealthy or
stale states, Runtime queues above the profile limit, KV usage above the
profile limit, and Runtimes without enough declared free token capacity. It
then selects the lowest-scoring candidate using KV pressure, token occupancy,
queue depth, running-request count, and scheduler utilization. The Rust
Router receives the selected concrete `model_id` and continues to choose among
that model's replicas using its normal worker policy.

The public endpoint exposes OpenAI-compatible `GET /v1/models`, which lists
both virtual routing profiles and every concrete Runtime model. `GET /health`
is a liveness check only. `GET /ready` returns 200 only when every virtual
profile has at least one healthy, fresh candidate Runtime, otherwise it returns
503. `GET /health/verbose` and `GET /v1/runtime-loads` expose the normalized
per-Runtime states used by the resolver. Successful routed responses include
`X-SGLang-Requested-Model` and `X-SGLang-Resolved-Model` headers. An explicit
concrete request such as `model=qwen-int4-tp4` bypasses ModelResolver selection
entirely. If no compatible candidate has fresh capacity, the endpoint returns
HTTP 503 rather than silently switching an explicit model or overcommitting KV
cache.

This first version uses Runtime-owned scheduler/KV metrics, which remain
accurate even when two test processes share a GPU. A host-level NVML agent can
be added later for physical GPU utilization and free-memory telemetry; it is
not required for the resolver's admission decisions today.
