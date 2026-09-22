# Multi-model runtime MVP plan

## Outcome

Provide one local command that starts several isolated SGLang model replicas and
exposes them through one SGLang Inference Gateway (IGW) endpoint. A request's
OpenAI `model` value selects a model-specific pool; repeated GPU groups for a
model are that model's standard DP replicas.

```text
request(model=qwen-chat) ─┐
                          ├─ IGW ModelRegistry(qwen-chat) ── Qwen replica 0/1
request(model=glm-reason) ┘
                             IGW ModelRegistry(glm-reason) ─ GLM replica 0/1
```

## MVP contract

- The entry point is `python -m sglang_router.launch_server --multi-model-config CONFIG.json`.
- `CONFIG.json` declares router settings and a list of models. Each `gpu_groups`
  item produces one runtime process tree with `dp_size=1`.
- The supervisor assigns a unique `CUDA_VISIBLE_DEVICES` slice to every replica,
  starts the router first, waits for readiness, starts workers, then registers
  each worker with IGW using its configured `model_id`.
- Every model uses an independent process tree, KV cache, scheduler, and NCCL
  topology. There is no cross-model batching, KV transfer, prefix reuse, or
  native SRT DP group.
- The MVP is static and single-node. Its lifecycle is startup, health checking,
  registration, and whole-process-tree cleanup on error or shutdown.

The JSON schema is exemplified in
`examples/multi_model_router/static_multi_model.json`. Replace its model paths
and GPU IDs with local values before running it.

## Implementation sequence

### 1. Static configuration and topology validation

Files: `sglang_router/multi_model.py`, unit tests.

Validate the configuration before starting CUDA work:

- model IDs must be unique;
- every GPU ID must be non-negative and globally assigned exactly once;
- a group's size must equal `tp_size * pp_size`;
- the supervisor owns `model_path`, `served_model_name`, host, port,
  `base_gpu_id`, and `dp_size`;
- DPA, multi-node, and PD options are explicitly rejected in this MVP rather
  than being silently launched with an incorrect topology.

Acceptance gate: parser-only tests cover valid multi-model expansion, GPU
overlap, topology mismatch, and rejected supervisor-owned fields.

### 2. Supervisor launch and isolated GPU slices

Files: `sglang_router/launch_server.py`.

Extend the existing one-model DP launcher rather than changing SRT. A new
worker-process argument carries an environment map. The child sets
`CUDA_VISIBLE_DEVICES` before importing the HTTP/gRPC entry point and uses
`base_gpu_id=0`; this makes both contiguous and non-contiguous physical GPU
groups safe for a replica.

Acceptance gate: existing legacy DP launch behavior remains unchanged; each
multi-model replica receives `dp_size=1`, a unique port, and its own GPU slice.

### 3. IGW registration and startup transaction

Start an empty IGW router with `enable_igw=true`, wait for `/health`, then:

1. launch all declared SGLang workers;
2. wait for each worker `/health`;
3. `POST /workers` with `{url, model_id}`;
4. poll the worker registry until that worker appears under the expected model;
5. expose the ready unified endpoint only after all registrations succeed.

Any startup exception or SIGINT/SIGTERM/SIGQUIT terminates every worker and the
router by process group, preventing orphan schedulers or a partially populated
gateway.

Acceptance gate: mocked unit tests cover registration success/failure and a
GPU-backed integration test confirms `/v1/models` contains every `model_id` and
requests route only to their own model pool.

### 4. Integration test with available local models

Use Qwen3-0.6B for the first live run. A true heterogeneous run requires a
second compatible local model and enough free GPU groups; it should verify both
`/v1/models` entries, a completion per model, repeated requests per model, and
clean shutdown. This stage must not use the Qwen result as proof that an
unavailable GLM checkpoint loads correctly.

## Deliberately deferred work

The next layers build on this supervisor without changing the model-isolation
contract:

1. **Per-model PD groups:** add `roles.prefill` and `roles.decode` to each
   model spec. P/D pairing, bootstrap ports, and KV transfer remain inside one
   model and compatible KV-layout namespace.
2. **Runtime lifecycle controller:** model states `Loading`, `Ready`,
   `Draining`, `Sleeping`, and `Failed`; graceful model-scale down and restart.
3. **Resource planner:** choose which DP replicas to add or remove based on
   model-level queueing, load, and GPU reservation. Do not try to hot-resize a
   TP/PP/DPA group; drain and recreate it with the new topology.
4. **Cluster control plane:** external orchestration and multi-node placement;
   retain IGW as request-plane routing and use model/revision/KV-layout scoped
   cache indexes.

This sequence keeps the existing SRT scheduler single-model, which is both
safer and consistent with the Router roadmap's model-aware engine registry and
route-plan direction.
