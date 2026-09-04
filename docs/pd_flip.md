# PD Flip runtime deployment

This branch adds in-process Prefill/Decode role switching to SGLang. It is based
on the official upstream `v0.5.13` tag
(`28b095c01005d4a3a2a5b637b7d028b07fba31b2`). The PD Flip implementation is
carried as one commit on top of that exact release baseline.

The retained implementation consists of:

- SGLang worker runtime-role switching and migration endpoints;
- active-request KV migration, output relay, P-to-D handoff, and recovery;
- Mooncake bootstrap registration and cache hot reconfiguration;
- the Rust `sgl-router` drain/role administration API;
- the controller and its SLO/topology policy modules.

Experiment runners, trace generators, benchmark reports, and retained artifacts
are intentionally excluded.

## Prerequisites

- The same model and tokenizer files on every worker.
- A working SGLang runtime image with the selected transfer backend. The
  operational path used by this implementation is Mooncake.
- Reachable worker HTTP and bootstrap ports, plus correctly selected RDMA
  devices/GIDs when RDMA is used.
- A private admin key supplied through an environment variable. Never commit
  the key or place it in a shared configuration file.

Before starting a cluster, verify GPU ownership, ports, model completeness,
driver health, mounts, disk space, and RDMA reachability on every node.

## Build

Install the local Python package in the runtime environment:

```bash
python3 -m pip install -e python
```

Build the router:

```bash
cargo build --release --manifest-path experimental/sgl-router/Cargo.toml
```

The resulting binary is
`experimental/sgl-router/target/release/sgl-router`.

## Start workers

Load the admin key from a private environment file or secret manager:

```bash
export ADMIN_API_KEY='<private value>'
```

Start each worker with an explicit initial role. The following is a template;
model, topology, memory, network, and device values must match the deployment.

```bash
python3 -m sglang.launch_server \
  --model-path /models/MODEL \
  --served-model-name MODEL_ID \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 4 \
  --dp-size 1 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8997 \
  --disaggregation-ib-device mlx5_0 \
  --enable-pd-flip-state-machine \
  --enable-pd-runtime-role-switch \
  --admin-api-key "${ADMIN_API_KEY}"
```

Use `--disaggregation-mode decode` for Decode workers. For cache-plane rebuilds
without reloading model weights, also use
`--enable-pd-runtime-cache-hot-reconfigure`. Shared L3 HiCache additionally
requires deployment-specific HiCache flags and
`--enable-pd-runtime-shared-hicache`.

Do not send traffic until every worker passes `/health` and its
`/pd_flip/runtime_role/status` reports the expected role.

## Start the router

The router admin key is environment-only and should match the worker/controller
credential:

```bash
export PD_FLIP_ROUTER_ADMIN_API_KEY="${ADMIN_API_KEY}"

experimental/sgl-router/target/release/sgl-router \
  --host 0.0.0.0 \
  --port 30001 \
  --model-id MODEL_ID \
  --tokenizer-path /models/MODEL/tokenizer.json \
  --worker-urls \
    http://worker-a:30000 \
    http://worker-b:30000 \
    http://worker-c:30000 \
    http://worker-d:30000
```

Verify the router's `/pd_flip/router/workers` response before enabling client
traffic. It must show the expected worker set, roles, drain state, and bootstrap
ports.

## Configure and run the controller

Create a local, uncommitted JSON configuration such as:

```json
{
  "router_url": "http://router:30001",
  "nodes": [
    {
      "name": "worker-a",
      "worker_url": "http://worker-a:30000",
      "router_worker_id": "http://worker-a:30000",
      "bootstrap_port": 8997
    },
    {
      "name": "worker-b",
      "worker_url": "http://worker-b:30000",
      "router_worker_id": "http://worker-b:30000",
      "bootstrap_port": 8997
    }
  ],
  "session_journal_path": "/private/pd-flip/session.json",
  "first_migration_ratio": 0.5,
  "observation_seconds": 2.0
}
```

Inspect metrics and build a non-mutating plan first:

```bash
python3 scripts/playground/disaggregation/pd_flip_controller.py \
  --config /private/pd-flip/cluster.json \
  --api-key-env ADMIN_API_KEY \
  metrics

python3 scripts/playground/disaggregation/pd_flip_controller.py \
  --config /private/pd-flip/cluster.json \
  --api-key-env ADMIN_API_KEY \
  dry-run --direction d_to_p
```

Execute a selected transition only after the dry-run identifies the intended
source and target:

```bash
python3 scripts/playground/disaggregation/pd_flip_controller.py \
  --config /private/pd-flip/cluster.json \
  --api-key-env ADMIN_API_KEY \
  execute --direction d_to_p \
  --source-name worker-b \
  --migration-target-name worker-a
```

For automatic operation, use `monitor-continuous` with explicit TTFT/TPOT SLOs,
worker bounds, cooldown, and a durable session journal. Run
`pd_flip_controller.py --help` for the complete policy interface.

## Operational invariants

- Start the router only after all workers are healthy and role-verified.
- Drain a worker in the router before mutating its runtime role.
- Keep the controller journal on durable storage and use a unique session ID.
- Never overlap two controllers for the same workers.
- On failure, stop new admissions, retain status/log evidence, and reconcile or
  abort the current session before retrying.
- A repaired attempt must use a new session/run identifier; do not overwrite
  failure evidence.
