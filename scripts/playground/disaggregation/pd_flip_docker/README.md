# Four-Node PD Runtime Role Flip Docker Harness

This harness runs the PD role-flip experiment on four physical servers. Each
server runs one SGLang worker container across its 8 local GPUs; the worker is
either active `prefill` or active `decode`, while both PD queues are initialized
because `--enable-pd-runtime-role-switch` is enabled.

## Files

- `env.example`: shared cluster variables.
- `run_worker.sh`: launch one worker container on the current physical node.
- `run_router.sh`: launch sgl-router against the four worker URLs.
- `run_controller.sh`: collect metrics, build a D->P/P->D dry-run plan, or
  execute the plan against live workers/router.

## Setup

```bash
cd scripts/playground/disaggregation/pd_flip_docker
cp env.example env.local
vi env.local
export ENV_FILE=$PWD/env.local
```

Set `SGLANG_REPO` to the checkout containing these changes. Set `MODEL_PATH`
to a path visible on every worker host, and set `NODE0`...`NODE3` to the worker
HTTP URLs that the router/controller can reach.

Static sgl-router discovery uses the worker URL itself as `worker_id`, so the
controller sends router admin calls with `router_worker_id=${NODE_URL}`.

## Launch Order

On node0:

```bash
./run_worker.sh prefill 0.0.0.0
```

On node1:

```bash
./run_worker.sh prefill 0.0.0.0
```

On node2:

```bash
./run_worker.sh decode 0.0.0.0
```

On node3:

```bash
./run_worker.sh decode 0.0.0.0
```

On the router host:

```bash
./run_router.sh
```

On the controller host:

```bash
./run_controller.sh metrics
DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh dry-run
DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh execute
```

## Checks

Worker runtime role:

```bash
curl -s "${NODE2}/pd_flip/runtime_role/status" | jq
```

Router view:

```bash
curl -s "http://${ROUTER_HOST}:${ROUTER_PORT}/pd_flip/router/workers" | jq
```

Controller dry-run:

```bash
DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh dry-run | jq '.actions'
```

Controller execution:

```bash
DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh execute | tee d_to_p-result.json
DIRECTION=p_to_d SOURCE_NAME=node0 ./run_controller.sh execute | tee p_to_d-result.json
```

The D->P plan should drain `node2`, pause its admission, migrate active decode
state to another decode node, call target prepare with `adopt_on_success=true`,
switch `node2` to `prefill`, refresh the router role, and undrain.

The execution result includes `success`, `message`, `source`, `migration_target`,
`migration_seconds`, `total_seconds`, and per-step `actions`. For the basic
experiment, accept the run only when `success=true`, router workers show the new
role, and client traffic records no unexpected 5xx errors during the flip.

## Notes

- `run_router.sh` first tries `experimental/sgl-router/target/release/sgl-router`.
  If it is missing and the Docker image has `cargo`, it builds/runs via
  `cargo run --release`. If the official image has no cargo, build the binary
  once in the mounted repo before starting the router.
- Add model-specific flags such as `--trust-remote-code` through
  `EXTRA_SGLANG_ARGS` in `env.local`.
- Add NIC/NCCL Docker environment flags through `EXTRA_DOCKER_ARGS`.
