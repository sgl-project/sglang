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
- `run_monitor.sh`: run the monitor-driven controller loop that observes
  TTFT/TPOT SLO attainment and commits or aborts two-phase D->P migration.
- `windows_four_node.ps1`: run SSH-based four-node orchestration from a Windows
  host whose SSH config can reach `cloud-099` through `cloud-102`.

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

The default host layout matches the four SSH aliases:

```text
cloud-099 -> node0 -> prefill
cloud-100 -> node1 -> prefill
cloud-101 -> node2 -> decode
cloud-102 -> node3 -> decode
```

If WSL cannot reach the SSH aliases but Windows can, run orchestration from a
Windows checkout of this same code. First make sure the Windows checkout and the
four cloud-node checkouts contain the same commit or patch as this working tree.
The easiest path is to push this branch, pull it on Windows, then run:

```powershell
cd scripts\playground\disaggregation\pd_flip_docker
copy env.example env.local
notepad env.local
.\windows_four_node.ps1 -Action preflight
.\windows_four_node.ps1 -Action sync-env
.\windows_four_node.ps1 -Action pull
```

If the changes are not pushed to a git remote yet, export/apply a patch into the
Windows checkout first, then use the same script to drive the cloud nodes.

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
./run_controller.sh monitor
```

To launch workers through the configured SSH aliases, keep the same `env.local`
path on each host or point `ENV_FILE` at a host-local copy:

```bash
ssh cloud-099 'cd /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=$PWD/env.local tmux new -d -s pd-node0 "./run_worker.sh prefill 0.0.0.0 |& tee worker.log"'
ssh cloud-100 'cd /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=$PWD/env.local tmux new -d -s pd-node1 "./run_worker.sh prefill 0.0.0.0 |& tee worker.log"'
ssh cloud-101 'cd /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=$PWD/env.local tmux new -d -s pd-node2 "./run_worker.sh decode 0.0.0.0 |& tee worker.log"'
ssh cloud-102 'cd /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=$PWD/env.local tmux new -d -s pd-node3 "./run_worker.sh decode 0.0.0.0 |& tee worker.log"'
```

From Windows, the equivalent scripted launch is:

```powershell
.\windows_four_node.ps1 -Action start-workers
.\windows_four_node.ps1 -Action start-router
.\windows_four_node.ps1 -Action start-monitor -MonitorIterations 120 -MonitorPollInterval 1
.\windows_four_node.ps1 -Action status
```

Use `.\windows_four_node.ps1 -Action logs` to tail worker/router/monitor logs
and `.\windows_four_node.ps1 -Action stop` to stop the tmux sessions.

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

Monitor loop:

```bash
PD_FLIP_MONITOR_ITERATIONS=120 PD_FLIP_MONITOR_POLL_INTERVAL=1 ./run_controller.sh monitor
```

The monitor computes prefill TTFT attainment and decode TPOT attainment over
the configured window. When prefill attainment crosses the enter threshold, the
D->P path drains `node2`, pauses admission, starts two-phase KV pre-transfer to
another decode node with `prepare_only=true`, and keeps observing SLO before any
role switch. If SLO recovers above the exit threshold, the controller aborts the
target/source migration and returns `node2` to `safe(decode)`. If SLO remains
risky after KV transfer, the controller commits the target migration, releases
source requests, switches `node2` to `prefill`, refreshes the router role, and
undrains.

The execution result includes `success`, `message`, `source`, `migration_target`,
`migration_seconds`, `total_seconds`, and per-step `actions`. For the basic
experiment, accept the run only when `success=true`, router workers show the new
role, and client traffic records no unexpected 5xx errors during the flip. For
the monitor experiment, collect the printed snapshots/actions as the decision
log and keep the worker `/server_info` and `/pd_flip/migration/status` snapshots
with the client latency logs.

## Notes

- `run_router.sh` first tries `experimental/sgl-router/target/release/sgl-router`.
  If it is missing and the Docker image has `cargo`, it builds/runs via
  `cargo run --release`. If the official image has no cargo, build the binary
  once in the mounted repo before starting the router.
- Add model-specific flags such as `--trust-remote-code` through
  `EXTRA_SGLANG_ARGS` in `env.local`.
- Add NIC/NCCL Docker environment flags through `EXTRA_DOCKER_ARGS`.
