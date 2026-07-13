# PD Flip Four-Node Linux Docker Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and validate the monitor-driven PD flip state machine on four remote Linux GPU servers in Docker, starting from the Windows Codex workstation and ending with saved baseline, recovery, commit, and failure artifacts.

**Architecture:** Treat the controller/monitor as the cluster-level state-machine owner for the first real experiment, while workers execute role switch and KV migration primitives. The remote Linux nodes run one Dockerized SGLang worker each with host networking; `cloud-099` also runs the router and controller unless preflight shows it cannot reach every worker. The experiment first proves the existing controller-owned two-phase D->P flow, then tightens the remaining gaps needed for the full diagram-level `safe -> preparing -> flipping -> safe(new role)` behavior.

**Tech Stack:** Windows PowerShell SSH orchestration, Linux bash on remote nodes, Docker host networking, NVIDIA GPUs, SGLang worker admin endpoints, `experimental/sgl-router`, `scripts/playground/disaggregation/pd_flip_docker/*`, `pd_flip_controller.py`, `pd_flip_monitor.py`, Prometheus-style `/metrics`, `/v1/loads`, Mooncake or fake transfer backend.

---

## Scope Check

This is two separable tracks:

1. **Experiment execution track:** Use the current harness to run four-node Docker experiments on `cloud-099..cloud-102`.
2. **State-machine completion track:** Patch the remaining semantic gaps before claiming it matches the full monitor-managed diagram.

Run the execution track first with the current code. Patch only the state-machine completion items that block the specific experiment branch being validated.

## No-Container Safe Mode

This mode is active until the user explicitly re-authorizes remote container or
GPU work. It exists because the four remote nodes may be running other people's
experiments.

Hard stops while this mode is active:

- Do not start, stop, restart, remove, inspect, log, or exec into any remote Docker container.
- Do not run `docker` commands on `cloud-099..cloud-102`, even read-only ones, unless explicitly re-authorized.
- Do not send traffic, health, admin, metrics, role-switch, drain, or router-refresh requests to existing remote services.
- Do not run GPU workloads, benchmark clients, Docker tests, or model warmup on the remote nodes.

Allowed work while this mode is active:

- Local static analysis with `rg` and targeted file reads.
- Local code, test, and plan edits.
- Local unit-test design and fake-client test planning that does not start Docker or touch GPUs.
- Remote read-only host checks only when they cannot interact with containers or services, such as `hostname`, file metadata, or disk metadata.

Current blocker for the real four-node execution:

- Existing remote workloads were observed to occupy GPU memory on the target nodes.
- A real DeepSeek/Mooncake run requires a dedicated experiment window, replacement idle nodes, or a much smaller isolated test model.
- The tasks below remain the future execution runbook. While No-Container Safe Mode is active, do not run any remote step that calls `docker`, `curl http://`, `run_worker.sh`, `run_router.sh`, `run_controller.sh`, `windows_four_node.ps1 -Action start-*`, `windows_four_node.ps1 -Action stop`, `windows_four_node.ps1 -Action status`, `windows_four_node.ps1 -Action logs`, `windows_four_node.ps1 -Action sync-code`, or `windows_four_node.ps1 -Action sync-env`.

## Authorized Experiment Execution Plan

Use this section when the user explicitly says the four-node Docker/GPU
experiment window is open. The intent is to start only this experiment's own
processes and to stop immediately if any gate suggests another workload would
be affected.

Required authorization sentence before running any command in this section:

```text
I authorize PD flip experiment execution on cloud-099..cloud-102. You may use Docker and GPUs for this experiment only. Do not stop or modify non-PD-flip containers.
```

Ownership rules for the experiment window:

- Allowed to start: `pd-node0`, `pd-node1`, `pd-node2`, `pd-node3`, `pd-router`, `pd-monitor`, and one-off `sglang-pd-switch:tianciJ` test containers started by this plan.
- Allowed to stop: only the tmux sessions or one-off containers started by this plan.
- Never stop, restart, remove, exec into, or change any pre-existing non-PD-flip container.
- Stop the run if required ports are occupied by another process or if GPU memory is not available.

### Phase 0: Local Readiness Gate

- [ ] **Step 0.1: Confirm local env matches the reserved ports**

Run:

```powershell
cd "C:\Users\Tianci J\Desktop\sglang"
Get-Content "scripts\playground\disaggregation\pd_flip_docker\env.local"
```

Expected:

```text
PORT=31000
BOOTSTRAP_PORT=18998
ROUTER_PORT=18000
NODE0=http://192.168.0.42:31000
NODE1=http://192.168.0.40:31000
NODE2=http://192.168.0.39:31000
NODE3=http://192.168.0.41:31000
```

- [ ] **Step 0.2: Avoid the current status-port footgun**

Do not use `windows_four_node.ps1 -Action status` until its hard-coded
`http://127.0.0.1:30000/pd_flip/runtime_role/status` check is changed to read
`${PORT}` from `env.local`. Use the explicit `curl` commands in Phase 4
instead.

### Phase 1: Remote Resource Gate

- [ ] **Step 1.1: Confirm host access without changing services**

Run:

```powershell
foreach ($h in "cloud-099","cloud-100","cloud-101","cloud-102") {
  ssh $h "hostname; python3 --version; hostname -I"
}
```

Expected: every host responds, and host Python may print `Python 3.6.8`.

- [ ] **Step 1.2: Check ports and GPU availability**

Run only after authorization:

```powershell
foreach ($h in "cloud-099","cloud-100","cloud-101","cloud-102") {
  ssh $h "echo host=$(hostname); ss -ltnp | egrep ':31000|:18998|:18000' || true; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader"
}
```

Expected: no listener on `31000`, `18998`, or `18000` unless it is this
experiment's own leftover process, and each node has enough free GPU memory for
the configured model. If any required port is owned by another process or GPU
memory is occupied by another active experiment, stop and ask for a new window.

- [ ] **Step 1.3: Check Docker image and model path**

Run only after authorization:

```powershell
foreach ($h in "cloud-099","cloud-100","cloud-101","cloud-102") {
  ssh $h "docker image inspect sglang-pd-switch:tianciJ >/dev/null && echo image_ok || echo image_missing; test -d /models/deepseek_v3.1_terminus && echo model_ok || echo model_missing"
}
```

Expected: every host prints `image_ok` and `model_ok`. If either is missing,
stop before launching workers.

### Phase 2: Sync And Unit-Test Gate

- [ ] **Step 2.1: Sync the exact local code and env**

Run:

```powershell
cd "C:\Users\Tianci J\Desktop\sglang\scripts\playground\disaggregation\pd_flip_docker"
.\windows_four_node.ps1 -Action sync-code -RemoteRepo "/root/sglang"
.\windows_four_node.ps1 -Action sync-env -RemoteRepo "/root/sglang"
```

Expected: all four nodes receive `/root/sglang`, executable shell scripts, and
the `env.local` with port `31000`.

- [ ] **Step 2.2: Run PD flip unit tests inside the image**

Run:

```powershell
ssh cloud-099 "docker run --rm --gpus all --network host -v /root/sglang:/sgl-workspace/sglang sglang-pd-switch:tianciJ bash -lc 'python3 --version; cd /sgl-workspace/sglang; PYTHONPATH=python python3 -m unittest discover -s test/srt -p test_pd_flip*.py -v'"
```

Expected: Python is 3.12.x inside the container and the PD flip unittest suite
passes. If it fails, do not start the four workers.

### Phase 3: Start Cluster

- [ ] **Step 3.1: Start four workers**

Run:

```powershell
cd "C:\Users\Tianci J\Desktop\sglang\scripts\playground\disaggregation\pd_flip_docker"
.\windows_four_node.ps1 -Action start-workers -RemoteRepo "/root/sglang"
```

Expected: `pd-node0`, `pd-node1`, `pd-node2`, and `pd-node3` sessions exist.

- [ ] **Step 3.2: Wait for workers and verify role endpoints explicitly**

Run:

```powershell
Start-Sleep -Seconds 180
ssh cloud-099 "curl -fsS http://127.0.0.1:31000/pd_flip/runtime_role/status"
ssh cloud-100 "curl -fsS http://127.0.0.1:31000/pd_flip/runtime_role/status"
ssh cloud-101 "curl -fsS http://127.0.0.1:31000/pd_flip/runtime_role/status"
ssh cloud-102 "curl -fsS http://127.0.0.1:31000/pd_flip/runtime_role/status"
```

Expected: node0/node1 report `prefill`, node2/node3 report `decode`, and each
worker reports runtime role switching enabled.

- [ ] **Step 3.3: Start router and verify workers**

Run:

```powershell
.\windows_four_node.ps1 -Action start-router -RemoteRepo "/root/sglang"
Start-Sleep -Seconds 20
ssh cloud-099 "curl -fsS http://127.0.0.1:18000/pd_flip/router/workers"
```

Expected: router sees four workers with two `prefill` and two `decode` roles.

### Phase 4: Manual Controller Branch

- [ ] **Step 4.1: Capture baseline metrics**

Run:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local ./run_controller.sh metrics | tee metrics-before.json"
```

Expected: every worker has correct role, load, running request count, waiting
request count, and router status.

- [ ] **Step 4.2: Dry-run D->P with explicit source**

Run:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh dry-run | tee d-to-p-dry-run.json"
```

Expected: the action list includes source drain, source admission pause, source
migration start, target prepare-only migration, source finish, role switch,
router refresh, admission resume, and router undrain.

- [ ] **Step 4.3: Execute D->P once**

Run:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh execute | tee d-to-p-result.json"
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local ./run_controller.sh metrics | tee metrics-after-d-to-p.json"
```

Expected: `d-to-p-result.json` has `success=true`, source `node2`, migration
target `node3`, and final topology is 3P/1D.

### Phase 5: Monitor Branches

- [ ] **Step 5.1: Reset to 2P/2D**

Run:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local DIRECTION=p_to_d SOURCE_NAME=node2 ./run_controller.sh execute | tee p-to-d-reset-node2.json"
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local ./run_controller.sh metrics | tee metrics-after-reset.json"
```

Expected: node0/node1 are `prefill`, node2/node3 are `decode`.

- [ ] **Step 5.2: Run safe monitor with no induced pressure**

Run:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local PD_FLIP_MONITOR_ITERATIONS=10 PD_FLIP_MONITOR_POLL_INTERVAL=1 ./run_controller.sh monitor | tee monitor-safe.json"
```

Expected: no flip is committed without risky SLO samples.

- [ ] **Step 5.3: Run recovery branch**

Start monitor, induce prefill pressure long enough to enter preparing, then
remove pressure before commit threshold remains risky.

Run the monitor side:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local PD_FLIP_WINDOW_SECONDS=3 PD_FLIP_MONITOR_ITERATIONS=120 PD_FLIP_MONITOR_POLL_INTERVAL=1 ./run_controller.sh monitor | tee monitor-recovery.json"
```

Expected: action log records target/source migration abort and cleanup, and
node2 remains `decode`.

- [ ] **Step 5.4: Run commit branch**

Start monitor again and keep prefill pressure risky through the KV transfer
window.

Run:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local PD_FLIP_WINDOW_SECONDS=3 PD_FLIP_MONITOR_ITERATIONS=120 PD_FLIP_MONITOR_POLL_INTERVAL=1 ./run_controller.sh monitor | tee monitor-commit.json"
```

Expected: action log records target commit, source finish, wait idle, runtime
role set to `prefill`, router role refresh, and final topology 3P/1D.

### Phase 6: Stop Only This Experiment And Collect Artifacts

- [ ] **Step 6.1: Collect artifacts before stopping sessions**

Run:

```powershell
New-Item -ItemType Directory -Force "C:\Users\Tianci J\Desktop\sglang\pd-flip-artifacts" | Out-Null
scp cloud-099:/root/sglang/scripts/playground/disaggregation/pd_flip_docker/*.json "C:\Users\Tianci J\Desktop\sglang\pd-flip-artifacts\"
```

Expected: local artifact directory contains `metrics-before.json`,
`d-to-p-dry-run.json`, `d-to-p-result.json`, `metrics-after-d-to-p.json`,
`p-to-d-reset-node2.json`, `monitor-safe.json`, `monitor-recovery.json`, and
`monitor-commit.json` when those phases have run.

- [ ] **Step 6.2: Stop only PD flip sessions**

Run:

```powershell
cd "C:\Users\Tianci J\Desktop\sglang\scripts\playground\disaggregation\pd_flip_docker"
.\windows_four_node.ps1 -Action stop -RemoteRepo "/root/sglang"
```

Expected: only `pd-node0`, `pd-node1`, `pd-node2`, `pd-node3`, `pd-router`,
and `pd-monitor` sessions are stopped. No non-PD-flip containers or sessions
are modified.

Abort criteria for every phase:

- A required port is already occupied by a non-PD-flip process.
- GPU memory is not available on any node.
- Any worker fails to expose role status after startup.
- Router does not see exactly four workers.
- Controller dry-run omits the expected migration or role-switch actions.
- Any command reports a failure after target commit; collect artifacts and stop
the experiment sessions rather than attempting an improvised rollback.

## Fixed Cluster Mapping

Use the SSH aliases from the workstation:

```text
AliECS     -> 121.89.86.41, ProxyJump bastion
cloud-099 -> 8.130.110.23 via AliECS, node0, initial prefill
cloud-100 -> 8.130.96.17  via AliECS, node1, initial prefill
cloud-101 -> 8.130.17.103 via AliECS, node2, initial decode, primary D->P source
cloud-102 -> 8.130.33.36  via AliECS, node3, initial decode, D->P migration target
```

Use the current `env.local` private worker URLs:

```bash
NODE0=http://192.168.0.42:31000
NODE1=http://192.168.0.40:31000
NODE2=http://192.168.0.39:31000
NODE3=http://192.168.0.41:31000
ROUTER_HOST=127.0.0.1
ROUTER_PORT=18000
```

Remote repo and model settings:

```bash
SGLANG_REPO=/root/sglang
IMAGE=sglang-pd-switch:tianciJ
MODEL_PATH=/models/deepseek_v3.1_terminus
MODEL_ID=deepseek_v3.1_terminus
TOKENIZER_PATH=/models/deepseek_v3.1_terminus
TP_SIZE=8
DP_SIZE=1
PORT=31000
BOOTSTRAP_PORT=18998
TRANSFER_BACKEND=mooncake
IB_DEVICE=mlx5_0
EXTRA_SGLANG_ARGS='--trust-remote-code --enable-metrics'
```

Important runtime split:

```text
Remote Linux host Python: 3.6.8
SGLang Docker image Python: 3.12.3
```

Do not run current SGLang Python scripts or tests directly on the remote host
Python. Use `run_controller.sh` with `PD_FLIP_CONTROLLER_USE_DOCKER=1` or run
tests through `docker run ... sglang-pd-switch:tianciJ ... python3`.

## File Structure

- `C:/Users/Tianci J/Desktop/sglang/scripts/playground/disaggregation/pd_flip_docker/windows_four_node.ps1`: Windows-side SSH orchestration for preflight, code sync, env sync, worker/router/monitor start, status, logs, stop.
- `C:/Users/Tianci J/Desktop/sglang/scripts/playground/disaggregation/pd_flip_docker/env.local`: local source of truth for remote Linux `env.local`.
- `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/env.local`: remote Linux env file copied to all four nodes.
- `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/run_worker.sh`: Linux worker container entrypoint.
- `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/run_router.sh`: Linux router container entrypoint on `cloud-099`.
- `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/run_controller.sh`: Linux controller/monitor entrypoint on `cloud-099`.
- `C:/Users/Tianci J/Desktop/sglang/scripts/playground/disaggregation/pd_flip_controller.py`: controller-owned two-phase D->P/P->D orchestration.
- `C:/Users/Tianci J/Desktop/sglang/scripts/playground/disaggregation/pd_flip_monitor.py`: SLO attainment monitor.
- `C:/Users/Tianci J/Desktop/sglang/python/sglang/srt/disaggregation/flip_state_machine.py`: worker-local FSM, currently observability-oriented rather than the controller's source of truth.
- `C:/Users/Tianci J/Desktop/sglang/python/sglang/srt/managers/scheduler.py`: worker runtime role switch and migration source/target behavior.

---

### Task 1: Windows SSH And Linux Preflight

**Files:**
- Read: `C:/Users/Tianci J/.ssh/config`
- Read: `C:/Users/Tianci J/Desktop/sglang/scripts/playground/disaggregation/pd_flip_docker/windows_four_node.ps1`

- [ ] **Step 1: Confirm the workstation SSH aliases resolve**

Run from PowerShell on Windows:

```powershell
ssh AliECS "hostname; date"
ssh cloud-099 "hostname; hostname -I"
ssh cloud-100 "hostname; hostname -I"
ssh cloud-101 "hostname; hostname -I"
ssh cloud-102 "hostname; hostname -I"
```

Expected: every command prints a Linux hostname and IP list. The `cloud-*` commands must work through `ProxyJump AliECS`.

- [ ] **Step 2: Run scripted preflight**

Run:

```powershell
cd "C:\Users\Tianci J\Desktop\sglang\scripts\playground\disaggregation\pd_flip_docker"
.\windows_four_node.ps1 -Action preflight
```

Expected: each remote node prints hostname, GPU count, Docker version, repo presence, and `env_local=ok` or `env_local=missing`.

- [ ] **Step 2.5: Confirm Docker Python for test execution**

Run:

```powershell
ssh cloud-099 "docker run --rm --gpus all --network host -v /root/sglang:/sgl-workspace/sglang sglang-pd-switch:tianciJ bash -lc 'python3 --version; cd /sgl-workspace/sglang; PYTHONPATH=python python3 -m unittest discover -s test/srt -p test_pd_flip*.py -v'"
```

Expected: container Python is 3.12.x and the PD flip unittest set passes. If
host Python prints 3.6.8, that is expected but should not be used for these
tests.

- [ ] **Step 3: Record the Linux facts**

Create this local note file:

```powershell
New-Item -ItemType Directory -Force "C:\Users\Tianci J\Desktop\sglang\pd-flip-artifacts\preflight" | Out-Null
.\windows_four_node.ps1 -Action preflight *> "C:\Users\Tianci J\Desktop\sglang\pd-flip-artifacts\preflight\preflight.txt"
```

Expected: `preflight.txt` contains four GPU counts, Docker versions, and repo/env status.

### Task 2: Sync Code And Linux Env Correctly

**Files:**
- Modify or verify: `C:/Users/Tianci J/Desktop/sglang/scripts/playground/disaggregation/pd_flip_docker/env.local`
- Execute remotely: `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/*.sh`

- [ ] **Step 1: Write the current env file from PowerShell**

Run:

```powershell
cd "C:\Users\Tianci J\Desktop\sglang\scripts\playground\disaggregation\pd_flip_docker"
.\windows_four_node.ps1 `
  -Action write-env `
  -RemoteRepo "/root/sglang" `
  -Image "sglang-pd-switch:tianciJ" `
  -ModelPath "/models/deepseek_v3.1_terminus" `
  -ModelId "deepseek_v3.1_terminus" `
  -TokenizerPath "/models/deepseek_v3.1_terminus" `
  -Node0Url "http://192.168.0.42:31000" `
  -Node1Url "http://192.168.0.40:31000" `
  -Node2Url "http://192.168.0.39:31000" `
  -Node3Url "http://192.168.0.41:31000" `
  -TpSize 8 `
  -DpSize 1 `
  -MemFractionStatic 0.88 `
  -TransferBackend "mooncake" `
  -IbDevice "mlx5_0" `
  -RouterHost "127.0.0.1" `
  -RouterPort 18000 `
  -ExtraSGLangArgs "--trust-remote-code --enable-metrics"
```

Expected: local `env.local` is ASCII and contains the exact node URLs above.

- [ ] **Step 2: Sync code to all Linux nodes**

Use archive sync when the remote branch may not match the local working tree:

```powershell
.\windows_four_node.ps1 -Action sync-code -RemoteRepo "/root/sglang"
```

Expected: `/root/sglang` exists on all four Linux nodes and executable bits are set on `pd_flip_docker/*.sh`.

- [ ] **Step 3: Sync env and remove CRLF remotely**

Run:

```powershell
.\windows_four_node.ps1 -Action sync-env -RemoteRepo "/root/sglang"
```

Expected: each node receives `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/env.local`, and the script runs `sed -i 's/\r$//'` on Linux.

- [ ] **Step 4: Verify Linux shell parsing**

Run:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && set -a && . ./env.local && set +a && printf 'repo=%s image=%s node2=%s backend=%s\n' `$SGLANG_REPO `$IMAGE `$NODE2 `$TRANSFER_BACKEND"
```

Expected:

```text
repo=/root/sglang image=sglang-pd-switch:tianciJ node2=http://192.168.0.39:31000 backend=mooncake
```

### Task 3: Docker, Model, Router Dependency Gate

**Files:**
- Execute: `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/prepare_router_deps.sh`
- Execute: `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/run_router.sh`

- [ ] **Step 1: Check image and GPU visibility on every Linux node**

Run:

```powershell
foreach ($h in "cloud-099","cloud-100","cloud-101","cloud-102") {
  ssh $h "docker image inspect sglang-pd-switch:tianciJ >/dev/null && echo image_ok || echo image_missing; docker run --rm --gpus all sglang-pd-switch:tianciJ nvidia-smi -L | wc -l"
}
```

Expected: every node prints `image_ok` and `8`.

- [ ] **Step 2: Check model path on every Linux node**

Run:

```powershell
foreach ($h in "cloud-099","cloud-100","cloud-101","cloud-102") {
  ssh $h "test -d /models/deepseek_v3.1_terminus && echo model_ok || echo model_missing; ls -ld /models/deepseek_v3.1_terminus"
}
```

Expected: every node prints `model_ok`.

- [ ] **Step 3: Check RDMA device for Mooncake**

Run:

```powershell
foreach ($h in "cloud-099","cloud-100","cloud-101","cloud-102") {
  ssh $h "ls /sys/class/infiniband; test -e /sys/class/infiniband/mlx5_0 && echo mlx5_0_ok || echo mlx5_0_missing"
}
```

Expected: every node prints `mlx5_0_ok`. If one node prints `mlx5_0_missing`, set `IB_DEVICE` to the actual common device and rerun Task 2 Step 1 through Step 4.

- [ ] **Step 4: Prepare router dependencies on `cloud-099`**

Run:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local ./prepare_router_deps.sh"
```

Expected: `/root/sglang/router_deps.tgz` is written, or `experimental/sgl-router/target/release/sgl-router` already exists. If the Docker image lacks cargo, build the router binary once on `cloud-099` outside the worker launch path.

### Task 4: Baseline Four-Worker Startup

**Files:**
- Execute: `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/run_worker.sh`
- Execute: `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/run_router.sh`

- [ ] **Step 1: Stop old sessions before starting**

Run:

```powershell
cd "C:\Users\Tianci J\Desktop\sglang\scripts\playground\disaggregation\pd_flip_docker"
.\windows_four_node.ps1 -Action stop -RemoteRepo "/root/sglang"
```

Expected: old worker, router, and monitor tmux sessions are gone.

- [ ] **Step 2: Start workers**

Run:

```powershell
.\windows_four_node.ps1 -Action start-workers -RemoteRepo "/root/sglang"
```

Expected: `pd-node0`, `pd-node1`, `pd-node2`, `pd-node3` sessions exist.

- [ ] **Step 3: Wait for worker HTTP health**

Run:

```powershell
Start-Sleep -Seconds 120
.\windows_four_node.ps1 -Action status -RemoteRepo "/root/sglang"
```

Expected: role status endpoints show node0/node1 as `prefill`, node2/node3 as `decode`, and `runtime_role_switch_enabled=true`.

- [ ] **Step 4: Start router**

Run:

```powershell
.\windows_four_node.ps1 -Action start-router -RemoteRepo "/root/sglang"
Start-Sleep -Seconds 20
ssh cloud-099 "source /root/sglang/scripts/playground/disaggregation/pd_flip_docker/env.local && curl -fsS http://127.0.0.1:`${ROUTER_PORT}/pd_flip/router/workers"
```

Expected: router worker list contains four workers with two prefill and two decode roles.

### Task 5: Controller Dry-Run And Manual Execute Branch

**Files:**
- Execute: `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/run_controller.sh`
- Read results: `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/*result*.json`

- [ ] **Step 1: Collect current metrics**

Run:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local ./run_controller.sh metrics | tee metrics-before.json"
```

Expected: all four nodes have correct `effective_role`, `running_reqs`, `waiting_reqs`, and router status.

- [ ] **Step 2: Dry-run D->P with explicit source**

Run:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh dry-run | tee d-to-p-dry-run.json"
```

Expected actions include:

```text
router_drain_source
pause_source_admission
start_decode_migration_source
prepare_decode_migration_target
wait_decode_migration_source
wait_decode_migration_target
finish_decode_migration_source
wait_source_idle
set_source_runtime_role
refresh_router_source_role
resume_source_admission
router_undrain_source
```

- [ ] **Step 3: Execute D->P commit branch**

Run:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh execute | tee d-to-p-result.json"
```

Expected: `success=true`, `source=node2`, `target_role=prefill`, and `migration_target=node3`.

- [ ] **Step 4: Verify final topology**

Run:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local ./run_controller.sh metrics | tee metrics-after-d-to-p.json"
```

Expected: node0/node1/node2 are `prefill`; node3 is `decode`.

### Task 6: Monitor-Driven State Machine Branches

**Files:**
- Execute: `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/run_monitor.sh`
- Modify if needed: `C:/Users/Tianci J/Desktop/sglang/scripts/playground/disaggregation/pd_flip_controller.py`
- Modify if needed: `C:/Users/Tianci J/Desktop/sglang/test/srt/test_pd_flip_controller.py`

- [ ] **Step 1: Reset topology to 2P/2D**

Run P->D on node2 if Task 5 changed it:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local DIRECTION=p_to_d SOURCE_NAME=node2 ./run_controller.sh execute | tee p-to-d-reset-node2.json"
```

Expected: node2 returns to decode.

- [ ] **Step 2: Start monitor without traffic**

Run:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local PD_FLIP_MONITOR_ITERATIONS=10 PD_FLIP_MONITOR_POLL_INTERVAL=1 ./run_controller.sh monitor | tee monitor-safe.json"
```

Expected: monitor returns `message=no flip decision` or no committed action because there are no risky SLO samples.

- [ ] **Step 3: Add a reliable trigger strategy**

Use one of these two strategies before claiming monitor automation:

```text
Strategy A: real traffic trigger
- Generate enough long-prompt traffic through router to make prefill TTFT attainment fall below 0.9.
- Stop or reduce traffic to test recovery.

Strategy B: controller test hook
- Add a test-only or CLI fixture mode that feeds synthetic SLO snapshots into pd_flip_controller.py.
- Use it only for branch validation, not for production performance claims.
```

Expected: the selected strategy can deterministically produce both below-threshold and recovered SLO windows.

- [ ] **Step 4: Validate recovery branch**

Run monitor while inducing then removing prefill pressure:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local PD_FLIP_WINDOW_SECONDS=3 PD_FLIP_MONITOR_ITERATIONS=120 PD_FLIP_MONITOR_POLL_INTERVAL=1 ./run_controller.sh monitor | tee monitor-recovery.json"
```

Expected: action log contains target/source migration abort and cleanup; final role for node2 remains `decode`.

- [ ] **Step 5: Validate commit branch**

Run monitor while keeping prefill pressure risky through KV transfer:

```powershell
ssh cloud-099 "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && ENV_FILE=./env.local PD_FLIP_WINDOW_SECONDS=3 PD_FLIP_MONITOR_ITERATIONS=120 PD_FLIP_MONITOR_POLL_INTERVAL=1 ./run_controller.sh monitor | tee monitor-commit.json"
```

Expected: action log contains target commit, source finish, wait idle, runtime role set to `prefill`, router role refresh, and final topology 3P/1D.

### Task 7: State-Machine Completion Gaps Before Full Diagram Claim

**Files:**
- Modify: `C:/Users/Tianci J/Desktop/sglang/scripts/playground/disaggregation/pd_flip_controller.py`
- Modify: `C:/Users/Tianci J/Desktop/sglang/python/sglang/srt/disaggregation/flip_state_machine.py`
- Modify: `C:/Users/Tianci J/Desktop/sglang/test/srt/test_pd_flip_controller.py`
- Modify: `C:/Users/Tianci J/Desktop/sglang/test/srt/test_pd_flip_state_machine.py`

Static findings from the No-Container review:

- D->P auto source selection still prefers the highest-load decode node in dry-run, execute, and monitor mode. For the diagram-style "choose the low-request decode node to become prefill" behavior, the source side should prefer the lowest migration-cost decode node, while the remaining decode node with spare capacity becomes the KV target.
- Monitor mode still uses direct SLO threshold helpers for TTFT/TPOT risk. `PDRatioSLOFlipEvaluator` exists, but controller monitor mode does not use it as the cluster policy owner and does not record P/D ratio curve metadata.
- Controller actions imply phases, but monitor output does not expose a first-class cluster timeline such as `safe -> preparing_kv_transfer -> flipping_role -> safe_new_role` or `safe_recovered`.
- Worker-local `FlipStateMachine` and controller two-phase migration are still parallel semantics. The controller is closer to the desired experiment owner, so worker FSM state should be treated as worker observability unless the implementation explicitly merges them.
- Post-commit failure compensation is under-specified. After target commit or source finish succeeds, a generic abort is no longer a complete rollback story; the controller should record partial-commit state and the next deterministic recovery action.
- Queue and load metrics are collected, but the monitor decision uses only SLO attainment. Queue length, running requests, token usage, and migration cost should feed source/target selection before claiming cost-aware state-machine behavior.

- [ ] **Step 1: Change D->P auto source selection to low-load source**

Add or update a controller test asserting D->P auto-selects the lowest-load decode source and the lowest-load remaining decode target.

Run:

```powershell
python3 -m unittest test.srt.test_pd_flip_controller -v
```

Expected before patch: the new source-selection test fails because controller currently passes `prefer_high_load=True`.

Patch `pd_flip_controller.py` D->P monitor/execute source selection to use low-load decode as source for the diagram-style experiment. Keep explicit `SOURCE_NAME=node2` working.

Expected after patch: controller tests pass and dry-run picks the low-load decode source when `SOURCE_NAME` is omitted.

- [ ] **Step 2: Connect P/D ratio evaluator to monitor mode**

Add a controller test asserting monitor can use `PDRatioSLOFlipEvaluator` or an equivalent controller-side policy object with enter, exit, and commit thresholds.

Run:

```powershell
python3 -m unittest test.srt.test_pd_flip_controller test.srt.test_pd_flip_state_machine -v
```

Expected before patch: monitor still uses `_prefill_risk()` and `_decode_risk()` threshold-only helpers.

Patch monitor mode so decision metadata records `current_pd_ratio`, `target_pd_ratio`, `enter_threshold`, `exit_threshold`, and `commit_threshold`.

Expected after patch: monitor output includes ratio metadata and tests pass.

- [ ] **Step 3: Make controller state explicit**

Add controller action records or monitor result fields for:

```text
safe
preparing_kv_transfer
flipping_role
safe_new_role
safe_recovered
failed_recovered
```

Expected: `monitor-commit.json` and `monitor-recovery.json` can be turned into a timeline without inferring state from HTTP action names.

- [ ] **Step 4: Harden post-commit compensation**

Add tests for failures after:

```text
target commit succeeds
source finish succeeds
runtime role set fails
router role refresh fails
resume admission fails
router undrain fails
```

Expected: controller result records the partial-commit state and a deterministic next recovery action. Do not silently call target/source abort after target commit if that can no longer restore the old topology.

### Task 8: Artifact Collection And Acceptance Report

**Files:**
- Create remotely: `/root/pd-flip-results/`
- Create locally: `C:/Users/Tianci J/Desktop/sglang/pd-flip-artifacts/`

- [ ] **Step 1: Collect logs from the cluster**

Run:

```powershell
cd "C:\Users\Tianci J\Desktop\sglang\scripts\playground\disaggregation\pd_flip_docker"
.\windows_four_node.ps1 -Action logs -RemoteRepo "/root/sglang" *> "C:\Users\Tianci J\Desktop\sglang\pd-flip-artifacts\cluster-logs.txt"
```

Expected: local `cluster-logs.txt` contains worker, router, and monitor tails.

- [ ] **Step 2: Package remote JSON results**

Run:

```powershell
ssh cloud-099 "mkdir -p /root/pd-flip-results && cp /root/sglang/scripts/playground/disaggregation/pd_flip_docker/*.json /root/pd-flip-results/ 2>/dev/null || true && tar -czf /root/pd-flip-results.tgz -C /root pd-flip-results"
scp cloud-099:/root/pd-flip-results.tgz "C:\Users\Tianci J\Desktop\sglang\pd-flip-artifacts\pd-flip-results.tgz"
```

Expected: local tarball exists and contains `metrics-before.json`, dry-run results, execute results, and monitor branch results.

- [ ] **Step 3: Write the acceptance summary**

Create `C:/Users/Tianci J/Desktop/sglang/pd-flip-artifacts/summary.md` with:

```markdown
# PD Flip Four-Node Experiment Summary

Date: 2026-06-30
Hosts: cloud-099, cloud-100, cloud-101, cloud-102
Model: /models/deepseek_v3.1_terminus
Image: sglang-pd-switch:tianciJ
Backend: mooncake
Initial topology: node0=prefill, node1=prefill, node2=decode, node3=decode

## Baseline
Router workers visible:
Worker role endpoints healthy:
Traffic through router:

## Manual D->P Execute
Source:
Target:
Success:
Migration seconds:
Total seconds:
Final topology:

## Monitor Recovery
Triggered preparing:
Aborted migration:
Final node2 role:

## Monitor Commit
Triggered preparing:
Committed migration:
Final node2 role:

## Remaining Gaps
Source selection:
P/D ratio curve:
Post-commit compensation:
Router exclusion by preparing/flipping state:
```

Expected: every field is filled with real values from logs and JSON results.

## Completion Definition

The experiment is complete when all of these are true:

- PowerShell can orchestrate all four Linux nodes through the SSH aliases.
- Docker workers start on all four nodes with 8 GPUs visible in each container.
- Router on `cloud-099` sees all four workers and correct initial roles.
- Controller `metrics` and D->P `dry-run` work from `cloud-099`.
- Manual D->P `execute` succeeds and produces 3P/1D topology.
- Monitor safe run does not flip without risky SLO samples.
- Monitor recovery branch records `preparing -> safe(decode)`.
- Monitor commit branch records `preparing -> flipping -> safe(prefill)`.
- Artifacts are saved under `C:/Users/Tianci J/Desktop/sglang/pd-flip-artifacts/`.
- Any remaining diagram-level gaps are explicitly listed rather than hidden inside a "successful" run.

## Self-Review

- Spec coverage: the plan covers the user-provided SSH aliases, Windows local orchestration, remote Linux command differences, Docker worker/router/controller startup, current state-machine gaps, monitor branches, and artifact capture.
- Placeholder scan: all commands use the current concrete paths, hosts, model path, image tag, ports, and node URLs from the workspace `env.local`.
- Type and name consistency: node names are `node0..node3`, SSH aliases are `cloud-099..cloud-102`, remote repo is `/root/sglang`, local repo is `C:/Users/Tianci J/Desktop/sglang`, and the controller source name for the primary D->P experiment is `node2`.
