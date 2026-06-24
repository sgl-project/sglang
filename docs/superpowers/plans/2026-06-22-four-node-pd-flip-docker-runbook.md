# Four Node PD Flip Docker Runbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to follow this runbook checkpoint-by-checkpoint. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the SGLang PD role flip experiment on four 8-GPU servers with Docker, starting from SSH access and ending with recorded D->P/P->D switch results.

**Architecture:** Four physical servers each run one SGLang worker container across all 8 local GPUs. Initial layout is two prefill workers and two decode workers; a separate router process routes traffic and a controller process drives drain, KV migration, role mutation, router refresh, and undrain.

**Tech Stack:** SSH, Docker with NVIDIA GPU runtime, SGLang official Docker image or offline image tarball, `scripts/playground/disaggregation/pd_flip_docker/*`, curl/jq, tmux, SGLang router and PD flip controller.

---

## How We Will Use This File

When you ask "下一步干嘛", paste either:

```text
我执行到 S2.3，输出如下：
...
```

or paste the command output directly. I will map the output to this runbook and tell you the next exact command.

Current known position as of 2026-06-22:

```text
S1.1-S1.3 completed on at least one server.
Docker is installed.
Each server reports 8 NVIDIA H20 GPUs.
Docker Hub pull timed out.
Next checkpoint: S2.1 docker images on all four servers.
```

## Server Naming

Use these names consistently in notes and shell history:

```text
node0: initial prefill
node1: initial prefill
node2: initial decode
node3: initial decode
router/controller: can be node0 or a fifth/login machine that can reach all NODE URLs
```

Record the real IPs before starting:

```text
node0_ip=<replace-with-node0-ip>
node1_ip=<replace-with-node1-ip>
node2_ip=<replace-with-node2-ip>
node3_ip=<replace-with-node3-ip>
router_ip=<replace-with-router-ip>
model_path=<same-model-path-on-all-worker-nodes>
```

## S0: Local Code Publish

- [ ] **S0.1 Check local branch state**

Run on your local development machine:

```bash
cd /home/tiancij/sglang
git status --short --branch
git log --oneline -n 3
```

Completion signal:

```text
Current branch is main.
Latest commit includes:
9238fe26f feat(pd-flip): execute docker role flip controller
```

- [ ] **S0.2 Push main if servers will pull from GitHub**

Run on your local development machine:

```bash
cd /home/tiancij/sglang
git push origin main
```

Completion signal:

```text
main is pushed to GitHub.
```

If servers cannot access GitHub, skip `git push` as an execution dependency and plan to copy the repository with `rsync` in S3.2.

## S1: SSH And Base Machine Check

- [ ] **S1.1 SSH into all four servers**

Open four terminals or four tmux panes from your local machine:

```bash
ssh root@<node0-ip>
ssh root@<node1-ip>
ssh root@<node2-ip>
ssh root@<node3-ip>
```

Completion signal:

```text
You have a shell prompt on all four servers.
```

- [ ] **S1.2 Record host IP and hostname on all four servers**

Run on each server:

```bash
hostname
hostname -I
```

Completion signal:

```text
You know which shell is node0/node1/node2/node3.
```

- [ ] **S1.3 Check Docker and GPU on all four servers**

Run on each server:

```bash
docker --version
nvidia-smi
```

Completion signal:

```text
Docker reports a version.
nvidia-smi reports 8 NVIDIA H20 GPUs.
No unexpected running GPU processes.
```

Known acceptable example:

```text
Docker version 26.1.3
8 x NVIDIA H20-3e
```

## S2: Docker Image Preparation

- [ ] **S2.1 Inspect existing images on all four servers**

Run on each server:

```bash
docker images
```

Completion signal:

```text
You know whether lmsysorg/sglang:latest already exists on every server.
```

If `lmsysorg/sglang:latest` exists on all four servers, go to S2.4.

- [ ] **S2.2 Create an offline SGLang image tarball if Docker Hub is blocked**

Run on a machine that can access Docker Hub:

```bash
docker pull lmsysorg/sglang:latest
docker save lmsysorg/sglang:latest -o sglang-latest.tar
ls -lh sglang-latest.tar
```

Completion signal:

```text
sglang-latest.tar exists.
```

- [ ] **S2.3 Copy and load the image on all four servers**

Run from the machine that has `sglang-latest.tar`:

```bash
scp sglang-latest.tar root@<node0-ip>:/root/
scp sglang-latest.tar root@<node1-ip>:/root/
scp sglang-latest.tar root@<node2-ip>:/root/
scp sglang-latest.tar root@<node3-ip>:/root/
```

Run on each server:

```bash
docker load -i /root/sglang-latest.tar
docker images | grep -E 'sglang|REPOSITORY'
```

Completion signal:

```text
lmsysorg/sglang latest appears in docker images on all four servers.
```

- [ ] **S2.4 Verify Docker can see GPUs inside the SGLang container**

Run on each server:

```bash
docker run --rm --gpus all lmsysorg/sglang:latest nvidia-smi
```

Completion signal:

```text
The container prints 8 NVIDIA H20 GPUs.
```

Stop condition:

```text
If this command fails before printing GPUs, paste the full output.
Do not continue to S3 until Docker GPU runtime works.
```

## S3: Code Synchronization On Servers

- [ ] **S3.1 Pull latest code from GitHub when servers have network access**

Run on each server:

```bash
mkdir -p /home/tiancij
cd /home/tiancij
if [ ! -d sglang ]; then
  git clone git@github.com:TianciJ/sglang.git
fi
cd /home/tiancij/sglang
git checkout main
git pull origin main
git rev-parse HEAD
```

Completion signal:

```text
HEAD starts with 9238fe26f.
```

- [ ] **S3.2 Copy repository manually if servers cannot access GitHub**

Run from your local development machine:

```bash
rsync -a --delete \
  --exclude .git \
  --exclude .venv \
  --exclude __pycache__ \
  /home/tiancij/sglang/ root@<node0-ip>:/home/tiancij/sglang/
rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ /home/tiancij/sglang/ root@<node1-ip>:/home/tiancij/sglang/
rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ /home/tiancij/sglang/ root@<node2-ip>:/home/tiancij/sglang/
rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ /home/tiancij/sglang/ root@<node3-ip>:/home/tiancij/sglang/
```

Completion signal:

```text
/home/tiancij/sglang exists on all four servers.
scripts/playground/disaggregation/pd_flip_docker/run_controller.sh contains execute).
```

Verify on each server:

```bash
grep -n 'execute)' /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker/run_controller.sh
```

## S4: Model Path Check

- [ ] **S4.1 Verify model path on all worker servers**

Run on each server:

```bash
ls -lah <model-path>
```

Completion signal:

```text
The same model directory exists on node0/node1/node2/node3.
```

If the model path is different across servers, create a consistent symlink on each server:

```bash
mkdir -p /models
ln -sfn <real-model-path> /models/pd-flip-model
ls -lah /models/pd-flip-model
```

Then use:

```text
MODEL_PATH=/models/pd-flip-model
MODEL_ID=pd-flip-model
TOKENIZER_PATH=/models/pd-flip-model
```

## S5: env.local Configuration

- [ ] **S5.1 Create env.local on all four servers**

Run on each server:

```bash
cd /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker
cp -n env.example env.local
```

- [ ] **S5.2 Edit env.local on all four servers**

Open:

```bash
vim /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker/env.local
```

Set these values, replacing IP and model path with your real values:

```bash
SGLANG_REPO=/home/tiancij/sglang
IMAGE=lmsysorg/sglang:latest

MODEL_PATH=/models/pd-flip-model
MODEL_ID=pd-flip-model
TOKENIZER_PATH=/models/pd-flip-model

TP_SIZE=8
DP_SIZE=1
PORT=30000
BOOTSTRAP_PORT=8998
MEM_FRACTION_STATIC=0.9

TRANSFER_BACKEND=fake
IB_DEVICE=mlx5_0

ROUTER_HOST=<router-ip>
ROUTER_PORT=8000
NODE0=http://<node0-ip>:30000
NODE1=http://<node1-ip>:30000
NODE2=http://<node2-ip>:30000
NODE3=http://<node3-ip>:30000

EXTRA_SGLANG_ARGS=
EXTRA_DOCKER_ARGS=
EXTRA_ROUTER_ARGS=
```

Completion signal:

```text
All four env.local files have identical NODE0-NODE3 and ROUTER_HOST values.
TRANSFER_BACKEND=fake for first flow validation.
```

- [ ] **S5.3 Source env.local and print critical values**

Run on each server:

```bash
cd /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker
export ENV_FILE=$PWD/env.local
source "$ENV_FILE"
printf 'IMAGE=%s\nMODEL_PATH=%s\nTP_SIZE=%s\nDP_SIZE=%s\nTRANSFER_BACKEND=%s\nNODE0=%s\nNODE1=%s\nNODE2=%s\nNODE3=%s\n' \
  "$IMAGE" "$MODEL_PATH" "$TP_SIZE" "$DP_SIZE" "$TRANSFER_BACKEND" "$NODE0" "$NODE1" "$NODE2" "$NODE3"
```

Completion signal:

```text
TP_SIZE=8
DP_SIZE=1
TRANSFER_BACKEND=fake
NODE URLs use the real server IPs.
```

## S6: Start Worker Containers

Use tmux or four persistent SSH terminals. Keep these processes running.

- [ ] **S6.1 Start node0 as prefill**

Run on node0:

```bash
cd /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker
export ENV_FILE=$PWD/env.local
./run_worker.sh prefill 0.0.0.0
```

- [ ] **S6.2 Start node1 as prefill**

Run on node1:

```bash
cd /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker
export ENV_FILE=$PWD/env.local
./run_worker.sh prefill 0.0.0.0
```

- [ ] **S6.3 Start node2 as decode**

Run on node2:

```bash
cd /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker
export ENV_FILE=$PWD/env.local
./run_worker.sh decode 0.0.0.0
```

- [ ] **S6.4 Start node3 as decode**

Run on node3:

```bash
cd /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker
export ENV_FILE=$PWD/env.local
./run_worker.sh decode 0.0.0.0
```

Completion signal:

```text
All four workers keep running.
No container exits immediately.
```

## S7: Worker Health And Role Check

- [ ] **S7.1 Check worker role endpoints from router/controller host**

Run on router/controller host:

```bash
curl -s http://<node0-ip>:30000/pd_flip/runtime_role/status | jq
curl -s http://<node1-ip>:30000/pd_flip/runtime_role/status | jq
curl -s http://<node2-ip>:30000/pd_flip/runtime_role/status | jq
curl -s http://<node3-ip>:30000/pd_flip/runtime_role/status | jq
```

Completion signal:

```text
node0 role=prefill
node1 role=prefill
node2 role=decode
node3 role=decode
runtime_role_switch_enabled=true
dual_queues_initialized=true
```

Stop condition:

```text
If curl cannot connect, check firewall, host networking, container logs, and PORT=30000.
Paste the curl error and the corresponding worker log.
```

## S8: Start Router

- [ ] **S8.1 Start router container**

Run on router host:

```bash
cd /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker
export ENV_FILE=$PWD/env.local
./run_router.sh
```

Completion signal:

```text
Router process keeps running on port 8000.
```

- [ ] **S8.2 Check router worker registry**

Run from another shell on router/controller host:

```bash
source /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker/env.local
curl -s "http://${ROUTER_HOST}:${ROUTER_PORT}/pd_flip/router/workers" | jq
```

Completion signal:

```text
Router sees four workers.
node0/node1 prefill.
node2/node3 decode.
draining=false for all workers.
```

## S9: Controller Metrics And Dry Run

- [ ] **S9.1 Collect metrics**

Run on controller host:

```bash
cd /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker
export ENV_FILE=$PWD/env.local
./run_controller.sh metrics | tee metrics-before.json
```

Completion signal:

```text
metrics-before.json contains four nodes with correct roles.
```

- [ ] **S9.2 Dry-run D->P switch**

Run on controller host:

```bash
DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh dry-run | tee d_to_p-dry-run.json
jq '.actions[].step' d_to_p-dry-run.json
```

Completion signal:

```text
Action list includes:
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

## S10: Basic Router Traffic

- [ ] **S10.1 Check model list through router**

Run on controller host:

```bash
source /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker/env.local
curl -s "http://${ROUTER_HOST}:${ROUTER_PORT}/v1/models" | jq
```

Completion signal:

```text
Router returns a model list or a valid JSON response.
```

- [ ] **S10.2 Send one streaming completion request**

Run on controller host, replacing `pd-flip-model` with `MODEL_ID` from env.local:

```bash
curl -N "http://${ROUTER_HOST}:${ROUTER_PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pd-flip-model",
    "prompt": "Write a long story about distributed inference.",
    "max_tokens": 512,
    "stream": true
  }'
```

Completion signal:

```text
Tokens stream back through the router.
```

## S11: Execute Fake-Backend D->P Switch

- [ ] **S11.1 Run D->P execute**

Run on controller host:

```bash
cd /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker
export ENV_FILE=$PWD/env.local
DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh execute | tee d_to_p-result.json
```

Completion signal:

```text
jq '.success' d_to_p-result.json returns true.
```

- [ ] **S11.2 Inspect D->P timing and roles**

Run on controller host:

```bash
jq '.success, .source, .migration_target, .migration_seconds, .total_seconds' d_to_p-result.json
./run_controller.sh metrics | tee metrics-after-d-to-p.json
```

Completion signal:

```text
source is node2.
node2 role is prefill after the switch.
success is true.
```

## S12: Execute Fake-Backend P->D Switch

- [ ] **S12.1 Dry-run P->D**

Run on controller host:

```bash
DIRECTION=p_to_d SOURCE_NAME=node0 ./run_controller.sh dry-run | tee p_to_d-dry-run.json
jq '.actions[].step' p_to_d-dry-run.json
```

Completion signal:

```text
Action list includes wait_source_idle and set_source_runtime_role.
No active decode migration steps are present.
```

- [ ] **S12.2 Run P->D execute**

Run on controller host:

```bash
DIRECTION=p_to_d SOURCE_NAME=node0 ./run_controller.sh execute | tee p_to_d-result.json
jq '.success, .source, .target_role, .total_seconds' p_to_d-result.json
```

Completion signal:

```text
success is true.
node0 role is decode after the switch.
```

## S13: Switch From Fake To Real KV Backend

Only start this after S11 and S12 pass with `TRANSFER_BACKEND=fake`.

- [ ] **S13.1 Stop workers and router**

Stop the four worker terminals and router terminal with Ctrl-C.

Completion signal:

```text
No old SGLang worker/router containers are running.
```

Check on each server:

```bash
docker ps
```

- [ ] **S13.2 Identify IB devices**

Run on each server:

```bash
ls /sys/class/infiniband
```

Completion signal:

```text
You know the actual IB device name, such as mlx5_0.
```

- [ ] **S13.3 Edit env.local for real backend on all four servers**

Set:

```bash
TRANSFER_BACKEND=mooncake
IB_DEVICE=<actual-ib-device-name>
```

Completion signal:

```text
All four env.local files use TRANSFER_BACKEND=mooncake and the same intended IB device.
```

- [ ] **S13.4 Repeat S6 through S12**

Run the same worker/router/controller sequence:

```text
S6 start workers
S7 check worker roles
S8 start router
S9 metrics and dry-run
S10 traffic
S11 D->P execute
S12 P->D execute
```

Completion signal:

```text
mooncake D->P result has success=true.
migration_seconds is recorded.
```

## S14: Result Collection

- [ ] **S14.1 Save result artifacts**

On controller host:

```bash
mkdir -p /home/tiancij/pd-flip-results
cp /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker/*result*.json /home/tiancij/pd-flip-results/ 2>/dev/null || true
cp /home/tiancij/sglang/scripts/playground/disaggregation/pd_flip_docker/*metrics*.json /home/tiancij/pd-flip-results/ 2>/dev/null || true
ls -lah /home/tiancij/pd-flip-results
```

- [ ] **S14.2 Record summary**

Create a text summary:

```bash
cat > /home/tiancij/pd-flip-results/summary.txt <<'EOF'
date:
node0:
node1:
node2:
node3:
router:
model:
backend:
d_to_p_source:
d_to_p_target:
d_to_p_success:
d_to_p_migration_seconds:
d_to_p_total_seconds:
p_to_d_source:
p_to_d_success:
p_to_d_total_seconds:
traffic_errors:
notes:
EOF
vim /home/tiancij/pd-flip-results/summary.txt
```

Completion signal:

```text
summary.txt has real values filled in.
JSON result files are saved.
```

## Troubleshooting Checkpoints

### Docker Pull Timeout

Observed error:

```text
Client.Timeout exceeded while awaiting headers
```

Next action:

```text
Use S2.2-S2.3 offline docker save/load.
```

### Container Cannot See GPUs

Failing command:

```bash
docker run --rm --gpus all lmsysorg/sglang:latest nvidia-smi
```

Next action:

```bash
docker info | grep -i runtime
nvidia-container-cli --version || true
```

Paste both outputs.

### Worker Port Unreachable

Failing command:

```bash
curl -s http://<node-ip>:30000/pd_flip/runtime_role/status
```

Next action on that node:

```bash
docker ps
ss -lntp | grep 30000 || true
```

Paste worker terminal logs and these command outputs.

### Router Cannot See Workers

Failing command:

```bash
curl -s "http://${ROUTER_HOST}:${ROUTER_PORT}/pd_flip/router/workers" | jq
```

Next action:

```bash
curl -s http://<node0-ip>:30000/pd_flip/runtime_role/status | jq
curl -s http://<node1-ip>:30000/pd_flip/runtime_role/status | jq
curl -s http://<node2-ip>:30000/pd_flip/runtime_role/status | jq
curl -s http://<node3-ip>:30000/pd_flip/runtime_role/status | jq
```

Paste router logs and these outputs.

### D->P Execute Fails

Next action:

```bash
jq '.success, .message, .actions[] | {step, success, message}' d_to_p-result.json
```

Paste the JSON output plus source and target worker logs.
