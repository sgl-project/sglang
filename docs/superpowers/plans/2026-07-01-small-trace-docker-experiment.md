# Small Trace Docker Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a small timestamped trace replay through the four-node PD flip Docker harness and verify that router traffic, monitor FSM, and D->P KV migration form a working link.

**Architecture:** Use the existing four-node Docker harness and run all server/client work in Docker. Generate a small `autobench` JSONL trace with OpenAI chat messages, output lengths, and millisecond timestamps; replay it with `python3 -m sglang.bench_serving --backend sglang-oai-chat --dataset-name autobench --use-trace-timestamps` against the router. Trigger one monitor iteration with intentionally strict TTFT SLO so the controller enters `safe -> preparing_kv_transfer -> flipping_role -> safe`, then verify client success, monitor state trace, router roles, logs, and cleanup.

**Tech Stack:** PowerShell orchestration, SSH, Docker, existing `pd_flip_docker` scripts, `sglang.bench_serving`, JSONL artifacts, `jq`/Python JSON parsing.

---

### Task 1: Safety Preflight

**Files:**
- No production file changes.

- [ ] **Step 1: Check four hosts without changing state**

Run from Windows:

```powershell
foreach ($h in @("cloud-099","cloud-100","cloud-101","cloud-102")) {
  @"
echo $h
ss -ltnp | grep -E ':31000|:18000|:18998' || true
pgrep -af '[s]glang.launch_server|[s]gl-router|[r]un_worker.sh|[r]un_router.sh|[p]d-node|[p]d-router|[p]d-monitor|[c]url.*v1/chat/completions' || true
docker ps --format '{{.Names}} {{.Image}}' | grep -E 'pd-node|pd-router|pd-monitor|sglang-pd-switch' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
"@ | ssh $h bash -s
}
```

Expected: each host prints only its hostname, with no matching listener/process/container/GPU compute rows.

- [ ] **Step 2: Verify remote monitor code has FSM fields**

Run:

```powershell
ssh cloud-099 "cd /root/sglang && grep -n 'state_trace\|class MonitorState\|_execute_p_to_d_monitor' scripts/playground/disaggregation/pd_flip_controller.py | head -20"
```

Expected: output includes `state_trace`, `class MonitorState`, and `_execute_p_to_d_monitor`.

### Task 2: Prepare Trace Input

**Files:**
- Create remote artifact: `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/traces/autobench_small_trace_32.jsonl`

- [ ] **Step 1: Generate a small timestamped OpenAI chat trace**

Run:

```powershell
@'
cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker
mkdir -p traces artifacts/trace-smoke
python3 - <<'PY'
import json
from pathlib import Path

path = Path("traces/autobench_small_trace_32.jsonl")
rows = []
for i in range(32):
    timestamp_ms = i * 250
    prompt = (
        "Trace smoke request %02d. Summarize this synthetic incident timeline "
        "in a concise paragraph and keep the answer deterministic. " % i
    )
    prompt += " ".join([f"event_{i}_{j}" for j in range(80)])
    rows.append(
        {
            "timestamp": timestamp_ms,
            "messages": [{"role": "user", "content": prompt}],
            "output_len": 96,
        }
    )

with path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

print(path)
print(len(rows))
PY
wc -l traces/autobench_small_trace_32.jsonl
head -n 1 traces/autobench_small_trace_32.jsonl
'@ | ssh cloud-099 bash -s
```

Expected: `wc -l` reports `32`; first JSON line contains `timestamp`, `messages`, and `output_len`.

### Task 3: Start Four-Node Docker Harness

**Files:**
- Remote logs under `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/*.log`

- [ ] **Step 1: Start workers**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Tianci J\Desktop\sglang\scripts\playground\disaggregation\pd_flip_docker\windows_four_node.ps1" -Action start-workers -RemoteRepo "/root/sglang"
```

Expected: tmux sessions `pd-node0` through `pd-node3` start on the four hosts.

- [ ] **Step 2: Wait for worker health**

Run:

```powershell
@'
set -e
for url in http://192.168.0.42:31000 http://192.168.0.40:31000 http://192.168.0.39:31000 http://192.168.0.41:31000; do
  echo "waiting $url"
  for i in $(seq 1 180); do
    if curl -fsS "$url/health" >/dev/null; then echo "$url healthy"; break; fi
    sleep 2
    if [ "$i" = "180" ]; then echo "$url not healthy"; exit 1; fi
  done
done
'@ | ssh cloud-099 bash -s
```

Expected: all four worker URLs report healthy.

- [ ] **Step 3: Start router**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Tianci J\Desktop\sglang\scripts\playground\disaggregation\pd_flip_docker\windows_four_node.ps1" -Action start-router -RemoteRepo "/root/sglang"
```

Expected: router listens on `127.0.0.1:18000` on `cloud-099`.

- [ ] **Step 4: Verify router sees all workers**

Run:

```powershell
ssh cloud-099 "curl -fsS http://127.0.0.1:18000/v1/models >/tmp/pd_trace_models.json && python3 -m json.tool /tmp/pd_trace_models.json | head -40"
ssh cloud-099 "curl -fsS http://127.0.0.1:18000/pd_flip/router/workers >/tmp/pd_trace_workers.json && python3 -m json.tool /tmp/pd_trace_workers.json | head -120"
```

Expected: `/v1/models` succeeds, and router workers include two prefill and two decode nodes.

### Task 4: Run Trace Replay And Monitor

**Files:**
- Remote artifacts under `/root/sglang/scripts/playground/disaggregation/pd_flip_docker/artifacts/trace-smoke/`

- [ ] **Step 1: Launch trace replay in a one-shot Docker client**

Run:

```powershell
@'
cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker
docker run --rm --name pd-trace-client-$(date +%s) \
  --network host \
  -v /root/sglang:/workspace \
  -w /workspace \
  -e PYTHONPATH=/workspace/python:/workspace \
  sglang-pd-switch:dev \
  python3 -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --base-url http://127.0.0.1:18000 \
    --dataset-name autobench \
    --dataset-path /workspace/scripts/playground/disaggregation/pd_flip_docker/traces/autobench_small_trace_32.jsonl \
    --num-prompts 32 \
    --use-trace-timestamps \
    --model default \
    --tokenizer "$MODEL_PATH" \
    --warmup-requests 1 \
    --max-concurrency 8 \
    --output-file /workspace/scripts/playground/disaggregation/pd_flip_docker/artifacts/trace-smoke/bench_trace_output.jsonl \
    --output-details \
  > artifacts/trace-smoke/bench_trace.log 2>&1 &
echo $! > artifacts/trace-smoke/bench_trace.pid
cat artifacts/trace-smoke/bench_trace.pid
'@ | ssh cloud-099 bash -s
```

Expected: a background client PID is written; the one-shot Docker container performs trace replay and exits by itself.

- [ ] **Step 2: Trigger monitor once while trace is active**

Run:

```powershell
@'
cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker
cp env.local artifacts/trace-smoke/env.monitor-trace
cat >> artifacts/trace-smoke/env.monitor-trace <<'EOF'
TTFT_SLO_SECONDS=0.000001
TPOT_SLO_SECONDS=0.02
PD_FLIP_WINDOW_SECONDS=30
PD_FLIP_ENTER_THRESHOLD=0.9
PD_FLIP_EXIT_THRESHOLD=0.95
PD_FLIP_COMMIT_THRESHOLD=0.9
PD_FLIP_MONITOR_ITERATIONS=1
PD_FLIP_MONITOR_POLL_INTERVAL=0
EOF
ENV_FILE=artifacts/trace-smoke/env.monitor-trace ./run_controller.sh monitor \
  > artifacts/trace-smoke/monitor_trace_result.json 2>&1
tail -80 artifacts/trace-smoke/monitor_trace_result.json
'@ | ssh cloud-099 bash -s
```

Expected: monitor prints JSON with `success: true` and `state_trace` ending in `safe`.

- [ ] **Step 3: Wait for trace replay completion**

Run:

```powershell
@'
cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker
pid=$(cat artifacts/trace-smoke/bench_trace.pid)
for i in $(seq 1 180); do
  if ! kill -0 "$pid" 2>/dev/null; then echo "bench completed"; exit 0; fi
  sleep 2
done
echo "bench still running after timeout"
tail -120 artifacts/trace-smoke/bench_trace.log
exit 1
'@ | ssh cloud-099 bash -s
```

Expected: `bench completed` within the timeout.

### Task 5: Verify Link And Clean Up

**Files:**
- Local artifacts copied to `C:\Users\Tianci J\Desktop\sglang\pd-flip-artifacts\trace-smoke\`

- [ ] **Step 1: Parse monitor and benchmark artifacts**

Run:

```powershell
@'
cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker
python3 - <<'PY'
import json
from pathlib import Path

base = Path("artifacts/trace-smoke")
text = (base / "monitor_trace_result.json").read_text()
start = text.find("{")
monitor = json.loads(text[start:])
print("monitor_success", monitor.get("success"))
print("monitor_message", monitor.get("message"))
print("monitor_states", " -> ".join(x["state"] for x in monitor.get("state_trace", [])))
print("monitor_actions", len(monitor.get("actions", [])))

bench = base / "bench_trace_output.jsonl"
rows = [json.loads(line) for line in bench.read_text().splitlines() if line.strip()]
print("bench_rows", len(rows))
print("bench_success_rows", sum(1 for r in rows if r.get("success")))
print("bench_error_rows", sum(1 for r in rows if not r.get("success")))
PY
'@ | ssh cloud-099 bash -s
```

Expected: monitor success is true; states include `preparing_kv_transfer` and `flipping_role`; benchmark has 32 rows and zero error rows.

- [ ] **Step 2: Check logs for critical errors**

Run:

```powershell
@'
cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker
grep -E 'pool memory leak|req_to_token_pool|Traceback|ERROR|Exception' router.log worker.log artifacts/trace-smoke/bench_trace.log || true
for h in cloud-100 cloud-101 cloud-102; do
  echo "== $h =="
  ssh "$h" "cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker && grep -E 'pool memory leak|req_to_token_pool|Traceback|ERROR|Exception' worker.log || true"
done
'@ | ssh cloud-099 bash -s
```

Expected: no critical pool leak or traceback rows attributable to this run.

- [ ] **Step 3: Copy artifacts locally**

Run:

```powershell
$dir = "C:\Users\Tianci J\Desktop\sglang\pd-flip-artifacts\trace-smoke"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
scp -r cloud-099:/root/sglang/scripts/playground/disaggregation/pd_flip_docker/artifacts/trace-smoke/* "$dir\"
scp cloud-099:/root/sglang/scripts/playground/disaggregation/pd_flip_docker/router.log "$dir\router.log"
scp cloud-099:/root/sglang/scripts/playground/disaggregation/pd_flip_docker/worker.log "$dir\worker-cloud-099.log"
scp cloud-100:/root/sglang/scripts/playground/disaggregation/pd_flip_docker/worker.log "$dir\worker-cloud-100.log"
scp cloud-101:/root/sglang/scripts/playground/disaggregation/pd_flip_docker/worker.log "$dir\worker-cloud-101.log"
scp cloud-102:/root/sglang/scripts/playground/disaggregation/pd_flip_docker/worker.log "$dir\worker-cloud-102.log"
```

Expected: artifacts are present locally.

- [ ] **Step 4: Stop only this harness and verify cleanup**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Tianci J\Desktop\sglang\scripts\playground\disaggregation\pd_flip_docker\windows_four_node.ps1" -Action stop -RemoteRepo "/root/sglang"

foreach ($h in @("cloud-099","cloud-100","cloud-101","cloud-102")) {
  @"
echo $h
ss -ltnp | grep -E ':31000|:18000|:18998' || true
pgrep -af '[s]glang.launch_server|[s]gl-router|[r]un_worker.sh|[r]un_router.sh|[p]d-node|[p]d-router|[p]d-monitor|[c]url.*v1/chat/completions' || true
docker ps --format '{{.Names}} {{.Image}}' | grep -E 'pd-node|pd-router|pd-monitor|pd-trace-client|sglang-pd-switch' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
"@ | ssh $h bash -s
}
```

Expected: no matching listeners, processes, containers, or GPU compute rows remain.
