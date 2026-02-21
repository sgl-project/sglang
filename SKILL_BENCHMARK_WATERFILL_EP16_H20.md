# Skill: EP16 Waterfill Benchmark on H20 Cluster (10.6.131.5/6)

This skill defines the EP16 benchmark procedure for the **waterfill** optimization on DeepSeek-V3, running on the 2-node H20 cluster with shared Lustre storage.

---

## Environment

| Item | Value |
|------|-------|
| Cluster | 2x H20-3e nodes (8x H20 per node), 400Gbps RoCE |
| Node IPs | `10.6.131.5` (node 0), `10.6.131.6` (node 1) |
| Container | `sglang_lb` (image: `lmsysorg/sglang:v0.5.5.post3`, with upgraded packages — see "Container Setup" section) |
| Storage | **Shared Lustre** — `/lustre/raplab/client` mounted in all containers, no rsync needed |
| Code Path | `/lustre/raplab/client/xutingz/workspace/gitsrc/sglang` (branch: `feat/deepep-waterfill-eplb-balance`) |
| Baseline Repo | `/lustre/raplab/client/xutingz/workspace/gitsrc/sglang_baseline_98a107d` |
| Model Path | `/lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3` |
| Bench / EPLB Dir | `/lustre/raplab/client/xutingz/workspace/bench/waterfill` |
| Torch Profile Dir | `/lustre/raplab/client/xutingz/workspace/bench/waterfill/torch_profile` |
| PyTorch | 2.9.1+cu129 (upgraded from 2.8.0 in base image) |
| sgl-kernel | 0.3.21 (upgraded from 0.3.17.post1 in base image) |
| flashinfer | 0.5.3 (upgraded from 0.5.2 in base image) |
| torchvision | 0.24.1+cu129 (upgraded from 0.23.0 in base image) |
| deep_ep | Custom build for PyTorch 2.9.1 (see "Container Setup") |
| nvshmem | 3.4.5 (source build at `/sgl-workspace/nvshmem/install/` in v0.5.5.post3 image) |
| Launch Wrapper | `/lustre/raplab/client/xutingz/workspace/bench/waterfill/launch_sglang.sh` (sets `ulimit -l unlimited`) |

> **Note**: `/home/xutingz` and `/lustre/raplab/client/xutingz` are the same path on the host, but **only** `/lustre/raplab/client/...` is mounted inside the container. Always use the full Lustre path in container commands.

---

## EP16 Configuration

| Parameter | Value |
|-----------|-------|
| TP | 16 |
| DP | 16 (dp_attention) |
| nnodes | 2 |
| MoE A2A Backend | deepep |
| DeepEP Mode | normal |
| CUDA Graph | Disabled (waterfill incompatible with graph capture) |

---

## Prerequisites

### 1. Enter Container (on node 0)

```bash
ssh 10.6.131.5
docker exec -it sglang_lb bash
```

### 2. Install sglang from Lustre (editable, inside container)

```bash
cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang
pip install -e "python[dev]" --no-deps -q
```

Verify:
```bash
python3 -c "import sglang; print(sglang.__version__)"
```

### 3. Verify Both Nodes Can Access Shared Storage

```bash
# From node 0 container:
ssh -o StrictHostKeyChecking=no 10.6.131.6 \
  "docker exec sglang_lb ls /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3/config.json"
```

### 4. Clean Stale Processes (both nodes)

```bash
for ip in 10.6.131.5 10.6.131.6; do
  ssh -o StrictHostKeyChecking=no $ip \
    "docker exec sglang_lb bash -c 'pkill -9 -f sglang 2>/dev/null; rm -f /dev/shm/nccl* /dev/shm/nvshmem* 2>/dev/null'"
done
```

---

## Part 1: Performance Benchmark (bench_one_batch)

### Using the Automated Multi-Node Script

The script `bench_waterfill_multinode.py` handles server launch/teardown on both nodes automatically.

**Before running**, the script's hardcoded `NODE_IPS` and `MODEL_PATH` must match this cluster. If they don't, override by editing locally or use the manual method below.

#### Step 1: Baseline vs Waterfill (no EPLB file needed)

```bash
docker exec sglang_lb bash -c '
  export SGLANG_LOG_MS=1
  python3 /lustre/raplab/client/xutingz/workspace/gitsrc/sglang/benchmark/deepseek_v3/bench_waterfill_multinode.py \
    --ep 16 \
    --modes baseline,waterfill \
    --out-dir /lustre/raplab/client/xutingz/workspace/bench/waterfill
'
```

#### Step 2: Generate EPLB File (first time only)

Check if the EPLB file already exists:
```bash
ls /lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_logical_count.pt
```

**If the file does NOT exist**, you must generate it before running EPLB modes. See "Generating EPLB Distribution File" section below.

**If the file exists**, skip to Step 3.

#### Step 3: EPLB vs EPLB+Waterfill (requires EPLB file from Step 2)

```bash
docker exec sglang_lb bash -c '
  export SGLANG_LOG_MS=1
  python3 /lustre/raplab/client/xutingz/workspace/gitsrc/sglang/benchmark/deepseek_v3/bench_waterfill_multinode.py \
    --ep 16 \
    --modes eplb,eplb_waterfill \
    --init-expert-location /lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_logical_count.pt \
    --out-dir /lustre/raplab/client/xutingz/workspace/bench/waterfill
'
```

#### All 4 Modes at Once (requires EPLB file from Step 2)

```bash
docker exec sglang_lb bash -c '
  export SGLANG_LOG_MS=1
  python3 /lustre/raplab/client/xutingz/workspace/gitsrc/sglang/benchmark/deepseek_v3/bench_waterfill_multinode.py \
    --ep 16 \
    --modes baseline,waterfill,eplb,eplb_waterfill \
    --init-expert-location /lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_logical_count.pt \
    --out-dir /lustre/raplab/client/xutingz/workspace/bench/waterfill
'
```

### Manual Method (Separate Server + Client)

This gives full control and access to individual server logs.

#### Launch Server (from inside container on node 0)

**Baseline (no waterfill):**
```bash
cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang_baseline_98a107d
pip install -e "python[dev]" --no-deps -q

export SGLANG_LOG_MS=1

# Node 1 (run on 10.6.131.6):
ssh -o StrictHostKeyChecking=no 10.6.131.6 "docker exec sglang_lb bash -c '
  export SGLANG_LOG_MS=1 &&
  cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang_baseline_98a107d &&
  pip install -e python[dev] --no-deps -q &&
  python3 -m sglang.launch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --trust-remote-code --host 0.0.0.0 --port 30000 \
    --tp 16 --dp-size 16 --enable-dp-attention \
    --moe-a2a-backend deepep --deepep-mode normal \
    --chunked-prefill-size -1 --disable-radix-cache \
    --max-prefill-tokens 8192 --max-running-requests 2048 \
    --load-balance-method round_robin --log-level info \
    --watchdog-timeout 600 --mem-fraction-static 0.75 \
    --skip-server-warmup --disable-cuda-graph \
    --dist-init-addr 10.6.131.5:20000 --nnodes 2 --node-rank 1
'" &

sleep 5

# Node 0 (local):
python3 -m sglang.launch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --trust-remote-code --host 0.0.0.0 --port 30000 \
    --tp 16 --dp-size 16 --enable-dp-attention \
    --moe-a2a-backend deepep --deepep-mode normal \
    --chunked-prefill-size -1 --disable-radix-cache \
    --max-prefill-tokens 8192 --max-running-requests 2048 \
    --load-balance-method round_robin --log-level info \
    --watchdog-timeout 600 --mem-fraction-static 0.75 \
    --skip-server-warmup --disable-cuda-graph \
    --dist-init-addr 10.6.131.5:20000 --nnodes 2 --node-rank 0 \
    2>&1 | tee server_baseline.log &
```

**Optimized (with waterfill):**
```bash
cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang
pip install -e "python[dev]" --no-deps -q

# Same as above but add: --enable-deepep-waterfill
# And optionally: --init-expert-location /lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_logical_count.pt
```

#### Run Bench Client (after server is ready)

```bash
CUDA_VISIBLE_DEVICES=99 python3 -m sglang.bench_one_batch_server \
    --model None \
    --base-url http://10.6.131.5:30000 \
    --batch-size 2048 \
    --input-len 1024 \
    --output-len 1 \
    --dataset-name random \
    --result-filename result_baseline.jsonl \
    --no-append-to-github-summary
```

> **Note**: `--batch-size 2048` is the **global** batch size (= local_bs 128 * dp_size 16). Adjust as needed.

#### Kill Server (after benchmark)

```bash
for ip in 10.6.131.5 10.6.131.6; do
  ssh -o StrictHostKeyChecking=no $ip \
    "docker exec sglang_lb bash -c 'pkill -9 -f sglang.launch_server 2>/dev/null; pkill -9 -f \"sglang::\" 2>/dev/null; rm -f /dev/shm/nccl* /dev/shm/nvshmem* 2>/dev/null'"
done
```

### Benchmark Cases

All cases use `output_len=1` and `deepep_mode=normal`. Batch size is **per DP rank**; the automated script scales to global (local_bs * 16).

> **Important**: `output_len=1` is required for waterfill benchmarking. Waterfill is a prefill-phase optimization. The primary metric is `input_throughput` (tok/s). `output_throughput` values with `output_len=1` are meaningless (inflated by near-zero decode time).

| Name | local_bs | global_bs | input_len | output_len |
|------|----------|-----------|-----------|------------|
| bs128_il512 | 128 | 2048 | 512 | 1 |
| bs64_il1024 | 64 | 1024 | 1024 | 1 |
| bs32_il2048 | 32 | 512 | 2048 | 1 |
| bs16_il4096 | 16 | 256 | 4096 | 1 |

### What to Check in Results

- `input_throughput` (tok/s) — prefill throughput
- `output_throughput` (tok/s) — decode throughput
- `latency` (s) — total latency
- `last_ttft` (s) — time to first token
- `last_gen_throughput` (tok/s) — decode gen throughput from server log

---

## Part 2: Torch Profile Trace

Launch server (baseline or optimized) as in Part 1, then:

```bash
CUDA_VISIBLE_DEVICES=99 python3 -m sglang.bench_one_batch_server \
    --model None \
    --base-url http://10.6.131.5:30000 \
    --batch-size 2048 \
    --input-len 1024 \
    --output-len 1 \
    --seed 1 \
    --profile \
    --profile-by-stage \
    --profile-steps 5 \
    --profile-prefix baseline- \
    --profile-output-dir /lustre/raplab/client/xutingz/workspace/bench/waterfill/torch_profile \
    --result-filename profile_result_baseline.jsonl \
    --no-append-to-github-summary
```

For optimized, change `--profile-prefix optimized-`.

### Trace Output

```
{profile-output-dir}/{timestamp}/
  server_args.json
  {prefix}bs-2048-il-1024-{ts}-TP-{i}-EP-{i}-EXTEND.trace.json.gz   # per-rank prefill
  {prefix}bs-2048-il-1024-{ts}-TP-{i}-EP-{i}-DECODE.trace.json.gz   # per-rank decode
  merged-{prefix}bs-2048-il-1024-{ts}-EXTEND.trace.json.gz           # all ranks merged
  merged-{prefix}bs-2048-il-1024-{ts}-DECODE.trace.json.gz           # all ranks merged
```

View merged files in Chrome `chrome://tracing` or Perfetto.

---

## Part 3: Accuracy Testing (MMLU)

Launch server, then:

```bash
python3 -m sglang.test.run_eval \
    --base-url http://10.6.131.5:30000 \
    --eval-name mmlu \
    --num-examples 64 \
    --num-threads 512
```

Expected DeepSeek-V3 score: ~0.90+. Baseline and optimized should be within <1% of each other.

---

## Part 4: Using the E2E Script

The all-in-one script automates baseline vs. waterfill comparison using two repos:

```bash
cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang

python3 benchmark/deepseek_v3/run_deepep_waterfill_e2e_test.py \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --tp 8 --ep 8 \
    --baseline-sglang-dir /lustre/raplab/client/xutingz/workspace/gitsrc/sglang_baseline_98a107d \
    --waterfill-sglang-dir /lustre/raplab/client/xutingz/workspace/gitsrc/sglang \
    --docker-container sglang_lb \
    --run-one-batch \
    --one-batch-num-prompts 256 \
    --one-batch-input-len 1024 \
    --one-batch-output-len 1 \
    --skip-accuracy \
    --skip-serving
```

> **Note**: The e2e script uses `--tp 8 --ep 8` for single-node EP8 comparison. For multi-node EP16, use `bench_waterfill_multinode.py` instead.

---

## Generating EPLB Distribution File (Required Before EPLB Modes)

First check if the file already exists:
```bash
ls /lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_logical_count.pt
```

If it exists, skip this section entirely. If not, follow the steps below to generate it.

### 1. Launch EP16 Server with Expert Distribution Recorder

On **both nodes** (inside `sglang_lb` container):

```bash
cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang
pip install -e "python[dev]" --no-deps -q

export SGLANG_LOG_MS=1
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/lustre/raplab/client/xutingz/workspace/bench/waterfill

python3 -m sglang.launch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --trust-remote-code --host 0.0.0.0 --port 30000 \
    --tp 16 --dp-size 16 --enable-dp-attention \
    --moe-a2a-backend deepep --deepep-mode normal \
    --chunked-prefill-size -1 --disable-radix-cache \
    --max-prefill-tokens 8192 --max-running-requests 128 \
    --load-balance-method round_robin --log-level info \
    --watchdog-timeout 600 --disable-cuda-graph --skip-server-warmup \
    --expert-distribution-recorder-mode stat \
    --expert-distribution-recorder-buffer-size 1000 \
    --dist-init-addr 10.6.131.5:20000 --nnodes 2 \
    --node-rank <0|1>
```

### 2. Record Expert Distribution (from node 0)

```bash
# Start recording
curl -X POST http://10.6.131.5:30000/start_expert_distribution_record

# Generate load
CUDA_VISIBLE_DEVICES=99 python3 -m sglang.bench_one_batch_server \
    --model None --base-url http://10.6.131.5:30000 \
    --batch-size 128 --input-len 1024 --output-len 10 \
    --dataset-name random --skip-warmup

# Stop and dump
curl -X POST http://10.6.131.5:30000/stop_expert_distribution_record
curl -X POST http://10.6.131.5:30000/dump_expert_distribution_record
```

### 3. Rename

```bash
mv /lustre/raplab/client/xutingz/workspace/bench/waterfill/expert_distribution_recorder_*.pt \
   /lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_logical_count.pt
```

No need to copy to other nodes — shared Lustre storage.

### 4. Kill Server

```bash
for ip in 10.6.131.5 10.6.131.6; do
  ssh -o StrictHostKeyChecking=no $ip \
    "docker exec sglang_lb bash -c 'pkill -9 -f sglang 2>/dev/null; rm -f /dev/shm/nccl* /dev/shm/nvshmem* 2>/dev/null'"
done
```

---

## Adapting bench_waterfill_multinode.py for This Cluster

The script has hardcoded values that may need updating. Check these constants at the top of `benchmark/deepseek_v3/bench_waterfill_multinode.py`:

```python
NODE_IPS = {
    16: ["10.6.131.5", "10.6.131.6"],
}
MODEL_PATH = "/lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3"
CONTAINER = "sglang_lb"
```

Also verify `env_vars` in `launch_server()` — should NOT set `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` (keep the default of 1 to avoid NVSHMEM bootstrap failures):

```python
env_vars = (
    "export SGLANG_LOG_MS=1; "
    "export NCCL_DEBUG=WARN; "
    "export SGLANG_DEBUG_WATERFILL_EPLB=1; "
    "export SGLANG_DEBUG_WATERFILL_EPLB_LAYER=all; "
    "export SGLANG_DEBUG_WATERFILL_EPLB_MAX_PRINTS=1; "
    "export SGLANG_DEBUG_WATERFILL_EPLB_MIN_TOKENS=64; "
)
```

---

## Known Issues & Solutions

### 1. CUDA graph disabled
Waterfill mode cannot use CUDA graph (DeepEP `Buffer.sync()` fails during graph capture). Disabled for all modes for fair comparison.

### 2. First forward pass slow (~40s)
DeepEP buffer initialization (NVSHMEM bootstrap, RDMA setup) happens on first forward. The `wait_server()` uses 1800s timeout.

### 3. Stale shared memory
After killing a server, always clean up: `rm -f /dev/shm/nccl* /dev/shm/nvshmem*` on all nodes.

### 4. `pkill -f sglang` self-kill
The benchmark script path contains "sglang". Use specific patterns like `sglang.launch_server`, `sglang::scheduler` to avoid killing the script itself.

### 5. Container sglang version
The container ships sglang 0.5.6 system-wide. After `pip install -e`, the editable install takes precedence. Verify with `python3 -c "import sglang; print(sglang.__file__)"` — should point to Lustre path.

### 6. CRITICAL: DeepGEMM JIT Cache — Pre-Warm + Precompile Required

DeepGEMM JIT-compiles ~385 GEMM kernels on the first server run and caches them at `/root/.cache/deep_gemm/cache/`. This cache is **per-node** (not shared).

**Problem 1 — Sequential bias**: When running multiple modes sequentially, the first mode bears all JIT compilation overhead (~190s), while the second mode reuses the disk cache (~80s). This makes the second mode appear ~2x faster.

**Problem 2 — NVSHMEM IBGDA timeout**: Setting `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` disables startup precompilation, which causes DeepGEMM to JIT-compile on the first forward pass. During the first forward, different ranks compile different kernels at different speeds, causing rank desynchronization during NVSHMEM bootstrap. This produces errors like:
```
socketStartConnect: exceeded retries (20000)
nvshmem setup connections failed
alltoall of rc failed
```

**Solution — Three-step approach**:
1. **Keep `SGLANG_JIT_DEEPGEMM_PRECOMPILE=1` (the default)**. Do NOT set it to 0. The precompile runs during model initialization (before NVSHMEM bootstrap), so all ranks synchronize properly.
2. **Pre-warm the JIT cache** on all nodes by running a baseline server + one warmup request before real benchmarks. The `bench_waterfill_multinode.py` script does this automatically in its "JIT CACHE PRE-WARM" phase.
3. **Sync JIT caches across nodes** if one node has more cached kernels than the other:
   ```bash
   # Copy from node with more kernels to shared filesystem
   docker exec sglang_lb bash -c 'cp -r /root/.cache/deep_gemm/cache/* /lustre/raplab/client/xutingz/workspace/bench/waterfill/deepgemm_cache/'
   # On other node(s), copy from shared filesystem
   ssh xutingz@10.6.131.6 "docker exec sglang_lb bash -c 'cp -rn /lustre/raplab/client/xutingz/workspace/bench/waterfill/deepgemm_cache/* /root/.cache/deep_gemm/cache/'"
   ```

**Historical note**: Earlier skill versions recommended `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` because precompile=1 caused CUDA errors. This was actually a misdiagnosis — the CUDA errors were caused by other issues. With populated JIT caches, precompile=1 simply validates the cache (~2-3s per kernel type) and does not cause issues.

### 7. CRITICAL: NVSHMEM IBGDA Bootstrap Failures on EP16

**Symptom**: Server fails to start or crashes on first forward pass with:
```
socketStartConnect: exceeded retries (20000)
nvshmem setup connections failed
alltoall of rc failed
```
Or on the remote node:
```
NULL value Unable to create ah.
create DCT share err.
connect EPS failed
```

**Root cause (IDENTIFIED 2026-02-17)**: NVSHMEM's UID bootstrap uses NCCL-derived TCP socket code to establish initial connections between nodes. By default, NVSHMEM scans available network interfaces and may pick an IB RoCE management interface (e.g., `ens1130f0np0` at `172.18.0.11/31`) instead of the management network (`bond0` at `10.6.131.x/24`). The IB RoCE interfaces on this cluster use `/31` subnets with point-to-point links that don't support arbitrary TCP connections between nodes, causing the bootstrap to timeout.

**The fix — Set `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME`**:
```bash
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0   # CRITICAL: force bootstrap over management network
export NCCL_SOCKET_IFNAME=bond0                    # Best practice: keep NCCL on same interface
```

These env vars MUST be set in ALL server launch commands on ALL nodes. The env var is confirmed in the NVSHMEM 3.4.5 source code at:
```
src/modules/bootstrap/common/env_defs.h:  NVSHMEMI_ENV_DEF(BOOTSTRAP_UID_SOCK_IFNAME, ...)
src/modules/bootstrap/uid/ncclSocket/ncclsocket_socket.cpp:  "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME set by environment to %s"
```

**Network topology context (H20-GPU-05/06 cluster)**:
- `bond0` (10.6.131.x/24): Management network — nodes can reach each other via TCP. Use for bootstrap.
- `ens1130f0np0` (172.18.0.x/31): IB RoCE interface — point-to-point, NOT suitable for TCP bootstrap.
- `ens1131f0np0`, `ens1033f0np0`, etc. (172.18.{32,64,96,128,160,192,224}.x/31): More IB RoCE interfaces.
- `docker0` (172.17.0.1/16): Docker bridge — NOT suitable for inter-node communication.

**How to diagnose on a new cluster**: If NVSHMEM bootstrap fails:
1. Check the error log for the IP it's trying to connect to
2. Run `ip addr show` inside the container to identify which interface owns that IP
3. Set `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME` to the interface that has inter-node TCP connectivity (usually the management/bond interface)

**Other contributing factors (still relevant)**:
1. **JIT cache synchronization**: If ranks are stalled by JIT compilation during NVSHMEM init, the bootstrap can timeout even on the correct interface. Keep `SGLANG_JIT_DEEPGEMM_PRECOMPILE=1` (default).
2. **Stale shared memory**: Always clean `/dev/shm/nvshmem*` between server runs.
3. **Port reuse**: Use a different `--dist-init-addr` port for each launch attempt to avoid stale TCP state.

**What does NOT fix it**:
- `NVSHMEM_REMOTE_TRANSPORT=ibrc` — different transport, still has bootstrap timeout issues
- `--skip-server-warmup` alone — bypasses the crash but costs ~33% throughput (no DeepGEMM warmup)
- Reverting code changes — the issue is a network interface selection problem, not a code bug

### 8. pip install can break package versions
Running `pip install -e "python[dev]"` (without `--no-deps`) may downgrade critical packages. **Always use `--no-deps`** to avoid this:
```bash
pip install -e '/lustre/raplab/client/xutingz/workspace/gitsrc/sglang/python[dev]' --no-deps
```

If you accidentally ran without `--no-deps`, re-run the container package upgrade procedure (see "Container Setup" section).

### 9. Container /dev/shm size
Docker containers default to 64MB or 1GB shm. NCCL with 16 GPUs needs ~32GB. Ensure containers are created with `--shm-size=32g`. Check with `df -h /dev/shm`.

### 10. EP8 waterfill CUDA crash (FIXED)
On EP8, `--enable-deepep-waterfill` used to trigger `CUDA_ERROR_ILLEGAL_ADDRESS`. Root cause: in the `num_tokens == 0` early-return path, `self.topk.empty_topk_output(device)` generated 8-column topk tensors, but waterfill mode expects 9 columns (8 routed + 1 shared). **Fix applied** in `deepseek_v2.py` (~line 1667): replaced `empty_topk_output()` with explicit 9-column tensor construction.

### 11. EP8 waterfill+EPLB is structurally unviable

**Conclusion**: Waterfill cannot produce positive throughput gain on EP8+EPLB. This is a structural limitation, not a tuning issue.

**Analysis**:
- Waterfill's fixed overhead (lost alt_stream overlap + extra AllReduce for global routed counts) costs ~5-6% throughput
- The imbalance improvement from waterfill is only ~2% (1.112 → 1.091 max/mean ratio), yielding ~1.3% throughput benefit
- Net result: -5.5% to -6.6% throughput regression
- EP8 has only 8 ranks, so the "thundering herd" effect is weaker and EPLB already achieves near-optimal balance

**Implication**: Waterfill+EPLB optimization efforts should focus exclusively on EP16+ where cross-node communication benefits and higher rank count create more room for improvement.

### 12. "Thundering Herd" in Waterfill Shared Dispatch

**Root cause of waterfill+EPLB underperformance on EP16**: All source ranks independently pick the same argmin destination rank for shared tokens, because they all see the same global routed_counts. When EPLB has already balanced routed load, the routed_counts are nearly uniform, so a small perturbation makes ALL ranks converge on the same "least loaded" rank — amplifying imbalance by ~world_size.

**Fix 1 — Adaptive threshold** (`adaptive_k_threshold=1.15`): Skip waterfill redistribution entirely for layers where `max(routed_counts)/mean(routed_counts) < 1.15`. These layers are already well-balanced by EPLB, and waterfill redistribution only adds overhead.

**Fix 2 — nnodes-scaled local preference** (`local_preference_factor = 1.0 + 0.2 * nnodes`): Penalize cross-node dispatch more aggressively on multi-node setups. EP16 (2 nodes) uses factor 1.4 instead of the previous fixed 1.2.

**Fix 3 (REJECTED) — Per-token Triton kernel branching**: Added a "close-enough" fallback in the Triton waterfill kernel (5% tolerance). Worsened throughput from -1.2% to -5.5% due to branch divergence overhead in the GPU kernel. **Reverted**.

### 13. CRITICAL: NVSHMEM IBGDA Transport — Docker memlock Limit

**Symptom**: NVSHMEM fails intermittently with `nvshmem setup connections failed` or `alltoall of rc failed` on multi-node EP16, even with JIT cache pre-warmed and `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`.

**Root cause**: Docker containers default to `ulimit -l 64` (64KB locked memory limit). NVSHMEM IBGDA transport requires unlimited locked memory for RDMA pinned buffers. When the limit is too low, IBGDA transport initialization fails non-deterministically.

**Solution — Wrapper script** (`/lustre/raplab/client/xutingz/workspace/bench/waterfill/launch_sglang.sh`):
```bash
#!/bin/bash
ulimit -l unlimited
ulimit -l  # print to verify
exec python3 "$@"
```

**Usage**: Replace `python3` with the wrapper script path in all server launch commands:
```bash
# Instead of:
python3 -m sglang.launch_server ...
# Use:
/lustre/raplab/client/xutingz/workspace/bench/waterfill/launch_sglang.sh -m sglang.launch_server ...
```

**Important notes**:
- The wrapper MUST be used for ALL multi-node EP16 launches (both baseline and waterfill)
- Even with the wrapper, NVSHMEM is intermittent — may need 2-3 launch attempts
- Use a DIFFERENT `--dist-init-addr` port each attempt (stale TCP state causes failures)
- Always kill + clean between attempts: `tmux kill-server; pkill -9 -f sglang; pkill -9 -f python3; rm -f /dev/shm/*`
- Wait 15-20s between kill and relaunch
- Launch node 1 (worker) first, wait 10s, then node 0 (master)

**Verification**: In the server log, look for `ulimit -l: unlimited` printed by the wrapper.

### 14. Debug Environment Variables for Imbalance Logging

To observe per-layer imbalance scores during benchmarking:
```bash
export SGLANG_DEBUG_WATERFILL_EPLB=1
export SGLANG_DEBUG_WATERFILL_EPLB_LAYER=all      # or specific layer ID
export SGLANG_DEBUG_WATERFILL_EPLB_MAX_PRINTS=1    # prints per layer
export SGLANG_DEBUG_WATERFILL_EPLB_MIN_TOKENS=64   # skip small batches
```

Output format in server log:
```
[deepep_eplb_load] mode=waterfill layer=3 ...
  pre_eplb   total=[...] max/mean=1.23 std/mean=0.15
  post_eplb  total=[...] max/mean=1.08 std/mean=0.05
  post_waterfill total=[...] max/mean=1.05 std/mean=0.03
```

The `bench_waterfill_multinode.py` script sets these automatically for all server launches.

### 15. CRITICAL: Container Image Selection — v0.5.5.post3, NOT v0.5.6

**The v0.5.5.post3 image is required** because it contains a source-built NVSHMEM at `/sgl-workspace/nvshmem/install/` that supports IBGDA transport. The pip-installed NVSHMEM (in v0.5.6 and other images) does NOT support IBGDA.

**Key discovery**: Only source-built NVSHMEM works for IBGDA on this cluster. The source build is at `/sgl-workspace/nvshmem/install/lib/libnvshmem.so` inside the v0.5.5.post3 image. You MUST set `LD_LIBRARY_PATH=/sgl-workspace/nvshmem/install/lib:$LD_LIBRARY_PATH` to override the pip-installed NVSHMEM.

**However**, the v0.5.5.post3 image ships PyTorch 2.8.0, which is too old for the current sglang code. Multiple packages need upgrading — see "Container Setup" section below.

### 16. NVSHMEM IBGDA Crash After Container Restart (Transient)

**Symptom**: After `docker restart sglang_lb`, the server fails on the first launch attempt with:
```
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.8/main_nvshmem/src/modules/transport/ibgda/ibgda.cpp:2174: NULL value Unable to create ah.
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.8/main_nvshmem/src/modules/transport/ibgda/ibgda.cpp:2916: non-zero status: 7 create DCT share err.
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.8/main_nvshmem/src/host/transport/transport.cpp:420: non-zero status: 7 connect EPS failed
nvshmem initialization failed, exiting
Scheduler or DataParallelController terminated with 255
```

**Root cause**: After a container restart, IB RoCE resources (address handles, DC transport objects) are in a transient state. The first NVSHMEM IBGDA init attempt immediately after restart fails.

**Solution — Restart, wait, retry**:
1. `docker restart sglang_lb` on both nodes
2. Wait ~10 seconds for IB subsystem to stabilize
3. Launch the server — if it fails with the above error, wait 30s and try again with a new `--dist-init-addr` port
4. Usually the **second attempt** succeeds

**This is different from Known Issue #7** (bootstrap interface selection). Issue #7 is caused by NVSHMEM picking the wrong network interface for TCP bootstrap. This issue (#16) is a transient IB resource initialization failure after container restart. Both `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0` and a fresh retry are needed.

### 17. CRITICAL: Baseline Must Use --init-expert-location for Fair A/B Comparison

**Symptom**: Waterfill shows +6% to +10% gain over baseline, far above the expected ~3-4%.

**Root cause**: The baseline was launched WITHOUT `--init-expert-location`, so it used trivial (round-robin) expert dispatch. Trivial dispatch is inherently ~1000 tok/s slower than EPLB static dispatch because experts are randomly placed across GPUs without any load-aware optimization. This artificially inflates the waterfill gain.

**The correct A/B comparison**:
- **Waterfill**: `--enable-deepep-waterfill --init-expert-location .../ep16_mmlu_logical_count.pt`
- **Baseline**: `--init-expert-location .../ep16_mmlu_logical_count.pt` (same EPLB file, NO waterfill flag)

The ONLY difference should be `--enable-deepep-waterfill`. Both must use EPLB.

**Verification**: Check the server log for `init_expert_location from init_by_eplb using ServerArgs.init_expert_location` in the startup output. If this line is missing from the baseline, the comparison is unfair.

**Historical proof**: The Feb 12 A/B test (`ep16_mmlu_ab_3rounds_20260213/`) used `--init-expert-location` for BOTH baseline and waterfill (verified from server logs), giving the correct +3-4% gain. The Feb 18 incorrect test omitted it from baseline, giving an inflated +9.6%.

| Test | Baseline Dispatch | Waterfill Dispatch | Baseline tput | Waterfill tput | Gain |
|------|-------------------|-------------------|---------------|----------------|------|
| Feb 12 (correct) | EPLB | EPLB + waterfill | 29,326 | 30,469 | +3.9% |
| Feb 18 (WRONG) | Trivial | EPLB + waterfill | 28,263 | 30,979 | +9.6% |
| Feb 18 (corrected) | EPLB | EPLB + waterfill | 29,745 | 30,979 | +4.1% |

---

## NVSHMEM Troubleshooting Runbook (Complete)

This section documents the full NVSHMEM IBGDA fix process discovered on 2026-02-17/18. Follow this when NVSHMEM fails on this cluster.

### Step 1: Identify the Failure Type

Check the server log for NVSHMEM errors. There are 3 failure types:

**Type A — Bootstrap Interface Wrong (Known Issue #7)**:
```
socketStartConnect: exceeded retries (20000)
nvshmem setup connections failed
```
Fix: `export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0`

**Type B — IBGDA Transport Init Failure (Known Issue #16)**:
```
NULL value Unable to create ah.
create DCT share err.
connect EPS failed
nvshmem initialization failed, exiting
```
Fix: Restart containers, wait 10s, retry with new port.

**Type C — Bootstrap Message Truncation**:
```
Message truncated : received 112 bytes instead of 40
allgather of ipc handles failed
```
Fix: Usually follows Type B. Fix Type B first (restart + retry).

### Step 2: Ensure Correct Environment Variables

ALL server launch commands must include:
```bash
export LD_LIBRARY_PATH=/sgl-workspace/nvshmem/install/lib:$LD_LIBRARY_PATH   # Source-built NVSHMEM
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0                                # Management network
export NCCL_SOCKET_IFNAME=bond0                                                # NCCL also management
ulimit -l unlimited                                                            # Unlimited locked memory
```

### Step 3: Full Recovery After Container Restart

When containers are restarted (`docker restart sglang_lb`), ALL package upgrades are preserved (installed to `/usr/local/lib/python3.12/dist-packages/` which persists across restarts), but:
- DeepGEMM JIT cache at `/root/.cache/deep_gemm/cache/` may be lost
- IB RoCE resources need time to stabilize

Recovery steps:
```bash
# 1. Verify packages are still there
docker exec sglang_lb python3 -c "import torch; print(torch.__version__); import sgl_kernel; import flashinfer; import deep_ep"

# 2. Restore DeepGEMM cache (if lost)
docker exec sglang_lb bash -c 'mkdir -p /root/.cache/deep_gemm/cache && cp -r /lustre/raplab/client/xutingz/workspace/bench/waterfill/deepgemm_cache/* /root/.cache/deep_gemm/cache/'

# 3. Re-install sglang (editable install uses symlink, should survive restart)
docker exec sglang_lb python3 -c "import sglang; print(sglang.__file__)"
# If it doesn't point to Lustre path, re-install:
docker exec sglang_lb pip install -e '/lustre/raplab/client/xutingz/workspace/gitsrc/sglang/python[dev]' --no-deps

# 4. Wait 10s before launching server
sleep 10
```

### Step 4: Zombie Process Handling

`pkill -9 -f sglang` often leaves zombie detokenizer/scheduler processes that hold ports and `/dev/shm`. When `ps aux | grep python3 | wc -l` shows processes after pkill:

```bash
# Nuclear option: restart container
docker restart sglang_lb
# Then re-run Step 3 above
```

**Port increment rule**: After each failed launch attempt, increment the `--dist-init-addr` port by 2 (e.g., 20042→20044→20046). Stale TCP state on old ports causes failures even after process cleanup.

### Step 5: The Complete Launch Sequence

```bash
# 1. Clean state
docker exec sglang_lb bash -c 'pkill -9 -f sglang; pkill -9 -f python3; rm -f /dev/shm/*'
ssh 10.6.131.5 "docker exec sglang_lb bash -c 'pkill -9 -f sglang; pkill -9 -f python3; rm -f /dev/shm/*'"

# 2. Check for zombies (should be 0)
docker exec sglang_lb bash -c 'ps aux | grep -E "sglang|python3" | grep -v grep | wc -l'
# If > 0: docker restart sglang_lb on affected node, then re-install sglang

# 3. Launch Node 0 first
ssh 10.6.131.5 "docker exec -d sglang_lb bash -c '...PORT=20050...'"

# 4. Launch Node 1 immediately after (within seconds)
docker exec -d sglang_lb bash -c '...PORT=20050...'

# 5. Wait ~3 min for model load + warmup
sleep 180

# 6. Check health
docker exec sglang_lb bash -c 'curl -s -o /dev/null -w "%{http_code}" http://localhost:30000/health'
# Should return 200

# 7. If health check fails, check log for error type (Step 1) and act accordingly
```

---

## Container Setup (Full Procedure)

This section documents how to create and configure the containers from scratch. All steps must be done on BOTH nodes.

### Step 1: Create Containers

```bash
# On EACH node (10.6.131.5 and 10.6.131.6):
docker run -d --name sglang_lb --gpus all --privileged --network=host --ipc=host \
  --shm-size 32g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /lustre/raplab/client/xutingz/workspace:/lustre/raplab/client/xutingz/workspace \
  lmsysorg/sglang:v0.5.5.post3 sleep infinity
```

**Critical flags**:
- `--ulimit memlock=-1`: Required for NVSHMEM IBGDA RDMA pinned buffers
- `--privileged`: Required for IB device access
- `--network=host`: Required for inter-node communication
- `--shm-size 32g`: NCCL with 16 GPUs needs ~32GB shared memory

### Step 2: Upgrade PyTorch (2.8.0 → 2.9.1)

```bash
docker exec sglang_lb bash -c '
  pip install torch==2.9.1+cu129 --index-url https://download.pytorch.org/whl/cu129
'
```

### Step 3: Upgrade ABI-Incompatible Packages

PyTorch 2.9.1 breaks ABI compatibility with packages compiled against 2.8.0. The following must be upgraded:

```bash
docker exec sglang_lb bash -c '
  # sgl-kernel: undefined symbol errors without upgrade
  pip install --upgrade sgl-kernel

  # flashinfer: segfault on import without upgrade
  pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3

  # torchvision: std::bad_alloc on import without upgrade
  pip install torchvision==0.24.1+cu129 --index-url https://download.pytorch.org/whl/cu129
'
```

### Step 4: Replace deep_ep with PyTorch 2.9.1-Compatible Version

The v0.5.5.post3 image's `deep_ep_cpp.so` was compiled against PyTorch 2.8.0. Replace it:

```bash
docker exec sglang_lb bash -c '
  # Replace the .so file
  cp /lustre/raplab/client/xutingz/workspace/bench/waterfill/deep_ep_291/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so \
     /usr/local/lib/python3.12/dist-packages/

  # Replace the Python package
  rm -rf /usr/local/lib/python3.12/dist-packages/deep_ep
  cp -r /lustre/raplab/client/xutingz/workspace/bench/waterfill/deep_ep_291/deep_ep \
     /usr/local/lib/python3.12/dist-packages/
'
```

### Step 5: Restore DeepGEMM JIT Cache

DeepGEMM has ~385 JIT-compiled kernel directories. Without the cache, first server startup takes ~190s extra. The cache is lost on container restart.

```bash
docker exec sglang_lb bash -c '
  mkdir -p /root/.cache/deep_gemm/cache
  cp -r /lustre/raplab/client/xutingz/workspace/bench/waterfill/deepgemm_cache/* /root/.cache/deep_gemm/cache/
'
```

### Step 6: Verify Environment

```bash
docker exec sglang_lb bash -c '
  python3 -c "
import torch; print(f\"PyTorch: {torch.__version__}\")
import sgl_kernel; print(f\"sgl-kernel OK\")
import flashinfer; print(f\"flashinfer OK\")
import torchvision; print(f\"torchvision OK\")
import deep_ep; print(f\"deep_ep OK\")
"
  # Verify NVSHMEM source build exists
  ls -la /sgl-workspace/nvshmem/install/lib/libnvshmem.so
'
```

Expected output:
```
PyTorch: 2.9.1+cu129
sgl-kernel OK
flashinfer OK
torchvision OK
deep_ep OK
```

### Post-Container-Restart Recovery

If the container is restarted (`docker restart sglang_lb`), Steps 2-5 are lost. Re-run them. A one-liner:

```bash
docker exec sglang_lb bash -c '
  pip install torch==2.9.1+cu129 --index-url https://download.pytorch.org/whl/cu129 &&
  pip install --upgrade sgl-kernel &&
  pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 &&
  pip install torchvision==0.24.1+cu129 --index-url https://download.pytorch.org/whl/cu129 &&
  cp /lustre/raplab/client/xutingz/workspace/bench/waterfill/deep_ep_291/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so /usr/local/lib/python3.12/dist-packages/ &&
  rm -rf /usr/local/lib/python3.12/dist-packages/deep_ep &&
  cp -r /lustre/raplab/client/xutingz/workspace/bench/waterfill/deep_ep_291/deep_ep /usr/local/lib/python3.12/dist-packages/ &&
  mkdir -p /root/.cache/deep_gemm/cache &&
  cp -r /lustre/raplab/client/xutingz/workspace/bench/waterfill/deepgemm_cache/* /root/.cache/deep_gemm/cache/
'
```

---

## Server Launch Commands (Canonical)

All launch commands MUST include the NVSHMEM env vars. Run from the **host machine** (not inside container).

### Required Environment Variables

```bash
# These MUST be set in ALL launch commands:
export LD_LIBRARY_PATH=/sgl-workspace/nvshmem/install/lib:$LD_LIBRARY_PATH   # Use source-built NVSHMEM
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0                                # Bootstrap over management network
export NCCL_SOCKET_IFNAME=bond0                                                # NCCL also over management network
```

### Waterfill Server Launch

```bash
# Node 0 (10.6.131.5):
ssh 10.6.131.5 "docker exec -d sglang_lb bash -c 'export LD_LIBRARY_PATH=/sgl-workspace/nvshmem/install/lib:\$LD_LIBRARY_PATH && export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 && export NCCL_SOCKET_IFNAME=bond0 && ulimit -l unlimited && python3 -m sglang.launch_server --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 --tp 16 --dp-size 16 --nnodes 2 --node-rank 0 --dist-init-addr 10.6.131.5:<PORT> --host 0.0.0.0 --port 30000 --trust-remote-code --moe-a2a-backend deepep --deepep-mode normal --enable-dp-attention --mem-fraction-static 0.75 --max-running-requests 2048 --watchdog-timeout 1800 --disable-radix-cache --disable-cuda-graph --chunked-prefill-size -1 --max-prefill-tokens 8192 --enable-deepep-waterfill --init-expert-location /lustre/raplab/client/xutingz/workspace/bench/waterfill/mmlu_expert_dist/ep16_mmlu_logical_count.pt >/lustre/raplab/client/xutingz/workspace/bench/waterfill/waterfill_node0.log 2>&1'"

# Node 1 (10.6.131.6):
docker exec -d sglang_lb bash -c 'export LD_LIBRARY_PATH=/sgl-workspace/nvshmem/install/lib:$LD_LIBRARY_PATH && export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 && export NCCL_SOCKET_IFNAME=bond0 && ulimit -l unlimited && python3 -m sglang.launch_server --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 --tp 16 --dp-size 16 --nnodes 2 --node-rank 1 --dist-init-addr 10.6.131.5:<PORT> --host 0.0.0.0 --port 30000 --trust-remote-code --moe-a2a-backend deepep --deepep-mode normal --enable-dp-attention --mem-fraction-static 0.75 --max-running-requests 2048 --watchdog-timeout 1800 --disable-radix-cache --disable-cuda-graph --chunked-prefill-size -1 --max-prefill-tokens 8192 --enable-deepep-waterfill --init-expert-location /lustre/raplab/client/xutingz/workspace/bench/waterfill/mmlu_expert_dist/ep16_mmlu_logical_count.pt >/lustre/raplab/client/xutingz/workspace/bench/waterfill/waterfill_node1.log 2>&1'
```

> **Note**: Replace `<PORT>` with a unique port for each launch attempt (e.g., 20020, 20022, 20024...). Reusing ports from a previous crashed run can cause failures.

### Baseline Server Launch (MUST ALSO USE --init-expert-location)

**CRITICAL**: The baseline MUST also use `--init-expert-location` for a fair comparison! The only difference between baseline and waterfill should be `--enable-deepep-waterfill`. Without `--init-expert-location`, baseline uses trivial (round-robin) expert dispatch which is ~1000 tok/s slower than EPLB dispatch, artificially inflating the waterfill gain from ~4% to ~10%.

Same as waterfill but **without** `--enable-deepep-waterfill`. Keep `--init-expert-location`.

### Benchmark Command

```bash
docker exec sglang_lb bash -c 'export LD_LIBRARY_PATH=/sgl-workspace/nvshmem/install/lib:$LD_LIBRARY_PATH && CUDA_VISIBLE_DEVICES=99 python3 /lustre/raplab/client/xutingz/workspace/bench/waterfill/tput_bench.py {waterfill|baseline} 4 8'
```

### Kill + Clean Procedure

```bash
# On both nodes:
docker exec sglang_lb bash -c 'pkill -9 -f sglang; pkill -9 -f python3; rm -f /dev/shm/*'
# Or from host for Node 0:
ssh 10.6.131.5 "docker exec sglang_lb bash -c 'pkill -9 -f sglang; pkill -9 -f python3; rm -f /dev/shm/*'"
```

> **Tip**: If zombie processes persist after kill, `docker restart sglang_lb` and then re-run the container recovery procedure.

---

## Key Files

| File | Purpose |
|------|---------|
| `benchmark/deepseek_v3/bench_waterfill_multinode.py` | Multi-node EP16 automated benchmark |
| `benchmark/deepseek_v3/run_deepep_waterfill_e2e_test.py` | Single-node e2e regression test |
| `python/sglang/bench_one_batch_server.py` | Single-batch latency/throughput benchmark |
| `python/sglang/srt/managers/scheduler_profiler_mixin.py` | Server-side profiler |
| `python/sglang/srt/utils/profile_merger.py` | Multi-rank trace merging |
| `python/sglang/test/run_eval.py` | MMLU/GSM8K evaluation |

---

## Background Execution (Recommended)

```bash
ssh 10.6.131.5 "nohup docker exec sglang_lb bash -c '
  export SGLANG_LOG_MS=1 &&
  cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang &&
  pip install -e python[dev] --no-deps -q &&
  python3 benchmark/deepseek_v3/bench_waterfill_multinode.py \
    --ep 16 \
    --modes baseline,waterfill \
    --out-dir /lustre/raplab/client/xutingz/workspace/bench/waterfill
' > /lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_run.log 2>&1 &"

# Monitor:
tail -f /lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_run.log
```

---

## Waterfill+EPLB Optimization (EP16)

### Problem Statement

With EPLB enabled, waterfill's throughput gain shrinks from +3-4% to -1% (regression). The root cause is the "thundering herd" effect (see Known Issues #12).

### Applied Fixes (in `deepseek_v2.py` and `deepep_waterfill.py`)

**Fix 1 — Adaptive threshold** (`adaptive_k_threshold=1.15`):
- Location: `DeepEPWaterfillBalancer.__init__()` and `prepare_dispatch()`
- Behavior: Before running waterfill, check `max(routed_counts) / mean(routed_counts)`. If < 1.15, the layer is already well-balanced by EPLB → skip waterfill and do all-local shared dispatch
- Effect: Eliminates waterfill overhead on ~50% of layers that are already balanced

**Fix 2 — nnodes-scaled local preference** (`local_preference_factor = 1.0 + 0.2 * nnodes`):
- Location: `DeepseekV2MoE.__init_deepep_waterfill()`
- Behavior: EP8 (1 node) → factor 1.2, EP16 (2 nodes) → factor 1.4
- Effect: Stronger bias toward local shared dispatch on multi-node, reducing cross-node communication

**Fix 3 (REJECTED) — Triton kernel close-enough fallback**:
- Added per-token branching in the Triton waterfill kernel: if remote count is within 5% of local, choose local
- Result: Worsened throughput from -1.2% to -5.5% due to branch divergence overhead
- **Reverted**: Per-token branching in Triton kernels is too expensive

### Tuning Parameters

| Parameter | Location | Default | EPLB | Description |
|-----------|----------|---------|------|-------------|
| `local_preference_factor` | `deepseek_v2.py` | 1.0 | 1.0 + 0.2*nnodes | Penalty multiplier for remote dispatch |
| `enable_sampling` | `deepseek_v2.py` | True | False | Disable random sampling under EPLB |
| `adaptive_k_threshold` | `deepseek_v2.py` | 0.0 | 1.15 | Skip waterfill if max/mean < threshold |

### Next Steps if Current Fixes Don't Produce Gain

1. Raise `adaptive_k_threshold` to 1.20 (more aggressive skip)
2. Conditional alt_stream/DeepEP routing per-token (avoid overhead for tokens that stay local)
3. Overlap the AllReduce with gate computation (pipeline the global counts)
4. Consider waterfill only on layers with highest imbalance (top 25%)

---

## Waterfill V2: Post-TopK Routed Expert Rebalance (EP16+EPLB)

### Problem with V1

V1 (original waterfill) serialized the shared expert into the MoE dispatch, losing alt_stream parallelism (~2% overhead). This structural overhead exceeded any benefit from better load balancing when EPLB was already active.

**V1 EP16 results**: -2.3% to -0.2% regression vs EPLB-only.

### V2 Approach

V2 keeps the shared expert on alt_stream (free parallelism), keeps the original 8-column dispatch, and adds a **post-topk routed expert swap** using local load counts. This has zero structural overhead.

**Key design**:
1. After topk selection, compute per-rank routed load using `torch.bincount` (local only, no AllReduce)
2. Check imbalance: if `max_load / mean_load < threshold` (default 1.05), skip rebalancing
3. Identify overloaded ranks (`load > mean * 1.02`)
4. For affected tokens on overloaded ranks: find the weakest expert (lowest router logit)
5. Mask router_logits: `-inf` for already-selected and overloaded-rank experts
6. Pick best alternative (highest logit on an underloaded rank)
7. Convert logical→physical expert IDs and apply the swap in topk_ids

### Activation

V2 is gated by environment variable only (no CLI flag needed):
```bash
export SGLANG_WATERFILL_V2=1
```

Optional threshold tuning:
```bash
export SGLANG_WATERFILL_V2_THRESHOLD=1.05  # default; lower = more aggressive rebalancing
```

### Implementation Files

| File | Change |
|------|--------|
| `python/sglang/srt/models/deepseek_v2.py` | V2 init logic (~line 645), `_rebalance_routed_topk()` method (~line 1349), hook in `forward_deepep` (~line 1553) |
| `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` | V2 env var check to skip V1 weight-loader adjustment |
| `benchmark/deepseek_v3/bench_waterfill_multinode.py` | `eplb_waterfill_v2` mode support |

### V2 Benchmark Results (2026-02-12, EP16, 2 nodes)

All runs with `--disable-cuda-graph`, `output_len=1`, `deepep_mode=normal`, `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`.

#### Input Throughput (tok/s) — Primary Metric

| Case | EPLB (baseline) | EPLB+V2 | Gain |
|------|-----------------|---------|------|
| bs128_il512 | 38,681 | 39,044 | **+0.94%** |
| bs64_il1024 | 38,158 | 38,279 | **+0.32%** |
| bs32_il2048 | 36,014 | 36,167 | **+0.43%** |
| bs16_il4096 | 32,074 | 32,475 | **+1.25%** |

#### All EP16 Results (Complete History)

| Case | Baseline | Waterfill | EPLB | EPLB+V1 | EPLB+V2 |
|------|----------|-----------|------|---------|---------|
| bs128_il512 | 35,357 | 36,615 | 38,681 | 37,723 (-2.3%) | 39,044 (+0.94%) |
| bs64_il1024 | 33,780 | 35,360 | 38,158 | 37,232 (-2.0%) | 38,279 (+0.32%) |
| bs32_il2048 | 31,790 | 33,071 | 36,014 | 35,387 (-1.9%) | 36,167 (+0.43%) |
| bs16_il4096 | 28,538 | 29,578 | 32,074 | 31,860 (-0.2%) | 32,475 (+1.25%) |

### Key Takeaways

1. **V2 achieves positive gain** in all 4 cases (+0.32% to +1.25%), while V1 was negative (-2.3% to -0.2%)
2. **Largest gain at bs16_il4096** (+1.25%): Higher per-token compute means rebalancing overhead is proportionally smaller
3. **Zero structural overhead**: No alt_stream serialization, no extra AllReduce
4. **Trade-off**: `.item()` calls in rebalancing prevent CUDA graph capture; OK since `--disable-cuda-graph` is already required for DeepEP

### Result Files

- EPLB baseline: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_v2_manual/ep16/eplb/results/`
- EPLB+V2: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_v2_manual/ep16/eplb_waterfill_v2/results/`
- V2 server logs: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_v2_manual/ep16/eplb_waterfill_v2/logs/`

---

## MMLU Throughput Benchmark Results

Benchmark using `tput_bench.py` with 14042 MMLU prompts, `max_tokens=1`, 4 warmup rounds + 8 measurement rounds. Full warmup (no `--skip-server-warmup`). Container: v0.5.5.post3 with upgraded packages. `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0` set.

### Corrected Results (2026-02-18) — Fair A/B Comparison

**CRITICAL LESSON**: Both waterfill AND baseline must use `--init-expert-location` (EPLB). Without it, baseline uses trivial expert dispatch (~28.3k tok/s), which is ~1k tok/s slower than EPLB dispatch (~29.7k), artificially inflating waterfill gain from ~4% to ~10%.

| Config | Trimmed Mean | All Rounds | Min | Max |
|--------|-------------|------------|-----|-----|
| **Waterfill Static** (EPLB + waterfill) | **30,979** | 30730, 31494, 31423, 30817, 30731, 31265, 30907, 30395 | 30395 | 31494 |
| **Baseline EPLB** (EPLB only, no waterfill) | **29,745** | 28828, 29977, 29873, 29528, 30288, 30543, 29093, 29714 | 28828 | 30543 |
| **Static Gain** | **+4.1%** ✅ | Matches Feb 12 historical ~3-4% | | |
| | | | | |
| **Waterfill Dynamic** (waterfill, no EPLB) | **29,241** | 29165, 28009, 29665, 29031, 29501, 29233, 29731, 28848 | 28009 | 29731 |
| **Baseline Trivial** (no EPLB, no waterfill) | **28,530** | 28482, 28667, 28335, 28617, 28212, 27850, 28866, 29176 | 27850 | 29176 |
| **Dynamic Gain** | **+2.5%** | | | |

### A/B Benchmark Methodology (MUST FOLLOW)

1. **Waterfill** uses waterfill worktree + `--enable-deepep-waterfill --init-expert-location .../ep16_mmlu_logical_count.pt`
2. **Baseline** uses baseline worktree (98a107d) + `--init-expert-location .../ep16_mmlu_logical_count.pt` (same EPLB file, NO waterfill flag)
3. The ONLY difference should be `--enable-deepep-waterfill` — baseline MUST also use `--init-expert-location`
4. Between switching waterfill→baseline: kill all, `docker restart` if zombies, reinstall sglang with `pip install -e ... --no-deps`
5. Use different `--dist-init-addr` port for each launch attempt

### How the Incorrect +9.6% Gain Was Produced (BUG RECORD)

On 2026-02-18, the first round of A/B testing showed waterfill at +9.6% gain (30,979 vs 28,263 tok/s). This was because the **baseline was launched WITHOUT `--init-expert-location`**, so it used trivial (round-robin) expert dispatch instead of EPLB. Trivial dispatch is ~1000 tok/s slower than EPLB dispatch because experts are not optimally placed.

The Feb 12 historical tests correctly used `--init-expert-location` for BOTH waterfill and baseline (verified from server logs at `ep16_mmlu_ab_3rounds_20260213/baseline_r1/node1.log`). After correcting the baseline to also use EPLB, the gain returned to the expected +4.1%.

**Rule**: When comparing waterfill vs baseline, ALWAYS verify both server logs show `init_expert_location from init_by_eplb` in the startup output.

### Comparison with Historical Results

| Date | Waterfill | Baseline (EPLB) | Gain | Notes |
|------|-----------|-----------------|------|-------|
| 2026-02-12 R1 | 30,469 | 29,326 | +3.9% | `sglang_lb_with_deepep` image, both use EPLB |
| 2026-02-12 R2 | 30,134 | 29,535 | +2.0% | Same |
| 2026-02-12 R3 | 30,501 | 29,502 | +3.4% | Same |
| **2026-02-18** | **30,979** | **29,745** | **+4.1%** | v0.5.5.post3 + upgraded packages, both use EPLB |

Waterfill throughput is consistent across dates (~30.1-31.0k). Baseline with EPLB is also consistent (~29.3-29.7k). The gain is consistently +3-4%.

### Key Parameters

```
# BOTH waterfill AND baseline MUST use:
--tp 16 --dp-size 16 --nnodes 2 --chunked-prefill-size -1 --max-prefill-tokens 8192
--disable-radix-cache --disable-cuda-graph --mem-fraction-static 0.75
--max-running-requests 2048 --moe-a2a-backend deepep --deepep-mode normal
--enable-dp-attention
--init-expert-location /lustre/.../ep16_mmlu_logical_count.pt   # BOTH must use this!

# ONLY waterfill adds:
--enable-deepep-waterfill
```

### Result Files

- Feb 12 log: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_mmlu_ab_3rounds_20260213/full_log.txt`
- Feb 18 server logs: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/waterfill_static_node{0,1}.log`, `baseline_eplb_node{0,1}.log`
- Feb 18 dynamic logs: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/waterfill_dynamic4_node{0,1}.log`, `baseline_dynamic2_node{0,1}.log`

---

## Benchmark Results (2026-02-10, waterfill_bench_v5)

All results use JIT cache pre-warming (fair comparison). All modes run with CUDA graph disabled, `output_len=1`, `deepep_mode=normal`.

### Input Throughput (tok/s) — Primary Metric

| Case | baseline | waterfill | eplb | eplb_waterfill |
|------|----------|-----------|------|----------------|
| bs128_il512 | 35,141 | 36,290 (+3.3%) | 38,831 (+10.5%) | 38,763 (+10.3%) |
| bs64_il1024 | 33,948 | 35,161 (+3.6%) | 36,465 (+7.4%) | 37,936 (+11.7%) |
| bs32_il2048 | 31,718 | 32,796 (+3.4%) | 36,129 (+13.9%) | 36,008 (+13.5%) |
| bs16_il4096 | 28,602 | 29,450 (+3.0%) | 31,841 (+11.3%) | 32,300 (+12.9%) |

### Latency (s)

| Case | baseline | waterfill | eplb | eplb_waterfill |
|------|----------|-----------|------|----------------|
| bs128_il512 | 29.84 | 28.90 (-3.2%) | 27.00 (-9.5%) | 27.05 (-9.4%) |
| bs64_il1024 | 30.89 | 29.82 (-3.4%) | 28.76 (-6.9%) | 27.64 (-10.5%) |
| bs32_il2048 | 33.06 | 31.97 (-3.3%) | 29.02 (-12.2%) | 29.12 (-11.9%) |
| bs16_il4096 | 36.66 | 35.61 (-2.9%) | 32.93 (-10.2%) | 32.47 (-11.4%) |

### Key Takeaways

1. **Waterfill alone**: Consistent +3.0% to +3.6% input throughput improvement over baseline (no EPLB needed).
2. **EPLB alone**: +7.4% to +13.9% improvement — expert load balancing is the dominant optimization.
3. **EPLB + waterfill**: Similar to EPLB alone (~0-4% additional gain on top of EPLB); the waterfill benefit is smaller when experts are already well-balanced.
4. **Best configuration**: EPLB or EPLB+waterfill, depending on workload. For bs64_il1024, EPLB+waterfill achieves the best result (+11.7%).

### Result Files

- Step 1 (baseline, waterfill): `/lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_step1_run8.log`
- Step 3 (eplb, eplb_waterfill): `/lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_step3_run1.log`
- Summary JSONs: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/waterfill_bench_v5/ep16/summary.json`
- EPLB file: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/ep16_logical_count.pt`

---

## EP8 Benchmark Results (2026-02-10, full_bench_v3)

Single-node (10.6.131.5), TP=8, DP=8. All modes with CUDA graph disabled, `output_len=1`, `deepep_mode=normal`. JIT cache pre-warmed.

### EP8 Input Throughput (tok/s) — Primary Metric

| Case | baseline (98a107d) | waterfill | eplb | eplb_waterfill |
|------|--------------------|-----------|------|----------------|
| bs128_il512 | 20,360 | 21,075 (+3.5%) | 20,757 (+2.0%) | 21,385 (+5.0%) |
| bs64_il1024 | 19,657 | **11,582 (-41.1%)** | 21,091 (+7.3%) | 20,839 (+6.0%) |
| bs32_il2048 | 18,380 | 19,187 (+4.4%) | 19,676 (+7.0%) | 19,707 (+7.2%) |
| bs16_il4096 | 16,387 | 17,076 (+4.2%) | 16,994 (+3.7%) | 17,563 (+7.2%) |

### EP8 Latency (s)

| Case | baseline | waterfill | eplb | eplb_waterfill |
|------|----------|-----------|------|----------------|
| bs128_il512 | 25.75 | 24.88 (-3.4%) | 25.26 (-1.9%) | 24.52 (-4.8%) |
| bs64_il1024 | 26.67 | **45.27 (+69.7%)** | 24.86 (-6.8%) | 25.16 (-5.7%) |
| bs32_il2048 | 28.52 | 27.33 (-4.2%) | 26.65 (-6.6%) | 26.60 (-6.7%) |
| bs16_il4096 | 32.00 | 30.70 (-4.1%) | 30.85 (-3.6%) | 29.85 (-6.7%) |

### EP8 Key Takeaways

1. **Waterfill crash fix works**: All modes completed without CUDA errors (fix in `deepseek_v2.py` for 9-column topk in `num_tokens == 0` path).
2. **Anomaly in waterfill bs64_il1024**: 11,582 tok/s (41% regression, 45.3s latency). All other waterfill cases show +3.5-4.4% gain. Likely a transient issue (stalled DP rank, server warmup artifact). Needs re-run with `--repeat 3` to confirm.
3. **eplb_waterfill is the best mode**: Consistent +5.0% to +7.2% over baseline across all cases.
4. **EPLB alone**: +2.0% to +7.3% improvement. Smaller gains than EP16 (expected — less cross-node communication to balance).
5. **EP8 vs EP16 comparison**: EP8 throughput is ~58% of EP16 (20k vs 35k tok/s for bs128_il512), consistent with H20 scaling expectations (8 vs 16 GPUs, but EP16 has cross-node overhead).

### EP8 Result Files

- Log: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/ep8_full_perf_v3.log`
- Summary JSON: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/full_bench_v3/ep8/ep8/summary.json`
- EPLB file: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/ep8_logical_count.pt`
