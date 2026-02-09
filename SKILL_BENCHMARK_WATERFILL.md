# Skill: E2E Benchmark for Waterfill (DeepSeek-V3)

This skill defines the end-to-end benchmark procedure for the **waterfill** optimization on DeepSeek-V3, covering **performance testing**, **torch profile tracing**, and **accuracy testing**.

---

## Environment

| Item | Value |
|------|-------|
| Container | `sglang_lb` (Docker, image: `lmsysorg/sglang:v0.5.6`) |
| Baseline Repo | `/home/xutingz/workspace/gitsrc/sglang_baseline_98a107d` |
| Optimized Repo | `/home/xutingz/workspace/gitsrc/sglang` |
| Model Path | `/lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3` |
| TP Size | 8 |
| EP Size | 8 |
| Baseline Commit | `98a107d491f4cbb6bcbe1bb3f156a35f5d31c4f0` |
| Optimized Commit | `484e12987d8ba5cc6f9e2558a772e00f3f580d79` (branch: `feat/deepep-waterfill-eplb-balance`) |
| Torch Profile Dir | `/home/xutingz/workspace/torch_profile/waterfill` |
| E2E Test Script | `benchmark/deepseek_v3/run_deepep_waterfill_e2e_test.py` (optimized repo only) |

> **Note**: `/home/xutingz` and `/lustre/raplab/client/xutingz` are the same path.
>
> **Two-repo strategy**: The e2e script does NOT support specifying commits. It requires two **separate directories**, each already checked out at the correct commit. The baseline repo `sglang_baseline_98a107d` is already at the baseline commit. The optimized repo `sglang` is on `feat/deepep-waterfill-eplb-balance`.
>
> **Important**: The e2e script (`run_deepep_waterfill_e2e_test.py`) only exists in the **optimized** repo. Always run it from the optimized repo. The baseline repo (older commit) does not have `--enable-deepep-waterfill` in its `ServerArgs` -- the e2e script handles this correctly by only adding that flag for waterfill mode.

---

## Prerequisites: Two-Repo Setup & Install

All commands run **inside** the `sglang_lb` container. To enter:
```bash
docker exec -it sglang_lb bash
```

Two separate directories are used so that the e2e script can switch between baseline and optimized without manual git operations:

| Role | Directory | Commit |
|------|-----------|--------|
| Baseline | `/home/xutingz/workspace/gitsrc/sglang_baseline_98a107d` | `98a107d491f4` (already checked out) |
| Optimized | `/home/xutingz/workspace/gitsrc/sglang` | `484e12987d` on branch `feat/deepep-waterfill-eplb-balance` |

### Verify & Install Baseline
```bash
cd /home/xutingz/workspace/gitsrc/sglang_baseline_98a107d
git log --oneline -1
# Expected: 98a107d49 Re-enable temp_prefill_info assertion after pairing fix (#16203)

pip install -e "python[dev]" --no-deps -q
```

### Verify & Install Optimized
```bash
cd /home/xutingz/workspace/gitsrc/sglang
git checkout feat/deepep-waterfill-eplb-balance
git log --oneline -1
# Expected: 484e12987 perf(deepep): make waterfill EPLB-aware at low imbalance

pip install -e "python[dev]" --no-deps -q
```

> **Note**: The e2e script runs `pip install -e python[dev] --no-deps -q` automatically before each mode, so manual install is only needed if running commands individually.

---

## Part 1: Performance Testing

Uses `bench_one_batch_server` to compare throughput between baseline and optimized code.

### Parameters
| Parameter | Value |
|-----------|-------|
| `--batch-size` | 256 |
| `--input-len` | 1024 |
| `--output-len` | 1 |
| `--disable-radix-cache` | Yes |
| CUDA Graph | Enabled (default; do NOT pass `--disable-cuda-graph`) |

### Server Launch (for each mode)

The server is launched by `bench_one_batch_server` internally, or you can launch separately and use `--base-url`.

#### Option A: Separate server + bench client (Recommended for manual runs)

Launch server and bench client separately. This gives you access to the full server log for analysis.

**Baseline**:
```bash
cd /home/xutingz/workspace/gitsrc/sglang_baseline_98a107d

# Launch server (no waterfill)
python3 -m sglang.launch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --tp 8 \
    --ep-size 8 \
    --moe-a2a-backend deepep \
    --trust-remote-code \
    --deepep-mode normal \
    --disable-radix-cache \
    --host 0.0.0.0 \
    --port 30000 \
    --log-level info \
    2>&1 | tee server_baseline.log &

# Wait for server ready, then run bench:
python3 -m sglang.bench_one_batch_server \
    --model-path none \
    --base-url http://127.0.0.1:30000 \
    --batch-size 256 \
    --input-len 1024 \
    --output-len 1 \
    --show-report \
    --result-filename result_baseline.jsonl \
    --no-append-to-github-summary

# Kill server after benchmark
pkill -9 -f "sglang"
```

**Optimized**:
```bash
cd /home/xutingz/workspace/gitsrc/sglang

# Launch server (with waterfill)
python3 -m sglang.launch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --tp 8 \
    --ep-size 8 \
    --moe-a2a-backend deepep \
    --trust-remote-code \
    --deepep-mode normal \
    --enable-deepep-waterfill \
    --disable-radix-cache \
    --host 0.0.0.0 \
    --port 30000 \
    --log-level info \
    2>&1 | tee server_optimized.log &

# Wait for server ready, then run bench:
python3 -m sglang.bench_one_batch_server \
    --model-path none \
    --base-url http://127.0.0.1:30000 \
    --batch-size 256 \
    --input-len 1024 \
    --output-len 1 \
    --show-report \
    --result-filename result_optimized.jsonl \
    --no-append-to-github-summary

# Kill server after benchmark
pkill -9 -f "sglang"
```

#### Option B: All-in-one (server + bench in one command)

`bench_one_batch_server` can also launch the server internally. This is simpler but the server log is mixed with bench output.

**Baseline**:
```bash
cd /home/xutingz/workspace/gitsrc/sglang_baseline_98a107d

python3 -m sglang.bench_one_batch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --tp 8 \
    --ep-size 8 \
    --moe-a2a-backend deepep \
    --trust-remote-code \
    --deepep-mode normal \
    --disable-radix-cache \
    --batch-size 256 \
    --input-len 1024 \
    --output-len 1 \
    --show-report \
    --result-filename result_baseline.jsonl \
    --no-append-to-github-summary \
    --log-level info \
    2>&1 | tee bench_baseline.log
```

**Optimized**:
```bash
cd /home/xutingz/workspace/gitsrc/sglang

python3 -m sglang.bench_one_batch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --tp 8 \
    --ep-size 8 \
    --moe-a2a-backend deepep \
    --trust-remote-code \
    --deepep-mode normal \
    --enable-deepep-waterfill \
    --disable-radix-cache \
    --batch-size 256 \
    --input-len 1024 \
    --output-len 1 \
    --show-report \
    --result-filename result_optimized.jsonl \
    --no-append-to-github-summary \
    --log-level info \
    2>&1 | tee bench_optimized.log
```

> **Note**: `--enable-deepep-waterfill` only exists in the optimized repo. Do NOT add it to the baseline command.

### What to Check in Server Logs

1. **CUDA Graph**: Look for `cuda graph: True` in the decode batch lines. Example:
   ```
   Decode batch, #running-req: 256, #token: 272640, token usage: 0.45, cuda graph: True, gen throughput (token/s): 34.49
   ```
   If `cuda graph: False`, there is a problem -- decode/verify should have CUDA graph enabled.

2. **Prefill Batches**: Look for lines like:
   ```
   Prefill batch, #new-seq: 8, #new-token: 8192, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 248
   ```
   Record: `new_seq` (batch size), `new_token` (tokens processed).

3. **Decode Batches**: Look for lines like:
   ```
   Decode batch, #running-req: 256, #token: 272640, cuda graph: True, gen throughput (token/s): 34.49
   ```
   Record: `running_req`, `gen_throughput`.

4. **Metrics from bench output**:
   - `input_throughput` (tok/s) -- prefill throughput
   - `output_throughput` (tok/s) -- decode throughput
   - `latency` (s) -- total latency
   - `last_ttft` (s) -- time to first token (prefill time)

### Analyzing Results

Compare the `result_baseline.jsonl` and `result_optimized.jsonl` files. Each line is a JSON object:
```json
{"run_name": "default", "batch_size": 256, "input_len": 1024, "output_len": 1, "latency": 12.34, "input_throughput": 21234.56, "output_throughput": 2650.12, "overall_throughput": 23884.68, "last_ttft": 1.23, "last_gen_throughput": 34.49, "acc_length": -1.0}
```

Determine if the performance bottleneck is in **prefill** (compare `input_throughput` and `last_ttft`) or **decode** (compare `output_throughput` and `last_gen_throughput`).

### Using the Existing E2E Script (Alternative)

The repo has a comprehensive e2e test script that automates baseline vs. waterfill comparison:

```bash
cd /home/xutingz/workspace/gitsrc/sglang

python3 benchmark/deepseek_v3/run_deepep_waterfill_e2e_test.py \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --tp 8 --ep 8 \
    --baseline-sglang-dir /home/xutingz/workspace/gitsrc/sglang_baseline_98a107d \
    --waterfill-sglang-dir /home/xutingz/workspace/gitsrc/sglang \
    --docker-container sglang_lb \
    --run-one-batch \
    --one-batch-num-prompts 256 \
    --one-batch-input-len 1024 \
    --one-batch-output-len 1 \
    --skip-accuracy \
    --skip-serving
```

The script automatically does `pip install -e python[dev] --no-deps -q` in each directory before running.

---

## Part 2: Torch Profile Trace

Uses `bench_one_batch_server --profile` to capture torch profiler traces. With `--profile-by-stage`, prefill (EXTEND) and decode (DECODE) are saved as **separate** trace files per rank. Multiple ranks' traces are automatically merged into a single file (via `merge_profiles=True` in `run_profile`).

### Profile Parameters
| Parameter | Value |
|-----------|-------|
| `--batch-size` | 256 |
| `--input-len` | 1024 |
| `--output-len` | 1 |
| `--profile` | Yes |
| `--profile-by-stage` | Yes (separate prefill/decode traces) |
| `--profile-steps` | 5 |
| `--profile-output-dir` | `/home/xutingz/workspace/torch_profile/waterfill` |

### Commands

First, launch the server (baseline or optimized). Then run the profiling bench:

**Baseline Profile**:
```bash
cd /home/xutingz/workspace/gitsrc/sglang_baseline_98a107d

# Launch server
python3 -m sglang.launch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --tp 8 \
    --ep-size 8 \
    --moe-a2a-backend deepep \
    --trust-remote-code \
    --deepep-mode normal \
    --disable-radix-cache \
    --host 0.0.0.0 \
    --port 30000 \
    --log-level info \
    2>&1 | tee server_baseline_profile.log &

# Wait for server ready, then:
python3 -m sglang.bench_one_batch_server \
    --model-path none \
    --base-url http://127.0.0.1:30000 \
    --batch-size 256 \
    --input-len 1024 \
    --output-len 1 \
    --seed 1 \
    --profile \
    --profile-by-stage \
    --profile-steps 5 \
    --profile-prefix baseline- \
    --profile-output-dir /home/xutingz/workspace/torch_profile/waterfill \
    --result-filename profile_result_baseline.jsonl \
    --no-append-to-github-summary
```

**Optimized Profile**:
```bash
cd /home/xutingz/workspace/gitsrc/sglang

# Launch server (with waterfill enabled)
python3 -m sglang.launch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --tp 8 \
    --ep-size 8 \
    --moe-a2a-backend deepep \
    --trust-remote-code \
    --deepep-mode normal \
    --enable-deepep-waterfill \
    --disable-radix-cache \
    --host 0.0.0.0 \
    --port 30000 \
    --log-level info \
    2>&1 | tee server_optimized_profile.log &

# Wait for server ready, then:
python3 -m sglang.bench_one_batch_server \
    --model-path none \
    --base-url http://127.0.0.1:30000 \
    --batch-size 256 \
    --input-len 1024 \
    --output-len 1 \
    --seed 1 \
    --profile \
    --profile-by-stage \
    --profile-steps 5 \
    --profile-prefix optimized- \
    --profile-output-dir /home/xutingz/workspace/torch_profile/waterfill \
    --result-filename profile_result_optimized.jsonl \
    --no-append-to-github-summary
```

### Trace File Layout

The profiler creates a timestamped subdirectory under `--profile-output-dir`:
```
/home/xutingz/workspace/torch_profile/waterfill/
  {timestamp}/                          # e.g., 1738857600.123456
    server_args.json                    # Server configuration
    baseline-bs-256-il-1024-{ts}-TP-0-EP-0-EXTEND.trace.json.gz
    baseline-bs-256-il-1024-{ts}-TP-0-EP-0-DECODE.trace.json.gz
    baseline-bs-256-il-1024-{ts}-TP-1-EP-1-EXTEND.trace.json.gz
    baseline-bs-256-il-1024-{ts}-TP-1-EP-1-DECODE.trace.json.gz
    ...  (one EXTEND + one DECODE per TP/EP rank)
    merged-baseline-bs-256-il-1024-{ts}-EXTEND.trace.json.gz   # All ranks merged (prefill)
    merged-baseline-bs-256-il-1024-{ts}-DECODE.trace.json.gz   # All ranks merged (decode)
```

- **EXTEND** suffix = prefill trace
- **DECODE** suffix = decode trace
- Each rank (TP-0-EP-0 through TP-7-EP-7) produces two files
- **merged-** prefix = all TP/EP ranks combined into one Chrome trace viewable file
- To view: open merged `.trace.json.gz` in Chrome `chrome://tracing` or Perfetto

### Using the Existing E2E Script (Alternative)

```bash
python3 benchmark/deepseek_v3/run_deepep_waterfill_e2e_test.py \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --tp 8 --ep 8 \
    --baseline-sglang-dir /home/xutingz/workspace/gitsrc/sglang_baseline_98a107d \
    --waterfill-sglang-dir /home/xutingz/workspace/gitsrc/sglang \
    --docker-container sglang_lb \
    --run-torch-profile \
    --torch-profile-root /home/xutingz/workspace/torch_profile/waterfill \
    --skip-accuracy \
    --skip-serving
```

---

## Part 3: Accuracy Testing (MMLU)

Uses sglang's MMLU evaluation script to verify correctness of the optimized code vs. baseline.

### Method 1: `run_eval.py` (Recommended, simpler)

This downloads MMLU data automatically and runs against a running server:

```bash
# Launch server first (baseline or optimized, as shown above), then:
python3 -m sglang.test.run_eval \
    --base-url http://127.0.0.1:30000 \
    --eval-name mmlu \
    --num-examples 64 \
    --num-threads 512
```

Output:
- Score printed to stdout (e.g., `Score: 0.906`)
- HTML report: `/tmp/mmlu_*.html`
- JSON results: `/tmp/mmlu_*.json`

Expected score for DeepSeek-V3: ~0.90+ (baseline and optimized should be very close).

### Method 2: `bench_sglang.py` (Legacy, more detailed per-subject)

Requires MMLU data to be downloaded first:
```bash
cd /home/xutingz/workspace/gitsrc/sglang/benchmark/mmlu
bash download_data.sh  # Downloads to ./data/
```

Then:
```bash
python3 bench_sglang.py \
    --backend srt \
    --host http://127.0.0.1 \
    --port 30000 \
    --parallel 8 \
    --ntrain 5 \
    --nsub 60 \
    --data_dir data \
    --result-file mmlu_result.jsonl
```

### Method 3: Using the E2E Script

```bash
python3 benchmark/deepseek_v3/run_deepep_waterfill_e2e_test.py \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --tp 8 --ep 8 \
    --baseline-sglang-dir /home/xutingz/workspace/gitsrc/sglang_baseline_98a107d \
    --waterfill-sglang-dir /home/xutingz/workspace/gitsrc/sglang \
    --docker-container sglang_lb \
    --skip-serving
```

This runs both GSM8K and MMLU accuracy tests for baseline and waterfill automatically.

### Speculative Decoding (if applicable)

If speculative decoding is enabled, the `bench_one_batch_server` output includes `acc_length` (average speculative accept length). Compare this value between baseline and optimized:
- Check `acc_length` in the result JSONL files
- Also available via server info endpoint: `GET /get_server_info` -> `internal_states[0].avg_spec_accept_length`

> **Note**: For this benchmark run, speculative decoding is **NOT** enabled. The `acc_length` field will show `-1.0`.

---

## Full Workflow Summary

### Step-by-step (manual, using two repos)

1. **Enter container**: `docker exec -it sglang_lb bash`

2. **Run baseline** (from `sglang_baseline_98a107d`):
   ```bash
   cd /home/xutingz/workspace/gitsrc/sglang_baseline_98a107d
   pip install -e "python[dev]" --no-deps -q
   ```
   - Run performance bench (Part 1 baseline)
   - Run torch profile (Part 2 baseline)
   - Run MMLU accuracy (Part 3 baseline)
   - **Kill the server** between runs: `pkill -9 -f sglang`

3. **Run optimized** (from `sglang`):
   ```bash
   cd /home/xutingz/workspace/gitsrc/sglang
   pip install -e "python[dev]" --no-deps -q
   ```
   - Run performance bench (Part 1 optimized)
   - Run torch profile (Part 2 optimized)
   - Run MMLU accuracy (Part 3 optimized)

4. **Compare results**:
   - Performance: compare `input_throughput`, `output_throughput`, `latency`, `last_gen_throughput`
   - Traces: open merged `.trace.json.gz` files in Chrome `chrome://tracing` or Perfetto
   - Accuracy: compare MMLU scores (should be similar, <1% difference)

### Using the All-in-One E2E Script (Recommended)

For the complete benchmark (all 3 parts at once), using two separate directories:

```bash
cd /home/xutingz/workspace/gitsrc/sglang

python3 benchmark/deepseek_v3/run_deepep_waterfill_e2e_test.py \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --tp 8 --ep 8 \
    --baseline-sglang-dir /home/xutingz/workspace/gitsrc/sglang_baseline_98a107d \
    --waterfill-sglang-dir /home/xutingz/workspace/gitsrc/sglang \
    --docker-container sglang_lb \
    --run-one-batch \
    --one-batch-num-prompts 256 \
    --one-batch-input-len 1024 \
    --one-batch-output-len 1 \
    --run-torch-profile \
    --torch-profile-root /home/xutingz/workspace/torch_profile/waterfill
```

The script handles `pip install` and server start/stop for each directory automatically. No git checkout needed since each directory is already at the correct commit.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `python/sglang/bench_one_batch_server.py` | Single-batch latency/throughput benchmark |
| `python/sglang/profiler.py` | Client-side torch profiler launcher |
| `python/sglang/srt/managers/scheduler_profiler_mixin.py` | Server-side profiler (trace file naming, stage separation) |
| `python/sglang/srt/utils/profile_merger.py` | Multi-rank trace merging |
| `python/sglang/test/run_eval.py` | MMLU/GSM8K/etc. evaluation entry point |
| `python/sglang/test/simple_eval_mmlu.py` | MMLU evaluation class |
| `benchmark/mmlu/bench_sglang.py` | Legacy MMLU benchmark (per-subject) |
| `benchmark/deepseek_v3/run_deepep_waterfill_e2e_test.py` | Full e2e regression test script |
| `benchmark/deepseek_v3/bench_waterfill_multinode.py` | Multi-node EP16/EP32 waterfill benchmark (H20 cluster) |

---

## Server Log Parsing Patterns

### Prefill batch
```
Prefill batch, #new-seq: {N}, #new-token: {T}, #cached-token: 0, token usage: X.XX, #running-req: {R}, #queue-req: {Q}
```

### Decode batch
```
Decode batch, #running-req: {N}, #token: {T}, token usage: X.XX, cuda graph: {True|False}, gen throughput (token/s): {THROUGHPUT}, #queue-req: 0
```

### Regex patterns (from `run_deepep_waterfill_e2e_test.py:parse_server_log`):
```python
prefill_pattern = r"Prefill batch.*?#new-seq:\s*(\d+).*?#new-token:\s*(\d+).*?#running-req:\s*(\d+)"
decode_pattern = r"Decode batch.*?#running-req:\s*(\d+).*?#token:\s*(\d+).*?cuda graph:\s*(True|False).*?gen throughput.*?:\s*([0-9.]+)"
```

---

## Part 4: Multi-Node EP16/EP32 Benchmark (H20 Cluster)

Automated multi-node benchmark using `bench_waterfill_multinode.py`. Supports four modes: **baseline**, **waterfill**, **eplb**, **eplb_waterfill**.

### Cluster Environment

| Item | Value |
|------|-------|
| Cluster | 6x H20-GPU nodes (8x H20 per node), NVLink NV18, 9x 400Gbps RoCE |
| Container | `sglang_eplb` (`lmsysorg/sglang:v0.5.5.post3`) |
| Model | `/raid/model/DeepSeek-R1` (local on each node) |
| Code | `/root/xutingz/gitsrc/sglang` (branch `feat/deepep-waterfill-eplb-balance`, editable install) |
| Storage | **Not shared** — must rsync code to all nodes before running |
| Dataset | `/root/xutingz/data/ShareGPT_V3_unfiltered_cleaned_split.json` |

### EP Configuration

| EP | Nodes | Node IPs | actual_tp | actual_dp | nnodes |
|----|-------|----------|-----------|-----------|--------|
| 16 | 2 | 10.6.131.20, .21 | 16 | 16 | 2 |
| 32 | 4 | 10.6.131.20, .21, .22, .23 | 16 | 2 | 4 |

### Benchmark Modes

| Mode | Waterfill | EPLB | Description |
|------|-----------|------|-------------|
| `baseline` | No | No | Vanilla DeepEP, trivial expert placement |
| `waterfill` | Yes | No | Waterfill shared expert dispatch, trivial placement |
| `eplb` | No | Yes | Static EPLB expert placement, no waterfill |
| `eplb_waterfill` | Yes | Yes | EPLB placement + waterfill shared dispatch |

### Benchmark Cases

All cases use `output_len=1` and `deepep_mode=normal`. Batch size is **per DP rank** (local); the script automatically scales to global batch size (local_bs * dp_size).

| Name | local_bs (per rank) | input_len | output_len |
|------|---------------------|-----------|------------|
| bs128_il512 | 128 | 512 | 1 |
| bs64_il1024 | 64 | 1024 | 1 |
| bs32_il2048 | 32 | 2048 | 1 |
| bs16_il4096 | 16 | 4096 | 1 |

### Required Environment Variables

```bash
export SGLANG_LOG_MS=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_HCA_LIST="mlx5_3:1,mlx5_2:1,mlx5_1:1,mlx5_0:1,mlx5_5:1,mlx5_4:1,mlx5_7:1,mlx5_6:1"
```

### EPLB Distribution Files

| EP | Path | How to generate |
|----|------|-----------------|
| 16 | `/root/xutingz/output/eplb/ep16_logical_count.pt` | Already exists |
| 32 | `/root/xutingz/output/eplb/ep32_logical_count.pt` | See "Generating EP32 EPLB" below |

### Prerequisites

1. **Sync code to all nodes** (storage is not shared):
   ```bash
   for ip in 10.6.131.21 10.6.131.22 10.6.131.23; do
     rsync -az /root/xutingz/gitsrc/sglang/ root@$ip:/root/xutingz/gitsrc/sglang/ &
   done
   wait
   ```

2. **Verify sglang install** on all nodes:
   ```bash
   for ip in 10.6.131.20 10.6.131.21 10.6.131.22 10.6.131.23; do
     echo "=== $ip ==="
     ssh root@$ip "docker exec sglang_eplb python3 -c 'import sglang; print(sglang.__version__)'"
   done
   ```

3. **Clean stale processes**:
   ```bash
   for ip in 10.6.131.20 10.6.131.21 10.6.131.22 10.6.131.23; do
     ssh root@$ip "docker exec sglang_eplb bash -c 'pkill -9 -f sglang 2>/dev/null; rm -f /dev/shm/nccl* /dev/shm/nvshmem* 2>/dev/null'"
   done
   ```

### Running the Benchmark

All commands run **inside** the `sglang_eplb` container on node 0 (10.6.131.20). The script automatically SSH's to worker nodes to launch/kill remote server processes.

#### EP16: EPLB vs EPLB+Waterfill (recommended comparison)

```bash
docker exec sglang_eplb bash -c '
  export SGLANG_LOG_MS=1
  export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
  export NVSHMEM_IB_GID_INDEX=3
  export NVSHMEM_HCA_LIST="mlx5_3:1,mlx5_2:1,mlx5_1:1,mlx5_0:1,mlx5_5:1,mlx5_4:1,mlx5_7:1,mlx5_6:1"
  python3 /root/xutingz/gitsrc/sglang/benchmark/deepseek_v3/bench_waterfill_multinode.py \
    --ep 16 \
    --modes eplb,eplb_waterfill \
    --init-expert-location /root/xutingz/output/eplb/ep16_logical_count.pt
'
```

#### EP16: All 4 modes

```bash
docker exec sglang_eplb bash -c '
  export SGLANG_LOG_MS=1
  export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
  export NVSHMEM_IB_GID_INDEX=3
  export NVSHMEM_HCA_LIST="mlx5_3:1,mlx5_2:1,mlx5_1:1,mlx5_0:1,mlx5_5:1,mlx5_4:1,mlx5_7:1,mlx5_6:1"
  python3 /root/xutingz/gitsrc/sglang/benchmark/deepseek_v3/bench_waterfill_multinode.py \
    --ep 16 \
    --modes baseline,waterfill,eplb,eplb_waterfill \
    --init-expert-location /root/xutingz/output/eplb/ep16_logical_count.pt
'
```

#### EP32: EPLB vs EPLB+Waterfill

```bash
docker exec sglang_eplb bash -c '
  export SGLANG_LOG_MS=1
  export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
  export NVSHMEM_IB_GID_INDEX=3
  export NVSHMEM_HCA_LIST="mlx5_3:1,mlx5_2:1,mlx5_1:1,mlx5_0:1,mlx5_5:1,mlx5_4:1,mlx5_7:1,mlx5_6:1"
  python3 /root/xutingz/gitsrc/sglang/benchmark/deepseek_v3/bench_waterfill_multinode.py \
    --ep 32 \
    --modes eplb,eplb_waterfill \
    --init-expert-location /root/xutingz/output/eplb/ep32_logical_count.pt
'
```

#### Background execution (recommended for long runs)

The benchmark takes ~20 min per mode (model load + bench cases). Use nohup from the host:

```bash
ssh root@10.6.131.20 "nohup docker exec sglang_eplb bash -c '
  export SGLANG_LOG_MS=1 &&
  export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 &&
  export NVSHMEM_IB_GID_INDEX=3 &&
  export NVSHMEM_HCA_LIST=\"mlx5_3:1,mlx5_2:1,mlx5_1:1,mlx5_0:1,mlx5_5:1,mlx5_4:1,mlx5_7:1,mlx5_6:1\" &&
  python3 /root/xutingz/gitsrc/sglang/benchmark/deepseek_v3/bench_waterfill_multinode.py \
    --ep 16 \
    --modes eplb,eplb_waterfill \
    --init-expert-location /root/xutingz/output/eplb/ep16_logical_count.pt
' > /root/xutingz/output/waterfill_bench/ep16_run.log 2>&1 &"

# Monitor progress:
ssh root@10.6.131.20 "tail -f /root/xutingz/output/waterfill_bench/ep16_run.log"
```

### Output

Results are saved to `/root/xutingz/output/waterfill_bench/ep{16,32}/`:

```
ep16/
  eplb/
    logs/server_node0.log, server_node1.log
    result_bs128_il512.jsonl
    result_bs128_il1024.jsonl
    ...
  eplb_waterfill/
    logs/server_node0.log, server_node1.log
    result_bs128_il512.jsonl
    ...
  summary.json          # All results + comparison table
```

The script prints a comparison table at the end. The `gain` column compares the first mode vs the last mode.

### Generating EP32 EPLB Distribution File

If `/root/xutingz/output/eplb/ep32_logical_count.pt` does not exist, generate it:

1. **Launch EP32 server with expert distribution recorder** (4 nodes, `--deepep-mode normal`):

   ```bash
   # On each node (rank 0-3), inside sglang_eplb container:
   export SGLANG_LOG_MS=1
   export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
   export NVSHMEM_IB_GID_INDEX=3
   export NVSHMEM_HCA_LIST="mlx5_3:1,mlx5_2:1,mlx5_1:1,mlx5_0:1,mlx5_5:1,mlx5_4:1,mlx5_7:1,mlx5_6:1"
   export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/root/xutingz/output/eplb

   python3 -m sglang.launch_server \
     --model-path /raid/model/DeepSeek-R1 --trust-remote-code \
     --host 0.0.0.0 --port 30000 \
     --tp 16 --dp-size 2 --enable-dp-attention \
     --moe-a2a-backend deepep --deepep-mode normal \
     --chunked-prefill-size -1 --disable-radix-cache \
     --max-prefill-tokens 8192 --max-running-requests 128 \
     --load-balance-method round_robin \
     --expert-distribution-recorder-mode stat \
     --expert-distribution-recorder-buffer-size 1000 \
     --dist-init-addr 10.6.131.20:20005 --nnodes 4 \
     --log-level info --watchdog-timeout 600 \
     --disable-cuda-graph --skip-server-warmup \
     --node-rank <0|1|2|3>
   ```

2. **Record expert distribution** (from node 0):

   ```bash
   # Start recording
   curl -X POST http://127.0.0.1:30000/start_expert_distribution_record

   # Generate load
   python3 -m sglang.bench_one_batch_server \
     --model None --base-url http://127.0.0.1:30000 \
     --batch-size 128 --input-len 1024 --output-len 10 \
     --dataset-name random \
     --dataset-path /root/xutingz/data/ShareGPT_V3_unfiltered_cleaned_split.json \
     --skip-warmup

   # Stop and dump
   curl -X POST http://127.0.0.1:30000/stop_expert_distribution_record
   curl -X POST http://127.0.0.1:30000/dump_expert_distribution_record
   ```

3. **Rename and distribute**:

   ```bash
   mv /root/xutingz/output/eplb/expert_distribution_recorder_*.pt \
      /root/xutingz/output/eplb/ep32_logical_count.pt

   for ip in 10.6.131.21 10.6.131.22 10.6.131.23; do
     scp /root/xutingz/output/eplb/ep32_logical_count.pt root@$ip:/root/xutingz/output/eplb/
   done
   ```

4. **Kill server**: `pkill -9 -f sglang.launch_server` on all nodes.

Alternatively, use the automated script:
```bash
python3 /root/xutingz/eplb_profile/run_ep32_e2e.py \
  --node-rank 0 \
  --init-expert-location /root/xutingz/output/eplb/ep32_logical_count.pt
```
This generates the EPLB file if it doesn't exist, then proceeds to profiling.

### Known Issues and Workarounds

1. **CUDA graph disabled for all modes**: Waterfill mode cannot use CUDA graph (DeepEP `Buffer.sync()` fails during graph capture). For fair comparison, the script disables CUDA graph for all modes.

2. **Waterfill deadlock fix**: `forward_deepep_waterfill` had a conditional `all_reduce` that caused deadlock when some DP ranks had zero tokens. Fixed by adding a dummy `all_reduce` in the zero-token path (`deepseek_v2.py`, commit `00c93fb00`).

3. **First forward pass is slow (~40s)**: DeepEP buffer initialization (NVSHMEM bootstrap, RDMA setup) happens on the first forward pass. The health check may return 503 during this time. The script's `wait_server()` handles this with a 1800s timeout.

4. **EP32 NVSHMEM instability**: 4-node DeepEP sometimes hits `invalid resource handle` during `Buffer.sync()`. Retry if it happens. Using `--skip-server-warmup` and `--disable-cuda-graph` helps.

5. **Stale NCCL/NVSHMEM shared memory**: After killing a server, clean up with `rm -f /dev/shm/nccl* /dev/shm/nvshmem*` on all nodes. The script's `kill_servers()` does this automatically.

6. **`pkill -f sglang` kills the benchmark script**: The benchmark script path contains "sglang". The `kill_servers()` function uses specific patterns (`sglang.launch_server`, `sglang::scheduler`, etc.) to avoid self-kill.

7. **sgl-kernel version**: Must use 0.3.17.post1. Newer versions have ABI incompatibility with PyTorch 2.8.0+cu129. The `engine.py` check is patched to accept 0.3.17+.

8. **bench_one_batch_server dp_size fix**: Token capacity threshold must be scaled by `dp_size` to avoid skipping large batch cases under DP attention. Patched in `bench_one_batch_server.py`.
