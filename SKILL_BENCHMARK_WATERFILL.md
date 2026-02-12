# Skill: E2E Benchmark for Waterfill (DeepSeek-V3)

This skill defines the end-to-end benchmark procedure for the **waterfill** optimization on DeepSeek-V3, covering **performance testing**, **torch profile tracing**, and **accuracy testing**.

> **See also**: `SKILL_BENCHMARK_WATERFILL_EP16_H20.md` — EP16 benchmark on the new H20 cluster (10.6.131.5/6, shared Lustre, `sglang_lb` container).

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

> **Important**: Use `--output-len 1` for waterfill benchmarking. Waterfill optimizes the MoE dispatch path which primarily affects the prefill (EXTEND) phase. Using `output_len=1` isolates prefill throughput as the metric. The key metric to compare is `input_throughput` (tok/s), not `output_throughput`.

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

### Accuracy Test Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--num-examples` | **2000** (default in bench script) | Sufficient for statistical significance; full MMLU is ~14042 |
| Seed | **0** (hardcoded in `MMLUEval`) | `random.Random(0).sample()` — deterministic across runs |
| `--num-threads` | 512 | Parallel eval threads |

> **Important**: MMLU seed is fixed to 0 in `simple_eval_mmlu.py:MMLUEval.__init__()`, so the same 2000 questions are always selected regardless of which mode runs. This guarantees apple-to-apple comparison across baseline/waterfill/eplb/eplb_waterfill.

### Method 1: Automated via `bench_waterfill_multinode.py` (Recommended)

The multi-node bench script supports integrated accuracy testing:

```bash
# EP8 accuracy only (all 4 modes, 2000 examples by default)
python3 benchmark/deepseek_v3/bench_waterfill_multinode.py --ep 8 \
    --modes baseline,waterfill,eplb,eplb_waterfill \
    --accuracy-only \
    --baseline-sglang-dir /lustre/.../sglang_baseline_98a107d \
    --init-expert-location /lustre/.../ep8_logical_count.pt

# Override num-examples if needed
python3 benchmark/deepseek_v3/bench_waterfill_multinode.py --ep 8 \
    --modes baseline,waterfill --accuracy-only --num-examples 500
```

### Method 2: `run_eval.py` (Manual, against running server)

```bash
# Launch server first (baseline or optimized, as shown above), then:
python3 -m sglang.test.run_eval \
    --base-url http://127.0.0.1:30000 \
    --eval-name mmlu \
    --num-examples 2000 \
    --num-threads 512
```

Output:
- Score printed to stdout (e.g., `Score: 0.906`)
- HTML report: `/tmp/mmlu_*.html`
- JSON results: `/tmp/mmlu_*.json`

Expected score for DeepSeek-V3: ~0.88+ (baseline and optimized should be within 0.002).

### EP8 Accuracy Results (2026-02-10, full MMLU 14042 examples)

| Mode | MMLU Score |
|------|-----------|
| baseline | 0.8820 |
| waterfill | 0.8820 |
| eplb | 0.8840 |
| eplb_waterfill | 0.8830 |

**Conclusion**: Waterfill does not impact accuracy. All modes within 0.002 of each other.

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

## Known Issues

### DeepGEMM JIT Cache Bias in Sequential Benchmarks

**CRITICAL**: DeepGEMM uses JIT compilation for GEMM kernels. The compiled kernels are cached on disk at `/root/.cache/deep_gemm/cache/` (~385 kernels for DeepSeek-V3). When running multiple modes sequentially (e.g., baseline then waterfill), the **first mode** bears all JIT compilation overhead, while the **second mode** reuses the disk cache. This can make the second mode appear **2x faster** — a completely misleading result.

**Symptom**: If the first mode shows latency ~2x of the second mode for the same workload, JIT cache bias is the likely cause. Swap the mode order to verify.

**Fix**: Pre-warm the JIT cache before running any benchmark modes. Launch a server, run one warmup request to populate `/root/.cache/deep_gemm/cache/` on all nodes, then kill the server. After this, both modes will use cached kernels and produce fair, comparable numbers.

**Important**: Do NOT set `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`. The default (1) is correct and required for multi-node NVSHMEM stability. See `SKILL_BENCHMARK_WATERFILL_EP16_H20.md` issue #6 for details.

### EP8 Waterfill+EPLB is Structurally Unviable

Waterfill cannot produce positive throughput gain on EP8+EPLB. The fixed overhead (~5-6%: lost alt_stream overlap + extra AllReduce) exceeds the benefit (~1.3% from reducing imbalance 1.112→1.091). This is unfixable without eliminating the AllReduce or finding a way to overlap it. See `SKILL_BENCHMARK_WATERFILL_EP16_H20.md` issue #11 for full analysis.

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
| `benchmark/deepseek_v3/bench_waterfill_multinode.py` | Multi-node EP16 waterfill benchmark |

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

## Part 4: Multi-Node EP16 Benchmark

For multi-node EP16 benchmark, see **SKILL_BENCHMARK_WATERFILL_EP16_H20.md** (H20 cluster at 10.6.131.5/6 with shared Lustre storage).
