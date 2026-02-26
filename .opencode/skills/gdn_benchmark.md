# GDN Integration E2E Benchmark Skill

## Project Overview

- **Project**: GDN Integration (K-last vs V-last SSM layout optimization)
- **Model path**: `/lustre/raplab/client/xutingz/workspace/model/Qwen3-Next-80B-A3B-Instruct`
- **Code path**: `/lustre/raplab/client/xutingz/workspace/gitsrc/sglang_gdn` (shared across containers via bind mount)
- **Baseline commit**: `b5493f65be22447af02edbfb6eb4cee71f2bc779` (V-last, no `--mamba-ssm-k-last` flag)
- **Optimized commit**: `f043e4ee1786f4fc7d58623fa5400d981697fe4e` (K-last, uses `--mamba-ssm-k-last` flag)
- **Trace output directory**: `/home/xutingz/workspace/torch_profile/gdn`
- **TP size**: 8
- **Note**: `/home/xutingz` and `/lustre/raplab/client/xutingz` are the same path

## Container Architecture

The benchmark uses **two separate containers** to avoid git checkout conflicts and FlashInfer version issues:

| | Baseline (V-last) | Optimized (K-last) |
|---|---|---|
| **Container** | `sglang_dev` | `sglang_klast` |
| **Server port** | `30000` | `30000` |
| **FlashInfer** | pip `flashinfer-python==0.5.3` (system) | Source install from `/lustre/raplab/client/xutingz/workspace/gitsrc/flashinfer` (v0.6.2, with GDN kernels) |
| **Git commit** | `b5493f6` (or current V-last branch) | `f043e4ee1` (K-last optimized) |
| **`--mamba-ssm-k-last`** | No | Yes |

**IMPORTANT**: Both containers use `--network host`, so they share the same network namespace. **Only one server can run on port 30000 at a time.** Always kill the server in one container before starting in the other.

### Running Commands in Containers

All benchmark commands must be run **inside** the appropriate container using `docker exec`:

```bash
# V-last commands run in sglang_dev
docker exec sglang_dev bash -c "<command>"

# K-last commands run in sglang_klast
docker exec sglang_klast bash -c "<command>"
```

For long-running server processes, use `docker exec -d` (detached):
```bash
docker exec -d <container> bash -c "<server launch command> 2>&1 | tee /tmp/server.log"
```

### Container Setup Notes

- `sglang_klast` was created from a snapshot of `sglang_dev` via `docker commit` + `docker run`
- `sglang_klast` has FlashInfer installed from source (`pip install -e .` from the flashinfer repo) which includes `flashinfer.gdn_prefill` and `flashinfer.gdn_decode` modules required for K-last mode
- The old `flashinfer-cubin` (0.5.3) package and its residual directory were removed in `sglang_klast` to avoid version mismatch errors
- Both containers share the same bind mount: `/lustre/raplab/client:/lustre/raplab/client`
- Both containers have access to all 8 GPUs (NVIDIA H20-3e, SM90)

---

## Common Setup

### Git Branch Management

Since the code path is shared across containers, manage git state carefully:

For **V-last baseline** (in `sglang_dev`):
```bash
docker exec sglang_dev bash -c "
cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang_gdn
git stash -q || true
git checkout b5493f65be22447af02edbfb6eb4cee71f2bc779
"
```

For **K-last optimized** (in `sglang_klast`):
```bash
docker exec sglang_klast bash -c "
cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang_gdn
git stash -q || true
git checkout f043e4ee1786f4fc7d58623fa5400d981697fe4e
"
```

**IMPORTANT**: Since both containers share the same code path via bind mount, switching git branches in one container affects the other. Run benchmarks sequentially: first V-last in `sglang_dev`, then K-last in `sglang_klast`. Always checkout the correct commit before installing and launching.

### Installation

After each git checkout, reinstall sglang **in the container that will run the server**.

For `sglang_dev` (V-last baseline):
```bash
docker exec sglang_dev bash -c "
cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang_gdn
pip install -e 'python[all]' --no-build-isolation -q
"
```

For `sglang_klast` (K-last optimized) -- **IMPORTANT**: sglang's dependency on `flashinfer_python==0.5.3` will overwrite the source-installed FlashInfer 0.6.2 (which has GDN kernels). After installing sglang, always re-install FlashInfer from source:
```bash
docker exec sglang_klast bash -c "
cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang_gdn
pip install -e 'python[all]' --no-build-isolation -q
# Re-install FlashInfer from source (sglang overwrites it with pip 0.5.3)
pip uninstall flashinfer-python -y
rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_cubin
cd /lustre/raplab/client/xutingz/workspace/gitsrc/flashinfer
pip install -e . --no-build-isolation -q
"
```

### Server Launch

**Kill existing server before launching a new one (run in BOTH containers to be safe):**
```bash
docker exec sglang_dev bash -c "lsof -ti:30000 | xargs -r kill -9 2>/dev/null || true"
docker exec sglang_klast bash -c "lsof -ti:30000 | xargs -r kill -9 2>/dev/null || true"
sleep 3
```

**Wait for server to be ready:**
Poll `http://127.0.0.1:30000/health` and `http://127.0.0.1:30000/v1/models` until both return 200. Timeout: 1800 seconds. Can poll from either container or the host since they share the network.

#### Non-spec mode (no speculative decoding)

Baseline (V-last) -- run in `sglang_dev`:
```bash
docker exec -d sglang_dev bash -c "
FLASHINFER_DISABLE_VERSION_CHECK=1 python -m sglang.launch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/Qwen3-Next-80B-A3B-Instruct \
    --tp 8 \
    --host 127.0.0.1 \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.70 \
    --disable-radix-cache \
    2>&1 | tee /tmp/server_baseline_nonspec.log
"
```

Optimized (K-last) -- run in `sglang_klast`:
```bash
docker exec -d sglang_klast bash -c "
python -m sglang.launch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/Qwen3-Next-80B-A3B-Instruct \
    --tp 8 \
    --host 127.0.0.1 \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.70 \
    --disable-radix-cache \
    --mamba-ssm-k-last \
    2>&1 | tee /tmp/server_optimized_nonspec.log
"
```

#### Spec mode (with speculative decoding - EAGLE)

Baseline (V-last) -- run in `sglang_dev`:
```bash
docker exec -d sglang_dev bash -c "
FLASHINFER_DISABLE_VERSION_CHECK=1 python -m sglang.launch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/Qwen3-Next-80B-A3B-Instruct \
    --tp 8 \
    --host 127.0.0.1 \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.70 \
    --disable-radix-cache \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    2>&1 | tee /tmp/server_baseline_spec.log
"
```

Optimized (K-last) -- run in `sglang_klast`:
```bash
docker exec -d sglang_klast bash -c "
python -m sglang.launch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/Qwen3-Next-80B-A3B-Instruct \
    --tp 8 \
    --host 127.0.0.1 \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.70 \
    --disable-radix-cache \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --mamba-ssm-k-last \
    2>&1 | tee /tmp/server_optimized_spec.log
"
```

### Verify CUDA Graph is Enabled

After server starts, check the server log for CUDA graph status:
```bash
docker exec <container> bash -c "grep -i 'cuda.*graph' /tmp/server*.log | head -5"
```
CUDA graph is enabled by default (do NOT pass `--disable-cuda-graph`). The decode/verify phase must have CUDA graph enabled for valid performance comparison.

---

## Part 1: Performance Test

**Tool**: `bench_one_batch_server`
**Parameters**: `batch-size=[1,4,8,16,32], input-len=1024, output-len=128, disable radix cache`

### Steps (repeat for both spec and non-spec modes)

1. **Checkout baseline commit, install, launch V-last server in `sglang_dev` (see Common Setup)**
2. **Run benchmark against baseline (from `sglang_dev`):**
```bash
docker exec sglang_dev bash -c "
python -m sglang.bench_one_batch_server \
    --model None \
    --base-url http://127.0.0.1:30000 \
    --batch-size 1 4 8 16 32 \
    --input-len 1024 \
    --output-len 128 \
    --result-filename /tmp/baseline_result.jsonl \
    --show-report
"
```

3. **Kill V-last server, checkout optimized commit, install, launch K-last server in `sglang_klast` (see Common Setup)**
4. **Run benchmark against optimized (from `sglang_klast`):**
```bash
docker exec sglang_klast bash -c "
python -m sglang.bench_one_batch_server \
    --model None \
    --base-url http://127.0.0.1:30000 \
    --batch-size 1 4 8 16 32 \
    --input-len 1024 \
    --output-len 128 \
    --result-filename /tmp/optimized_result.jsonl \
    --show-report
"
```

### Analyzing Results

From the server log, extract prefill and decode batch metrics:
```bash
# Prefill batch info
docker exec <container> bash -c "grep -E '(prefill|Prefill).*batch' /tmp/server*.log | tail -20"

# Decode batch info
docker exec <container> bash -c "grep -E '(decode|Decode).*batch' /tmp/server*.log | tail -20"
```

**Key metrics from `bench_one_batch_server` output:**
- `input throughput` (tok/s) - prefill throughput
- `output throughput` (tok/s) - decode throughput
- `last_ttft` - time to first token (prefill latency indicator)
- `latency` - overall latency
- `acc_length` - accept length (only meaningful in spec mode)

**Organizing results**: Create a table separating prefill and decode metrics for baseline vs optimized, for both spec and non-spec modes. This helps identify whether the performance gain is from prefill or decode phase.

| Mode | Phase | Baseline (V-last) | Optimized (K-last) | Gain |
|------|-------|--------------------|---------------------|------|
| Non-spec | Prefill throughput (tok/s) | ... | ... | ...% |
| Non-spec | Decode throughput (tok/s) | ... | ... | ...% |
| Spec | Prefill throughput (tok/s) | ... | ... | ...% |
| Spec | Decode throughput (tok/s) | ... | ... | ...% |
| Spec | Accept length | ... | ... | ... |

---

## Part 2: Torch Profile Trace

**Tool**: `bench_one_batch_server --profile`
**Output**: Traces saved to `/home/xutingz/workspace/torch_profile/gdn/<TIMESTAMP>/`
**Requirement**: Prefill and decode steps in one trace file. Multi-rank traces merged into one merged trace.

### Steps (repeat for both spec and non-spec modes)

1. **Server must be running** (from Part 1, or launch separately).
2. **Run profiling for baseline (from `sglang_dev`):**
```bash
docker exec sglang_dev bash -c "
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
TRACE_DIR='/home/xutingz/workspace/torch_profile/gdn/\${TIMESTAMP}_baseline'
mkdir -p \${TRACE_DIR}

python -m sglang.bench_one_batch_server \
    --model None \
    --base-url http://127.0.0.1:30000 \
    --batch-size 1 4 8 16 32 \
    --input-len 1024 \
    --output-len 128 \
    --profile \
    --profile-steps 5 \
    --profile-output-dir \${TRACE_DIR} \
    --profile-prefix baseline
"
```

3. **Kill V-last server, switch to K-last in `sglang_klast`, run profiling:**
```bash
docker exec sglang_klast bash -c "
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
TRACE_DIR='/home/xutingz/workspace/torch_profile/gdn/\${TIMESTAMP}_optimized'
mkdir -p \${TRACE_DIR}

python -m sglang.bench_one_batch_server \
    --model None \
    --base-url http://127.0.0.1:30000 \
    --batch-size 1 4 8 16 32 \
    --input-len 1024 \
    --output-len 128 \
    --profile \
    --profile-steps 5 \
    --profile-output-dir \${TRACE_DIR} \
    --profile-prefix optimized
"
```

### Alternative: Profile with prefill/decode separated using `--profile-by-stage`

```bash
docker exec <container> bash -c "
python -m sglang.bench_one_batch_server \
    --model None \
    --base-url http://127.0.0.1:30000 \
    --batch-size 1 4 8 16 32 \
    --input-len 1024 \
    --output-len 128 \
    --profile \
    --profile-steps 5 \
    --profile-by-stage \
    --profile-output-dir \${TRACE_DIR} \
    --profile-prefix <baseline|optimized>
"
```

### Alternative: Use `sglang.profiler` module directly for merge control

```bash
docker exec <container> bash -c "
python -m sglang.profiler \
    --url http://127.0.0.1:30000 \
    --output-dir \${TRACE_DIR} \
    --num-steps 5 \
    --gpu \
    --cpu \
    --merge-profiles
"
```

### Trace File Details

- Individual rank traces: `{profile_id}-TP-{tp}-DP-{dp}-PP-{pp}-EP-{ep}.trace.json.gz`
- Merged trace (all ranks): `merged-{profile_id}.trace.json.gz`
- The merge happens automatically when using `bench_one_batch_server --profile` (the server-side profiler handles merge via `profile_merger.py`)
- View traces at: https://perfetto.dev or `chrome://tracing`

---

## Part 3: Accuracy Test

**Tool**: `sglang.test.run_eval` (MMLU evaluation)
**Requirement**: Compare MMLU scores between baseline and optimized. For spec mode, also compare accept length.

### Steps (repeat for both spec and non-spec modes)

1. **Server must be running** (launch as described in Common Setup).

2. **Run MMLU eval for baseline (from `sglang_dev`):**
```bash
docker exec sglang_dev bash -c "
python -m sglang.test.run_eval \
    --eval-name mmlu \
    --num-examples 1000 \
    --num-threads 128 \
    --max-tokens 512 \
    --host 127.0.0.1 \
    --port 30000
"
```

3. **Kill V-last server, launch K-last server in `sglang_klast`, run MMLU eval:**
```bash
docker exec sglang_klast bash -c "
python -m sglang.test.run_eval \
    --eval-name mmlu \
    --num-examples 1000 \
    --num-threads 128 \
    --max-tokens 512 \
    --host 127.0.0.1 \
    --port 30000
"
```

### Key Accuracy Metrics

- **MMLU Score**: from `run_eval` output line `Score: <float>`
- **Total latency**: from output line `Total latency: <float>`
- **Accept length** (spec mode only): obtained from `bench_one_batch_server` output (`acc_length` field) or from server info endpoint (`avg_spec_accept_length`)

### Verifying Accept Length (spec mode only)

After running `bench_one_batch_server` in spec mode, the accept length is reported in the output. You can also query it directly:
```bash
curl -s http://127.0.0.1:30000/get_server_info | python -m json.tool | grep spec_accept
```

### Organizing Accuracy Results

| Mode | Metric | Baseline (V-last) | Optimized (K-last) | Difference |
|------|--------|--------------------|---------------------|------------|
| Non-spec | MMLU Score | ... | ... | ... |
| Spec | MMLU Score | ... | ... | ... |
| Spec | Accept Length | ... | ... | ... |

**Expected**: Optimized (K-last) should have MMLU score within acceptable tolerance of baseline (V-last). Accept length should be comparable (similar or better).

---

## Full E2E Workflow Summary

The complete benchmark requires running all 3 parts for both spec and non-spec modes. The flow uses two containers to avoid git conflicts:

### For each mode (non-spec, spec):

1. **Baseline (V-last)** -- in `sglang_dev`:
   - `docker exec sglang_dev bash -c "cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang_gdn && git checkout b5493f65be22447af02edbfb6eb4cee71f2bc779"`
   - `docker exec sglang_dev bash -c "cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang_gdn && pip install -e 'python[all]' --no-build-isolation -q"`
   - Launch V-last server in `sglang_dev` (without `--mamba-ssm-k-last`)
   - Run performance benchmark (`bench_one_batch_server`) from `sglang_dev`
   - Run torch profiling (`bench_one_batch_server --profile`) from `sglang_dev`
   - Run MMLU accuracy test (`sglang.test.run_eval`) from `sglang_dev`
   - Kill server

2. **Optimized (K-last)** -- in `sglang_klast`:
   - `docker exec sglang_klast bash -c "cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang_gdn && git checkout f043e4ee1786f4fc7d58623fa5400d981697fe4e"`
   - `docker exec sglang_klast bash -c "cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang_gdn && pip install -e 'python[all]' --no-build-isolation -q"`
   - Launch K-last server in `sglang_klast` (with `--mamba-ssm-k-last`)
   - Run performance benchmark (`bench_one_batch_server`) from `sglang_klast`
   - Run torch profiling (`bench_one_batch_server --profile`) from `sglang_klast`
   - Run MMLU accuracy test (`sglang.test.run_eval`) from `sglang_klast`
   - Kill server

3. **Compare results** across all dimensions.

---

## Existing Helper Scripts

The repo already contains several benchmark scripts that can be referenced or reused:

- **Full E2E comparison**: `benchmark/run_klast_vlast_e2e_comparison.sh`
- **Baseline-only benchmark**: `benchmark/run_baseline_vlast_benchmark.sh`
- **Quick comparison**: `run_benchmark_comparison.sh`
- **Precision comparison (Python)**: `benchmark/compare_klast_vlast_precision.py`
  - Supports `--mmlu-eval`, `--gsm8k-eval`, `--enable-profile`, `--disable-speculative`
  - Example: `docker exec sglang_klast bash -c "python benchmark/compare_klast_vlast_precision.py --model-path /lustre/raplab/client/xutingz/workspace/model/Qwen3-Next-80B-A3B-Instruct --tp 8 --mmlu-eval --enable-profile"`

---

## Troubleshooting

### FlashInfer GDN kernels not available in `sglang_klast`

If you see `No module named 'flashinfer.gdn_prefill'`, the source FlashInfer install may have been overwritten:
```bash
# Verify FlashInfer is from source
docker exec sglang_klast python3 -c "import flashinfer; print(flashinfer.__file__)"
# Should print: /lustre/raplab/client/xutingz/workspace/gitsrc/flashinfer/flashinfer/__init__.py

# If not, reinstall from source:
docker exec sglang_klast bash -c "
pip uninstall flashinfer-python flashinfer -y
rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_cubin
cd /lustre/raplab/client/xutingz/workspace/gitsrc/flashinfer
pip install -e . --no-build-isolation -q
"
```

### Port conflict between containers

Both containers share `--network host`. Only one server can listen on port 30000:
```bash
# Kill server in both containers
docker exec sglang_dev bash -c "lsof -ti:30000 | xargs -r kill -9 2>/dev/null || true"
docker exec sglang_klast bash -c "lsof -ti:30000 | xargs -r kill -9 2>/dev/null || true"
```

### Container not running

```bash
# Check status
docker ps -a --filter name=sglang_klast --format "{{.Names}} {{.Status}}"
docker ps -a --filter name=sglang_dev --format "{{.Names}} {{.Status}}"

# Restart if needed
docker start sglang_klast
docker start sglang_dev
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `python/sglang/bench_one_batch_server.py` | Performance benchmark with server |
| `python/sglang/profiler.py` | Torch profiler module |
| `python/sglang/test/run_eval.py` | MMLU and other eval scripts |
| `python/sglang/srt/server_args.py` | All server CLI arguments |
| `python/sglang/srt/utils/profile_merger.py` | Multi-rank trace merger |
| `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | GDN attention backend (K-last/V-last logic) |
| `benchmark/compare_klast_vlast_precision.py` | Full comparison Python script |
| `benchmark/run_klast_vlast_e2e_comparison.sh` | E2E comparison shell script |
| `docs/developer_guide/benchmark_and_profiling.md` | Profiling documentation |
