---
name: benchmark-and-profile-reference
description: Reference commands and workflow for denoise benchmarks and profiling in SGLang Diffusion.
---

# SGLang Diffusion Benchmark and Profile Guide

**Primary Metric: Denoise Latency**
- Denoise latency is the total DiT forward-pass time across all inference steps.
- It is the dominant cost for diffusion inference, typically more than 80% of end-to-end time.
- It is the **sole optimization target** for kernel work.
- End-to-end latency is a secondary sanity check only.

> **Correctness First**: Faster but incorrect output is not an improvement. Always compare generated images/videos against a reference baseline before and after any change.

---

## Prerequisites

```bash
ENV_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/diffusion_skill_env.py
ROOT=$(python3 "$ENV_PY" print-root)
cd "$ROOT"
python3 "$ENV_PY" check-write-access >/dev/null

export HF_TOKEN=<your_hf_token>  # required for gated repos such as black-forest-labs/FLUX.*
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=$(python3 "$ENV_PY" print-idle-gpus --count 1)

ASSET_DIR=$(python3 "$ENV_PY" print-assets-dir --mkdir)
BENCH_DIR=$(python3 "$ENV_PY" print-output-dir --kind benchmarks --mkdir)
PROFILE_DIR=$(python3 "$ENV_PY" print-output-dir --kind profiles --mkdir)
NCU_DIR=$(python3 "$ENV_PY" print-output-dir --kind ncu --mkdir)
export PROFILE_DIR

check() {
  local label="$1"
  shift
  "$@" &>/dev/null && echo "[OK]  $label" || echo "[MISS] $label"
}

check "sglang" python3 -c "import sglang"
check "torch+CUDA" python3 -c "import torch; assert torch.cuda.is_available()"
check "torch.profiler" python3 -c "import torch.profiler"
check "nsys (Level 2)" which nsys
check "ncu (Level 3)" which ncu
check "pandas" python3 -c "import pandas"
check "plotly" python3 -c "import plotly"
check "regex" python3 -c "import regex"
```

Environment notes:
- **Minimum for benchmarking**: `sglang`, `torch` with CUDA.
- **Level 1 profiling**: `torch.profiler` (bundled with torch).
- **Level 2 profiling**: `nsys`, `pandas`, `plotly`, `regex`, and `gputrc2graph.py` from the sglang repo.
- All commands below assume you are inside the configured diffusion container shell and already `cd`'d to the repo root derived from `sglang.__file__`.
- Export `HF_TOKEN` before running any command against a gated Hugging Face repo such as `black-forest-labs/FLUX.*`. Without it, the top-level `sglang generate` auto-detection can fail before model loading and report a misleading `Generate subcommand is not yet supported for model ...`.
- Export `FLASHINFER_DISABLE_VERSION_CHECK=1` before any benchmark or profiler command.
- Re-run `print-idle-gpus` before each perf command if GPU availability may have changed.
- Keep benchmark commands within 4 GPUs or fewer.

Download input images required by some models:
```bash
wget -O "${ASSET_DIR}/cat.png" \
  https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png
wget -O "${ASSET_DIR}/astronaut.jpg" \
  https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg
wget -O "${ASSET_DIR}/mova_single_person.jpg" \
  https://github.com/OpenMOSS/MOVA/raw/main/assets/single_person.jpg
```

---

## Benchmark Commands

All commands include `--warmup` and `--enable-torch-compile` for real production performance. Add `--perf-dump-path <file>.json` for machine-readable output.

If you want a checked-in preset runner instead of copying commands manually, use `scripts/bench_diffusion_denoise.py --model <preset> --label <name>`. It writes the same perf dump JSONs used by `compare_perf.py`.

### Perf dump & before/after compare

For every benchmark run, always write a perf dump JSON:

```bash
sglang generate ... --warmup --perf-dump-path "${BENCH_DIR}/<result>.json"
```

Before/after comparison (outputs a Markdown table suitable for PR descriptions):

```bash
# Baseline (on main branch or before changes)
sglang generate ... --warmup --perf-dump-path "${BENCH_DIR}/baseline.json"

# New (after changes)
sglang generate ... --warmup --perf-dump-path "${BENCH_DIR}/new.json"

python3 python/sglang/multimodal_gen/benchmarks/compare_perf.py \
  "${BENCH_DIR}/baseline.json" "${BENCH_DIR}/new.json"
```

### Qwen-Image-2512 (1024×1024, 50 steps)
```bash
sglang generate \
  --model-path=Qwen/Qwen-Image-2512 \
  --prompt="A futuristic cyberpunk city at night, neon lights reflecting on wet streets, highly detailed, 8k" \
  '--negative-prompt= ' \
  --width=1024 --height=1024 --num-inference-steps=50 --guidance-scale=4.0 \
  --seed=42 --save-output --enable-torch-compile --warmup \
  --dit-cpu-offload false --text-encoder-cpu-offload false
```

### Qwen-Image-Edit-2511 (image editing, 1024×1024, 50 steps)
```bash
sglang generate \
  --model-path=Qwen/Qwen-Image-Edit-2511 \
  '--prompt=Transform into anime style' '--negative-prompt= ' \
  --image-path="${ASSET_DIR}/cat.png" \
  --width=1024 --height=1024 --num-inference-steps=50 --guidance-scale=4.0 \
  --seed=42 --save-output --enable-torch-compile --warmup \
  --dit-cpu-offload false --text-encoder-cpu-offload false
```

### FLUX.1-dev (1024×1024, 50 steps)
```bash
sglang generate \
  --model-path=black-forest-labs/FLUX.1-dev \
  --prompt="A futuristic cyberpunk city at night, neon lights reflecting on wet streets, highly detailed, 8k" \
  --width=1024 --height=1024 --num-inference-steps=50 --guidance-scale=4.0 \
  --seed=42 --save-output --enable-torch-compile --warmup
```

### FLUX.2-dev (1024×1024)
```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --width=1024 --height=1024 \
  --dit-layerwise-offload false --enable-torch-compile --warmup \
  --dit-cpu-offload false --text-encoder-cpu-offload true --vae-cpu-offload false
```

### Z-Image-Turbo (1024×1024, 9 steps)
```bash
sglang generate \
  --model-path=Tongyi-MAI/Z-Image-Turbo \
  --prompt='A fantasy landscape with mountains and a river, detailed, vibrant colors' \
  --width=1024 --height=1024 --num-inference-steps=9 --guidance-scale=0.0 \
  --seed=42 --save-output --enable-torch-compile --warmup \
  --dit-cpu-offload false --text-encoder-cpu-offload false
```

### Wan2.2-T2V-A14B 720P (4 GPUs, 81 frames, 2 steps)
```bash
# Select four idle GPUs first:
# export CUDA_VISIBLE_DEVICES=$(python3 "$ENV_PY" print-idle-gpus --count 4)
sglang generate \
  --model-path=Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon." \
  --negative-prompt=" " --720p --num-inference-steps=2 --num-frames=81 \
  --guidance-scale=5.0 --seed=42 --save-output \
  --num-gpus=4 --ulysses-degree=4 \
  --text-encoder-cpu-offload --pin-cpu-memory \
  --warmup --enable-torch-compile
```

### Wan2.2-TI2V-5B 720P (single GPU, 81 frames, 50 steps)
```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --prompt "An astronaut hatching from an egg, on the surface of the moon..." \
  --negative-prompt "Bright tones, overexposed, static, blurred details..." \
  --image-path="${ASSET_DIR}/astronaut.jpg" \
  --num-frames 81 --720p --num-inference-steps 50 --guidance-scale 5.0 \
  --seed 42 --save-output \
  --dit-layerwise-offload false --dit-cpu-offload false \
  --vae-cpu-offload false --text-encoder-cpu-offload false \
  --enable-torch-compile --warmup
```

### HunyuanVideo (848×480, 65 frames, 30 steps)
```bash
sglang generate \
  --model-path=hunyuanvideo-community/HunyuanVideo \
  --text-encoder-cpu-offload --pin-cpu-memory \
  --prompt="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window." \
  --save-output --num-frames=65 --width=848 --height=480 \
  --num-inference-steps=30 \
  --warmup --enable-torch-compile
```

### MOVA-720p (4 GPUs, 193 frames, 2 steps)
```bash
# Select four idle GPUs first:
# export CUDA_VISIBLE_DEVICES=$(python3 "$ENV_PY" print-idle-gpus --count 4)
sglang generate \
  --model-path=OpenMOSS-Team/MOVA-720p \
  --prompt="A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks. At one point, he says, \"I would also believe that this advance in AI recently wasn’t unexpected.\"" \
  --image-path="${ASSET_DIR}/mova_single_person.jpg" \
  --adjust-frames=false \
  --num-gpus=4 --ring-degree=1 --ulysses-degree=4 \
  --num-frames=193 --fps=24 \
  --num-inference-steps=2 \
  --enable-torch-compile --save-output --warmup
```

**Key metrics** (all models): denoise latency ★, end-to-end latency, peak GPU memory.

---

## Performance Bottleneck Workflow

### Step 1: Identify the Slow DiT Operation

Add `--log-level=info` and observe:
- **Denoise loop latency** ★ — primary target
- Per-step DiT latency — denoise ÷ steps

### Step 2: Profile with torch.profiler (Level 1)

**Compile-safety rule for fused or rewritten kernels**
- Any new kernel must be checked for `torch.compile` graph breaks before trusting its benchmark result.
- If a direct Python/library call triggers tracing issues, wrap it as a custom op first.
- For external libraries, use `register_custom_op_from_extern(...)`.
- For SGLang JIT kernels, use `@register_custom_op(...)` and keep the JIT/module loading inside the custom op body.
- Re-run `torch._dynamo.explain` on representative shapes and verify the optimized path still gets `graph_count=1` and `graph_break_count=0`.

```bash
SGLANG_TORCH_PROFILER_DIR="${PROFILE_DIR}/torch" \
sglang generate \
  --model-path=black-forest-labs/FLUX.1-dev \
  --prompt="A futuristic cyberpunk city at night" \
  --width=1024 --height=1024 --num-inference-steps=50 \
  --seed=42 --enable-torch-compile --warmup \
  --profile --num-profiled-timesteps 3
```

Parse the trace without a browser:
```python
import gzip, json, collections, glob, os

log_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR", "./logs")
trace_path = sorted(glob.glob(f"{log_dir}/*.trace.json.gz"), key=os.path.getmtime, reverse=True)[0]
with gzip.open(trace_path, "rb") as f:
    data = json.loads(f.read())

cuda_ops = collections.defaultdict(lambda: {"total_us": 0, "count": 0})
for e in data.get("traceEvents", []):
    if e.get("cat") in ("kernel", "gpu_memcpy") and "dur" in e:
        cuda_ops[e.get("name","unknown")]["total_us"] += e["dur"]
        cuda_ops[e.get("name","unknown")]["count"] += 1

print(f"{'Kernel':<80} {'Total(ms)':>10} {'Count':>6}")
for name, s in sorted(cuda_ops.items(), key=lambda x: -x[1]["total_us"])[:30]:
    print(f"{name:<80} {s['total_us']/1000:>10.3f} {s['count']:>6}")
```

Add `record_function` scopes in the DiT block for per-layer attribution:
```python
with torch.profiler.record_function(f"dit_block_{idx}.attn"):
    x = self.attn(x)
with torch.profiler.record_function(f"dit_block_{idx}.norm"):
    x = self.norm(x)
```

**Expected dominant kernels per DiT sub-component:**

| Sub-component | Expected kernel |
|--------------|-----------------|
| QKV / output / MLP projections | `cutlass_gemm` / `ampere_*_gemm` |
| Attention | `flash_attn_fwd` / `fmha_*` (FA3/FA4) |
| AdaLN modulation | `fuse_scale_shift_kernel` |
| RMSNorm / LayerNorm | `sgl_kernel_rmsnorm` / Triton norm |
| SiLU gate | `vectorized_elementwise_kernel` |
| RoPE | `apply_rotary_embedding` (Triton) |
| QK Norm | `fused_inplace_qknorm` (JIT) |

### Step 3: Deep CUDA Kernel Breakdown (Level 2 — nsys)

Workflow:
- **Pass A**: collect the `nsys` trace.
- **Pass B**: measure wall-clock runtime without profiling.
- Write the non-profiled wall-clock time into `ELAPSED_SEC`.

```bash
# Pass A — collect nsys trace (skip warmup with --delay)
nsys profile -t cuda -o "${PROFILE_DIR}/flux_dev" -f true \
  --trace-fork-before-exec=true --delay 120 --duration 60 \
  sglang generate \
    --model-path=black-forest-labs/FLUX.1-dev \
    --prompt="A futuristic cyberpunk city at night" \
    --width=1024 --height=1024 --num-inference-steps=50 \
    --seed=42 --enable-torch-compile --warmup

# Pass B — measure wall-clock time without profiling
time sglang generate --model-path=black-forest-labs/FLUX.1-dev \
  --width=1024 --height=1024 --num-inference-steps=50 --seed=42 \
  --enable-torch-compile --warmup
# Record ELAPSED_SEC from Pass B
```

Create classification JSON at `examples/profiler/nsys_profile_tools/sglang_diffusion_engine_model.json`:
```json
{
  "sglang": {
    "diffusion": {
      "gemm|nvjet|cutlass": "gemm",
      "flash|fmha|fwd_flash": "attn",
      "fuse_scale_shift|scale_shift_gate": "adaln_modulation",
      "_norm_|Norm|rmsnorm|fused_add_rmsnorm": "norm",
      "rotary|rope": "rope",
      "act_and_mul|silu|gelu": "activation",
      "ncclDevKernel|all_gather|all_reduce": "nccl_comm",
      "triton": "triton_kernel",
      "CUDA mem": "non-gpu-H_D_memops",
      ".*": "misc"
    }
  }
}
```

Notes:
- `gputrc2graph.py` only recognizes `sglang,diffusion,...` after this JSON file exists in `examples/profiler/nsys_profile_tools/`.
- If you only want a quick structural check, set `ELAPSED_SEC=0`. The report will still generate, but `CPU(non-GPU)` time can be inflated.

Run analysis:
```bash
ELAPSED_SEC=12.34
cd "$ROOT/examples/profiler/nsys_profile_tools"
python3 gputrc2graph.py \
  --in_file "${PROFILE_DIR}/flux_dev.nsys-rep,sglang,diffusion,${ELAPSED_SEC}" \
  --out_dir "${PROFILE_DIR}/analysis" \
  --title "FLUX.1-dev denoise kernel breakdown"

# Read results
python3 - << 'EOF'
import os
import pandas as pd

df = pd.read_csv(f"{os.environ['PROFILE_DIR']}/analysis/result.csv")
summary = df.groupby("Category")["Elapsed Time (sec)"].sum().sort_values(ascending=False)
total = summary.sum()
for cat, sec in summary.items():
    print(f"{cat:<30} {sec:>8.3f}s  ({sec/total*100:>5.1f}%)")
EOF
```

**What the category breakdown tells you:**

| Category high | Investigation |
|--------------|---------------|
| `gemm` dominant | Check tensor parallelism; QKV/MLP bottleneck |
| `attn` dominant | Verify FA3/FA4 is active |
| `adaln_modulation` high | Verify fused `fuse_scale_shift_kernel` is used |
| `norm` high | Verify `sgl_kernel_rmsnorm` / CuTe DSL path; check D alignment |
| `nccl_comm` high | Multi-GPU: tune Ulysses degree |
| `triton_kernel` high | Identify which Triton kernel; consider CUDA replacement |
| `non-gpu-H_D_memops` high | Accidental CPU offload or `.cpu()` calls mid-denoising |
| `CPU(non-GPU)` high | Python dispatch overhead / torch.compile graph breaks |

### Step 3.5: Per-Kernel Deep Analysis (Level 3 — ncu)

**CRITICAL**: `ncu` (Nsight Compute) is the essential tool for kernel-level optimization. While nsys and torch.profiler tell you **which** kernels are slow, only ncu tells you **why** — memory bandwidth utilization, compute throughput, occupancy limiters, warp stall reasons, and roofline position. **Always use ncu when optimizing or writing custom kernels.**

#### When to use ncu

- After writing a new Triton or CUDA kernel — verify it saturates hardware bandwidth
- When a kernel shows up as a top bottleneck in Level 1/2 profiling
- When comparing your fused kernel vs PyTorch baseline or torch.compile output
- When tuning Triton autotune configs (block sizes, num_warps)
- When profiling `sglang generate`, add `--target-processes all` so child worker processes are included

#### Basic ncu workflow

```bash
# 1. Profile a specific kernel by name (skip warmup launches, collect 3 invocations)
ncu --target-processes all \
    --kernel-name "_fused_gated_residual_add_kernel" \
    --launch-skip 10 --launch-count 3 \
    --set full \
    -o "${NCU_DIR}/gated_residual" \
    sglang generate \
      --model-path=black-forest-labs/FLUX.1-dev \
      --prompt="test" --width=1024 --height=1024 \
      --num-inference-steps=5 --seed=42

# 2. Profile all kernels in a short run (use few steps to limit time)
ncu --target-processes all \
    --launch-skip 50 --launch-count 200 \
    --set full \
    -o "${NCU_DIR}/all_kernels" \
    sglang generate \
      --model-path=black-forest-labs/FLUX.1-dev \
      --prompt="test" --width=1024 --height=1024 \
      --num-inference-steps=3 --seed=42

# 3. For CUDA graph mode, keep --graph-profiling=node on the ncu side.
# Note: `--enable-piecewise-cuda-graph` is a server flag, not a valid
# `sglang generate` flag, so do not append it here.
ncu --target-processes all \
    --graph-profiling node \
    --kernel-name "_fused_gated_residual_add_kernel" \
    --launch-skip 5 --launch-count 3 \
    --set full \
    -o "${NCU_DIR}/gated_residual_cudagraph" \
    sglang generate \
      --model-path=black-forest-labs/FLUX.1-dev \
      --prompt="test" --width=1024 --height=1024 \
      --num-inference-steps=5 --seed=42
```

#### Reading ncu results (CLI, no GUI needed)

```bash
# Summary of all profiled kernels
ncu --import "${NCU_DIR}/gated_residual.ncu-rep" --page raw --csv 2>/dev/null | head -50

# Key metrics to extract:
ncu --import "${NCU_DIR}/gated_residual.ncu-rep" \
    --page details --csv 2>/dev/null | python3 -c "
import csv, sys
reader = csv.DictReader(sys.stdin)
key_metrics = {
    'gpu__time_duration.avg': 'Duration',
    'sm__throughput.avg.pct_of_peak_sustained_elapsed': 'Compute (SM) Throughput',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed': 'DRAM Throughput',
    'l1tex__throughput.avg.pct_of_peak_sustained_elapsed': 'L1/TEX Cache Throughput',
    'sm__warps_active.avg.pct_of_peak_sustained_active': 'Achieved Occupancy',
    'launch__occupancy_limit_registers': 'Block Limit Registers',
    'launch__occupancy_limit_shared_mem': 'Block Limit Shared Mem',
}
for row in reader:
    name = row.get('Metric Name', '')
    if any(alias in name or metric in name for metric, alias in key_metrics.items()):
        print(f'{name:<60} {row.get(\"Metric Value\",\"\")}')
"
```

#### Interpreting ncu results for kernel optimization

| Metric | Good | Action if bad |
|--------|------|--------------|
| DRAM throughput > 80% peak | Memory-bound, near optimal | Already saturating HBM — fuse with adjacent ops to reduce total memory traffic |
| DRAM throughput < 50% peak | Not saturating memory bandwidth | Check coalescing, increase vector width, tune BLOCK sizes |
| SM throughput > 60% peak | Compute-bound, near optimal | Reduce arithmetic, use faster instructions (e.g., FMA) |
| SM throughput < 30% peak | Underutilized compute | Increase occupancy, reduce warp stalls, check instruction mix |
| Achieved occupancy > 50% | Acceptable for most kernels | — |
| Achieved occupancy < 25% | Too few active warps | Reduce register pressure or shared memory; increase block size |

#### Comparing before/after with ncu

```bash
# Profile baseline kernel
ncu --target-processes all \
    --kernel-name "vectorized_elementwise_kernel" \
    --launch-skip 10 --launch-count 3 --set full \
    -o "${NCU_DIR}/baseline" ./program

# Profile optimized kernel
ncu --target-processes all \
    --kernel-name "_fused_gated_residual_add_kernel" \
    --launch-skip 10 --launch-count 3 --set full \
    -o "${NCU_DIR}/optimized" ./program

# Compare key metrics
for report in baseline optimized; do
  echo "=== $report ==="
  ncu --import "${NCU_DIR}/${report}.ncu-rep" \
      --page details --csv 2>/dev/null | grep -E "time_duration|throughput.*pct|occupancy"
done
```

**Decision rule after ncu analysis:**
- Kernel already at >80% DRAM bandwidth → fuse with neighbors to reduce total traffic
- Kernel at <50% DRAM bandwidth → tune block sizes, fix coalescing, increase vectorization
- Kernel compute-bound (SM util high, DRAM low) → reduce FLOPs or switch to a faster algorithm
- Low occupancy → reduce registers (simplify kernel) or increase block size in autotune configs

### Step 4: Apply Kernel Optimization

After pinpointing the slow op, choose the right tool:

| Scenario | Skill to use |
|----------|-------------|
| New fused elementwise, norm variant, RoPE variant | **`sglang-diffusion-triton-kernel`** — Triton JIT, faster iteration, NPU fallback |
| Bandwidth-bound reduction (RMSNorm) needing max vectorization | **`sglang-diffusion-cuda-kernel`** — CUDA JIT with `AlignedVector`, warp reductions |
| Attention or tile-based op needing shared memory tuning | **`sglang-diffusion-cuda-kernel`** — full control over CUDA primitives |
| Slow op already covered by an existing fused kernel | **`existing-fast-paths.md`** — check constraints and enable it |

**Quick decision rule**: start with Triton. Switch to CUDA JIT only when profiling shows Triton can't saturate hardware bandwidth.

Both kernel types use SGLang's JIT compilation:
- **Triton**: `python/sglang/jit_kernel/diffusion/triton/<op>.py`
- **CUDA JIT**: `python/sglang/jit_kernel/csrc/diffusion/<op>.cuh` + wrapper `python/sglang/jit_kernel/diffusion/<op>.py`

### Step 5: torch.compile Coverage

```bash
TORCH_COMPILE_DEBUG=1 sglang generate ...
```
- Dynamic shape changes trigger recompilation → fix resolution and frame count when benchmarking
- `tensor.item()` in conditional branches causes graph breaks → rewrite as tensor ops

### Step 6: Multi-GPU Efficiency (Wan2.2-T2V-A14B / MOVA)

- Verify `--ulysses-degree` evenly divides `--num-gpus`
- Keep the command shape fixed when comparing kernels; for quick checks, reduce only `--num-inference-steps`
- If a run OOMs or jitters because of host contention, first confirm there are no leaked scheduler processes on the chosen GPU set

---

## Optimization Workflow Summary

```
0. BASELINE
   sglang generate --seed=42 --save-output → save reference images/videos
       ↓
1. BENCHMARK
   Run benchmark commands above → record denoise latency baseline
       ↓
2. LEVEL 1 PROFILE (torch.profiler)
   --profile --num-profiled-timesteps 3
   → parse .trace.json.gz → rank ops by CUDA time
   → identify slow DiT layer (norm / attn / mlp / rope / adaln)
       ↓
3. LEVEL 2 PROFILE (nsys + gputrc2graph.py)
   → result.csv category breakdown (gemm / attn / adaln / norm / triton / cpu)
   → confirm where GPU time is concentrated
       ↓
4. LEVEL 3 PROFILE (ncu — per-kernel deep analysis) ★ CRITICAL
   → ncu --set full on target kernel(s)
   → extract DRAM bandwidth util, SM throughput, achieved occupancy
   → determine if kernel is memory-bound, compute-bound, or latency-bound
   → for CUDA graph: use --graph-profiling node
       ↓
5. KERNEL OPTIMIZATION
   Existing fused kernel?  → existing-fast-paths.md
   New Triton kernel?      → sglang-diffusion-triton-kernel
   New CUDA JIT kernel?    → sglang-diffusion-cuda-kernel
   After writing kernel    → ncu again to verify bandwidth/occupancy ★
       ↓
6. VERIFY CORRECTNESS
   sglang generate --seed=42 --save-output → diff against reference
   If output differs beyond tolerance → reject optimization
       ↓
7. RE-BENCHMARK
   Verify denoise latency improvement; no regression on other models
```

---

## Checklist Before Merging

### Correctness (must pass first)
- [ ] Reference outputs collected with `--seed=42 --save-output` **before** any change
- [ ] After change: regenerate with identical args and compare
- [ ] No visible quality degradation in generated images / videos
- [ ] Correctness verified on all benchmark models

### Performance (only after correctness passes)
- [ ] All benchmark models executed; denoise latency ★, end-to-end, peak memory recorded
- [ ] No regression in denoise latency vs. previous baseline (±2% tolerance)
- [ ] New kernel shows measurable improvement on at least 2 models
- [ ] No new torch.compile graph breaks introduced
- [ ] Results reproducible with all offloads disabled and fixed `--seed=42`
