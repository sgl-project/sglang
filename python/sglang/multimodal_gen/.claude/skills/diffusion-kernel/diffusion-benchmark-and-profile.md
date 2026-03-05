---
name: diffusion-benchmark-and-profile
description: Denoise-stage benchmark and per-layer kernel profiling guide for SGLang Diffusion models. Use when measuring denoising latency, profiling DiT kernel breakdown with torch.profiler or nsys+gputrc2graph.py, investigating performance bottlenecks, or optimizing with custom Triton/CUDA kernels. Always verify output correctness before and after any optimization.
---

# SGLang Diffusion Benchmark and Profile Guide

**Primary Metric: Denoise Latency**
The denoising loop latency — total DiT forward pass time across all inference steps — is the dominant cost (>80% of end-to-end) and the **sole optimization target** for kernel work. End-to-end latency is recorded as a secondary check only.

> **Correctness First**: Faster but incorrect output is not an improvement. Always compare generated images/videos against a reference baseline before and after any change.

---

## Prerequisites

```bash
#!/usr/bin/env bash
# Quick dependency check
check() { "$@" &>/dev/null && echo "[OK]  $1" || echo "[MISS] $1"; }
check "sglang"         python3 -c "import sglang"
check "torch+CUDA"     python3 -c "import torch; assert torch.cuda.is_available()"
check "torch.profiler" python3 -c "import torch.profiler"
check "nsys (Level 2)" which nsys
check "pandas"         python3 -c "import pandas"
check "plotly"         python3 -c "import plotly"
```

**Minimum for benchmarking**: `sglang`, `torch` with CUDA.
**Level 1 profiling**: `torch.profiler` (bundled with torch).
**Level 2 profiling**: `nsys`, `pandas`, `plotly` + `gputrc2graph.py` from the sglang repo.

Download input images required by some models:
```bash
mkdir -p /workspace/gen_benchmark/figs
wget -O /workspace/gen_benchmark/figs/cat.png \
  https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png
wget -O /workspace/gen_benchmark/figs/astronaut.jpg \
  https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg
```

---

## Benchmark Commands

All commands include `--warmup` and `--enable-torch-compile` for real production performance. Add `--perf-dump-path <file>.json` for machine-readable output (see `diffusion-perf` skill for comparison).

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
  --image-path=/workspace/gen_benchmark/figs/cat.png \
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

### Wan2.2-T2V-A14B 720P (8 GPUs, 81 frames, 40 steps)
```bash
sglang generate \
  --model-path=Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon." \
  --negative-prompt=" " --720p --num-inference-steps=40 --num-frames=81 \
  --guidance-scale=5.0 --seed=42 --save-output \
  --num-gpus=8 --enable-cfg-parallel --ulysses-degree=4 \
  --dit-layerwise-offload true --dit-cpu-offload false \
  --vae-cpu-offload false --text-encoder-cpu-offload true \
  --warmup --enable-torch-compile
```

### Wan2.2-TI2V-5B 720P (single GPU, 81 frames, 50 steps)
```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --prompt "An astronaut hatching from an egg, on the surface of the moon..." \
  --negative-prompt "Bright tones, overexposed, static, blurred details..." \
  --image-path=/workspace/gen_benchmark/figs/astronaut.jpg \
  --num-frames 81 --720p --num-inference-steps 50 --guidance-scale 5.0 \
  --seed 42 --save-output \
  --dit-layerwise-offload false --dit-cpu-offload false \
  --vae-cpu-offload false --text-encoder-cpu-offload false \
  --enable-torch-compile --warmup
```

**Key metrics** (all models): denoise latency ★, end-to-end latency, peak GPU memory.

---

## Performance Bottleneck Workflow

### Step 1: Identify the Slow DiT Operation

Add `--log-level=info` and observe:
- **Denoise loop latency** ★ — primary target
- Per-step DiT latency — denoise ÷ steps

### Step 2: Profile with torch.profiler (Level 1)

```bash
SGLANG_TORCH_PROFILER_DIR=/workspace/profiles \
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

```bash
# Pass A — collect nsys trace (skip warmup with --delay)
nsys profile -t cuda -o /workspace/profiles/flux_dev -f true \
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

Run analysis:
```bash
cd examples/profiler/nsys_profile_tools
python3 gputrc2graph.py \
  --in_file /workspace/profiles/flux_dev.nsys-rep,sglang,diffusion,ELAPSED_SEC \
  --out_dir /workspace/profiles/analysis \
  --title "FLUX.1-dev denoise kernel breakdown"

# Read results
python3 - << 'EOF'
import pandas as pd
df = pd.read_csv("/workspace/profiles/analysis/result.csv")
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

### Step 4: Apply Kernel Optimization

After pinpointing the slow op, choose the right tool:

| Scenario | Skill to use |
|----------|-------------|
| New fused elementwise, norm variant, RoPE variant | **`add-triton-kernel.md`** — Triton JIT, faster iteration, NPU fallback |
| Bandwidth-bound reduction (RMSNorm) needing max vectorization | **`add-cuda-kernel.md`** — CUDA JIT with `AlignedVector`, warp reductions |
| Attention or tile-based op needing shared memory tuning | **`add-cuda-kernel.md`** — full control over CUDA primitives |
| Slow op already covered by existing fused kernel | **`use-efficient-diffusion-kernels.md`** — check constraints & enable |

**Quick decision rule**: start with Triton. Switch to CUDA JIT only when profiling shows Triton can't saturate hardware bandwidth.

Both kernel types use SGLang's JIT compilation:
- **Triton**: `python/sglang/jit_kernel/diffusion/triton/<op>.py`
- **CUDA JIT**: `python/sglang/jit_kernel/csrc/diffusion/<op>.cuh` + wrapper `python/sglang/jit_kernel/diffusion_<op>.py`

### Step 5: torch.compile Coverage

```bash
TORCH_COMPILE_DEBUG=1 sglang generate ...
```
- Dynamic shape changes trigger recompilation → fix resolution and frame count when benchmarking
- `tensor.item()` in conditional branches causes graph breaks → rewrite as tensor ops

### Step 6: Multi-GPU Efficiency (Wan2.2-T2V-A14B)

- Verify `--ulysses-degree` evenly divides `--num-gpus`
- Confirm `--enable-cfg-parallel` is active (requires `guidance_scale > 1`)
- `--dit-layerwise-offload true` introduces CPU↔GPU transfer overhead; disable when memory permits

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
4. KERNEL OPTIMIZATION
   Existing fused kernel?  → use-efficient-diffusion-kernels.md
   New Triton kernel?      → add-triton-kernel.md
   New CUDA JIT kernel?    → add-cuda-kernel.md
       ↓
5. VERIFY CORRECTNESS
   sglang generate --seed=42 --save-output → diff against reference
   If output differs beyond tolerance → reject optimization
       ↓
6. RE-BENCHMARK
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
