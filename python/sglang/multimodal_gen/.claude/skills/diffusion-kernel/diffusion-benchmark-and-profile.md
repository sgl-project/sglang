---
name: diffusion-benchmark-and-profile
description: End-to-end benchmark and per-layer kernel profiling guide for SGLang Diffusion models. Use when measuring SGLang Diffusion generation performance, running latency benchmarks across Qwen-Image, FLUX, Z-Image-Turbo, and Wan2.2 models, profiling DiT layer kernel breakdown with torch.profiler or nsys+gputrc2graph.py, investigating performance bottlenecks, or tracking performance regressions. Always verify output correctness before and after any optimization.
---

# SGLang Diffusion Benchmark and Profile Guide

**Overview**
This skill covers how to run end-to-end benchmarks for SGLang Diffusion (`sglang generate`) across a standard set of models, profile per-layer kernel execution inside the DiT, and use those results to continuously improve performance. These metrics collectively reflect the overall performance of the current SGLang Diffusion release.

> **Correctness First**: Any performance optimization must be validated for correctness before being considered complete. Faster but incorrect output is not an improvement. Always compare generated images/videos against a reference baseline before and after any change.

**Primary Metric: Denoise Latency**
The most important latency signal is the **denoising loop latency** — the total time spent running the DiT forward pass across all inference steps. This is the dominant cost in every diffusion model and the main target for optimization. End-to-end latency (including VAE decode and text encoding) is also recorded as a secondary metric, but denoising latency is the key indicator of DiT model performance.

---

## Prerequisites

### Tool Dependency Check

Before running any benchmark or profiling command, verify that all required tools are available. Run the following check script:

```bash
#!/usr/bin/env bash
set -euo pipefail
PASS=0; FAIL=0

check() {
  local label=$1; shift
  if "$@" &>/dev/null; then
    echo "  [OK]  $label"
    ((PASS++)) || true
  else
    echo "  [MISS] $label"
    ((FAIL++)) || true
  fi
}

echo "=== SGLang Diffusion Benchmark Prerequisites ==="

# Core runtime
check "sglang CLI"             python3 -c "import sglang"
check "torch"                  python3 -c "import torch; assert torch.cuda.is_available()"
check "CUDA available"         python3 -c "import torch; torch.zeros(1).cuda()"

# Profiling (torch.profiler is built into torch — no extra install needed)
check "torch.profiler"         python3 -c "import torch.profiler"

# nsys (optional, for Level 2 profiling)
check "nsys in PATH"           which nsys

# gputrc2graph.py dependencies
# The tool lives in the sglang repo at examples/profiler/nsys_profile_tools/gputrc2graph.py
# Set SGLANG_REPO to your local sglang repo root, e.g.:
#   export SGLANG_REPO=/workspace/sglang
SGLANG_REPO="${SGLANG_REPO:-$(python3 -c "import sglang, os; print(os.path.abspath(os.path.join(os.path.dirname(sglang.__file__), '../../..')))" 2>/dev/null || echo "")}"
GPUTRC="${SGLANG_REPO}/examples/profiler/nsys_profile_tools/gputrc2graph.py"
check "gputrc2graph.py exists (set SGLANG_REPO if missing)" test -f "$GPUTRC"
check "pandas"                 python3 -c "import pandas"
check "regex"                  python3 -c "import regex"
check "plotly (optional)"      python3 -c "import plotly"

echo ""
echo "Result: $PASS passed, $FAIL missing"
if [ "$FAIL" -gt 0 ]; then
  echo ""
  echo "Install missing dependencies:"
  echo "  pip install pandas regex plotly   # for gputrc2graph.py"
  echo "  # nsys: install NVIDIA Nsight Systems from https://developer.nvidia.com/nsight-systems"
fi
```

**Minimum required for benchmarking**: `sglang`, `torch` with CUDA.
**Additional for Level 1 profiling**: `torch.profiler` (bundled with torch, always available).
**Additional for Level 2 profiling**: `nsys` on PATH, `gputrc2graph.py` present, `pandas`, `regex`.
`plotly` is only needed to generate the HTML chart; `result.csv` is generated regardless.

---

### Download Required Input Images

Some models (image editing / image-guided video generation) require input images:

```bash
mkdir -p /workspace/gen_benchmark/figs
cd /workspace/gen_benchmark/figs

wget -O cat.png https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png
wget -O astronaut.jpg https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg
```

---

## Standard Benchmark Model Suite

All commands include `--warmup` (pre-warm torch.compile) and `--enable-torch-compile` to reflect real production deployment performance. Timing is measured after warmup completes.

### 1. Qwen/Qwen-Image-2512 (Text-to-Image, single GPU)

**Task**: Text-to-Image
**Resolution**: 1024×1024, 50 steps

```bash
sglang generate \
  --model-path=Qwen/Qwen-Image-2512 \
  --prompt="A futuristic cyberpunk city at night, neon lights reflecting on wet streets, highly detailed, 8k" \
  '--negative-prompt= ' \
  --width=1024 \
  --height=1024 \
  --num-inference-steps=50 \
  --guidance-scale=4.0 \
  --seed=42 \
  --save-output \
  --enable-torch-compile \
  --warmup \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload false
```

**Key metrics**: denoise latency (s, primary), end-to-end latency (s/image), peak GPU memory (GB)

---

### 2. Qwen/Qwen-Image-Edit-2511 (Image Editing, single GPU)

**Task**: Text-guided Image Editing
**Prerequisite**: `cat.png` (see Prerequisites)
**Resolution**: 1024×1024, 50 steps

```bash
sglang generate \
  --model-path=Qwen/Qwen-Image-Edit-2511 \
  '--prompt=Transform into anime style' \
  '--negative-prompt= ' \
  --image-path=/workspace/gen_benchmark/figs/cat.png \
  --width=1024 \
  --height=1024 \
  --num-inference-steps=50 \
  --guidance-scale=4.0 \
  --seed=42 \
  --save-output \
  --enable-torch-compile \
  --warmup \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload false
```

**Key metrics**: denoise latency (s, primary), end-to-end latency (s/image), peak GPU memory (GB)

---

### 3. black-forest-labs/FLUX.1-dev (Text-to-Image, single GPU)

**Task**: Text-to-Image
**Resolution**: 1024×1024, 50 steps

```bash
sglang generate \
  --model-path=black-forest-labs/FLUX.1-dev \
  --prompt="A futuristic cyberpunk city at night, neon lights reflecting on wet streets, highly detailed, 8k" \
  --width=1024 \
  --height=1024 \
  --num-inference-steps=50 \
  --guidance-scale=4.0 \
  --seed=42 \
  --save-output \
  --warmup \
  --enable-torch-compile
```

**Key metrics**: denoise latency (s, primary), end-to-end latency (s/image), peak GPU memory (GB)

---

### 4. black-forest-labs/FLUX.2-dev (Text-to-Image, single GPU)

**Task**: Text-to-Image
**Resolution**: 1024×1024

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --width=1024 \
  --height=1024 \
  --dit-layerwise-offload false \
  --enable-torch-compile \
  --warmup \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload true \
  --vae-cpu-offload false
```

**Key metrics**: denoise latency (s, primary), end-to-end latency (s/image), peak GPU memory (GB)

---

### 5. Tongyi-MAI/Z-Image-Turbo (Turbo Text-to-Image, single GPU)

**Task**: Text-to-Image (few-step turbo mode, guidance=0)
**Resolution**: 1024×1024, **9 steps**

```bash
sglang generate \
  --model-path=Tongyi-MAI/Z-Image-Turbo \
  --log-level=info \
  --prompt='A fantasy landscape with mountains and a river, detailed, vibrant colors' \
  --width=1024 \
  --height=1024 \
  --num-inference-steps=9 \
  --guidance-scale=0.0 \
  --seed=42 \
  --save-output \
  --enable-torch-compile \
  --warmup \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload false
```

**Key metrics**: denoise latency (s, primary), end-to-end latency (s/image), peak GPU memory (GB)

---

### 6. Wan-AI/Wan2.2-T2V-A14B-Diffusers 720P (Text-to-Video, 8 GPUs)

**Task**: Text-to-Video
**Resolution**: 720P, 81 frames, 40 steps
**Parallelism**: 8 GPUs, Ulysses degree=4, CFG parallel + layerwise offload

```bash
sglang generate \
  --model-path=Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --log-level=info \
  --prompt="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window." \
  --negative-prompt=" " \
  --720p \
  --num-inference-steps=40 \
  --num-frames=81 \
  --guidance-scale=5.0 \
  --seed=42 \
  --save-output \
  --num-gpus=8 \
  --enable-cfg-parallel \
  --ulysses-degree=4 \
  --dit-layerwise-offload true \
  --dit-cpu-offload false \
  --vae-cpu-offload false \
  --text-encoder-cpu-offload true \
  --warmup \
  --enable-torch-compile
```

**Key metrics**: denoise latency (s, primary), total end-to-end latency (s/video), peak GPU memory per device (GB)

---

### 7. Wan-AI/Wan2.2-TI2V-5B-Diffusers 720P (Text-Image-to-Video, single GPU)

**Task**: Text-Image-to-Video
**Prerequisite**: `astronaut.jpg` (see Prerequisites)
**Resolution**: 720P, 81 frames, 50 steps

```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --log-level info \
  --warmup \
  --dit-layerwise-offload false \
  --dit-cpu-offload false \
  --vae-cpu-offload false \
  --text-encoder-cpu-offload false \
  --enable-torch-compile \
  --prompt "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot." \
  --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
  --image-path=/workspace/gen_benchmark/figs/astronaut.jpg \
  --num-frames 81 \
  --720p \
  --num-inference-steps 50 \
  --guidance-scale 5.0 \
  --seed 42 \
  --save-output
```

**Key metrics**: denoise latency (s, primary), total end-to-end latency (s/video), peak GPU memory (GB)

---

## Result Recording Format

Record results in the following format for cross-version comparison:

Denoise latency is the primary optimization target. End-to-end latency includes text encoding and VAE decode on top of denoise.

```
| Model                             | Task       | Steps | Resolution  | Denoise (s) ★ | E2E (s) | Peak Mem (GB) | SGLang Ver | GPU |
|-----------------------------------|------------|-------|-------------|---------------|---------|---------------|------------|-----|
| Qwen-Image-2512                   | T2I        |  50   | 1024×1024   |               |         |               |            |     |
| Qwen-Image-Edit-2511              | Img Edit   |  50   | 1024×1024   |               |         |               |            |     |
| FLUX.1-dev                        | T2I        |  50   | 1024×1024   |               |         |               |            |     |
| FLUX.2-dev                        | T2I        |   -   | 1024×1024   |               |         |               |            |     |
| Z-Image-Turbo                     | T2I Turbo  |   9   | 1024×1024   |               |         |               |            |     |
| Wan2.2-T2V-A14B 720P (8 GPU)      | T2V        |  40   | 720P 81fr   |               |         |               |            |     |
| Wan2.2-TI2V-5B 720P               | TI2V       |  50   | 720P 81fr   |               |         |               |            |     |
```

★ Denoise latency = total time of the denoising loop (all DiT forward passes). Reported separately from text encoding and VAE decode.

---

## Performance Bottleneck Investigation Workflow

### Step 1: Identify the Slow Stage

Add `--log-level=info` (or `debug`) and observe the following stages in order of priority:

- **Denoise loop latency** ★ — total time across all DiT forward passes; this is the primary optimization target and usually accounts for >80% of end-to-end latency
- **Per-step DiT forward latency** — denoise latency divided by number of steps; useful for pinpointing per-step overhead
- VAE decode latency — runs once after the denoising loop; significant for video models
- Text encoder encoding latency — runs once before the denoising loop
- Warmup / torch.compile compilation time — excluded from all reported latency numbers

Focus optimization effort on denoise latency first. Improvements to VAE or text encoder only matter after denoise is already well-optimized.

### Step 2: Cross-reference Kernel Optimization Skill

After identifying the slow stage, refer to `use-efficient-diffusion-kernels.md`:
- Slow attention → check whether attention backend is FlashAttention (FA3/FA4)
- Slow AdaLN modulation → verify `LayerNormScaleShift` / `fuse_scale_shift_kernel` is active
- Slow RMSNorm → verify `sgl_kernel.rmsnorm` / `fused_add_rmsnorm` is hit
- Slow RoPE → check FlashInfer inplace RoPE or Triton RoPE fallback
- Slow QK Norm → verify `fused_inplace_qknorm` path; confirm `head_dim` is in the supported list (`64, 128, 256, 512, 1024`)

If no existing fused kernel covers the slow operation (e.g., a new elementwise fusion opportunity, a norm variant, or a custom DiT sub-op), implement a new Triton kernel using **`add-triton-kernel.md`**. That skill covers the full workflow: kernel authoring, autotune, `torch.compile` compatibility, NPU fallback, layer integration, and tests.

### Step 3: Check torch.compile Coverage

```bash
TORCH_COMPILE_DEBUG=1 sglang generate ...
```

Key things to watch:
- Dynamic shape changes trigger recompilation — fix resolution and frame count when benchmarking
- Conditional branches containing `tensor.item()` cause graph breaks and must be rewritten

### Step 4: Multi-GPU Parallel Efficiency (Wan2.2-T2V-A14B)

- Verify `--ulysses-degree` evenly divides `--num-gpus`
- Confirm `--enable-cfg-parallel` is active (requires `guidance_scale > 1`)
- Use `torch.distributed` profiling or `nsys` to identify communication bottlenecks
- `--dit-layerwise-offload true` introduces CPU↔GPU transfer overhead; only enable when memory-constrained

### Step 5: Offload Strategy Trade-offs

| Offload Flag                   | Effect                         | Cost                            |
|--------------------------------|--------------------------------|---------------------------------|
| `--dit-cpu-offload true`       | Move all DiT weights to CPU    | Significantly increases per-step latency |
| `--dit-layerwise-offload true` | Load DiT layer-by-layer on demand | Moderate latency, saves GPU memory |
| `--text-encoder-cpu-offload`   | Move text encoder to CPU       | Only affects encoding phase (once per run) |
| `--vae-cpu-offload`            | Move VAE to CPU                | Affects decode phase only       |

When establishing a GPU performance baseline, disable all offloading (`false`).

---

## Per-Layer Kernel Profiling

When denoise latency is higher than expected, the next step is to understand **which layer and which kernel** inside the DiT is responsible. Two profiling levels are used, from coarse to fine:

```
Level 1 — torch.profiler (built-in --profile flag)  →  per-PyTorch-op breakdown, per DiT layer
Level 2 — nsys + gputrc2graph.py                    →  CUDA kernel category breakdown, CPU vs GPU time
```

Use Level 1 to find the slow DiT layer and sub-component; use Level 2 to get the full CUDA kernel category distribution and confirm where GPU time is spent.

---

### Level 1: torch.profiler — Built-in `--profile` Flag

SGLang Diffusion has a built-in profiler (`SGLDiffusionProfiler` in `python/sglang/multimodal_gen/runtime/utils/profiler.py`) that is activated via CLI flags. No driver script needed.

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--profile` | off | Enable torch profiler for the denoising stage |
| `--num-profiled-timesteps N` | 5 | Profile N denoising steps (use `-1` for all steps) |
| `--profile-all-stages` | off | Also profile text encoding and VAE decode stages |

**Output:** gzipped Chrome trace JSON at `$SGLANG_TORCH_PROFILER_DIR/{uuid4}-{mode}-global-rank0.trace.json.gz` (default dir: `./logs`).
The `{uuid4}` part is a randomly generated request ID. Use `ls -t` or `ls -1` to find the latest trace file.

**Example — profile 3 denoising steps of FLUX.1-dev:**

```bash
SGLANG_TORCH_PROFILER_DIR=/workspace/gen_benchmark/profiles \
sglang generate \
  --model-path=black-forest-labs/FLUX.1-dev \
  --prompt="A futuristic cyberpunk city at night, neon lights reflecting on wet streets, highly detailed, 8k" \
  --width=1024 --height=1024 \
  --num-inference-steps=50 \
  --guidance-scale=4.0 \
  --seed=42 \
  --enable-torch-compile \
  --warmup \
  --profile \
  --num-profiled-timesteps 3
```

**Example — profile all pipeline stages (text encoder + denoise + VAE):**

```bash
SGLANG_TORCH_PROFILER_DIR=/workspace/gen_benchmark/profiles \
sglang generate \
  --model-path=black-forest-labs/FLUX.1-dev \
  --prompt="A futuristic cyberpunk city at night" \
  --width=1024 --height=1024 \
  --num-inference-steps=5 \
  --seed=42 \
  --enable-torch-compile \
  --warmup \
  --profile \
  --profile-all-stages
```

**Reading the trace without a UI:**

The output is a `.trace.json.gz` file (Chrome trace format). Parse it with Python to get a ranked op table — no browser needed:

```python
import gzip, json, collections, glob, os

# The request_id is a uuid4, so the filename is e.g.:
#   550e8400-e29b-41d4-a716-446655440000-3_steps-global-rank0.trace.json.gz
# Find the latest trace file automatically:
log_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR", "./logs")
traces = sorted(glob.glob(f"{log_dir}/*.trace.json.gz"), key=os.path.getmtime, reverse=True)
assert traces, f"No trace files found in {log_dir}"
trace_path = traces[0]
print(f"Reading: {trace_path}")
with gzip.open(trace_path, "rb") as f:
    data = json.loads(f.read())

events = data.get("traceEvents", [])

# Collect CUDA kernel durations per op name
cuda_ops = collections.defaultdict(lambda: {"total_us": 0, "count": 0})
for e in events:
    if e.get("cat") in ("kernel", "gpu_memcpy") and "dur" in e:
        name = e.get("name", "unknown")
        cuda_ops[name]["total_us"] += e["dur"]
        cuda_ops[name]["count"] += 1

# Print top 40 by total CUDA time
sorted_ops = sorted(cuda_ops.items(), key=lambda x: x[1]["total_us"], reverse=True)
print(f"{'Kernel Name':<80} {'Total (ms)':>12} {'Count':>8}")
print("-" * 102)
for name, stats in sorted_ops[:40]:
    print(f"{name:<80} {stats['total_us']/1000:>12.3f} {stats['count']:>8}")
```

**Add `record_function` scopes for per-layer attribution:**

The built-in profiler captures all PyTorch ops, but adding named scopes makes it easy to attribute costs to specific DiT layers. Add these directly in the DiT block `forward()`:

```python
import torch

# Inside DiT transformer block forward() — e.g., FluxTransformerBlock, QwenDiTBlock
with torch.profiler.record_function(f"dit_block_{block_idx}.norm"):
    x = self.norm1(x, scale, shift)
with torch.profiler.record_function(f"dit_block_{block_idx}.attn"):
    x = self.attn(x)
with torch.profiler.record_function(f"dit_block_{block_idx}.mlp"):
    x = self.mlp(x)
```

These scopes appear as named spans in the trace and in `key_averages()` output.

**Key ops to watch per DiT sub-component:**

| Sub-component        | Expected dominant kernel                                    |
|----------------------|-------------------------------------------------------------|
| QKV projection       | `cutlass_gemm` / `ampere_*_gemm`                            |
| Attention compute    | `flash_attn_fwd` / `fmha_*` (FA3/FA4)                      |
| Output projection    | `cutlass_gemm`                                              |
| AdaLN modulation     | `fuse_scale_shift_kernel` / `fuse_scale_shift_gate_*`       |
| RMSNorm / LayerNorm  | `sgl_kernel_rmsnorm` / `fused_add_rmsnorm` / Triton norm    |
| MLP fc1 / fc2        | `cutlass_gemm`                                              |
| SiLU gate            | `vectorized_elementwise_kernel` / `silu_and_mul`            |
| RoPE                 | `apply_rotary_embedding` (Triton) / FlashInfer inplace      |
| QK Norm              | `fused_inplace_qknorm` (JIT) or fallback `rmsnorm`          |

---

### Level 2: nsys + gputrc2graph.py — CUDA Kernel Category Breakdown

SGLang ships a CLI analysis tool at `examples/profiler/nsys_profile_tools/gputrc2graph.py` ([PR #9314](https://github.com/sgl-project/sglang/pull/9314)). It processes `.nsys-rep` files and outputs:
- `result.csv` — kernel-to-category mapping with elapsed time per category (read directly on server)
- `result.html` — stacked bar chart (for environments with a browser)

**Step 1: Install dependencies**

```bash
pip install pandas regex plotly
```

**Step 2: Collect nsys trace**

Run two passes: one with profiling (to get the trace) and one without (to measure the true runtime for CPU time calculation):

```bash
# Pass A — with profiling (skip warmup using --delay)
# Estimate DELAY as: warmup time + a few seconds buffer
nsys profile \
  -t cuda \
  -o /workspace/gen_benchmark/profiles/flux_dev \
  -f true \
  --trace-fork-before-exec=true \
  --delay 120 \
  --duration 60 \
  sglang generate \
    --model-path=black-forest-labs/FLUX.1-dev \
    --prompt="A futuristic cyberpunk city at night, neon lights reflecting on wet streets, highly detailed, 8k" \
    --width=1024 --height=1024 \
    --num-inference-steps=50 \
    --guidance-scale=4.0 \
    --seed=42 \
    --enable-torch-compile \
    --warmup

# Pass B — without profiling, record total wall-clock time
time sglang generate \
  --model-path=black-forest-labs/FLUX.1-dev \
  --prompt="A futuristic cyberpunk city at night, neon lights reflecting on wet streets, highly detailed, 8k" \
  --width=1024 --height=1024 --num-inference-steps=50 \
  --guidance-scale=4.0 --seed=42 \
  --enable-torch-compile --warmup
# Record the elapsed seconds as ELAPSED_SEC
```

**Step 3: Add a diffusion kernel classification JSON**

`gputrc2graph.py` loads all `.json` files in its directory. Create a diffusion-specific one at `examples/profiler/nsys_profile_tools/sglang_diffusion_engine_model.json`:

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

**Step 4: Run analysis (CLI-only, no UI)**

```bash
cd examples/profiler/nsys_profile_tools

python3 gputrc2graph.py \
  --in_file /workspace/gen_benchmark/profiles/flux_dev.nsys-rep,sglang,diffusion,ELAPSED_SEC \
  --out_dir /workspace/gen_benchmark/profiles/analysis \
  --title "FLUX.1-dev denoise kernel breakdown"
```

Replace `ELAPSED_SEC` with the wall-clock seconds measured in Pass B (e.g. `132` if `time sglang generate ...` reported `real 2m12s`). The tool uses this value to compute CPU (non-GPU) idle time = `ELAPSED_SEC - total_GPU_sec`. Passing `0` is also valid but will use the nsys-measured elapsed time, which may inflate non-GPU time if profiling overhead is significant.

**Step 5: Read results without a UI**

```bash
# Read the CSV directly — sorted by elapsed time, largest first
cat /workspace/gen_benchmark/profiles/analysis/result.csv | column -t -s,

# Or with Python for a clean summary
python3 - << 'EOF'
import pandas as pd
df = pd.read_csv("/workspace/gen_benchmark/profiles/analysis/result.csv")
summary = df.groupby("Category")["Elapsed Time (sec)"].sum().sort_values(ascending=False)
total = summary.sum()
for cat, sec in summary.items():
    print(f"{cat:<30} {sec:>8.3f}s  ({sec/total*100:>5.1f}%)")
EOF
```

**What to look for in the category breakdown:**

| Category shows high time | Investigation |
|--------------------------|---------------|
| `gemm` dominant          | Check QKV / output / MLP projections; verify tensor parallelism is active |
| `attn` dominant          | Verify FA3/FA4 is active; check sequence length and head_dim |
| `adaln_modulation` unexpectedly high | Verify fused `fuse_scale_shift_kernel` path is used |
| `norm` high              | Verify `sgl_kernel_rmsnorm` / CuTe DSL fused path; check D alignment |
| `nccl_comm` high         | Multi-GPU: check Ulysses degree; consider reducing TP degree |
| `triton_kernel` high     | Identify which Triton kernel; check if a CuTe DSL or sgl-kernel replacement exists |
| `non-gpu-H_D_memops` high | CPU↔GPU copy detected; check for accidental offload or `.cpu()` calls mid-denoising |
| `CPU(non-GPU)` high      | Python dispatch overhead; check for graph breaks in torch.compile |

**Comparing two versions:**

The 4th field in `--in_file` is `elapsed_nonprofiled_sec` — the **total wall-clock time (seconds) of the same run without profiling** (measured via `time sglang generate ...` in Pass B above). The tool uses this to calculate how much time was spent on CPU (non-GPU) work. Replace `120` and `118` with the actual measured seconds for each run:

```bash
python3 gputrc2graph.py \
  --in_file \
    before.nsys-rep,sglang,diffusion,120 \   # replace 120 with Pass B elapsed seconds for "before"
    after.nsys-rep,sglang,diffusion,118 \    # replace 118 with Pass B elapsed seconds for "after"
  --out_dir /workspace/gen_benchmark/profiles/compare \
  --title "FLUX.1-dev before vs after optimization"

# Then read the comparison CSV
python3 - << 'EOF'
import pandas as pd
df = pd.read_csv("/workspace/gen_benchmark/profiles/compare/result.csv")
pivot = df.pivot_table(values="Elapsed Time (sec)", index="Category",
                       columns="Model_Engine", aggfunc="sum").round(3)
pivot["delta_s"] = pivot.iloc[:, 1] - pivot.iloc[:, 0]
pivot["delta_%"] = (pivot["delta_s"] / pivot.iloc[:, 0] * 100).round(1)
print(pivot.sort_values("delta_s"))
EOF
```

---

### Profiling Workflow Summary

```
0. CORRECTNESS BASELINE
   sglang generate --seed=42 --save-output → save reference images/videos
   (do this before any optimization; compare against it after every change)
       ↓
1. Run benchmark → identify slow model (denoise latency > baseline)
       ↓
2. sglang generate --profile --num-profiled-timesteps 3
   → parse .trace.json.gz → rank ops by self_cuda_time_total
   → identify slow DiT layer / sub-component (norm / attn / mlp / rope)
       ↓
3. nsys profile + gputrc2graph.py → result.csv
   → read category breakdown (gemm / attn / adaln / norm / nccl / cpu-overhead)
   → confirm where GPU time is concentrated
       ↓
4. Cross-reference use-efficient-diffusion-kernels.md
   → apply existing fused kernel if available
   → if no existing kernel covers the case, use add-triton-kernel.md
     to implement and integrate a new Triton kernel
       ↓
5. VERIFY CORRECTNESS FIRST
   sglang generate --seed=42 --save-output → diff against reference baseline
   If output differs beyond tolerance → reject the optimization regardless of speedup
       ↓
6. Re-run benchmark → verify denoise latency improvement; no regression elsewhere
```

---

## Continuous Improvement Checklist

Before merging any PR that affects diffusion performance, run the full benchmark suite and compare.

> **Rule: correctness gates performance.** A PR that improves latency but changes output is not acceptable. Correctness checks must pass before performance numbers are even considered.

### Correctness Checks (must pass first)

- [ ] Reference outputs collected with `--seed=42 --save-output` **before** any change
- [ ] After change: regenerate with identical args and compare against reference
- [ ] No visible quality degradation in generated images / videos
- [ ] For numerical changes: pixel-level diff or PSNR/SSIM within agreed tolerance
- [ ] Correctness verified on **all 7 benchmark models**, not just the model being optimized

### Performance Checks (only after correctness passes)

- [ ] All 7 model benchmarks executed; denoise latency (★), end-to-end latency, and peak memory recorded
- [ ] No regression in denoise latency vs. previous baseline (allow ±2% variance)
- [ ] New optimization shows measurable improvement in denoise latency on at least 2 models
- [ ] No new graph breaks introduced (verify via `torch._dynamo` logs)
- [ ] Results reproducible with all offloads disabled and fixed `--seed=42`
