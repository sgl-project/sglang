---
name: benchmark-and-profile-reference
description: Reference commands and workflow for denoise benchmarks, perf dumps, and torch.profiler analysis in SGLang Diffusion.
---

# SGLang Diffusion Benchmark and Profile Guide

**Primary Metric: Denoise Latency**
- Denoise latency is the total DiT forward-pass time across all inference steps.
- It is the dominant cost for diffusion inference and the main optimization target.
- End-to-end latency and peak memory are secondary sanity checks.

> **Correctness First**: Faster but incorrect output is not an improvement. Always compare generated images or videos against a reference baseline before and after any change.

## Scope

This guide intentionally stops at:
- checked-in denoise benchmarks
- structured perf dumps
- `torch.profiler` trace capture
- hotspot ranking
- mapping hotspots to known fast paths

If the hotspot survives this checklist, hand the work to
`sglang-diffusion-ako4all-kernel` or another specialized kernel-optimization
skill. Do not grow this skill back into a general Nsight or kernel-authoring
guide.

## Prerequisites

```bash
ENV_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/diffusion_skill_env.py
BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
ROOT=$(python3 "$ENV_PY" print-root)
cd "$ROOT"
python3 "$ENV_PY" check-write-access >/dev/null

export HF_TOKEN=<your_hf_token>  # required for gated repos such as black-forest-labs/FLUX.*
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=$(python3 "$ENV_PY" print-idle-gpus --count 1)

ASSET_DIR=$(python3 "$ENV_PY" print-assets-dir --mkdir)
BENCH_DIR=$(python3 "$ENV_PY" print-output-dir --kind benchmarks --mkdir)
PROFILE_DIR=$(python3 "$ENV_PY" print-output-dir --kind profiles --mkdir)
export PROFILE_DIR

check() {
  local label="$1"
  shift
  "$@" &>/dev/null && echo "[OK]  $label" || echo "[MISS] $label"
}

check "sglang" python3 -c "import sglang"
check "torch+CUDA" python3 -c "import torch; assert torch.cuda.is_available()"
check "torch.profiler" python3 -c "import torch.profiler"
```

## Native Backend Gate

Every benchmark and profile result in this guide must come from the native SGLang diffusion backend.

If the command log contains any of:
- `Falling back to diffusers backend`
- `Using diffusers backend`
- `Loaded diffusers pipeline`

then stop immediately:
- do not record the perf dump or trace as valid benchmark evidence
- do not compare it against other runs
- do not continue to hotspot ranking or kernel optimization
- first fix backend selection so the model stays on the native SGLang diffusion path

The checked-in benchmark helper pins `--backend=sglang` so native presets fail
fast instead of silently falling back through `--backend=auto`. Do the same for
manual native profiling commands unless you are intentionally collecting a
diffusers baseline.

Environment notes:
- all commands below assume you are inside the configured diffusion container shell
- export `HF_TOKEN` before any gated Hugging Face model run
- export `FLASHINFER_DISABLE_VERSION_CHECK=1` before any benchmark or profiler run
- re-run `print-idle-gpus` before each perf command if GPU availability may have changed
- keep benchmark commands within 4 GPUs or fewer

Download input images required by some presets:

```bash
wget -O "${ASSET_DIR}/cat.png" \
  https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png
wget -O "${ASSET_DIR}/mova_single_person.jpg" \
  https://github.com/OpenMOSS/MOVA/raw/main/assets/single_person.jpg
```

## Benchmark Presets

Treat `"$BENCH_PY"` as the source of truth for preset order.

Nightly diffusion comparison is server/API based (`sglang serve` plus requests).
This skill stays on `sglang generate` for local benchmarking and profiling, but
the nightly-aligned presets in `bench_diffusion_denoise.py` mirror
`scripts/ci/utils/diffusion/comparison_configs.json` on model, task, prompt,
reference image, size, frames, seed, GPU count, serve args, and the request
defaults used by `run_comparison.py` when a case omits steps or guidance.
When in doubt, re-check that JSON before trusting this reference.

List the current preset order:

```bash
PYTHONPATH=python python3 "$BENCH_PY" --list-models
```

Run one preset and save a perf dump:

```bash
PYTHONPATH=python python3 "$BENCH_PY" \
  --model ltx2 \
  --label baseline \
  --output-dir "${BENCH_DIR}"
```

Keep `torch.compile` off when the task requires it:

```bash
PYTHONPATH=python python3 "$BENCH_PY" \
  --model flux \
  --label baseline \
  --output-dir "${BENCH_DIR}" \
  --no-torch-compile
```

Run the `LTX-2.3` one-stage skill preset:

```bash
PYTHONPATH=python python3 "$BENCH_PY" \
  --model ltx23-one-stage \
  --label baseline \
  --output-dir "${BENCH_DIR}"
```

Run the nightly-aligned `LTX-2.3` TI2V two-stage preset:

```bash
PYTHONPATH=python python3 "$BENCH_PY" \
  --model ltx23-ti2v-two-stage \
  --label baseline \
  --output-dir "${BENCH_DIR}"
```

Run the `LTX-2.3` two-stage skill preset:

```bash
PYTHONPATH=python python3 "$BENCH_PY" \
  --model ltx23-two-stage \
  --label baseline \
  --output-dir "${BENCH_DIR}"
```

Run the full preset sweep:

```bash
PYTHONPATH=python python3 "$BENCH_PY" \
  --all \
  --label prXXXX \
  --output-dir "${BENCH_DIR}"
```

Nightly-aligned presets come first; skill-only presets stay available after them.

| Preset | Model | Nightly | Notes |
| --- | --- | --- | --- |
| `flux` | `black-forest-labs/FLUX.1-dev` | Yes: `flux1_dev_t2i_1024` | Aligned to nightly prompt plus `--dit-layerwise-offload false` |
| `flux2` | `black-forest-labs/FLUX.2-dev` | Yes: `flux2_dev_t2i_1024` | Aligned to nightly prompt, 50 steps, guidance 4.0 |
| `qwen` | `Qwen/Qwen-Image-2512` | Yes: `qwen_image_2512_t2i_1024` | Aligned to nightly prompt and steps |
| `qwen-edit` | `Qwen/Qwen-Image-Edit-2511` | Yes: `qwen_image_edit_2511` | Uses the nightly cat image and edit prompt |
| `zimage` | `Tongyi-MAI/Z-Image-Turbo` | Yes: `zimage_turbo_t2i_1024` | Aligned to nightly prompt and guidance 4.0 |
| `wan-t2v` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | Yes: `wan22_t2v_a14b_720p` | Aligned to nightly CFG-parallel 4-GPU launch |
| `wan-ti2v` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Yes: `wan22_ti2v_5b_720p` | Uses the nightly cat image and motion prompt |
| `ltx2` | `Lightricks/LTX-2` | Yes: `ltx2_twostage_t2v` | Uses `LTX2TwoStagePipeline`; 2 GPUs, CFG parallel, 768x512, 121 frames, seed 42 |
| `ltx23-ti2v-two-stage` | `Lightricks/LTX-2.3` | Yes: `ltx2.3_twostage_ti2v_2gpus` | Uses the nightly cat image, motion prompt, `LTX2TwoStagePipeline`, 2 GPUs, 768x512, 121 frames, seed 42 |
| `wan-i2v` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Yes: `wan22_i2v_a14b_720p` | Aligned to nightly CFG-parallel 4-GPU launch |
| `ltx23-one-stage` | `Lightricks/LTX-2.3` | No | Skill-only extra preset for the native `LTX-2.3` one-stage baseline; 2 GPUs, 768x512, 121 frames, fps 24, 30 steps, guidance 3.0, seed 1234 |
| `ltx23-two-stage` | `Lightricks/LTX-2.3` | No | Skill-only high-resolution stress preset for the native `LTX-2.3` two-stage path; uses `LTX2TwoStagePipeline`, 2 GPUs, 1536x1024, 121 frames, fps 24, 30 steps, guidance 3.0, seed 1234 |
| `hunyuanvideo` | `hunyuanvideo-community/HunyuanVideo` | No | Skill-only extra preset |
| `mova-720p` | `OpenMOSS-Team/MOVA-720p` | No | Skill-only extra preset |
| `helios` | `BestWishYsh/Helios-Base` | No | Skill-only extra preset |

For Wan2.2 video models, remember the difference between **nightly alignment**
and **best latency tuning**:
- the nightly-aligned 4-GPU commands intentionally keep `--enable-cfg-parallel --ulysses-degree=2` so CFG and ring behavior stay covered
- do not assume that is the fastest topology
- for pure latency tuning, benchmark pure Ulysses too, for example `--ulysses-degree=4 --ring-degree=1` on 4 GPUs, and on 8 GPUs compare pure `--ulysses-degree=8` against `--enable-cfg-parallel --ulysses-degree=4`

### Manual command example: LTX-2 Two-Stage

```bash
sglang generate \
  --model-path=Lightricks/LTX-2 \
  --pipeline-class-name=LTX2TwoStagePipeline \
  --prompt="A cat and a dog baking a cake together in a kitchen." \
  --width=768 --height=512 \
  --num-frames=121 \
  --num-inference-steps=50 --guidance-scale=4.0 \
  --seed=42 --num-gpus=2 --enable-cfg-parallel \
  --save-output --enable-torch-compile --warmup
```

`LTX2TwoStagePipeline` is a native path. The spatial upsampler and distilled
LoRA are auto-resolved from the same model snapshot unless you override them.

### Manual command example: LTX-2.3 TI2V Two-Stage

```bash
sglang generate \
  --model-path=Lightricks/LTX-2.3 \
  --pipeline-class-name=LTX2TwoStagePipeline \
  --prompt="The cat starts walking slowly towards the camera." \
  --image-path="${ASSET_DIR}/cat.png" \
  --width=768 --height=512 \
  --num-frames=121 \
  --num-inference-steps=50 --guidance-scale=4.0 \
  --seed=42 --num-gpus=2 \
  --save-output --enable-torch-compile --warmup
```

This matches the nightly comparison case `ltx2.3_twostage_ti2v_2gpus`.

### Manual command example: LTX-2.3 One-Stage

```bash
sglang generate \
  --model-path=Lightricks/LTX-2.3 \
  --prompt="A beautiful sunset over the ocean" \
  --negative-prompt="shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static." \
  --width=768 --height=512 \
  --num-frames=121 --fps=24 \
  --num-inference-steps=30 --guidance-scale=3.0 \
  --seed=1234 --num-gpus=2 \
  --save-output --enable-torch-compile --warmup
```

Use this when you want the native `LTX2Pipeline` baseline for `LTX-2.3` at the
validated one-stage resolution.

### Manual command example: LTX-2.3 Two-Stage High-Resolution Stress

```bash
sglang generate \
  --model-path=Lightricks/LTX-2.3 \
  --pipeline-class-name=LTX2TwoStagePipeline \
  --prompt="A beautiful sunset over the ocean" \
  --negative-prompt="shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static." \
  --width=1536 --height=1024 \
  --num-frames=121 --fps=24 \
  --num-inference-steps=30 --guidance-scale=3.0 \
  --seed=1234 --num-gpus=2 \
  --save-output --enable-torch-compile --warmup
```

This matches the skill-only `ltx23-two-stage` preset. Use it as a
high-resolution stress target, not as a nightly comparison case.

### Manual command example: Wan2.2-I2V-A14B 720P

```bash
# Select four idle GPUs first:
# export CUDA_VISIBLE_DEVICES=$(python3 "$ENV_PY" print-idle-gpus --count 4)
sglang generate \
  --model-path=Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --prompt="The cat starts walking slowly towards the camera." \
  --image-path="${ASSET_DIR}/cat.png" \
  --720p --num-inference-steps=2 --num-frames=81 \
  --guidance-scale=5.0 --seed=42 --save-output \
  --num-gpus=4 --enable-cfg-parallel --ulysses-degree=2 \
  --text-encoder-cpu-offload --pin-cpu-memory \
  --warmup --enable-torch-compile
```

`Wan2.2-I2V-A14B` uses the 720p max-area config by default, and explicit
`--width/--height` overrides control the target area while preserving the
reference-image aspect ratio.

## Perf Dump Workflow

For every benchmark run, write a perf dump JSON:

```bash
sglang generate ... --warmup --perf-dump-path "${BENCH_DIR}/<result>.json"
```

Before/after comparison:

```bash
python3 python/sglang/multimodal_gen/benchmarks/compare_perf.py \
  "${BENCH_DIR}/baseline.json" \
  "${BENCH_DIR}/new.json"
```

Always keep:
- denoise latency
- end-to-end latency
- peak GPU memory
- exact command line, model shape, dtype, and GPU topology

Never keep a perf dump produced after a diffusers-backend fallback.

## `torch.profiler` Workflow

### 1. Establish the baseline

```bash
PYTHONPATH=python python3 "$BENCH_PY" \
  --model flux \
  --label baseline \
  --output-dir "${BENCH_DIR}"
```

Keep model shape, seed, and GPU topology fixed for every comparison. Save one
reference image or video before changing code. If the active task requires
`torch.compile` off, add `--no-torch-compile` here too.

### 2. Capture a representative trace

By default SGLang profiles the denoising stage. The default sampling window is
5 profiled timesteps after warmup.

```bash
SGLANG_DIFFUSION_TORCH_PROFILER_DIR="${PROFILE_DIR}/torch" \
sglang generate \
  --model-path=black-forest-labs/FLUX.1-dev \
  --prompt="A futuristic cyberpunk city at night" \
  --width=1024 --height=1024 --num-inference-steps=50 \
  --seed=42 --enable-torch-compile --warmup \
  --profile
```

Use `--profile-all-stages` only when you really need text encoder, VAE, or
other non-denoise stages too.

The generated trace path is printed in the console and also lands under
`SGLANG_DIFFUSION_TORCH_PROFILER_DIR`. The diffusion profiler falls back to
`SGLANG_TORCH_PROFILER_DIR` and then `./logs` when the diffusion-specific env
var is unset. Open the trace in Perfetto if you want a timeline view:
- https://ui.perfetto.dev/

### 3. Rank the hot CUDA kernels

Use this parser for a quick top-k table without opening a browser:

```python
import collections
import glob
import gzip
import json
import os

log_dir = (
    os.environ.get("SGLANG_DIFFUSION_TORCH_PROFILER_DIR")
    or os.environ.get("SGLANG_TORCH_PROFILER_DIR")
    or "./logs"
)
trace_path = sorted(
    glob.glob(f"{log_dir}/*.trace.json.gz"),
    key=os.path.getmtime,
    reverse=True,
)[0]

with gzip.open(trace_path, "rb") as f:
    data = json.loads(f.read())

cuda_ops = collections.defaultdict(lambda: {"total_us": 0, "count": 0})
for event in data.get("traceEvents", []):
    if event.get("cat") in ("kernel", "gpu_memcpy") and "dur" in event:
        cuda_ops[event.get("name", "unknown")]["total_us"] += event["dur"]
        cuda_ops[event.get("name", "unknown")]["count"] += 1

print(f"{'Kernel':<90} {'Total(ms)':>10} {'Count':>6}")
for name, stat in sorted(cuda_ops.items(), key=lambda item: -item[1]["total_us"])[:30]:
    print(f"{name:<90} {stat['total_us'] / 1000:>10.3f} {stat['count']:>6}")
```

If you need better attribution, add `record_function(...)` scopes around DiT
attention, norm, modulation, MLP, or communication boundaries and re-run.

### 4. Classify the hotspot with `existing-fast-paths.md`

Do not jump from a hot kernel straight into new code. First classify it against
the known mainline families.

| What the trace shows | First interpretation |
| --- | --- |
| `fused_inplace_qknorm_rope` missing, but separate qk norm plus rope show up | Check whether the fused diffusion `QK norm + RoPE` path should have engaged |
| `to_q -> to_k -> to_v` on NVFP4 or Nunchaku FLUX-family checkpoints | Treat as a packed-QKV fast-path miss or checkpoint-format mismatch |
| `fused_norm_tanh_mul_add*` missing on Z-Image | Treat as a missing mainline modulation path, not a new fusion request |
| `all_to_all`, ring attention, or async A2A dominate | Classify against Ulysses, USP, or turbo-layer overlap first |
| split `fc1 -> gelu -> quant -> fc2.lora_down` on Nunchaku FLUX | Treat as a missing fused GELU MLP path |
| attention kernels dominate | Confirm backend, topology, and shape guards before proposing a new kernel |

If the hot path is already covered by a mainline optimization family, fix the
enablement, shape guard, backend choice, or checkpoint mapping first.

### 5. Hand off only real kernel work

Only after the hotspot survives the fast-path checklist:

1. save a baseline perf dump
2. save a representative `torch.profiler` trace
3. note the exact model, shape, dtype, and GPU topology
4. hand the work to `sglang-diffusion-ako4all-kernel` or another future specialized optimization skill

This skill intentionally stops here. It tells you whether you are looking at:
- a missing existing optimization
- a configuration or backend problem
- or a real kernel opportunity worth handing off

## Minimal Merge Checklist

- [ ] fixed-shape baseline perf dump saved
- [ ] fixed-shape new perf dump saved
- [ ] `compare_perf.py` table generated
- [ ] one representative `torch.profiler` trace saved
- [ ] hotspot classified against `existing-fast-paths.md`
- [ ] reference image or video checked for correctness
- [ ] any remaining kernel work handed to a specialized optimization skill
