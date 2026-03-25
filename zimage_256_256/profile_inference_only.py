#!/usr/bin/env python3
"""
Profile ONLY inference (not warmup) using cudaProfilerStart/Stop.

nsys --capture-range=cudaProfilerApi only records GPU activity between
torch.cuda.cudart().cudaProfilerStart() and cudaProfilerStop().

This script:
  1. Creates DiffGenerator with warmup (DeepGemm JIT compile happens here)
  2. Runs one warmup inference (not profiled)
  3. cudaProfilerStart()
  4. Runs one inference (PROFILED)
  5. cudaProfilerStop()

Usage:
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    -t cuda -o /path/to/output -f true \
    python zimage_256_256/profile_inference_only.py

  # Then analyze:
  nsys stats /path/to/output.nsys-rep --report cuda_gpu_kern_sum --format csv \
    | grep -i transpose
"""

import torch

# Must be before any sglang import that triggers CUDA
torch.cuda.init()

from sglang import DiffGenerator

MODEL = "/mnt/geminihzceph/rhyshen/models/Z-Image-Turbo"
TRANSFORMER_PATH = f"{MODEL}/transformer-FP8-block128"
PROMPT = "A beautiful sunset over the ocean with golden clouds"
HEIGHT = 256
WIDTH = 256

print("=" * 60)
print("Profile Inference Only (cudaProfilerApi)")
print("=" * 60)

# 1. Create generator with warmup
print("\n[1] Creating DiffGenerator (includes JIT warmup)...")
gen = DiffGenerator.from_pretrained(
    model_path=MODEL,
    transformer_path=TRANSFORMER_PATH,
    text_encoder_precisions="bf16",
    warmup=True,
)

# 2. Warmup inference (not profiled)
print("\n[2] Running warmup inference (not profiled)...")
_ = gen.generate(
    sampling_params_kwargs={
        "prompt": PROMPT,
        "height": HEIGHT,
        "width": WIDTH,
        "seed": 42,
    }
)
torch.cuda.synchronize()
print("    Warmup inference done.")

# 3. Start profiling
print("\n[3] cudaProfilerStart() — profiling begins")
torch.cuda.cudart().cudaProfilerStart()

# 4. Profiled inference
print("[4] Running profiled inference...")
result = gen.generate(
    sampling_params_kwargs={
        "prompt": PROMPT,
        "height": HEIGHT,
        "width": WIDTH,
        "seed": 42,
    }
)
torch.cuda.synchronize()

# 5. Stop profiling
torch.cuda.cudart().cudaProfilerStop()
print("[5] cudaProfilerStop() — profiling ends")

print("\n" + "=" * 60)
print("Done. Analyze the nsys output with:")
print("  nsys stats <output>.nsys-rep --report cuda_gpu_kern_sum --format csv")
print("  # Then grep for transpose:")
print("  grep -i transpose <output>_cuda_gpu_kern_sum.csv")
print("=" * 60)
