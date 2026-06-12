"""Phase 1.2-1.5: Compile kernel, extract PTX, analyze instructions, roofline, occupancy"""
import os
import re
from pathlib import Path

os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_gpu_1"

import torch
import triton

# Clear cache for fresh compilation
import shutil
cache_dir = Path("/tmp/triton_cache_gpu_1")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

from sglang.srt.layers.attention.triton_ops.prefill_attention import _fwd_kernel, context_attention_fwd

# Representative inputs: batch=1, seq_len=2048, num_heads=32, head_dim=128
batch = 1
seq_len = 2048
num_heads = 32
kv_heads = 32
head_dim = 128
device = "cuda:0"

total_tokens = batch * seq_len
q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device=device)
k = torch.randn(total_tokens, kv_heads, head_dim, dtype=torch.float16, device=device)
v = torch.randn(total_tokens, kv_heads, head_dim, dtype=torch.float16, device=device)
o = torch.zeros_like(q)

b_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
b_seq_len = torch.tensor([seq_len], dtype=torch.int32, device=device)

# Launch kernel to compile
context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, seq_len, is_causal=True)
torch.cuda.synchronize()

print("Kernel compiled. Searching for PTX...")

# Find PTX
ptx_files = sorted(cache_dir.rglob("*.ptx"), key=lambda p: p.stat().st_mtime, reverse=True)
print(f"Found {len(ptx_files)} PTX files")

if ptx_files:
    ptx = ptx_files[0].read_text()

    # Save baseline PTX
    out_dir = Path(__file__).parent
    (out_dir / "ptx_baseline.txt").write_text(ptx)
    print(f"Saved baseline PTX to ptx_baseline.txt ({len(ptx)} chars)")

    # 1.3: Annotate PTX - count instructions
    print("\n" + "=" * 60)
    print("PTX INSTRUCTION COUNTS")
    print("=" * 60)

    instructions = {
        "ld.global": r"ld\.global",
        "st.global": r"st\.global",
        "ld.shared": r"ld\.shared",
        "st.shared": r"st\.shared",
        "mul.f32": r"mul\.f32",
        "mul.f16": r"mul\.f16",
        "fma.rn.f32": r"fma\.rn\.f32",
        "fma.rn.f16": r"fma\.rn\.f16",
        "div.full.f32": r"div\.full\.f32",
        "div.rn.f32": r"div\.rn\.f32",
        "div.approx": r"div\.approx",
        "ex2.approx": r"ex2\.approx",
        "rcp.approx": r"rcp\.approx",
        "cvt.*": r"cvt\.",
        "cp.async": r"cp\.async",
        "mma": r"mma\.",
        "bar.sync": r"bar\.sync",
        "setp": r"setp\.",
    }

    for name, pattern in instructions.items():
        count = len(re.findall(pattern, ptx))
        if count > 0:
            print(f"  {name:20s}: {count}")

    # Total lines in PTX
    lines = ptx.strip().split("\n")
    print(f"\n  Total PTX lines: {len(lines)}")

    # Also check for ldmatrix, wmma, or tensor core instructions
    for pattern_name, pattern in [("ldmatrix", r"ldmatrix"), ("wmma", r"wmma"), ("wgmma", r"wgmma"), ("mma.sync", r"mma\.sync")]:
        count = len(re.findall(pattern, ptx))
        if count > 0:
            print(f"  {pattern_name:20s}: {count}")

# 1.4: Roofline Analysis
print("\n" + "=" * 60)
print("ROOFLINE ANALYSIS")
print("=" * 60)

BLOCK_M = 128
BLOCK_N = 128
BLOCK_D = 128

# Per inner loop iteration:
# Load K: BLOCK_D x BLOCK_N x 2 bytes (fp16) = 128*128*2 = 32KB
# Load V: BLOCK_N x BLOCK_D x 2 bytes (fp16) = 128*128*2 = 32KB
bytes_per_iter = 2 * BLOCK_D * BLOCK_N * 2  # K + V loads
# Store: only once at end, amortized

# FLOPs per iteration:
# dot(q, k): 2 * BLOCK_M * BLOCK_N * BLOCK_D = 2 * 128 * 128 * 128
# dot(p, v): 2 * BLOCK_M * BLOCK_D * BLOCK_N = 2 * 128 * 128 * 128
# exp, mul, etc are small compared to dots
flops_per_iter = 2 * (2 * BLOCK_M * BLOCK_N * BLOCK_D)

ai = flops_per_iter / bytes_per_iter
print(f"Bytes per inner iter: {bytes_per_iter} ({bytes_per_iter/1024:.0f} KB)")
print(f"FLOPs per inner iter: {flops_per_iter} ({flops_per_iter/1e6:.1f} MFLOP)")
print(f"Arithmetic Intensity: {ai:.1f} FLOP/byte")
print(f"H200 crossover point: ~16.7 FLOP/byte")
print(f"Classification: {'COMPUTE-BOUND' if ai > 16.7 else 'MEMORY-BOUND'}")

# 1.5: Occupancy Check
print("\n" + "=" * 60)
print("OCCUPANCY CHECK")
print("=" * 60)

props = torch.cuda.get_device_properties(0)
num_warps = 8  # for head_dim=128
threads_per_block = num_warps * 32
blocks_per_sm = props.max_threads_per_multi_processor // threads_per_block
total_blocks = batch * num_heads * triton.cdiv(seq_len, BLOCK_M)
waves = total_blocks / (props.multi_processor_count * blocks_per_sm)

print(f"Threads per block: {threads_per_block}")
print(f"Max blocks per SM: {blocks_per_sm}")
print(f"Total blocks (batch={batch}, heads={num_heads}, seq={seq_len}): {total_blocks}")
print(f"SMs: {props.multi_processor_count}")
print(f"Waves: {waves:.2f}")
print(f"Wave efficiency: {(waves % 1) * 100:.0f}% tail" if waves > 1 else f"Blocks/SM coverage: {total_blocks / props.multi_processor_count:.2f}")
print(f"\nNote: At larger batch/seq, occupancy improves. Key concern is at small sizes.")
