"""Phase 4: PTX diff between baseline and best variant."""
import os
import re
import shutil
from pathlib import Path

os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_gpu_1"

import torch
import triton
from sglang.srt.layers.attention.triton_ops.prefill_attention import _fwd_kernel


def compile_and_get_ptx(seq_len, num_heads, head_dim, num_warps, num_stages, BLOCK_M, BLOCK_N, label):
    """Compile kernel with given config and extract PTX."""
    # Clear cache
    cache_dir = Path("/tmp/triton_cache_gpu_1")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    batch = 1
    device = "cuda:0"
    total_tokens = batch * seq_len
    kv_heads = num_heads

    q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(total_tokens, kv_heads, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(total_tokens, kv_heads, head_dim, dtype=torch.float16, device=device)
    o = torch.zeros_like(q)

    b_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
    b_seq_len = torch.tensor([seq_len], dtype=torch.int32, device=device)
    sm_scale = 1.0 / (head_dim ** 0.5)
    kv_group_num = num_heads // kv_heads
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)

    grid = (batch, num_heads, triton.cdiv(seq_len, BLOCK_M))

    _fwd_kernel[grid](
        q, k, v, sm_scale,
        b_start_loc, b_seq_len, o,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=True,
        num_warps=num_warps,
        num_stages=num_stages,
        Lk=head_dim,
    )
    torch.cuda.synchronize()

    # Get PTX
    ptx_files = sorted(cache_dir.rglob("*.ptx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ptx_files:
        print(f"No PTX found for {label}")
        return None

    ptx = ptx_files[0].read_text()
    out_dir = Path(__file__).parent
    (out_dir / f"ptx_{label}.txt").write_text(ptx)
    return ptx


def count_instructions(ptx, label):
    """Count PTX instructions."""
    patterns = {
        "ld.global": r"ld\.global",
        "st.global": r"st\.global",
        "ld.shared": r"ld\.shared",
        "st.shared": r"st\.shared",
        "mul.f32": r"\bmul\.f32\b",
        "fma.rn.f32": r"fma\.rn\.f32",
        "div.full.f32": r"div\.full\.f32",
        "div.rn.f32": r"div\.rn\.f32",
        "ex2.approx": r"ex2\.approx",
        "rcp.approx": r"rcp\.approx",
        "cvt": r"\bcvt\.",
        "cp.async": r"cp\.async",
        "mma/wgmma": r"(?:mma\.|wgmma)",
        "ldmatrix": r"ldmatrix",
        "bar.sync": r"bar\.sync",
        "setp": r"setp\.",
    }

    print(f"\n{label}:")
    counts = {}
    for name, pattern in patterns.items():
        count = len(re.findall(pattern, ptx))
        counts[name] = count
        if count > 0:
            print(f"  {name:20s}: {count}")
    print(f"  {'Total lines':20s}: {len(ptx.strip().split(chr(10)))}")
    return counts


# Compare for head_dim=128 (the most common case)
print("=" * 60)
print("PTX DIFF: head_dim=128")
print("=" * 60)
print("\nBaseline: num_warps=8, num_stages=1, BLOCK_M=128, BLOCK_N=128")
ptx_base = compile_and_get_ptx(2048, 32, 128, 8, 1, 128, 128, "baseline")
counts_base = count_instructions(ptx_base, "Baseline (w8_s1_m128_n128)")

print("\nBest: num_warps=4, num_stages=3, BLOCK_M=64, BLOCK_N=64")
ptx_best = compile_and_get_ptx(2048, 32, 128, 4, 3, 64, 64, "best")
counts_best = count_instructions(ptx_best, "Best (w4_s3_m64_n64)")

# Diff
print("\n" + "=" * 60)
print("INSTRUCTION DIFF (best - baseline)")
print("=" * 60)
all_keys = set(counts_base.keys()) | set(counts_best.keys())
for key in sorted(all_keys):
    base = counts_base.get(key, 0)
    best = counts_best.get(key, 0)
    diff = best - base
    if base > 0 or best > 0:
        direction = "+" if diff > 0 else ""
        print(f"  {key:20s}: {base:4d} -> {best:4d}  ({direction}{diff})")

# Also compare for head_dim=64
print("\n" + "=" * 60)
print("PTX DIFF: head_dim=64")
print("=" * 60)
print("\nBaseline: num_warps=4, num_stages=1, BLOCK_M=128, BLOCK_N=128")
ptx_base64 = compile_and_get_ptx(2048, 32, 64, 4, 1, 128, 128, "baseline_d64")
counts_base64 = count_instructions(ptx_base64, "Baseline d64 (w4_s1_m128_n128)")

print("\nBest: num_warps=4, num_stages=3, BLOCK_M=64, BLOCK_N=128")
ptx_best64 = compile_and_get_ptx(2048, 32, 64, 4, 3, 64, 128, "best_d64")
counts_best64 = count_instructions(ptx_best64, "Best d64 (w4_s3_m64_n128)")

print("\n" + "=" * 60)
print("INSTRUCTION DIFF head_dim=64 (best - baseline)")
print("=" * 60)
all_keys = set(counts_base64.keys()) | set(counts_best64.keys())
for key in sorted(all_keys):
    base = counts_base64.get(key, 0)
    best = counts_best64.get(key, 0)
    diff = best - base
    if base > 0 or best > 0:
        direction = "+" if diff > 0 else ""
        print(f"  {key:20s}: {base:4d} -> {best:4d}  ({direction}{diff})")
