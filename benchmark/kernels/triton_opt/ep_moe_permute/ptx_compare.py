"""Extract and compare PTX for baseline (BS512/W4) vs best (BS1024/W8)."""
import os
import re
import shutil
from pathlib import Path

import torch
import triton
import triton.language as tl

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_gpu_3"

OUT_DIR = Path(__file__).parent

@triton.jit
def kernel_baseline(
    input_ptr, gateup_input_ptr, src2dst_ptr, topk_ids_ptr, a1_scales_ptr,
    topk, hidden_size, BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty
    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    src_ptr = input_ptr + src_idx * hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)


device = "cuda:0"
nt, hs, tk = 1024, 4096, 6
inp = torch.randn(nt, hs, device=device, dtype=torch.bfloat16)
src2dst = torch.arange(nt * tk, device=device, dtype=torch.int64).reshape(nt, tk)
gateup = torch.empty(nt * tk, hs, device=device, dtype=torch.bfloat16)
topk_ids = torch.randint(0, 64, (nt, tk), device=device, dtype=torch.int64)

cache_dir = Path("/tmp/triton_cache_gpu_3")

# Baseline: BS=512, W=4
if cache_dir.exists():
    shutil.rmtree(cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

kernel_baseline[(nt,)](inp, gateup, src2dst, topk_ids, None, tk, hs, BLOCK_SIZE=512, num_warps=4)
torch.cuda.synchronize()

ptx_files = sorted(cache_dir.rglob("*.ptx"), key=lambda p: p.stat().st_mtime, reverse=True)
ptx_bl = ptx_files[0].read_text()
(OUT_DIR / "ptx_baseline.txt").write_text(ptx_bl)

# Best: BS=1024, W=8
if cache_dir.exists():
    shutil.rmtree(cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

kernel_baseline[(nt,)](inp, gateup, src2dst, topk_ids, None, tk, hs, BLOCK_SIZE=1024, num_warps=8)
torch.cuda.synchronize()

ptx_files = sorted(cache_dir.rglob("*.ptx"), key=lambda p: p.stat().st_mtime, reverse=True)
ptx_best = ptx_files[0].read_text()
(OUT_DIR / "ptx_best.txt").write_text(ptx_best)


def count_all(ptx):
    cats = {
        "ld.global": r"ld\.global",
        "st.global": r"st\.global",
        "ld.shared": r"ld\.shared",
        "st.shared": r"st\.shared",
        "cp.async": r"cp\.async",
        "mul": r"mul\.",
        "add": r"add\.",
        "fma": r"fma\.",
        "cvt": r"cvt\.",
        "setp": r"setp\.",
        "mov": r"mov\.",
        "bar": r"bar\.",
        "bra": r"bra\b",
        "reqntid": r"reqntid",
    }
    return {k: len(re.findall(v, ptx)) for k, v in cats.items()}


bl_c = count_all(ptx_bl)
best_c = count_all(ptx_best)
print(f"{'Category':15s} {'Baseline':>10s} {'Best':>10s} {'Delta':>10s}")
print("-" * 50)
for cat in bl_c:
    print(f"  {cat:15s} {bl_c[cat]:10d} {best_c[cat]:10d} {best_c[cat] - bl_c[cat]:+10d}")

print(f"\nBaseline .reqntid: {re.findall(r'reqntid (\\d+)', ptx_bl)}")
print(f"Best .reqntid: {re.findall(r'reqntid (\\d+)', ptx_best)}")
