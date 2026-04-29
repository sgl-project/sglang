"""Cold-GPU re-bench of pure-INT4 baseline at M_global=4096.

Reproduces exactly what compose_optimal.py measure() does for x=0:
- Build Zipf routing at M=4096 with seed=42
- All 128 experts in cold set (sparse_active_dispatch passes topk_ids
  unchanged because every expert is in the active set)
- Run fused_marlin_moe inside bench() (CUDA graph + L2 flush + median-of-50)
- Repeat 5 times to gauge variance
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from _utils import (
    KERN_E, KERN_NUM_BITS, bench, make_int4_weights,
    make_zipf_routing, sparse_active_dispatch,
)

assert torch.cuda.is_available()
device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")

int4_w1, int4_w2, int4_s1, int4_s2 = make_int4_weights(device, seed=0)
M = 4096
x_, topk_w, topk_ids, gating, _ = make_zipf_routing(M, device, seed=42)

# x=0 → cold set is all 128 experts; sparse_active_dispatch returns the
# original topk_ids unchanged in that case.
cold_active = torch.arange(KERN_E, device=device, dtype=topk_ids.dtype)
cold_ids, cold_w = sparse_active_dispatch(topk_ids, topk_w, cold_active, "marlin")

from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe

def fn():
    fused_marlin_moe(
        x_, int4_w1, int4_w2, int4_s1, int4_s2, gating, cold_w, cold_ids,
        num_bits=KERN_NUM_BITS, is_k_full=True,
    )

# Warmup once outside bench's own warmup
for _ in range(3):
    fn()
torch.cuda.synchronize()

# 5 repeats with a small cooldown between
for rep in range(5):
    t0 = time.perf_counter()
    lat = bench(fn, device)
    elapsed = time.perf_counter() - t0
    temp = torch.cuda.temperature() if hasattr(torch.cuda, "temperature") else "n/a"
    print(f"rep {rep+1}: lat = {lat:.4f} ms (bench took {elapsed:.1f}s)")
    time.sleep(2.0)  # cooldown
