"""Kernel-level decode benchmark: flashinfer fp16 (scalar vs mma) vs our fp4 fused.

Runs in the container (editable sglang at /sgl-workspace/sglang):
    python3 /sgl-workspace/sglang/paper_docs/tests/kernel_bench.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from sglang.srt.layers.jit_kernels.mxfp4_kv import (
    quantize_and_store,
    decode_fused,
    decode_fused_mma,
    stage_decode_inputs,
)

torch.manual_seed(0)
B, QH, KH, HD, SEQ = 100, 32, 8, 128, 1024
S = B * SEQ
sm_scale = HD ** -0.5

kv_bf16 = torch.randn(S, KH, HD, dtype=torch.bfloat16, device="cuda") * 0.7
k_data = torch.zeros(S, KH, HD // 2, dtype=torch.uint8, device="cuda")
k_scale = torch.zeros(S, KH, HD // 32, dtype=torch.uint8, device="cuda")
v_data = torch.zeros(S, KH, HD // 2, dtype=torch.uint8, device="cuda")
v_scale = torch.zeros(S, KH, HD // 32, dtype=torch.uint8, device="cuda")
loc = torch.arange(S, dtype=torch.int64, device="cuda")

kv_indices = torch.arange(S, dtype=torch.int32, device="cuda")
kv_indptr = torch.arange(B + 1, dtype=torch.int32, device="cuda") * SEQ

q = torch.randn(B, QH, HD, dtype=torch.bfloat16, device="cuda") * 0.5
o = torch.empty(B, QH, HD, dtype=torch.bfloat16, device="cuda")
lse = torch.empty(B, QH, device="cuda")

# Kernel-level bench: bypass the sglang staging layer (32MiB caps), call ext directly.
from sglang.srt.layers.jit_kernels import mxfp4_kv

_ext = mxfp4_kv._load_ext()
_stream = torch.cuda.current_stream().cuda_stream
_ext.mxfp4_quantize_store(kv_bf16, loc, k_data, k_scale, S, KH, _stream)
_ext.mxfp4_quantize_store(kv_bf16, loc, v_data, v_scale, S, KH, _stream)


def bench(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(True) for _ in range(iters)]
    ends = [torch.cuda.Event(True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return sum(ts) / len(ts), ts[len(ts) // 2]


# --- 1. our fp4 fused (CUDA cores) ---
staged = (q, kv_indices, kv_indptr)  # already kernel-ready tensors
t, m = bench(lambda: decode_fused(q, k_data, k_scale, v_data, v_scale,
                                  kv_indices, kv_indptr, o, lse, sm_scale, staged))
print(f"fp4 fused (CUDA cores)     : mean {t:8.3f} ms  median {m:8.3f} ms")

# --- 1b. our fp4 fused (mma / tensor cores, ldmatrix) ---
t, m = bench(lambda: decode_fused_mma(q, k_data, k_scale, v_data, v_scale,
                                      kv_indices, kv_indptr, o, lse, sm_scale, staged))
print(f"fp4 fused (mma ldmatrix)   : mean {t:8.3f} ms  median {m:8.3f} ms")

# --- 2. flashinfer fp16 scalar vs mma ---
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

kv_cache = torch.stack([kv_bf16.view(S, 1, KH, HD)] * 2, dim=1)  # [S, 2, page_size=1, KH, HD]
last_page = torch.ones(B, dtype=torch.int32, device="cuda")
for tc in (False, True):
    ws = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device="cuda")
    w = BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_cuda_graph=False,
                                           use_tensor_cores=tc)
    w.plan(kv_indptr, kv_indices, last_page, QH, KH, HD, 1, q_data_type=torch.bfloat16)
    out = w.forward(q, kv_cache)
    t, m = bench(lambda: w.forward(q, kv_cache))
    print(f"flashinfer fp16 tc={tc}     : mean {t:8.3f} ms  median {m:8.3f} ms")
