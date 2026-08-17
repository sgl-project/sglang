"""Minimal kernel runner for NCU profiling (one kernel call per type)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from sglang.srt.layers.jit_kernels.mxfp4_kv import (
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

from sglang.srt.layers.jit_kernels import mxfp4_kv

ext = mxfp4_kv._load_ext()
stream = torch.cuda.current_stream().cuda_stream
ext.mxfp4_quantize_store(kv_bf16, loc, k_data, k_scale, S, KH, stream)
ext.mxfp4_quantize_store(kv_bf16, loc, v_data, v_scale, S, KH, stream)
staged = stage_decode_inputs(q, kv_indices, kv_indptr)

for _ in range(3):  # warmup
    decode_fused(q, k_data, k_scale, v_data, v_scale, kv_indices, kv_indptr,
                 o, lse, sm_scale, staged)
    decode_fused_mma(q, k_data, k_scale, v_data, v_scale, kv_indices, kv_indptr,
                     o, lse, sm_scale, staged)
torch.cuda.synchronize()

decode_fused(q, k_data, k_scale, v_data, v_scale, kv_indices, kv_indptr,
             o, lse, sm_scale, staged)
decode_fused_mma(q, k_data, k_scale, v_data, v_scale, kv_indices, kv_indptr,
                 o, lse, sm_scale, staged)
torch.cuda.synchronize()
print("done")
