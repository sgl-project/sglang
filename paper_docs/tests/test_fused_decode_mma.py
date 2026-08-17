"""Validate the mma (tensor-core) fused decode kernel vs manual reference."""
import os
import sys
import math

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from sglang.srt.layers.jit_kernels.mxfp4_kv import (
    quantize_and_store,
    decode_fused,
    decode_fused_mma,
)

HEAD_DIM = 128


def reference_decode(q, k_ref, v_ref, kv_indptr, sm_scale):
    batch, qh, _ = q.shape
    group = qh // k_ref.shape[1]
    o = torch.zeros(batch, qh, HEAD_DIM, dtype=torch.float32, device="cuda")
    lse = torch.zeros(batch, qh, device="cuda")
    qf = q.float()
    kf = k_ref.float()
    vf = v_ref.float()
    for b in range(batch):
        start, end = kv_indptr[b].item(), kv_indptr[b + 1].item()
        for h in range(qh):
            kh_ = h // group
            s = torch.einsum("d,nd->n", qf[b, h], kf[start:end, kh_, :]) * sm_scale
            m = s.max()
            p = torch.exp(s - m)
            d = p.sum()
            o[b, h] = (p[:, None] * vf[start:end, kh_, :]).sum(0) / d
            lse[b, h] = m / math.log(2) + torch.log2(d)
    return o.to(torch.bfloat16), lse


def run_case(batch, qh, kh, lens, seed):
    torch.manual_seed(seed)
    S = max(sum(lens) + 16, 64)
    group = qh // kh
    k_data = torch.zeros(S, kh, HEAD_DIM // 2, dtype=torch.uint8, device="cuda")
    k_scale = torch.zeros(S, kh, HEAD_DIM // 32, dtype=torch.uint8, device="cuda")
    v_data = torch.zeros(S, kh, HEAD_DIM // 2, dtype=torch.uint8, device="cuda")
    v_scale = torch.zeros(S, kh, HEAD_DIM // 32, dtype=torch.uint8, device="cuda")
    kv_bf16 = torch.randn(sum(lens), kh, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.7

    q = torch.randn(batch, qh, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
    kv_indices = torch.arange(sum(lens), dtype=torch.int32, device="cuda")
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    off = 0
    for b, ln in enumerate(lens):
        x = kv_bf16[off:off + ln]
        loc = torch.arange(off, off + ln, dtype=torch.int64, device="cuda")
        quantize_and_store(x, loc, k_data, k_scale)
        quantize_and_store(x, loc, v_data, v_scale)
        kv_indptr[b + 1] = off + ln
        off += ln

    o_ref, lse_ref = reference_decode(q, kv_bf16, kv_bf16, kv_indptr, HEAD_DIM ** -0.5)
    o = torch.empty(batch, qh, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(batch, qh, device="cuda")
    for fn, name in ((decode_fused, "scalar"), (decode_fused_mma, "mma")):
        o.zero_()
        lse.zero_()
        fn(q, k_data, k_scale, v_data, v_scale, kv_indices, kv_indptr, o, lse,
           HEAD_DIM ** -0.5)
        torch.cuda.synchronize()
        max_o = (o.float() - o_ref.float()).abs().max().item()
        max_l = (lse - lse_ref).abs().max().item()
        bad = (o != o_ref).sum().item()
        print(f"[{name}] lens={lens} max_o={max_o:.5f} max_lse={max_l:.5f} "
              f"diff_elems={bad}/{o.numel()}")


run_case(3, 32, 8, [257, 64, 130], 0)
run_case(2, 32, 8, [16, 33], 1)
run_case(2, 32, 8, [1, 64], 2)
run_case(4, 32, 8, [130, 200, 40, 512], 3)
run_case(5, 32, 8, [1024] * 5, 4)
