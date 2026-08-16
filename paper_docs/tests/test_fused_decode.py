"""Validate fused MXFP4 decode kernel vs flashinfer decode reference."""
import os
import sys

import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from sglang.srt.layers.jit_kernels.mxfp4_kv import (
    quantize_and_store,
    dequantize_indices,
    decode_fused,
)

HEAD_DIM = 128


def reference_decode(q, k_ref, v_ref, kv_indptr, sm_scale):
    """Manual attention reference on dequantized KV, per-request slices."""
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
            # kernel reports lse in log2 domain: m/ln2 + log2(d)
            lse[b, h] = m / math.log(2) + torch.log2(d)
    return o.to(torch.bfloat16), lse


def main():
    torch.manual_seed(0)
    batch, qh, kh = 3, 32, 8
    lens = [257, 64, 130]  # mixed lengths, one > 256 to exercise pipeline
    S = 512
    group = qh // kh

    # build kv pool (packed fp4) + kv_indices/indptr
    k_data = torch.zeros(S, kh, HEAD_DIM // 2, dtype=torch.uint8, device="cuda")
    k_scale = torch.zeros(S, kh, HEAD_DIM // 32, dtype=torch.uint8, device="cuda")
    v_data = torch.zeros(S, kh, HEAD_DIM // 2, dtype=torch.uint8, device="cuda")
    v_scale = torch.zeros(S, kh, HEAD_DIM // 32, dtype=torch.uint8, device="cuda")
    kv_bf16 = torch.zeros(sum(lens), kh, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    q = (torch.randn(batch, qh, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5)
    kv_indices = []
    indptr = [0]
    off = 0
    for b, ln in enumerate(lens):
        x = torch.randn(ln, kh, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.7
        loc = torch.arange(off, off + ln, dtype=torch.int64, device="cuda")
        quantize_and_store(x, loc, k_data, k_scale)
        quantize_and_store(x, loc, v_data, v_scale)
        kv_bf16[off : off + ln] = x
        kv_indices.append(torch.arange(off, off + ln, dtype=torch.int32, device="cuda"))
        off += ln
        indptr.append(off)
    kv_indices = torch.cat(kv_indices)
    # reference must use the *dequantized* KV (quantization error excluded)
    k_ref = torch.zeros(sum(lens), kh, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v_ref = torch.zeros(sum(lens), kh, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    dequantize_indices(k_data, k_scale, kv_indices, k_ref)
    dequantize_indices(v_data, v_scale, kv_indices, v_ref)
    kv_indptr = torch.tensor(indptr, dtype=torch.int32, device="cuda")

    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    # fused kernel
    o = torch.zeros(batch, qh, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(batch, qh, dtype=torch.float32, device="cuda")
    decode_fused(q, k_data, k_scale, v_data, v_scale, kv_indices, kv_indptr, o, lse, sm_scale)
    torch.cuda.synchronize()

    # reference (on dequantized KV)
    o_ref, lse_ref = reference_decode(q, k_ref, v_ref, kv_indptr, sm_scale)

    o_err = (o.float() - o_ref.float()).abs().max().item()
    o_rel = ((o.float() - o_ref.float()).abs() / (o_ref.float().abs() + 1e-3)).mean().item()
    lse_err = (lse - lse_ref).abs().max().item()
    print(f"o max_abs_err={o_err:.5f} o mean_rel_err={o_rel:.5f} lse_max_err={lse_err:.5f}")

    # check against flashinfer decode (workspace path) too
    try:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
        # workspace bf16 kv
        kv_ws = kv_bf16
        indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
        indptr[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32, device="cuda"), 0)
        idx = torch.arange(sum(lens), dtype=torch.int32, device="cuda")
        last = torch.ones(batch, dtype=torch.int32, device="cuda")
        wrapper.plan(indptr, idx, last, qh, kh, HEAD_DIM, 1, "NONE", torch.bfloat16)
        o_fi = wrapper.forward(q.contiguous(), (kv_ws, kv_ws))
        torch.cuda.synchronize()
        fi_err = (o.float() - o_fi.float()).abs().max().item()
        print(f"flashinfer ref max_abs_err={fi_err:.5f}")
    except Exception as e:
        print("flashinfer ref skipped:", e)

    assert o_err < 0.02, f"fused kernel output mismatch: {o_err}"
    assert lse_err < 0.01, f"lse mismatch: {lse_err}"
    print("FUSED DECODE OK")


if __name__ == "__main__":
    main()
