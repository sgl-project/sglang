"""Diagnostic: isolate whether the slowdown is the PACKED AoS FORMAT+exp2 or the
decode kernel itself. Benchmark three decoders on identical shapes:

  (A) bf16 baseline           : sparse_attn_v4_paged_decode (unified bf16)
  (B) QUANT_KV clean SoA fp8  : sparse_attn_v4_paged_decode(kv_scales=...)
                                unified_kv fp8[pages,512] + fp32 scale[pages,8]
                                (the existing TUNED fp8 path; 1x64 block-scale)
  (C) v2 packed AoS           : sparse_attn_v4_paged_decode_split_src_v2

If (B) ~ 1.2-1.5x (A) but (C) >> (A), the mixed-packed AoS byte layout + ue8m0
exp2 is the culprit, and the production path should use a clean SoA fp8 layout
(nope fp8 contiguous + rope bf16 contiguous + scale contiguous).
"""
import time

import torch

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    sparse_attn_v4_paged_decode,
    _FP8_DTYPE,
    _FP8_GROUP_SIZE,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode_split_src_v2 import (
    sparse_attn_v4_paged_decode_split_src_v2,
)
from poc_split_src_decode import _build_packed_buffer


def _bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def _quantize_soa(kv_bf16):
    """Clean 1x64 block-scale fp8 quant: returns (fp8[N,512], scale[N,8] fp32)."""
    N, D = kv_bf16.shape
    G = _FP8_GROUP_SIZE
    x = kv_bf16.float().view(N, D // G, G)
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    fmax = torch.finfo(_FP8_DTYPE).max
    scale = amax / fmax
    q = (x / scale).clamp(-fmax, fmax).to(_FP8_DTYPE).view(N, D)
    return q.contiguous(), scale.squeeze(-1).contiguous().float()


def _case(*, T, H, D, swa_pages, C, page_size, kv_len, ratio_comp, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(T, H, D, device=device, dtype=dtype) * 0.5
    swa_kv = torch.randn(swa_pages, D, device=device, dtype=dtype) * 0.5
    compressed_bf16 = torch.randn(C, D, device=device, dtype=dtype) * 0.5
    attn_sink = torch.randn(H, device=device, dtype=torch.float32)
    scale = 1.0 / (D**0.5)

    # (A) bf16 unified
    packed, _ = _build_packed_buffer(compressed_bf16, page_size)
    loc = torch.arange(C, dtype=torch.int32, device=device)
    comp_dq = dequantize_k_cache_paged(packed, loc, page_size).view(C, D)
    unified_bf16 = torch.cat([swa_kv, comp_dq], 0).contiguous()

    # (B) QUANT_KV clean SoA fp8 for the WHOLE unified buffer
    fp8_kv, kv_scales = _quantize_soa(unified_bf16)

    # monotone [swa|comp] indices
    torch.manual_seed(seed + 7)
    comp_len = int(round(kv_len * ratio_comp)); swa_l = kv_len - comp_len
    indptr = torch.arange(0, (T + 1) * kv_len, kv_len, dtype=torch.int32, device=device)
    parts = []
    for _t in range(T):
        s = torch.randint(0, swa_pages, (swa_l,), device=device, dtype=torch.int32)
        c = swa_pages + torch.randint(0, C, (comp_len,), device=device, dtype=torch.int32)
        parts.append(torch.cat([s, c]))
    indices = torch.cat(parts).to(torch.int32)
    swa_len = torch.full((T,), swa_l, dtype=torch.int32, device=device)

    A = lambda: sparse_attn_v4_paged_decode(q, unified_bf16, indices, indptr, attn_sink, scale)  # noqa
    B = lambda: sparse_attn_v4_paged_decode(q, fp8_kv, indices, indptr, attn_sink, scale, kv_scales=kv_scales)  # noqa
    Cc = lambda: sparse_attn_v4_paged_decode_split_src_v2(q, swa_kv, packed, indices, indptr, swa_len, attn_sink, scale, swa_pages=swa_pages, packed_page_size=page_size)  # noqa

    ta, tb, tc = _bench(A), _bench(B), _bench(Cc)
    print(
        f"T={T:>3} H={H:>3} kv_len={kv_len:>5} rc={ratio_comp:.1f} | "
        f"(A)bf16={ta:7.1f}  (B)fp8-SoA={tb:7.1f} [{tb/ta:4.2f}x]  "
        f"(C)packed-AoS-v2={tc:8.1f} [{tc/ta:5.2f}x]"
    )


def main():
    assert torch.cuda.is_available()
    print(f"device={torch.cuda.get_device_name(0)}")
    cases = [
        dict(T=1, H=128, kv_len=2048, ratio_comp=1.0),
        dict(T=1, H=128, kv_len=8192, ratio_comp=1.0),
        dict(T=16, H=128, kv_len=4096, ratio_comp=0.5),
        dict(T=32, H=128, kv_len=4096, ratio_comp=0.5),
        dict(T=32, H=128, kv_len=16384, ratio_comp=1.0),
        dict(T=128, H=128, kv_len=8192, ratio_comp=0.9),
    ]
    for i, c in enumerate(cases):
        _case(D=512, swa_pages=2048, C=32768, page_size=64, seed=i, **c)
    print("BENCH DONE")


if __name__ == "__main__":
    main()
