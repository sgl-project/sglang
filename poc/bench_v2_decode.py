"""Stage-3 PoC v2 performance test: two-segment single-source unified_kv decode
vs the all-bf16 baseline (existing sparse_attn_v4_paged_decode, the Triton HIP
kernel that is the current unmodified unified_kv decode path).

Indices are built in the REAL runtime layout: per request [ SWA | compressed ]
contiguous, so each BLOCK_K tile is single-source. Reports v2/baseline ratio and
the resident-byte saving (bf16 1024 B/row -> packed 584 B/row).
"""
import time

import torch

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    sparse_attn_v4_paged_decode,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode_split_src_v2 import (
    sparse_attn_v4_paged_decode_split_src_v2,
)
from poc_split_src_decode import _build_packed_buffer  # noqa: E402


def _bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def _monotone(T, kv_len, swa_pages, C, ratio_comp, device, seed):
    torch.manual_seed(seed + 7)
    comp_len = int(round(kv_len * ratio_comp))
    swa_l = kv_len - comp_len
    indptr = torch.arange(0, (T + 1) * kv_len, kv_len, dtype=torch.int32, device=device)
    parts = []
    for _t in range(T):
        s = torch.randint(0, swa_pages, (swa_l,), device=device, dtype=torch.int32)
        c = swa_pages + torch.randint(0, C, (comp_len,), device=device, dtype=torch.int32)
        parts.append(torch.cat([s, c]))
    indices = torch.cat(parts).to(torch.int32)
    swa_len = torch.full((T,), swa_l, dtype=torch.int32, device=device)
    return indices, indptr, swa_len


def _case(*, T, H, D, swa_pages, C, page_size, kv_len, ratio_comp, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(T, H, D, device=device, dtype=dtype) * 0.5
    swa_kv = torch.randn(swa_pages, D, device=device, dtype=dtype) * 0.5
    compressed_bf16 = torch.randn(C, D, device=device, dtype=dtype) * 0.5
    attn_sink = torch.randn(H, device=device, dtype=torch.float32)
    scale = 1.0 / (D**0.5)

    packed, _ = _build_packed_buffer(compressed_bf16, page_size)
    loc = torch.arange(C, dtype=torch.int32, device=device)
    comp_dq = dequantize_k_cache_paged(packed, loc, page_size).view(C, D)
    unified_bf16 = torch.cat([swa_kv, comp_dq], 0).contiguous()

    indices, indptr, swa_len = _monotone(T, kv_len, swa_pages, C, ratio_comp, device, seed)

    ref = lambda: sparse_attn_v4_paged_decode(  # noqa: E731
        q, unified_bf16, indices, indptr, attn_sink, scale
    )
    v2 = lambda: sparse_attn_v4_paged_decode_split_src_v2(  # noqa: E731
        q, swa_kv, packed, indices, indptr, swa_len, attn_sink, scale,
        swa_pages=swa_pages, packed_page_size=page_size,
    )
    t_ref = _bench(ref)
    t_v2 = _bench(v2)
    print(
        f"T={T:>3} H={H:>3} kv_len={kv_len:>5} ratio_comp={ratio_comp:.1f} | "
        f"bf16-baseline={t_ref:8.1f}us  v2={t_v2:8.1f}us  ratio={t_v2 / t_ref:5.2f}x"
    )


def main():
    assert torch.cuda.is_available()
    print(f"device={torch.cuda.get_device_name(0)}")
    bf16_row = 512 * 2
    packed_row = 448 + 64 * 2 + 8
    print(
        f"compressed row bytes: bf16={bf16_row}  packed={packed_row}  "
        f"saving={100 * (1 - packed_row / bf16_row):.1f}%"
    )
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
