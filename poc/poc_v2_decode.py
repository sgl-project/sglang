"""Stage-3 PoC v2 self-test: two-segment single-source unified_kv decode.

Same oracle as v1 (production dequant -> bf16 unified -> existing decode kernel),
but the index stream is built in the REAL runtime layout: per request the slots
are CONTIGUOUS and MONOTONE  [ SWA prefix (swa_len[t]) | compressed tail ].
We pass swa_len[t] to the v2 kernel as the crossover point.
"""

import torch

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    sparse_attn_v4_paged_decode,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode_split_src_v2 import (
    sparse_attn_v4_paged_decode_split_src_v2,
)

# reuse the validated packed-buffer builder from v1 poc
from poc_split_src_decode import _build_packed_buffer


def _build_monotone_indices(T, swa_pages, C, ratio_comp, device, seed):
    """Per request: [swa_len swa slots][comp_len compressed slots], concatenated."""
    torch.manual_seed(seed + 1)
    total_lens = torch.randint(4, 48, (T,), device=device)
    comp_lens = (total_lens.float() * ratio_comp).round().to(torch.int64)
    comp_lens = torch.minimum(comp_lens, total_lens)
    swa_lens = total_lens - comp_lens

    indptr = torch.zeros(T + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(total_lens, 0)
    total = int(indptr[-1].item())
    indices = torch.empty(total, dtype=torch.int32, device=device)

    off = 0
    for t in range(T):
        sl = int(swa_lens[t].item())
        cl = int(comp_lens[t].item())
        if sl > 0:
            indices[off : off + sl] = torch.randint(
                0, swa_pages, (sl,), device=device, dtype=torch.int32
            )
        if cl > 0:
            indices[off + sl : off + sl + cl] = swa_pages + torch.randint(
                0, C, (cl,), device=device, dtype=torch.int32
            )
        off += sl + cl
    return indices, indptr, swa_lens.to(torch.int32)


def _run_case(*, T, H, D, swa_pages, C, page_size, seed=0, ratio_comp=0.5):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(T, H, D, device=device, dtype=dtype) * 0.5
    swa_kv = torch.randn(swa_pages, D, device=device, dtype=dtype) * 0.5
    compressed_bf16 = torch.randn(C, D, device=device, dtype=dtype) * 0.5
    attn_sink = torch.randn(H, device=device, dtype=torch.float32)
    softmax_scale = 1.0 / (D**0.5)

    packed, _bpp = _build_packed_buffer(compressed_bf16, page_size)
    loc = torch.arange(C, dtype=torch.int32, device=device)
    comp_dq = dequantize_k_cache_paged(packed, loc, page_size).view(C, D)
    unified_bf16 = torch.cat([swa_kv, comp_dq], dim=0).contiguous()

    indices, indptr, swa_len = _build_monotone_indices(
        T, swa_pages, C, ratio_comp, device, seed
    )

    out_ref = sparse_attn_v4_paged_decode(
        q, unified_bf16, indices, indptr, attn_sink, softmax_scale
    )
    out_poc = sparse_attn_v4_paged_decode_split_src_v2(
        q,
        swa_kv,
        packed,
        indices,
        indptr,
        swa_len,
        attn_sink,
        softmax_scale,
        swa_pages=swa_pages,
        packed_page_size=page_size,
    )

    diff = (out_poc.float() - out_ref.float()).abs()
    max_abs = diff.max().item()
    denom = out_ref.float().abs().max().item() + 1e-6
    rel = max_abs / denom
    ok = torch.allclose(out_poc, out_ref, atol=2e-2, rtol=2e-2)
    print(
        f"[T={T} H={H} D={D} swa_pages={swa_pages} C={C} ps={page_size} "
        f"ratio_comp={ratio_comp}] max_abs={max_abs:.3e} rel={rel:.3e} "
        f"-> {'OK' if ok else 'FAIL'}"
    )
    return ok


def main():
    assert torch.cuda.is_available(), "PoC needs a CUDA/HIP device"
    cases = [
        dict(T=1, H=16, D=512, swa_pages=256, C=1024, page_size=64, ratio_comp=1.0),
        dict(T=1, H=16, D=512, swa_pages=256, C=1024, page_size=64, ratio_comp=0.0),
        dict(T=1, H=128, D=512, swa_pages=512, C=4096, page_size=64, ratio_comp=0.5),
        dict(T=16, H=128, D=512, swa_pages=512, C=4096, page_size=64, ratio_comp=0.5),
        dict(T=32, H=16, D=512, swa_pages=1024, C=8192, page_size=128, ratio_comp=0.5),
        dict(T=32, H=64, D=512, swa_pages=1024, C=8192, page_size=1, ratio_comp=0.7),
    ]
    all_ok = True
    for i, c in enumerate(cases):
        all_ok &= _run_case(seed=i, **c)
    print("ALL OK" if all_ok else "SOME FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
