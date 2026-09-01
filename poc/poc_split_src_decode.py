"""Stage-3 PoC self-test: split-source unified_kv decode.

Oracle: dequantize the packed compressed buffer to bf16 with the PRODUCTION
dequant (dsv4.dequant_k_cache), splice it into a fully-bf16 unified buffer, and
run the EXISTING unified decode kernel (sparse_attn_v4_paged_decode). The PoC
kernel reads swa(bf16) + packed(uint8) directly and must reproduce that output.

Since the oracle's compressed rows == dequant(packed) and the PoC dequants the
same bytes with the same addressing, the two KV tiles fed to the dot are
bit-identical, so outputs must match to fp32-accumulation tolerance.
"""

import torch

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    NOPE_ROPE_BYTES,
    PADDED_SCALE_PER_TOKEN,
    dequantize_k_cache_paged,
)
from sglang.kernels.ops.attention.dsv4.index_buf_accessor import SetKAndS
from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    sparse_attn_v4_paged_decode,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode_split_src import (
    sparse_attn_v4_paged_decode_split_src,
)


class _FakePool:
    """Minimal pool shim for SetKAndS (needs .page_size)."""

    def __init__(self, page_size):
        self.page_size = page_size


def _build_packed_buffer(compressed_bf16, page_size):
    """Quantize [C, 512] bf16 -> production packed [num_pages, bpp] uint8.
    Returns (packed_uint8, bytes_per_page). Row i stored at loc=i."""
    C = compressed_bf16.shape[0]
    device = compressed_bf16.device
    raw_bytes_per_page = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = (
        (raw_bytes_per_page + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES
    ) * NOPE_ROPE_BYTES
    num_pages = (C + page_size - 1) // page_size + 1  # +1 pad page (null slot)
    packed = torch.zeros((num_pages, bytes_per_page), dtype=torch.uint8, device=device)

    pack = quant_to_nope_fp8_rope_bf16_pack_triton(compressed_bf16.contiguous())
    loc = torch.arange(C, dtype=torch.int64, device=device)
    SetKAndS.execute(_FakePool(page_size), packed, loc, pack)
    return packed, bytes_per_page


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

    # Production dequant of the packed rows -> bf16 ground-truth compressed KV.
    loc = torch.arange(C, dtype=torch.int32, device=device)
    comp_dq = dequantize_k_cache_paged(packed, loc, page_size).view(C, D)

    # Build a fully-bf16 unified buffer: [0, swa_pages) SWA ++ [swa_pages, ...) comp.
    unified_bf16 = torch.cat([swa_kv, comp_dq], dim=0).contiguous()

    # Random ragged indices mixing SWA slots and compressed slots.
    torch.manual_seed(seed + 1)
    lens = torch.randint(1, 40, (T,), device=device, dtype=torch.int32)
    indptr = torch.zeros(T + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(lens, 0)
    total = int(indptr[-1].item())
    r = torch.rand(total, device=device)
    swa_slots = torch.randint(0, swa_pages, (total,), device=device, dtype=torch.int32)
    comp_slots = (
        swa_pages
        + torch.randint(0, C, (total,), device=device, dtype=torch.int32)
    )
    indices = torch.where(r < ratio_comp, comp_slots, swa_slots).to(torch.int32)

    # Oracle: existing kernel over the fully-bf16 unified buffer.
    out_ref = sparse_attn_v4_paged_decode(
        q, unified_bf16, indices, indptr, attn_sink, softmax_scale
    )
    # PoC: split-source kernel over swa(bf16) + packed(uint8).
    out_poc = sparse_attn_v4_paged_decode_split_src(
        q,
        swa_kv,
        packed,
        indices,
        indptr,
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
