# Fused gather-transpose: NHD KV pool slots -> compact HND paged KV.
#
# fmha_sm100's sparse prefill consumes paged KV as [num_pages, Hkv, page_size, D]
# (HND) and unconditionally calls .contiguous() on it (MSA cute/interface.py).
# The sglang MHA pool is slot-major NHD [max_slots, Hkv, D], so for Hkv > 1 the
# permuted whole-pool view is non-contiguous and .contiguous() would copy the
# ENTIRE per-layer pool on every forward. This kernel instead gathers only the
# pages referenced by the batch page table (kv_indices), transposing NHD->HND on
# the fly in a single read+write pass over batch KV; the page table handed to
# MSA then becomes the identity (arange).
#
# For Hkv == 1 the permuted pool view is already contiguous (size-1 dim), so
# callers must skip the gather entirely — see msa_sparse_prefill_main.
#
# The kernel is dtype-agnostic: K/V are reinterpreted as int32 words
# (head_dim * itemsize must be a multiple of 4 bytes — always true for the
# D=128 bf16/fp8 caches MSA supports), giving identical code for bf16 and
# fp8_e4m3. The innermost dim is contiguous on both source and destination
# (256 B rows for bf16, 128 B for fp8), so loads and stores vectorize.

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_kv_hnd_kernel(
    k_src_ptr,  # int32 view of NHD pool: [max_slots, H, DW]
    v_src_ptr,
    k_dst_ptr,  # int32 view of HND out: [n_pages, H, P, DW]
    v_dst_ptr,
    page_ids_ptr,  # int32 [n_pages] physical page id per packed page slot
    H: tl.constexpr,  # num kv heads
    P: tl.constexpr,  # page size (tokens per page)
    DW: tl.constexpr,  # head_dim in int32 words
    BLOCK_S: tl.constexpr,  # tokens per program
):
    pg = tl.program_id(0)  # packed (destination) page index
    h = tl.program_id(1)  # kv head
    sb = tl.program_id(2)  # token block within the page

    phys = tl.load(page_ids_ptr + pg).to(tl.int64)

    s = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    d = tl.arange(0, DW)
    s_mask = (s < P)[:, None]

    # src slot row = phys * P + s; NHD element offset = (slot * H + h) * DW + d
    src_off = ((phys * P + s[:, None].to(tl.int64)) * H + h) * DW + d[None, :]
    # dst HND element offset = ((pg * H + h) * P + s) * DW + d
    dst_off = ((pg.to(tl.int64) * H + h) * P + s[:, None].to(tl.int64)) * DW + d[
        None, :
    ]

    tl.store(
        k_dst_ptr + dst_off, tl.load(k_src_ptr + src_off, mask=s_mask), mask=s_mask
    )
    tl.store(
        v_dst_ptr + dst_off, tl.load(v_src_ptr + src_off, mask=s_mask), mask=s_mask
    )


def gather_kv_hnd(
    k_cache: torch.Tensor,  # [max_slots, H, D] contiguous (NHD pool, any 1/2-byte dtype)
    v_cache: torch.Tensor,  # [max_slots, H, D] contiguous, same dtype/shape as k_cache
    page_ids: torch.Tensor,  # [n_pages] int32 physical page ids (packed by request)
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather referenced pool pages into compact, contiguous HND paged KV.

    Returns (k_hnd, v_hnd) of shape [n_pages, H, page_size, D] in the cache
    dtype. Page ``i`` of the output holds pool page ``page_ids[i]``, so the
    matching MSA ``kv_indices`` is ``arange(n_pages)``.
    """
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    assert k_cache.shape == v_cache.shape and k_cache.dtype == v_cache.dtype
    assert page_ids.dtype == torch.int32
    max_slots, H, D = k_cache.shape
    P = page_size
    assert max_slots % P == 0, f"max_slots={max_slots} not divisible by P={P}"

    itemsize = k_cache.element_size()
    assert (D * itemsize) % 4 == 0, f"head_dim*itemsize={D * itemsize} not 4B-aligned"
    DW = (D * itemsize) // 4
    # tl.arange(0, DW) in the kernel requires a power-of-two extent. DW is 64
    # (bf16) or 32 (fp8_e4m3) for the D=128 caches MSA supports; reject other
    # head dims here instead of failing inside Triton compilation.
    assert DW & (DW - 1) == 0, f"head_dim*itemsize/4={DW} must be a power of two"

    n_pages = page_ids.numel()
    k_out = torch.empty((n_pages, H, P, D), dtype=k_cache.dtype, device=k_cache.device)
    v_out = torch.empty_like(k_out)
    if n_pages == 0:
        return k_out, v_out

    k_src = k_cache.view(torch.int32)
    v_src = v_cache.view(torch.int32)
    k_dst = k_out.view(torch.int32)
    v_dst = v_out.view(torch.int32)

    # ~32 KiB moved per program for bf16 D=128 (BLOCK_S=64: 2 tensors x 64 rows
    # x 256 B); small enough to spread across SMs, large enough to amortize.
    BLOCK_S = min(64, triton.next_power_of_2(P))
    grid = (n_pages, H, triton.cdiv(P, BLOCK_S))
    _gather_kv_hnd_kernel[grid](
        k_src,
        v_src,
        k_dst,
        v_dst,
        page_ids,
        H=H,
        P=P,
        DW=DW,
        BLOCK_S=BLOCK_S,
        num_warps=4,
    )
    return k_out, v_out
