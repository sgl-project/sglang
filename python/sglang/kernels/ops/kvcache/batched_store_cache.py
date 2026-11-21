from typing import Optional, Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def _batched_store_cache_kernel(
    K,
    V,
    K_CACHES,
    V_CACHES,
    LOCS,
    COMMIT_LENS,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    K_CACHE_STRIDES: tl.constexpr,
    V_CACHE_STRIDES: tl.constexpr,
    CAPACITIES: tl.constexpr,
    N,
    ROW_DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_COMMIT_LENS: tl.constexpr,
    SKIP_ZERO: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    layer = tl.program_id(2)
    rows = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.program_id(1).to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)
    live = rows < N
    if HAS_COMMIT_LENS:
        lens = tl.load(COMMIT_LENS + rows // WIDTH, live, other=0)
        live = live & (rows % WIDTH < lens)
    for i in tl.static_range(len(K_CACHES)):
        if layer == i:
            slots = tl.load(LOCS[i] + rows, live, other=-1).to(tl.int64)
            live = live & (slots >= 0) & (slots < CAPACITIES[i])
            if SKIP_ZERO:
                live = live & (slots != 0)
            mask = live[:, None] & (cols[None, :] < ROW_DIM)
            k = tl.load(
                K + i * K_STRIDES[0] + rows[:, None] * K_STRIDES[1] + cols[None, :],
                mask,
                other=0,
            )
            v = tl.load(
                V + i * V_STRIDES[0] + rows[:, None] * V_STRIDES[1] + cols[None, :],
                mask,
                other=0,
            )
            tl.store(
                K_CACHES[i] + slots[:, None] * K_CACHE_STRIDES[i] + cols[None, :],
                k,
                mask,
            )
            tl.store(
                V_CACHES[i] + slots[:, None] * V_CACHE_STRIDES[i] + cols[None, :],
                v,
                mask,
            )


def batched_store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_caches: Sequence[torch.Tensor],
    v_caches: Sequence[torch.Tensor],
    locs: Sequence[torch.Tensor],
    commit_lens: Optional[torch.Tensor] = None,
    locs_row_width: int = 1,
) -> None:
    """Scatter native `[layers, tokens, heads, dim]` K/V in one launch.

    Cache rows and source layers may have padding; each row's head/dim axes
    must be contiguous. Locations are contiguous physical ids per layer.
    Ordinary writes preserve reserved slot zero. Prefix writes commit only
    columns below `commit_lens`, including slot zero like the per-layer writer.
    """
    layers, tokens, heads, dim = k.shape
    assert v.shape == k.shape and len(k_caches) == len(v_caches) == len(locs) == layers
    assert k.dtype in (torch.float16, torch.bfloat16) and v.dtype == k.dtype
    assert k.stride(2) == v.stride(2) == dim and k.stride(3) == v.stride(3) == 1
    for kc, vc, loc in zip(k_caches, v_caches, locs):
        assert kc.shape == vc.shape and kc.shape[1:] == (heads, dim)
        assert (
            kc.dtype == vc.dtype == k.dtype
            and kc.device == vc.device == k.device == v.device
        )
        assert kc.stride(1) == vc.stride(1) == dim and kc.stride(2) == vc.stride(2) == 1
        assert kc.stride(0) >= heads * dim and vc.stride(0) >= heads * dim
        assert loc.device == k.device and loc.dtype in (torch.int32, torch.int64)
        assert loc.is_contiguous() and loc.numel() == tokens
    if commit_lens is not None:
        assert commit_lens.device == k.device and commit_lens.is_contiguous()
        assert commit_lens.dtype in (torch.int32, torch.int64)
        assert locs_row_width > 0 and commit_lens.numel() * locs_row_width == tokens
    if tokens == 0 or layers == 0:
        return
    _batched_store_cache_kernel[
        (triton.cdiv(tokens, 4), triton.cdiv(heads * dim, 256), layers)
    ](
        k,
        v,
        tuple(k_caches),
        tuple(v_caches),
        tuple(locs),
        commit_lens,
        k.stride()[:2],
        v.stride()[:2],
        tuple(c.stride(0) for c in k_caches),
        tuple(c.stride(0) for c in v_caches),
        tuple(c.shape[0] for c in k_caches),
        tokens,
        heads * dim,
        locs_row_width,
        commit_lens is not None,
        commit_lens is None,
        4,
        256,
        num_warps=4,
    )
