import torch
import triton
import triton.language as tl


@triton.jit
def _gather_mamba_track_indices_kernel(
    mapping,
    mapping_row_stride,
    req_pool_indices,
    positions,
    out,
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    req = tl.load(req_pool_indices + offs, mask=mask, other=0).to(tl.int64)
    pos = tl.load(positions + offs, mask=mask, other=-1).to(tl.int64)
    valid = mask & (pos >= 0)
    slot = tl.load(
        mapping + req * mapping_row_stride + tl.maximum(pos, 0), mask=valid, other=0
    )
    tl.store(out + offs, tl.where(valid, slot, 0), mask=mask)


def gather_mamba_track_indices(
    mapping: torch.Tensor, req_pool_indices: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """``mapping[req_pool_indices, positions]`` with 0 where ``positions < 0``, in
    one launch (the torch form is clamp / index / compare / where)."""
    n = positions.numel()
    out = torch.empty((n,), dtype=mapping.dtype, device=mapping.device)
    if n:
        _gather_mamba_track_indices_kernel[(1,)](
            mapping,
            mapping.stride(0),
            req_pool_indices,
            positions,
            out,
            n,
            BLOCK=max(16, triton.next_power_of_2(n)),
            num_warps=1,
        )
    return out
