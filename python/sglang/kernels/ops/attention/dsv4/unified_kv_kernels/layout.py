"""Row layout of the two-pool fp8 unified_kv cache, shared by its writers.

The pools are separate allocations with the same row count and one row index
addresses both, so these numbers belong with the kernels that write the rows
rather than with the pool that allocates them. Neither writer bounds-checks that
index -- the Triton scatter walks off the end of the shorter pool, aiter's fused
store aborts the process with nothing on stderr -- so the pair has to be checked
before the launch.
"""

from __future__ import annotations

from typing import Optional

import torch

# The fp8 nope row is a fixed 512 B whatever the payload: 448 B latent, then
# 14 B of E8M0 tile scales (7 tiles, each written twice -- the asm reader reads
# every tile scale twice), then 50 B nobody touches. Keep in sync with aiter's
# pack_v4_nope_scale and with kFp8TwoPoolRowBytes in
# jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh; the 512 B stride is what the
# reader assumes and nothing checks it across the language boundary.
DSV4_FP8_NOPE_ROW_BYTES = 512
DSV4_FP8_QUANT_TILE = 64


def check_two_pool_pair(
    nope_pool: torch.Tensor,
    rope_pool: Optional[torch.Tensor],
    *,
    rope_width: int,
    rope_dtype: torch.dtype,
) -> None:
    """Reject two pools that aren't a pair, before anything is written.

    ``rope_width`` is what the caller believes the rope row is (rot_dim for the
    fused store, the source row width for the scatter). Both writers take the rope
    row stride off the tensor, so a wider row would still land in the right place;
    a width that disagrees with the caller means the wrong pool was fetched.
    """
    assert (
        rope_pool is not None
    ), "the fp8 layout needs a rope pool next to the nope pool"
    assert (
        nope_pool.shape[0] == rope_pool.shape[0]
    ), f"pool rows differ: nope {nope_pool.shape[0]} vs rope {rope_pool.shape[0]}"
    assert (
        nope_pool.element_size() == 1 and nope_pool.shape[-1] == DSV4_FP8_NOPE_ROW_BYTES
    ), (
        f"nope pool must be the packed {DSV4_FP8_NOPE_ROW_BYTES} B fp8 row, got "
        f"{nope_pool.shape[-1]} x {nope_pool.dtype}"
    )
    assert rope_pool.shape[-1] == rope_width and rope_pool.dtype == rope_dtype, (
        f"rope pool is {rope_pool.shape[-1]} x {rope_pool.dtype}, expected "
        f"{rope_width} x {rope_dtype}"
    )
    assert nope_pool.is_contiguous() and rope_pool.is_contiguous()
