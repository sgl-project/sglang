# SPDX-License-Identifier: Apache-2.0

# KDA provenance: this kernel was automatically optimized by the Humanize2
# workflow (https://github.com/PolyArch/humanize) and Kernel Design Agents
# (https://github.com/mit-han-lab/kernel-design-agents).
# Source: https://github.com/BBuf/KDA-Pilot/pull/195 @
# 516c976cee824a236679adf6eb525275a0a9a120.
"""SM120 scale-factor layouts used by the candidate CuTe kernel."""

from __future__ import annotations

import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dsl_user_op
def sm120_make_smem_layout_sfa(
    tiled_mma,
    tile_shape_mnk,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
):
    del loc, ip
    assert sf_vec_size in (16, 32)
    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size
    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)
    assert tile_shape_mnk[0] % 64 == 0
    sfa_tile_m = max(blk_mn, _ceil_div(tile_shape_mnk[0], blk_mn) * blk_mn)
    sfa_shape_m = (mn_basic_block_shape, sfa_tile_m // blk_mn)
    sf_stride_m = (mn_basic_block_stride, blk_elems)
    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0
    assert tile_shape_mnk[2] % (sf_vec_size * blk_sf) == 0
    assert blk_sf % mma_nsf == 0
    sfa_shape_k = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sf_stride_k = (
        k_basic_block_stride,
        mma_nsf,
        sfa_tile_m // blk_mn * blk_elems,
    )
    layout = cute.make_layout(
        (sfa_shape_m, sfa_shape_k), stride=(sf_stride_m, sf_stride_k)
    )
    return cute.append(
        layout,
        cute.make_layout(num_stages, stride=cute.cosize(cute.filter_zeros(layout))),
    )


@dsl_user_op
def sm120_make_smem_layout_sfb(
    tiled_mma,
    tile_shape_mnk,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
):
    del loc, ip
    assert sf_vec_size in (16, 32)
    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf
    assert tile_shape_mnk[1] % 16 == 0
    assert tile_shape_mnk[2] % sf_vec_size == 0
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size
    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)
    sfb_tile_n = max(blk_mn, _ceil_div(tile_shape_mnk[1], blk_mn) * blk_mn)
    sfb_shape_n = (mn_basic_block_shape, sfb_tile_n // blk_mn)
    sf_stride_n = (mn_basic_block_stride, blk_elems)
    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0
    assert tile_shape_mnk[2] % (sf_vec_size * blk_sf) == 0
    assert blk_sf % mma_nsf == 0
    sfb_shape_k = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sf_stride_k = (
        k_basic_block_stride,
        mma_nsf,
        sfb_tile_n // blk_mn * blk_elems,
    )
    layout = cute.make_layout(
        (sfb_shape_n, sfb_shape_k), stride=(sf_stride_n, sf_stride_k)
    )
    return cute.append(
        layout,
        cute.make_layout(num_stages, stride=cute.cosize(cute.filter_zeros(layout))),
    )
