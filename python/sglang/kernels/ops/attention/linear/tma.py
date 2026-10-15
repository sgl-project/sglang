# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared TMA layouts for the Blackwell GDN and KDA chunk kernels."""

import cutlass
from cutlass import cute
from cutlass.cute.nvgpu import cpasync


@cute.jit
def make_chunk_tma_args(
    tensor: cute.Tensor,
    dim: cutlass.Constexpr[int],
    op: cpasync.TmaCopyOp,
    stages: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
):
    # Divide the contiguous dimension so the descriptor issues one 4D TMA.
    slayout = cute.make_layout(
        (chunk_size, 1, (64, dim // 64), stages),
        stride=(64, 0, (1, chunk_size * 64), chunk_size * dim),
    )
    slayout = cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, slayout)
    atom, tma_tensor = cpasync.make_tiled_tma_atom(
        op,
        cute.logical_divide(tensor, (None, None, 64)),
        slayout,
        cta_tiler=(chunk_size, 1, dim),
    )
    return atom, tma_tensor, slayout


@cute.jit
def make_recurrent_state_tma_args(
    tensor: cute.Tensor,
    op: cpasync.TmaCopyOp,
    key_dim: cutlass.Constexpr[int],
    value_dim: cutlass.Constexpr[int],
):
    num_elems = 128 // (tensor.element_type.width // 8)
    slayout = cute.make_layout(
        (1, 1, value_dim, (num_elems, key_dim // num_elems)),
        stride=(0, 0, num_elems, (1, value_dim * num_elems)),
    )
    slayout = cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, slayout)
    atom, tma_tensor = cpasync.make_tiled_tma_atom(
        op,
        cute.logical_divide(tensor, (None, None, None, num_elems)),
        slayout,
        cta_tiler=(1, 1, value_dim, key_dim),
    )
    return atom, tma_tensor, slayout


@cute.jit
def make_output_state_tma_args(
    tensor: cute.Tensor,
    op: cpasync.TmaCopyOp,
    stages: cutlass.Constexpr[int],
    key_dim: cutlass.Constexpr[int],
    value_dim: cutlass.Constexpr[int],
):
    num_elems = 128 // (tensor.element_type.width // 8)
    slayout = cute.make_layout(
        (1, value_dim, (num_elems, key_dim // num_elems), stages),
        stride=(0, num_elems, (1, value_dim * num_elems), value_dim * key_dim),
    )
    slayout = cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, slayout)
    atom, tma_tensor = cpasync.make_tiled_tma_atom(
        op,
        cute.logical_divide(tensor, (None, None, num_elems)),
        slayout,
        cta_tiler=(1, value_dim, key_dim),
    )
    return atom, tma_tensor, slayout
