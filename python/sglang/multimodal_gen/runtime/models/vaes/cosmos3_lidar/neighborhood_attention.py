# SPDX-License-Identifier: Apache-2.0
"""Neighborhood attention of the Cosmos3 LiDAR range-map VAE without NATTEN.

The windows follow NATTEN's ``na2d``. Along an axis of length ``L`` with kernel
``k`` and dilation ``d``, query ``i`` attends to the ``k`` positions of its residue
class ``i % d`` starting at ``clamp(i // d - k // 2, 0, L_d - k)``, where ``L_d``
is the size of that class: the window shifts inward at the borders instead of
shrinking. The caller owns the circular azimuth padding.

The attention runs as a compiled FlexAttention block mask in IEEE fp32: the
tokenizer is an fp32 model, so TF32 is not applied here whatever the
process-wide matmul policy says. Uncompiled FlexAttention materializes the
dense score matrix (about 1 TB for one 20-sweep chunk), so CUDA always compiles
and only tiny CPU grids may run eagerly, for tests.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

BLOCK_SIZE = 64
_QUERY_CHUNK = 4096
_EAGER_CPU_MAX_TOKENS = 4096
# 32x32 tiles: measured on RTX 5880 Ada at the encoder's three levels (fp32, 20
# frames) within 1.3-2x of NATTEN's CUTLASS kernel everywhere, while 64x64 tiles
# run 6-12x slower than NATTEN at head dim 64. Must divide BLOCK_SIZE.
_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "num_warps": 4,
    "num_stages": 1,
    "USE_TMA": False,
    # Triton dot precision for this operator only, including P @ V.
    "FLOAT32_PRECISION": "'ieee'",
}
# One block mask per (grid, kernel, dilation, device); an encoder has three levels.
_BLOCK_MASK_CACHE_MAX = 16
_block_masks: OrderedDict[tuple, BlockMask] = OrderedDict()
_compiled_flex_attention = None


def neighborhood_window_start(
    index: torch.Tensor, length: int, kernel: int, dilation: int
) -> torch.Tensor:
    """First attended position of ``index``'s residue class, in class units."""
    residue = index % dilation
    class_length = (length - 1 - residue) // dilation + 1
    return torch.minimum(
        (index // dilation - kernel // 2).clamp(min=0), class_length - kernel
    )


def neighborhood_mask_mod(
    height: int, width: int, kernel: tuple[int, int], dilation: tuple[int, int]
):
    kernel_h, kernel_w = kernel
    dilation_h, dilation_w = dilation

    def mask_mod(batch, head, q, kv):
        del batch, head
        q_h, q_w = q // width, q % width
        kv_h, kv_w = kv // width, kv % width
        start_h = neighborhood_window_start(q_h, height, kernel_h, dilation_h)
        start_w = neighborhood_window_start(q_w, width, kernel_w, dilation_w)
        return (
            (q_h % dilation_h == kv_h % dilation_h)
            & (q_w % dilation_w == kv_w % dilation_w)
            & (kv_h // dilation_h >= start_h)
            & (kv_h // dilation_h < start_h + kernel_h)
            & (kv_w // dilation_w >= start_w)
            & (kv_w // dilation_w < start_w + kernel_w)
        )

    return mask_mod


def _check_geometry(
    height: int, width: int, kernel: tuple[int, int], dilation: tuple[int, int]
) -> None:
    for name, length, k, d in (
        ("height", height, kernel[0], dilation[0]),
        ("width", width, kernel[1], dilation[1]),
    ):
        if k < 1 or d < 1:
            raise ValueError(
                f"LiDAR neighborhood attention needs positive kernel and dilation, "
                f"got kernel {kernel}, dilation {dilation}."
            )
        # Every residue class must hold a whole window, the shortest one included.
        if length < k * d:
            raise ValueError(
                f"LiDAR neighborhood attention {name} {length} is smaller than "
                f"kernel {k} x dilation {d}."
            )


def _kv_block_table(
    height: int,
    width: int,
    kernel: tuple[int, int],
    dilation: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per query block: the number and the ids of key blocks any of its windows touch."""
    tokens = height * width
    blocks = math.ceil(tokens / BLOCK_SIZE)
    counts = torch.empty(blocks, dtype=torch.int32, device=device)
    indices = torch.empty(blocks, blocks, dtype=torch.int32, device=device)
    offsets_h = torch.arange(kernel[0], device=device)
    offsets_w = torch.arange(kernel[1], device=device)
    for first in range(0, tokens, _QUERY_CHUNK):
        q = torch.arange(first, min(first + _QUERY_CHUNK, tokens), device=device)
        q_h, q_w = q // width, q % width
        rows = (
            neighborhood_window_start(q_h, height, kernel[0], dilation[0])[:, None]
            + offsets_h
        ) * dilation[0] + (q_h % dilation[0])[:, None]
        cols = (
            neighborhood_window_start(q_w, width, kernel[1], dilation[1])[:, None]
            + offsets_w
        ) * dilation[1] + (q_w % dilation[1])[:, None]
        kv_blocks = (
            (rows[:, :, None] * width + cols[:, None, :]) // BLOCK_SIZE
        ).flatten(1)
        q_rows = (q - first) // BLOCK_SIZE
        chunk_rows = math.ceil(q.numel() / BLOCK_SIZE)
        present = torch.zeros(chunk_rows * blocks, dtype=torch.bool, device=device)
        present[(q_rows[:, None] * blocks + kv_blocks).flatten()] = True
        present = present.view(chunk_rows, blocks)
        start = first // BLOCK_SIZE
        counts[start : start + chunk_rows] = present.sum(-1, dtype=torch.int32)
        # Present blocks first, in ascending order; the kernel reads counts[row] of them.
        indices[start : start + chunk_rows] = (
            present.to(torch.int8).argsort(dim=-1, descending=True, stable=True)
        ).to(torch.int32)
    return counts, indices


def neighborhood_block_mask(
    height: int,
    width: int,
    kernel: tuple[int, int],
    dilation: tuple[int, int],
    device: torch.device,
) -> BlockMask:
    _check_geometry(height, width, kernel, dilation)
    key = (height, width, kernel, dilation, str(device))
    mask = _block_masks.get(key)
    if mask is not None:
        _block_masks.move_to_end(key)
        return mask
    counts, indices = _kv_block_table(height, width, kernel, dilation, device)
    tokens = height * width
    mask = BlockMask.from_kv_blocks(
        counts[None, None],
        indices[None, None],
        BLOCK_SIZE=(BLOCK_SIZE, BLOCK_SIZE),
        mask_mod=neighborhood_mask_mod(height, width, kernel, dilation),
        seq_lengths=(tokens, tokens),
        compute_q_blocks=False,
    )
    if len(_block_masks) >= _BLOCK_MASK_CACHE_MAX:
        _block_masks.popitem(last=False)
    _block_masks[key] = mask
    return mask


def _compiled() -> object:
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
    return _compiled_flex_attention


def neighborhood_attention_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kernel_size: tuple[int, int],
    dilation: tuple[int, int],
    scale: float,
) -> torch.Tensor:
    """``na2d`` semantics on fp32 ``[B, H, W, heads, head_dim]`` tensors."""
    if q.ndim != 5 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            "LiDAR neighborhood attention needs matching [B, H, W, heads, head_dim] "
            f"tensors, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}."
        )
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError("LiDAR neighborhood attention runs the fp32 tokenizer only.")
    batch, height, width, heads, head_dim = q.shape
    kernel = (int(kernel_size[0]), int(kernel_size[1]))
    dilation = (int(dilation[0]), int(dilation[1]))
    mask = neighborhood_block_mask(height, width, kernel, dilation, q.device)

    def to_flex(x: torch.Tensor) -> torch.Tensor:
        return (
            x.reshape(batch, height * width, heads, head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    q_flex, k_flex, v_flex = to_flex(q), to_flex(k), to_flex(v)
    if q.is_cuda:
        # The frame count per streaming chunk varies; keep it symbolic so the
        # three spatial levels stay within dynamo's recompile budget.
        if batch > 1:
            for tensor in (q_flex, k_flex, v_flex):
                torch._dynamo.mark_dynamic(tensor, 0)
        out = _compiled()(
            q_flex,
            k_flex,
            v_flex,
            block_mask=mask,
            scale=scale,
            kernel_options=_KERNEL_OPTIONS,
        )
    else:
        if height * width > _EAGER_CPU_MAX_TOKENS:
            raise RuntimeError(
                "Eager CPU LiDAR neighborhood attention is for small test grids "
                f"(at most {_EAGER_CPU_MAX_TOKENS} tokens); use CUDA."
            )
        out = flex_attention(q_flex, k_flex, v_flex, block_mask=mask, scale=scale)
    return out.transpose(1, 2).reshape(batch, height, width, heads, head_dim)
