from __future__ import annotations

from typing import Optional

import msgspec
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_IDX_SENTINEL = tl.constexpr(2147483647)
_MARKOV_BLOCK_V = 256
_MARKOV_BLOCK_R = 32


class MarkovGreedyStep:
    """One greedy markov draft step, fused.

    Computes ``argmax_v(base_logits[:, v] + dot(w2_weight[v, :], prev_embeds))``
    in a single pass over the (vocab x rank) weight: no full-vocab bias or
    step-logits materialization, no separate GEMV / add / two-pass argmax
    launches. Numerics: the dot and the add accumulate in fp32, while
    MarkovSampleStep rounds the bias and the sum to the step dtype; near-tie
    winners can differ, so a one-row block can draft differently from the same
    row in a larger batch. Drafts are proposals only (target verify guards the
    output), so the impact is bounded to accept-rate noise on near ties.
    """

    @classmethod
    def execute(
        cls,
        *,
        base_logits: torch.Tensor,
        prev_embeds: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.is_cuda:
            return cls.triton(
                base_logits=base_logits, prev_embeds=prev_embeds, w2_weight=w2_weight
            )
        return cls.torch(
            base_logits=base_logits, prev_embeds=prev_embeds, w2_weight=w2_weight
        )

    @classmethod
    def torch(
        cls,
        *,
        base_logits: torch.Tensor,
        prev_embeds: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> torch.Tensor:
        return markov_greedy_step(
            base_logits=base_logits, prev_embeds=prev_embeds, w2_weight=w2_weight
        )

    @classmethod
    def triton(
        cls,
        *,
        base_logits: torch.Tensor,
        prev_embeds: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> torch.Tensor:
        return markov_greedy_step_triton(
            base_logits=base_logits, prev_embeds=prev_embeds, w2_weight=w2_weight
        )


def markov_greedy_step(
    *,
    base_logits: torch.Tensor,
    prev_embeds: torch.Tensor,
    w2_weight: torch.Tensor,
) -> torch.Tensor:
    step_logits = base_logits + F.linear(prev_embeds, w2_weight)
    return torch.argmax(step_logits, dim=-1)


@triton.jit
def _markov_greedy_partial_kernel(
    base_ptr,
    embed_ptr,
    w2_ptr,
    tile_val_ptr,
    tile_idx_ptr,
    V,
    R,
    stride_base_row,
    stride_embed_row,
    stride_w2_v,
    n_tiles,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V
    acc = tl.zeros([BLOCK_V], dtype=tl.float32)
    for r0 in range(0, R, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R
        embed = tl.load(
            embed_ptr + row * stride_embed_row + offs_r, mask=mask_r, other=0.0
        ).to(tl.float32)
        w2 = tl.load(
            w2_ptr + offs_v[:, None] * stride_w2_v + offs_r[None, :],
            mask=mask_v[:, None] & mask_r[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w2 * embed[None, :], axis=1)
    base = tl.load(
        base_ptr + row * stride_base_row + offs_v, mask=mask_v, other=float("-inf")
    ).to(tl.float32)
    score = tl.where(mask_v, base + acc, float("-inf"))
    tile_best = tl.max(score, axis=0)
    # First-index tie-break within the tile, matching torch.argmax.
    idx = tl.where(score == tile_best, offs_v, _IDX_SENTINEL)
    tl.store(tile_val_ptr + row * n_tiles + tile, tile_best)
    tl.store(tile_idx_ptr + row * n_tiles + tile, tl.min(idx, axis=0))


@triton.jit
def _markov_greedy_combine_kernel(
    tile_val_ptr,
    tile_idx_ptr,
    next_tokens_ptr,
    n_tiles,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < n_tiles
    vals = tl.load(tile_val_ptr + row * n_tiles + offs, mask=mask, other=float("-inf"))
    idxs = tl.load(tile_idx_ptr + row * n_tiles + offs, mask=mask, other=_IDX_SENTINEL)
    best = tl.max(vals, axis=0)
    # Lowest global index among equal-valued tiles, matching torch.argmax.
    cand = tl.where(vals == best, idxs, _IDX_SENTINEL)
    tl.store(next_tokens_ptr + row, tl.min(cand, axis=0).to(tl.int64))


def markov_greedy_step_triton(
    *,
    base_logits: torch.Tensor,
    prev_embeds: torch.Tensor,
    w2_weight: torch.Tensor,
) -> torch.Tensor:
    bs, vocab = base_logits.shape
    rank = w2_weight.shape[1]
    device = base_logits.device
    assert base_logits.stride(1) == 1, "base_logits rows must be contiguous"
    assert w2_weight.stride(1) == 1, "markov_w2 weight rows must be contiguous"
    prev_embeds = prev_embeds.contiguous()

    n_tiles = triton.cdiv(vocab, _MARKOV_BLOCK_V)
    tile_vals = torch.empty((bs, n_tiles), dtype=torch.float32, device=device)
    tile_idxs = torch.empty((bs, n_tiles), dtype=torch.int32, device=device)
    next_tokens = torch.empty((bs,), dtype=torch.int64, device=device)

    _markov_greedy_partial_kernel[(bs, n_tiles)](
        base_logits,
        prev_embeds,
        w2_weight,
        tile_vals,
        tile_idxs,
        vocab,
        rank,
        base_logits.stride(0),
        prev_embeds.stride(0),
        w2_weight.stride(0),
        n_tiles,
        BLOCK_V=_MARKOV_BLOCK_V,
        BLOCK_R=_MARKOV_BLOCK_R,
    )
    _markov_greedy_combine_kernel[(bs,)](
        tile_vals,
        tile_idxs,
        next_tokens,
        n_tiles,
        BLOCK_TILES=triton.next_power_of_2(n_tiles),
    )
    return next_tokens


_STACKED_WEIGHT_CACHE: dict[int, _StackedWkvWeight] = {}


class CommitKvProj:
    @classmethod
    def execute(
        cls,
        *,
        main_x: torch.Tensor,
        wkv_linears: list[torch.nn.Module],
        allow_strided_output: bool = False,
    ) -> list[torch.Tensor]:
        if main_x.is_cuda and _fused_commit_kv_proj_supported(wkv_linears=wkv_linears):
            return cls.triton(
                main_x=main_x,
                wkv_linears=wkv_linears,
                allow_strided_output=allow_strided_output,
            )
        return cls.torch(main_x=main_x, wkv_linears=wkv_linears)

    @classmethod
    def torch(
        cls,
        *,
        main_x: torch.Tensor,
        wkv_linears: list[torch.nn.Module],
    ) -> list[torch.Tensor]:
        return commit_kv_proj(main_x=main_x, wkv_linears=wkv_linears)

    @classmethod
    def triton(
        cls,
        *,
        main_x: torch.Tensor,
        wkv_linears: list[torch.nn.Module],
        allow_strided_output: bool = False,
    ) -> list[torch.Tensor]:
        return commit_kv_proj_fused(
            main_x=main_x,
            wkv_linears=wkv_linears,
            allow_strided_output=allow_strided_output,
        )


def commit_kv_proj(
    *,
    main_x: torch.Tensor,
    wkv_linears: list[torch.nn.Module],
) -> list[torch.Tensor]:
    return [linear(main_x)[0] for linear in wkv_linears]


def commit_kv_proj_fused(
    *,
    main_x: torch.Tensor,
    wkv_linears: list[torch.nn.Module],
    allow_strided_output: bool = False,
) -> list[torch.Tensor]:
    num_stages = len(wkv_linears)
    stacked = _stacked_wkv_weight(wkv_linears=wkv_linears)

    if stacked.mxfp8_scale is not None:
        quant_method = wkv_linears[0].quant_method
        if quant_method.mxfp8_dense_backend.is_gfx95_mxfp8_aiter():
            kv_all = quant_method.w8a8_mxfp8_linear(
                input=main_x,
                weight=stacked.weight,
                weight_scale_ue8m0=stacked.mxfp8_scale,
                input_scale=None,
                bias=None,
            )
        else:
            kv_all = quant_method.w8a8_mxfp8_linear(
                input=main_x,
                weight=stacked.weight,
                weight_scale=stacked.mxfp8_scale,
                input_scale=None,
                bias=None,
            )
    elif stacked.fp8_scale is not None:
        quant_method = wkv_linears[0].quant_method
        kv_all = quant_method.w8a8_block_fp8_linear(
            input=main_x,
            weight=stacked.weight,
            block_size=quant_method.quant_config.weight_block_size,
            weight_scale=stacked.fp8_scale,
            input_scale=None,
            bias=None,
        )
    else:
        kv_all = torch.nn.functional.linear(main_x, stacked.weight)

    head_dim = kv_all.shape[-1] // num_stages
    slices = list(kv_all.split(head_dim, dim=-1))
    return slices if allow_strided_output else [kv.contiguous() for kv in slices]


class _StackedWkvWeight(msgspec.Struct):
    weight: torch.Tensor
    fp8_scale: Optional[torch.Tensor]
    mxfp8_scale: Optional[torch.Tensor] = None


def _stacked_wkv_weight(*, wkv_linears: list[torch.nn.Module]) -> _StackedWkvWeight:
    key = id(wkv_linears[0])
    cached = _STACKED_WEIGHT_CACHE.get(key)
    if cached is None:
        cached = _build_stacked_wkv_weight(wkv_linears=wkv_linears)
        _STACKED_WEIGHT_CACHE[key] = cached
    return cached


def _block_quant_stack_applies(*, wkv_linears: list[torch.nn.Module]) -> bool:
    quant_method = wkv_linears[0].quant_method
    block_quant = hasattr(quant_method, "block_quant") and quant_method.block_quant
    if not (block_quant and hasattr(quant_method, "w8a8_block_fp8_linear")):
        return False
    # on gfx950 only the aiter MXFP8 route stacks (the Triton block kernel's ue8m0 activation
    # quant is CUDA-only); the others serve 32x32-block weights through apply()
    backend = getattr(quant_method, "mxfp8_dense_backend", None)
    if backend is not None and backend.is_gfx95():
        if not (
            backend.is_gfx95_mxfp8_aiter()
            and all(
                getattr(linear, "block_fp8_mxfp8_ready", False)
                for linear in wkv_linears
            )
        ):
            return False
    block_out = quant_method.quant_config.weight_block_size[0]
    # the gfx950 native MXFP8 route keeps the weight in a 3-D lane-order layout
    return all(
        linear.weight.dtype == torch.float8_e4m3fn
        and linear.weight.dim() == 2
        and linear.weight.shape[0] % block_out == 0
        for linear in wkv_linears
    )


def _dequant_supported(linear: torch.nn.Module) -> bool:
    """Mirrors the preconditions asserted in _dequant_linear_weight."""
    weight = linear.weight
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return True
    # the gfx950 native MXFP8 route keeps the weight in a 3-D lane-order layout
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        return False
    block = 128
    out_dim, in_dim = weight.shape
    expected_scale_shape = (
        (out_dim + block - 1) // block,
        (in_dim + block - 1) // block,
    )
    return tuple(linear.weight_scale_inv.shape) == expected_scale_shape


def _fused_commit_kv_proj_supported(*, wkv_linears: list[torch.nn.Module]) -> bool:
    """Whether _build_stacked_wkv_weight can handle these weights; unsupported
    quant schemes fall back to the per-linear torch path in execute()."""
    if _block_quant_stack_applies(wkv_linears=wkv_linears):
        return True
    return all(_dequant_supported(linear) for linear in wkv_linears)


def _build_stacked_wkv_weight(
    *, wkv_linears: list[torch.nn.Module]
) -> _StackedWkvWeight:
    if _block_quant_stack_applies(wkv_linears=wkv_linears):
        weight = torch.cat([linear.weight for linear in wkv_linears], dim=0)
        backend = getattr(wkv_linears[0].quant_method, "mxfp8_dense_backend", None)
        if (
            backend is not None
            and (backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl())
            and all(
                getattr(linear, "block_fp8_mxfp8_ready", False)
                and linear.weight.shape[0] % 128 == 0
                for linear in wkv_linears
            )
        ):
            # 128-row-aligned scale tiles concatenate without breaking the swizzle.
            scale = torch.cat(
                [linear.weight_scale_inv_swizzled.reshape(-1) for linear in wkv_linears]
            )
            return _StackedWkvWeight(weight=weight, fp8_scale=None, mxfp8_scale=scale)
        if backend is not None and backend.is_gfx95_mxfp8_aiter():
            # compact ue8m0 block scales, one row per 32 output rows: whole blocks stack
            scale = torch.cat(
                [linear.weight_scale_mx_e8m0 for linear in wkv_linears], dim=0
            )
            return _StackedWkvWeight(weight=weight, fp8_scale=None, mxfp8_scale=scale)
        if wkv_linears[0].weight_scale_inv.dtype == torch.int32:
            from sglang.srt.layers.quantization.fp8_utils import (
                inverse_transform_scale_ue8m0,
                transform_scale_ue8m0,
            )

            sf_fp32 = torch.cat(
                [
                    inverse_transform_scale_ue8m0(
                        linear.weight_scale_inv, mn=linear.weight.shape[0]
                    )
                    for linear in wkv_linears
                ],
                dim=0,
            )
            scale = transform_scale_ue8m0(sf_fp32, mn=weight.shape[0])
            return _StackedWkvWeight(weight=weight, fp8_scale=scale)
        scale = torch.cat([linear.weight_scale_inv for linear in wkv_linears], dim=0)
        if scale.dim() >= 2 and scale.stride(-2) != 1:
            scale = scale.transpose(-2, -1).contiguous().transpose(-2, -1)
        return _StackedWkvWeight(weight=weight, fp8_scale=scale)
    weight = torch.cat(
        [_dequant_linear_weight(linear) for linear in wkv_linears], dim=0
    )
    return _StackedWkvWeight(weight=weight, fp8_scale=None)


def _dequant_linear_weight(linear: torch.nn.Module) -> torch.Tensor:
    weight = linear.weight
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return weight.to(torch.bfloat16)
    assert weight.dtype == torch.float8_e4m3fn, (
        f"unsupported wkv weight dtype {weight.dtype} for the fused commit kv proj; "
        f"execute() should have routed this to the torch path "
        f"(_fused_commit_kv_proj_supported)"
    )
    block = 128
    scale = linear.weight_scale_inv
    out_dim, in_dim = weight.shape
    expected_scale_shape = (
        (out_dim + block - 1) // block,
        (in_dim + block - 1) // block,
    )
    assert tuple(scale.shape) == expected_scale_shape, (
        f"wkv weight_scale_inv shape {tuple(scale.shape)} does not match the "
        f"128x128 block grid {expected_scale_shape} for weight {tuple(weight.shape)}; "
        f"execute() should have routed this to the torch path "
        f"(_fused_commit_kv_proj_supported)"
    )
    scale_full = scale.repeat_interleave(block, dim=0)[:out_dim]
    scale_full = scale_full.repeat_interleave(block, dim=1)[:, :in_dim]
    return (weight.to(torch.float32) * scale_full.to(torch.float32)).to(torch.bfloat16)
