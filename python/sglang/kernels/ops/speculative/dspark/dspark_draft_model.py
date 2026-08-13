from __future__ import annotations

from typing import Optional

import msgspec
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda

_BLOCK_V = 1024
_IDX_SENTINEL = tl.constexpr(2147483647)


class SampleStepTokens:
    @classmethod
    def execute(
        cls,
        *,
        step_logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        exp_noise: torch.Tensor,
    ) -> torch.Tensor:
        if step_logits.is_cuda:
            return cls.triton(
                step_logits=step_logits,
                temperatures=temperatures,
                greedy_mask=greedy_mask,
                exp_noise=exp_noise,
            )
        return cls.torch(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
        )

    @classmethod
    def torch(
        cls,
        *,
        step_logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        exp_noise: torch.Tensor,
    ) -> torch.Tensor:
        return sample_step_tokens(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
        )

    @classmethod
    def triton(
        cls,
        *,
        step_logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        exp_noise: torch.Tensor,
    ) -> torch.Tensor:
        return sample_step_tokens_triton(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
        )


def sample_step_tokens(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
) -> torch.Tensor:
    probs = torch.softmax(step_logits.float() / temperatures[:, None], dim=-1)
    noise = torch.where(greedy_mask[:, None], 1.0, exp_noise)
    return probs.div_(noise).argmax(dim=-1)


@triton.jit
def _online_partial_kernel(
    logits_ptr,
    temperatures_ptr,
    greedy_mask_ptr,
    exp_noise_ptr,
    tile_max_ptr,
    partial_key_ptr,
    partial_idx_ptr,
    V,
    stride_row,
    n_tiles,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < V
    logits = tl.load(
        logits_ptr + row * stride_row + offs, mask=mask, other=float("-inf")
    ).to(tl.float32)
    temperature = tl.load(temperatures_ptr + row)
    s = logits / temperature
    tile_max = tl.max(s, axis=0)
    greedy = tl.load(greedy_mask_ptr + row) != 0
    noise = tl.load(exp_noise_ptr + row * V + offs, mask=mask, other=1.0)
    denom = tl.where(greedy, 1.0, noise)
    key = tl.exp(s - tile_max) / denom
    key = tl.where(mask, key, -1.0)
    tile_best = tl.max(key, axis=0)
    idx = tl.where(key == tile_best, offs, _IDX_SENTINEL)
    tl.store(tile_max_ptr + row * n_tiles + tile, tile_max)
    tl.store(partial_key_ptr + row * n_tiles + tile, tile_best)
    tl.store(partial_idx_ptr + row * n_tiles + tile, tl.min(idx, axis=0))


@triton.jit
def _online_combine_kernel(
    tile_max_ptr,
    partial_key_ptr,
    partial_idx_ptr,
    next_tokens_ptr,
    n_tiles,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < n_tiles
    tile_max = tl.load(
        tile_max_ptr + row * n_tiles + offs, mask=mask, other=float("-inf")
    )
    keys = tl.load(partial_key_ptr + row * n_tiles + offs, mask=mask, other=-1.0)
    idxs = tl.load(
        partial_idx_ptr + row * n_tiles + offs, mask=mask, other=_IDX_SENTINEL
    )
    global_max = tl.max(tile_max, axis=0)
    rescaled = keys * tl.exp(tile_max - global_max)
    rescaled = tl.where(mask, rescaled, -1.0)
    best = tl.max(rescaled, axis=0)
    cand = tl.where(rescaled == best, idxs, _IDX_SENTINEL)
    next_token = tl.min(cand, axis=0)
    # Degenerate rows (e.g. all -inf logits) leave cand all-sentinel; clamp to 0
    # so a valid token id is emitted instead of an out-of-range 2147483647.
    next_token = tl.where(next_token == _IDX_SENTINEL, 0, next_token)
    tl.store(next_tokens_ptr + row, next_token.to(tl.int64))


def sample_step_tokens_triton(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
) -> torch.Tensor:
    bs, V = step_logits.shape
    device = step_logits.device
    assert step_logits.stride(1) == 1, "step_logits rows must be contiguous"
    stride_row = step_logits.stride(0)
    temperatures = temperatures.to(torch.float32).contiguous()
    greedy_mask = greedy_mask.to(torch.int32).contiguous()
    exp_noise = exp_noise.to(torch.float32).contiguous()

    n_tiles = triton.cdiv(V, _BLOCK_V)
    block_tiles = triton.next_power_of_2(n_tiles)

    tile_max = torch.empty((bs, n_tiles), dtype=torch.float32, device=device)
    partial_key = torch.empty((bs, n_tiles), dtype=torch.float32, device=device)
    partial_idx = torch.empty((bs, n_tiles), dtype=torch.int32, device=device)
    next_tokens = torch.empty((bs,), dtype=torch.int64, device=device)

    tile_grid = (bs, n_tiles)
    row_grid = (bs,)

    _online_partial_kernel[tile_grid](
        step_logits,
        temperatures,
        greedy_mask,
        exp_noise,
        tile_max,
        partial_key,
        partial_idx,
        V,
        stride_row,
        n_tiles,
        BLOCK_V=_BLOCK_V,
    )
    _online_combine_kernel[row_grid](
        tile_max,
        partial_key,
        partial_idx,
        next_tokens,
        n_tiles,
        BLOCK_TILES=block_tiles,
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
    ) -> list[torch.Tensor]:
        if main_x.is_cuda and _fused_commit_kv_proj_supported(wkv_linears=wkv_linears):
            return cls.triton(main_x=main_x, wkv_linears=wkv_linears)
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
    ) -> list[torch.Tensor]:
        return commit_kv_proj_fused(main_x=main_x, wkv_linears=wkv_linears)


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
) -> list[torch.Tensor]:
    num_stages = len(wkv_linears)
    stacked = _stacked_wkv_weight(wkv_linears=wkv_linears)

    if stacked.fp8_scale is not None:
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
    return [
        kv_all[:, i * head_dim : (i + 1) * head_dim].contiguous()
        for i in range(num_stages)
    ]


class _StackedWkvWeight(msgspec.Struct):
    weight: torch.Tensor
    fp8_scale: Optional[torch.Tensor]


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
    block_out = quant_method.quant_config.weight_block_size[0]
    return all(
        linear.weight.dtype == torch.float8_e4m3fn
        and linear.weight.shape[0] % block_out == 0
        for linear in wkv_linears
    )


def _dequant_supported(linear: torch.nn.Module) -> bool:
    """Mirrors the preconditions asserted in _dequant_linear_weight."""
    weight = linear.weight
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return True
    if weight.dtype != torch.float8_e4m3fn:
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


_BLOCK = 1024


class BuildStepLocal:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(cls, *, bias: torch.Tensor, base_local: torch.Tensor) -> torch.Tensor:
        return build_step_local(bias=bias, base_local=base_local)

    @classmethod
    def triton(cls, *, bias: torch.Tensor, base_local: torch.Tensor) -> torch.Tensor:
        return build_step_local_triton(bias=bias, base_local=base_local)


def build_step_local(*, bias: torch.Tensor, base_local: torch.Tensor) -> torch.Tensor:
    per_partition = base_local.shape[-1]
    pad = per_partition - bias.shape[-1]
    padded = (
        F.pad(bias.to(torch.float32), (0, pad)) if pad > 0 else bias.to(torch.float32)
    )
    return base_local + padded


@triton.jit
def _build_step_local_kernel(
    bias_ptr,
    base_ptr,
    out_ptr,
    org_width,
    per_partition,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < per_partition
    base = tl.load(base_ptr + row * per_partition + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    bias = tl.load(
        bias_ptr + row * org_width + offs, mask=offs < org_width, other=0.0
    ).to(tl.float32)
    tl.store(out_ptr + row * per_partition + offs, base + bias, mask=mask)


def build_step_local_triton(
    *, bias: torch.Tensor, base_local: torch.Tensor
) -> torch.Tensor:
    bs, per_partition = base_local.shape
    org_width = bias.shape[-1]
    base_local = base_local.contiguous()
    bias = bias.contiguous()
    out = torch.empty(
        (bs, per_partition), dtype=torch.float32, device=base_local.device
    )
    grid = (bs, triton.cdiv(per_partition, _BLOCK))
    _build_step_local_kernel[grid](
        bias, base_local, out, org_width, per_partition, BLOCK=_BLOCK
    )
    return out
