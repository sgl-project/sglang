"""Read-only forward Triton convolution for diffusion blocks in Clover."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bd_gdn_preview_conv_kernel(
    input_ptr,
    clean_state_ptr,
    cache_indices_ptr,
    weight_ptr,
    bias_ptr,
    right_weight_ptr,
    active_valid_ptr,
    output_ptr,
    input_stride_t,
    input_stride_c,
    state_stride_slot,
    state_stride_c,
    state_stride_w,
    weight_stride_c,
    weight_stride_w,
    right_weight_stride_c,
    right_weight_stride_w,
    output_stride_t,
    output_stride_c,
    NUM_CHANNELS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INDEPENDENT_SYMMETRIC: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NUM_TOKEN_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channel_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    token_program = tl.program_id(1)
    request_idx = token_program // NUM_TOKEN_TILES
    token_tile = token_program % NUM_TOKEN_TILES
    token_offsets = token_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    channel_mask = channel_offsets < NUM_CHANNELS
    token_mask = token_offsets < BLOCK_SIZE
    token_indices = request_idx * BLOCK_SIZE + token_offsets

    cache_index = tl.load(cache_indices_ptr + request_idx)
    request_active = cache_index >= 0
    safe_cache_index = tl.maximum(cache_index, 0)
    output_active = (
        tl.load(active_valid_ptr + token_indices, mask=token_mask, other=0) != 0
    ) & request_active

    if HAS_BIAS:
        bias = tl.load(bias_ptr + channel_offsets, mask=channel_mask, other=0.0).to(
            tl.float32
        )
    else:
        bias = tl.zeros((BLOCK_C,), dtype=tl.float32)
    accumulator = tl.broadcast_to(bias[None, :], (BLOCK_T, BLOCK_C))

    for tap in tl.static_range(WIDTH):
        relative_source = token_offsets + tap - (WIDTH - 1)
        from_current = relative_source >= 0
        current_token = request_idx * BLOCK_SIZE + tl.maximum(relative_source, 0)
        current_valid = (
            tl.load(
                active_valid_ptr + current_token,
                mask=token_mask & from_current,
                other=0,
            )
            != 0
        ) & request_active
        current_value = tl.load(
            input_ptr
            + current_token[:, None] * input_stride_t
            + channel_offsets[None, :] * input_stride_c,
            mask=(token_mask & from_current & current_valid)[:, None]
            & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        history_offset = relative_source + (WIDTH - 1)
        history_value = tl.load(
            clean_state_ptr
            + safe_cache_index * state_stride_slot
            + channel_offsets[None, :] * state_stride_c
            + tl.maximum(history_offset, 0)[:, None] * state_stride_w,
            mask=(token_mask & ~from_current & request_active)[:, None]
            & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        source_value = tl.where(from_current[:, None], current_value, history_value)
        weight = tl.load(
            weight_ptr + channel_offsets * weight_stride_c + tap * weight_stride_w,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += source_value * weight[None, :]

    if INDEPENDENT_SYMMETRIC:
        right_accumulator = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for tap in tl.static_range(WIDTH - 1):
            distance = WIDTH - 1 - tap
            right_offset = token_offsets + distance
            right_token = request_idx * BLOCK_SIZE + right_offset
            right_valid = (
                tl.load(
                    active_valid_ptr + right_token,
                    mask=token_mask & (right_offset < BLOCK_SIZE),
                    other=0,
                )
                != 0
            ) & request_active
            right_value = tl.load(
                input_ptr
                + right_token[:, None] * input_stride_t
                + channel_offsets[None, :] * input_stride_c,
                mask=(token_mask & (right_offset < BLOCK_SIZE) & right_valid)[:, None]
                & channel_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                right_weight_ptr
                + channel_offsets * right_weight_stride_c
                + (distance - 1) * right_weight_stride_w,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            right_accumulator += right_value * weight[None, :]
        accumulator += right_accumulator

    output = accumulator * tl.sigmoid(accumulator)
    output = tl.where(output_active[:, None], output, 0.0)
    tl.store(
        output_ptr
        + token_indices[:, None] * output_stride_t
        + channel_offsets[None, :] * output_stride_c,
        output,
        mask=token_mask[:, None] & channel_mask[None, :],
    )


def supports_clover_local_memory_conv(
    mixed_qkv: torch.Tensor,
    *,
    conv_weights: torch.Tensor,
    block_size: int,
    activation: str,
) -> bool:
    channels = mixed_qkv.shape[-1] if mixed_qkv.ndim == 2 else 0
    width = conv_weights.numel() // channels if channels else 0
    return (
        mixed_qkv.is_cuda
        and mixed_qkv.dtype in (torch.bfloat16, torch.float16)
        and mixed_qkv.ndim == 2
        and block_size in (1, 2, 4, 8, 16, 32)
        and width == 4
        and activation in ("silu", "swish")
        and torch.cuda.get_device_capability(mixed_qkv.device)[0] in (9, 10)
    )


def clover_local_memory_conv(
    mixed_qkv: torch.Tensor,
    *,
    clean_conv_states: torch.Tensor,
    cache_indices: torch.Tensor,
    conv_weights: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str,
    block_size: int,
    active_valid_mask: torch.Tensor,
    conv_mode: str = "causal",
    conv_right_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convolve a noisy block without copying or updating clean history."""
    num_tokens, channels = mixed_qkv.shape
    batch_size = cache_indices.numel()
    if num_tokens != batch_size * block_size:
        raise ValueError(
            "BD-GDN preview requires one fixed block per request: "
            f"{num_tokens=} != {batch_size=} * {block_size=}"
        )
    if not supports_clover_local_memory_conv(
        mixed_qkv,
        conv_weights=conv_weights,
        block_size=block_size,
        activation=activation,
    ):
        raise NotImplementedError(
            "BD-GDN preview conv Triton supports CUDA BF16/FP16, block_size in "
            "{1, 2, 4, 8, 16, 32}, "
            "width=4, and SiLU"
        )
    width = conv_weights.numel() // channels
    if clean_conv_states.ndim != 3 or clean_conv_states.shape[1:] != (
        channels,
        width - 1,
    ):
        raise ValueError(
            "BD-GDN clean conv state must use [slot,C,W-1] layout, got "
            f"{tuple(clean_conv_states.shape)}"
        )
    if active_valid_mask.numel() != num_tokens:
        raise ValueError(f"{active_valid_mask.numel()=} does not match {num_tokens=}")
    independent_symmetric = conv_mode == "independent_symmetric"
    if conv_mode not in ("causal", "independent_symmetric"):
        raise NotImplementedError(
            f"BD-GDN preview convolution does not support conv_mode={conv_mode!r}"
        )
    weights = conv_weights.reshape(channels, width)
    expected_right_shape = (channels, 1, width - 1)
    if independent_symmetric and (
        conv_right_weight is None
        or tuple(conv_right_weight.shape) != expected_right_shape
    ):
        raise ValueError(
            "independent_symmetric BD-GDN convolution requires right weight "
            f"shape {expected_right_shape}, got "
            f"{None if conv_right_weight is None else tuple(conv_right_weight.shape)}"
        )
    right_weights = (
        conv_right_weight.reshape(channels, width - 1)
        if conv_right_weight is not None
        else weights
    )
    output = torch.empty_like(mixed_qkv)
    block_t = 32 if block_size == 128 else triton.next_power_of_2(block_size)
    num_token_tiles = triton.cdiv(block_size, block_t)
    # For the latency-critical MRR=1 / block-32 preview on Blackwell, a
    # 64-channel tile gives the small convolution more parallel programs and
    # is faster than the generic 128-channel tile.  Keep the established
    # geometry for other request shapes.
    block_c = 64 if batch_size == 1 and block_size == 32 else 128
    grid = (triton.cdiv(channels, block_c), batch_size * num_token_tiles)
    _bd_gdn_preview_conv_kernel[grid](
        mixed_qkv,
        clean_conv_states,
        cache_indices,
        weights,
        bias if bias is not None else mixed_qkv,
        right_weights,
        active_valid_mask,
        output,
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        clean_conv_states.stride(0),
        clean_conv_states.stride(1),
        clean_conv_states.stride(2),
        weights.stride(0),
        weights.stride(1),
        right_weights.stride(0),
        right_weights.stride(1),
        output.stride(0),
        output.stride(1),
        NUM_CHANNELS=channels,
        BLOCK_SIZE=block_size,
        WIDTH=width,
        HAS_BIAS=bias is not None,
        INDEPENDENT_SYMMETRIC=independent_symmetric,
        BLOCK_T=block_t,
        NUM_TOKEN_TILES=num_token_tiles,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=2,
    )
    return output
