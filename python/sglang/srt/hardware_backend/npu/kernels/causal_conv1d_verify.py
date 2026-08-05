from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_linear_verify_kernel(
    x_ptr,
    state_ptr,
    weight_ptr,
    bias_ptr,
    state_indices_ptr,
    snapshot_ptr,
    snapshot_indices_ptr,
    out_ptr,
    channels,
    stride_x_batch,
    stride_x_channel,
    stride_x_step,
    stride_state_slot,
    stride_state_channel,
    stride_state_window,
    stride_weight_channel,
    stride_weight_window,
    stride_snapshot_slot,
    stride_snapshot_step,
    stride_snapshot_channel,
    stride_snapshot_window,
    stride_out_batch,
    stride_out_channel,
    stride_out_step,
    KERNEL_WIDTH: tl.constexpr,
    STEPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    UPDATE_PERSISTENT_STATE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    channel = pid_channel * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channel < channels

    state_idx = tl.load(state_indices_ptr + pid_batch).to(tl.int64)
    snapshot_idx = tl.load(snapshot_indices_ptr + pid_batch).to(tl.int64)
    active = (state_idx >= 0) & (snapshot_idx >= 0)
    mask = active & channel_mask

    state_base = (
        state_ptr
        + state_idx * stride_state_slot
        + channel * stride_state_channel
    )
    weight_base = weight_ptr + channel * stride_weight_channel

    history0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    history1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    history2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    history3 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    history4 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    if KERNEL_WIDTH >= 2:
        history0 = tl.load(
            state_base, mask=mask, other=0.0
        ).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        history1 = tl.load(
            state_base + stride_state_window, mask=mask, other=0.0
        ).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        history2 = tl.load(
            state_base + 2 * stride_state_window, mask=mask, other=0.0
        ).to(tl.float32)
    if KERNEL_WIDTH >= 5:
        history3 = tl.load(
            state_base + 3 * stride_state_window, mask=mask, other=0.0
        ).to(tl.float32)
    if KERNEL_WIDTH >= 6:
        history4 = tl.load(
            state_base + 4 * stride_state_window, mask=mask, other=0.0
        ).to(tl.float32)

    weight0 = tl.load(weight_base, mask=channel_mask, other=0.0).to(tl.float32)
    weight1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    weight2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    weight3 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    weight4 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    weight5 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    if KERNEL_WIDTH >= 2:
        weight1 = tl.load(
            weight_base + stride_weight_window,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        weight2 = tl.load(
            weight_base + 2 * stride_weight_window,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        weight3 = tl.load(
            weight_base + 3 * stride_weight_window,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
    if KERNEL_WIDTH >= 5:
        weight4 = tl.load(
            weight_base + 4 * stride_weight_window,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
    if KERNEL_WIDTH >= 6:
        weight5 = tl.load(
            weight_base + 5 * stride_weight_window,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + channel, mask=channel_mask, other=0.0).to(
            tl.float32
        )
    else:
        bias = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for step in tl.static_range(0, STEPS):
        current = tl.load(
            x_ptr
            + pid_batch * stride_x_batch
            + channel * stride_x_channel
            + step * stride_x_step,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc = bias
        if KERNEL_WIDTH == 1:
            acc += current * weight0
        elif KERNEL_WIDTH == 2:
            acc += history0 * weight0 + current * weight1
            history0 = current
        elif KERNEL_WIDTH == 3:
            acc += history0 * weight0 + history1 * weight1 + current * weight2
            history0 = history1
            history1 = current
        elif KERNEL_WIDTH == 4:
            acc += (
                history0 * weight0
                + history1 * weight1
                + history2 * weight2
                + current * weight3
            )
            history0 = history1
            history1 = history2
            history2 = current
        elif KERNEL_WIDTH == 5:
            acc += (
                history0 * weight0
                + history1 * weight1
                + history2 * weight2
                + history3 * weight3
                + current * weight4
            )
            history0 = history1
            history1 = history2
            history2 = history3
            history3 = current
        elif KERNEL_WIDTH == 6:
            acc += (
                history0 * weight0
                + history1 * weight1
                + history2 * weight2
                + history3 * weight3
                + history4 * weight4
                + current * weight5
            )
            history0 = history1
            history1 = history2
            history2 = history3
            history3 = history4
            history4 = current

        if SILU:
            acc = acc / (1.0 + tl.exp(-acc))
        tl.store(
            out_ptr
            + pid_batch * stride_out_batch
            + channel * stride_out_channel
            + step * stride_out_step,
            acc,
            mask=mask,
        )

        snapshot_base = (
            snapshot_ptr
            + snapshot_idx * stride_snapshot_slot
            + step * stride_snapshot_step
            + channel * stride_snapshot_channel
        )
        if KERNEL_WIDTH >= 2:
            tl.store(snapshot_base, history0, mask=mask)
        if KERNEL_WIDTH >= 3:
            tl.store(
                snapshot_base + stride_snapshot_window, history1, mask=mask
            )
        if KERNEL_WIDTH >= 4:
            tl.store(
                snapshot_base + 2 * stride_snapshot_window,
                history2,
                mask=mask,
            )
        if KERNEL_WIDTH >= 5:
            tl.store(
                snapshot_base + 3 * stride_snapshot_window,
                history3,
                mask=mask,
            )
        if KERNEL_WIDTH >= 6:
            tl.store(
                snapshot_base + 4 * stride_snapshot_window,
                history4,
                mask=mask,
            )

    if UPDATE_PERSISTENT_STATE:
        if KERNEL_WIDTH >= 2:
            tl.store(state_base, history0, mask=mask)
        if KERNEL_WIDTH >= 3:
            tl.store(state_base + stride_state_window, history1, mask=mask)
        if KERNEL_WIDTH >= 4:
            tl.store(state_base + 2 * stride_state_window, history2, mask=mask)
        if KERNEL_WIDTH >= 5:
            tl.store(state_base + 3 * stride_state_window, history3, mask=mask)
        if KERNEL_WIDTH >= 6:
            tl.store(state_base + 4 * stride_state_window, history4, mask=mask)


def causal_conv1d_linear_verify_npu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    activation: Optional[str] = "silu",
    update_persistent_state: bool = False,
) -> torch.Tensor:
    """Run fixed-width linear-chain verification and save every conv state.

    Shapes are ``x[B, C, T]``, ``conv_state[pool, C, W]``,
    ``weight[C, W + 1]`` and
    ``intermediate_conv_window[scratch, T, C, W]``.
    """
    if x.ndim != 3 or conv_state.ndim != 3 or weight.ndim != 2:
        raise ValueError("x and conv_state must be 3D and weight must be 2D")
    batch, channels, steps = x.shape
    state_window = conv_state.shape[2]
    kernel_width = weight.shape[1]
    if kernel_width != state_window + 1:
        raise ValueError("weight width must equal conv state window plus one")
    if not 2 <= kernel_width <= 6:
        raise ValueError("kernel width must be in [2, 6]")
    if conv_state.shape[1] != channels or weight.shape[0] != channels:
        raise ValueError("channel dimensions do not match")
    if intermediate_conv_window.ndim != 4 or tuple(
        intermediate_conv_window.shape[1:]
    ) != (steps, channels, state_window):
        raise ValueError(
            "intermediate_conv_window must have shape [scratch, T, C, W]"
        )
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != channels):
        raise ValueError("bias must have shape [C]")
    if activation not in (None, "silu", "swish"):
        raise ValueError("activation must be None, silu, or swish")
    tensors = [x, conv_state, weight, intermediate_conv_window]
    if bias is not None:
        tensors.append(bias)
    if any(t.device != x.device for t in tensors):
        raise ValueError("all data tensors must be on the same device")
    if conv_state.dtype != x.dtype:
        raise ValueError("x and conv_state must have the same dtype")
    if intermediate_conv_window.dtype != conv_state.dtype:
        raise ValueError(
            "intermediate_conv_window and conv_state must have the same dtype"
        )
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all data tensors must be contiguous")
    if conv_state_indices.shape != (batch,) or intermediate_state_indices.shape != (
        batch,
    ):
        raise ValueError("state index tensors must have shape [B]")
    if (
        conv_state_indices.device != x.device
        or intermediate_state_indices.device != x.device
    ):
        raise ValueError("state index tensors must be on the input device")

    conv_state_indices = conv_state_indices.to(torch.int32).contiguous()
    intermediate_state_indices = intermediate_state_indices.to(
        torch.int32
    ).contiguous()
    out = torch.zeros_like(x)
    # Eight verify steps keep the full convolution history live. A 512-channel
    # tile overflows the 192 KiB UB on Ascend 910; 256 leaves enough room for
    # compiler-generated multi-buffer temporaries.
    block_c = min(256, triton.next_power_of_2(channels))
    grid = (batch, triton.cdiv(channels, block_c))
    _causal_conv1d_linear_verify_kernel[grid](
        x,
        conv_state,
        weight,
        bias,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        out,
        channels,
        *x.stride(),
        *conv_state.stride(),
        *weight.stride(),
        *intermediate_conv_window.stride(),
        *out.stride(),
        KERNEL_WIDTH=kernel_width,
        STEPS=steps,
        HAS_BIAS=bias is not None,
        SILU=activation in ("silu", "swish"),
        UPDATE_PERSISTENT_STATE=update_persistent_state,
        BLOCK_C=block_c,
    )
    return out
