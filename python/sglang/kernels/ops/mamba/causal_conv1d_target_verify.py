# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1


@triton.jit()
def _causal_conv1d_target_verify_npu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    intermediate_conv_window_ptr,
    intermediate_state_indices_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    retrieve_parent_token_ptr,
    out_ptr,
    batch,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    stride_x_batch,
    stride_x_dim: tl.constexpr,
    stride_x_token,
    stride_weight_dim: tl.constexpr,
    stride_weight_width: tl.constexpr,
    stride_state_batch,
    stride_state_dim: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_state_indices,
    stride_inter_batch,
    stride_inter_step,
    stride_inter_dim: tl.constexpr,
    stride_inter_window,
    stride_intermediate_state_indices,
    stride_next_batch,
    stride_next_token,
    stride_sibling_batch,
    stride_sibling_token,
    stride_parent_batch,
    stride_parent_token,
    stride_out_batch,
    stride_out_dim: tl.constexpr,
    stride_out_token,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_TREE: tl.constexpr,
    NP2_SEQLEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    B_TILE: tl.constexpr,
):
    """Compute verify outputs and checkpoints without mutating committed state."""
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)

    channels = pid_channel * BLOCK_N + tl.arange(0, BLOCK_N)
    channel_mask = channels < dim

    weight_base = weight_ptr + channels * stride_weight_dim
    weight0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    weight1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    weight2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    weight3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if KERNEL_WIDTH >= 1:
        weight0 = tl.load(
            weight_base, mask=channel_mask, other=0.0
        ).to(tl.float32)
    if KERNEL_WIDTH >= 2:
        weight1 = tl.load(
            weight_base + stride_weight_width, mask=channel_mask, other=0.0
        ).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        weight2 = tl.load(
            weight_base + 2 * stride_weight_width,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        weight3 = tl.load(
            weight_base + 3 * stride_weight_width,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(
            tl.float32
        )
    else:
        bias = tl.zeros((BLOCK_N,), dtype=tl.float32)

    token_offsets = tl.arange(0, NP2_SEQLEN)
    token_mask = token_offsets < seqlen

    for tile_offset in tl.static_range(0, B_TILE):
        batch_idx = pid_batch * B_TILE + tile_offset
        active = batch_idx < batch

        state_idx = tl.load(
            conv_state_indices_ptr + batch_idx * stride_state_indices,
            mask=active,
            other=pad_slot_id,
        ).to(tl.int64)
        active = active & (state_idx != pad_slot_id)
        intermediate_idx = tl.load(
            intermediate_state_indices_ptr
            + batch_idx * stride_intermediate_state_indices,
            mask=active,
            other=0,
        ).to(tl.int64)

        state_base = (
            conv_state_ptr
            + state_idx * stride_state_batch
            + channels * stride_state_dim
        )
        history0 = tl.zeros((BLOCK_N,), dtype=conv_state_ptr.dtype.element_ty)
        history1 = tl.zeros((BLOCK_N,), dtype=conv_state_ptr.dtype.element_ty)
        history2 = tl.zeros((BLOCK_N,), dtype=conv_state_ptr.dtype.element_ty)
        if KERNEL_WIDTH >= 2:
            history0 = tl.load(
                state_base, mask=active & channel_mask, other=0.0
            )
        if KERNEL_WIDTH >= 3:
            history1 = tl.load(
                state_base + stride_state_token,
                mask=active & channel_mask,
                other=0.0,
            )
        if KERNEL_WIDTH >= 4:
            history2 = tl.load(
                state_base + 2 * stride_state_token,
                mask=active & channel_mask,
                other=0.0,
            )

        x_base = x_ptr + batch_idx * stride_x_batch + channels * stride_x_dim
        out_base = (
            out_ptr + batch_idx * stride_out_batch + channels * stride_out_dim
        )
        intermediate_base = (
            intermediate_conv_window_ptr
            + intermediate_idx * stride_inter_batch
            + channels * stride_inter_dim
        )

        if HAS_TREE:
            next_tokens = tl.load(
                retrieve_next_token_ptr
                + batch_idx * stride_next_batch
                + token_offsets * stride_next_token,
                mask=active & token_mask,
                other=-1,
            ).to(tl.int32)
            next_siblings = tl.load(
                retrieve_next_sibling_ptr
                + batch_idx * stride_sibling_batch
                + token_offsets * stride_sibling_token,
                mask=active & token_mask,
                other=-1,
            ).to(tl.int32)
            parent_tokens = tl.zeros((NP2_SEQLEN,), dtype=tl.int32)

            # Keep tree traversal as a loop to avoid excessive code growth for
            # larger EAGLE trees. The common linear chain below is fully unrolled.
            for token_idx in tl.range(0, seqlen):
                child_idx = tl.sum(
                    tl.where(token_offsets == token_idx, next_tokens, 0)
                )
                if child_idx != -1:
                    parent_tokens = tl.where(
                        token_offsets == child_idx, token_idx, parent_tokens
                    )

                sibling_idx = tl.sum(
                    tl.where(token_offsets == token_idx, next_siblings, 0)
                )
                if sibling_idx != -1:
                    token_parent = tl.sum(
                        tl.where(token_offsets == token_idx, parent_tokens, 0)
                    )
                    parent_tokens = tl.where(
                        token_offsets == sibling_idx, token_parent, parent_tokens
                    )

                acc = bias
                ancestor_idx = token_idx
                value = tl.load(
                    x_base + ancestor_idx * stride_x_token,
                    mask=active & channel_mask,
                    other=0.0,
                ).to(conv_state_ptr.dtype.element_ty)

                for tap in tl.static_range(0, KERNEL_WIDTH):
                    if KERNEL_WIDTH == 2:
                        tap_weight = weight1 if tap == 0 else weight0
                    elif KERNEL_WIDTH == 3:
                        if tap == 0:
                            tap_weight = weight2
                        elif tap == 1:
                            tap_weight = weight1
                        else:
                            tap_weight = weight0
                    else:
                        if tap == 0:
                            tap_weight = weight3
                        elif tap == 1:
                            tap_weight = weight2
                        elif tap == 2:
                            tap_weight = weight1
                        else:
                            tap_weight = weight0

                    acc += value.to(tl.float32) * tap_weight

                    if tap < KERNEL_WIDTH - 1:
                        tl.store(
                            intermediate_base
                            + token_idx * stride_inter_step
                            + (KERNEL_WIDTH - tap - 2) * stride_inter_window,
                            value,
                            mask=active & channel_mask,
                        )

                        if ancestor_idx > 0:
                            ancestor_idx = tl.sum(
                                tl.where(
                                    token_offsets == ancestor_idx, parent_tokens, 0
                                )
                            )
                            value = tl.load(
                                x_base + ancestor_idx * stride_x_token,
                                mask=active & channel_mask,
                                other=0.0,
                            ).to(conv_state_ptr.dtype.element_ty)
                        else:
                            if KERNEL_WIDTH == 2:
                                value = history0
                            elif KERNEL_WIDTH == 3:
                                value = history1 if ancestor_idx == 0 else history0
                            else:
                                if ancestor_idx == 0:
                                    value = history2
                                elif ancestor_idx == -1:
                                    value = history1
                                else:
                                    value = history0
                            ancestor_idx = ancestor_idx - 1

                if SILU_ACTIVATION:
                    acc = acc / (1.0 + tl.exp(-acc))
                tl.store(
                    out_base + token_idx * stride_out_token,
                    acc,
                    mask=active & channel_mask,
                )

            tl.store(
                retrieve_parent_token_ptr
                + batch_idx * stride_parent_batch
                + token_offsets * stride_parent_token,
                parent_tokens,
                mask=active & token_mask & (pid_channel == 0),
            )
        else:
            for token_idx in tl.static_range(0, seqlen):
                current = tl.load(
                    x_base + token_idx * stride_x_token,
                    mask=active & channel_mask,
                    other=0.0,
                ).to(conv_state_ptr.dtype.element_ty)
                acc = bias
                if KERNEL_WIDTH == 2:
                    acc += history0.to(tl.float32) * weight0
                    acc += current.to(tl.float32) * weight1
                    history0 = current
                elif KERNEL_WIDTH == 3:
                    acc += history0.to(tl.float32) * weight0
                    acc += history1.to(tl.float32) * weight1
                    acc += current.to(tl.float32) * weight2
                    history0 = history1
                    history1 = current
                else:
                    acc += history0.to(tl.float32) * weight0
                    acc += history1.to(tl.float32) * weight1
                    acc += history2.to(tl.float32) * weight2
                    acc += current.to(tl.float32) * weight3
                    history0 = history1
                    history1 = history2
                    history2 = current

                step_base = intermediate_base + token_idx * stride_inter_step
                if KERNEL_WIDTH >= 2:
                    tl.store(
                        step_base,
                        history0,
                        mask=active & channel_mask,
                    )
                if KERNEL_WIDTH >= 3:
                    tl.store(
                        step_base + stride_inter_window,
                        history1,
                        mask=active & channel_mask,
                    )
                if KERNEL_WIDTH >= 4:
                    tl.store(
                        step_base + 2 * stride_inter_window,
                        history2,
                        mask=active & channel_mask,
                    )

                if SILU_ACTIVATION:
                    acc = acc / (1.0 + tl.exp(-acc))
                tl.store(
                    out_base + token_idx * stride_out_token,
                    acc,
                    mask=active & channel_mask,
                )


def _batch_tile_size(batch: int, dim: int, block_n: int) -> int:
    """Keep roughly two waves of work on a 40-vector-core Ascend device."""
    programs = batch * triton.cdiv(dim, block_n)
    raw_tile = max(1, triton.cdiv(programs, 80))
    if raw_tile <= 1:
        return 1
    if raw_tile <= 2:
        return 2
    if raw_tile <= 4:
        return 4
    return 8


def causal_conv1d_target_verify_npu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data: bool = False,
) -> torch.Tensor:
    """Ascend target-verify causal conv with per-token state checkpoints.

    This helper is dispatched only when the update operator receives speculative
    state indices. It writes every candidate's post-convolution window to scratch;
    the accepted window is committed later, so committed conv state stays read-only.
    Weight may use the compatibility layout ``[dim, width]`` or the optimized
    channel-contiguous layout ``[width, dim]``.
    """
    if conv_state_indices is None or intermediate_state_indices is None:
        raise ValueError(
            "target verify requires conv_state_indices and intermediate_state_indices"
        )
    if intermediate_conv_window is None:
        raise ValueError("target verify requires intermediate_conv_window")

    if isinstance(activation, bool):
        activation = "silu" if activation else None
    elif activation not in (None, "silu", "swish"):
        raise ValueError("activation must be None, silu, or swish")

    if x.dim() not in (2, 3):
        raise ValueError(f"x must be rank 2 or 3, got shape {tuple(x.shape)}")
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    batch, dim, seqlen = x.shape
    if batch == 0 or seqlen == 0:
        raise ValueError("target verify requires non-empty batch and token dimensions")
    if weight.dim() != 2:
        raise ValueError("weight must have shape [dim, width] or [width, dim]")
    if weight.shape[0] == dim:
        weight_channel_last = False
        width = weight.shape[1]
        stride_weight_dim, stride_weight_width = weight.stride()
    elif weight.shape[1] == dim:
        weight_channel_last = True
        width = weight.shape[0]
        stride_weight_width, stride_weight_dim = weight.stride()
    else:
        raise ValueError("weight must have shape [dim, width] or [width, dim]")
    if width < 2 or width > 4:
        raise ValueError(f"target-verify NPU kernel supports width 2..4, got {width}")
    if conv_state.dim() != 3 or (
        conv_state.shape[1] != dim or conv_state.shape[2] < width - 1
    ):
        raise ValueError(
            "conv_state must have shape [cache_lines, dim, state_len >= width - 1]"
        )
    if bias is not None and (bias.dim() != 1 or bias.numel() != dim):
        raise ValueError("bias must have shape [dim]")
    if conv_state_indices.dim() != 1 or intermediate_state_indices.dim() != 1:
        raise ValueError("state index tensors must be one-dimensional")
    if conv_state_indices.numel() < batch or intermediate_state_indices.numel() < batch:
        raise ValueError(
            "state index tensors must contain at least one entry per batch"
        )
    if intermediate_conv_window.dim() != 4 or (
        intermediate_conv_window.shape[1] < seqlen
        or intermediate_conv_window.shape[2] != dim
        or intermediate_conv_window.shape[3] < width - 1
    ):
        raise ValueError(
            "intermediate_conv_window must have shape "
            "[scratch_size, steps >= seqlen, dim, window >= width - 1]"
        )

    has_tree = retrieve_next_token is not None
    if has_tree:
        if retrieve_next_sibling is None or retrieve_parent_token is None:
            raise ValueError(
                "tree verify requires next-token, next-sibling, and parent tensors"
            )
        tree_tensors = (
            retrieve_next_token,
            retrieve_next_sibling,
            retrieve_parent_token,
        )
        if any(
            tensor.dim() != 2
            or tensor.shape[0] < batch
            or tensor.shape[1] < seqlen
            for tensor in tree_tensors
        ):
            raise ValueError(
                "tree mapping tensors must have shape "
                "[rows >= batch, steps >= seqlen]"
            )
    elif retrieve_next_sibling is not None or retrieve_parent_token is not None:
        raise ValueError(
            "tree mapping tensors must either all be provided or all be None"
        )

    if validate_data:
        if x.stride(1) != 1:
            raise ValueError("x must be contiguous along the channel dimension")
        if conv_state.stride(1) != 1:
            raise ValueError(
                "conv_state must be contiguous along the channel dimension"
            )
        if weight_channel_last:
            if stride_weight_dim != 1:
                raise ValueError(
                    "[width, dim] weight must be contiguous along the channel "
                    "dimension"
                )
        elif stride_weight_width != 1:
            raise ValueError(
                "[dim, width] weight must be contiguous along the width dimension"
            )

    # A linear chain consumes x in token order and keeps its rolling history in
    # registers, so each input token can be overwritten after it is loaded. Tree
    # verify must retain all original tokens for later ancestor reads.
    out = torch.empty_like(x) if has_tree else x
    # Match causal_conv1d_update_v2's channel tiling for the common large-KDA
    # shape. Tree traversal has higher register pressure and keeps the smaller
    # tile used by the generic path.
    block_n = 512 if not has_tree and dim >= 512 else 256
    batch_tile = _batch_tile_size(batch, dim, block_n)
    np2_seqlen = triton.next_power_of_2(seqlen)

    stride_x_batch, stride_x_dim, stride_x_token = x.stride()
    stride_state_batch, stride_state_dim, stride_state_token = conv_state.stride()
    stride_inter_batch, stride_inter_step, stride_inter_dim, stride_inter_window = (
        intermediate_conv_window.stride()
    )
    stride_out_batch, stride_out_dim, stride_out_token = out.stride()

    if has_tree:
        stride_next_batch, stride_next_token = retrieve_next_token.stride()
        stride_sibling_batch, stride_sibling_token = retrieve_next_sibling.stride()
        stride_parent_batch, stride_parent_token = retrieve_parent_token.stride()
        next_ptr = retrieve_next_token
        sibling_ptr = retrieve_next_sibling
        parent_ptr = retrieve_parent_token
    else:
        stride_next_batch = stride_next_token = 0
        stride_sibling_batch = stride_sibling_token = 0
        stride_parent_batch = stride_parent_token = 0
        next_ptr = x
        sibling_ptr = x
        parent_ptr = x

    grid = (triton.cdiv(batch, batch_tile), triton.cdiv(dim, block_n))
    _causal_conv1d_target_verify_npu_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        next_ptr,
        sibling_ptr,
        parent_ptr,
        out,
        batch,
        dim,
        seqlen,
        stride_x_batch,
        stride_x_dim,
        stride_x_token,
        stride_weight_dim,
        stride_weight_width,
        stride_state_batch,
        stride_state_dim,
        stride_state_token,
        conv_state_indices.stride(0),
        stride_inter_batch,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_window,
        intermediate_state_indices.stride(0),
        stride_next_batch,
        stride_next_token,
        stride_sibling_batch,
        stride_sibling_token,
        stride_parent_batch,
        stride_parent_token,
        stride_out_batch,
        stride_out_dim,
        stride_out_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        HAS_TREE=has_tree,
        NP2_SEQLEN=np2_seqlen,
        BLOCK_N=block_n,
        B_TILE=batch_tile,
    )

    if unsqueeze:
        out = out.squeeze(-1)
    return out
