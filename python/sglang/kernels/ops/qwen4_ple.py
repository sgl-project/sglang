"""Fused kernels for Qwen4 PLE decode and target verification."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_QWEN4_NGRAM_SIZE = 3
_QWEN4_HEADS_PER_NGRAM = 8
_QWEN4_NGRAM_HEADS = 16
_QWEN4_HC_COUNT = 4
_QWEN4_HIDDEN_SIZE = 2560
_QWEN4_MAX_SHORT_CONV_STATE_LEN = 16


@triton.jit
def _round_bf16_to_fp32(value):
    """RNE-round fp32 to BF16 precision while retaining an fp32 register."""

    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _qwen4_ngram_hash_kernel(
    contexts_ptr,
    multipliers_ptr,
    vocab_sizes_ptr,
    offsets_ptr,
    output_ptr,
    num_outputs,
    eos_token_id,
    NGRAM_SIZE: tl.constexpr,
    HEADS_PER_NGRAM: tl.constexpr,
    NGRAM_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < num_outputs
    token_idx = output_idx // NGRAM_HEADS
    head_idx = output_idx % NGRAM_HEADS
    context_base = token_idx * NGRAM_SIZE

    token_0 = tl.load(contexts_ptr + context_base, mask=mask, other=0)
    token_1 = tl.load(contexts_ptr + context_base + 1, mask=mask, other=0)
    token_2 = tl.load(contexts_ptr + context_base + 2, mask=mask, other=0)
    multiplier_0 = tl.load(multipliers_ptr)
    multiplier_1 = tl.load(multipliers_ptr + 1)
    multiplier_2 = tl.load(multipliers_ptr + 2)

    # Only the final position of each three-token context is materialized.
    previous_1 = tl.where(token_1 == eos_token_id, eos_token_id, token_1)
    previous_2 = tl.where(
        (token_0 == eos_token_id) | (token_1 == eos_token_id),
        eos_token_id,
        token_0,
    )
    mixed = (token_2 * multiplier_0) ^ (previous_1 * multiplier_1)
    mixed_3 = mixed ^ (previous_2 * multiplier_2)
    mixed = tl.where(head_idx < HEADS_PER_NGRAM, mixed, mixed_3)

    vocab_size = tl.load(vocab_sizes_ptr + head_idx, mask=mask, other=1)
    offset = tl.load(offsets_ptr + head_idx, mask=mask, other=0)
    tl.store(output_ptr + output_idx, mixed % vocab_size + offset, mask=mask)


def can_fuse_qwen4_ngram_hash(
    contexts: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> bool:
    """Return whether inputs match the fixed Qwen4 PLE hash contract."""

    return (
        contexts.is_cuda
        and contexts.dtype == torch.long
        and contexts.dim() == 2
        and contexts.shape[1] == _QWEN4_NGRAM_SIZE
        and contexts.is_contiguous()
        and multipliers.is_cuda
        and multipliers.dtype == torch.long
        and multipliers.numel() == _QWEN4_NGRAM_SIZE
        and vocab_sizes.is_cuda
        and vocab_sizes.dtype == torch.long
        and vocab_sizes.numel() == _QWEN4_NGRAM_HEADS
        and offsets.is_cuda
        and offsets.dtype == torch.long
        and offsets.numel() == _QWEN4_NGRAM_HEADS
    )


def fused_qwen4_ngram_hash(
    contexts: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """Return the 16 Qwen4 PLE N-gram IDs in one kernel launch."""

    if not can_fuse_qwen4_ngram_hash(contexts, multipliers, vocab_sizes, offsets):
        raise ValueError("unsupported input for fused Qwen4 PLE N-gram hash")
    output = torch.empty(
        (contexts.shape[0], _QWEN4_NGRAM_HEADS),
        dtype=torch.long,
        device=contexts.device,
    )
    num_outputs = output.numel()
    if num_outputs:
        block_size = 256
        _qwen4_ngram_hash_kernel[(triton.cdiv(num_outputs, block_size),)](
            contexts,
            multipliers,
            vocab_sizes,
            offsets,
            output,
            num_outputs,
            eos_token_id,
            NGRAM_SIZE=_QWEN4_NGRAM_SIZE,
            HEADS_PER_NGRAM=_QWEN4_HEADS_PER_NGRAM,
            NGRAM_HEADS=_QWEN4_NGRAM_HEADS,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
    return output


@triton.jit
def _qwen4_gate_value_kernel(
    gate_ptr,
    value_ptr,
    output_ptr,
    num_tokens,
    HC_COUNT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_group = tl.program_id(0)
    token = token_group // HC_COUNT
    hidden = tl.arange(0, BLOCK_SIZE)
    mask = (token < num_tokens) & (hidden < HIDDEN_SIZE)

    # `gate` arrives already rounded to bf16 by the eager multiply/reduce/divide;
    # every remaining eager bf16 rounding boundary is reproduced below.
    gate = tl.load(gate_ptr + token_group).to(tl.float32)
    magnitude = tl.maximum(tl.abs(gate), 1.0e-6)
    root = _round_bf16_to_fp32(tl.sqrt(magnitude))
    sign = tl.where(gate > 0.0, 1.0, tl.where(gate < 0.0, -1.0, 0.0))
    transformed = _round_bf16_to_fp32(root * sign)
    activated = _round_bf16_to_fp32(tl.sigmoid(transformed))

    value = tl.load(value_ptr + token * HIDDEN_SIZE + hidden, mask=mask, other=0.0).to(
        tl.float32
    )
    output_offset = token_group * HIDDEN_SIZE + hidden
    tl.store(output_ptr + output_offset, activated * value, mask=mask)


def can_fuse_qwen4_gate_value(gate: torch.Tensor, value: torch.Tensor) -> bool:
    """Return whether inputs match Qwen4's fixed BF16 gate/value contract."""

    return (
        gate.is_cuda
        and gate.dtype == torch.bfloat16
        and gate.dim() == 3
        and gate.shape[1:] == (_QWEN4_HC_COUNT, 1)
        and gate.is_contiguous()
        and value.is_cuda
        and value.dtype == gate.dtype
        and value.shape == (gate.shape[0], _QWEN4_HIDDEN_SIZE)
        and value.is_contiguous()
    )


def fused_qwen4_gate_value(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply Qwen4's post-reduction gate and value broadcast in one kernel."""

    if not can_fuse_qwen4_gate_value(gate, value):
        raise ValueError("unsupported input for fused Qwen4 PLE gate/value")
    output = torch.empty(
        (gate.shape[0], _QWEN4_HC_COUNT, _QWEN4_HIDDEN_SIZE),
        dtype=value.dtype,
        device=value.device,
    )
    if gate.shape[0]:
        _qwen4_gate_value_kernel[(gate.shape[0] * _QWEN4_HC_COUNT,)](
            gate,
            value,
            output,
            gate.shape[0],
            HC_COUNT=_QWEN4_HC_COUNT,
            HIDDEN_SIZE=_QWEN4_HIDDEN_SIZE,
            BLOCK_SIZE=4096,
            num_warps=8,
        )
    return output


@triton.jit
def _qwen4_short_conv_state_kernel(
    state_ptr,
    state_indices_ptr,
    x_ptr,
    conv_input_ptr,
    num_tokens,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_STATE_LEN: tl.constexpr,
):
    token = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)[:, None]
    state_col = tl.arange(0, BLOCK_STATE_LEN)[None, :]
    channel_mask = (token < num_tokens) & (channel < CHANNELS)
    state_mask = channel_mask & (state_col < STATE_LEN)
    state_index = tl.load(state_indices_ptr + token, mask=token < num_tokens, other=0)
    state_base = state_index * CHANNELS * STATE_LEN
    state_offset = state_base + channel * STATE_LEN + state_col
    output_base = token * CHANNELS * (STATE_LEN + 1)
    output_offset = output_base + channel * (STATE_LEN + 1) + state_col

    # Materialize every old state value before advancing the in-place cache,
    # so the convolution input equals the native index_select + cat result.
    old_state = tl.load(state_ptr + state_offset, mask=state_mask, other=0.0)
    tl.store(conv_input_ptr + output_offset, old_state, mask=state_mask)
    x = tl.load(x_ptr + token * CHANNELS + channel, mask=channel_mask, other=0.0)
    tl.store(
        conv_input_ptr + output_base + channel * (STATE_LEN + 1) + STATE_LEN,
        x,
        mask=channel_mask,
    )
    tl.debug_barrier()

    # Slot 0 is the CUDA-graph padding slot and may appear in several rows;
    # its post-step value is unobservable, so skip it to avoid duplicate writers.
    update_mask = state_mask & (state_col < STATE_LEN - 1) & (state_index != 0)
    next_value = tl.load(
        conv_input_ptr + output_offset + 1, mask=update_mask, other=0.0
    )
    tl.store(state_ptr + state_offset, next_value, mask=update_mask)
    tl.store(
        state_ptr + state_base + channel * STATE_LEN + STATE_LEN - 1,
        x,
        mask=channel_mask & (state_index != 0),
    )


def can_fuse_qwen4_short_conv_state(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    x: torch.Tensor,
) -> bool:
    """Return whether decode state movement can use the exact fused kernel."""

    return (
        state.is_cuda
        and state.dtype in (torch.bfloat16, torch.float16)
        and state.dim() == 3
        and state.is_contiguous()
        and 0 < state.shape[2] <= _QWEN4_MAX_SHORT_CONV_STATE_LEN
        and state_indices.is_cuda
        and state_indices.dtype == torch.long
        and state_indices.dim() == 1
        and state_indices.is_contiguous()
        and x.is_cuda
        and x.dtype == state.dtype
        and x.dim() == 2
        and x.is_contiguous()
        and x.shape == (state_indices.shape[0], state.shape[1])
    )


def fused_qwen4_short_conv_state(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Build ``[selected state, x]`` and advance real decode slots in one launch."""

    if not can_fuse_qwen4_short_conv_state(state, state_indices, x):
        raise ValueError("unsupported input for fused Qwen4 short-conv state")
    state_len = state.shape[2]
    conv_input = torch.empty(
        (x.shape[0], x.shape[1], state_len + 1),
        dtype=x.dtype,
        device=x.device,
    )
    if x.shape[0]:
        block_channels = 128
        block_state_len = triton.next_power_of_2(state_len)
        _qwen4_short_conv_state_kernel[
            (x.shape[0], triton.cdiv(x.shape[1], block_channels))
        ](
            state,
            state_indices,
            x,
            conv_input,
            x.shape[0],
            CHANNELS=state.shape[1],
            STATE_LEN=state_len,
            BLOCK_CHANNELS=block_channels,
            BLOCK_STATE_LEN=block_state_len,
            num_warps=8,
        )
    return conv_input


@triton.jit
def _qwen4_gate_reduce_kernel(
    key_ptr,
    query_ptr,
    value_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    hidden = tl.arange(0, BLOCK_SIZE)
    mask = hidden < HIDDEN_SIZE
    key = tl.load(key_ptr + group * HIDDEN_SIZE + hidden, mask, other=0).to(tl.float32)
    query = tl.load(query_ptr + group * HIDDEN_SIZE + hidden, mask, other=0).to(
        tl.float32
    )
    product = _round_bf16_to_fp32(key * query)
    gate = _round_bf16_to_fp32(tl.sum(product, 0))
    gate = _round_bf16_to_fp32(gate * (HIDDEN_SIZE**-0.5))
    magnitude = _round_bf16_to_fp32(tl.maximum(tl.abs(gate), 1.0e-6))
    root = _round_bf16_to_fp32(tl.sqrt(magnitude))
    sign = tl.where(gate > 0, 1.0, tl.where(gate < 0, -1.0, 0.0))
    activated = _round_bf16_to_fp32(tl.sigmoid(_round_bf16_to_fp32(root * sign)))
    value = tl.load(
        value_ptr + (group // HC_COUNT) * HIDDEN_SIZE + hidden, mask, other=0
    ).to(tl.float32)
    tl.store(output_ptr + group * HIDDEN_SIZE + hidden, activated * value, mask)


def can_fuse_qwen4_gate_reduce(key, query, value):
    return (
        key.is_cuda
        and key.dtype == torch.bfloat16
        and key.ndim == 3
        and key.shape[1:] == (_QWEN4_HC_COUNT, _QWEN4_HIDDEN_SIZE)
        and key.is_contiguous()
        and query.shape == key.shape
        and query.dtype == key.dtype
        and query.device == key.device
        and query.is_contiguous()
        and value.shape == (key.shape[0], _QWEN4_HIDDEN_SIZE)
        and value.dtype == key.dtype
        and value.device == key.device
        and value.is_contiguous()
    )


def fused_qwen4_gate_reduce(key, query, value):
    if not can_fuse_qwen4_gate_reduce(key, query, value):
        raise ValueError("unsupported input for Qwen4 PLE gate reduction")
    output = torch.empty_like(key)
    if key.shape[0]:
        _qwen4_gate_reduce_kernel[(key.shape[0] * _QWEN4_HC_COUNT,)](
            key,
            query,
            value,
            output,
            HIDDEN_SIZE=_QWEN4_HIDDEN_SIZE,
            HC_COUNT=_QWEN4_HC_COUNT,
            BLOCK_SIZE=triton.next_power_of_2(_QWEN4_HIDDEN_SIZE),
            enable_fp_fusion=False,
        )
    return output


@triton.jit
def _qwen4_verify_conv_prepare_kernel(
    x_ptr,
    state_ptr,
    indices_ptr,
    valid_ptr,
    conv_ptr,
    intermediate_ptr,
    state_stride0,
    state_stride1,
    state_stride2,
    index_stride,
    cache_stride0,
    cache_stride1,
    cache_stride2,
    cache_stride3,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    HAS_INTERMEDIATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(indices_ptr + req * index_stride)
    channel = offsets // (STATE_LEN + WIDTH)
    col = offsets % (STATE_LEN + WIDTH)
    mask = channel < CHANNELS
    state = tl.load(
        state_ptr
        + slot * state_stride0
        + channel * state_stride1
        + col * state_stride2,
        mask & (col < STATE_LEN),
        other=0,
    ).to(x_ptr.dtype.element_ty)
    x = tl.load(
        x_ptr + (req * WIDTH + col - STATE_LEN) * CHANNELS + channel,
        mask & (col >= STATE_LEN),
        other=0,
    )
    tl.store(
        conv_ptr + req * CHANNELS * (STATE_LEN + WIDTH) + offsets,
        tl.where(col < STATE_LEN, state, x),
        mask,
    )
    if HAS_INTERMEDIATE and STATE_LEN > 0:
        step = offsets // (CHANNELS * STATE_LEN)
        channel = (offsets // STATE_LEN) % CHANNELS
        state_col = offsets % STATE_LEN
        col = step + 1 + state_col
        mask = step < WIDTH
        valid = tl.load(valid_ptr + req * WIDTH + step, mask, other=0)
        state = tl.load(
            state_ptr
            + slot * state_stride0
            + channel * state_stride1
            + col * state_stride2,
            mask & (col < STATE_LEN),
            other=0,
        ).to(x_ptr.dtype.element_ty)
        x = tl.load(
            x_ptr + (req * WIDTH + col - STATE_LEN) * CHANNELS + channel,
            mask & (col >= STATE_LEN),
            other=0,
        )
        value = tl.where(valid, tl.where(col < STATE_LEN, state, x), 0)
        tl.store(
            intermediate_ptr
            + req * cache_stride0
            + step * cache_stride1
            + channel * cache_stride2
            + state_col * cache_stride3,
            value,
            mask,
        )


@triton.jit
def _qwen4_verify_conv_finish_kernel(
    conv_ptr,
    residual_ptr,
    valid_ptr,
    output_ptr,
    num_elements,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    token = offsets // CHANNELS
    channel = offsets % CHANNELS
    req = token // WIDTH
    step = token % WIDTH
    conv = tl.load(
        conv_ptr + (req * CHANNELS + channel) * WIDTH + step, mask, other=0
    ).to(tl.float32)
    activated = _round_bf16_to_fp32(conv * tl.sigmoid(conv))
    residual = tl.load(residual_ptr + offsets, mask, other=0).to(tl.float32)
    valid = tl.load(valid_ptr + token, mask, other=0)
    tl.store(output_ptr + offsets, tl.where(valid, activated + residual, 0), mask)


def can_fuse_qwen4_verify_conv(
    x, residual, weight, state, state_indices, valid, width, dilation, intermediate
):
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and residual.shape == x.shape
        and residual.dtype == x.dtype
        and residual.device == x.device
        and residual.is_contiguous()
        and 0 < width <= 32
        and x.shape[0] == state_indices.numel() * width
        and state_indices.ndim == 1
        and state_indices.dtype in (torch.int32, torch.int64)
        and state_indices.device == x.device
        and valid.shape == (x.shape[0],)
        and valid.dtype == torch.bool
        and valid.device == x.device
        and valid.is_contiguous()
        and state.ndim == 3
        and state.shape[1] == x.shape[1]
        and state.device == x.device
        and state.dtype in (torch.bfloat16, torch.float32)
        and 0 <= state.shape[2] <= _QWEN4_MAX_SHORT_CONV_STATE_LEN
        and weight.ndim == 3
        and weight.shape[:2] == (x.shape[1], 1)
        and weight.device == x.device
        and weight.dtype == x.dtype
        and dilation > 0
        and state.shape[2] == (weight.shape[2] - 1) * dilation
        and (
            intermediate is None
            or (
                intermediate.ndim == 4
                and intermediate.shape[0] >= state_indices.numel()
                and intermediate.shape[1] >= width
                and intermediate.shape[2:] == state.shape[1:]
                and intermediate.device == x.device
                and intermediate.dtype in (torch.bfloat16, torch.float32)
            )
        )
    )


def fused_qwen4_verify_conv(
    x, residual, weight, state, state_indices, valid, width, dilation, intermediate
):
    if not can_fuse_qwen4_verify_conv(
        x, residual, weight, state, state_indices, valid, width, dilation, intermediate
    ):
        raise ValueError("unsupported input for Qwen4 PLE verify convolution")
    output = torch.empty_like(x)
    if x.shape[0] == 0:
        return output
    requests = state_indices.numel()
    channels, state_len = state.shape[1:]
    conv_input = torch.empty(
        (requests, channels, state_len + width), dtype=x.dtype, device=x.device
    )
    work = channels * max(
        state_len + width, width * state_len if intermediate is not None else 0
    )
    _qwen4_verify_conv_prepare_kernel[(requests, triton.cdiv(work, 256))](
        x,
        state,
        state_indices,
        valid,
        conv_input,
        intermediate if intermediate is not None else x,
        *state.stride(),
        state_indices.stride(0),
        *(intermediate.stride() if intermediate is not None else (0, 0, 0, 0)),
        CHANNELS=channels,
        WIDTH=width,
        STATE_LEN=state_len,
        HAS_INTERMEDIATE=intermediate is not None,
        BLOCK=256,
    )
    conv = torch.nn.functional.conv1d(
        conv_input, weight, dilation=dilation, groups=channels
    )
    _qwen4_verify_conv_finish_kernel[(triton.cdiv(x.numel(), 256),)](
        conv,
        residual,
        valid,
        output,
        x.numel(),
        CHANNELS=channels,
        WIDTH=width,
        BLOCK=256,
        enable_fp_fusion=False,
    )
    return output
