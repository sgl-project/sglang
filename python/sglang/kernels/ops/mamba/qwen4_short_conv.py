from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

from sglang.kernels.numerics import round_bf16_to_fp32

from sglang.kernels.ops.pdl_small import ENABLED as _PDL_SMALL

logger = logging.getLogger(__name__)

_QWEN4_MAX_SHORT_CONV_STATE_LEN = 16

# SGLANG_PLE_FUSED_VERIFY_CONV (default off, read once): fused_qwen4_verify_conv in one launch
# (_qwen4_verify_conv_one_kernel) instead of prepare + torch depthwise conv1d + finish.
VERIFY_CONV_ONE_KERNEL_ENV = "SGLANG_PLE_FUSED_VERIFY_CONV"
VERIFY_CONV_ONE_KERNEL = os.environ.get(VERIFY_CONV_ONE_KERNEL_ENV, "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# ATen canUse32BitIndexMath on the contiguous [requests, channels, state + width] conv input:
# below it stock F.conv1d runs the native depthwise kernel (DepthwiseConv2d.cu), whose
# arithmetic the one kernel reproduces; above it stock keeps its own path.
_INT32_MAX = 2**31 - 1
_noted = set()


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
    PDL_SMALL: tl.constexpr = False,
):
    # Protect dynamic reads and reused-buffer writes before relaxing this edge.
    if PDL_SMALL:
        tl.extra.cuda.gdc_wait()
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
    if PDL_SMALL:
        tl.extra.cuda.gdc_launch_dependents()


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
    PDL_SMALL: tl.constexpr = False,
):
    # Protect dynamic reads and reused-buffer writes before relaxing this edge.
    if PDL_SMALL:
        tl.extra.cuda.gdc_wait()
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    token = offsets // CHANNELS
    channel = offsets % CHANNELS
    req = token // WIDTH
    step = token % WIDTH
    conv = tl.load(
        conv_ptr + (req * CHANNELS + channel) * WIDTH + step, mask, other=0
    ).to(tl.float32)
    activated = round_bf16_to_fp32(conv * tl.sigmoid(conv))
    residual = tl.load(residual_ptr + offsets, mask, other=0).to(tl.float32)
    valid = tl.load(valid_ptr + token, mask, other=0)
    tl.store(output_ptr + offsets, tl.where(valid, activated + residual, 0), mask)
    if PDL_SMALL:
        tl.extra.cuda.gdc_launch_dependents()


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
    if VERIFY_CONV_ONE_KERNEL:
        if requests * channels * (state_len + width) < _INT32_MAX:
            _note_once(
                "on",
                "%s: on: PLE verify conv in one kernel, first call requests=%d channels=%d "
                "width=%d state=%d intermediate=%s",
                VERIFY_CONV_ONE_KERNEL_ENV, requests, channels, width, state_len,
                None if intermediate is None else intermediate.dtype,
            )
            _launch_verify_conv_one_kernel(
                x, residual, weight, state, state_indices, valid, width, dilation,
                intermediate, output,
            )
            return output
        _note_once(
            "stock",
            "%s: kept the stock chain for a call stock F.conv1d does not run on its native "
            "depthwise kernel (requests=%d channels=%d columns=%d)",
            VERIFY_CONV_ONE_KERNEL_ENV, requests, channels, state_len + width,
        )
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
        launch_pdl=_PDL_SMALL,
        PDL_SMALL=_PDL_SMALL,
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
        launch_pdl=_PDL_SMALL,
        PDL_SMALL=_PDL_SMALL,
    )
    return output


def _note_once(key, fmt, *args):
    if key not in _noted:
        _noted.add(key)
        logger.info(fmt, *args)


@triton.jit
def _fma_rn_f32(a, b, c):
    return tl.inline_asm_elementwise(
        asm="fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _verify_conv_rows(
    x_ptr,
    residual_ptr,
    weight_ptr,
    state_ptr,
    valid_ptr,
    output_ptr,
    req,
    slot,
    part,
    state_stride0,
    state_stride1,
    state_stride2,
    weight_stride0,
    weight_stride2,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    KSIZE: tl.constexpr,
    DILATION: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Output rows req * WIDTH + step for BLOCK_C channels. Tap k of step s is column
    # s + k * DILATION of stock's conv input [state (cast to x's dtype), the request's x rows],
    # read in place; the sum is the native depthwise kernel's: FP32 from zero, one fused
    # multiply-add per tap in tap order, rounded once to x's dtype. Then stock finish's arithmetic.
    step = tl.arange(0, BLOCK_W)
    channel = part * BLOCK_C + tl.arange(0, BLOCK_C)
    step_mask = step < WIDTH
    channel_mask = channel < CHANNELS
    mask = step_mask[:, None] & channel_mask[None, :]
    acc = tl.zeros((BLOCK_W, BLOCK_C), dtype=tl.float32)
    for k in tl.static_range(KSIZE):
        col = (step + k * DILATION)[:, None]
        from_state = col < STATE_LEN
        state = tl.load(
            state_ptr
            + slot * state_stride0
            + channel[None, :] * state_stride1
            + col * state_stride2,
            mask & from_state,
            other=0,
        ).to(x_ptr.dtype.element_ty)
        xv = tl.load(
            x_ptr + (req * WIDTH + col - STATE_LEN) * CHANNELS + channel[None, :],
            mask & (col >= STATE_LEN),
            other=0,
        )
        tap = tl.where(from_state, state, xv).to(tl.float32)
        w = tl.load(
            weight_ptr + channel * weight_stride0 + k * weight_stride2,
            channel_mask,
            other=0,
        ).to(tl.float32)
        acc = _fma_rn_f32(tl.broadcast_to(w[None, :], (BLOCK_W, BLOCK_C)), tap, acc)
    conv = acc.to(x_ptr.dtype.element_ty).to(tl.float32)
    activated = round_bf16_to_fp32(conv * tl.sigmoid(conv))
    token = req * WIDTH + step
    offsets = token[:, None] * CHANNELS + channel[None, :]
    residual = tl.load(residual_ptr + offsets, mask, other=0).to(tl.float32)
    valid = tl.load(valid_ptr + token, step_mask, other=0)
    tl.store(
        output_ptr + offsets,
        tl.where(valid[:, None], activated + residual, 0),
        mask,
    )


@triton.jit
def _verify_conv_windows_contiguous(
    x_ptr,
    state_ptr,
    valid_ptr,
    intermediate_ptr,
    req,
    slot,
    part,
    offsets,
    state_stride0,
    cache_stride0,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # offsets = step * C * S + channel * S + state_col; the state column read is
    # col = step + 1 + state_col, at channel * S + col = offsets - step * C * S + step + 1.
    mask = offsets < WIDTH * CHANNELS * STATE_LEN
    if (CHANNELS * STATE_LEN) % BLOCK == 0:
        # A block lies in one step: one valid flag, and an invalid step only writes zeros.
        step = (part * BLOCK) // (CHANNELS * STATE_LEN)
        within = offsets - step * (CHANNELS * STATE_LEN)
        valid = tl.load(valid_ptr + req * WIDTH + step)
        if valid:
            channel = within // STATE_LEN
            col = step + 1 + within - channel * STATE_LEN
            state = tl.load(
                state_ptr + slot * state_stride0 + within + step + 1,
                mask & (col < STATE_LEN),
                other=0,
            ).to(x_ptr.dtype.element_ty)
            x = tl.load(
                x_ptr + (req * WIDTH + col - STATE_LEN) * CHANNELS + channel,
                mask & (col >= STATE_LEN),
                other=0,
            )
            value = tl.where(col < STATE_LEN, state, x)
        else:
            value = tl.zeros([BLOCK], dtype=x_ptr.dtype.element_ty)
        tl.store(intermediate_ptr + req * cache_stride0 + offsets, value, mask)
    else:
        step = offsets // (CHANNELS * STATE_LEN)
        within = offsets - step * (CHANNELS * STATE_LEN)
        channel = within // STATE_LEN
        state_col = within - channel * STATE_LEN
        col = step + 1 + state_col
        valid = tl.load(valid_ptr + req * WIDTH + step, mask, other=0)
        state = tl.load(
            state_ptr + slot * state_stride0 + within + step + 1,
            mask & (col < STATE_LEN),
            other=0,
        ).to(x_ptr.dtype.element_ty)
        x = tl.load(
            x_ptr + (req * WIDTH + col - STATE_LEN) * CHANNELS + channel,
            mask & (col >= STATE_LEN),
            other=0,
        )
        value = tl.where(valid, tl.where(col < STATE_LEN, state, x), 0)
        tl.store(intermediate_ptr + req * cache_stride0 + offsets, value, mask)


@triton.jit
def _verify_conv_windows(
    x_ptr,
    state_ptr,
    valid_ptr,
    intermediate_ptr,
    req,
    slot,
    part,
    state_stride0,
    state_stride1,
    state_stride2,
    cache_stride0,
    cache_stride1,
    cache_stride2,
    cache_stride3,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
):
    # The intermediate windows, as _qwen4_verify_conv_prepare_kernel writes them.
    offsets = part * BLOCK + tl.arange(0, BLOCK)
    if CONTIGUOUS:
        # Contiguous [C, S] state rows and [W, C, S] windows: the same entries and values,
        # addressed as one span per request so the stores vectorize.
        _verify_conv_windows_contiguous(
            x_ptr, state_ptr, valid_ptr, intermediate_ptr, req, slot, part, offsets,
            state_stride0, cache_stride0, CHANNELS, WIDTH, STATE_LEN, BLOCK,
        )
    else:
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
def _qwen4_verify_conv_one_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    state_ptr,
    indices_ptr,
    valid_ptr,
    output_ptr,
    intermediate_ptr,
    state_stride0,
    state_stride1,
    state_stride2,
    index_stride,
    weight_stride0,
    weight_stride2,
    cache_stride0,
    cache_stride1,
    cache_stride2,
    cache_stride3,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    KSIZE: tl.constexpr,
    DILATION: tl.constexpr,
    CONV_BLOCKS: tl.constexpr,
    HAS_INTERMEDIATE: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    PDL_SMALL: tl.constexpr = False,
):
    # Stock prepare + F.conv1d + finish of fused_qwen4_verify_conv in one launch: programs
    # [0, CONV_BLOCKS) of a request write output rows, the rest its intermediate windows.
    if PDL_SMALL:
        tl.extra.cuda.gdc_wait()
    req = tl.program_id(0)
    part = tl.program_id(1)
    slot = tl.load(indices_ptr + req * index_stride)
    if part < CONV_BLOCKS:
        _verify_conv_rows(
            x_ptr, residual_ptr, weight_ptr, state_ptr, valid_ptr, output_ptr,
            req, slot, part,
            state_stride0, state_stride1, state_stride2, weight_stride0, weight_stride2,
            CHANNELS, WIDTH, STATE_LEN, KSIZE, DILATION, BLOCK_W, BLOCK_C,
        )
    elif HAS_INTERMEDIATE:
        _verify_conv_windows(
            x_ptr, state_ptr, valid_ptr, intermediate_ptr,
            req, slot, part - CONV_BLOCKS,
            state_stride0, state_stride1, state_stride2,
            cache_stride0, cache_stride1, cache_stride2, cache_stride3,
            CHANNELS, WIDTH, STATE_LEN, BLOCK, CONTIGUOUS,
        )
    if PDL_SMALL:
        tl.extra.cuda.gdc_launch_dependents()


# Launch shape of the one kernel: BLOCK_C channels x the verify width per conv program,
# BLOCK flattened (step, channel, state column) entries per intermediate program.
# Picked by the H577 unit sweep on VR200 at 48 requests (all shapes bit-identical).
_ONE_KERNEL_BLOCK_C = 256
_ONE_KERNEL_BLOCK = 2048
_ONE_KERNEL_WARPS = 4


def _launch_verify_conv_one_kernel(
    x, residual, weight, state, state_indices, valid, width, dilation, intermediate, output
):
    requests = state_indices.numel()
    channels, state_len = state.shape[1:]
    has_intermediate = intermediate is not None and state_len > 0
    contiguous = has_intermediate and (
        state.stride(1) == state_len
        and state.stride(2) == 1
        and intermediate.stride(1) == channels * state_len
        and intermediate.stride(2) == state_len
        and intermediate.stride(3) == 1
    )
    conv_blocks = triton.cdiv(channels, _ONE_KERNEL_BLOCK_C)
    intermediate_blocks = (
        triton.cdiv(width * channels * state_len, _ONE_KERNEL_BLOCK)
        if has_intermediate
        else 0
    )
    _qwen4_verify_conv_one_kernel[(requests, conv_blocks + intermediate_blocks)](
        x,
        residual,
        weight,
        state,
        state_indices,
        valid,
        output,
        intermediate if has_intermediate else x,
        *state.stride(),
        state_indices.stride(0),
        weight.stride(0),
        weight.stride(2),
        *(intermediate.stride() if has_intermediate else (0, 0, 0, 0)),
        CHANNELS=channels,
        WIDTH=width,
        STATE_LEN=state_len,
        KSIZE=weight.shape[2],
        DILATION=dilation,
        CONV_BLOCKS=conv_blocks,
        HAS_INTERMEDIATE=has_intermediate,
        BLOCK_W=max(2, triton.next_power_of_2(width)),
        BLOCK_C=_ONE_KERNEL_BLOCK_C,
        BLOCK=_ONE_KERNEL_BLOCK,
        CONTIGUOUS=contiguous,
        num_warps=_ONE_KERNEL_WARPS,
        enable_fp_fusion=False,
        launch_pdl=_PDL_SMALL,
        PDL_SMALL=_PDL_SMALL,
    )
    return output
