"""Fused Triton kernels for the sampling pipeline.

Fuses temperature scaling + softmax into a single kernel to reduce
kernel launch overhead and global memory traffic during decode.

Two kernel variants:
  - Single-pass: vocab fits in one tile (1 read + 1 write). Used when
    next_power_of_2(vocab) <= 32768.
  - Multi-pass: 2-pass online softmax with autotune (2 reads + 1 write).
    Used for large vocabs (e.g. 128K+).
"""

import torch
import triton
import triton.language as tl

_MAX_SINGLE_PASS_BLOCK = 32768

# ---------------------------------------------------------------------------
# Single-pass kernel: entire vocab fits in one BLOCK_SIZE tile.
# Data stays in registers — only 1 global memory read + 1 write.
# ---------------------------------------------------------------------------


@triton.jit
def _single_pass_temperature_softmax_kernel(
    logits_ptr,
    temperatures_ptr,
    output_ptr,
    vocab_size,
    logits_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    temp = tl.load(temperatures_ptr + row_idx)
    inv_temp = 1.0 / temp

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < vocab_size

    x = tl.load(
        logits_ptr + row_idx * logits_stride + offsets,
        mask=mask,
        other=float("-inf"),
    )
    x = (x * inv_temp).to(tl.float32)

    x_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - x_max)
    prob = exp_x / tl.sum(exp_x, axis=0)

    tl.store(output_ptr + row_idx * output_stride + offsets, prob, mask=mask)


@triton.jit
def _single_pass_temperature_softmax_inplace_kernel(
    logits_ptr,
    temperatures_ptr,
    vocab_size,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    temp = tl.load(temperatures_ptr + row_idx)
    inv_temp = 1.0 / temp

    row_start = logits_ptr + row_idx * stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < vocab_size

    x = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
    x = (x * inv_temp).to(tl.float32)

    x_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - x_max)
    prob = exp_x / tl.sum(exp_x, axis=0)

    tl.store(row_start + offsets, prob, mask=mask)


# ---------------------------------------------------------------------------
# Multi-pass kernel: vocab too large for one tile.
# 2-pass online softmax with autotune over (BLOCK_SIZE, num_warps).
# ---------------------------------------------------------------------------

_MULTI_PASS_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=16),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=4),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=16),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=32),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=32, num_stages=4),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=16),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=32),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=32, num_stages=4),
    triton.Config({"BLOCK_SIZE": 32768}, num_warps=32),
    triton.Config({"BLOCK_SIZE": 32768}, num_warps=32, num_stages=4),
]


@triton.autotune(configs=_MULTI_PASS_AUTOTUNE_CONFIGS, key=["vocab_size"])
@triton.jit
def _multi_pass_temperature_softmax_kernel(
    logits_ptr,
    temperatures_ptr,
    output_ptr,
    vocab_size,
    logits_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    temp = tl.load(temperatures_ptr + row_idx)
    inv_temp = 1.0 / temp

    logits_row = logits_ptr + row_idx * logits_stride
    output_row = output_ptr + row_idx * output_stride

    # Pass 1: online softmax — find max and accumulate sum(exp).
    running_max = tl.full([], value=float("-inf"), dtype=tl.float32)
    running_sum = tl.full([], value=0.0, dtype=tl.float32)

    for start in range(0, vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        x = tl.load(logits_row + offsets, mask=mask, other=float("-inf"))
        x = (x * inv_temp).to(tl.float32)

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, block_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(x - new_max), axis=0
        )
        running_max = new_max

    log_sum = tl.log(running_sum)

    # Pass 2: normalize and write probabilities.
    for start in range(0, vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        x = tl.load(logits_row + offsets, mask=mask, other=float("-inf"))
        x = (x * inv_temp).to(tl.float32)

        prob = tl.exp(x - running_max - log_sum)
        tl.store(output_row + offsets, prob, mask=mask)


@triton.autotune(configs=_MULTI_PASS_AUTOTUNE_CONFIGS, key=["vocab_size"])
@triton.jit
def _multi_pass_temperature_softmax_inplace_kernel(
    logits_ptr,
    temperatures_ptr,
    vocab_size,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    temp = tl.load(temperatures_ptr + row_idx)
    inv_temp = 1.0 / temp

    row_start = logits_ptr + row_idx * stride

    running_max = tl.full([], value=float("-inf"), dtype=tl.float32)
    running_sum = tl.full([], value=0.0, dtype=tl.float32)

    for start in range(0, vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        x = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
        x = (x * inv_temp).to(tl.float32)

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, block_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(x - new_max), axis=0
        )
        running_max = new_max

    log_sum = tl.log(running_sum)

    for start in range(0, vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        x = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
        x = (x * inv_temp).to(tl.float32)

        prob = tl.exp(x - running_max - log_sum)
        tl.store(row_start + offsets, prob, mask=mask)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _single_pass_num_warps(block_size: int) -> int:
    return max(4, min(32, block_size // 256))


def fused_temperature_softmax(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """Fused temperature scaling + softmax in a single Triton kernel.

    Args:
        logits: Raw logits of shape ``(batch_size, vocab_size)``.
        temperatures: Per-request temperatures of shape ``(batch_size, 1)``.

    Returns:
        Probability tensor of shape ``(batch_size, vocab_size)`` in float32.
    """
    batch_size, vocab_size = logits.shape
    if batch_size == 0:
        return torch.empty(0, vocab_size, dtype=torch.float32, device=logits.device)

    output = torch.empty(
        batch_size, vocab_size, dtype=torch.float32, device=logits.device
    )
    temperatures_flat = temperatures.view(-1)
    grid = (batch_size,)

    block_size = triton.next_power_of_2(vocab_size)
    if block_size <= _MAX_SINGLE_PASS_BLOCK:
        _single_pass_temperature_softmax_kernel[grid](
            logits,
            temperatures_flat,
            output,
            vocab_size,
            logits.stride(0),
            output.stride(0),
            BLOCK_SIZE=block_size,
            num_warps=_single_pass_num_warps(block_size),
        )
    else:
        _multi_pass_temperature_softmax_kernel[grid](
            logits,
            temperatures_flat,
            output,
            vocab_size,
            logits.stride(0),
            output.stride(0),
        )
    return output


def fused_temperature_softmax_inplace(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
) -> None:
    """In-place fused temperature scaling + softmax.

    After this call, ``logits`` contains probabilities.

    Args:
        logits: Raw logits of shape ``(batch_size, vocab_size)``. Modified in-place.
        temperatures: Per-request temperatures of shape ``(batch_size, 1)``.
    """
    batch_size, vocab_size = logits.shape
    if batch_size == 0:
        return

    temperatures_flat = temperatures.view(-1)
    grid = (batch_size,)

    block_size = triton.next_power_of_2(vocab_size)
    if block_size <= _MAX_SINGLE_PASS_BLOCK:
        _single_pass_temperature_softmax_inplace_kernel[grid](
            logits,
            temperatures_flat,
            vocab_size,
            logits.stride(0),
            BLOCK_SIZE=block_size,
            num_warps=_single_pass_num_warps(block_size),
        )
    else:
        _multi_pass_temperature_softmax_inplace_kernel[grid](
            logits,
            temperatures_flat,
            vocab_size,
            logits.stride(0),
        )
