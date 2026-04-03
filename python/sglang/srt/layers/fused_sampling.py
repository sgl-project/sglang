"""Fused Triton kernels for the sampling pipeline.

Fuses temperature scaling + softmax into a single kernel to reduce
kernel launch overhead and global memory traffic during decode.

Two kernel variants:
  - Single-pass: vocab fits in one tile (1 read + 1 write). Used when
    next_power_of_2(vocab) <= 32768.
  - Multi-pass: 2-pass online softmax with autotune (2 reads + 1 write).
    Used for large vocabs (e.g. 128K+).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

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

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < vocab_size

    x = tl.load(
        logits_ptr + row_idx * logits_stride + offsets,
        mask=mask,
        other=float("-inf"),
    )
    x = (x / temp).to(tl.float32)

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

    row_start = logits_ptr + row_idx * stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < vocab_size

    x = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
    x = (x / temp).to(tl.float32)

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

    logits_row = logits_ptr + row_idx * logits_stride
    output_row = output_ptr + row_idx * output_stride

    # Pass 1: find global max (matches PyTorch's first reduction pass)
    global_max = tl.full([], value=float("-inf"), dtype=tl.float32)
    for start in range(0, vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        x = tl.load(logits_row + offsets, mask=mask, other=float("-inf"))
        x = (x / temp).to(tl.float32)
        global_max = tl.maximum(global_max, tl.max(x, axis=0))

    # Pass 2: compute sum of exp(x - max) (matches PyTorch's second pass)
    sum_exp = tl.full([], value=0.0, dtype=tl.float32)
    for start in range(0, vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        x = tl.load(logits_row + offsets, mask=mask, other=float("-inf"))
        x = (x / temp).to(tl.float32)
        sum_exp += tl.sum(tl.exp(x - global_max), axis=0)

    # Pass 3: normalize (matches PyTorch's exp(x-max)/sum)
    for start in range(0, vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        x = tl.load(logits_row + offsets, mask=mask, other=float("-inf"))
        x = (x / temp).to(tl.float32)

        prob = tl.exp(x - global_max) / sum_exp
        tl.store(output_row + offsets, prob, mask=mask)


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

    row_start = logits_ptr + row_idx * stride

    # Pass 1: find global max (matches PyTorch's first reduction pass)
    global_max = tl.full([], value=float("-inf"), dtype=tl.float32)
    for start in range(0, vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        x = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
        x = (x / temp).to(tl.float32)
        global_max = tl.maximum(global_max, tl.max(x, axis=0))

    # Pass 2: compute sum of exp(x - max) (matches PyTorch's second pass)
    sum_exp = tl.full([], value=0.0, dtype=tl.float32)
    for start in range(0, vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        x = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
        x = (x / temp).to(tl.float32)
        sum_exp += tl.sum(tl.exp(x - global_max), axis=0)

    # Pass 3: normalize (matches PyTorch's exp(x-max)/sum)
    for start in range(0, vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        x = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
        x = (x / temp).to(tl.float32)

        prob = tl.exp(x - global_max) / sum_exp
        tl.store(row_start + offsets, prob, mask=mask)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_MULTI_PASS_CONFIG = {"BLOCK_SIZE": 4096, "num_warps": 16}

# Populated by warmup from the out-of-place kernel's autotune result.
_multi_pass_inplace_config: dict | None = None


def _single_pass_num_warps(block_size: int) -> int:
    return max(4, min(32, block_size // 256))


def _get_multi_pass_inplace_config() -> dict:
    """Return the launch config for the multi-pass in-place kernel."""
    if _multi_pass_inplace_config is not None:
        return _multi_pass_inplace_config
    return _DEFAULT_MULTI_PASS_CONFIG


def _dispatch_kernel(
    logits: torch.Tensor,
    temperatures_flat: torch.Tensor,
    vocab_size: int,
    batch_size: int,
    output: torch.Tensor = None,
) -> None:
    """Dispatch to single-pass or multi-pass kernel. output=None means in-place."""
    grid = (batch_size,)
    block_size = triton.next_power_of_2(vocab_size)
    inplace = output is None

    if block_size <= _MAX_SINGLE_PASS_BLOCK:
        if inplace:
            _single_pass_temperature_softmax_inplace_kernel[grid](
                logits,
                temperatures_flat,
                vocab_size,
                logits.stride(0),
                BLOCK_SIZE=block_size,
                num_warps=_single_pass_num_warps(block_size),
            )
        else:
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
        if inplace:
            cfg = _get_multi_pass_inplace_config()
            _multi_pass_temperature_softmax_inplace_kernel[grid](
                logits,
                temperatures_flat,
                vocab_size,
                logits.stride(0),
                **cfg,
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


def fused_temperature_softmax(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """Fused temperature scaling + softmax. Returns float32 probabilities."""
    batch_size, vocab_size = logits.shape
    if batch_size == 0:
        return torch.empty(0, vocab_size, dtype=torch.float32, device=logits.device)

    if not logits.is_contiguous():
        logits = logits.contiguous()

    output = torch.empty(
        batch_size, vocab_size, dtype=torch.float32, device=logits.device
    )
    temperatures_flat = temperatures.contiguous().view(-1)
    _dispatch_kernel(logits, temperatures_flat, vocab_size, batch_size, output)
    return output


def fused_temperature_softmax_inplace(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
) -> None:
    """In-place fused temperature scaling + softmax. Overwrites logits with probabilities."""
    batch_size, vocab_size = logits.shape
    if batch_size == 0:
        return

    if not logits.is_contiguous():
        work = logits.contiguous()
        fused_temperature_softmax_inplace(work, temperatures)
        logits.copy_(work)
        return

    temperatures_flat = temperatures.contiguous().view(-1)
    _dispatch_kernel(logits, temperatures_flat, vocab_size, batch_size)


def warmup_fused_temperature_softmax(
    vocab_size: int,
    device: torch.device | int | None = None,
    logits_dtype: torch.dtype = torch.float32,
) -> None:
    """Pre-compile and autotune kernels at startup so first request has no latency spike.

    For multi-pass kernels the out-of-place variant is autotuned (safe — separate
    input/output buffers), and its winning config is reused for the in-place
    variant so that no autotune ever runs on a live logits buffer.

    ``logits_dtype`` should match ``next_token_logits`` at inference (usually
    ``model_config.dtype``) so Triton specializes the same way as in production.
    """
    global _multi_pass_inplace_config

    if device is None:
        device = torch.cuda.current_device()

    block_size = triton.next_power_of_2(vocab_size)
    is_multi_pass = block_size > _MAX_SINGLE_PASS_BLOCK
    label = "multi-pass autotune" if is_multi_pass else "single-pass JIT"
    logger.info(
        "Warming up fused_temperature_softmax (%s, vocab_size=%d, logits_dtype=%s) ...",
        label,
        vocab_size,
        logits_dtype,
    )

    dummy_logits = torch.randn(1, vocab_size, dtype=logits_dtype, device=device)
    dummy_temps = torch.ones(1, 1, dtype=torch.float32, device=device)

    # 1. Out-of-place kernel: autotune runs here (safe, separate buffers).
    fused_temperature_softmax(dummy_logits, dummy_temps)

    # 2. Propagate best config to the in-place kernel (no autotune needed).
    if is_multi_pass:
        best = getattr(_multi_pass_temperature_softmax_kernel, "best_config", None)
        if best is not None:
            _multi_pass_inplace_config = {
                "BLOCK_SIZE": best.kwargs["BLOCK_SIZE"],
                "num_warps": best.num_warps,
            }
            if best.num_stages is not None:
                _multi_pass_inplace_config["num_stages"] = best.num_stages
            ns = _multi_pass_inplace_config.get("num_stages", "default")
            logger.info(
                "Multi-pass autotune result: BLOCK_SIZE=%d, num_warps=%d, num_stages=%s",
                _multi_pass_inplace_config["BLOCK_SIZE"],
                _multi_pass_inplace_config["num_warps"],
                ns,
            )
        else:
            _multi_pass_inplace_config = None
            logger.warning(
                "Multi-pass fused softmax: autotune did not set best_config; "
                "using default launch config for in-place kernel."
            )

    # 3. In-place kernel: JIT compile only (uses the config from step 2).
    fused_temperature_softmax_inplace(dummy_logits.clone(), dummy_temps)
    torch.cuda.synchronize(device)

    logger.info("fused_temperature_softmax warmup done (vocab_size=%d).", vocab_size)
