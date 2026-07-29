"""Fused vocabulary reductions for dLLM decoding on Ascend NPU."""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _scrub_argmax_kernel(
        logits_ptr,
        scrub_ptr,
        num_rows,
        vocab_size,
        row_stride,
        delete_token_id: tl.constexpr,
        split_token_id: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row_pid = tl.program_id(0)
        num_programs = tl.num_programs(0)
        for row in tl.range(row_pid, num_rows, num_programs):
            base = row * row_stride
            running_max = tl.full((), float("-inf"), tl.float32)
            running_argmax = tl.zeros((), tl.int64)
            for start in range(0, vocab_size, BLOCK_V):
                offsets = start + tl.arange(0, BLOCK_V)
                valid = offsets < vocab_size
                values = tl.load(
                    logits_ptr + base + offsets,
                    mask=valid,
                    other=float("-inf"),
                )
                values = tl.where(
                    (offsets != delete_token_id) & (offsets != split_token_id),
                    values,
                    float("-inf"),
                )
                chunk_max = tl.max(values, axis=0)
                chunk_argmax = tl.argmax(values, axis=0).to(tl.int64) + start
                running_argmax = tl.where(
                    chunk_max > running_max,
                    chunk_argmax,
                    running_argmax,
                )
                running_max = tl.maximum(running_max, chunk_max)
            tl.store(scrub_ptr + row, running_argmax)

    @triton.jit
    def _argmax_prob_kernel(
        logits_ptr,
        argmax_ptr,
        prob_ptr,
        num_rows,
        vocab_size,
        row_stride,
        BLOCK_V: tl.constexpr,
    ):
        row_pid = tl.program_id(0)
        num_programs = tl.num_programs(0)
        for row in tl.range(row_pid, num_rows, num_programs):
            base = row * row_stride
            running_max = tl.full((), float("-inf"), tl.float32)
            running_argmax = tl.zeros((), tl.int64)
            running_sum = tl.zeros((), tl.float32)
            for start in range(0, vocab_size, BLOCK_V):
                offsets = start + tl.arange(0, BLOCK_V)
                valid = offsets < vocab_size
                values = tl.load(
                    logits_ptr + base + offsets,
                    mask=valid,
                    other=float("-inf"),
                ).to(tl.float32)
                chunk_max = tl.max(values, axis=0)
                new_max = tl.maximum(running_max, chunk_max)
                running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
                    tl.exp(values - new_max), axis=0
                )
                chunk_argmax = tl.argmax(values, axis=0).to(tl.int64) + start
                running_argmax = tl.where(
                    chunk_max > running_max,
                    chunk_argmax,
                    running_argmax,
                )
                running_max = new_max
            tl.store(argmax_ptr + row, running_argmax)
            # exp(max - logsumexp(logits))
            tl.store(prob_ptr + row, 1.0 / running_sum)


def _num_programs(num_rows: int) -> int:
    try:
        from sgl_kernel_npu.utils.triton_utils import get_device_properties

        _, num_cores = get_device_properties()
    except Exception:
        num_cores = 40
    return min(num_cores, num_rows)


def argmax_softmax_prob_fused(
    logits: torch.Tensor, block_v: int = 8192
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return argmax and its softmax probability with one vocabulary scan."""
    assert logits.dim() == 2
    num_rows, vocab_size = logits.shape
    logits = logits.contiguous()
    argmax = torch.empty(num_rows, dtype=torch.int64, device=logits.device)
    probability = torch.empty(num_rows, dtype=torch.float32, device=logits.device)
    _argmax_prob_kernel[(_num_programs(num_rows),)](
        logits,
        argmax,
        probability,
        num_rows,
        vocab_size,
        logits.stride(0),
        BLOCK_V=block_v,
    )
    return argmax, probability


def scrub_argmax_fused(
    logits: torch.Tensor,
    delete_token_id: int,
    split_token_id: int,
    block_v: int = 8192,
) -> torch.Tensor:
    """Return the best token other than DELETE and SPLIT."""
    assert logits.dim() == 2
    num_rows, vocab_size = logits.shape
    logits = logits.contiguous()
    scrub = torch.empty(num_rows, dtype=torch.int64, device=logits.device)
    _scrub_argmax_kernel[(_num_programs(num_rows),)](
        logits,
        scrub,
        num_rows,
        vocab_size,
        logits.stride(0),
        delete_token_id=delete_token_id,
        split_token_id=split_token_id,
        BLOCK_V=block_v,
        multibuffer=False,
    )
    return scrub


if not _TRITON_AVAILABLE:
    argmax_softmax_prob_fused = None
    scrub_argmax_fused = None
