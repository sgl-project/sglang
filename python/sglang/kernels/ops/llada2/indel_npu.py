"""LLaDA2 insertion/deletion kernels for Ascend NPU."""

from functools import lru_cache

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _fallback_and_scrub_argmax_kernel(
    logits_ptr,
    fallback_ptr,
    scrub_ptr,
    num_rows,
    vocab_size,
    row_stride,
    mask_token_id: tl.constexpr,
    delete_token_id: tl.constexpr,
    split_token_id: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row_pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row in tl.range(row_pid, num_rows, num_programs):
        base = row * row_stride
        fallback_max = tl.full((), float("-inf"), tl.float32)
        fallback_argmax = tl.zeros((), tl.int64)
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
            fallback_values = tl.where(
                offsets != mask_token_id,
                values,
                float("-inf"),
            )
            chunk_fallback_max = tl.max(fallback_values, axis=0)
            chunk_fallback_argmax = (
                tl.argmax(fallback_values, axis=0).to(tl.int64) + start
            )
            fallback_argmax = tl.where(
                chunk_fallback_max > fallback_max,
                chunk_fallback_argmax,
                fallback_argmax,
            )
            fallback_max = tl.maximum(fallback_max, chunk_fallback_max)
            values = tl.where(
                (offsets != mask_token_id)
                & (offsets != delete_token_id)
                & (offsets != split_token_id),
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
        tl.store(fallback_ptr + row, fallback_argmax)
        tl.store(scrub_ptr + row, running_argmax)


@lru_cache(maxsize=1)
def _num_vector_cores() -> int:
    device = torch.npu.current_device()
    properties = triton.runtime.driver.active.utils.get_device_properties(device)
    num_vector_cores = properties.get("num_vectorcore", -1)
    if num_vector_cores <= 0:
        raise RuntimeError("Failed to detect the number of NPU vector cores")
    return num_vector_cores


def scrub_argmax_fused(
    logits: torch.Tensor,
    mask_token_id: int,
    delete_token_id: int,
    split_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each row's best non-MASK and non-reserved tokens."""
    num_rows, vocab_size = logits.shape
    logits = logits.contiguous()
    fallback = torch.empty(num_rows, dtype=torch.int64, device=logits.device)
    scrub = torch.empty(num_rows, dtype=torch.int64, device=logits.device)
    _fallback_and_scrub_argmax_kernel[(min(_num_vector_cores(), num_rows),)](
        logits,
        fallback,
        scrub,
        num_rows,
        vocab_size,
        logits.stride(0),
        mask_token_id=mask_token_id,
        delete_token_id=delete_token_id,
        split_token_id=split_token_id,
        BLOCK_V=8192,
        multibuffer=False,
    )
    return fallback, scrub
