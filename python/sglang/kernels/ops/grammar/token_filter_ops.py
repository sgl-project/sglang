# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Triton kernels for token filter operations."""

from collections import OrderedDict
from typing import List

import torch
import triton
import triton.language as tl

from sglang.srt.utils import get_device_core_count


@triton.jit
def reset_vocab_mask_kernel(
    vocab_mask_ptr,
    batch_idx: int,
    num_elements: int,
    reset_value: tl.constexpr,
):
    """Reset the vocab mask for a specific batch index to a given value.

    Parameters
    ----------
    vocab_mask_ptr : tl.tensor
        Pointer to the vocab mask tensor.

    batch_idx : int
        The batch index to reset.

    num_elements : int
        Number of int32 elements in the vocab mask for each batch.

    reset_value : int
        The value to reset the vocab mask to (typically -1 or 0).
    """
    pid = tl.program_id(0)
    num_threads = tl.num_programs(0)

    for i in tl.range(pid, num_elements, num_threads):
        offset = batch_idx * num_elements + i
        tl.store(vocab_mask_ptr + offset, reset_value)


@triton.jit
def set_token_filter_batch_kernel(
    vocab_mask_ptr,
    token_ids_ptr,
    batch_idx: int,
    num_tokens: int,
    num_elements: int,
    is_allowed: tl.constexpr,
):
    """Set or clear specific tokens in the vocab mask for a batch.

    Each token ID maps to a specific bit in the int32 bitmask array.
    The kernel sets or clears those bits using atomic operations.

    Parameters
    ----------
    vocab_mask_ptr : tl.tensor
        Pointer to the vocab mask tensor.

    token_ids_ptr : tl.tensor
        Pointer to the token IDs to set/clear.

    batch_idx : int
        The batch index to modify.

    num_tokens : int
        Number of tokens to process.

    num_elements : int
        Number of int32 elements in the vocab mask for each batch.

    is_allowed : bool
        If True, set the bit to 1 (allow token).
        If False, clear the bit to 0 (block token).
    """
    pid = tl.program_id(0)
    num_threads = tl.num_programs(0)

    for i in tl.range(pid, num_tokens, num_threads):
        token_id = tl.load(token_ids_ptr + i)
        element_idx = token_id // 32
        bit_idx = token_id % 32

        offset = batch_idx * num_elements + element_idx

        if is_allowed:
            tl.atomic_or(vocab_mask_ptr + offset, 1 << bit_idx)
        else:
            tl.atomic_and(vocab_mask_ptr + offset, ~(1 << bit_idx))


_cached_num_sms = None
_cached_token_id_tensors: OrderedDict[tuple[int, tuple[int, ...]], torch.Tensor] = (
    OrderedDict()
)
_MAX_TOKEN_ID_TENSOR_CACHE_SIZE = 32


def _compute_grid(work_items: int):
    global _cached_num_sms
    if _cached_num_sms is None:
        _cached_num_sms = get_device_core_count()
    if _cached_num_sms > 0:
        return (min(_cached_num_sms, work_items),)
    return (work_items,)


def _get_cached_token_ids_tensor(
    token_ids: List[int], device: torch.device
) -> torch.Tensor:
    key = (device.index or 0, tuple(token_ids))
    cached = _cached_token_id_tensors.get(key)
    if cached is not None:
        _cached_token_id_tensors.move_to_end(key)
        return cached

    token_ids_tensor = torch.tensor(token_ids, dtype=torch.int32, device=device)
    _cached_token_id_tensors[key] = token_ids_tensor
    if len(_cached_token_id_tensors) > _MAX_TOKEN_ID_TENSOR_CACHE_SIZE:
        _cached_token_id_tensors.popitem(last=False)
    return token_ids_tensor


def set_token_filter_triton(
    vocab_mask: torch.Tensor,
    token_ids: List[int],
    batch_idx: int,
    is_allowed: bool = True,
    reset_vocab_mask: bool = True,
):
    """Set or clear specific tokens in the vocab mask using Triton."""
    assert vocab_mask.device.type == "cuda"

    num_elements = vocab_mask.shape[1]

    if reset_vocab_mask:
        reset_value = 0 if is_allowed else -1
        reset_vocab_mask_kernel[_compute_grid(num_elements)](
            vocab_mask,
            batch_idx,
            num_elements,
            reset_value,
            num_warps=4,
        )

    if not token_ids:
        return

    num_tokens = len(token_ids)
    token_ids_tensor = _get_cached_token_ids_tensor(token_ids, vocab_mask.device)
    set_token_filter_batch_kernel[_compute_grid(num_tokens)](
        vocab_mask,
        token_ids_tensor,
        batch_idx,
        num_tokens,
        num_elements,
        is_allowed,
        num_warps=4,
    )
