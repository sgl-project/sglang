# Copyright 2023-2024 SGLang Team
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

    # Each thread processes multiple elements in a strided loop
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

    This kernel handles the case where num_elements may be different from
    num_tokens // 32.

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

    # Each thread processes multiple tokens in a strided loop
    for i in tl.range(pid, num_tokens, num_threads):
        token_id = tl.load(token_ids_ptr + i)
        element_idx = token_id // 32
        bit_idx = token_id % 32

        offset = batch_idx * num_elements + element_idx

        if is_allowed:
            # Set the bit to 1 using atomic OR operation
            tl.atomic_or(vocab_mask_ptr + offset, 1 << bit_idx)
        else:
            # Clear the bit to 0 using atomic AND operation
            tl.atomic_and(vocab_mask_ptr + offset, ~(1 << bit_idx))


def set_token_filter_triton(
    vocab_mask: torch.Tensor,
    token_ids: List[int],
    batch_idx: int,
    is_allowed: bool = True,
    reset_vocab_mask: bool = True,
):
    """Set or clear specific tokens in the vocab mask using Triton.

    This is a Triton-accelerated version of the set_token_filter function.

    Parameters
    ----------
    vocab_mask : torch.Tensor
        The vocab mask tensor of shape [batch_size, num_elements] where each
        int32 element contains 32 bits representing 32 tokens.

    token_ids : List[int]
        List of token IDs to set or clear.

    batch_idx : int
        The batch index to modify.

    is_allowed : bool, default=True
        If True, set the bits for the given token_ids to 1 (allow tokens).
        If False, clear the bits for the given token_ids to 0 (block tokens).

    reset_vocab_mask : bool, default=True
        If True, reset the entire vocab mask row before setting/clearing tokens.
        The reset value is:
        - 0 if is_allowed=True (block all tokens, then allow specified ones)
        - -1 if is_allowed=False (allow all tokens, then block specified ones)
    """

    # Ensure vocab_mask is on GPU device for Triton kernel
    assert vocab_mask.device.type == "cuda"

    if not token_ids:
        if reset_vocab_mask:
            # Just reset the entire row
            num_elements = vocab_mask.shape[1]
            NUM_SMS = get_device_core_count()
            if NUM_SMS > 0:
                grid = (min(NUM_SMS, num_elements),)
            else:
                grid = (num_elements,)

            reset_value = 0 if is_allowed else -1
            reset_vocab_mask_kernel[grid](
                vocab_mask,
                batch_idx,
                num_elements,
                reset_value,
                num_warps=4,
            )
        return

    num_tokens = len(token_ids)
    num_elements = vocab_mask.shape[1]
    token_ids_tensor = torch.tensor(
        token_ids, dtype=torch.int32, device=vocab_mask.device
    )

    if reset_vocab_mask:
        # First: reset the vocab mask
        NUM_SMS = get_device_core_count()
        if NUM_SMS > 0:
            grid = (min(NUM_SMS, num_elements),)
        else:
            grid = (num_elements,)

        reset_value = 0 if is_allowed else -1
        reset_vocab_mask_kernel[grid](
            vocab_mask,
            batch_idx,
            num_elements,
            reset_value,
            num_warps=4,
        )

        # Second: set/clear specific tokens (synchronization happens between kernel launches)
        if NUM_SMS > 0:
            grid = (min(NUM_SMS, num_tokens),)
        else:
            grid = (num_tokens,)

        set_token_filter_batch_kernel[grid](
            vocab_mask,
            token_ids_tensor,
            batch_idx,
            num_tokens,
            num_elements,
            is_allowed,
            num_warps=4,
        )
    else:
        # Only set/clear specific tokens without reset
        NUM_SMS = get_device_core_count()
        if NUM_SMS > 0:
            grid = (min(NUM_SMS, num_tokens),)
        else:
            grid = (num_tokens,)

        set_token_filter_batch_kernel[grid](
            vocab_mask,
            token_ids_tensor,
            batch_idx,
            num_tokens,
            num_elements,
            is_allowed,
            num_warps=4,
        )


def demo_test():
    """Demo test to verify the Triton implementation."""
    import torch

    # Test 1: Set specific tokens as allowed (reset to 0, then set bits)
    vocab_mask = torch.zeros(2, 4, dtype=torch.int32, device="cuda")
    token_ids = [10, 20, 30, 40, 50]

    set_token_filter_triton(
        vocab_mask, token_ids, batch_idx=0, is_allowed=True, reset_vocab_mask=True
    )

    # Verify: all elements should be 0 except where tokens are set
    print("Test 1 - Set tokens as allowed:")
    print(f"vocab_mask[0] = {vocab_mask[0].cpu().tolist()}")
    # Token 10: element_idx=0, bit_idx=10, value should have bit 10 set
    # Token 20: element_idx=0, bit_idx=20, value should have bit 20 set
    # Token 30: element_idx=0, bit_idx=30, value should have bit 30 set
    # Token 40: element_idx=1, bit_idx=8, value should have bit 8 set
    # Token 50: element_idx=1, bit_idx=18, value should have bit 18 set

    # Test 2: Block specific tokens (reset to -1, then clear bits)
    vocab_mask = torch.full((2, 4), -1, dtype=torch.int32, device="cuda")
    token_ids = [10, 20, 30, 40, 50]

    set_token_filter_triton(
        vocab_mask, token_ids, batch_idx=1, is_allowed=False, reset_vocab_mask=True
    )

    print("\nTest 2 - Block tokens:")
    print(f"vocab_mask[1] = {vocab_mask[1].cpu().tolist()}")
    # All should be -1 except bits 10, 20, 30 in element 0 and bits 8, 18 in element 1 should be cleared

    # Test 3: Set tokens without reset
    vocab_mask = torch.full((2, 4), -1, dtype=torch.int32, device="cuda")
    token_ids = [10, 20]

    set_token_filter_triton(
        vocab_mask, token_ids, batch_idx=0, is_allowed=True, reset_vocab_mask=False
    )

    print("\nTest 3 - Set tokens without reset:")
    print(f"vocab_mask[0] = {vocab_mask[0].cpu().tolist()}")
    # Should still be -1 with bits 10 and 20 set (but -1 already has all bits set)

    # Test 4: Empty token list with reset
    vocab_mask = torch.ones((2, 4), dtype=torch.int32, device="cuda")

    set_token_filter_triton(
        vocab_mask, [], batch_idx=0, is_allowed=True, reset_vocab_mask=True
    )

    print("\nTest 4 - Empty token list with reset:")
    print(f"vocab_mask[0] = {vocab_mask[0].cpu().tolist()}")
    # Should be all 0s

    print("\nAll tests completed!")


if __name__ == "__main__":
    demo_test()
