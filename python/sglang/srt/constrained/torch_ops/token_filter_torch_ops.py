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
"""Torch impl for token filter operations."""

from typing import List

import numpy as np
import torch


def set_token_filter_torch(
    vocab_mask: torch.Tensor,
    token_ids: List[int],
    batch_idx: int,
    is_allowed: bool = True,
    reset_vocab_mask: bool = True,
):
    if reset_vocab_mask:
        mask_val = -1 if (not is_allowed) else 0
        vocab_mask[batch_idx].fill_(mask_val)

    for token_id in token_ids:
        element_idx = token_id // 32
        bit_idx = token_id % 32
        current_value = vocab_mask[batch_idx, element_idx].item()

        if is_allowed:
            new_value = current_value | (1 << bit_idx)
        else:
            new_value = current_value & (~(1 << bit_idx) & 0xFFFFFFFF)
        vocab_mask[batch_idx, element_idx] = np.int32(new_value)


def demo_test():
    """Demo test to verify the Triton implementation."""
    import torch

    # Test 1: Set specific tokens as allowed (reset to 0, then set bits)
    vocab_mask = torch.zeros(2, 4, dtype=torch.int32, device="cuda")
    token_ids = [10, 20, 30, 40, 50]

    set_token_filter_torch(
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

    for i in range(100):
        set_token_filter_torch(
            vocab_mask, token_ids, batch_idx=1, is_allowed=False, reset_vocab_mask=True
        )

    print("\nTest 2 - Block tokens:")
    print(f"vocab_mask[1] = {vocab_mask[1].cpu().tolist()}")
    # All should be -1 except bits 10, 20, 30 in element 0 and bits 8, 18 in element 1 should be cleared

    # Test 3: Set tokens without reset
    vocab_mask = torch.full((2, 4), -1, dtype=torch.int32, device="cuda")
    token_ids = [10, 20]

    set_token_filter_torch(
        vocab_mask, token_ids, batch_idx=0, is_allowed=True, reset_vocab_mask=False
    )

    print("\nTest 3 - Set tokens without reset:")
    print(f"vocab_mask[0] = {vocab_mask[0].cpu().tolist()}")
    # Should still be -1 with bits 10 and 20 set (but -1 already has all bits set)

    # Test 4: Empty token list with reset
    vocab_mask = torch.ones((2, 4), dtype=torch.int32, device="cuda")

    set_token_filter_torch(
        vocab_mask, [], batch_idx=0, is_allowed=True, reset_vocab_mask=True
    )

    print("\nTest 4 - Empty token list with reset:")
    print(f"vocab_mask[0] = {vocab_mask[0].cpu().tolist()}")
    # Should be all 0s

    print("\nAll tests completed!")


if __name__ == "__main__":
    demo_test()
