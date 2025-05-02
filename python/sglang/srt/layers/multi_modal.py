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
"""Logits processing."""

import torch
import triton
import triton.language as tl


@triton.jit
def simple_hash_kernel(
    input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, PRIME: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    data = tl.load(input_ptr + offsets, mask=mask, other=0)

    # Simple hash computation (multiplication and shifts, no xor/shuffle)
    hash_val = data * PRIME
    hash_val = hash_val + (hash_val >> 16)
    hash_val = hash_val * PRIME
    hash_val = hash_val + (hash_val >> 13)

    # Store intermediate hash values
    tl.store(output_ptr + offsets, hash_val, mask=mask)


def gpu_tensor_hash(tensor: torch.Tensor, prime=0x01000193) -> int:
    assert tensor.is_cuda, "Tensor must be on CUDA device"
    tensor = tensor.contiguous().view(torch.int32)

    n_elements = tensor.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    intermediate_hashes = torch.empty(
        n_elements, dtype=torch.int32, device=tensor.device
    )

    simple_hash_kernel[grid](
        tensor, intermediate_hashes, n_elements, BLOCK_SIZE=BLOCK_SIZE, PRIME=prime
    )

    # Final reduction on GPU (simple sum reduction)
    final_hash = intermediate_hashes.sum().item()

    return final_hash
