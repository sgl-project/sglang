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
def hash_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    PRIME: tl.constexpr,
    XCONST: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    data = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.int64)
    mixed = data ^ (offsets.to(tl.int64) + XCONST)
    hash_val = mixed * PRIME
    hash_val = hash_val ^ (hash_val >> 16)
    hash_val = hash_val * (PRIME ^ XCONST)
    hash_val = hash_val ^ (hash_val >> 13)

    tl.store(output_ptr + offsets, hash_val, mask=mask)


PRIME_1 = -(11400714785074694791 ^ 0xFFFFFFFFFFFFFFFF) - 1
PRIME_2 = -(14029467366897019727 ^ 0xFFFFFFFFFFFFFFFF) - 1


def gpu_tensor_hash(tensor: torch.Tensor) -> int:
    assert tensor.is_cuda
    tensor = tensor.contiguous().view(torch.int32)
    n = tensor.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    intermediate_hashes = torch.empty(n, dtype=torch.int64, device=tensor.device)

    hash_kernel[grid](
        tensor,
        intermediate_hashes,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
        PRIME=PRIME_1,
        XCONST=PRIME_2,
    )

    # TODO: threads can't be synced on triton kernel
    final_hash = intermediate_hashes.sum().item()

    return final_hash
