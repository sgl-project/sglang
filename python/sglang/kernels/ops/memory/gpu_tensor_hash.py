# SPDX-License-Identifier: Apache-2.0
"""Legacy scalar checksum; multimodal caches use full BLAKE3 identities."""

import torch

from sglang.kernels.ops.memory.gpu_tensor_hash_blake3 import gpu_tensor_blake3


def gpu_tensor_hash(tensor: torch.Tensor) -> int:
    """Compatibility 64-bit checksum, not an authoritative cache identity.

    Replaces the additive two-lane hash whose structured collisions were
    trivial to construct. New cache callers must retain gpu_tensor_blake3's
    entire 32-byte result and include tensor structure in the outer identity.
    """
    return int.from_bytes(gpu_tensor_blake3(tensor)[:8], "big")
