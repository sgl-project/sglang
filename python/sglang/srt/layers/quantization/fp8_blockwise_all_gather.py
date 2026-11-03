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
"""One all-gather for FP8 activations and their FP32 per-group scales.

The producer writes disjoint views into a token-major packet. A float32
bit-view keeps the existing custom-AG path available without converting values.
The consumer gets row-strided FP8 activations and column-major scale storage;
only scales are copied. The GEMM must support the activation's leading stride.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def _log_layout(hidden_size, group_size, world_size):
    logging.getLogger(__name__).info(
        "Packed FP8 blockwise all-gather: hidden=%s group=%s tp=%s, FP32 bit-view transport",
        hidden_size,
        group_size,
        world_size,
    )


@dataclass
class Fp8BlockwisePacket:
    data: torch.Tensor
    hidden_size: int
    fp8_dtype: torch.dtype
    group_size: int = 128

    @classmethod
    def allocate(cls, rows, hidden_size, fp8_dtype, device, group_size=128):
        if rows < 0 or group_size <= 0 or hidden_size <= 0:
            raise ValueError("Invalid FP8 blockwise packet dimensions")
        if hidden_size % group_size or hidden_size % 4:
            raise ValueError("Packet hidden size must be divisible by group_size and 4")
        if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            raise ValueError("FP8 blockwise packets require E4M3 activations")
        words = hidden_size // 4 + hidden_size // group_size
        return cls(
            torch.empty((rows, words), dtype=torch.float32, device=device),
            hidden_size,
            fp8_dtype,
            group_size,
        )

    @property
    def quantized(self):
        return self.data.view(torch.uint8)[:, : self.hidden_size].view(self.fp8_dtype)

    @property
    def scales(self):
        return self.data[:, self.hidden_size // 4 :]


def all_gather_fp8_blockwise(
    packet: Fp8BlockwisePacket,
    world_size: int,
    all_gather: Callable[[torch.Tensor, torch.Tensor], None],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather once, preserving FP8/FP32 bits; return the CK scale layout.

    Allocate under the caller's collective memory-pool context. Inputs must
    have equal padded row counts within this collective's group.
    """
    if world_size < 1:
        raise ValueError("world_size must be positive")
    _log_layout(packet.hidden_size, packet.group_size, world_size)
    rows, words = packet.data.shape
    gathered = torch.empty(
        (rows * world_size, words), dtype=torch.float32, device=packet.data.device
    )
    all_gather(gathered, packet.data)
    output = Fp8BlockwisePacket(
        gathered, packet.hidden_size, packet.fp8_dtype, packet.group_size
    )
    groups = packet.hidden_size // packet.group_size
    # The GEMM consumes [groups, global_rows] storage, presented as [rows, groups].
    scales = torch.empty(
        (rows * world_size, groups), dtype=torch.float32, device=gathered.device
    )
    scales.view(groups, rows * world_size).copy_(output.scales.transpose(0, 1))
    return output.quantized, scales
