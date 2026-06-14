from __future__ import annotations

"""
Copyright 2023-2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
MambaCheckpointPool — the radix prefix cache's int8-compressed store for cached
linear-attention (KDA / GDN) states.

It decouples the *cached* states (radix-owned, idle, compressed) from the *active*
``MambaPool`` (running requests, bf16/fp32, kernel-facing). The radix stores one
cached state per node HERE; on a prefix-cache hit it is dequantized back into a
fresh active slot (copy-on-write).

Per cached slot it holds:
  * the SSM temporal state in **int8** (per-(head,k-channel) symmetric), via
    ``Int8CheckpointStore`` — ~2x more cached states than bf16, quality-safe
    (one-time rounding on store; validated GSM8K 0.888 vs 0.898).
  * the conv1d window state in **bf16** (tiny, W-1 tokens; not worth quantizing).

This is strategy-agnostic: whether the active slot to be cached was produced by
the ``no_buffer`` donate (copy_from) or the ``extra_buffer`` ping-pong track
buffer (spec path), both converge on "an active slot becomes the cached
``mamba_value``" — which is exactly the (store_from_active) hook here. Slot
lifecycle is owned by the caller via the embedded ``MambaSlotAllocator``.
"""

from typing import List

import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.int8_checkpoint_store import Int8CheckpointStore


class MambaCheckpointPool:
    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_shapes: List[tuple],
        conv_dtype: torch.dtype,
        device: str,
    ):
        self.num_slots = num_slots
        self.device = device
        self.temporal = Int8CheckpointStore(
            num_layers=num_layers,
            num_slots=num_slots + 1,  # slot 0 reserved (matches MambaSlotAllocator)
            num_heads=num_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
            device=device,
        )
        # conv windows stay bf16 (small); one buffer per conv tensor in the State
        self.conv = [
            torch.empty(
                (num_layers, num_slots + 1) + tuple(shape),
                dtype=conv_dtype,
                device=device,
            )
            for shape in conv_shapes
        ]
        self.allocator = MambaSlotAllocator(size=num_slots, device=device)

    # ---- lifecycle (delegates to the embedded allocator) ----

    def alloc(self, n: int = 1):
        return self.allocator.alloc(n)

    def free(self, slots: torch.Tensor):
        self.allocator.free(slots)

    def available_size(self) -> int:
        return self.allocator.available_size()

    def clear(self) -> None:
        """Release every checkpoint slot (radix flush/reset). The int8 qdata is
        left as-is; slots are reused/overwritten on the next store."""
        self.allocator.clear()

    # ---- state transfer between the active MambaPool and this store ----

    def store_from_active(self, active_mamba_pool, active_slots, ckpt_slots) -> None:
        """Quantize temporal + copy conv from the active pool into checkpoint slots
        (the radix donate / cache-store)."""
        cache = active_mamba_pool.mamba_cache
        self.temporal.store_from_bf16_pool(cache.temporal, active_slots, ckpt_slots)
        for i, c in enumerate(self.conv):
            c[:, ckpt_slots] = cache.conv[i][:, active_slots]

    def load_to_active(self, active_mamba_pool, ckpt_slots, active_slots) -> None:
        """Dequantize temporal + copy conv from checkpoint slots into the active pool
        (the cache-hit copy-on-write)."""
        cache = active_mamba_pool.mamba_cache
        self.temporal.copy_to_bf16_pool(cache.temporal, ckpt_slots, active_slots)
        for i, c in enumerate(self.conv):
            cache.conv[i][:, active_slots] = c[:, ckpt_slots].to(cache.conv[i].dtype)

    def mem_usage_bytes(self) -> int:
        conv_bytes = sum(c.numel() * c.element_size() for c in self.conv)
        return self.temporal.mem_usage_bytes() + conv_bytes
