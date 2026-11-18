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
"""Kimi-K3 ROCm MLA decode: write the query and the latent KV cache in one launch.

K3 MLA applies no RoPE. The gfx950 HIP op has only a RoPE form, so it runs against
one-row identity cos/sin tables with all-zero positions; the Triton fallback uses
AITER's no-RoPE kernel.
"""

from typing import Optional, Tuple

import torch
from torch import nn

from sglang.kernels.ops.attention import (
    mla_q_cache_aiter_hip,
    mla_q_cache_aiter_triton,
)
from sglang.srt.runtime_context import get_buffer
from sglang.srt.utils.device import is_device_stream_capturing
from sglang.srt.utils.log_utils import print_warning_once

# With save_kv_cache=False, these backends read only the BF16 Q and the KV cache;
# Triton target-verify runs its extend kernel, which also reads k.
_DECODE_BACKENDS = frozenset({"aiter", "triton"})
_TARGET_VERIFY_BACKENDS = frozenset({"aiter"})


class _ZeroPositions:
    """Int64 zeros that every K3 MLA layer passes as the identity-RoPE positions.

    Only eager steps lease or grow the buffer; the graph runner's eager warmups before
    each capture size it, so a capture never allocates.
    """

    def __init__(self) -> None:
        self.buf: Optional[torch.Tensor] = None
        self.retired: list[torch.Tensor] = []

    def get(self, tokens: int, device: torch.device) -> Optional[torch.Tensor]:
        """The first `tokens` zeros, or None inside a capture the warmup did not size."""
        buf = self.buf
        if buf is None or buf.numel() < tokens or buf.device != device:
            if is_device_stream_capturing(device):
                return None
            if buf is not None:
                # Captured graphs keep reading the old pointer;
                # power-of-two growth keeps retired buffers smaller than the live one.
                self.retired.append(buf)
            rows = 1 << (tokens - 1).bit_length()
            self.buf = buf = torch.zeros((rows,), dtype=torch.int64, device=device)
        return buf[:tokens]


def create_q_cache_writer(
    *, kv_lora_rank: int, qk_rope_head_dim: int
) -> Optional["KimiK3RocmMlaQCacheWriter"]:
    """The writer for this device, or None where neither fused kernel can run."""
    use_hip = mla_q_cache_aiter_hip.available()
    use_triton = mla_q_cache_aiter_triton.available()
    if not (use_hip or use_triton):
        print_warning_once(
            "Kimi-K3 MLA Q/cache fusion: no fused kernel for this device or AITER "
            "build (the HIP op needs gfx950, the no-RoPE Triton kernel is missing on "
            "gfx1250 and older AITER); keeping the split chain."
        )
        return None
    return KimiK3RocmMlaQCacheWriter(
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        use_hip=use_hip,
        use_triton=use_triton,
    )


class KimiK3RocmMlaQCacheWriter(nn.Module):
    def __init__(
        self,
        *,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        use_hip: bool,
        use_triton: bool,
    ) -> None:
        super().__init__()
        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim
        self._head_dim = kv_lora_rank + qk_rope_head_dim
        # The HIP op is preferred; the Triton kernel takes the inputs it rejects.
        self._use_hip = use_hip
        self._use_triton = use_triton
        half_rope_dim = qk_rope_head_dim // 2
        self.register_buffer(
            "_identity_cos",
            torch.ones((1, half_rope_dim), dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "_identity_sin",
            torch.zeros((1, half_rope_dim), dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "_unit_scale", torch.ones((1,), dtype=torch.float32), persistent=False
        )

    def write(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache: torch.Tensor,
        k_scale_float: Optional[float],
        attention_backend: str,
        is_target_verify: bool,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (q, k placeholder) after writing Q and the KV cache,
        or None to leave both to the split rope/concat/cache chain."""
        backends = _TARGET_VERIFY_BACKENDS if is_target_verify else _DECODE_BACKENDS
        if (
            attention_backend not in backends
            # The split chain's AITER write casts without the scale,
            # its Triton decode write divides by it; they agree only at 1.
            or k_scale_float not in (None, 1.0)
            or not self._shapes_match(
                q_nope=q_nope,
                q_pe=q_pe,
                k_nope=k_nope,
                k_pe=k_pe,
                slot_mapping=slot_mapping,
            )
        ):
            return None

        kernel_args = dict(
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            k_scale=self._unit_scale,
        )
        q = self._write_hip(kernel_args=kernel_args) if self._use_hip else None
        if q is None and self._use_triton:
            q = mla_q_cache_aiter_triton.run(**kernel_args)
        if q is None:
            print_warning_once(
                "Kimi-K3 MLA Q/cache fusion: inputs are outside the fused kernels' "
                "contract (dtype, stride or cache layout); keeping the split chain."
            )
            return None
        # The KV cache is already written; attention reads only the shape of k.
        k_placeholder = torch.empty(
            (q_nope.shape[0], 1, self._head_dim),
            dtype=k_nope.dtype,
            device=k_nope.device,
        )
        return q, k_placeholder

    def _shapes_match(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> bool:
        tokens, num_heads = q_nope.shape[0], q_nope.shape[1]
        return (
            tokens > 0
            and q_nope.shape == (tokens, num_heads, self._kv_lora_rank)
            and q_pe.shape == (tokens, num_heads, self._qk_rope_head_dim)
            and k_nope.shape == (tokens, 1, self._kv_lora_rank)
            and k_pe.shape == (tokens, 1, self._qk_rope_head_dim)
            and slot_mapping.shape == (tokens,)
        )

    def _write_hip(self, *, kernel_args: dict) -> Optional[torch.Tensor]:
        q_nope = kernel_args["q_nope"]
        tokens, num_heads = q_nope.shape[0], q_nope.shape[1]
        # The identity tables have one row and not every AITER build clamps positions
        # in every kernel, so every token reads row 0.
        positions = get_buffer("kimi_k3_mla_zero_positions", _ZeroPositions).get(
            tokens=tokens, device=q_nope.device
        )
        if positions is None:
            return None
        return mla_q_cache_aiter_hip.run(
            **kernel_args,
            positions=positions,
            cos_cache=self._identity_cos,
            sin_cache=self._identity_sin,
            out=torch.empty(
                (tokens, num_heads, self._head_dim),
                dtype=q_nope.dtype,
                device=q_nope.device,
            ),
            q_scale=self._unit_scale,
        )
