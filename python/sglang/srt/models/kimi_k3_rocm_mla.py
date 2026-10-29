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

K3 MLA applies no RoPE, so the RoPE stage of the fused kernels runs against
one-row identity cos/sin tables with all-zero positions.
"""

from typing import Optional, Tuple

import torch
from torch import nn

from sglang.kernels.ops.attention import mla_q_cache_aiter_hip
from sglang.srt.runtime_context import get_buffer
from sglang.srt.utils.common import print_warning_once

# With save_kv_cache=False, these backends read only the BF16 Q and the KV cache;
# Triton target-verify runs its extend kernel, which also reads k.
_DECODE_BACKENDS = frozenset({"aiter", "triton"})
_TARGET_VERIFY_BACKENDS = frozenset({"aiter"})


class _ZeroPositions:
    """Int64 zeros that every K3 MLA layer passes as the identity-RoPE positions."""

    def __init__(self) -> None:
        self.buf: Optional[torch.Tensor] = None
        self.retired: list[torch.Tensor] = []

    def get(self, tokens: int, device: torch.device) -> torch.Tensor:
        buf = self.buf
        if buf is None or buf.numel() < tokens or buf.device != device:
            if torch.cuda.is_current_stream_capturing():
                # A memset recorded into a graph runs only at replay,
                # so it cannot seed the shared buffer.
                return torch.zeros((tokens,), dtype=torch.int64, device=device)
            if buf is not None:
                # Captured graphs keep reading the old pointer;
                # power-of-two growth keeps retired buffers smaller than the live one.
                self.retired.append(buf)
            rows = 1 << (tokens - 1).bit_length()
            self.buf = buf = torch.zeros((rows,), dtype=torch.int64, device=device)
        return buf[:tokens]


class KimiK3RocmMlaQCacheWriter(nn.Module):
    def __init__(self, *, kv_lora_rank: int, qk_rope_head_dim: int) -> None:
        super().__init__()
        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim
        self._head_dim = kv_lora_rank + qk_rope_head_dim
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
        out_cache_loc: torch.Tensor,
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
                out_cache_loc=out_cache_loc,
            )
        ):
            return None

        tokens, num_heads = q_nope.shape[0], q_nope.shape[1]
        kernel_args = dict(
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            kv_cache=kv_cache,
            slot_mapping=out_cache_loc,
            # The identity tables have one row and not every kernel clamps positions,
            # so every token reads row 0.
            positions=get_buffer("kimi_k3_mla_zero_positions", _ZeroPositions).get(
                tokens=tokens, device=q_nope.device
            ),
            k_scale=self._unit_scale,
        )
        q = self._write_aiter(kernel_args=kernel_args, num_heads=num_heads)
        if q is None:
            q = self._write_triton(kernel_args=kernel_args)
        # The KV cache is already written; attention reads only the shape of k.
        k_placeholder = torch.empty(
            (tokens, 1, self._head_dim), dtype=k_nope.dtype, device=k_nope.device
        )
        return q, k_placeholder

    def _shapes_match(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> bool:
        tokens, num_heads = q_nope.shape[0], q_nope.shape[1]
        return (
            tokens > 0
            and q_nope.shape == (tokens, num_heads, self._kv_lora_rank)
            and q_pe.shape == (tokens, num_heads, self._qk_rope_head_dim)
            and k_nope.shape == (tokens, 1, self._kv_lora_rank)
            and k_pe.shape == (tokens, 1, self._qk_rope_head_dim)
            and out_cache_loc.shape == (tokens,)
        )

    def _write_aiter(
        self, *, kernel_args: dict, num_heads: int
    ) -> Optional[torch.Tensor]:
        if not mla_q_cache_aiter_hip.available():
            print_warning_once(
                "Kimi-K3 MLA Q/cache fusion: the AITER op needs gfx950 and an "
                "AITER build with fused_qk_rope_concat_and_cache_mla; using the "
                "AITER Triton kernel instead."
            )
            return None
        q_nope = kernel_args["q_nope"]
        aiter_args = dict(
            kernel_args,
            cos_cache=self._identity_cos,
            sin_cache=self._identity_sin,
            out=torch.empty(
                (q_nope.shape[0], num_heads, self._head_dim),
                dtype=q_nope.dtype,
                device=q_nope.device,
            ),
            q_scale=self._unit_scale,
        )
        if not mla_q_cache_aiter_hip.covered(**aiter_args):
            print_warning_once(
                "Kimi-K3 MLA Q/cache fusion: inputs are outside the AITER op's "
                "contract (dtype, stride or cache layout); using the AITER "
                "Triton kernel instead."
            )
            return None
        return mla_q_cache_aiter_hip.run(**aiter_args)

    def _write_triton(self, *, kernel_args: dict) -> torch.Tensor:
        from sglang.srt.layers.rocm_linear_utils import (
            fused_qk_rope_cat_and_cache_mla,
        )

        q, _, _, _ = fused_qk_rope_cat_and_cache_mla(
            kernel_args["q_nope"],
            kernel_args["q_pe"],
            kernel_args["k_nope"],
            kernel_args["k_pe"],
            kernel_args["kv_cache"],
            kernel_args["slot_mapping"],
            kernel_args["positions"],
            self._identity_cos,
            self._identity_sin,
            kernel_args["k_scale"],
            True,  # is_neox
            q_out_dtype=kernel_args["q_nope"].dtype,
        )
        return q
