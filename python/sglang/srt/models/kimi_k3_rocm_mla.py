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
one-row identity cos/sin tables with all-zero positions. ``kimi_k3.py`` builds
the writer behind ``_is_hip``; the shared ROCm MLA forward calls it.
"""

from typing import Callable, Optional, Tuple

import torch
from torch import nn

from sglang.kernels.ops.attention import mla_q_cache_aiter_hip

# MLA decode and target verify on these backends read only Q and the KV cache,
# never k, and take the BF16 Q the split chain would build.
_FUSED_DECODE_BACKENDS = frozenset({"aiter", "triton"})


class KimiK3RocmMlaQCacheWriter(nn.Module):
    def __init__(self, *, kv_lora_rank: int, qk_rope_head_dim: int) -> None:
        super().__init__()
        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim
        self._head_dim = kv_lora_rank + qk_rope_head_dim
        self._graph_buffers: dict[str, torch.Tensor] = {}
        self._retired_graph_buffers: list[torch.Tensor] = []
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
        k_scale: Optional[torch.Tensor],
        attention_backend: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (q, k placeholder) after writing Q and the KV cache, or None
        to leave both to the split rope/concat/cache chain."""
        tokens, num_heads = q_nope.shape[0], q_nope.shape[1]
        if attention_backend not in _FUSED_DECODE_BACKENDS or not self._shapes_match(
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            out_cache_loc=out_cache_loc,
        ):
            return None

        # The identity tables have one row and not every kernel clamps the
        # position, so every token reads row 0.
        zero_positions = self._graph_buffer(
            name="zero_positions",
            tokens=tokens,
            make=lambda rows: torch.zeros(
                (rows,), dtype=torch.int64, device=q_nope.device
            ),
        )
        kernel_args = dict(
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            kv_cache=kv_cache,
            slot_mapping=out_cache_loc,
            positions=zero_positions,
            k_scale=self._unit_scale if k_scale is None else k_scale,
        )
        q = self._write_aiter(kernel_args=kernel_args, num_heads=num_heads)
        if q is None:
            q = self._write_triton(kernel_args=kernel_args)
        # The KV cache is already written; attention reads only the shape of k.
        k_placeholder = self._graph_buffer(
            name=f"k_placeholder_{k_nope.dtype}",
            tokens=tokens,
            make=lambda rows: torch.empty(
                (rows, 1, self._head_dim), dtype=k_nope.dtype, device=k_nope.device
            ),
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
            return None
        q_nope = kernel_args["q_nope"]
        q_out = self._graph_buffer(
            name=f"q_out_{q_nope.dtype}_{num_heads}",
            tokens=q_nope.shape[0],
            make=lambda rows: torch.empty(
                (rows, num_heads, self._head_dim),
                dtype=q_nope.dtype,
                device=q_nope.device,
            ),
        )
        aiter_args = dict(
            kernel_args,
            cos_cache=self._identity_cos,
            sin_cache=self._identity_sin,
            out=q_out,
            q_scale=self._unit_scale,
        )
        if not mla_q_cache_aiter_hip.covered(**aiter_args):
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

    def _graph_buffer(
        self, *, name: str, tokens: int, make: Callable[[int], torch.Tensor]
    ) -> torch.Tensor:
        # A captured graph keeps the pointer it saw, so an outgrown buffer is
        # retired rather than freed.
        buf = self._graph_buffers.get(name)
        if buf is None or buf.shape[0] < tokens:
            if buf is not None:
                self._retired_graph_buffers.append(buf)
            buf = make(tokens)
            self._graph_buffers[name] = buf
        return buf[:tokens]
