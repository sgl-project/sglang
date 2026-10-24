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
"""ROCm fused MLA Q/cache path for Kimi-K3.

Shared ``kimi_k3.py`` keeps an ``_is_hip`` call. These functions still
take the attention module as ``self``.
"""

from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs


def _bf16_q_decode(self, heads: int, kv_cache: torch.Tensor) -> bool:
    # Gluon and Triton decode take the 12-head BF16 Q as written. ASM pads Q
    # to 16 heads itself, so it keeps the split chain.
    if self.current_attention_backend in ("triton", "triton_mla"):
        return True
    from sglang.srt.layers.attention.aiter_mla_gluon import (
        prefer_mla_gluon_decode,
    )

    return prefer_mla_gluon_decode(
        head_pad_mode="zero",
        num_head=heads,
        kv_cache_dtype=kv_cache.dtype,
        q_dtype=torch.bfloat16,
    )


def _mla_q_cache_shapes_match(
    self,
    q_nope_out: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    tokens: int,
    heads: int,
) -> bool:
    return (
        q_nope_out.shape == (tokens, heads, self.kv_lora_rank)
        and q_pe.shape == (tokens, heads, self.qk_rope_head_dim)
        and k_nope.shape == (tokens, 1, self.kv_lora_rank)
        and k_pe.shape == (tokens, 1, self.qk_rope_head_dim)
        and out_cache_loc.shape == (tokens,)
        and positions.shape == (tokens,)
    )


def try_fused_mla_q_cache(
    self,
    q_nope_out: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not self._k3_mla_q_cache_fusion:
        return None
    tokens, heads = q_nope_out.shape[0], q_nope_out.shape[1]

    from sglang.srt.layers.rocm_linear_utils import (
        fused_qk_rope_cat_and_cache_mla,
    )
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    kv_cache = get_token_to_kv_pool().get_key_buffer(self.attn_mqa.layer_id)
    scale = self.attn_mqa.k_scale
    if scale is None:
        scale = self._k3_mla_q_cache_scale
    cos_cache = self._k3_identity_rope_cos
    sin_cache = self._k3_identity_rope_sin

    if not _bf16_q_decode(self, heads, kv_cache):
        return None
    if not _mla_q_cache_shapes_match(
        self,
        q_nope_out,
        q_pe,
        k_nope,
        k_pe,
        positions,
        out_cache_loc,
        tokens,
        heads,
    ):
        return None
    q_out = _q_out_buffer(
        self,
        tokens=tokens,
        heads=heads,
        head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
        dtype=q_nope_out.dtype,
        device=q_nope_out.device,
    )
    from sglang.kernels.ops.attention import mla_q_cache_aiter_hip

    # K3 MLA has no RoPE: cos/sin are one-row identity tables. Not every kernel
    # clamps the position, so all-zero positions keep the reads on row 0.
    zero_positions = _zero_positions(self, tokens=tokens, device=positions.device)
    if mla_q_cache_aiter_hip.covered(
        q_nope_out,
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        out_cache_loc,
        zero_positions,
        scale,
        cos_cache,
        sin_cache,
        q_out,
        q_scale=self._k3_mla_q_cache_scale,
    ):
        q = mla_q_cache_aiter_hip.run(
            q_nope=q_nope_out,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            kv_cache=kv_cache,
            slot_mapping=out_cache_loc,
            positions=zero_positions,
            k_scale=scale,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            out=q_out,
            q_scale=self._k3_mla_q_cache_scale,
        )
    else:
        q, _, _, _ = fused_qk_rope_cat_and_cache_mla(
            q_nope_out,
            q_pe,
            k_nope,
            k_pe,
            kv_cache,
            out_cache_loc,
            zero_positions,
            cos_cache,
            sin_cache,
            scale,
            True,
            q_out_dtype=q_nope_out.dtype,
        )
    k_placeholder = _k_placeholder(
        self, tokens=k_nope.shape[0], dtype=k_nope.dtype, device=k_nope.device
    )
    return q, k_placeholder


def _persistent(self, name: str, tokens: int, make) -> torch.Tensor:
    # A captured graph keeps the pointer it saw, so an outgrown buffer is
    # retired rather than freed.
    buf = self._k3_mla_buffers.get(name)
    if buf is None or buf.shape[0] < tokens:
        if buf is not None:
            self._k3_mla_retired_buffers.append(buf)
        buf = make(tokens)
        self._k3_mla_buffers[name] = buf
    return buf[:tokens]


def _q_out_buffer(
    self,
    *,
    tokens: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return _persistent(
        self,
        f"q_out_{dtype}_{heads}",
        tokens,
        lambda rows: torch.empty((rows, heads, head_dim), dtype=dtype, device=device),
    )


def _zero_positions(self, *, tokens: int, device: torch.device) -> torch.Tensor:
    return _persistent(
        self,
        "zero_positions",
        tokens,
        lambda rows: torch.zeros((rows,), dtype=torch.int64, device=device),
    )


def _k_placeholder(
    self, *, tokens: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    # The fused kernel already wrote K to the cache; attention only reads its shape.
    width = self.kv_lora_rank + self.qk_rope_head_dim
    return _persistent(
        self,
        f"k_placeholder_{dtype}",
        tokens,
        lambda rows: torch.empty((rows, 1, width), dtype=dtype, device=device),
    )


def init_mla_rocm_state(self) -> None:
    self._k3_mla_q_cache_fusion = envs.SGLANG_OPT_HIP_K3_MLA_Q_CACHE_FUSION.get()
    self._k3_mla_buffers: dict[str, torch.Tensor] = {}
    self._k3_mla_retired_buffers: list[torch.Tensor] = []
    # One-row identity RoPE tables: K3 MLA applies no RoPE, so cos=1 and sin=0.
    rope_cos = rope_sin = q_cache_scale = None
    if self._k3_mla_q_cache_fusion:
        rope_cos = torch.ones((1, 32), dtype=torch.bfloat16)
        rope_sin = torch.zeros((1, 32), dtype=torch.bfloat16)
        q_cache_scale = torch.ones((1,), dtype=torch.float32)
    self.register_buffer("_k3_identity_rope_cos", rope_cos, persistent=False)
    self.register_buffer("_k3_identity_rope_sin", rope_sin, persistent=False)
    self.register_buffer("_k3_mla_q_cache_scale", q_cache_scale, persistent=False)
