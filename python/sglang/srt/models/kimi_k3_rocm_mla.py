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

_aiter_mla_gate = envs.SGLANG_ROCM_K3_AITER_MLA_GATE.get()


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
    if (
        not isinstance(scale, torch.Tensor)
        or not isinstance(self._k3_mla_q_cache_scale, torch.Tensor)
        or cos_cache is None
        or sin_cache is None
    ):
        return None

    # AITER's asm decode uses FP8 Q when the cache is FP8. Gluon keeps Q in
    # BF16 while retaining the fused FP8 cache write; its h12/bh16 kernel is
    # both faster and more accurate for K3's long-context decode regime.
    triton_decode = self.current_attention_backend in ("triton", "triton_mla")
    tokens, heads = q_nope_out.shape[0], q_nope_out.shape[1]
    from sglang.srt.layers.attention.aiter_mla_gluon import (
        prefer_mla_gluon_decode,
    )

    gluon_decode = not triton_decode and prefer_mla_gluon_decode(
        head_pad_mode="zero",
        num_head=heads,
        kv_cache_dtype=kv_cache.dtype,
        q_dtype=torch.bfloat16,
    )
    q_out_dtype = q_nope_out.dtype if triton_decode or gluon_decode else kv_cache.dtype
    if (
        q_nope_out.shape != (tokens, heads, self.kv_lora_rank)
        or q_pe.shape != (tokens, heads, self.qk_rope_head_dim)
        or k_nope.shape != (tokens, 1, self.kv_lora_rank)
        or k_pe.shape != (tokens, 1, self.qk_rope_head_dim)
        or out_cache_loc.shape != (tokens,)
        or positions.shape != (tokens,)
    ):
        return None
    # AITER's qh16 decode kernel needs 16 Q heads. K3 TP8 has 12; the
    # previous producer emitted 12 and the backend `F.pad`ed, which is a
    # FillFunctor per MLA layer. Write the 12 real heads into a persistent
    # zeroed 16-head buffer instead so decode can skip that launch.
    # Do not use uninitialized pad heads: that is the fill-elim that
    # already failed GSM8K.
    aiter_pad_heads = (
        heads if gluon_decode else (16 if (heads < 16 and 16 % heads != 0) else heads)
    )
    q_out = mla_q_out_buffer(
        self,
        tokens,
        aiter_pad_heads,
        self.kv_lora_rank + self.qk_rope_head_dim,
        q_out_dtype,
        q_nope_out.device,
        zero_init=aiter_pad_heads > heads,
    )
    from sglang.kernels.ops.attention import mla_q_cache_aiter_hip

    if mla_q_cache_aiter_hip.covered(
        q_nope_out,
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        out_cache_loc,
        positions,
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
            positions=positions,
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
            positions,
            cos_cache,
            sin_cache,
            scale,
            True,
            q_scale=self._k3_mla_q_cache_scale,
            q_out_dtype=q_out_dtype,
            compute_all_q_rope=False,
            identity_rope=True,
        )
    k_placeholder = mla_k_placeholder(
        self, k_nope.shape[0], k_nope.dtype, k_nope.device
    )
    return q, k_placeholder


def mla_q_out_buffer(
    self,
    tokens: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    zero_init: bool = False,
) -> torch.Tensor:
    """Persistent fused-Q workspace, including optional MLA pad heads."""
    buf = getattr(self, "_k3_mla_q_out", None)
    if (
        not isinstance(buf, torch.Tensor)
        or buf.dtype != dtype
        or buf.device != device
        or buf.shape[1] != heads
        or buf.shape[2] != head_dim
        or buf.shape[0] < tokens
    ):
        buf = torch.empty((tokens, heads, head_dim), dtype=dtype, device=device)
        if zero_init:
            buf.view(torch.uint8).zero_()
        self._k3_mla_q_out = buf
    return buf[:tokens]


def mla_k_placeholder(
    self, tokens: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    width = self.kv_lora_rank + self.qk_rope_head_dim
    buf = getattr(self, "_k3_mla_k_placeholder", None)
    if (
        not isinstance(buf, torch.Tensor)
        or buf.dtype != dtype
        or buf.device != device
        or buf.shape[1] != 1
        or buf.shape[2] != width
        or buf.shape[0] < tokens
    ):
        buf = torch.empty((tokens, 1, width), dtype=dtype, device=device)
        self._k3_mla_k_placeholder = buf
    return buf[:tokens]


def init_mla_rocm_state(self) -> None:
    self._k3_mla_q_cache_fusion = envs.SGLANG_ROCM_K3_AITER_MLA_Q_CACHE_FUSION.get()
    if self._k3_mla_q_cache_fusion:
        self.register_buffer(
            "_k3_identity_rope_cos",
            torch.ones((1, 32), dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "_k3_identity_rope_sin",
            torch.zeros((1, 32), dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "_k3_mla_q_cache_scale",
            torch.ones((1,), dtype=torch.float32),
            persistent=False,
        )
        self.attn_mqa.register_buffer(
            "_k3_mla_q_scale",
            torch.ones((1,), dtype=torch.float32),
            persistent=False,
        )
    else:
        self.register_buffer("_k3_identity_rope_cos", None, persistent=False)
        self.register_buffer("_k3_identity_rope_sin", None, persistent=False)
        self.register_buffer("_k3_mla_q_cache_scale", None, persistent=False)


def prepare_mla_rocm(self) -> None:
    if _aiter_mla_gate:
        from sglang.kernels.ops.attention import mla_gate_aiter_hip

        mla_gate_aiter_hip.warmup(self.g_proj.weight)


def apply_output_gate(self, gate_input: torch.Tensor, x: torch.Tensor):
    """x * sigmoid(g_proj(gate_input)) through the ROCm fused kernels."""
    from sglang.kernels.ops.attention import mla_gate_aiter_hip, mla_output_gate
    from sglang.srt.models.kimi_k3_rocm_fusion import _k3_fuse_mla_gate_ptpc

    if (
        self._gate_pending_stream is None
        and _aiter_mla_gate
        and mla_gate_aiter_hip.covered(gate_input, self.g_proj.weight, x)
    ):
        return mla_gate_aiter_hip.run(gate_input, self.g_proj.weight, x)
    gate = self._compute_output_gate(gate_input)
    fused_gate = _k3_fuse_mla_gate_ptpc(x, gate=gate, o_proj=self.o_proj)
    if fused_gate is not None:
        return fused_gate
    if mla_output_gate.covered(x, gate):
        return mla_output_gate.kimi_k3_mla_output_gate(x, gate)
    return x * torch.sigmoid(gate)
