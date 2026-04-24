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
# ==============================================================================
"""MaskedFlashAttention — flash-varlen attention with optional per-key bool mask.

Drop-in replacement for ``diffusers.models.attention.Attention`` in DiT-style
diffusion stacks.  Built-in diffusers-named projections (``to_q``, ``to_k``,
``to_v``, ``to_out.0``) so upstream checkpoints load via plain
``load_state_dict`` without any name remapping.

Two dispatch paths (single flash kernel, no SDPA fallback):

* self-attn (``encoder_hidden_states=None``, ``attention_mask=None``):
  pack ``(B, Lq, H, D)`` Q/K/V as ``(B*Lq, H, D)`` with uniform cu_seqlens
  and call ``flash_attn_varlen_func``.
* masked cross-attn (``encoder_hidden_states`` given, ``attention_mask:
  (B, Lk) bool``): gather valid K/V rows per request into a packed
  ``(sum(valid_i), H, D)`` buffer; compute ``cu_seqlens_k`` from
  ``mask.sum(dim=1)``; Q stays dense with uniform cu_seqlens_q.

Designed for DiT-class diffusion models with padded cross-attn
conditioning (Stable Diffusion, PixArt, FLUX, HunyuanDiT, Groot N1.7,
...).  Not for decoder-only flow matching (alpamayo, pi0) — those rely
on sglang's ``RadixAttention`` + shared VLM KV cache, a fundamentally
different topology.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.jit_kernel.flash_attention import flash_attn_varlen_func


class MaskedFlashAttention(nn.Module):
    """Flash-varlen attention with optional per-key bool mask."""

    def __init__(
        self,
        query_dim: int,
        kv_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        attention_bias: bool = False,
        out_bias: bool = True,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        kv_dim = kv_dim if kv_dim is not None else query_dim

        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim
        self.softmax_scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=attention_bias)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=attention_bias)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=attention_bias)
        self.to_out = nn.ModuleList(
            [
                nn.Linear(inner_dim, query_dim, bias=out_bias),
                nn.Dropout(dropout),
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch=None,  # threaded-only; unused at dispatch
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(
                "hidden_states must be (B, Lq, query_dim); got "
                f"{tuple(hidden_states.shape)}"
            )
        if not hidden_states.is_cuda:
            raise RuntimeError(
                "MaskedFlashAttention requires CUDA tensors; got device "
                f"{hidden_states.device}"
            )
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(
                "MaskedFlashAttention requires bf16 or fp16 inputs; got "
                f"{hidden_states.dtype}"
            )

        B, Lq, _ = hidden_states.shape
        H, D = self.heads, self.dim_head
        device = hidden_states.device

        kv_src = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        q = self.to_q(hidden_states).view(B, Lq, H, D)
        k = self.to_k(kv_src)
        v = self.to_v(kv_src)

        q_packed = q.reshape(B * Lq, H, D).contiguous()
        cu_q = torch.arange(0, (B + 1) * Lq, Lq, dtype=torch.int32, device=device)
        max_q = Lq

        if attention_mask is None:
            Lk = kv_src.shape[1]
            k_packed = k.view(B, Lk, H, D).reshape(B * Lk, H, D).contiguous()
            v_packed = v.view(B, Lk, H, D).reshape(B * Lk, H, D).contiguous()
            cu_k = torch.arange(0, (B + 1) * Lk, Lk, dtype=torch.int32, device=device)
            max_k = Lk
        else:
            if attention_mask.dtype != torch.bool:
                raise RuntimeError(
                    "attention_mask must be a bool tensor of shape (B, Lk); got "
                    f"dtype={attention_mask.dtype}"
                )
            if attention_mask.dim() != 2:
                raise RuntimeError(
                    "attention_mask must be 2D (B, Lk); got shape "
                    f"{tuple(attention_mask.shape)}"
                )
            Lk = kv_src.shape[1]
            k = k.view(B, Lk, H, D)
            v = v.view(B, Lk, H, D)
            # Row-major gather drops invalid positions and preserves per-request
            # ordering: the first valid_lens[0] rows belong to request 0, the
            # next valid_lens[1] rows belong to request 1, etc.
            k_packed = k[attention_mask].contiguous()
            v_packed = v[attention_mask].contiguous()
            valid_lens = attention_mask.sum(dim=1).to(torch.int32)
            cu_k = F.pad(valid_lens.cumsum(dim=0), (1, 0)).to(torch.int32)
            max_k = int(valid_lens.max().item())
            # Short-circuit: if any request has zero valid keys, flash-varlen
            # would divide by zero on that row.  Fall back to an all-zeros
            # output contribution there by raising early — this should not
            # happen in normal DiT usage.
            if max_k == 0:
                raise RuntimeError(
                    "MaskedFlashAttention: all-false attention_mask row "
                    "(no valid keys) is not supported."
                )

        out_packed = flash_attn_varlen_func(
            q_packed,
            k_packed,
            v_packed,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=self.softmax_scale,
            causal=False,
        )

        out = out_packed.view(B, Lq, self.inner_dim)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out
