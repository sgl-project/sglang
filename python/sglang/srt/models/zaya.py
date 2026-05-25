# SPDX-License-Identifier: Apache-2.0
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
"""Inference-only Zyphra ZAYA1 (CCA attention + MoE) model implementation.

Architecture summary (see docs/supported_models/text_generation/zaya_design.md
for the full design notes):

- Even-indexed layers run :class:`ZayaAttention`, which feeds hidden states to
  the :class:`CCA` (Compressed Convolutional Attention) projection. CCA emits
  q/k/v via two small (``kernel_size=2``) depthwise + grouped 1D convolutions
  over the time axis plus a learnable per-K-head temperature. The conv needs a
  two-token left padding that is sourced from a per-request state cache owned
  by the CCA module itself. The q/k/v then go through partial rotary embedding
  (``partial_rotary_factor=0.5``) and SGLang's :class:`RadixAttention` for the
  softmax MHA. The implementation only uses ``torch`` / ``torch.nn`` ops, so the
  same code runs on NVIDIA and AMD GPUs.
- Odd-indexed layers run :class:`ZayaBlock`, an MoE mixer built around SGLang's
  :class:`FusedMoE`. Expert routing uses a 3-layer MLP with EDA (depth-wise
  averaging across MoE layers) and MOD (mixture-of-depths skip expert).
- Per-layer :class:`ResidualScaling` keeps the residual stream in fp32 with
  affine scale/bias both on the residual and on the post-mixer hidden states.
- Per-request CCA state (``conv_state`` + ``prev_hs``) is held by each CCA
  module as buffers indexed by ``forward_batch.req_pool_indices``; it is sized
  lazily on the first forward pass.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.zaya import ZayaConfig
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Residual scaling
# ---------------------------------------------------------------------------


class ResidualScaling(nn.Module):
    """Affine fp32 scaling applied to the residual / hidden_states streams.

    Layer 0 has no incoming residual stream, so its checkpoint omits
    ``residual_scale`` / ``residual_bias`` and ``has_residual`` stays False.
    """

    def __init__(self, config: ZayaConfig, layer_n: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.has_residual = layer_n != 0
        self.hidden_states_scale = nn.Parameter(torch.ones(self.hidden_size))
        self.hidden_states_bias = nn.Parameter(torch.zeros(self.hidden_size))
        if self.has_residual:
            self.residual_scale = nn.Parameter(torch.ones(self.hidden_size))
            self.residual_bias = nn.Parameter(torch.zeros(self.hidden_size))

    def forward(
        self,
        residual: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        hs_scale = self.hidden_states_scale.to(torch.float32)
        hs_bias = self.hidden_states_bias.to(torch.float32)
        hidden_states = (hidden_states.float() + hs_bias) * hs_scale

        if self.has_residual and residual is not None:
            res_scale = self.residual_scale.to(torch.float32)
            res_bias = self.residual_bias.to(torch.float32)
            residual = (residual.float() + res_bias) * res_scale

        return residual, hidden_states


def _apply_norm_with_fp32_residual(
    norm: nn.Module,
    residual: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize ``residual`` (typically fp32) and cast back to ``target_dtype``.

    SGLang's fused RMSNorm kernel assumes the input dtype matches the weight
    dtype, so when the residual is kept in fp32 we route through the eager
    ``forward_native`` path before casting back to the model dtype.
    """
    if isinstance(norm, RMSNorm):
        if residual.dtype != norm.weight.dtype:
            hidden_states = norm.forward_native(residual)
        else:
            hidden_states = norm(residual)
        return hidden_states.to(target_dtype)
    return norm(residual.to(target_dtype))


# ---------------------------------------------------------------------------
# CCA: Compressed Convolutional Attention QKV projection
# ---------------------------------------------------------------------------


class CCA(nn.Module):
    """Compressed Convolutional Attention QKV projection.

    Given hidden states ``hs`` of shape ``[S, H]`` this layer produces
    ``(q, k, v)`` where:

        q = (W_q hs + Conv(W_q hs ‖ W_k hs)_q) / 2
            + mean_group(W_k hs) / 2                      (fp32, RMSNorm'd)
        k = (W_k hs + Conv(W_q hs ‖ W_k hs)_k) / 2
            + mean_group(W_q hs) / 2,  scaled by per-head temperature
        v = concat(W_{v1} hs, W_{v2} hs_prev_shifted)

    The two-stage conv on ``(W_q hs ‖ W_k hs)`` needs
    ``total_padding = (cca_time0 - 1) + (cca_time1 - 1)`` tokens of left padding.
    For the first prefill chunk of a request the padding is zero; for a resumed
    prefill or for decode it is read from a per-request cache that this module
    maintains internally.
    """

    def __init__(
        self,
        config: ZayaConfig,
        cca_num_k_heads: int,
        cca_num_q_heads: int,
        hidden_size: int,
        head_dim: int,
        cca_time0: int,
        cca_time1: int,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = int(hidden_size)
        self.head_dim = int(head_dim)
        self.cca_time0 = int(cca_time0)
        self.cca_time1 = int(cca_time1)
        self.padding0 = self.cca_time0 - 1
        self.padding1 = self.cca_time1 - 1
        self.total_padding = self.padding0 + self.padding1
        self.num_k_heads = int(cca_num_k_heads)
        self.num_q_heads = int(cca_num_q_heads)
        assert self.num_q_heads % self.num_k_heads == 0
        self.gqa_groups = self.num_q_heads // self.num_k_heads

        self.latent_k_dim = self.num_k_heads * self.head_dim
        self.latent_q_dim = self.num_q_heads * self.head_dim
        self.in_out_ch = self.latent_k_dim + self.latent_q_dim
        self.sqrt_head_dim = float(self.head_dim) ** 0.5
        self.clamp_temp = bool(getattr(config, "clamp_temp", False))

        bias = bool(getattr(config, "attention_bias", False))
        self.linear_q = ReplicatedLinear(
            self.hidden_size,
            self.latent_q_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_q", prefix),
        )
        self.linear_k = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_k", prefix),
        )
        self.val_proj1 = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim // 2,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("val_proj1", prefix),
        )
        self.val_proj2 = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim // 2,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("val_proj2", prefix),
        )

        # Two-stage depthwise + grouped conv along the time axis. Wrapping the
        # two nn.Conv1d modules in nn.Sequential makes the HF checkpoint keys
        #   conv_qk.0.{weight,bias}, conv_qk.1.{weight,bias}
        # map onto submodules 1:1 without any key rewriting at load time.
        self.conv_qk = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_out_ch,
                out_channels=self.in_out_ch,
                kernel_size=self.cca_time0,
                groups=self.in_out_ch,
                padding=0,
                stride=1,
            ),
            nn.Conv1d(
                in_channels=self.in_out_ch,
                out_channels=self.in_out_ch,
                kernel_size=self.cca_time1,
                groups=(self.num_k_heads + self.num_q_heads),
                padding=0,
                stride=1,
            ),
        )

        # Per-K-head learnable temperature scalar.
        self.temp = nn.Parameter(torch.zeros(self.num_k_heads))

        # Per-request state buffers. Sized lazily on the first forward call
        # once the actual request-pool layout is observable.
        self.register_buffer(
            "conv_state_pool",
            torch.zeros(1, self.in_out_ch, self.total_padding),
            persistent=False,
        )
        self.register_buffer(
            "prev_hs_pool",
            torch.zeros(1, self.hidden_size),
            persistent=False,
        )

    # ----- helpers ---------------------------------------------------------

    def _ensure_pool(self, forward_batch: ForwardBatch) -> None:
        """Grow the per-request state pools to cover all seen ``req_pool_indices``.

        ``ForwardBatch`` does not expose ``req_to_token_pool`` to the model, so
        the required size is inferred from the current batch's indices and from
        the server-side ``max_running_requests``. Pools only ever grow: a
        warmup / profile batch with N indices triggers a single resize, then
        all subsequent forwards reuse the same buffer.
        """
        device = (
            forward_batch.input_ids.device
            if forward_batch.input_ids is not None
            else self.conv_state_pool.device
        )

        if (
            getattr(forward_batch, "req_pool_indices", None) is not None
            and forward_batch.req_pool_indices.numel() > 0
        ):
            current_max = int(forward_batch.req_pool_indices.max().item()) + 1
        else:
            current_max = 1

        try:
            from sglang.srt.server_args import get_global_server_args

            server_args = get_global_server_args()
            server_cap = int(getattr(server_args, "max_running_requests", 0) or 0)
        except Exception:
            server_cap = 0

        required = max(current_max, server_cap, self.conv_state_pool.shape[0])
        if (
            required <= self.conv_state_pool.shape[0]
            and self.conv_state_pool.device == device
        ):
            return

        dtype = self.conv_state_pool.dtype
        self.conv_state_pool = torch.zeros(
            required,
            self.in_out_ch,
            self.total_padding,
            dtype=dtype,
            device=device,
        )
        self.prev_hs_pool = torch.zeros(
            required,
            self.hidden_size,
            dtype=dtype,
            device=device,
        )

    def _normalize_qk(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm (no learnable weight) + sqrt(head_dim) scaling on q and k,
        plus per-K-head temperature on k. Computed in fp32 for stability.
        """
        eps = 1e-12
        sqrt_head_dim = float(self.sqrt_head_dim)
        query_fp32 = query.to(torch.float32)
        q_norm = torch.linalg.vector_norm(query_fp32, ord=2, dim=-1, keepdim=True)
        query_fp32 = query_fp32 * torch.rsqrt(q_norm * q_norm + eps) * sqrt_head_dim

        key_fp32 = key.to(torch.float32)
        k_norm = torch.linalg.vector_norm(key_fp32, ord=2, dim=-1, keepdim=True)
        key_fp32 = key_fp32 * torch.rsqrt(k_norm * k_norm + eps) * sqrt_head_dim
        temp = self.temp.to(torch.float32).view(1, self.num_k_heads, 1)
        if self.clamp_temp:
            temp = torch.exp(torch.clamp(temp, 1e-7, 2.0))
        key_fp32 = key_fp32 * temp
        return query_fp32, key_fp32

    def _add_grouped_qk_means(
        self,
        query_conv: torch.Tensor,
        key_conv: torch.Tensor,
        query_pre: torch.Tensor,
        key_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend the post-conv q/k with the per-GQA-group mean of the
        pre-conv (raw projection) q/k, matching the ZAYA1 training formula.

        Shapes (T = num_tokens):
            query_conv : [T, num_q_heads, head_dim]      (fp32, post conv)
            key_conv   : [T, num_k_heads, head_dim]      (fp32, post conv)
            query_pre  : [T, num_q_heads, head_dim]      (raw W_q hs)
            key_base   : [T, num_k_heads, head_dim]      (raw W_k hs)
        """
        num_k_heads = key_base.shape[-2]
        key_base_fp32 = key_base.to(torch.float32)
        query_pre_grouped = query_pre.view(
            query_pre.shape[0], num_k_heads, self.gqa_groups, query_pre.shape[-1]
        )
        query_out_grouped = query_conv.view_as(query_pre_grouped).float()
        query_out_grouped = (
            query_out_grouped
            + 0.5 * query_pre_grouped.float()
            + 0.5 * key_base_fp32.unsqueeze(-2)
        )
        query_out = query_out_grouped.reshape(query_pre.shape[0], -1, query_pre.shape[-1])

        query_pre_mean = torch.mean(
            query_pre_grouped.float(), dim=-2, dtype=torch.float32
        )
        key_out = key_conv.float() + 0.5 * query_pre_mean + 0.5 * key_base_fp32
        return query_out, key_out

    def _conv_qk_run(self, padded: torch.Tensor) -> torch.Tensor:
        """Run ``conv_qk`` on ``[N, C, S + total_padding]`` → ``[N, C, S]``."""
        return self.conv_qk(padded)

    # ----- forward modes ---------------------------------------------------

    def _forward_no_state(self, hs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference path: process the entire ``hs`` of shape ``[S, H]`` with
        a zero initial conv state and a zero ``prev_hs``.

        Exercised by the CCA unit tests so the prefill / decode paths can be
        compared against a single-shot torch reference, and used as a fallback
        for profile / warmup runs where no state cache is meaningful.
        """
        S = hs.shape[0]
        hs_3d = hs.unsqueeze(1)  # [S, 1, H]

        q_raw, _ = self.linear_q(hs_3d)
        k_raw, _ = self.linear_k(hs_3d)
        qk = torch.cat([q_raw, k_raw], dim=-1)  # [S, 1, latent_q + latent_k]

        query_pre = q_raw.view(S, self.num_q_heads, self.head_dim)
        key_base = k_raw.view(S, self.num_k_heads, self.head_dim)

        # [1, C, S+pad] -> [1, C, S]
        qk_perm = qk.permute(1, 2, 0)
        qk_pad = F.pad(qk_perm, (self.total_padding, 0))
        qk_out = self._conv_qk_run(qk_pad).permute(2, 0, 1).squeeze(1)  # [S, C]

        query_conv = qk_out[:, : self.latent_q_dim].view(
            S, self.num_q_heads, self.head_dim
        )
        key_conv = qk_out[:, self.latent_q_dim :].view(
            S, self.num_k_heads, self.head_dim
        )

        query, key = self._add_grouped_qk_means(
            query_conv, key_conv, query_pre, key_base
        )
        query, key = self._normalize_qk(query, key)

        # val_proj2 uses a right-shifted hidden_state. First val_proj2 input is 0.
        hs_shifted = F.pad(hs_3d[:-1], (0, 0, 0, 0, 1, 0))  # [S, 1, H]
        v1, _ = self.val_proj1(hs_3d)
        v2, _ = self.val_proj2(hs_shifted)
        value = torch.cat([v1, v2], dim=-1).squeeze(1).view(
            S, self.num_k_heads, self.head_dim
        ).to(torch.float32)
        return query, key, value

    def _forward_extend(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefill / extend path.

        Walks every request in the batch, applies the conv with each request's
        own initial state (zero on first chunk, cached otherwise), writes the
        updated state and ``prev_hs`` back into the per-request pools, and
        returns the concatenated q/k/v in the original token layout.
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        T = hidden_states.shape[0]

        q_raw, _ = self.linear_q(hidden_states)  # [T, latent_q]
        k_raw, _ = self.linear_k(hidden_states)
        qk = torch.cat([q_raw, k_raw], dim=-1)  # [T, in_out_ch]

        query_pre = q_raw.view(T, self.num_q_heads, self.head_dim)
        key_base = k_raw.view(T, self.num_k_heads, self.head_dim)

        qk_out = torch.empty_like(qk)
        v2_input = torch.empty_like(hidden_states)

        extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu or []
        extend_prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu or []
        req_pool_indices = forward_batch.req_pool_indices

        start = 0
        for i, seq_len in enumerate(extend_seq_lens_cpu):
            end = start + int(seq_len)
            req_idx = int(req_pool_indices[i].item())
            has_prefix = int(extend_prefix_lens_cpu[i]) > 0

            qk_cur = qk[start:end].transpose(0, 1).unsqueeze(0)  # [1, C, S_cur]
            if has_prefix:
                left_pad = self.conv_state_pool[req_idx].unsqueeze(0).to(dtype)
            else:
                left_pad = qk_cur.new_zeros((1, self.in_out_ch, self.total_padding))
            padded = torch.cat([left_pad, qk_cur], dim=-1)

            out = self._conv_qk_run(padded)  # [1, C, S_cur]
            qk_out[start:end] = out.squeeze(0).transpose(0, 1)

            # Persist the tail (last `total_padding` columns) of (left_pad ‖ qk_cur)
            # as the conv state for the next chunk of this request.
            new_state = padded[..., -self.total_padding :]
            self.conv_state_pool[req_idx] = new_state.squeeze(0).to(
                self.conv_state_pool.dtype
            )

            # val_proj2 sees the hidden_state right-shifted by one within the
            # request; the very first slot is filled from the cached prev_hs (or
            # zero for the first chunk).
            hs_cur = hidden_states[start:end]
            if has_prefix:
                first = self.prev_hs_pool[req_idx].to(dtype).unsqueeze(0)
            else:
                first = hidden_states.new_zeros((1, self.hidden_size))
            shifted = torch.cat([first, hs_cur[:-1]], dim=0)
            v2_input[start:end] = shifted

            self.prev_hs_pool[req_idx] = hs_cur[-1].to(self.prev_hs_pool.dtype)

            start = end

        query_conv = qk_out[:, : self.latent_q_dim].view(
            T, self.num_q_heads, self.head_dim
        )
        key_conv = qk_out[:, self.latent_q_dim :].view(
            T, self.num_k_heads, self.head_dim
        )

        query, key = self._add_grouped_qk_means(
            query_conv, key_conv, query_pre, key_base
        )
        query, key = self._normalize_qk(query, key)

        v1, _ = self.val_proj1(hidden_states)
        v2, _ = self.val_proj2(v2_input)
        value = (
            torch.cat([v1, v2], dim=-1)
            .view(T, self.num_k_heads, self.head_dim)
            .to(torch.float32)
        )
        return query, key, value

    def _forward_decode(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-token decode path for a whole batch.

        Reads each request's cached conv state and ``prev_hs`` via
        ``index_select``, runs the conv on the small ``[T, C, total_padding+1]``
        window, and writes the updated state back via ``index_copy_``.
        """
        T = hidden_states.shape[0]
        dtype = hidden_states.dtype
        req_idx = forward_batch.req_pool_indices.to(torch.long)

        q_raw, _ = self.linear_q(hidden_states)
        k_raw, _ = self.linear_k(hidden_states)
        qk = torch.cat([q_raw, k_raw], dim=-1)  # [T, C]

        query_pre = q_raw.view(T, self.num_q_heads, self.head_dim)
        key_base = k_raw.view(T, self.num_k_heads, self.head_dim)

        left_pad = self.conv_state_pool.index_select(0, req_idx).to(dtype)
        cur = qk.unsqueeze(-1)  # [T, C, 1]
        padded = torch.cat([left_pad, cur], dim=-1)  # [T, C, total_padding+1]
        # Use the standard conv module so cuDNN/MIOpen accumulate in fp32 and
        # match the prefill path numerically on bf16 inputs. A hand-rolled
        # einsum here accumulates in the input dtype and silently drifts.
        out = self._conv_qk_run(padded)  # [T, C, 1]
        qk_out = out.squeeze(-1)  # [T, C]

        # New conv state = last total_padding columns of `padded` (the first
        # column is shifted out for the next step).
        new_state = padded[..., -self.total_padding :]
        self.conv_state_pool.index_copy_(
            0, req_idx, new_state.to(self.conv_state_pool.dtype)
        )

        query_conv = qk_out[:, : self.latent_q_dim].view(
            T, self.num_q_heads, self.head_dim
        )
        key_conv = qk_out[:, self.latent_q_dim :].view(
            T, self.num_k_heads, self.head_dim
        )

        query, key = self._add_grouped_qk_means(
            query_conv, key_conv, query_pre, key_base
        )
        query, key = self._normalize_qk(query, key)

        # val_proj2 consumes the previous hidden_state (one per request); after
        # we use it, refresh the pool with the current token's hidden_state.
        prev_hs = self.prev_hs_pool.index_select(0, req_idx).to(dtype)
        v1, _ = self.val_proj1(hidden_states)
        v2, _ = self.val_proj2(prev_hs)
        value = (
            torch.cat([v1, v2], dim=-1)
            .view(T, self.num_k_heads, self.head_dim)
            .to(torch.float32)
        )
        self.prev_hs_pool.index_copy_(
            0, req_idx, hidden_states.to(self.prev_hs_pool.dtype)
        )
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project ``hidden_states`` into ``(q, k, v)`` honoring per-request state.

        Returns float32 tensors with shapes::

            q : [T, num_q_heads, head_dim]
            k : [T, num_k_heads, head_dim]
            v : [T, num_k_heads, head_dim]
        """
        # Zero-token batches (e.g. idle DP rank, dummy forward) bypass the
        # pool entirely.
        if hidden_states.shape[0] == 0:
            zero = hidden_states.new_zeros((0,))
            return (
                zero.view(0, self.num_q_heads, self.head_dim).to(torch.float32),
                zero.view(0, self.num_k_heads, self.head_dim).to(torch.float32),
                zero.view(0, self.num_k_heads, self.head_dim).to(torch.float32),
            )

        self._ensure_pool(forward_batch)

        if forward_batch.forward_mode.is_decode_or_idle():
            return self._forward_decode(hidden_states, forward_batch)
        # EXTEND / MIXED / DRAFT_EXTEND / DLLM_EXTEND all share the prefill loop.
        return self._forward_extend(hidden_states, forward_batch)


# ---------------------------------------------------------------------------
# Attention layer (CCA QKV + rotary + RadixAttention)
# ---------------------------------------------------------------------------


class ZayaAttention(nn.Module):
    def __init__(
        self,
        config: ZayaConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_q_heads = config.num_attention_heads
        self.num_k_heads = config.num_query_groups
        self.head_dim = config.head_dim
        self.q_dim = self.num_q_heads * self.head_dim
        self.k_dim = self.num_k_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        # The HF checkpoint stores the CCA QKV projection under
        # ``self_attn.qkv.*``, so the CCA submodule is registered with that
        # exact name to keep weight loading a 1:1 key mapping.
        self.qkv = CCA(
            config=config,
            cca_num_k_heads=self.num_k_heads,
            cca_num_q_heads=self.num_q_heads,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            cca_time0=config.cca_time0,
            cca_time1=config.cca_time1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("qkv", prefix),
        )

        self.o_proj = ReplicatedLinear(
            self.q_dim,
            self.hidden_size,
            bias=bool(getattr(config, "attention_bias", False)),
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        rope_theta = float(getattr(config, "rope_theta", 1_000_000.0))
        partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 0.5))
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=int(config.max_position_embeddings),
            base=int(rope_theta),
            is_neox_style=True,
            partial_rotary_factor=partial_rotary_factor,
        )

        self.attn = RadixAttention(
            num_heads=self.num_q_heads,
            head_dim=self.head_dim,
            scaling=self.scale,
            num_kv_heads=self.num_k_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # CCA returns fp32 ``[T, heads, head_dim]`` tensors; flatten the head
        # dim and cast to the model dtype before rotary + RadixAttention.
        q, k, v = self.qkv(hidden_states, forward_batch)
        target_dtype = hidden_states.dtype
        q = q.reshape(q.shape[0], -1).to(target_dtype)
        k = k.reshape(k.shape[0], -1).to(target_dtype)
        v = v.reshape(v.shape[0], -1).to(target_dtype)

        q, k = self.rotary_emb(positions, q, k)
        # Some rotary backends (notably AITER on ROCm) hand back tensors with
        # a different stride than the input. RadixAttention's KV-store kernel
        # asserts contiguous layout, so normalize q/k/v before the attention.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# Router (EDA + MOD) and MoE block
# ---------------------------------------------------------------------------


class ZayaRouter(nn.Module):
    """ZAYA1 expert router: 3-layer MLP with optional EDA and MOD.

    EDA (Exponential Decay Averaging) adds a scaled copy of the previous MoE
    layer's router hidden_state to the current layer's input, threading state
    across MoE layers.

    MOD (Mixture of Depths) reserves the last expert slot as a "skip" expert
    whose contribution to the residual stream is just the routing probability
    times the unprocessed hidden_state, letting individual tokens bypass the
    MoE entirely when the router scores the skip expert highest.
    """

    def __init__(
        self,
        config: ZayaConfig,
        layer_id: int,
        num_moe_experts: int,
        moe_router_topk: int,
        mlp_expansion: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.router_softmax_fp32 = bool(getattr(config, "zaya_high_prec", False))

        self.use_mod = bool(getattr(config, "zaya_use_mod", False))
        self.num_experts = (num_moe_experts + 1) if self.use_mod else num_moe_experts
        self.topk = int(moe_router_topk)
        self.mlp_expansion = int(mlp_expansion)

        self.down_proj = ReplicatedLinear(
            self.hidden_size,
            self.mlp_expansion,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

        # EDA threads router state from the previous MoE layer through
        # ``router_states_scale``. The first MoE layer in the model has no
        # previous state; whether to fold it in is decided at call time based on
        # ``prev_router_hidden_states``.
        ln_eps = float(getattr(config, "norm_epsilon", 1e-5))
        self.use_eda = bool(getattr(config, "zaya_use_eda", False))
        self.rmsnorm_eda = RMSNorm(self.mlp_expansion, eps=ln_eps)
        if self.use_eda:
            self.router_states_scale = nn.Parameter(torch.ones(self.mlp_expansion))

        self.non_linearity = nn.GELU()
        self.router_mlp = nn.Sequential(
            ReplicatedLinear(
                self.mlp_expansion,
                self.mlp_expansion,
                bias=True,
                quant_config=quant_config,
                prefix=add_prefix("router_mlp.0", prefix),
            ),
            self.non_linearity,
            ReplicatedLinear(
                self.mlp_expansion,
                self.mlp_expansion,
                bias=True,
                quant_config=quant_config,
                prefix=add_prefix("router_mlp.2", prefix),
            ),
            self.non_linearity,
            ReplicatedLinear(
                self.mlp_expansion,
                self.num_experts,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("router_mlp.4", prefix),
            ),
        )

        self.register_buffer(
            "balancing_biases",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=True,
        )
        if self.use_mod:
            with torch.no_grad():
                self.balancing_biases[-1] = -1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ``hidden_states`` is ``[T, H]``.
        hs, _ = self.down_proj(hidden_states)
        if (
            self.use_eda
            and prev_router_hidden_states is not None
            and hasattr(self, "router_states_scale")
        ):
            hs = hs + prev_router_hidden_states * self.router_states_scale

        router_hidden_states_next = hs.clone()

        hs_norm = self.rmsnorm_eda(hs)

        # Step through the Sequential manually so the ``(tensor, bias)`` tuple
        # returned by each ReplicatedLinear is unpacked correctly.
        out = hs_norm
        for stage in self.router_mlp:
            if isinstance(stage, ReplicatedLinear):
                out, _ = stage(out)
            else:
                out = stage(out)
        logits = out

        if self.router_softmax_fp32:
            expert_prob = torch.softmax(logits, dim=-1, dtype=torch.float32)
        else:
            expert_prob = torch.softmax(logits, dim=-1)

        biased = expert_prob.detach().to(torch.float32) + self.balancing_biases
        _, expert_choice = torch.topk(biased, self.topk, dim=-1)

        if self.topk > 1 and self.use_mod:
            skip_idx = self.num_experts - 1
            n_mask = expert_choice == skip_idx
            cumsum_mask = torch.cumsum(n_mask, dim=-1)
            expert_choice = expert_choice.masked_fill(cumsum_mask > 0, skip_idx)

        route_prob = torch.gather(expert_prob, dim=1, index=expert_choice)
        if route_prob.dtype != hidden_states.dtype:
            route_prob = route_prob.to(hidden_states.dtype)

        return route_prob, expert_choice, router_hidden_states_next


class ZayaBlock(nn.Module):
    """ZAYA1 MoE mixer: ZayaRouter feeding FusedMoE, with optional MOD residual blend."""

    def __init__(
        self,
        config: ZayaConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.num_moe_experts = int(config.num_experts)
        self.mlp_expansion = int(config.zaya_mlp_expansion)
        self.topk = int(getattr(config, "moe_router_topk", 1))

        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > self.num_moe_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than the "
                f"number of experts {self.num_moe_experts}"
            )

        assert (
            config.activation_func == "swiglu"
        ), "ZayaBlock only supports SwiGLU activation"
        assert config.gated_linear_unit, "ZayaBlock requires gated_linear_unit=True"

        self.router = ZayaRouter(
            config=config,
            layer_id=layer_id,
            num_moe_experts=self.num_moe_experts,
            moe_router_topk=self.topk,
            mlp_expansion=self.mlp_expansion,
            quant_config=quant_config,
            prefix=add_prefix("router", prefix),
        )

        # ffn_hidden_size is the merged (gate+up) hidden dim; the per-side
        # intermediate is half.
        intermediate = int(config.ffn_hidden_size) // 2
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=self.num_moe_experts,
            top_k=self.topk,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            activation="silu",
            reduce_results=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states.new_zeros((0, self.mlp_expansion))

        probs, indices, router_hs_next = self.router(
            hidden_states, prev_router_hidden_states
        )

        topk_out = StandardTopKOutput(
            topk_weights=probs.to(hidden_states.dtype),
            topk_ids=indices.to(torch.int32),
            router_logits=probs.to(hidden_states.dtype),
        )

        if self.config.zaya_use_mod:
            # MOD: clamp the "skip expert" id (== num_moe_experts) into the
            # valid expert range so FusedMoE never indexes out of bounds; the
            # mask below decides per-token whether to actually use experts or
            # the skip path.
            clamped_ids = torch.clamp(indices, min=0, max=self.num_moe_experts - 1).to(
                torch.int32
            )
            topk_out = topk_out._replace(topk_ids=clamped_ids)

            experts_out = self.experts(hidden_states, topk_out)
            mod_out = hidden_states * probs

            if self.tp_size > 1:
                mod_out = tensor_model_parallel_all_reduce(mod_out)
                experts_out = tensor_model_parallel_all_reduce(experts_out)

            mod_mask = (indices != self.num_moe_experts).to(experts_out.dtype)
            hidden_out = mod_mask * experts_out + (1.0 - mod_mask) * mod_out
        else:
            hidden_out = self.experts(hidden_states, topk_out)
            if self.tp_size > 1:
                hidden_out = tensor_model_parallel_all_reduce(hidden_out)

        return hidden_out, router_hs_next


# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


class ZayaDecoderATTLayer(nn.Module):
    """Attention decoder layer: ``res_scale → input_norm → ZayaAttention``."""

    def __init__(
        self,
        config: ZayaConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.self_attn = ZayaAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.input_norm = self._build_norm(config)
        if config.scale_residual_merge:
            self.res_scale = ResidualScaling(config, layer_id)
        else:
            self.res_scale = None

    @staticmethod
    def _build_norm(config: ZayaConfig) -> nn.Module:
        if config.normalization == "RMSNorm":
            return RMSNorm(config.hidden_size, eps=config.norm_epsilon)
        if config.normalization == "LayerNorm":
            return nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        raise ValueError(f"Unsupported normalization: {config.normalization}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        target_dtype = (
            self.input_norm.weight.dtype
            if isinstance(self.input_norm, RMSNorm)
            else hidden_states.dtype
        )
        if self.res_scale is not None:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        if residual is not None:
            residual = residual.float() + hidden_states.float()
        else:
            residual = hidden_states.float()
        hidden_states = _apply_norm_with_fp32_residual(
            self.input_norm, residual, target_dtype
        )
        hidden_states = self.self_attn(hidden_states, positions, forward_batch)
        return hidden_states, residual, prev_router_hidden_states


class ZayaDecoderMLPLayer(nn.Module):
    """MoE decoder layer: ``res_scale → input_norm → ZayaBlock``."""

    def __init__(
        self,
        config: ZayaConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.zaya_block = ZayaBlock(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("zaya_block", prefix),
        )
        self.input_norm = ZayaDecoderATTLayer._build_norm(config)
        if config.scale_residual_merge:
            self.res_scale = ResidualScaling(config, layer_id)
        else:
            self.res_scale = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        target_dtype = (
            self.input_norm.weight.dtype
            if isinstance(self.input_norm, RMSNorm)
            else hidden_states.dtype
        )
        if self.res_scale is not None:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        if residual is not None:
            residual = residual.float() + hidden_states.float()
        else:
            residual = hidden_states.float()
        hidden_states = _apply_norm_with_fp32_residual(
            self.input_norm, residual, target_dtype
        )
        hidden_states, prev_router_hidden_states = self.zaya_block(
            hidden_states, prev_router_hidden_states
        )
        return hidden_states, residual, prev_router_hidden_states


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


def _build_layer(
    layer_id: int,
    config: ZayaConfig,
    quant_config: Optional[QuantizationConfig],
    prefix: str,
) -> nn.Module:
    # Even layer ids are attention, odd layer ids are MoE. This matches the HF
    # checkpoint keys: ``model.layers.<2k>.self_attn.*`` (CCA) versus
    # ``model.layers.<2k+1>.zaya_block.*`` (MoE).
    if layer_id % 2 == 0:
        return ZayaDecoderATTLayer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
        )
    return ZayaDecoderMLPLayer(
        config=config,
        layer_id=layer_id,
        quant_config=quant_config,
        prefix=prefix,
    )


class ZayaModel(nn.Module):
    def __init__(
        self,
        config: ZayaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: _build_layer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.final_norm = ZayaDecoderATTLayer._build_norm(config)
            if config.scale_residual_merge:
                self.res_scale = ResidualScaling(config, config.num_hidden_layers)
            else:
                self.res_scale = None
        else:
            self.final_norm = PPMissingLayer()
            self.res_scale = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        prev_router_hidden_states: Optional[torch.Tensor] = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual, prev_router_hidden_states = layer(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                forward_batch=forward_batch,
                prev_router_hidden_states=prev_router_hidden_states,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        if self.res_scale is not None:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        target_dtype = (
            self.final_norm.weight.dtype
            if isinstance(self.final_norm, RMSNorm)
            else hidden_states.dtype
        )
        if residual is not None:
            merged = hidden_states.float() + residual.float()
        else:
            merged = hidden_states.float()
        hidden_states = _apply_norm_with_fp32_residual(
            self.final_norm, merged, target_dtype
        )
        return hidden_states


class ZayaForCausalLM(nn.Module):
    def __init__(
        self,
        config: ZayaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.model = ZayaModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                bias=bool(getattr(config, "lm_head_bias", False)),
                quant_config=None,
                prefix=add_prefix("lm_head", prefix),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=inputs_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self.pp_group.is_last_rank:
            return hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    # ---------------- weight loading ----------------

    _EXPERT_RE = re.compile(
        r"^(.*\.zaya_block\.experts)\.local_experts\.(\d+)\.(linear_fc1|linear_fc2)\.weight$"
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load an HF ZAYA1 safetensors checkpoint into the SGLang module tree.

        Most keys map 1:1 because the module names already mirror the HF
        checkpoint layout. Two cases need rewriting:

        1. ``self_attn.qkv.{linear_q, linear_k, conv_qk.{0,1}, val_proj{1,2}, temp}``
           lands directly on the registered ``CCA`` submodule (which is named
           ``qkv`` exactly to keep this mapping trivial).
        2. ``zaya_block.experts.local_experts.<i>.linear_fc1.weight`` (gate
           and up projections concatenated along dim 0) is split and routed
           to FusedMoE shards ``w1`` (first half) and ``w3`` (second half);
           ``linear_fc2.weight`` becomes the FusedMoE ``w2`` shard.
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        # ``balancing_biases`` is a persistent buffer; FusedMoE may also expose
        # buffers. Expose them all through ``params_dict`` so that the regular
        # ``default_weight_loader`` can write to them.
        for key, buf in buffers_dict.items():
            params_dict.setdefault(key, buf)

        fused_moe_modules: dict[str, nn.Module] = {}
        for name, module in self.named_modules():
            if module.__class__.__name__ == "FusedMoE" or hasattr(module, "w13_weight"):
                fused_moe_modules[name] = module

        loaded_params: set[str] = set()

        for ckpt_name, loaded_weight in weights:
            # Skip keys that have no runtime counterpart in this model.
            if ckpt_name.startswith("lm_head") and self.config.tie_word_embeddings:
                continue
            if "rotary_emb" in ckpt_name:
                continue

            match = self._EXPERT_RE.match(ckpt_name)
            if match is not None:
                experts_prefix = match.group(1)  # e.g. model.layers.1.zaya_block.experts
                expert_id = int(match.group(2))
                kind = match.group(3)
                moe_module = fused_moe_modules.get(experts_prefix)
                if moe_module is None:
                    logger.warning(
                        "FusedMoE module %s not found; skipping %s",
                        experts_prefix,
                        ckpt_name,
                    )
                    continue
                weight_loader = moe_module.weight_loader
                if kind == "linear_fc1":
                    param_name = f"{experts_prefix}.w13_weight"
                    param = params_dict.get(param_name)
                    if param is None:
                        logger.warning("No param %s for %s", param_name, ckpt_name)
                        continue
                    half = loaded_weight.shape[0] // 2
                    weight_loader(
                        param,
                        loaded_weight[:half],
                        ckpt_name,
                        shard_id="w1",
                        expert_id=expert_id,
                    )
                    weight_loader(
                        param,
                        loaded_weight[half:],
                        ckpt_name,
                        shard_id="w3",
                        expert_id=expert_id,
                    )
                    loaded_params.add(param_name)
                else:  # linear_fc2
                    param_name = f"{experts_prefix}.w2_weight"
                    param = params_dict.get(param_name)
                    if param is None:
                        logger.warning("No param %s for %s", param_name, ckpt_name)
                        continue
                    weight_loader(
                        param,
                        loaded_weight,
                        ckpt_name,
                        shard_id="w2",
                        expert_id=expert_id,
                    )
                    loaded_params.add(param_name)
                continue

            # HF stores CCA tensors under ``self_attn.qkv.*``, which already
            # matches our submodule registration, so no rename is needed.
            if ckpt_name not in params_dict:
                # ``conv_qk`` is an ``nn.Sequential`` of two ``nn.Conv1d``,
                # whose keys end in ``.0.{weight,bias}`` / ``.1.{weight,bias}``
                # and are exposed through ``named_parameters()`` automatically.
                # Anything else is genuinely unknown – warn and skip.
                logger.warning(
                    "WARNING: checkpoint key %s has no matching parameter; skipping",
                    ckpt_name,
                )
                continue

            param = params_dict[ckpt_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(ckpt_name)

        return loaded_params


EntryClass = ZayaForCausalLM
