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
- Per-request CCA state (``conv_state`` + ``prev_hs``) is managed by
  SGLang's centralized ``MambaPool`` inside ``HybridReqToTokenPool``,
  accessed via ``get_req_to_token_pool().mamba2_layer_cache()``.
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
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
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
from sglang.srt.model_executor.forward_context import get_req_to_token_pool
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, make_layers, set_weight_attrs

logger = logging.getLogger(__name__)


# Attribute names used to memoize the per-request MambaPool slot indices on the
# ForwardBatch. The req -> slot mapping is identical for every CCA layer in a
# step, so caching it here makes the lookup (and its GPU->CPU sync) run once per
# forward step instead of once per attention layer.
_MAMBA_INDICES_ATTR = "_zaya_mamba_indices"
_MAMBA_INDICES_CPU_ATTR = "_zaya_mamba_indices_cpu"


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

    The fp32 residual stream is preserved by the caller (the residual tensor
    is kept around for the next accumulation), so the norm itself can run at
    ``target_dtype`` -- this lets us hit the fused sgl_kernel rmsnorm path
    instead of the eager ``forward_native`` fallback (5+ kernel launches per
    call, ×120 norms per step).
    """
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

    Parallelism: when ``tp_size > 1`` the CCA is head-parallel. Both the
    grouped-mean step and the second ``conv_qk`` stage with
    ``groups=num_q_heads+num_k_heads`` are head-local (each GQA group lives on
    a single rank), so the entire QKV projection runs without any cross-rank
    collective. The QKV projections become ``ColumnParallelLinear`` and the
    two ``nn.Conv1d`` layers are sized per-rank with custom weight loaders
    that slice the HF checkpoint rows into ``[rank's q heads, rank's k heads]``.
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
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
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

        if tp_rank is None:
            tp_rank = get_parallel().tp_rank
        if tp_size is None:
            tp_size = get_parallel().tp_size
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)

        # Full (global) head counts retained for weight loading and shape asserts.
        self.num_q_heads_full = int(cca_num_q_heads)
        self.num_k_heads_full = int(cca_num_k_heads)
        assert (
            self.num_q_heads_full % self.num_k_heads_full == 0
        ), "num_q_heads must be a multiple of num_k_heads"
        self.gqa_groups = self.num_q_heads_full // self.num_k_heads_full

        # Head-parallel TP requires both head counts to be divisible by tp_size.
        # KV-replication-style TP (tp_size > num_k_heads) is not yet supported.
        assert self.num_q_heads_full % self.tp_size == 0, (
            f"num_q_heads ({self.num_q_heads_full}) must be divisible by "
            f"tp_size ({self.tp_size}) for ZAYA1 head-parallel CCA"
        )
        assert self.num_k_heads_full % self.tp_size == 0, (
            f"num_k_heads ({self.num_k_heads_full}) must be divisible by "
            f"tp_size ({self.tp_size}); KV-replication TP is not supported "
            "for ZAYA1 because both grouped-mean and conv_qk.1 are per-head"
        )

        # Per-rank head counts.
        self.num_q_heads = self.num_q_heads_full // self.tp_size
        self.num_k_heads = self.num_k_heads_full // self.tp_size

        # Per-rank channel layout.
        self.latent_q_dim_full = self.num_q_heads_full * self.head_dim
        self.latent_k_dim_full = self.num_k_heads_full * self.head_dim
        self.in_out_ch_full = self.latent_q_dim_full + self.latent_k_dim_full
        self.latent_q_dim = self.num_q_heads * self.head_dim
        self.latent_k_dim = self.num_k_heads * self.head_dim
        self.in_out_ch = self.latent_q_dim + self.latent_k_dim
        self.sqrt_head_dim = float(self.head_dim) ** 0.5
        self.clamp_temp = bool(getattr(config, "clamp_temp", False))

        bias = bool(getattr(config, "attention_bias", False))
        # ``linear_q`` / ``linear_k`` outputs are laid out as a contiguous head
        # sequence in the HF checkpoint, so the natural ColumnParallel shard
        # (``tp_rank * shard``) lands rank ``r`` on the head set
        # ``[r * heads_per_rank, (r+1) * heads_per_rank)``.
        #
        # At ``tp_size == 1`` there is nothing to shard, and on ROCm/aiter the
        # ColumnParallelLinear path selects a slower GEMM for the large-M prefill
        # (1.6-2.25x slower than ReplicatedLinear in bench_one_batch), so the
        # single-GPU case uses ReplicatedLinear. ``tp_size > 1`` keeps
        # ColumnParallelLinear for the per-rank head shard.
        if self.tp_size > 1:
            self.linear_q = ColumnParallelLinear(
                self.hidden_size,
                self.latent_q_dim_full,
                bias=bias,
                gather_output=False,
                quant_config=quant_config,
                prefix=add_prefix("linear_q", prefix),
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            )
            self.linear_k = ColumnParallelLinear(
                self.hidden_size,
                self.latent_k_dim_full,
                bias=bias,
                gather_output=False,
                quant_config=quant_config,
                prefix=add_prefix("linear_k", prefix),
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            )
        else:
            self.linear_q = ReplicatedLinear(
                self.hidden_size,
                self.latent_q_dim_full,
                bias=bias,
                quant_config=quant_config,
                prefix=add_prefix("linear_q", prefix),
            )
            self.linear_k = ReplicatedLinear(
                self.hidden_size,
                self.latent_k_dim_full,
                bias=bias,
                quant_config=quant_config,
                prefix=add_prefix("linear_k", prefix),
            )
        # The HF V-projection layout maps val_proj1 to the FIRST half of K
        # heads and val_proj2 to the SECOND half (after ``cat([v1, v2]).view(
        # T, num_k_heads_full, head_dim)``). That doesn't align with a simple
        # output-dim ColumnParallel shard, so val_proj1 / val_proj2 are kept
        # Replicated and the per-rank K-head slice is taken in the forward
        # passes after ``cat + view``. The replicated weight memory is small
        # (~0.5 MB / layer) and the wasted compute is negligible compared to
        # linear_q / linear_k / o_proj.
        self.val_proj1 = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim_full // 2,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("val_proj1", prefix),
        )
        self.val_proj2 = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim_full // 2,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("val_proj2", prefix),
        )

        # Per-rank K head range, used for slicing the replicated v tensor.
        self.k_head_start = self.tp_rank * self.num_k_heads
        self.k_head_end = self.k_head_start + self.num_k_heads

        # Two-stage depthwise + grouped conv along the time axis, sized for
        # this rank's head subset. Wrapping the two nn.Conv1d modules in
        # nn.Sequential makes the HF checkpoint keys ``conv_qk.{0,1}.weight``
        # / ``conv_qk.{0,1}.bias`` map onto submodules 1:1, with TP slicing
        # handled by the custom weight_loader attached below.
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

        # Per-K-head learnable temperature scalar (per-rank slice).
        self.temp = nn.Parameter(torch.zeros(self.num_k_heads))

        # Attach TP-aware weight loaders to conv_qk weights/biases and ``temp``
        # so the existing ``load_weights`` dispatch (``getattr(param,
        # "weight_loader", default_weight_loader)``) automatically slices the
        # HF checkpoint into rank-local rows.
        if self.tp_size > 1:
            self._install_tp_weight_loaders()

    # ----- TP weight loaders ----------------------------------------------

    def _install_tp_weight_loaders(self) -> None:
        """Attach TP-aware ``weight_loader`` attributes to parameters whose
        full-tensor → per-rank slicing cannot be expressed by a generic
        ColumnParallelLinear loader: the two ``conv_qk`` Conv1d weights and
        biases (where the per-rank "row" set is the discontiguous union of
        this rank's q heads and this rank's k heads) and the per-K-head
        ``temp`` parameter.
        """
        head_dim = self.head_dim
        latent_q_dim_full = self.latent_q_dim_full
        num_q_heads_per_rank = self.num_q_heads
        num_k_heads_per_rank = self.num_k_heads
        tp_rank = self.tp_rank

        q_start = tp_rank * num_q_heads_per_rank * head_dim
        q_end = q_start + num_q_heads_per_rank * head_dim
        k_start = latent_q_dim_full + tp_rank * num_k_heads_per_rank * head_dim
        k_end = k_start + num_k_heads_per_rank * head_dim
        k_temp_start = tp_rank * num_k_heads_per_rank
        k_temp_end = k_temp_start + num_k_heads_per_rank

        def conv_row_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            # Both Conv1d.weight ([C_out, in_per_group, K]) and Conv1d.bias
            # ([C_out]) slice along the leading (output channel) dim. The
            # per-rank rows are the rank's q heads (contiguous) followed by
            # the rank's k heads (contiguous in the second half of the full
            # tensor).
            sliced = torch.cat(
                [loaded_weight[q_start:q_end], loaded_weight[k_start:k_end]],
                dim=0,
            )
            assert (
                sliced.shape == param.data.shape
            ), f"conv shard shape mismatch: {sliced.shape} vs {param.data.shape}"
            param.data.copy_(sliced)

        def temp_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            sliced = loaded_weight[k_temp_start:k_temp_end]
            assert (
                sliced.shape == param.data.shape
            ), f"temp shard shape mismatch: {sliced.shape} vs {param.data.shape}"
            param.data.copy_(sliced)

        set_weight_attrs(self.conv_qk[0].weight, {"weight_loader": conv_row_loader})
        set_weight_attrs(self.conv_qk[0].bias, {"weight_loader": conv_row_loader})
        set_weight_attrs(self.conv_qk[1].weight, {"weight_loader": conv_row_loader})
        set_weight_attrs(self.conv_qk[1].bias, {"weight_loader": conv_row_loader})
        set_weight_attrs(self.temp, {"weight_loader": temp_loader})

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _get_mamba_indices(forward_batch: ForwardBatch) -> torch.Tensor:
        """Per-request MambaPool slot indices as an int64 device tensor.

        The req -> slot mapping depends only on ``forward_batch.req_pool_indices``,
        which is constant for every CCA layer within one forward step. Computing
        it inside each of the ~60 attention layers would issue one redundant
        gather per layer, so it is computed once and memoized on the ForwardBatch
        (whose lifetime is exactly one forward step). The lookup is pure on-device
        work, so this stays compatible with CUDA graph capture on the decode path.
        """
        cached = getattr(forward_batch, _MAMBA_INDICES_ATTR, None)
        if cached is None:
            cached = (
                get_req_to_token_pool()
                .get_mamba_indices(forward_batch.req_pool_indices)
                .to(torch.long)
            )
            setattr(forward_batch, _MAMBA_INDICES_ATTR, cached)
        return cached

    @staticmethod
    def _get_mamba_indices_cpu(
        forward_batch: ForwardBatch, mamba_indices: torch.Tensor
    ) -> list[int]:
        """Host mirror of :meth:`_get_mamba_indices`, memoized per forward step.

        Only the extend/prefill path needs the indices on the host to drive its
        per-request Python loop; the decode path indexes the pool entirely
        on-device. Memoizing turns the previous one-``.tolist()``-sync-per-layer
        behavior into a single GPU->CPU sync per forward step. This helper is
        never reached on the decode path that CUDA graphs capture.
        """
        cached = getattr(forward_batch, _MAMBA_INDICES_CPU_ATTR, None)
        if cached is None:
            cached = mamba_indices.tolist()
            setattr(forward_batch, _MAMBA_INDICES_CPU_ATTR, cached)
        return cached

    def _get_pool_state(self, forward_batch: ForwardBatch):
        """Retrieve per-request CCA state from the centralized MambaPool.

        ``conv_state`` / ``prev_hs_state`` are layer-local pool views, but the
        ``mamba_indices`` req -> slot mapping is shared across layers and so is
        memoized on the ForwardBatch (see :meth:`_get_mamba_indices`).
        """
        layer_cache = get_req_to_token_pool().mamba2_layer_cache(self.layer_id)
        conv_state = layer_cache.conv[0]
        prev_hs_state = layer_cache.conv[1]
        mamba_indices = self._get_mamba_indices(forward_batch)
        return conv_state, prev_hs_state, mamba_indices

    def _normalize_qk(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm (no learnable weight) + sqrt(head_dim) scaling on q and k,
        plus per-K-head temperature on k. Computed in fp32 for stability.
        """
        eps = 1e-12
        sqrt_head_dim = float(self.sqrt_head_dim)
        query_fp32 = query.to(torch.float32)
        inv_q = (
            torch.rsqrt(query_fp32.pow(2).sum(-1, keepdim=True) + eps) * sqrt_head_dim
        )
        query_fp32 = query_fp32 * inv_q

        key_fp32 = key.to(torch.float32)
        inv_k = torch.rsqrt(key_fp32.pow(2).sum(-1, keepdim=True) + eps) * sqrt_head_dim
        key_fp32 = key_fp32 * inv_k
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
        query_pre_grouped_fp32 = query_pre_grouped.to(torch.float32)
        query_out_grouped = (
            query_conv.view_as(query_pre_grouped).to(torch.float32)
            + 0.5 * query_pre_grouped_fp32
            + 0.5 * key_base_fp32.unsqueeze(-2)
        )
        query_out = query_out_grouped.reshape(
            query_pre.shape[0], -1, query_pre.shape[-1]
        )

        query_pre_mean = query_pre_grouped_fp32.mean(dim=-2, dtype=torch.float32)
        key_out = (
            key_conv.to(torch.float32) + 0.5 * query_pre_mean + 0.5 * key_base_fp32
        )
        return query_out, key_out

    def _conv_qk_run(self, padded: torch.Tensor) -> torch.Tensor:
        """Run ``conv_qk`` on ``[N, C, S + total_padding]`` → ``[N, C, S]``."""
        return self.conv_qk(padded)

    # ----- forward modes ---------------------------------------------------

    def _slice_v_per_rank(self, value_full: torch.Tensor) -> torch.Tensor:
        """Take this rank's K-head slice of the full ``value`` tensor.

        Returns a no-op view when ``tp_size == 1``. For ``tp_size > 1`` the
        full V tensor is computed on every rank (see the comment on
        ``val_proj1`` / ``val_proj2``) and the rank's contiguous K-head range
        is selected here, leaving the downstream RadixAttention call with a
        per-rank shape ``[T, num_k_heads_per_rank, head_dim]``.
        """
        if self.tp_size == 1:
            return value_full
        return value_full[:, self.k_head_start : self.k_head_end, :].contiguous()

    def _forward_no_state(
        self, hs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference path: process the entire ``hs`` of shape ``[S, H]`` with
        a zero initial conv state and a zero ``prev_hs``.

        Exercised by the CCA unit tests so the prefill / decode paths can be
        compared against a single-shot torch reference, and used as a fallback
        for profile / warmup runs where no state cache is meaningful.
        """
        S = hs.shape[0]
        hs_3d = hs.unsqueeze(1)  # [S, 1, H]

        q_raw, _ = self.linear_q(hs_3d)  # [S, 1, latent_q_dim_per_rank]
        k_raw, _ = self.linear_k(hs_3d)  # [S, 1, latent_k_dim_per_rank]
        qk = torch.cat([q_raw, k_raw], dim=-1)  # [S, 1, in_out_ch_per_rank]

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

        # val_proj1 / val_proj2 are replicated; compute the full V tensor and
        # then take this rank's K-head slice.
        # val_proj2 uses a right-shifted hidden_state. First val_proj2 input is 0.
        hs_shifted = F.pad(hs_3d[:-1], (0, 0, 0, 0, 1, 0))  # [S, 1, H]
        v1, _ = self.val_proj1(hs_3d)
        v2, _ = self.val_proj2(hs_shifted)
        value_full = (
            torch.cat([v1, v2], dim=-1)
            .squeeze(1)
            .view(S, self.num_k_heads_full, self.head_dim)
        )
        value = self._slice_v_per_rank(value_full)
        return query, key, value

    def _forward_extend(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefill / extend path.

        Walks every request in the batch, applies the conv with each request's
        own initial state (zero on first chunk, cached otherwise), writes the
        updated state and ``prev_hs`` back into the centralized MambaPool, and
        returns the concatenated q/k/v in the original token layout.
        """
        dtype = hidden_states.dtype
        T = hidden_states.shape[0]

        q_raw, _ = self.linear_q(hidden_states)  # [T, latent_q]
        k_raw, _ = self.linear_k(hidden_states)
        qk = torch.cat([q_raw, k_raw], dim=-1)  # [T, in_out_ch]

        query_pre = q_raw.view(T, self.num_q_heads, self.head_dim)
        key_base = k_raw.view(T, self.num_k_heads, self.head_dim)

        qk_out = torch.empty_like(qk)
        v2_input = torch.empty_like(hidden_states)

        conv_state, prev_hs_state, mamba_indices = self._get_pool_state(forward_batch)
        # Host view of the slot indices to drive the per-request loop below.
        # Memoized on the ForwardBatch, so the GPU->CPU sync runs once per forward
        # step rather than once per attention layer (~60 syncs/step otherwise).
        mamba_idx_cpu = self._get_mamba_indices_cpu(forward_batch, mamba_indices)

        extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        extend_prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu

        # Fresh-prefill fast path: when no request has a cached prefix the
        # per-request convs (one launch each, ×60 attention layers ×B requests)
        # can be coalesced into a single packed convolution. The conv chain is
        # two ``kernel_size=2`` convs (effective receptive field = 3), so each
        # request's S valid outputs are produced from the packed positions
        # ``[a_i, a_i + S_i - 1]`` where the input segment for request i is
        # ``[pad, pad, x_0, ..., x_{S-1}]`` of length ``S_i + total_padding``.
        all_fresh = bool(extend_seq_lens_cpu) and not any(
            int(p) > 0 for p in extend_prefix_lens_cpu
        )

        if all_fresh:
            seq_lens = [int(s) for s in extend_seq_lens_cpu]
            pad = self.total_padding
            # Build packed buffer: per request -> [pad zeros, S_i tokens].
            offsets_in = [0]
            for s in seq_lens:
                offsets_in.append(offsets_in[-1] + s + pad)
            packed = qk.new_zeros((1, self.in_out_ch, offsets_in[-1]))
            start = 0
            for i, s in enumerate(seq_lens):
                end = start + s
                packed[0, :, offsets_in[i] + pad : offsets_in[i + 1]] = qk[
                    start:end
                ].transpose(0, 1)
                start = end

            packed_out = self._conv_qk_run(packed)  # [1, C, offsets_in[-1] - pad]

            start = 0
            for i, s in enumerate(seq_lens):
                end = start + s
                a_i = offsets_in[i]
                qk_out[start:end] = packed_out[0, :, a_i : a_i + s].transpose(0, 1)
                new_state = packed[0, :, a_i + s : a_i + s + pad]
                conv_state[mamba_idx_cpu[i]] = new_state.to(conv_state.dtype)

                hs_cur = hidden_states[start:end]
                first = hidden_states.new_zeros((1, self.hidden_size))
                v2_input[start:end] = torch.cat([first, hs_cur[:-1]], dim=0)
                prev_hs_state[mamba_idx_cpu[i]] = (
                    hs_cur[-1].unsqueeze(-1).to(prev_hs_state.dtype)
                )
                start = end
        else:
            start = 0
            for i, seq_len in enumerate(extend_seq_lens_cpu):
                end = start + int(seq_len)
                mamba_idx = mamba_idx_cpu[i]
                has_prefix = int(extend_prefix_lens_cpu[i]) > 0

                qk_cur = qk[start:end].transpose(0, 1).unsqueeze(0)  # [1, C, S_cur]
                if has_prefix:
                    left_pad = conv_state[mamba_idx].unsqueeze(0).to(dtype)
                else:
                    left_pad = qk_cur.new_zeros((1, self.in_out_ch, self.total_padding))
                padded = torch.cat([left_pad, qk_cur], dim=-1)

                out = self._conv_qk_run(padded)  # [1, C, S_cur]
                qk_out[start:end] = out.squeeze(0).transpose(0, 1)

                new_state = padded[..., -self.total_padding :]
                conv_state[mamba_idx] = new_state.squeeze(0).to(conv_state.dtype)

                hs_cur = hidden_states[start:end]
                if has_prefix:
                    first = prev_hs_state[mamba_idx].squeeze(-1).to(dtype).unsqueeze(0)
                else:
                    first = hidden_states.new_zeros((1, self.hidden_size))
                shifted = torch.cat([first, hs_cur[:-1]], dim=0)
                v2_input[start:end] = shifted

                prev_hs_state[mamba_idx] = (
                    hs_cur[-1].unsqueeze(-1).to(prev_hs_state.dtype)
                )

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
        value_full = torch.cat([v1, v2], dim=-1).view(
            T, self.num_k_heads_full, self.head_dim
        )
        value = self._slice_v_per_rank(value_full)
        return query, key, value

    def _forward_decode(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-token decode path for a whole batch.

        Reads each request's cached conv state and ``prev_hs`` from the
        centralized MambaPool via ``index_select``, runs the conv on the
        small ``[T, C, total_padding+1]`` window, and writes back via
        ``index_copy_``.
        """
        T = hidden_states.shape[0]
        dtype = hidden_states.dtype

        conv_state, prev_hs_state, mamba_indices = self._get_pool_state(forward_batch)

        q_raw, _ = self.linear_q(hidden_states)
        k_raw, _ = self.linear_k(hidden_states)
        qk = torch.cat([q_raw, k_raw], dim=-1)  # [T, C]

        query_pre = q_raw.view(T, self.num_q_heads, self.head_dim)
        key_base = k_raw.view(T, self.num_k_heads, self.head_dim)

        left_pad = conv_state.index_select(0, mamba_indices).to(dtype)
        cur = qk.unsqueeze(-1)  # [T, C, 1]
        padded = torch.cat([left_pad, cur], dim=-1)  # [T, C, total_padding+1]
        out = self._conv_qk_run(padded)  # [T, C, 1]
        qk_out = out.squeeze(-1)  # [T, C]

        new_state = padded[..., -self.total_padding :]
        conv_state.index_copy_(0, mamba_indices, new_state.to(conv_state.dtype))

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

        prev_hs = prev_hs_state.index_select(0, mamba_indices).squeeze(-1).to(dtype)
        v1, _ = self.val_proj1(hidden_states)
        v2, _ = self.val_proj2(prev_hs)
        value_full = torch.cat([v1, v2], dim=-1).view(
            T, self.num_k_heads_full, self.head_dim
        )
        value = self._slice_v_per_rank(value_full)
        prev_hs_state.index_copy_(
            0, mamba_indices, hidden_states.unsqueeze(-1).to(prev_hs_state.dtype)
        )
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project ``hidden_states`` into ``(q, k, v)`` honoring per-request state.

        ``q`` / ``k`` are returned in fp32 (the normalize step keeps fp32 for
        stability); ``v`` is returned in the input dtype since the caller
        casts everything back to ``hidden_states.dtype`` before rotary +
        attention anyway.

        Shapes::

            q : [T, num_q_heads, head_dim]
            k : [T, num_k_heads, head_dim]
            v : [T, num_k_heads, head_dim]
        """
        if hidden_states.shape[0] == 0:
            zero = hidden_states.new_zeros((0,))
            return (
                zero.view(0, self.num_q_heads, self.head_dim).to(torch.float32),
                zero.view(0, self.num_k_heads, self.head_dim).to(torch.float32),
                zero.view(0, self.num_k_heads, self.head_dim),
            )

        if forward_batch.forward_mode.is_decode_or_idle():
            return self._forward_decode(hidden_states, forward_batch)
        # EXTEND / MIXED / DLLM_EXTEND all share the prefill loop.
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
        self.num_q_heads_full = config.num_attention_heads
        self.num_k_heads_full = config.num_query_groups
        self.head_dim = config.head_dim

        # Head-parallel TP: split both Q and KV heads across ranks. Since the
        # grouped-mean and conv_qk.1 are head-local, no cross-rank collective
        # is required inside the QKV projection. Both head counts must be
        # divisible by tp_size; the KV-replicated GQA-TP variant (tp_size >
        # num_k_heads) is intentionally rejected with a clear error message
        # because both per-K-head paths assume each rank holds whole K heads.
        self.tp_rank = get_parallel().tp_rank
        self.tp_size = get_parallel().tp_size
        # The head split, the ``o_proj`` RowParallel all-reduce, and the
        # RadixAttention KV cache are all organized on the *global* TP group,
        # and ``ZayaConfig.mamba2_cache_params`` sizes the conv-state cache on
        # that same group. DP attention would run attention on the smaller
        # attention-TP group (and ``o_proj`` would need
        # ``use_dp_attention_reduce``), which this model does not wire up, so
        # require the two groups to coincide and fail fast instead of silently
        # mis-sizing the conv-state cache.
        attn_tp_size = get_parallel().attn_tp_size
        assert attn_tp_size == self.tp_size, (
            f"ZAYA1 head-parallel attention requires the attention TP group "
            f"({attn_tp_size}) to equal the global TP group ({self.tp_size}); "
            "DP attention (enable_dp_attention) is not supported for ZAYA1."
        )
        assert self.num_q_heads_full % self.tp_size == 0, (
            f"num_attention_heads ({self.num_q_heads_full}) must be divisible "
            f"by tp_size ({self.tp_size}) for ZAYA1 head-parallel attention"
        )
        assert self.num_k_heads_full % self.tp_size == 0, (
            f"num_query_groups ({self.num_k_heads_full}) must be divisible by "
            f"tp_size ({self.tp_size}); set tp_size <= num_k_heads to keep "
            "both grouped-mean and conv_qk.1 head-local on each rank"
        )
        self.num_q_heads = self.num_q_heads_full // self.tp_size
        self.num_k_heads = self.num_k_heads_full // self.tp_size
        self.q_dim_full = self.num_q_heads_full * self.head_dim
        self.scale = self.head_dim**-0.5

        # The HF checkpoint stores the CCA QKV projection under
        # ``self_attn.qkv.*``, so the CCA submodule is registered with that
        # exact name to keep weight loading a 1:1 key mapping.
        self.qkv = CCA(
            config=config,
            cca_num_k_heads=self.num_k_heads_full,
            cca_num_q_heads=self.num_q_heads_full,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            cca_time0=config.cca_time0,
            cca_time1=config.cca_time1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("qkv", prefix),
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )

        # RowParallel o_proj: per-rank input is the rank's q heads, full
        # output is replicated via the end-of-forward all-reduce.
        self.o_proj = RowParallelLinear(
            self.q_dim_full,
            self.hidden_size,
            bias=bool(getattr(config, "attention_bias", False)),
            input_is_parallel=True,
            reduce_results=True,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
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
        # CCA returns fp32 q/k and input-dtype v as ``[T, heads, head_dim]``
        # tensors; flatten the head dim and cast all to the model dtype before
        # rotary + RadixAttention.
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

        # ``hs`` is a freshly-allocated tensor (output of ``down_proj`` or the
        # EDA add above) and ``rmsnorm_eda`` is non-residual / out-of-place,
        # so we can hand the same buffer to the next layer without cloning.
        router_hidden_states_next = hs

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


def mod_premask_experts(
    experts_out: torch.Tensor,
    indices: torch.Tensor,
    num_moe_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask the (per-rank, pre-all-reduce) expert output for the MOD skip path.

    Returns ``(mod_mask, masked_experts)`` where ``mod_mask`` is ``1`` for
    tokens routed to a real expert and ``0`` for tokens routed to the skip
    slot (``indices == num_moe_experts``), and
    ``masked_experts = mod_mask * experts_out``.

    The masking is applied *before* the cross-rank all-reduce so the single
    reduction yields ``mask · sum_r(partial_r) = mask · experts_out_full``
    without the replicated ``mod_out`` term being summed ``tp_size`` times.
    Pairs with :func:`mod_blend`, which adds the skip-path term back after the
    reduce. Kept as a free function so the MOD math is unit-testable without a
    live ``torch.distributed`` group.
    """
    mod_mask = (indices != num_moe_experts).to(experts_out.dtype)
    return mod_mask, mod_mask * experts_out


def mod_blend(
    masked_experts_reduced: torch.Tensor,
    mod_mask: torch.Tensor,
    mod_out: torch.Tensor,
) -> torch.Tensor:
    """Combine the already-all-reduced masked expert output with the skip path.

    ``mod_out`` (the skip-expert residual, ``hidden_states * prob``) is
    replicated on every rank, so it is folded in here -- after the reduce of
    ``masked_experts`` -- weighted by ``(1 - mod_mask)``. See
    :func:`mod_premask_experts`.
    """
    return masked_experts_reduced + (1.0 - mod_mask) * mod_out


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

        self.tp_size = get_parallel().tp_size
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
            # ``mod_out`` is computed identically on every TP rank (both
            # ``hidden_states`` and ``probs`` are replicated). Fold the skip
            # mask into the per-rank partial experts output *before*
            # all-reduce so the single reduction yields:
            #   sum_r(mask · partial_r) + (1 - mask) · mod_out
            # = mask · experts_out_full + (1 - mask) · mod_out
            # without double-counting ``mod_out`` by tp_size. The two steps are
            # ``mod_premask_experts`` / ``mod_blend`` so the math is testable
            # without a live distributed group.
            mod_out = hidden_states * probs
            mod_mask, masked_experts = mod_premask_experts(
                experts_out, indices, self.num_moe_experts
            )
            if self.tp_size > 1:
                masked_experts = tensor_model_parallel_all_reduce(masked_experts)
            hidden_out = mod_blend(masked_experts, mod_mask, mod_out)
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
                experts_prefix = match.group(
                    1
                )  # e.g. model.layers.1.zaya_block.experts
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
