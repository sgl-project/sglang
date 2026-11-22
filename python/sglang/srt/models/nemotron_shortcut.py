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
"""Nemotron Shortcut: the GDP / SDSA hybrid text stack and the Pixtral vision tower.

Tensor parallel shards GDP and attention by head and the experts by intermediate
size; the SDSA indexer, the MoE router and latent projections are replicated.
"""

from __future__ import annotations

import functools
import logging
from array import array
from collections.abc import Iterable
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.configs.nemotron_shortcut import (
    GDP_LAYER_TYPES,
    MOE_LAYER_TYPES,
    gdp_conv_dim,
    gdp_key_dim,
    gdp_value_dim,
    variant_setting,
)
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType, get_moe_runner_backend
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import general_mm_embed_routine, pad_mm_tokens
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.nemotron_shortcut_vision import (
    NemotronShortcutMultiModalProjector,
    nemotron_shortcut_image_features,
)
from sglang.srt.models.pixtral import PixtralHFVisionModel
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


def clamped_relu2(x: torch.Tensor, clamp_value: Optional[float]) -> torch.Tensor:
    """Soft-clamped squared ReLU, relu(c * tanh(x / c))^2, with the soft clamp
    taken in FP32 as in the reference."""
    if clamp_value is not None:
        x = (clamp_value * torch.tanh(x.float() / clamp_value)).to(x.dtype)
    return torch.square(F.relu(x))


# ---------------------------------------------------------------------------
# Wide (multi-stream) residual
# ---------------------------------------------------------------------------


def _split_streams(streams: torch.Tensor, num_streams: int) -> torch.Tensor:
    return streams.view(streams.shape[0], num_streams, -1)


class NemotronShortcutWideResidualRead(nn.Module):
    """sum_k sigmoid(logit_k) * stream_k; the shortcut-MoE routed read and
    the final readout."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.num_streams = config.wide_residual_num_streams
        self.logit = nn.Parameter(torch.empty(self.num_streams, dtype=torch.float32))
        # Branches run in the model dtype also when the streams are FP32.
        self.out_dtype = torch.get_default_dtype()

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        factors = torch.sigmoid(self.logit).to(streams.dtype)
        read = torch.matmul(factors, _split_streams(streams, self.num_streams))
        return read.to(self.out_dtype)


class NemotronShortcutWideResidualConnection(nn.Module):
    """Branch read with its own coefficients; write Y_k = gamma_k X_k + 2 sigmoid(w_k) U."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.num_streams = config.wide_residual_num_streams
        self.max_forget = config.wide_residual_retention_max_forget
        shape = (self.num_streams,)
        self.read_logit = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.write_logit = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.retention_logit = (
            nn.Parameter(torch.empty(shape, dtype=torch.float32))
            if config.wide_residual_learned_retention
            else None
        )
        self.out_dtype = torch.get_default_dtype()

    def read(self, streams: torch.Tensor) -> torch.Tensor:
        factors = torch.sigmoid(self.read_logit).to(streams.dtype)
        read = torch.matmul(factors, _split_streams(streams, self.num_streams))
        return read.to(self.out_dtype)

    def write(self, streams: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        split = _split_streams(streams, self.num_streams)
        write = (2.0 * torch.sigmoid(self.write_logit)).to(streams.dtype)
        if self.retention_logit is not None:
            retention = 1.0 - self.max_forget * torch.sigmoid(-self.retention_logit)
            split = split * retention.to(streams.dtype).view(1, -1, 1)
        out = torch.addcmul(
            split, update.to(streams.dtype).unsqueeze(1), write.view(1, -1, 1)
        )
        return out.view(streams.shape)


class NemotronShortcutPlainResidual(nn.Module):
    """Ordinary additive residual behind the wide-residual read/write interface
    (the MTP block has no wide residual)."""

    def read(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def write(self, hidden_states: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return hidden_states + update.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------


class NemotronShortcutMLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.clamp_value = config.activation_clamp_value
        self.up_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up_proj(x)
        x, _ = self.down_proj(clamped_relu2(x, self.clamp_value))
        return x


class _Conv1dWeight(nn.Module):
    """Holds the depthwise conv weight under the checkpoint's `conv1d.weight` name."""

    def __init__(self, conv_dim: int, kernel: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(conv_dim, 1, kernel))


class NemotronShortcutGDPMixer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        n = config.gdp_num_householder
        tp_size = get_parallel().tp_size
        # This rank's heads; every per-head block of in_proj and conv1d is
        # sharded on its own (gdp_in_proj_blocks).
        self.num_heads = config.gdp_num_heads // tp_size
        self.value_dim = gdp_value_dim(config) // tp_size
        self.conv_dim = gdp_conv_dim(config) // tp_size
        self.g_clamp_value = variant_setting(config, "gdp_g_clamp_value")
        # in_proj rows: [z, V_0..V_{n-1}, K_0..K_{n-1}, Q, b_0..b_{n-1}, a].
        self.split_sizes = [
            self.value_dim,
            self.conv_dim,
            n * self.num_heads,
            self.num_heads,
        ]
        self.in_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=sum(rows for rows, _ in gdp_in_proj_blocks(config)),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("in_proj", prefix),
        )
        self.conv1d = _Conv1dWeight(self.conv_dim, config.gdp_conv_kernel)
        # FP32 so FlashInfer's decode kernel reads them in place.
        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.norm = RMSNormGated(
            self.value_dim,
            eps=config.gdp_norm_eps,
            group_size=config.gdp_head_dim,
            norm_before_gate=False,
            activation="swish",
        )
        self.out_proj = RowParallelLinear(
            input_size=gdp_value_dim(config),
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_heads,
            num_k_heads=self.num_heads,
            num_v_heads=self.num_heads,
            head_q_dim=config.gdp_key_head_dim,
            head_k_dim=config.gdp_key_head_dim,
            head_v_dim=config.gdp_head_dim,
            conv_weights=self.conv1d.weight.view(self.conv_dim, -1),
            bias=None,
            activation="silu",
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        projected, _ = self.in_proj(hidden_states)
        z, mixed_vkq, b, a = projected.split(self.split_sizes, dim=-1)
        if self.g_clamp_value is not None:
            a = self._clamp_decay_input(a)
        core = self.attn(forward_batch, mixed_qkv=mixed_vkq.contiguous(), a=a, b=b)
        core = core.reshape(-1, self.value_dim)
        core = self.norm(core, z.contiguous())
        out, _ = self.out_proj(core)
        return out

    def _clamp_decay_input(self, a: torch.Tensor) -> torch.Tensor:
        """g.clamp_min(-c) applied through a: the kernels form
        g = -exp(A_log) * softplus(a + dt_bias), so g >= -c is exactly
        a <= log(expm1(c * exp(-A_log))) - dt_bias."""
        bound = torch.expm1(self.g_clamp_value * torch.exp(-self.A_log))
        upper = torch.log(bound.clamp_min(torch.finfo(torch.float32).tiny))
        return torch.minimum(a.float(), upper - self.dt_bias).to(a.dtype)


def gdp_in_proj_blocks(config: PretrainedConfig) -> list[tuple[int, int]]:
    """in_proj's full-size row blocks as (rows, num_householder): each
    householder slot of a block is head-major and sharded on its own."""
    n = config.gdp_num_householder
    heads = config.gdp_num_heads
    return [
        (gdp_value_dim(config), 1),
        (n * gdp_value_dim(config), n),
        (n * gdp_key_dim(config), n),
        (gdp_key_dim(config), 1),
        (n * heads, n),
        (heads, 1),
    ]


def shard_gdp_rows(
    loaded_weight: torch.Tensor, blocks: list[tuple[int, int]], rank: int, size: int
) -> torch.Tensor:
    """This rank's heads from every householder slot of every block."""
    shards = []
    offset = 0
    for rows, householder in blocks:
        per_slot = rows // householder
        shard = per_slot // size
        for slot in range(householder):
            start = offset + slot * per_slot + rank * shard
            shards.append(loaded_weight[start : start + shard])
        offset += rows
    return torch.cat(shards, dim=0)


class NemotronShortcutAttention(nn.Module):
    """NoPE attention: position information only enters the model through the
    GDP mixers."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        use_sdsa: bool = True,
    ):
        super().__init__()
        tp_size = get_parallel().tp_size
        self.head_dim = config.head_dim
        # This rank's query heads; the one KV head is replicated.
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        # SDSA: the indexer's one-head query/key pick each query's top-k keys
        # (layers/attention/sdsa_backend.py).
        self.indexer = (
            NemotronShortcutIndexer(
                config, quant_config=quant_config, prefix=add_prefix("indexer", prefix)
            )
            if use_sdsa and config.index_topk is not None
            else None
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.indexer is None:
            out = self.attn(q, k, v, forward_batch)
        else:
            index_q, _ = self.indexer.q_proj(hidden_states)
            index_k, _ = self.indexer.k_proj(hidden_states)
            out = self.attn(
                q, k, v, forward_batch, sdsa_index_q=index_q, sdsa_index_k=index_k
            )
        out, _ = self.o_proj(out)
        return out


class NemotronShortcutIndexer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.q_proj = ReplicatedLinear(
            config.hidden_size,
            config.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ReplicatedLinear(
            config.hidden_size,
            config.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )


def route_logit_space(
    *,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """moe_router_bias_space="logit": select topk(logits + bias), weight by the
    unbiased sigmoid scores (routed_scaling_factor is applied by the caller)."""
    logits = gating_output.float()
    topk_ids = torch.topk(logits + bias, k=topk, dim=-1, sorted=False)[1]
    topk_weights = logits.gather(1, topk_ids).sigmoid()
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_weights, topk_ids.to(torch.int32)


def _check_clamped_relu2_backend() -> None:
    """The experts' soft-clamped squared ReLU runs on Marlin (W4A16), Triton (the
    BF16 MTP experts) and FlashInfer's CuTe-DSL NVFP4 / MXFP8 MoE
    (FlashInfer >= 0.7.4)."""
    backend = get_moe_runner_backend()
    if backend.is_marlin() or backend.is_triton():
        return
    if backend.is_flashinfer_cutedsl():
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
            cutedsl_supports_clamped_relu2,
        )

        if cutedsl_supports_clamped_relu2():
            return
        raise ValueError(
            "Nemotron Shortcut's soft-clamped squared-ReLU experts need "
            "FlashInfer >= 0.7.4 for "
            "--moe-runner-backend flashinfer_cutedsl."
        )
    raise ValueError(
        "Nemotron Shortcut experts use a soft-clamped squared ReLU; launch with "
        "--moe-runner-backend marlin or flashinfer_cutedsl "
        "(and --speculative-moe-runner-backend triton for MTP)."
    )


class NemotronShortcutMoE(nn.Module):
    """Latent MoE: route at model width, run the experts in the latent space,
    RMSNorm the latent sum, project back; the shared expert reads its own input."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )
        clamp_value = config.activation_clamp_value
        if clamp_value is not None:
            _check_clamped_relu2_backend()
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.moe_latent_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            activation="relu2",
            gemm1_clamp_limit=clamp_value,
            layer_id=layer_id,
            is_gated=False,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        logit_space = variant_setting(config, "moe_router_bias_space") == "logit"
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            use_grouped_topk=not logit_space,
            topk_group=None if logit_space else 1,
            num_expert_group=None if logit_space else 1,
            renormalize=config.num_experts_per_tok > 1,
            scoring_func="sigmoid",
            correction_bias=None if logit_space else self.gate.e_score_correction_bias,
            custom_routing_function=(
                functools.partial(
                    route_logit_space, bias=self.gate.e_score_correction_bias
                )
                if logit_space
                else None
            ),
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=(
                self.experts.should_fuse_routed_scaling_factor_in_topk
            ),
        )
        self.topk.bind_runner_backend(
            self.experts.runner_backend,
            routing_method_type=self.experts.routing_method_type,
        )
        self.fc1_latent_proj = ReplicatedLinear(
            config.hidden_size,
            config.moe_latent_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fc1_latent_proj", prefix),
        )
        self.fc2_norm = (
            RMSNorm(config.moe_latent_size, eps=config.rms_norm_eps)
            if config.moe_use_norm_before_up_proj
            else None
        )
        self.fc2_latent_proj = ReplicatedLinear(
            config.moe_latent_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fc2_latent_proj", prefix),
        )
        self.shared_experts = (
            NemotronShortcutMLP(
                config,
                intermediate_size=config.moe_shared_expert_intermediate_size,
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
            )
            if config.moe_shared_expert_intermediate_size is not None
            else None
        )
        # Each rank's experts give a partial latent sum, reduced before fc2_norm.
        self.reduce_routed = get_parallel().tp_size > 1

    def forward(
        self, routed_input: torch.Tensor, shared_input: torch.Tensor
    ) -> torch.Tensor:
        # Megatron routes in FP32; BF16 operands with FP32 accumulation match it.
        router_logits = torch.mm(
            routed_input, self.gate.weight.t(), out_dtype=torch.float32
        )
        topk_output = self.topk(routed_input, router_logits)
        latent, _ = self.fc1_latent_proj(routed_input)
        routed = self.experts(latent, topk_output)
        if self.reduce_routed:
            routed = tensor_model_parallel_all_reduce(routed)
        if self.fc2_norm is not None:
            routed = self.fc2_norm(routed)
        out, _ = self.fc2_latent_proj(routed)
        if self.shared_experts is not None:
            out = out + self.shared_experts(shared_input)
        return out


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def _build_mixer(
    config, layer_type, layer_id, quant_config, prefix, use_sdsa: bool = True
) -> nn.Module:
    if layer_type in GDP_LAYER_TYPES:
        return NemotronShortcutGDPMixer(
            config, layer_id, quant_config, prefix=add_prefix("mixer", prefix)
        )
    return NemotronShortcutAttention(
        config,
        layer_id,
        quant_config,
        prefix=add_prefix("self_attn", prefix),
        use_sdsa=use_sdsa,
    )


class NemotronShortcutSingleBranchLayer(nn.Module):
    """linear_attention, full_attention and mlp blocks: read, norm, branch, write."""

    def __init__(self, config, layer_type, layer_id, quant_config, prefix):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.residual_connection = NemotronShortcutWideResidualConnection(config)
        self.is_mlp = layer_type == "mlp"
        self.uses_gdp = layer_type in GDP_LAYER_TYPES
        if self.is_mlp:
            self.mlp = NemotronShortcutMLP(
                config,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            # Checkpoint names: `mixer` for GDP, `self_attn` for attention.
            mixer = _build_mixer(config, layer_type, layer_id, quant_config, prefix)
            if self.uses_gdp:
                self.mixer = mixer
            else:
                self.self_attn = mixer

    def forward(
        self, streams: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        branch_input = self.input_layernorm(self.residual_connection.read(streams))
        if self.is_mlp:
            update = self.mlp(branch_input)
        elif self.uses_gdp:
            update = self.mixer(branch_input, forward_batch)
        else:
            update = self.self_attn(branch_input, forward_batch)
        return self.residual_connection.write(streams, update)


class ShortcutMoELayer(nn.Module):
    """Mixer + latent MoE. The routed branch reads the block input (before the
    mixer); the shared branch reads the post-mixer streams."""

    def __init__(
        self,
        config,
        layer_type,
        layer_id,
        quant_config,
        prefix,
        *,
        wide_residual: bool = True,
        use_sdsa: bool = True,
    ):
        super().__init__()

        def residual():
            if wide_residual:
                return NemotronShortcutWideResidualConnection(config)
            return NemotronShortcutPlainResidual()

        self.shortcut_read = (
            NemotronShortcutWideResidualRead(config) if wide_residual else None
        )
        self.shortcut_input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer_residual_connection = residual()
        self.pre_moe_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = NemotronShortcutMoE(
            config, layer_id, quant_config, prefix=add_prefix("moe", prefix)
        )
        self.shortcut_post_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.moe_shortcut_post_norm
            else None
        )
        self.moe_residual_connection = residual()
        self.uses_gdp = layer_type == "shortcut_moe_gdp"
        mixer = _build_mixer(
            config, layer_type, layer_id, quant_config, prefix, use_sdsa=use_sdsa
        )
        if self.uses_gdp:
            self.mixer = mixer
        else:
            self.self_attn = mixer

    def forward(
        self, streams: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        routed_read = (
            streams if self.shortcut_read is None else self.shortcut_read(streams)
        )
        routed_input = self.shortcut_input_layernorm(routed_read)
        mixer_input = self.input_layernorm(self.mixer_residual_connection.read(streams))
        mixer = self.mixer if self.uses_gdp else self.self_attn
        streams = self.mixer_residual_connection.write(
            streams, mixer(mixer_input, forward_batch)
        )
        shared_input = self.pre_moe_layernorm(
            self.moe_residual_connection.read(streams)
        )
        update = self.moe(routed_input, shared_input)
        if self.shortcut_post_norm is not None:
            update = self.shortcut_post_norm(update)
        return self.moe_residual_connection.write(streams, update)


class NemotronShortcutMoEAttentionLayer(nn.Module):
    """Attention then latent MoE on a plain residual: the MTP block under
    moe_shortcut_disable_mtp. Routed and shared experts both read the
    post-attention state."""

    def __init__(self, config, layer_id, quant_config, prefix, *, use_sdsa: bool):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = NemotronShortcutAttention(
            config,
            layer_id,
            quant_config,
            prefix=add_prefix("self_attn", prefix),
            use_sdsa=use_sdsa,
        )
        self.pre_moe_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = NemotronShortcutMoE(
            config, layer_id, quant_config, prefix=add_prefix("moe", prefix)
        )

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), forward_batch
        )
        moe_input = self.pre_moe_layernorm(hidden_states)
        return hidden_states + self.moe(moe_input, moe_input)


class NemotronShortcutTextModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.config = config
        self.num_streams = config.wide_residual_num_streams
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        layers = []
        for layer_id, layer_type in enumerate(config.layer_types):
            layer_cls = (
                ShortcutMoELayer
                if layer_type in MOE_LAYER_TYPES
                else NemotronShortcutSingleBranchLayer
            )
            layers.append(
                layer_cls(
                    config,
                    layer_type,
                    layer_id,
                    quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.residual_readout = NemotronShortcutWideResidualRead(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        streams = input_embeds.repeat(1, self.num_streams)
        if self.config.residual_in_fp32:
            streams = streams.float()
        for layer in self.layers:
            streams = layer(streams, forward_batch)
        return self.norm(self.residual_readout(streams))


class NemotronShortcutModel(nn.Module):
    """Mirrors the checkpoint's `model.{language_model,vision_tower,
    multi_modal_projector}` nesting; vision_config None builds the text stack only."""

    def __init__(self, text_config, quant_config, prefix, vision_parent_config=None):
        super().__init__()
        self.language_model = NemotronShortcutTextModel(
            text_config, quant_config, prefix=add_prefix("language_model", prefix)
        )
        if vision_parent_config is not None:
            # The vision tower and projector are unquantized in the checkpoint.
            self.vision_tower = PixtralHFVisionModel(
                vision_parent_config.vision_config,
                None,
                prefix=add_prefix("vision_tower", prefix),
            )
            self.multi_modal_projector = NemotronShortcutMultiModalProjector(
                vision_parent_config
            )


class NemotronShortcutForConditionalGeneration(nn.Module):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(
        self,
        *,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        text_config = config.get_text_config()
        parallel = get_parallel()
        if parallel.attn_tp_size != parallel.tp_size or parallel.moe_ep_size != 1:
            raise NotImplementedError(
                "Nemotron Shortcut supports tensor parallel only (no DP attention, EP=1)."
            )
        self.config = config
        self.text_config = text_config
        self.quant_config = quant_config
        self.has_vision = not config.language_model_only
        self.model = NemotronShortcutModel(
            text_config,
            quant_config,
            prefix=add_prefix("model", prefix),
            vision_parent_config=config if self.has_vision else None,
        )
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(text_config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.language_model.embed_tokens

    def get_embed_and_head(self):
        return self.model.language_model.embed_tokens.weight, self.lm_head.weight

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        return pad_mm_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return nemotron_shortcut_image_features(
            vision_tower=self.model.vision_tower,
            projector=self.model.multi_modal_projector,
            items=items,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.has_vision and input_embeds is None:
            hidden_states = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.model.language_model,
                multimodal_model=self,
            )
        else:
            hidden_states = self.model.language_model(
                input_ids, forward_batch, input_embeds
            )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def _skip_weight(self, name: str) -> bool:
        is_vision = name.startswith(
            ("model.vision_tower.", "model.multi_modal_projector.")
        )
        return (
            (is_vision and not self.has_vision)
            or ".mtp." in name
            # NVFP4 KV-cache global scales; an FP8 KV cache runs at scale 1.
            or name.endswith(".k_scale")
            or name.endswith(".v_scale")
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        expert_params_mapping = nemotron_shortcut_expert_params_mapping(
            self.text_config.n_routed_experts
        )
        params_dict = dict(self.named_parameters())
        head_sharded = gdp_head_sharded_params(self, self.text_config)
        loaded: set[str] = set()
        vision_tower_weights = []
        for name, loaded_weight in weights:
            if self._skip_weight(name):
                continue
            if name.startswith("model.vision_tower."):
                vision_tower_weights.append(
                    (name.removeprefix("model.vision_tower."), loaded_weight)
                )
                continue
            load_nemotron_shortcut_weight(
                params_dict,
                name,
                loaded_weight,
                loaded,
                expert_params_mapping,
                head_sharded=head_sharded,
            )

        if vision_tower_weights:
            self.model.vision_tower.load_weights(vision_tower_weights)
            loaded.update(
                name for name in params_dict if name.startswith("model.vision_tower.")
            )
        warn_unloaded_params(params_dict, loaded)


_STACKED_QKV = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
)

# Experts stacked into one [num_experts, out, in] tensor per projection (the
# MXFP8 checkpoint, and its BF16 MTP experts): FusedMoE param and shard.
_FUSED_EXPERT_TENSORS = {
    "up_proj": ("w13_weight", "w1"),
    "up_proj_scale": ("w13_weight_scale_inv", "w1"),
    "down_proj": ("w2_weight", "w2"),
    "down_proj_scale": ("w2_weight_scale_inv", "w2"),
}


def nemotron_shortcut_expert_params_mapping(num_experts: int):
    # Non-gated experts: the checkpoint's up_proj is FusedMoE's w1.
    return FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="up_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="",
        num_experts=num_experts,
    )


def gdp_head_sharded_params(
    model: nn.Module, config: PretrainedConfig
) -> dict[str, list[tuple[int, int]]]:
    """GDP parameters loaded through shard_gdp_rows under TP, with their
    full-size row blocks: in_proj and the block scales that follow its rows,
    conv1d, and the per-head A_log, dt_bias and gated-norm weight."""
    if get_parallel().tp_size == 1:
        return {}
    in_proj_blocks = gdp_in_proj_blocks(config)
    heads = [(config.gdp_num_heads, 1)]
    sharded = {}
    for prefix, module in model.named_modules():
        if not isinstance(module, NemotronShortcutGDPMixer):
            continue
        rows = module.in_proj.weight.shape[0]
        for name, param in module.in_proj.named_parameters():
            if param.dim() == 2 and param.shape[0] == rows:
                sharded[f"{prefix}.in_proj.{name}"] = in_proj_blocks
        sharded[f"{prefix}.conv1d.weight"] = in_proj_blocks[1:4]
        sharded[f"{prefix}.A_log"] = heads
        sharded[f"{prefix}.dt_bias"] = heads
        sharded[f"{prefix}.norm.weight"] = [(gdp_value_dim(config), 1)]
    return sharded


def _fp8_linear_scale_name(params_dict, name: str) -> str:
    """MXFP8 linear block scales: the checkpoint's weight_scale is
    Fp8LinearMethod's weight_scale_inv."""
    if name.endswith(".weight_scale") and name not in params_dict:
        return name + "_inv"
    return name


def load_nemotron_shortcut_weight(
    params_dict,
    name,
    loaded_weight,
    loaded,
    expert_params_mapping,
    head_sharded: Optional[dict[str, list[tuple[int, int]]]] = None,
) -> None:
    """Load one checkpoint tensor whose name already matches the module tree,
    stacking q/k/v into qkv_proj and routing per-expert tensors into FusedMoE."""
    if ".self_attn." in name and ".indexer." not in name:
        for param_name, weight_name, shard_id in _STACKED_QKV:
            if f".{weight_name}." in name:
                mapped = _fp8_linear_scale_name(
                    params_dict, name.replace(weight_name, param_name)
                )
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(mapped)
                return
    if ".moe.experts." in name:
        module, _, tensor = name.rpartition(".moe.experts.")
        if tensor in _FUSED_EXPERT_TENSORS:
            param_suffix, shard_id = _FUSED_EXPERT_TENSORS[tensor]
            mapped = f"{module}.moe.experts.{param_suffix}"
            param = params_dict[mapped]
            for expert_id in range(loaded_weight.shape[0]):
                param.weight_loader(
                    param,
                    loaded_weight[expert_id],
                    mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
            loaded.add(mapped)
            return
        for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
            if weight_name in name:
                mapped = name.replace(weight_name, param_name)
                param = params_dict[mapped]
                param.weight_loader(
                    param, loaded_weight, mapped, shard_id=shard_id, expert_id=expert_id
                )
                loaded.add(mapped)
                return
        raise KeyError(f"Unmapped Nemotron Shortcut expert weight: {name}")
    name = _fp8_linear_scale_name(params_dict, name)
    if name not in params_dict:
        raise KeyError(f"Unexpected Nemotron Shortcut checkpoint weight: {name}")
    param = params_dict[name]
    if head_sharded and name in head_sharded:
        param.data.copy_(
            shard_gdp_rows(
                loaded_weight,
                head_sharded[name],
                rank=get_parallel().tp_rank,
                size=get_parallel().tp_size,
            ).view(param.shape)
        )
        loaded.add(name)
        return
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight)
    loaded.add(name)


def warn_unloaded_params(params_dict, loaded) -> None:
    missing = sorted(set(params_dict) - loaded)
    if missing:
        logger.warning(
            "Nemotron Shortcut: %d parameters not loaded from the checkpoint, e.g. %s",
            len(missing),
            missing[:8],
        )


class Nemotron4ForConditionalGeneration(NemotronShortcutForConditionalGeneration):
    """The checkpoint's architecture name."""


EntryClass = [Nemotron4ForConditionalGeneration]
