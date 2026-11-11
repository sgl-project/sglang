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
"""Nemotron Shortcut: the hybrid text stack.

Not served yet: the Pixtral vision tower, the MTP head, the top-k sparse
attention indexer (attention is dense, exact for contexts <= index_topk) and
TP/EP > 1.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.configs.nemotron_shortcut import (
    GDP_LAYER_TYPES,
    MOE_LAYER_TYPES,
    gdp_conv_dim,
    gdp_value_dim,
)
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
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
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

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        factors = torch.sigmoid(self.logit).to(streams.dtype)
        return torch.matmul(factors, _split_streams(streams, self.num_streams))


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

    def read(self, streams: torch.Tensor) -> torch.Tensor:
        factors = torch.sigmoid(self.read_logit).to(streams.dtype)
        return torch.matmul(factors, _split_streams(streams, self.num_streams))

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
        self.num_heads = config.gdp_num_heads
        self.value_dim = gdp_value_dim(config)
        self.conv_dim = gdp_conv_dim(config)
        # in_proj rows: [z, V_0..V_{n-1}, K_0..K_{n-1}, Q, b_0..b_{n-1}, a].
        self.split_sizes = [
            self.value_dim,
            self.conv_dim,
            n * self.num_heads,
            self.num_heads,
        ]
        self.in_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=sum(self.split_sizes),
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
            group_size=self.value_dim // config.gdp_num_groups,
            norm_before_gate=False,
            activation="swish",
        )
        self.out_proj = RowParallelLinear(
            input_size=self.value_dim,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=config.gdp_num_groups,
            num_k_heads=config.gdp_num_groups,
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
        core = self.attn(forward_batch, mixed_qkv=mixed_vkq.contiguous(), a=a, b=b)
        core = core.reshape(-1, self.value_dim)
        core = self.norm(core, z.contiguous())
        out, _ = self.out_proj(core)
        return out


class NemotronShortcutAttention(nn.Module):
    """NoPE attention: position information only enters the model through the
    GDP mixers."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            input_size=self.q_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        # TODO(nemotron_shortcut): top-k sparse attention; the indexer is loaded, unused.
        self.indexer = (
            NemotronShortcutIndexer(
                config, quant_config=quant_config, prefix=add_prefix("indexer", prefix)
            )
            if config.index_topk is not None
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
        out = self.attn(q, k, v, forward_batch)
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
        if clamp_value is not None and not get_moe_runner_backend().is_marlin():
            raise ValueError(
                "Nemotron Shortcut experts use a soft-clamped squared ReLU that only the "
                "Marlin "
                "NVFP4 MoE implements; launch with --moe-runner-backend marlin."
            )
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
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            use_grouped_topk=True,
            topk_group=1,
            num_expert_group=1,
            renormalize=config.num_experts_per_tok > 1,
            scoring_func="sigmoid",
            correction_bias=self.gate.e_score_correction_bias,
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
        if self.fc2_norm is not None:
            routed = self.fc2_norm(routed)
        out, _ = self.fc2_latent_proj(routed)
        if self.shared_experts is not None:
            out = out + self.shared_experts(shared_input)
        return out


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def _build_mixer(config, layer_type, layer_id, quant_config, prefix) -> nn.Module:
    if layer_type in GDP_LAYER_TYPES:
        return NemotronShortcutGDPMixer(
            config, layer_id, quant_config, prefix=add_prefix("mixer", prefix)
        )
    return NemotronShortcutAttention(
        config, layer_id, quant_config, prefix=add_prefix("self_attn", prefix)
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

    def __init__(self, config, layer_type, layer_id, quant_config, prefix):
        super().__init__()
        self.shortcut_read = NemotronShortcutWideResidualRead(config)
        self.shortcut_input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer_residual_connection = NemotronShortcutWideResidualConnection(config)
        self.pre_moe_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = NemotronShortcutMoE(
            config, layer_id, quant_config, prefix=add_prefix("moe", prefix)
        )
        self.shortcut_post_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.moe_shortcut_post_norm
            else None
        )
        self.moe_residual_connection = NemotronShortcutWideResidualConnection(config)
        self.uses_gdp = layer_type == "shortcut_moe_gdp"
        mixer = _build_mixer(config, layer_type, layer_id, quant_config, prefix)
        if self.uses_gdp:
            self.mixer = mixer
        else:
            self.self_attn = mixer

    def forward(
        self, streams: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        routed_input = self.shortcut_input_layernorm(self.shortcut_read(streams))
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

    def forward(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        streams = input_embeds.repeat(1, self.num_streams)
        for layer in self.layers:
            streams = layer(streams, forward_batch)
        return self.norm(self.residual_readout(streams))


class NemotronShortcutModel(nn.Module):
    """Mirrors the checkpoint's `model.language_model` nesting."""

    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.language_model = NemotronShortcutTextModel(
            config, quant_config, prefix=add_prefix("language_model", prefix)
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
        if parallel.tp_size != 1 or parallel.moe_ep_size != 1:
            raise NotImplementedError(
                "Nemotron Shortcut currently supports TP=1, EP=1 only."
            )
        self.config = config
        self.text_config = text_config
        self.quant_config = quant_config
        self.model = NemotronShortcutModel(
            text_config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(text_config)
        if text_config.index_topk is not None:
            logger.warning(
                "Nemotron Shortcut sparse attention is not implemented; attention is dense "
                "and matches the model only for contexts <= %d tokens.",
                text_config.index_topk,
            )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.language_model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model.language_model(
            input_ids, forward_batch, input_embeds
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    @staticmethod
    def _skip_weight(name: str) -> bool:
        return (
            name.startswith("model.vision_tower.")
            or name.startswith("model.multi_modal_projector.")
            or ".mtp." in name
            # NVFP4 KV-cache scales; the KV cache is served unquantized.
            or name.endswith(".k_scale")
            or name.endswith(".v_scale")
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        # Non-gated experts: the checkpoint's up_proj is FusedMoE's w1.
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="up_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="",
            num_experts=self.text_config.n_routed_experts,
        )
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, loaded_weight in weights:
            if self._skip_weight(name):
                continue
            if ".self_attn." in name and ".indexer." not in name:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if f".{weight_name}." not in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    param = params_dict[mapped]
                    param.weight_loader(param, loaded_weight, shard_id)
                    loaded.add(mapped)
                    break
                else:
                    self._load_plain(params_dict, name, loaded_weight, loaded)
                continue
            if ".moe.experts." in name:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    param = params_dict[mapped]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded.add(mapped)
                    break
                else:
                    raise KeyError(f"Unmapped Nemotron Shortcut expert weight: {name}")
                continue
            self._load_plain(params_dict, name, loaded_weight, loaded)

        missing = sorted(set(params_dict) - loaded)
        if missing:
            logger.warning(
                "Nemotron Shortcut: %d parameters not loaded from the checkpoint, e.g. %s",
                len(missing),
                missing[:8],
            )

    @staticmethod
    def _load_plain(params_dict, name, loaded_weight, loaded) -> None:
        if name not in params_dict:
            raise KeyError(f"Unexpected Nemotron Shortcut checkpoint weight: {name}")
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        loaded.add(name)


class Nemotron4ForConditionalGeneration(NemotronShortcutForConditionalGeneration):
    """The checkpoint's architecture name."""


EntryClass = [Nemotron4ForConditionalGeneration]
