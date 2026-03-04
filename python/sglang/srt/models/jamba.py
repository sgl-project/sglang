# Copyright 2023-2025 SGLang Team
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
"""Inference-only Jamba model.

Jamba is AI21's hybrid Transformer-Mamba1-MoE model.
This implementation uses period-based layer selection:
- Attention layers: every attn_layer_period layers (default: 8)
- MoE layers: every expert_layer_period layers (default: 2)
- Mamba1 layers: all non-attention layers

Mamba1 uses selective_scan_fn from mamba-ssm for prefill and a custom
triton kernel for decode.
"""

from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn

from sglang.srt.configs.jamba import ATTENTION, MAMBA, MLP, MOE, JambaConfig
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    Mamba1AttnBackend,
)
from sglang.srt.layers.attention.mamba.mamba1 import MambaMixer1
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers
from sglang.utils import logger


class JambaMLP(nn.Module):
    """Jamba dense MLP layer with SiLU-gated activation."""

    def __init__(
        self,
        config: JambaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class JambaMoE(nn.Module):
    """Jamba Sparse MoE layer with top-k routing."""

    def __init__(
        self,
        config: JambaConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.router",
        )

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=True,
            layer_id=layer_idx,
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts
            + get_global_server_args().ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            layer_id=layer_idx,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape

        # Router logits
        router_logits, _ = self.router(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)

        # Expert forward
        final_hidden_states = self.experts(hidden_states, topk_output)

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class JambaAttention(nn.Module):
    """Jamba attention layer with GQA and RoPE."""

    def __init__(
        self,
        config: JambaConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=10000,
            is_neox_style=True,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class JambaMambaDecoderLayer(nn.Module):
    """Jamba decoder layer with Mamba1 mixer and MLP/MoE FFN."""

    def __init__(
        self,
        config: JambaConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Mamba1 mixer (generic reusable layer, like MambaMixer2 for Mamba2 models)
        self.mamba = MambaMixer1(
            cache_params=config.mamba1_cache_params,
            hidden_size=config.hidden_size,
            dt_rank=config.mamba_dt_rank_value,
            use_conv_bias=config.mamba_conv_bias,
            use_bias=config.mamba_proj_bias,
            use_dt_bc_layernorm=True,
            rms_norm_eps=config.rms_norm_eps,
            activation="silu",
            quant_config=quant_config,
            prefix=f"{prefix}.mamba",
        )

        # FFN: MLP or MoE based on layer pattern
        _, ffn_type = config.get_layer_types(layer_idx)
        if ffn_type == MOE:
            self.feed_forward = JambaMoE(
                config,
                layer_idx,
                quant_config=quant_config,
                prefix=f"{prefix}.feed_forward",
            )
        else:
            self.feed_forward = JambaMLP(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.feed_forward",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm for Mamba
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Use Mamba1AttnBackend for forward pass (output tensor pattern)
        attn_backend = forward_batch.attn_backend
        assert isinstance(attn_backend, HybridLinearAttnBackend)
        assert isinstance(attn_backend.linear_attn_backend, Mamba1AttnBackend)
        output = torch.empty_like(hidden_states)
        attn_backend.linear_attn_backend.forward(
            mixer=self.mamba,
            hidden_states=hidden_states,
            output=output,
            layer_id=self.layer_idx,
        )
        hidden_states = output + residual

        # Pre-norm for FFN
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)

        # FFN
        hidden_states = self.feed_forward(hidden_states)

        return hidden_states, residual


class JambaAttentionDecoderLayer(nn.Module):
    """Jamba decoder layer with attention and MLP/MoE FFN."""

    def __init__(
        self,
        config: JambaConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = JambaAttention(
            config,
            layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # FFN: MLP or MoE based on layer pattern
        _, ffn_type = config.get_layer_types(layer_idx)
        if ffn_type == MOE:
            self.feed_forward = JambaMoE(
                config,
                layer_idx,
                quant_config=quant_config,
                prefix=f"{prefix}.feed_forward",
            )
        else:
            self.feed_forward = JambaMLP(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.feed_forward",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm for attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Attention
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = hidden_states + residual

        # Pre-norm for FFN
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)

        # FFN
        hidden_states = self.feed_forward(hidden_states)

        return hidden_states, residual


# Type aliases
JambaDecoderLayer = Union[JambaMambaDecoderLayer, JambaAttentionDecoderLayer]


class JambaModel(nn.Module):
    """Jamba model backbone."""

    def __init__(
        self,
        *,
        config: JambaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(idx: int, prefix: str):
            block_type, _ = config.get_layer_types(idx)
            if block_type == ATTENTION:
                return JambaAttentionDecoderLayer(
                    config, idx, quant_config=quant_config, prefix=prefix
                )
            else:
                return JambaMambaDecoderLayer(
                    config, idx, quant_config=quant_config, prefix=prefix
                )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            get_layer,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

        if self.pp_group.is_last_rank:
            self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.final_layernorm = PPMissingLayer(return_tuple=True)

        # Track which layers are Mamba layers for state management
        self.mamba_layer_indices = [
            i for i in range(self.start_layer, self.end_layer)
            if config._get_layer_block_type(i) == MAMBA
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
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

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            # Both Mamba and Attention layers share the same forward signature
            # The Mamba1AttnBackend handles state management internally
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class JambaForCausalLM(nn.Module):
    """Jamba model for causal language modeling."""

    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    def __init__(
        self,
        *,
        config: JambaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = JambaModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.pp_group = get_pp_group()

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.unpadded_vocab_size = config.vocab_size
                self.lm_head = ParallelLMHead(
                    self.unpadded_vocab_size,
                    config.hidden_size,
                    org_num_embeddings=config.vocab_size,
                    padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        if self.pp_group.world_size > 1 and self.config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            elif self.pp_group.is_last_rank:
                emb_token_weight = self.pp_group.recv(
                    size=self.lm_head.weight.shape,
                    dtype=next(self.model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, pp_proxy_tensors, input_embeds
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> None:
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if is_mtp:
                if "mtp" not in name:
                    continue
                name = name.replace("mtp.layers.", "model.layers.")
                if "embeddings" in name:
                    name = name.replace("embeddings", "model.embed_tokens")
                    if name.startswith("backbone."):
                        name = name.replace("backbone.", "")

            if not is_mtp and "mtp" in name:
                continue

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "embed_tokens" in name and not self.pp_group.is_first_rank:
                continue

            if (
                "final_layernorm" in name or "lm_head" in name
            ) and not self.pp_group.is_last_rank:
                continue

            # Handle stacked params (QKV, gate_up)
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Skip expert weights here - they are handled by
                # expert_params_mapping below. Without this guard,
                # "gate_proj" in "experts.0.gate_proj" would match and
                # corrupt the name to "experts.0.gate_up_proj".
                if "feed_forward.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle expert weights
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.debug(f"Parameter {name} not found in params_dict")


EntryClass = [JambaForCausalLM]
