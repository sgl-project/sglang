# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Inference-only Laguna (poolside/Laguna-XS.2) model."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.laguna import LagunaConfig
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import should_skip_post_experts_all_reduce
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import LazyValue, add_prefix, make_layers

logger = logging.getLogger(__name__)


class LagunaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        # Skip the in-block reduce when LayerCommunicator will fuse it or when
        # the next layer expects reduce-scatter — otherwise we'd double-reduce.
        x, _ = self.down_proj(
            x,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
        )
        return x


class LagunaMoEGate(nn.Module):
    def __init__(
        self,
        config: LagunaConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        )
        # Released checkpoint stores this under `mlp.experts.e_score_correction_bias`
        # (load_weights remaps it) but every value is 0.0; zero-init keeps us
        # correct if a future checkpoint omits the tensor entirely.
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.to(torch.float32), self.weight, None)


class LagunaMoE(nn.Module):
    def __init__(
        self,
        config: LagunaConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.moe_routed_scaling_factor
        self.router_logit_softcapping = getattr(
            config, "moe_router_logit_softcapping", 0.0
        )

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"TP size {self.tp_size} > num_experts {config.num_experts}."
            )

        self.gate = LagunaMoEGate(config, prefix=add_prefix("gate", prefix))

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts
            + get_global_server_args().ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            reduce_results=False,
            apply_router_weight_on_input=bool(config.moe_apply_router_weight_on_input),
            prefix=add_prefix("experts", prefix),
        )

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            renormalize=True,
            use_grouped_topk=False,
            scoring_func="sigmoid",
            correction_bias=self.gate.e_score_correction_bias,
        )

        # HF safetensors key is singular `shared_expert.…`; mirror so the
        # default loader picks it up without remapping.
        self.shared_expert = LagunaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("shared_expert", prefix),
        )

    def get_moe_weights(self):
        return [x.data for x in self.experts.parameters()]

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        shared_out = self.shared_expert(hidden_states)

        router_logits = self.gate(hidden_states)
        if self.router_logit_softcapping > 0.0:
            cap = self.router_logit_softcapping
            router_logits = torch.tanh(router_logits / cap) * cap
        topk_output = self.topk(hidden_states, router_logits)
        routed_out = self.experts(hidden_states, topk_output)

        # Non-grouped TopK doesn't honor apply_routed_scaling_factor_on_output,
        # so scale routed manually before adding the unscaled shared expert.
        if self.routed_scaling_factor != 1.0:
            routed_out = routed_out * self.routed_scaling_factor
        final = routed_out + shared_out

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final = tensor_model_parallel_all_reduce(final)
        return final


class LagunaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rms_norm_eps: float,
        rope_theta: float,
        rope_scaling: Optional[Dict[str, Any]],
        partial_rotary_factor: float,
        max_position_embeddings: int,
        attention_bias: bool,
        sliding_window_size: int,
        layer_type: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.layer_id = layer_id

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        # Per-head softplus gate (`gating=True` in HF). Shard like Q so the
        # local output dim matches `num_heads`.
        self.g_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads,
            bias=False,
            gather_output=False,
            quant_config=None,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("g_proj", prefix),
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
        )

        assert layer_type in {"sliding_attention", "full_attention"}
        use_sliding = layer_type == "sliding_attention"
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            sliding_window_size=sliding_window_size if use_sliding else -1,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
        )
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)

        gate, _ = self.g_proj(hidden_states)
        gate = F.softplus(gate.float()).to(attn_output.dtype)
        attn_output = attn_output.view(-1, self.num_heads, self.head_dim)
        attn_output = attn_output * gate.view(-1, self.num_heads, 1)
        attn_output = attn_output.reshape(-1, self.num_heads * self.head_dim)

        output, _ = self.o_proj(attn_output)
        return output


class LagunaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LagunaConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        layer_types = config.layer_types
        layer_type = layer_types[layer_id]
        is_swa = layer_type == "sliding_attention"

        layer_num_heads = config.num_attention_heads_per_layer[layer_id]

        if is_swa:
            rope_theta = config.swa_rope_theta
            rope_scaling = config.swa_rope_scaling
            partial_rotary_factor = config.swa_partial_rotary_factor
        else:
            rope_theta = config.rope_theta
            rope_scaling = config.full_rope_scaling
            partial_rotary_factor = config.partial_rotary_factor

        self.self_attn = LagunaAttention(
            hidden_size=self.hidden_size,
            num_heads=layer_num_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            layer_id=layer_id,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
            max_position_embeddings=config.max_position_embeddings,
            attention_bias=config.attention_bias,
            # SGLang's window is exclusive; HF's `sliding_window` is inclusive.
            sliding_window_size=config.sliding_window - 1,
            layer_type=layer_type,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        mlp_types = config.mlp_layer_types
        self.is_layer_sparse = mlp_types[layer_id] == "sparse"
        is_previous_layer_sparse = layer_id > 0 and mlp_types[layer_id - 1] == "sparse"
        is_next_layer_sparse = (
            layer_id + 1 < config.num_hidden_layers
            and mlp_types[layer_id + 1] == "sparse"
        )

        if self.is_layer_sparse:
            self.mlp = LagunaMoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = LagunaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=True,
                prefix=add_prefix("mlp", prefix),
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(
            hidden_states,
            forward_batch=forward_batch,
            should_allreduce_fusion=should_allreduce_fusion,
            use_reduce_scatter=use_reduce_scatter,
        )

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )
        return hidden_states, residual


class LagunaModel(nn.Module):
    def __init__(
        self,
        config: LagunaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type = LagunaDecoderLayer,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        decoder_layer_type = decoder_layer_type or LagunaDecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
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
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LagunaForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: LagunaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.model = LagunaModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)

        # Only walk this rank's local layers — out-of-range entries can be PPMissingLayer.
        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.start_layer, self.end_layer)
                if isinstance(self.model.layers[layer_id].mlp, LagunaMoE)
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        return hidden_states

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())

        # (layer, expert, shard) tuples that hit the per-expert loader,
        # cross-checked against `expected` below to fail on dropped weights.
        loaded_expert_shards: set[Tuple[int, int, str]] = set()
        moe_layer_ids = [
            i
            for i, mt in enumerate(self.config.mlp_layer_types)
            if mt == "sparse" and self.start_layer <= i < self.end_layer
        ]

        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if layer_id is not None and (
                layer_id < self.start_layer or layer_id >= self.end_layer
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            # HF stores the router correction bias under the experts namespace;
            # our parameter lives on the gate. Remap before dispatch.
            if name.endswith("mlp.experts.e_score_correction_bias"):
                name = name.replace(
                    "mlp.experts.e_score_correction_bias",
                    "mlp.gate.e_score_correction_bias",
                )

            # Stacked dense (QKV / gate_up). The `mlp.experts.` guard stops
            # `up_proj` substring from false-matching `experts.{i}.up_proj.weight`.
            matched_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts." in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped.endswith(".bias") and name_mapped not in params_dict:
                    continue
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                matched_stacked = True
                break
            if matched_stacked:
                continue

            matched_expert = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                param.weight_loader(
                    param,
                    loaded_weight,
                    name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                if layer_id is not None:
                    loaded_expert_shards.add((layer_id, expert_id, shard_id))
                matched_expert = True
                break
            if matched_expert:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                logger.warning("Parameter %s not found in params_dict", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        # If any routed-expert tensor was silently dropped (e.g. a future
        # checkpoint renaming `gate_proj`, or a ckpt-vs-mapping shape mismatch),
        # fail loud here instead of generating garbage.
        expected = {
            (layer_id, expert_id, shard_id)
            for layer_id in moe_layer_ids
            for expert_id in range(self.config.num_experts)
            for shard_id in ("w1", "w2", "w3")
        }
        missing = expected - loaded_expert_shards
        if missing:
            sample = sorted(missing)[:5]
            raise RuntimeError(
                f"{len(missing)} routed-expert tensors were not loaded "
                f"(sample: {sample}). Expected {len(expected)} (layers={moe_layer_ids}, "
                f"num_experts={self.config.num_experts}, shards=3)."
            )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = LagunaForCausalLM
