# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
# Adapted from:
# https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/model_executor/models/cohere2_moe.py
"""Inference-only Cohere2Moe (Command A Plus) model compatible with HuggingFace weights."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, get_compiler_backend, is_cuda, make_layers


@torch.compile(backend=get_compiler_backend())
def _cohere_layer_norm(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    mean = hidden_states.mean(-1, keepdim=True)
    variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    hidden_states = (hidden_states - mean) * torch.rsqrt(variance + variance_epsilon)
    hidden_states = weight.to(torch.float32) * hidden_states
    return hidden_states.to(input_dtype)


class Cohere2MoeLayerNorm(nn.Module):
    """Centered layer norm with learnable scale only (no bias)."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return _cohere_layer_norm(hidden_states, self.weight, self.variance_epsilon)


def cohere2_sigmoid_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    """Sigmoid -> top-k (-> renormalize) routing."""
    scores = gating_output.float().sigmoid()
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


class Cohere2MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
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

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Cohere2MoeAttention(nn.Module):
    """Attention with optional RoPE on sliding-window layers only."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = getattr(
            config, "model_max_length", None
        ) or getattr(config, "max_position_embeddings", 8192)
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is None:
            rope_parameters = {
                "rope_theta": getattr(config, "rope_theta", 10000.0),
                "rope_type": "default",
            }
        self.rope_theta = rope_parameters.get(
            "rope_theta", getattr(config, "rope_theta", 10000.0)
        )
        self.rope_scaling = rope_parameters

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            reduce_results=False,
        )

        layer_types = getattr(config, "layer_types", None)
        self.is_sliding = (
            layer_types is not None and layer_types[layer_id] == "sliding_attention"
        )
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        prefix_dense_sliding_window_pattern = getattr(
            config, "prefix_dense_sliding_window_pattern", 1
        )
        self.force_rope = bool(
            first_k_dense_replace
            and prefix_dense_sliding_window_pattern == 1
            and layer_id < first_k_dense_replace
        )
        sliding_window = getattr(config, "sliding_window", None)
        self.sliding_window_size = (
            sliding_window if (self.is_sliding and sliding_window is not None) else -1
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
            is_neox_style=False,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.is_sliding or self.force_rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Cohere2MoeSparseMoeBlock(nn.Module):
    """Sigmoid-routed MoE with optional shared experts (combined via 'sum' or 'average')."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.layer_id = layer_id

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.expert_selection_fn = getattr(config, "expert_selection_fn", "softmax")
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)

        if self.expert_selection_fn == "sigmoid":
            custom_routing_function = cohere2_sigmoid_topk
            scoring_func = "sigmoid"
        else:
            custom_routing_function = None
            scoring_func = "softmax"

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        self.topk = TopK(
            top_k=self.top_k,
            renormalize=self.norm_topk_prob,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            layer_id=layer_id,
        )

        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            reduce_results=False,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=add_prefix("experts", prefix),
        )

        num_shared_experts = getattr(config, "num_shared_experts", 0)
        self.num_shared_experts = num_shared_experts
        if num_shared_experts > 0:
            self.shared_experts = Cohere2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size * num_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )
        else:
            self.shared_experts = None

        self.shared_expert_combination_strategy = getattr(
            config, "shared_expert_combination_strategy", "sum"
        )
        assert self.shared_expert_combination_strategy in ("average", "sum")

        # Auxiliary CUDA stream so shared_experts can overlap with the
        # gate + routed-experts path inside a captured CUDA graph. Only used
        # during capture/replay; outside capture the sync overhead outweighs it.
        self.alt_stream = (
            torch.cuda.Stream()
            if is_cuda() and self.shared_experts is not None
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        if self.shared_experts is None:
            router_logits, _ = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
            final_hidden_states = self.experts(hidden_states, topk_output)
            return final_hidden_states.view(orig_shape)

        # FusedMoE.experts can write back into its input buffer (observed for
        # the unquantized triton BF16 path). Snapshot the post-norm input so
        # the shared-expert branch sees the original layernorm output.
        shared_input = hidden_states.clone()

        if self.alt_stream is not None and get_is_capture_mode():
            # Multi-stream overlap: shared_experts on alt stream, in parallel
            # with gate + topk + routed experts on the main stream.
            current_stream = torch.cuda.current_stream()
            shared_input.record_stream(self.alt_stream)
            self.alt_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.alt_stream):
                shared_out = self.shared_experts(shared_input)
            router_logits, _ = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
            routed_out = self.experts(hidden_states, topk_output)
            current_stream.wait_stream(self.alt_stream)
        else:
            router_logits, _ = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
            routed_out = self.experts(hidden_states, topk_output)
            shared_out = self.shared_experts(shared_input)

        final_hidden_states = routed_out + shared_out
        if self.shared_expert_combination_strategy == "average":
            final_hidden_states = final_hidden_states / 2
        # Returned un-reduced: the decoder layer folds attn + MoE TP-partials
        # into a single all-reduce.
        return final_hidden_states.view(orig_shape)


class Cohere2MoeDecoderLayer(nn.Module):
    """Parallel attention + MLP: out = residual + attn(norm(x)) + mlp(norm(x))."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        self.self_attn = Cohere2MoeAttention(
            config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        if layer_id < first_k_dense_replace:
            self.mlp = Cohere2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=getattr(
                    config, "prefix_dense_intermediate_size", config.intermediate_size
                ),
                quant_config=quant_config,
                # Folded into the decoder layer's single all-reduce.
                reduce_results=False,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = Cohere2MoeSparseMoeBlock(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        norm_eps = getattr(config, "layer_norm_eps", 1e-5)
        self.input_layernorm = Cohere2MoeLayerNorm(config.hidden_size, eps=norm_eps)
        self.tp_size = get_tensor_model_parallel_world_size()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Parallel structure: y = x + attn(norm(x)) + mlp(norm(x)). The single
        # residual lets the two TP all-reduces (attn.o_proj, mlp) fold into one
        # sum-then-allreduce, halving per-layer all-reduces.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        mlp_out = self.mlp(hidden_states)
        combined = attn_out + mlp_out
        if self.tp_size > 1:
            combined = tensor_model_parallel_all_reduce(combined)
        return residual + combined


class Cohere2MoeModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Cohere2MoeDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        norm_eps = getattr(config, "layer_norm_eps", 1e-5)
        self.norm = Cohere2MoeLayerNorm(config.hidden_size, eps=norm_eps)

    def get_input_embeddings(self, input_ids: Optional[torch.Tensor] = None):
        """Return the embedding module, or the embedded tensor if ``input_ids``
        is given (SGLang's mm utils call this with no args)."""
        if input_ids is None:
            return self.embed_tokens
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Cohere2MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.logit_scale = getattr(config, "logit_scale", None)
        self.logits_processor = LogitsProcessor(config, logit_scale=self.logit_scale)
        self.model = Cohere2MoeModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

    def get_input_embeddings(self, input_ids: Optional[torch.Tensor] = None):
        if input_ids is None:
            return self.model.embed_tokens
        return self.model.get_input_embeddings(input_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        if get_embedding:
            return hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.model.embed_tokens, forward_batch
        )

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
        loaded_params = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Skip all-zero bias tensors that the checkpoint carries for
            # bias-free Cohere layers (input_layernorm.bias, norm.bias,
            # o_proj.bias, mlp.gate.bias, experts.*.[gate|up|down]_proj.bias).
            if (name.endswith(".bias") or name.endswith("_bias")) and (
                name not in params_dict
                and name.replace("q_proj", "qkv_proj") not in params_dict
                and name.replace("gate_proj", "gate_up_proj") not in params_dict
            ):
                continue

            # Stacked attention / MLP weights.
            matched = False
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                new_name = name.replace(shard_name, param_name)
                if new_name.endswith(".bias") and new_name not in params_dict:
                    matched = True
                    break
                if new_name not in params_dict:
                    matched = True
                    break
                param = params_dict[new_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(new_name)
                matched = True
                break
            if matched:
                continue

            # Expert weights.
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name not in params_dict:
                    continue
                param = params_dict[new_name]
                weight_loader = param.weight_loader
                weight_loader(
                    param,
                    loaded_weight,
                    new_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded_params.add(new_name)
                matched = True
                break
            if matched:
                continue

            # lm_head is tied with embed_tokens; skip if missing.
            if "lm_head.weight" in name:
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


EntryClass = Cohere2MoeForCausalLM
