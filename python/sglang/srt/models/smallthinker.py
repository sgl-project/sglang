"""Inference-only SmallThinker model compatible with HuggingFace weights."""
"""Adapted from Qwen2MoE, Qwen3MoE and Mixtral"""

import logging
from collections.abc import Iterable
from typing import Iterable, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    get_local_attention_dp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.utils import add_prefix, make_layers
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)

"""Inference-only SmallThinker model compatible with HuggingFace weights."""

class SmallThinkerMoeBlock(nn.Module):    
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.moe_ffn_hidden_size = config.moe_ffn_hidden_size
        self.num_primary_experts = config.moe_num_primary_experts
        self.num_active_primary_experts = config.moe_num_active_primary_experts
        self.moe_primary_router_apply_softmax = config.moe_primary_router_apply_softmax
        self.tp_size = get_tensor_model_parallel_world_size()

        # Primary router
        self.primary_router = ReplicatedLinear(self.hidden_dim, self.num_primary_experts, bias=False, skip_bias_add=False, 
                                               quant_config=quant_config, prefix=f"{prefix}.primary_router")

        def custom_topk(hidden_states, gating_output, topk, renormalize):
            router_logits, selected_experts = torch.topk(gating_output[0], topk, dim=-1)
            if self.moe_primary_router_apply_softmax:
                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            else:
                routing_weights = F.sigmoid(router_logits)
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            return routing_weights, selected_experts
        
        self.experts = FusedMoE(num_experts=self.num_primary_experts,
                                top_k=self.num_active_primary_experts,
                                custom_routing_function=custom_topk,
                                hidden_size=self.hidden_dim,
                                intermediate_size=self.moe_ffn_hidden_size,
                                reduce_results=False,
                                renormalize=True,
                                quant_config=quant_config,
                                activation='relu',
                                prefix=f"{prefix}.experts")

    def forward(self, router_input: torch.Tensor, hidden_states: torch.Tensor, forward_batch: ForwardBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length, hidden_dim = hidden_states.shape
        
        orig_shape = hidden_states.shape
        # Flatten for processing
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_input = router_input.view(-1, hidden_dim)
        
        router_logits = self.primary_router(router_input)

        final_hidden_states = self.experts( hidden_states=hidden_states,
                                            router_logits=router_logits)
        final_hidden_states = final_hidden_states
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        
        return final_hidden_states.view(orig_shape), router_logits


class SmallThinkerAttention(nn.Module):
    """Multi-head attention with optional sliding window."""
    
    def __init__(
        self,
        config,
        layer_idx: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.rope_theta = config.rope_theta
        self.layer_idx = layer_idx
        self.rope_scaling = getattr(config, "rope_scaling", None)
        
        # Sliding window configuration
        self.sliding_window_size = None
        if hasattr(config, 'sliding_window_layout') and config.sliding_window_layout:
            if layer_idx < len(config.sliding_window_layout) and config.sliding_window_layout[layer_idx]:
                self.sliding_window_size = config.sliding_window_size
        
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.tp_rank = get_tensor_model_parallel_rank()

        # QKV projection
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=f"{prefix}.qkv_proj",
        )
        
        # Output projection
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=True,
            prefix=f"{prefix}.o_proj",
        )

        # RoPE
        if config.rope_layout[self.layer_idx]:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=self.rope_theta,
                rope_scaling=self.rope_scaling
            )
        else:
            self.rotary_emb = lambda positions, q, k: (q, k)

        # Attention mechanism with sliding window support
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=self.layer_idx,
            prefix=prefix,
            sliding_window_size=self.sliding_window_size,
            quant_config=quant_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class SmallThinkerDecoderLayer(nn.Module):
    """Decoder layer combining attention and MLP/MoE."""
    
    def __init__(
        self,
        config,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.local_dp_size = get_local_attention_dp_size()
        
        # Attention
        self.self_attn = SmallThinkerAttention(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        
        self.mlp = SmallThinkerMoeBlock(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        
        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            router_input = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            router_input = hidden_states + residual
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        
        # MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, _ = self.mlp(router_input, hidden_states, forward_batch)
        
        return hidden_states, residual

class SmallThinkerModel(nn.Module):
    """Main model class."""
    
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        # Embeddings
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Decoder layers
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SmallThinkerDecoderLayer(
                config=config,
                layer_idx=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,

            prefix=f"{prefix}.layers",
        )

        # Output norm
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

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
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

class SmallThinkerForCausalLM(nn.Module):
    """Causal language model with vLLM optimization."""
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__()

        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.model = SmallThinkerModel(config=config, quant_config=quant_config, prefix=add_prefix("model", prefix))

        # Language model head
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix)
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        hidden_states = self.model(input_ids, positions, forward_batch, pp_proxy_tensors, inputs_embeds)
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (".bias", "_bias", ".k_scale", "_k_scale",
                           ".v_scale", "_v_scale", ".weight_scale",
                           "_weight_scale", ".input_scale", "_input_scale")

        # Rule 1: QKV Fusion
        # Original weights:
        #   model.layers.{i}.self_attn.q_proj.weight
        #   model.layers.{i}.self_attn.k_proj.weight
        #   model.layers.{i}.self_attn.v_proj.weight
        # Fused weights:
        #   model.layers.{i}.self_attn.qkv_proj.weight (shards "q", "k", "v")
        # Format:
        #   (param_name, shard_name, shard_id)
        stacked_attn_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        # Params for weights
        # Rule 2: Expert fusion
        # Original weights:
        #   model.layers.{i}.block_sparse_moe.experts.{j}.gate.weight   (w1)
        #   model.layers.{i}.block_sparse_moe.experts.{j}.up.weight     (w3)
        #   model.layers.{i}.block_sparse_moe.experts.{j}.down.weight   (w2)
        # Fused weight:
        #   model.layers.{i}.mlp.experts.w13_weight (a lot of shards)
        #   model.layers.{i}.mlp.experts.w2_weight (a lot of shards)
        # Mapping data structure:
        #   (param_name, weight_name, expert_id, shard_id)
        #   ('experts.w13_', 'experts.0.gate.', 0, 'w1')
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate",
            ckpt_down_proj_name="down",
            ckpt_up_proj_name="up",
            num_experts=self.config.moe_num_primary_experts)
        
        # Rule 3: Router rename
        # Original weights:
        #   *.block_sparse_moe.*
        # Renamed weights:
        #   *.mlp.*
        router_rename_mapping = [("mlp.primary_router", "block_sparse_moe.primary_router")]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Skip tied LMHead
            if self.config.tie_word_embeddings and name.endswith(
                "lm_head.weight"):
                continue

            # Attention process (Rule 1)
            # Check for {qkv}_proj in weight name to skip non-stacked modules
            for (param_name, weight_name, shard_id) in stacked_attn_params_mapping:
                
                if weight_name not in name:
                    continue

                # Skip layers that are not in the current PP rank.
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
                # Original name:
                #   model.layers.0.self_attn.q_proj.weight
                # Mapped name:
                #   model.layers.0.self_attn.qkv_proj.weight
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                # Skip layers on other devices.
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # FFN Process (Rule 2)
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Skip layers that are not in the current PP rank.
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

                    name = name.replace(weight_name, param_name)
                    
                    # SmallThinker: expert name replace
                    name = name.replace("block_sparse_moe", "mlp")
                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(
                            ignore_suffixes) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Router process (Rule 3)
                    for mapping in router_rename_mapping:
                        param_name, weight_name = mapping
                        if weight_name not in name:
                            continue

                        # Skip layers that are not in the current PP rank.
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
                        
                        name = name.replace(weight_name, param_name)
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if name.endswith(
                                ignore_suffixes) and name not in params_dict:
                            continue
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight)
                        break
                    else:
                        if name not in params_dict:
                            continue
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if name.endswith(
                                ignore_suffixes) and name not in params_dict:
                            continue
                        # Remapping the name of FP8 kv-scale.
                        if name.endswith("kv_scale"):
                            remapped_kv_scale_name = name.replace(
                                ".kv_scale", ".attn.kv_scale")
                            if remapped_kv_scale_name not in params_dict:
                                logger.warning_once(
                                    "Found kv scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv-scale is not loaded.",  # noqa: E501
                                    name,
                                    remapped_kv_scale_name,
                                )
                                continue
                            else:
                                name = remapped_kv_scale_name
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
    
EntryClass = [SmallThinkerForCausalLM]