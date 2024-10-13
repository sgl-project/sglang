"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral.py#L1
"""Inference-only Grok1 model."""
import warnings
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.layers.fused_moe import FusedMoE
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class Grok1MoE(nn.Module):
    """A tensor-parallel MoE implementation for Grok1 that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
        )

        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=True,
            renormalize=False,
            quant_config=quant_config,
            tp_size=tp_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        router_logits = 30.0 * F.tanh(router_logits / 30.0)
        final_hidden_states = self.experts(hidden_states, router_logits)
        return final_hidden_states.view(orig_shape)


class Grok1Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        logit_cap: float = 30,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = 128
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=logit_cap,
        )
        # TODO(lianmin): load logit cap from config

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


class Grok1DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = Grok1Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            quant_config=quant_config,
        )
        self.block_sparse_moe = Grok1MoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
        )
        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Self Attention
        hidden_states = (
            self.post_attn_norm(
                self.self_attn(
                    positions=positions,
                    hidden_states=self.pre_attn_norm(hidden_states),
                    forward_batch=forward_batch,
                )
            )
            + hidden_states
        )

        # Fully Connected
        hidden_states = (
            self.post_moe_norm(self.block_sparse_moe(self.pre_moe_norm(hidden_states)))
            + hidden_states
        )
        return hidden_states


class Grok1Model(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                Grok1DecoderLayer(config, i, quant_config=quant_config)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
            hidden_states.mul_(self.config.embedding_multiplier_scale)
        else:
            hidden_states = input_embeds

        for i in range(len(self.layers)):
            hidden_states = self.layers[i](positions, hidden_states, forward_batch)
        hidden_states = self.norm(hidden_states)
        hidden_states.mul_(self.config.output_multiplier_scale)
        return hidden_states


class Grok1ForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Grok1Model(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

        # Monkey patch _prepare_weights to load pre-sharded weights
        setattr(DefaultModelLoader, "_prepare_weights", _prepare_presharded_weights)

        self.use_presharded_weights = True

        warnings.filterwarnings("ignore", category=FutureWarning)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head.weight, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if self.use_presharded_weights:
                        extra_kwargs = {
                            "use_presharded_weights": self.use_presharded_weights
                        }
                    else:
                        extra_kwargs = {}

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        weight_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        **extra_kwargs,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name is None:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


old_prepare_weights = getattr(DefaultModelLoader, "_prepare_weights")


def _prepare_presharded_weights(
    self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
) -> Tuple[str, List[str], bool]:
    import glob
    import os

    if get_tensor_model_parallel_world_size() == 1:
        return old_prepare_weights(self, model_name_or_path, revision, fall_back_to_pt)

    tp_rank = get_tensor_model_parallel_rank()
    allow_patterns = [f"*-{tp_rank:03d}.bin"]

    hf_folder = model_name_or_path

    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
    use_safetensors = False

    return hf_folder, hf_weights_files, use_safetensors


class Grok1ModelForCausalLM(Grok1ForCausalLM):
    """An alias for backward-compatbility."""

    pass


EntryClass = [Grok1ForCausalLM, Grok1ModelForCausalLM]
