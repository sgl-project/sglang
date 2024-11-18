# Adapted from vLLM's GPT-J implementation for SGLang
# Original Authors: The vLLM team and HuggingFace Team
# License: Apache License, Version 2.0

from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import GPTJConfig

print(" ** This model is being loaded from built-in source at IR2-project **")
# from sglang.srt.layers.activation import get_act_fn
from vllm.model_executor.layers.activation import get_act_fn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention

# from sglang.srt.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class GPTJAttention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: GPTJConfig,
        cache_config=None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_size,
            self.total_num_heads,
            bias=False,
            quant_config=quant_config,
            # prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            # prefix=f"{prefix}.out_proj",
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        self.scale = self.head_size**-0.5
        assert getattr(config, "rotary", True)
        assert config.rotary_dim % 2 == 0
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.rotary_emb = get_rope(
            self.head_size,
            rotary_dim=config.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=False,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_size,
            scaling=self.scale,
            num_kv_heads=self.total_num_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class GPTJMLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: GPTJConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.n_embd
        self.fc_in = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            quant_config=quant_config,
            # prefix=f"{prefix}.fc_in",
        )
        self.fc_out = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            # prefix=f"{prefix}.fc_out",
        )
        self.act = get_act_fn(
            config.activation_function, quant_config, intermediate_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc_out(hidden_states)
        return hidden_states


class GPTJBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: GPTJConfig,
        cache_config=None,
        quant_config: Optional[QuantizationConfig] = None,
        # prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = 4 * config.n_embd if config.n_inner is None else config.n_inner
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.attn = GPTJAttention(
            layer_id,
            config,
            cache_config,
            quant_config,
            # prefix=f"{prefix}.attn",
        )
        self.mlp = GPTJMLP(
            inner_dim,
            config,
            quant_config,
            # prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        mlp_output = self.mlp(hidden_states)
        hidden_states = attn_output + mlp_output + residual
        return hidden_states


class GPTJModel(nn.Module):
    def __init__(
        self,
        config: GPTJConfig,
        cache_config=None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.n_embd
        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            self.embed_dim,
            # prefix=f"{prefix}.wte",
        )
        self.h = nn.ModuleList(
            [
                GPTJBlock(i, config, cache_config, quant_config)
                for i in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)

        for layer in self.h:
            hidden_states = layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPTJForCausalLM(nn.Module):
    def __init__(
        self,
        config: GPTJConfig,
        cache_config=None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.transformer = GPTJModel(
            config, cache_config, quant_config, prefix="transformer"
        )
        self.lm_head = self.transformer.wte
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head.weight, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # for name, loaded_weight in weights:
        #     if "attn.bias" in name or "attn.masked_bias" in name:
        #         continue

        #     if not name.startswith("transformer."):
        #         name = "transformer." + name

        #     param = params_dict.get(name)
        #     if param is None:
        #         continue

        #     # Handle weight transposition for Conv1D layers
        #     conv1d_layers = ["qkv_proj", "out_proj", "fc_in", "fc_out"]
        #     if any(conv1d_layer in name for conv1d_layer in conv1d_layers):
        #         if name.endswith(".weight"):
        #             loaded_weight = loaded_weight.t()

        #     weight_loader = getattr(param, "weight_loader",
        #                             default_weight_loader)
        #     weight_loader(param, loaded_weight)

        for name, loaded_weight in weights:
            if "attn.bias" in name or "attn.masked_bias" in name:
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
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = GPTJForCausalLM
