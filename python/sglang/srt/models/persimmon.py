from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from transformers import PersimmonConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers


class PersimmonMLP(nn.Module):

    def __init__(
        self, config: PersimmonConfig, quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, quant_config=quant_config
        )
        self.dense_4h_to_h = RowParallelLinear(
            config.intermediate_size, config.hidden_size, quant_config=quant_config
        )
        self.act = get_act_fn(config.hidden_act)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states, _ = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.dense_4h_to_h(hidden_states)
        return hidden_states


class PersimmonAttention(nn.Module):

    def __init__(
        self,
        config: PersimmonConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id: int = 0,
    ):
        super().__init__()
        self.config = config
        tensor_parallel_world_size = get_tensor_model_parallel_world_size()

        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tensor_parallel_world_size
        self.head_dim = self.hidden_size // self.total_num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.is_causal = True

        assert (self.head_dim * self.total_num_heads) == self.hidden_size
        assert self.total_num_heads % tensor_parallel_world_size == 0

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            bias=True,
            quant_config=quant_config,
        )
        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
        )
        self.is_qk_layernorm = config.qk_layernorm

        if self.is_qk_layernorm:
            self.q_layernorm = nn.LayerNorm(self.head_dim)
            self.k_layernorm = nn.LayerNorm(self.head_dim)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
        )
        self.scaling = self.head_dim**-0.5
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        seq_length = x.shape[0]
        return x.view(seq_length, self.num_heads, self.head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        seq_length = x.shape[0]
        return x.view(seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)

        if self.is_qk_layernorm:
            q = self._split_heads(q)
            k = self._split_heads(k)

            q = self.q_layernorm(q)
            k = self.k_layernorm(k)

            q = self._merge_heads(q)
            k = self._merge_heads(k)

        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v, forward_batch=forward_batch)
        output, _ = self.dense(attn_output)
        return output


class PersimmonDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PersimmonConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PersimmonAttention(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            layer_id=idx,
        )
        self.mlp = PersimmonMLP(config, quant_config=quant_config)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = hidden_states
        return outputs


class PersimmonModel(nn.Module):

    def __init__(
        self,
        config: PersimmonConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: PersimmonDecoderLayer(
                config, quant_config=quant_config, prefix=prefix, idx=idx
            ),
            prefix="model.layers",
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
        )

        if self.pp_group.is_last_rank:
            self.final_layernorm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
        else:
            self.final_layernorm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
        else:
            hidden_states = forward_batch.pp_input_hidden
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(
                position_ids=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
            )
        return self.final_layernorm(hidden_states)


class PersimmonForCausalLM(nn.Module):

    def __init__(
        self,
        config: PersimmonConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = PersimmonModel(
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(
            input_ids=input_ids,
            forward_batch=forward_batch,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name not in params_dict:
                if name == "lm_head.weight":
                    continue
                print(f"Warning: weight {name} not found in model.")
                continue
            param = params_dict[name]
            if "query_key_value" in name:
                output_dim = getattr(param, "output_dim", None)
                if output_dim is not None:
                    loaded_weight_shape = loaded_weight.shape
                    num_heads = self.config.num_attention_heads
                    loaded_weight = loaded_weight.view(
                        loaded_weight_shape[:output_dim]
                        + (num_heads, 3, -1)
                        + loaded_weight_shape[output_dim + 1 :]
                    )
                    loaded_weight = loaded_weight.transpose(output_dim, output_dim + 1)
                    loaded_weight = loaded_weight.reshape(loaded_weight_shape)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = PersimmonForCausalLM
