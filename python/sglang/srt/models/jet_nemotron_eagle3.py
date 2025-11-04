from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from transformers.utils import logging

from sglang.srt.configs.jet_nemotron import JetNemotronConfig
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.jet_nemotron import (
    JetBlock,
    JetNemotronAttention,
    JetNemotronForCausalLM,
    JetNemotronMLP,
    JetNemotronRMSNorm,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.get_logger(__name__)


class JetNemotronDecoderLayerEagle3(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.midlayer_type == "attn":
            self.self_attn = JetNemotronAttention(
                config, layer_id, quant_config, prefix
            )
        elif config.midlayer_type == "swa":
            assert (
                config.efficient_attention_config is not None
            ), "Efficient attention config must be provided in JetNemotronConfig."
            self.self_attn = JetNemotronAttention(
                config,
                layer_id,
                quant_config,
                prefix,
                sliding_window=config.efficient_attention_config["swa"]["window_size"],
            )
        else:
            config.hidden_size = config.hidden_size * 2
            self.self_attn = JetBlock(config, layer_id, quant_config, prefix)
            config.hidden_size = config.hidden_size // 2
            self.self_attn.o_proj = nn.Linear(
                self.self_attn.value_dim, config.hidden_size, bias=False
            )
        if config.midlayer_type in ["attn", "swa"]:
            self.self_attn.qkv_proj = QKVParallelLinear(
                2 * config.hidden_size,
                self.self_attn.head_dim,
                config.num_attention_heads,
                config.num_key_value_heads,
                bias=True,
                quant_config=quant_config,
                tp_rank=self.self_attn.attn_tp_rank,
                tp_size=self.self_attn.attn_tp_size,
            )

        self.mlp = JetNemotronMLP(config, quant_config, prefix)
        self.input_layernorm = JetNemotronRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = JetNemotronRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.hidden_norm = JetNemotronRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        input_embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class JetNemotronModelEagle3(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.norm = JetNemotronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.midlayer = JetNemotronDecoderLayerEagle3(config, 0, quant_config, prefix)
        if hasattr(config, "target_hidden_size"):
            self.fc = torch.nn.Linear(
                config.target_hidden_size * 3, config.hidden_size, bias=False
            )
        else:
            self.fc = torch.nn.Linear(
                config.hidden_size * 3, config.hidden_size, bias=False
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        hidden_states = forward_batch.spec_info.hidden_states
        if forward_batch.forward_mode.is_prefill():
            hidden_states = self.fc(hidden_states)

        hidden_states = self.midlayer(positions, embeds, hidden_states, forward_batch)
        hidden_states_norm = self.norm(hidden_states)

        return hidden_states_norm, [hidden_states]


class JetNemotronForCausalLMEagle3(JetNemotronForCausalLM):
    def __init__(
        self,
        config: JetNemotronConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.config = config
        self.vocab_size = config.vocab_size

        if config.draft_vocab_size is None:
            config.draft_vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(
            config.draft_vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            bias=False,
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )

        self.model = JetNemotronModelEagle3(config, quant_config, prefix)

        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                self.hot_token_id = self.hot_token_id.to(torch.int64)
                continue
            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    name = "model." + name
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    name = "model." + name
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [JetNemotronForCausalLMEagle3]
