import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer, DeepseekV4ForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

COMPRESS_RATIO_DSPARK_LAYER = 0


class DSparkMarkovHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        markov_rank: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size,
            markov_rank,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("markov_w1", prefix),
        )
        self.markov_w2 = ParallelLMHead(
            vocab_size,
            markov_rank,
            quant_config=quant_config,
            prefix=add_prefix("markov_w2", prefix),
        )

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def project_bias(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.linear(embeddings, self.markov_w2.weight)


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        features = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(features.float()).squeeze(-1)


class DeepseekV4DSparkModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.block_size = config.dspark_block_size
        self.markov_rank = config.dspark_markov_rank
        self.noise_token_id = config.dspark_noise_token_id
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.num_dspark_layers = num_dspark_layers = get_dspark_num_layers(config)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.main_proj = ReplicatedLinear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("main_proj", prefix),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    is_nextn=True,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_streams=None,
                    compress_ratio_override=COMPRESS_RATIO_DSPARK_LAYER,
                )
                for layer_id in range(num_dspark_layers)
            ]
        )

        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            config.dspark_markov_rank,
            quant_config=quant_config,
            prefix=add_prefix("markov_head", prefix),
        )
        self.confidence_head = DSparkConfidenceHead(
            config.hidden_size + config.dspark_markov_rank
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.rms_norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def project_main_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        projected, _ = self.main_proj(main_hidden)
        return self.main_norm(projected)

    def forward_backbone(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        prev_residual, prev_post, prev_comb = None, None, None
        last_layer = None
        for layer in self.layers:
            last_layer = layer
            hidden_states, prev_residual, prev_post, prev_comb = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                input_ids=input_ids,
                input_ids_global=input_ids,
                prev_residual=prev_residual,
                prev_post=prev_post,
                prev_comb=prev_comb,
            )
        if last_layer is not None and prev_residual is not None:
            hidden_states = last_layer.hc_post(
                hidden_states, prev_residual, prev_post, prev_comb
            )
        return hidden_states

    def collapse_block_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        return self.shared_head.norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.forward_backbone(input_ids, positions, forward_batch)
        return self.collapse_block_hidden(hidden_states)


def get_dspark_num_layers(config: PretrainedConfig) -> int:
    return int(getattr(config, "dspark_num_layers", 0) or 3)


class DeepseekV4ForCausalLMDSpark(DeepseekV4ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()

        self.model = DeepseekV4DSparkModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @property
    def block_size(self) -> int:
        return self.model.block_size

    @property
    def num_dspark_layers(self) -> int:
        return self.model.num_dspark_layers

    def project_main_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        return self.model.project_main_hidden(main_hidden)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            return self.model(input_ids, positions, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights, is_nextn=False, is_dspark=True)

    def post_load_weights(self, is_nextn=False, is_dspark=False, weight_names=None):
        super().post_load_weights(is_dspark=True, weight_names=weight_names)


EntryClass = [DeepseekV4ForCausalLMDSpark]
