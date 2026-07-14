"""GigaChat 3.5 multi-head MTP.

Multiple heads are served Step-3.5 style by sglang's multi-layer EAGLE worker
(``MultiLayerEagleDraftWorker``): one ``GigaChat35ForCausalLMNextN`` instance per
speculative step, selected by ``draft_model_idx``, with hidden-state chaining
(``chain_mtp_hidden_states``) so each head consumes the previous head's output
hidden state instead of always reusing the target model's.

The block mirrors ``DeepseekModelNextN``'s *shape and naming* (so the checkpoint
weight keys map cleanly through ``DeepseekV2WeightLoaderMixin``) but is built
standalone from the GigaChat modules.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
    NextNDisabledConfig,
    NextNEnabledConfig,
)
from sglang.srt.models.gigachat35 import (
    GigaChat35Config,
    GigaChat35DecoderLayer,
    _remap_gigachat_weight_names,
    build_norm,
)
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import BumpAllocator, add_prefix


class GigaChat35ModelNextN(nn.Module):
    def __init__(
        self,
        config: GigaChat35Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.enorm = build_norm(config, config.hidden_size)
        self.hnorm = build_norm(config, config.hidden_size)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        self.alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        self.decoder = GigaChat35DecoderLayer(
            config=config,
            layer_id=0,
            quant_config=quant_config,
            prefix=add_prefix("decoder", prefix),
            alt_stream=self.alt_stream,
            is_nextn=True,
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = build_norm(config, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=(
                input_embeds.device if input_embeds is not None else input_ids.device
            ),
        )

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        if hidden_states.shape[0] > 0:
            hidden_states = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )

        residual = None
        hidden_states, residual = self.decoder(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=residual,
            zero_allocator=zero_allocator,
        )

        hidden_states_before_norm = None
        if not forward_batch.forward_mode.is_idle():
            hidden_states_before_norm = (
                hidden_states if residual is None else hidden_states + residual
            )
            if residual is not None:
                hidden_states, _ = self.shared_head.norm(hidden_states, residual)
            else:
                hidden_states = self.shared_head.norm(hidden_states)

        return hidden_states, hidden_states_before_norm


class GigaChat35ForCausalLMNextN(DeepseekV2WeightLoaderMixin, nn.Module):
    def __init__(
        self,
        config: GigaChat35Config,
        quant_config: Optional[QuantizationConfig] = None,
        draft_model_idx: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_fused_shared_experts = 0
        self.draft_model_idx = draft_model_idx or 0

        self.model = GigaChat35ModelNextN(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states, hidden_states_before_norm = self.model(
            input_ids, positions, forward_batch
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            hidden_states_before_norm=hidden_states_before_norm,
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _initialize_nextn_conf(self, is_nextn: bool):
        if not is_nextn:
            return NextNDisabledConfig()
        if not hasattr(self.config, "num_nextn_predict_layers"):
            raise ValueError("num_nextn_predict_layers is not in the config")
        nextn_layer_id = (
            0
            if self.config.num_hidden_layers == 1
            else self.config.num_hidden_layers + self.draft_model_idx
        )
        return NextNEnabledConfig(
            num_nextn_layers=1,
            nextn_layer_id=nextn_layer_id,
            nextn_layer_prefix=f"model.layers.{nextn_layer_id}",
            nextn_spec_weight_names=["shared_head.norm", "eh_proj", "enorm", "hnorm"],
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        self.do_load_weights(_remap_gigachat_weight_names(weights), is_nextn=True)

    def post_load_weights(self, is_nextn: bool = True, weight_names=None) -> None:
        super().post_load_weights(is_nextn=True, weight_names=weight_names)


EntryClass = [GigaChat35ForCausalLMNextN]
