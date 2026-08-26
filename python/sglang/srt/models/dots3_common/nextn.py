"""Inference-only full-sharing Dots3 MTP / NextN draft model."""

import logging
from collections.abc import Iterable

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
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
from sglang.srt.models.dots3_common.modeling import (
    Dots3DecoderLayer,
    Dots3LanguageModelForCausalLM,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import BumpAllocator, add_prefix

logger = logging.getLogger(__name__)


class Dots3MTPHead(nn.Module):
    """The single MTP layer, recursively reused by every draft step."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("eh_proj", prefix),
        )
        self.decoder = Dots3DecoderLayer(
            config,
            layer_id=0,
            quant_config=quant_config,
            is_nextn=True,
            prefix=add_prefix("decoder", prefix),
        )
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Dot3NoteModelNextN(nn.Module):
    """Text-only draft model containing one full-sharing MTP layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.num_nextn_predict_layers != 1:
            raise ValueError(
                "Dots3 MTP currently supports one full-sharing layer only."
            )
        if list(config.layer_types) != ["sliding_attention"]:
            raise ValueError("Dots3 MTP full-sharing layer must use sliding_attention.")

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )
        # The weight loader maps the shared MTP layer to heads.0.
        self.heads = nn.ModuleList(
            [Dots3MTPHead(config, quant_config, add_prefix("heads.0", prefix))]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=2, dtype=torch.float32, device=device
        )
        hidden_states = (
            self._embed_input_ids(input_ids) if input_embeds is None else input_embeds
        )
        head = self.heads[0]
        if hidden_states.shape[0] > 0:
            hidden_states, _ = head.eh_proj(
                torch.cat(
                    (
                        head.enorm(hidden_states),
                        head.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )

        residual = None
        with get_global_expert_distribution_recorder().disable_this_region():
            hidden_states, residual = head.decoder(
                positions, hidden_states, forward_batch, residual, zero_allocator
            )

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = head.shared_head.norm(hidden_states)
            else:
                hidden_states, _ = head.shared_head.norm(hidden_states, residual)
        return hidden_states

    def _embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Multimodal sentinels use target hidden states, so clamp their unused
        # draft embedding indices to the vocabulary.
        return self.embed_tokens(input_ids.clamp(min=0, max=self.vocab_size - 1))


class Dots3NoteForCausalLMNextN(Dots3LanguageModelForCausalLM):
    """Full-sharing Dots3 MTP draft registered for NEXTN decoding."""

    fused_shared_experts_architecture = "Dots3NoteForCausalLMNextN"

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.fuse_qkv_a_g_proj = True
        self.packed_modules_mapping = {
            "fused_qkv_a_g_proj_with_mqa": [
                "q_a_proj",
                "kv_a_proj_with_mqa",
                "g_proj",
            ]
        }
        self.determine_num_fused_shared_experts()

        self.model = Dot3NoteModelNextN(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_parallel().config.enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self._mtp_loaded_embed = False

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights = list(weights)
        self._mtp_loaded_embed = any(
            name.startswith("model.mtp.embed_tokens.") for name, _ in weights
        )
        super().load_weights(weights, is_nextn=True)

    def set_embed_and_head(self, embed, head):
        # Preserve a checkpoint-provided MTP embedding; share the output head.
        if not self._mtp_loaded_embed:
            del self.model.embed_tokens.weight
            self.model.embed_tokens.weight = embed
        else:
            logger.info("Keeping the checkpoint's MTP-specific input embedding.")
        del self.lm_head.weight
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
