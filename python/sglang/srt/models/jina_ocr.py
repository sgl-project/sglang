# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only Jina-OCR-v1 (DeepSeek-OCR + FastMTP speculative head).

Checkpoint: jinaai/jina-ocr-v1. Its config declares DeepseekOCRForCausalLM; the
config loader remaps it to JinaOCRForCausalLM when mtp_num_heads is present, and
the draft side is mapped to JinaOCRForCausalLMNextN in model_config.py.

FastMTP trains one dense MTP block with recursive feedback: draft step k+1
consumes step k's post-norm hidden state. SGLang's EAGLE draft loop already
feeds the draft's own output back in, so this is a plain NextN draft model
rather than a new speculative algorithm.
"""

import copy
import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.deepseek_ocr import DeepseekVLV2Config
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek import DeepseekDecoderLayer
from sglang.srt.models.deepseek_ocr import DeepseekOCRForCausalLM

logger = logging.getLogger(__name__)

# Checkpoint tensors that belong to the FastMTP head, not the target model.
MTP_WEIGHT_PREFIXES = ("mtp_module.", "mtp_embed_tokens")
MTP_HEAD_PREFIX = "mtp_module.heads.0."
MTP_BLOCK_PREFIX = MTP_HEAD_PREFIX + "mtp_block."


class JinaOCRForCausalLM(DeepseekOCRForCausalLM):
    """Target model: DeepSeek-OCR unchanged, minus the FastMTP tensors."""

    # gets the embedding table and lm head, needed by the FastMTP head
    def get_embed_and_head(self):
        return self.model.model.embed_tokens.weight, self.model.lm_head.weight

    # this override avoids crash, as Jina contains extra tensors not in Deepseek OCR
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(
            (name, loaded_weight)
            for name, loaded_weight in weights
            if not name.startswith(MTP_WEIGHT_PREFIXES)
        )


class JinaOCRModelNextN(nn.Module):
    """One dense FastMTP block, shared across all draft steps."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        dense_config = copy.deepcopy(config)
        dense_config.n_routed_experts = None
        self.decoder = DeepseekDecoderLayer(
            dense_config,
            layer_id=0,
            quant_config=quant_config,
            prefix=add_prefix("decoder", prefix),
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # compute embeddings from input ids; follows same pattern as deepseek nextn
    def _embed_input_ids(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        input_embeds = forward_batch.mm_input_embeds
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.contains_mm_inputs()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            assert input_embeds is not None
            last_indices = (
                forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
            ).long()
            input_embeds[last_indices] = self.embed_tokens(input_ids[last_indices])
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        return input_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self._embed_input_ids(input_ids, forward_batch)

        if input_embeds.shape[0] == 0:
            return input_embeds

        input_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, input_embeds)

        hidden_states = self.eh_proj(
            torch.cat(
                (
                    self.enorm(input_embeds),
                    self.hnorm(forward_batch.spec_info.hidden_states),
                ),
                dim=-1,
            )
        )
        hidden_states, residual = self.decoder(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=None,
        )
        hidden_states, _ = self.shared_head.norm(hidden_states, residual=residual)
        return hidden_states


class JinaOCRForCausalLMNextN(nn.Module):
    """FastMTP draft model; embedding and lm_head are shared with the target."""

    _stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    _shared_param_names = {"model.embed_tokens.weight", "lm_head.weight"}

    def __init__(
        self,
        *,
        config: DeepseekVLV2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.mtp_num_heads != 1 or not config.mtp_recursive:
            raise ValueError(
                "JinaOCRForCausalLMNextN supports a single recursive FastMTP head; "
                f"got mtp_num_heads={config.mtp_num_heads}, "
                f"mtp_recursive={config.mtp_recursive}"
            )
        self.config = config
        text_config = config.text_config
        self.model = JinaOCRModelNextN(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(text_config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @staticmethod
    def _map_weight_name(name: str) -> Optional[str]:
        """Checkpoint name -> draft parameter name, or None to skip it."""
        if name.startswith(MTP_BLOCK_PREFIX):
            return "model.decoder." + name[len(MTP_BLOCK_PREFIX) :]
        if name.startswith(MTP_HEAD_PREFIX):
            return "model." + name[len(MTP_HEAD_PREFIX) :]
        if name == "model.norm.weight":
            return "model.shared_head.norm.weight"
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            name = self._map_weight_name(name)
            if name is None:
                continue
            for param_name, shard_name, shard_id in self._stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        missing = params_dict.keys() - loaded_params - self._shared_param_names
        if missing:
            raise RuntimeError(
                f"FastMTP weights not found in checkpoint: {sorted(missing)}"
            )


EntryClass = [JinaOCRForCausalLM, JinaOCRForCausalLMNextN]
