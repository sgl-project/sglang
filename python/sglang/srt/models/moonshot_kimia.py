# Copyright 2025 SGLang Team
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

import logging
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig

from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.transformers import maybe_prefix

logger = logging.getLogger(__name__)


class MoonshotKimiaForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        logger.info(
            "Using MoonshotKimia Transformers backend without SGLang attention."
        )

        self.quant_config = quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size

        self.model = AutoModel.from_config(
            self.config,
            torch_dtype=torch.get_default_dtype(),
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.config._attn_implementation = "sdpa"

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.get_input_embeddings().weight

        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        assert get_embedding is False, "embedding is not supported yet"
        aux_hidden_states = None
        seq_lens = forward_batch.seq_lens
        total_len = input_ids.shape[0]
        dtype = self.model.dtype
        attention_mask = torch.full(
            (total_len, total_len),
            torch.finfo(dtype).min,
            device=input_ids.device,
            dtype=dtype,
        )
        start = 0
        for seq_len in seq_lens.tolist():
            end = start + seq_len
            block = torch.zeros(
                (seq_len, seq_len),
                device=input_ids.device,
                dtype=dtype,
            )
            block = block.masked_fill_(
                torch.triu(
                    torch.ones(
                        (seq_len, seq_len),
                        device=input_ids.device,
                        dtype=torch.bool,
                    ),
                    diagonal=1,
                ),
                torch.finfo(dtype).min,
            )
            attention_mask[start:end, start:end] = block
            start = end
        attention_mask = attention_mask[None, None, ...]
        hidden_states = self.model(
            input_ids[None, ...],
            use_cache=False,
            position_ids=positions[None, ...],
            attention_mask=attention_mask,
            return_dict=False,
        )[0][0, ...]

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def load_weights(self, weights):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name not in params_dict:
                name = f"{self.model.base_model_prefix}.{name}"
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = MoonshotKimiaForCausalLM
