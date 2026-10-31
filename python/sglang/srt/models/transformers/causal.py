# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team


from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

from .utils import maybe_prefix


class CausalMixin:
    def __init__(self, *args, prefix: str = "", **kwargs):
        super().__init__(*args, prefix=prefix, **kwargs)
        tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", False)
        if tie_word_embeddings:
            self.skip_prefixes.append("lm_head.")
        if not self.pp_group.is_last_rank:
            self._register_missing_prefix("lm_head")
            return
        self.lm_head = ParallelLMHead(
            self.vocab_size,
            self.text_config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if tie_word_embeddings:
            self.lm_head.weight = self.model.get_input_embeddings().weight
        logit_scale = getattr(self.text_config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.text_config, logit_scale=logit_scale
        )
