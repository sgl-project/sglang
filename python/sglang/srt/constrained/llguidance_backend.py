# Copyright 2023-2024 SGLang Team
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
"""Constrained decoding with llguidance backend."""

import json
import logging
import os
from typing import List, Optional, Tuple

import torch
from llguidance import LLMatcher, LLTokenizer, StructTag, grammar_from
from llguidance.hf import from_tokenizer
from llguidance.torch import (
    allocate_token_bitmask,
    apply_token_bitmask_inplace,
    fill_next_token_bitmask,
)

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
)

logger = logging.getLogger(__name__)


class GuidanceGrammar(BaseGrammarObject):

    def __init__(self, llguidance_tokenizer: LLTokenizer, serialized_grammar: str):
        super().__init__()
        self.llguidance_tokenizer = llguidance_tokenizer
        self.serialized_grammar = serialized_grammar

        self.ll_matcher = LLMatcher(
            self.llguidance_tokenizer,
            self.serialized_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )
        self.finished = False
        self.bitmask = None

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        ff_tokens = self.ll_matcher.compute_ff_tokens()
        if ff_tokens:
            return ff_tokens, ""
        else:
            return None

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        return "", -1

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        pass

    def accept_token(self, token: int):
        if not self.ll_matcher.consume_token(token):
            logger.warning(f"matcher error: {self.ll_matcher.get_error()}")
            self.finished = True

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        if self.ll_matcher.is_stopped():
            self.finished = True

        fill_next_token_bitmask(self.ll_matcher, vocab_mask, idx)

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        if self.bitmask is None or self.bitmask.shape[0] < batch_size:
            # only create bitmask when batch gets larger
            self.bitmask = allocate_token_bitmask(
                batch_size, self.llguidance_tokenizer.vocab_size
            )
            bitmask = self.bitmask
        else:
            bitmask = self.bitmask[:batch_size]

        return bitmask

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        apply_token_bitmask_inplace(logits, vocab_mask)

    def copy(self):
        return GuidanceGrammar(
            llguidance_tokenizer=self.llguidance_tokenizer,
            serialized_grammar=self.serialized_grammar,
        )


class GuidanceBackend(BaseGrammarBackend):

    def __init__(
        self,
        tokenizer,
        whitespace_pattern: Optional[str] = None,
        n_vocab: Optional[int] = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.whitespace_pattern = whitespace_pattern
        self.llguidance_tokenizer = from_tokenizer(self.tokenizer, n_vocab)

    def _from_serialized(self, serialized_grammar) -> Optional[GuidanceGrammar]:
        try:
            return GuidanceGrammar(
                llguidance_tokenizer=self.llguidance_tokenizer,
                serialized_grammar=serialized_grammar,
            )
        except Exception as e:
            logger.warning(f"Skip invalid grammar: {serialized_grammar}, {e=}")
            return None

    def dispatch_json(self, key_string: str) -> Optional[GuidanceGrammar]:
        serialized_grammar = LLMatcher.grammar_from_json_schema(
            key_string,
            defaults={
                "whitespace_pattern": self.whitespace_pattern,
            },
        )
        return self._from_serialized(serialized_grammar)

    def dispatch_regex(self, key_string: str) -> Optional[GuidanceGrammar]:
        serialized_grammar = grammar_from("regex", key_string)
        return self._from_serialized(serialized_grammar)

    def dispatch_ebnf(self, key_string: str) -> Optional[GuidanceGrammar]:
        try:
            serialized_grammar = grammar_from("ebnf", key_string)
            return self._from_serialized(serialized_grammar)
        except ValueError as e:
            logger.warning(f"Skip invalid ebnf: regex={key_string}, {e=}")
            return None

    def dispatch_structural_tag(self, key_string: str) -> Optional[GuidanceGrammar]:
        try:
            structural_tag = json.loads(key_string)
            tags = [
                StructTag(
                    begin=structure["begin"],
                    grammar=structure["schema"],
                    end=structure["end"],
                    trigger=structural_tag["triggers"][0],  # TODO?
                )
                for structure in structural_tag["structures"]
            ]
            g = StructTag.to_grammar(tags)
            return self._from_serialized(g)
        except Exception as e:
            logging.warning(f"Skip invalid structural_tag: {key_string}, {e=}")
            return None
