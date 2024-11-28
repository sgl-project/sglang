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
"""Constrained decoding with xgrammar backend."""

import logging
from typing import List, Tuple

import torch
from xgrammar import (
    CompiledGrammar,
    GrammarCompiler,
    GrammarMatcher,
    TokenizerInfo,
    allocate_token_bitmask,
    apply_token_bitmask_inplace,
)

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
)

logger = logging.getLogger(__name__)


MAX_ROLLBACK_TOKENS = 200


class XGrammarGrammar(BaseGrammarObject):

    def __init__(
        self, matcher: GrammarMatcher, vocab_size: int, ctx: CompiledGrammar
    ) -> None:
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx

    def accept_token(self, token: int):
        assert self.matcher.accept_token(token)

    def try_jump_forward(self, tokenizer) -> Tuple[List[int], str]:
        s = self.matcher.find_jump_forward_string()
        if s:
            return [], s
        return None

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        _, data = helper
        return data, -1

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        k = 0
        for i, old_id in enumerate(old_output_ids):
            if old_id == new_output_ids[i]:
                k = i + 1
            else:
                break

        # rollback to the last token that is the same
        if k < len(old_output_ids):
            self.matcher.rollback(len(old_output_ids) - k)

        for i in range(k, len(new_output_ids)):
            assert self.matcher.accept_token(new_output_ids[i])

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, vocab_size)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(vocab_mask, idx)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if vocab_mask.device.type != logits.device.type:
            # vocab_mask must then be on the same device as logits
            # when applying the token bitmask, so we check and move if needed
            vocab_mask = vocab_mask.to(logits.device)

        apply_token_bitmask_inplace(logits, vocab_mask)

    def copy(self):
        matcher = GrammarMatcher(self.ctx, max_rollback_tokens=MAX_ROLLBACK_TOKENS)
        return XGrammarGrammar(matcher, self.vocab_size, self.ctx)


class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
    ):
        super().__init__()

        tokenizer_info = TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size
        )
        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
        self.vocab_size = vocab_size

    def init_value_impl(self, key: Tuple[str, str]) -> XGrammarGrammar:

        key_type, key_string = key
        if key_type == "json":
            try:
                ctx = self.grammar_compiler.compile_json_schema(schema=key_string)
            except RuntimeError as e:
                logging.warning(
                    f"Skip invalid json_schema: json_schema={key_string}, {e=}"
                )
                return None
        elif key_type == "regex":
            logger.warning(
                "regex hasn't been supported by xgrammar yet. This is skipped."
            )
            return None
        else:
            raise ValueError(f"Invalid key_type: {key_type}")

        matcher = GrammarMatcher(ctx, max_rollback_tokens=MAX_ROLLBACK_TOKENS)
        return XGrammarGrammar(matcher, self.vocab_size, ctx)

    def reset(self):
        if self.grammar_compiler:
            self.grammar_compiler.clear_cache()
