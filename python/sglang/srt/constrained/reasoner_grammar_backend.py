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
"""The baseclass of a backend for reasoner grammar-guided constrained decoding."""

from typing import List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sglang.srt.environ import envs
from sglang.srt.parser.reasoning_parser import ReasoningParser

from .base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    BaseGrammarBackend,
    BaseGrammarObject,
)


class ReasonerGrammarObject(BaseGrammarObject):
    def __init__(self, grammar: BaseGrammarObject, think_end_id: int):
        super().__init__()
        self.grammar = grammar
        self.think_end_id = think_end_id
        # -1    means thinking has not ended yet
        # 0     means just ended thinking in the last token
        # +     means number of tokens after thinking ended
        self.tokens_after_think_end = -1

    def maybe_init_reasoning(self, reasoning: bool):
        self.tokens_after_think_end = -1 if reasoning else 0

    def transfer_state(self, token: int) -> int:
        if self.tokens_after_think_end == -1 and token == self.think_end_id:
            self.tokens_after_think_end = 0
        elif self.tokens_after_think_end >= 0:
            self.tokens_after_think_end += 1

    def rollback_state(self):
        if self.tokens_after_think_end == 0:
            self.tokens_after_think_end = -1
        elif self.tokens_after_think_end > 0:
            self.tokens_after_think_end -= 1

    def accept_token(self, token: int):
        if self.tokens_after_think_end >= 0:
            self.grammar.accept_token(token)
        self.transfer_state(token)

    def is_terminated(self):
        return self.grammar.is_terminated()

    def rollback(self, k):
        steps_after_think = min(k, self.tokens_after_think_end)
        if steps_after_think > 0:
            self.grammar.rollback(steps_after_think)

        for _ in range(k):
            self.rollback_state()

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return self.grammar.allocate_vocab_mask(vocab_size, batch_size, device)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        if self.tokens_after_think_end >= 0:
            self.grammar.fill_vocab_mask(vocab_mask, idx)

    def move_vocab_mask(self, vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return self.grammar.move_vocab_mask(vocab_mask, device)

    @property
    def apply_vocab_mask(self):
        return self.grammar.apply_vocab_mask

    def copy(self) -> BaseGrammarObject:
        return ReasonerGrammarObject(self.grammar.copy(), self.think_end_id)

    @property
    def finished(self):
        return self.grammar.finished

    @finished.setter
    def finished(self, finished):
        self.grammar.finished = finished

    def try_jump_forward(self, tokenizer):
        return self.grammar.try_jump_forward(tokenizer)

    def jump_forward_str_state(self, helper):
        return self.grammar.jump_forward_str_state(helper)

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        return self.grammar.jump_and_retokenize(
            old_output_ids, new_output_ids, next_state
        )


class StrictReasonerGrammarObject(ReasonerGrammarObject):
    def __init__(
        self,
        grammar: BaseGrammarObject,
        grammar_backend: BaseGrammarBackend,
        think_start_id: int,
        think_end_id: int,
        strict_reasoning_format: bool = False,
        think_excluded_token_ids: Optional[List[int]] = None,
    ):
        super().__init__(grammar, think_end_id)
        self.grammar = grammar
        self.grammar_backend = grammar_backend
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.strict_reasoning_format = strict_reasoning_format
        self.think_excluded_token_ids = think_excluded_token_ids
        self.enable_think_token_filter = (
            strict_reasoning_format
            and think_excluded_token_ids
            and getattr(grammar_backend, "is_support_token_filter", False)
        )
        self.is_init_reasoning = False
        self.max_think_tokens = envs.SGLANG_MAX_THINK_TOKENS.get()
        # -1    means thinking has not began yet
        # 0     means just began thinking in the last token
        # +     means number of tokens in thinking
        self.tokens_in_think = -1
        # -1    means thinking has not ended yet
        # 0     means just ended thinking in the last token
        # +     means number of tokens after thinking ended
        self.tokens_after_think_end = -1

    def maybe_init_reasoning(self, reasoning: bool):
        self.tokens_in_think = 0 if reasoning else -1  # Enforce thinking begin
        self.is_init_reasoning = reasoning

    def _is_initial_state(self):
        return (self.tokens_in_think == -1) and (self.tokens_after_think_end == -1)

    def _is_reasoning_started(self):
        return (self.tokens_in_think >= 0) and (self.tokens_after_think_end == -1)

    def _is_reasoning_ended(self):
        return self.tokens_after_think_end >= 0

    def transfer_state(self, token: int) -> int:
        if self._is_initial_state() and token == self.think_start_id:
            self.tokens_in_think = 0
            return

        if self._is_reasoning_started() and token == self.think_end_id:
            self.tokens_after_think_end = 0
            return

        if self._is_reasoning_started():
            self.tokens_in_think += 1
        elif self._is_reasoning_ended():
            self.tokens_after_think_end += 1

    def rollback_state(self):
        if self._is_initial_state():
            # Nothing to do
            return

        if self._is_reasoning_started():
            if self.tokens_in_think > 0:
                self.tokens_in_think -= 1
            elif self.tokens_in_think == 0:
                # This is too tricky to avoid double think start tokens ...
                self.tokens_in_think = 0 if self.is_init_reasoning else -1
            return
        elif self._is_reasoning_ended():
            if self.tokens_after_think_end == 0:
                self.tokens_after_think_end = -1
            elif self.tokens_after_think_end > 0:
                self.tokens_after_think_end -= 1
            return

    def accept_token(self, token: int):
        if self._is_reasoning_ended():
            self.grammar.accept_token(token)
        self.transfer_state(token)

    def set_token_filter(
        self,
        vocab_mask: torch.Tensor,
        token_ids: List[int],
        idx: int,
        is_allowed: bool = True,
    ):
        if not self.enable_think_token_filter:
            return
        self.grammar_backend.set_token_filter(vocab_mask, token_ids, idx, is_allowed)

    def _can_think_more(self):
        return (self.max_think_tokens < 0) or (
            self.tokens_in_think < self.max_think_tokens
        )

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        if self._is_initial_state():
            # The think_start_token not presented yet, do nothing.
            return

        if self._is_reasoning_started():
            if self._can_think_more():
                # Filter out all the special tokens that breaks reasoning format
                self.set_token_filter(
                    vocab_mask, self.think_excluded_token_ids, idx, is_allowed=False
                )
            else:
                # Break the reasoning by constraint
                self.set_token_filter(
                    vocab_mask,
                    [
                        self.think_end_id,
                    ],
                    idx,
                    is_allowed=True,
                )
            return

        if self._is_reasoning_ended():
            self.grammar.fill_vocab_mask(vocab_mask, idx)

    def copy(self) -> BaseGrammarObject:
        return StrictReasonerGrammarObject(
            self.grammar.copy(),
            self.grammar_backend,
            self.think_start_id,
            self.think_end_id,
            self.strict_reasoning_format,
            self.think_excluded_token_ids,
        )


class ReasonerGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        grammar_backend: BaseGrammarBackend,
        reasoning_parser: ReasoningParser,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ):
        super().__init__()
        self.grammar_backend = grammar_backend
        self.think_start_id = tokenizer.encode(
            reasoning_parser.detector.think_start_token
        )[0]
        self.think_end_id = tokenizer.encode(reasoning_parser.detector.think_end_token)[
            0
        ]
        self.strict_reasoning_format = reasoning_parser.detector.strict_reasoning_format
        self.think_excluded_token_ids = self._get_think_excluded_token_ids(
            reasoning_parser, tokenizer
        )

    def _get_think_excluded_token_ids(
        self,
        reasoning_parser: ReasoningParser,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> Optional[List[int]]:
        excluded_ids = []
        if (not self.strict_reasoning_format) or (
            not reasoning_parser.detector.think_excluded_tokens
        ):
            return None
        for token in reasoning_parser.detector.think_excluded_tokens:
            new_ids = tokenizer.encode(token)
            excluded_ids += new_ids
        return excluded_ids

    def _init_value_dispatch(
        self, key: Tuple[str, str], reasoning: bool
    ) -> Optional[BaseGrammarObject]:
        ret = self.grammar_backend._init_value_dispatch(key, reasoning)
        # avoid wrapping invalid grammar, so that the scheduler can detect it
        if ret is None or ret is INVALID_GRAMMAR_OBJ:
            return ret
        if not self.strict_reasoning_format:
            obj = ReasonerGrammarObject(ret, self.think_end_id)
        else:
            obj = StrictReasonerGrammarObject(
                ret,
                self.grammar_backend,
                self.think_start_id,
                self.think_end_id,
                self.strict_reasoning_format,
                self.think_excluded_token_ids,
            )
        obj.maybe_init_reasoning(reasoning)
        return obj
