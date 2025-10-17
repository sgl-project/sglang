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

from typing import List, Optional, Tuple

import torch

from .base_grammar_backend import BaseGrammarBackend, BaseGrammarObject


class ReasonerGrammarObject(BaseGrammarObject):
    def __init__(
        self,
        grammar: BaseGrammarObject,
        think_end_id: Optional[int],
        think_start_ids: Optional[List[int]] = None,
        initial_in_reasoning: bool = True,
    ):
        super().__init__()
        self.grammar = grammar
        self.think_end_id = think_end_id
        self.think_start_ids = tuple(think_start_ids or [])
        self.initial_in_reasoning = bool(initial_in_reasoning)
        self.is_in_reasoning = self.initial_in_reasoning
        self._start_match_index = 0
        self._pending_start_tokens: List[int] = []

    def set_initial_reasoning_state(self, is_in_reasoning: Optional[bool]):
        if is_in_reasoning is None:
            target_state = self.initial_in_reasoning
        else:
            target_state = bool(is_in_reasoning)
        self.is_in_reasoning = target_state
        self._start_match_index = 0
        self._pending_start_tokens.clear()

    def accept_token(self, token: int):
        if self.think_end_id is not None and token == self.think_end_id:
            self.is_in_reasoning = False
            self._start_match_index = 0
            self._pending_start_tokens.clear()
            return

        if self.is_in_reasoning:
            return

        if self.think_start_ids:
            expected = self.think_start_ids[self._start_match_index]
            if token == expected:
                self._pending_start_tokens.append(token)
                self._start_match_index += 1
                if self._start_match_index == len(self.think_start_ids):
                    # We have fully matched the start token sequence; enter reasoning.
                    self.is_in_reasoning = True
                    self._start_match_index = 0
                    self._pending_start_tokens.clear()
                    return
            else:
                if self._start_match_index > 0:
                    # Flush previously buffered tokens since the sequence broke.
                    for buffered_token in self._pending_start_tokens:
                        self.grammar.accept_token(buffered_token)
                    self._pending_start_tokens.clear()
                    self._start_match_index = 0
                    # The current token might be the first token of the sequence again.
                    if token == self.think_start_ids[0]:
                        self._pending_start_tokens.append(token)
                        self._start_match_index = 1
                        return

        if self._start_match_index == 0:
            self.grammar.accept_token(token)

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return self.grammar.allocate_vocab_mask(vocab_size, batch_size, device)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        if not self.is_in_reasoning:
            self.grammar.fill_vocab_mask(vocab_mask, idx)

    def move_vocab_mask(self, vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return self.grammar.move_vocab_mask(vocab_mask, device)

    @property
    def apply_vocab_mask(self):
        return self.grammar.apply_vocab_mask

    def copy(self) -> BaseGrammarObject:
        copied = ReasonerGrammarObject(
            self.grammar.copy(),
            self.think_end_id,
            list(self.think_start_ids),
            self.initial_in_reasoning,
        )
        copied.is_in_reasoning = self.is_in_reasoning
        copied._start_match_index = self._start_match_index
        copied._pending_start_tokens = list(self._pending_start_tokens)
        return copied

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


class ReasonerGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        grammar_backend: BaseGrammarBackend,
        think_end_id: Optional[int],
        think_start_ids: Optional[List[int]] = None,
        initial_in_reasoning: bool = True,
    ):
        super().__init__()
        self.grammar_backend = grammar_backend
        self.think_end_id = think_end_id
        self.think_start_ids = think_start_ids or []
        self.initial_in_reasoning = initial_in_reasoning

    def _init_value_dispatch(
        self, key: Tuple[str, str]
    ) -> Optional[ReasonerGrammarObject]:
        ret = self.grammar_backend._init_value_dispatch(key)
        if ret is None:
            return None
        if self.think_end_id is None:
            return ret
        return ReasonerGrammarObject(
            ret,
            self.think_end_id,
            self.think_start_ids,
            self.initial_in_reasoning,
        )
