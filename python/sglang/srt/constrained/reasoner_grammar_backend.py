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


class ReasonerGrammarBackend(BaseGrammarBackend):
    def __init__(self, grammar_backend: BaseGrammarBackend, think_end_id):
        super().__init__()
        self.grammar_backend = grammar_backend
        self.think_end_id = think_end_id

    def _init_value_dispatch(
        self, key: Tuple[str, str], reasoning: bool
    ) -> Optional[BaseGrammarObject]:
        ret = self.grammar_backend._init_value_dispatch(key, reasoning)
        # avoid wrapping invalid grammar, so that the scheduler can detect it
        if ret is None or ret is INVALID_GRAMMAR_OBJ:
            return ret
        obj = ReasonerGrammarObject(ret, self.think_end_id)
        obj.maybe_init_reasoning(reasoning)
        return obj
