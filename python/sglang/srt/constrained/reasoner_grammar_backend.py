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

from concurrent.futures import Future
from typing import List, Optional, Tuple

import torch

from .base_grammar_backend import BaseGrammarBackend, BaseGrammarObject


class ReasonerGrammarObject(BaseGrammarObject):
    def __init__(self, grammar: BaseGrammarObject, think_end_id):
        super().__init__()
        self.grammar = grammar
        self.think_end_id = think_end_id
        self.is_in_reasoning = True

    @property
    def finished(self):
        return self.grammar.finished

    @finished.setter
    def finished(self, finished):
        self.grammar.finished = finished

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

    def accept_token(self, token: int):
        if token == self.think_end_id:
            self.is_in_reasoning = False

        if not self.is_in_reasoning and token != self.think_end_id:
            self.grammar.accept_token(token)

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

    def copy(self) -> BaseGrammarObject:
        return ReasonerGrammarObject(self.grammar.copy(), self.think_end_id)


class ReasonerGrammarBackend(BaseGrammarBackend):
    def __init__(self, grammar_backend: BaseGrammarBackend, think_end_id):
        self.grammar_backend = grammar_backend
        self.think_end_id = think_end_id

    def get_cached_value(self, key: Tuple[str, str]) -> Optional[ReasonerGrammarObject]:
        grammar = self.grammar_backend.get_cached_value(key)
        return ReasonerGrammarObject(grammar, self.think_end_id) if grammar else None

    def get_future_value(self, key: Tuple[str, str]) -> Future:
        grammar = Future()

        def callback(f: Future):
            if result := f.result():
                grammar.set_result(ReasonerGrammarObject(result, self.think_end_id))
            else:
                grammar.set_result(None)

        self.grammar_backend.get_future_value(key).add_done_callback(callback)
        return grammar

    def reset(self):
        self.grammar_backend.reset()
