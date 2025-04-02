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

from abc import ABC
from concurrent.futures import Future
from typing import Optional, Tuple

import torch

from .base_grammar_backend import BaseGrammarBackend, BaseGrammarObject


class ReasonerGrammarObject(ABC):
    def __init__(self, grammar: Optional[BaseGrammarObject] = None, think_end_id=0):
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


class ReasonerGrammarBackend(ABC):
    def __init__(
        self, grammar_backend: Optional[BaseGrammarBackend] = None, think_end_id=0
    ):
        self.grammar_backend = grammar_backend
        self.think_end_id = think_end_id

    def get_cached_value(self, key: Tuple[str, str]) -> Optional[ReasonerGrammarObject]:
        grammar = self.grammar_backend.get_cached_value(key)
        return ReasonerGrammarObject(grammar, self.think_end_id) if grammar else None

    def get_future_value(self, key: Tuple[str, str]) -> Future:
        grammar = Future()
        self.grammar_backend.get_future_value(key).add_done_callback(
            lambda f: grammar.set_result(
                ReasonerGrammarObject(f.result(), self.think_end_id)
            )
        )
        return grammar

    def reset(self):
        self.grammar_backend.reset()
