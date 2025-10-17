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

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch

from .base_grammar_backend import BaseGrammarBackend, BaseGrammarObject


@dataclass
class _ReasonerHistoryEntry:
    forwarded_count: int
    prev_is_in_reasoning: bool
    prev_start_index: int
    prev_end_index: int
    prev_pending_start: List[int]
    prev_pending_end: List[int]


class ReasonerGrammarObject(BaseGrammarObject):
    def __init__(
        self,
        grammar: BaseGrammarObject,
        think_end_ids: Optional[Union[int, Sequence[int]]] = None,
        think_start_ids: Optional[Sequence[int]] = None,
        initial_in_reasoning: bool = True,
    ):
        super().__init__()
        self.grammar = grammar
        if think_end_ids is None:
            self.think_end_ids: Tuple[int, ...] = ()
        elif isinstance(think_end_ids, int):
            self.think_end_ids = (think_end_ids,)
        else:
            self.think_end_ids = tuple(think_end_ids)
        self.think_start_ids = tuple(think_start_ids or [])
        self.initial_in_reasoning = bool(initial_in_reasoning)
        self.is_in_reasoning = self.initial_in_reasoning
        self._start_match_index = 0
        self._pending_start_tokens: List[int] = []
        self._end_match_index = 0
        self._pending_end_tokens: List[int] = []
        self._history: List[_ReasonerHistoryEntry] = []

    def set_initial_reasoning_state(self, is_in_reasoning: Optional[bool]):
        if is_in_reasoning is None:
            target_state = self.initial_in_reasoning
        else:
            target_state = bool(is_in_reasoning)
        self.is_in_reasoning = target_state
        self._start_match_index = 0
        self._pending_start_tokens.clear()
        self._end_match_index = 0
        self._pending_end_tokens.clear()
        self._history.clear()

    def _process_token(self, token: int, apply_to_grammar: bool) -> None:
        prev_state = _ReasonerHistoryEntry(
            forwarded_count=0,
            prev_is_in_reasoning=self.is_in_reasoning,
            prev_start_index=self._start_match_index,
            prev_end_index=self._end_match_index,
            prev_pending_start=list(self._pending_start_tokens),
            prev_pending_end=list(self._pending_end_tokens),
        )

        if self.is_in_reasoning:
            if self.think_end_ids:
                expected = self.think_end_ids[self._end_match_index]
                if token == expected:
                    self._pending_end_tokens.append(token)
                    self._end_match_index += 1
                    if self._end_match_index == len(self.think_end_ids):
                        self.is_in_reasoning = False
                        self._end_match_index = 0
                        self._pending_end_tokens.clear()
                    self._history.append(prev_state)
                    return
                if self._end_match_index > 0:
                    if token == self.think_end_ids[0]:
                        self._pending_end_tokens = [token]
                        self._end_match_index = 1
                    else:
                        self._pending_end_tokens.clear()
                        self._end_match_index = 0
                    self._history.append(prev_state)
                    return
            self._history.append(prev_state)
            return

        if self.think_start_ids:
            expected = self.think_start_ids[self._start_match_index]
            if token == expected:
                self._pending_start_tokens.append(token)
                self._start_match_index += 1
                if self._start_match_index == len(self.think_start_ids):
                    # Fully matched the start sequence; enter reasoning.
                    self.is_in_reasoning = True
                    self._start_match_index = 0
                    self._pending_start_tokens.clear()
                    self._end_match_index = 0
                    self._pending_end_tokens.clear()
                self._history.append(prev_state)
                return
            if self._start_match_index > 0:
                buffered_tokens = list(self._pending_start_tokens)
                prev_state.forwarded_count += len(buffered_tokens)
                if apply_to_grammar:
                    for buffered_token in buffered_tokens:
                        self.grammar.accept_token(buffered_token)
                self._pending_start_tokens.clear()
                self._start_match_index = 0
                if token == self.think_start_ids[0]:
                    self._pending_start_tokens.append(token)
                    self._start_match_index = 1
                    self._history.append(prev_state)
                    return

        if self._start_match_index == 0:
            prev_state.forwarded_count += 1
            if apply_to_grammar:
                self.grammar.accept_token(token)

        self._history.append(prev_state)

    def accept_token(self, token: int):
        self._process_token(token, apply_to_grammar=True)

    def rollback(self, k: int):
        if k <= 0 or not self._history:
            return
        if k > len(self._history):
            k = len(self._history)
        for _ in range(k):
            entry = self._history.pop()
            if entry.forwarded_count:
                self.grammar.rollback(entry.forwarded_count)
            self.is_in_reasoning = entry.prev_is_in_reasoning
            self._start_match_index = entry.prev_start_index
            self._end_match_index = entry.prev_end_index
            self._pending_start_tokens = list(entry.prev_pending_start)
            self._pending_end_tokens = list(entry.prev_pending_end)

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
            self.think_end_ids,
            list(self.think_start_ids),
            self.initial_in_reasoning,
        )
        copied.is_in_reasoning = self.is_in_reasoning
        copied._start_match_index = self._start_match_index
        copied._pending_start_tokens = list(self._pending_start_tokens)
        copied._end_match_index = self._end_match_index
        copied._pending_end_tokens = list(self._pending_end_tokens)
        copied._history = [
            _ReasonerHistoryEntry(
                forwarded_count=entry.forwarded_count,
                prev_is_in_reasoning=entry.prev_is_in_reasoning,
                prev_start_index=entry.prev_start_index,
                prev_end_index=entry.prev_end_index,
                prev_pending_start=list(entry.prev_pending_start),
                prev_pending_end=list(entry.prev_pending_end),
            )
            for entry in self._history
        ]
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
        self.grammar.jump_and_retokenize(old_output_ids, new_output_ids, next_state)
        self._rebuild_reasoner_state(new_output_ids)

    def _rebuild_reasoner_state(self, output_ids: List[int]) -> None:
        self._history.clear()
        self._pending_start_tokens.clear()
        self._pending_end_tokens.clear()
        self._start_match_index = 0
        self._end_match_index = 0
        self.is_in_reasoning = self.initial_in_reasoning
        for token in output_ids:
            self._process_token(token, apply_to_grammar=False)


class ReasonerGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        grammar_backend: BaseGrammarBackend,
        think_end_ids: Optional[Sequence[int]] = None,
        think_start_ids: Optional[Sequence[int]] = None,
        initial_in_reasoning: bool = True,
    ):
        super().__init__()
        self.grammar_backend = grammar_backend
        if think_end_ids is None:
            self.think_end_ids: Tuple[int, ...] = ()
        else:
            self.think_end_ids = tuple(think_end_ids)
        self.think_start_ids = list(think_start_ids or [])
        self.initial_in_reasoning = initial_in_reasoning

    def _init_value_dispatch(
        self, key: Tuple[str, str]
    ) -> Optional[ReasonerGrammarObject]:
        ret = self.grammar_backend._init_value_dispatch(key)
        if ret is None:
            return None
        if not self.think_end_ids:
            return ret
        return ReasonerGrammarObject(
            ret,
            self.think_end_ids,
            self.think_start_ids,
            self.initial_in_reasoning,
        )
