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

import logging
from typing import List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sglang.srt.environ import envs
from sglang.srt.parser.reasoning_parser import ReasoningParser

from .base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)

logger = logging.getLogger(__name__)


class ReasonerGrammarObject(BaseGrammarObject):
    """Wraps a grammar object to handle reasoning (think/generation) phases.

    State machine (must call maybe_init_reasoning before use):
      THINKING (tokens_in_think >= 0, tokens_after_end == -1)
        -> grammar not consulted, optional token filtering
      GENERATION (tokens_after_end >= 0)
        -> grammar consulted for accept/fill/rollback

    When enable_token_filter=True (strict mode), fill_vocab_mask filters
    excluded tokens during THINKING and enforces max_think_tokens budget.
    When the budget is exhausted, only think_end_id is allowed, forcing the
    model to exit the thinking phase.
    When enable_token_filter=False (non-strict mode), fill_vocab_mask is
    a no-op during THINKING.
    """

    def __init__(
        self,
        grammar: Optional[BaseGrammarObject],
        think_end_id: int,
        think_excluded_token_ids: Optional[List[int]] = None,
        max_think_tokens: int = -1,
        enable_token_filter: bool = False,
        token_filter_fn=None,
        allocate_vocab_mask_fn=None,
        move_vocab_mask_fn=None,
        apply_vocab_mask_fn=None,
    ):
        super().__init__()
        self.grammar = grammar
        self.think_end_id = think_end_id
        self.think_excluded_token_ids = think_excluded_token_ids
        self.max_think_tokens = max_think_tokens
        self.enable_token_filter = enable_token_filter
        self.token_filter_fn = token_filter_fn
        self.allocate_vocab_mask_fn = allocate_vocab_mask_fn
        self.move_vocab_mask_fn = move_vocab_mask_fn
        self.apply_vocab_mask_fn = apply_vocab_mask_fn
        self._think_end_id_list = [think_end_id]

        self.tokens_in_think = -1
        self.tokens_after_end = -1

    def maybe_init_reasoning(self, reasoning: bool):
        if reasoning:
            self.tokens_in_think = 0
        else:
            self.tokens_in_think = -1
            self.tokens_after_end = 0

    def _is_thinking(self):
        return self.tokens_in_think >= 0 and self.tokens_after_end == -1

    def _is_generation(self):
        return self.tokens_after_end >= 0

    def transfer_state(self, token: int) -> None:
        if self._is_thinking():
            if token == self.think_end_id:
                self.tokens_after_end = 0
            else:
                self.tokens_in_think += 1
        elif self._is_generation():
            self.tokens_after_end += 1

    def rollback_state(self):
        if self._is_thinking():
            if self.tokens_in_think > 0:
                self.tokens_in_think -= 1
        elif self._is_generation():
            if self.tokens_after_end == 0:
                self.tokens_after_end = -1
            elif self.tokens_after_end > 0:
                self.tokens_after_end -= 1

    def accept_token(self, token: int):
        if self._is_generation() and self.grammar is not None:
            self.grammar.accept_token(token)
        self.transfer_state(token)

    def is_terminated(self):
        if self.grammar is not None:
            return self.grammar.is_terminated()
        return False

    def rollback(self, k):
        if self.grammar is not None:
            steps_after = min(k, max(0, self.tokens_after_end))
            if steps_after > 0:
                self.grammar.rollback(steps_after)
        for _ in range(k):
            self.rollback_state()

    def _can_think_more(self):
        return self.max_think_tokens < 0 or self.tokens_in_think < self.max_think_tokens

    def _do_token_filter(self, vocab_mask, token_ids, idx, is_allowed=True):
        if self.token_filter_fn is not None:
            self.token_filter_fn(vocab_mask, token_ids, idx, is_allowed)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        if self._is_thinking():
            if not self.enable_token_filter:
                return
            if self._can_think_more():
                self._do_token_filter(
                    vocab_mask, self.think_excluded_token_ids, idx, is_allowed=False
                )
            else:
                self._do_token_filter(
                    vocab_mask, self._think_end_id_list, idx, is_allowed=True
                )
            return
        if self._is_generation() and self.grammar is not None:
            self.grammar.fill_vocab_mask(vocab_mask, idx)

    def allocate_vocab_mask(self, vocab_size, batch_size, device):
        if self.grammar is not None:
            return self.grammar.allocate_vocab_mask(vocab_size, batch_size, device)
        if self.allocate_vocab_mask_fn is not None:
            return self.allocate_vocab_mask_fn(vocab_size, batch_size, device)
        return None

    def move_vocab_mask(self, vocab_mask, device):
        if self.grammar is not None:
            return self.grammar.move_vocab_mask(vocab_mask, device)
        if self.move_vocab_mask_fn is not None:
            return self.move_vocab_mask_fn(vocab_mask, device)
        return vocab_mask

    @property
    def apply_vocab_mask(self):
        if self.grammar is not None:
            return self.grammar.apply_vocab_mask
        return self.apply_vocab_mask_fn

    def copy(self):
        new_obj = ReasonerGrammarObject(
            self.grammar.copy() if self.grammar is not None else None,
            self.think_end_id,
            self.think_excluded_token_ids,
            self.max_think_tokens,
            self.enable_token_filter,
            self.token_filter_fn,
            self.allocate_vocab_mask_fn,
            self.move_vocab_mask_fn,
            self.apply_vocab_mask_fn,
        )
        new_obj.tokens_in_think = self.tokens_in_think
        new_obj.tokens_after_end = self.tokens_after_end
        new_obj._finished = self._finished
        return new_obj

    @property
    def finished(self):
        if self.grammar is not None:
            return self.grammar.finished
        return self._finished

    @finished.setter
    def finished(self, finished):
        if self.grammar is not None:
            self.grammar.finished = finished
        else:
            self._finished = finished

    def try_jump_forward(self, tokenizer):
        if self.grammar is not None:
            return self.grammar.try_jump_forward(tokenizer)
        return None

    def jump_forward_str_state(self, helper):
        if self.grammar is not None:
            return self.grammar.jump_forward_str_state(helper)
        return None

    def jump_and_retokenize(self, old_output_ids, new_output_ids, next_state):
        if self.grammar is not None:
            return self.grammar.jump_and_retokenize(
                old_output_ids, new_output_ids, next_state
            )


class ReasonerGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        grammar_backend: BaseGrammarBackend,
        reasoning_parser: ReasoningParser,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        enable_strict_thinking: bool = False,
    ):
        super().__init__()
        self.grammar_backend = grammar_backend
        think_end_ids = tokenizer.encode(
            reasoning_parser.detector.think_end_token, add_special_tokens=False
        )
        if not think_end_ids:
            raise ValueError(
                f"think_end_token '{reasoning_parser.detector.think_end_token}' "
                f"could not be encoded by the tokenizer."
            )
        if len(think_end_ids) != 1:
            raise ValueError(
                f"think_end_token '{reasoning_parser.detector.think_end_token}' "
                "must encode to exactly one token for constrained reasoning."
            )
        self.think_end_id = think_end_ids[0]
        self._enable_strict_thinking = enable_strict_thinking
        self.think_excluded_token_ids = self._get_think_excluded_token_ids(
            reasoning_parser, tokenizer
        )
        self.max_think_tokens = envs.SGLANG_MAX_THINK_TOKENS.get()
        if (
            self.enable_strict_thinking
            and self.think_excluded_token_ids is not None
            and not self.grammar_backend.is_support_token_filter
        ):
            raise ValueError(
                "Strict reasoning format requested but the grammar backend does not "
                "support token filtering. Use a grammar backend that supports token "
                "filtering (e.g., xgrammar) or disable strict reasoning mode."
            )
        self.enable_token_filter = (
            self.enable_strict_thinking
            and self.think_excluded_token_ids is not None
            and self.grammar_backend.is_support_token_filter
        )
        self._token_filter_fn = (
            self.grammar_backend.set_token_filter if self.enable_token_filter else None
        )

    def _get_think_excluded_token_ids(
        self,
        reasoning_parser: ReasoningParser,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> Optional[List[int]]:
        excluded_ids = []
        if (not self.enable_strict_thinking) or (
            not reasoning_parser.detector.think_excluded_tokens
        ):
            return None
        for token in reasoning_parser.detector.think_excluded_tokens:
            new_ids = tokenizer.encode(token, add_special_tokens=False)
            if not new_ids:
                raise ValueError(
                    f"think_excluded_token '{token}' could not be encoded by the "
                    f"tokenizer. All excluded tokens must be encodable for strict "
                    f"reasoning mode to function correctly."
                )
            excluded_ids += new_ids
        return excluded_ids

    def _make_grammar_object(
        self, grammar: Optional[BaseGrammarObject], reasoning: bool
    ) -> ReasonerGrammarObject:
        obj = ReasonerGrammarObject(
            grammar=grammar,
            think_end_id=self.think_end_id,
            think_excluded_token_ids=self.think_excluded_token_ids,
            max_think_tokens=self.max_think_tokens,
            enable_token_filter=self.enable_token_filter,
            token_filter_fn=self._token_filter_fn,
            allocate_vocab_mask_fn=self.grammar_backend.allocate_vocab_mask,
            move_vocab_mask_fn=self.grammar_backend.move_vocab_mask,
            apply_vocab_mask_fn=self.grammar_backend.apply_vocab_mask,
        )
        obj.maybe_init_reasoning(reasoning)
        return obj

    def init_strict_reasoning_grammar(
        self, reasoning: bool
    ) -> Optional[BaseGrammarObject]:
        """Create a grammar object for strict token filtering only (no inner grammar)."""
        if not self.enable_strict_thinking:
            return None
        return self._make_grammar_object(None, reasoning)

    def _init_value_dispatch(
        self, key: Tuple[str, str], reasoning: bool
    ) -> Optional[BaseGrammarObject]:
        ret = self.grammar_backend._init_value_dispatch(key, reasoning)
        if ret is None or isinstance(ret, InvalidGrammarObject):
            return ret
        return self._make_grammar_object(ret, reasoning)
