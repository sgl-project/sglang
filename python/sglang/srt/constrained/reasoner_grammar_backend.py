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
from collections import OrderedDict
from concurrent.futures import Future
from typing import List, Optional, Sequence, Tuple, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sglang.srt.environ import envs
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils.token_sequence_matcher import TokenSequenceMatcher

from .base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)

logger = logging.getLogger(__name__)

_OPEN_ANSWER_VARIANT = "open_answer"
_THINK_BUDGET_VARIANT = "think_budget="
# Arbitrary bound; budgets are a per-request knob, so only recent variants stay cached.
_BUDGET_VARIANT_CACHE_SIZE = 32


class ReasonerGrammarObject(BaseGrammarObject):
    """Defers grammar constraints until the reasoning end sequence is complete."""

    def __init__(
        self,
        grammar: Optional[BaseGrammarObject],
        think_end_ids: Sequence[int],
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
        self.think_end_ids = tuple(think_end_ids)
        self._think_end_matcher = TokenSequenceMatcher(self.think_end_ids)
        self.think_excluded_token_ids = think_excluded_token_ids
        self.max_think_tokens = max_think_tokens
        self.enable_token_filter = enable_token_filter
        self.token_filter_fn = token_filter_fn
        self.allocate_vocab_mask_fn = allocate_vocab_mask_fn
        self.move_vocab_mask_fn = move_vocab_mask_fn
        self.apply_vocab_mask_fn = apply_vocab_mask_fn

        self.tokens_in_think = -1
        self.tokens_after_end = -1
        self._matched_think_end_tokens = 0
        self._thinking_match_history: List[int] = []

    def maybe_init_reasoning(self, reasoning: bool):
        self._matched_think_end_tokens = 0
        self._thinking_match_history.clear()
        if reasoning:
            self.tokens_in_think = 0
            self.tokens_after_end = -1
        else:
            self.tokens_in_think = -1
            self.tokens_after_end = 0

    def set_request_think_end_ids(self, think_end_ids: Sequence[int]) -> None:
        """Select one request-specific reasoning terminator before decoding."""
        token_ids = tuple(think_end_ids)
        if not token_ids:
            raise ValueError("request reasoning terminator must not be empty")
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise ValueError("request reasoning terminator must contain token IDs")
        if (
            self.current_token is not None
            or self._thinking_match_history
            or (self.tokens_in_think, self.tokens_after_end) not in ((0, -1), (-1, 0))
        ):
            raise RuntimeError(
                "request reasoning terminator must be selected before decoding"
            )

        self.think_end_ids = token_ids
        self._think_end_matcher = TokenSequenceMatcher(token_ids)
        self._matched_think_end_tokens = 0

    def _is_thinking(self):
        return self.tokens_in_think >= 0 and self.tokens_after_end == -1

    def _is_generation(self):
        return self.tokens_after_end >= 0

    def transfer_state(self, token: int) -> None:
        if self._is_thinking():
            previous_match = self._matched_think_end_tokens
            self._thinking_match_history.append(previous_match)
            matched = self._think_end_matcher.advance(previous_match, token)
            if matched == len(self._think_end_matcher):
                self._matched_think_end_tokens = 0
                self.tokens_after_end = 0
            else:
                self.tokens_in_think += previous_match + 1 - matched
                self._matched_think_end_tokens = matched
        elif self._is_generation():
            self.tokens_after_end += 1

    def rollback_state(self):
        if self._is_thinking():
            if self._thinking_match_history:
                previous_match = self._thinking_match_history.pop()
                self.tokens_in_think -= (
                    previous_match + 1 - self._matched_think_end_tokens
                )
                self._matched_think_end_tokens = previous_match
        elif self._is_generation():
            if self.tokens_after_end == 0:
                if self._thinking_match_history:
                    self.tokens_after_end = -1
                    self._matched_think_end_tokens = self._thinking_match_history.pop()
            elif self.tokens_after_end > 0:
                self.tokens_after_end -= 1

    def accept_token(self, token: int):
        # Track the last accepted token on the wrapper itself (mirroring
        # XGrammarGrammar.accept_token). Disaggregation's process_prebuilt uses
        # `grammar.current_token is None` to detect a retracted request whose
        # token was already accepted and must not be re-accepted. Without this,
        # a ReasonerGrammarObject's current_token stays None forever (the inner
        # grammar's is updated, not the wrapper's), so the guard never fires and
        # the token is accepted twice -> "Tokens not accepted" -> FINISH_ABORT.
        self.current_token = token
        if self._is_generation() and self.grammar is not None:
            self.grammar.accept_token(token)
        self.transfer_state(token)

    def is_terminated(self):
        if self._is_generation() and self.grammar is not None:
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
        return (
            self.max_think_tokens < 0
            or self.tokens_in_think + self._matched_think_end_tokens
            < self.max_think_tokens
        )

    def _do_token_filter(self, vocab_mask, token_ids, idx, is_allowed=True):
        if self.token_filter_fn is not None:
            self.token_filter_fn(vocab_mask, token_ids, idx, is_allowed)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        if self._is_thinking():
            if not self.enable_token_filter:
                return
            if self._can_think_more():
                if self.think_excluded_token_ids is not None:
                    self._do_token_filter(
                        vocab_mask,
                        self.think_excluded_token_ids,
                        idx,
                        is_allowed=False,
                    )
            else:
                self._do_token_filter(
                    vocab_mask,
                    [self.think_end_ids[self._matched_think_end_tokens]],
                    idx,
                    is_allowed=True,
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
            grammar=self.grammar.copy() if self.grammar is not None else None,
            think_end_ids=self.think_end_ids,
            think_excluded_token_ids=self.think_excluded_token_ids,
            max_think_tokens=self.max_think_tokens,
            enable_token_filter=self.enable_token_filter,
            token_filter_fn=self.token_filter_fn,
            allocate_vocab_mask_fn=self.allocate_vocab_mask_fn,
            move_vocab_mask_fn=self.move_vocab_mask_fn,
            apply_vocab_mask_fn=self.apply_vocab_mask_fn,
        )
        new_obj.tokens_in_think = self.tokens_in_think
        new_obj.tokens_after_end = self.tokens_after_end
        new_obj._matched_think_end_tokens = self._matched_think_end_tokens
        new_obj._thinking_match_history = list(self._thinking_match_history)
        new_obj._finished = self._finished
        new_obj.current_token = self.current_token
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
        if self._is_generation() and self.grammar is not None:
            return self.grammar.try_jump_forward(tokenizer)
        return None

    def jump_forward_str_state(self, helper):
        if self._is_generation() and self.grammar is not None:
            return self.grammar.jump_forward_str_state(helper)
        return None

    def jump_and_retokenize(self, old_output_ids, new_output_ids, next_state):
        if self._is_generation() and self.grammar is not None:
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
        self._budget_variant_keys: OrderedDict[Tuple[str, str], None] = OrderedDict()
        think_end_ids = reasoning_parser.detector.get_think_end_token_ids(tokenizer)
        if not think_end_ids:
            raise ValueError(
                f"think_end_token '{reasoning_parser.detector.think_end_token}' "
                f"could not be encoded by the tokenizer."
            )
        self.think_end_ids = think_end_ids
        self._enable_strict_thinking = enable_strict_thinking
        self._answer_frame = reasoning_parser.detector.get_answer_frame()
        if self._answer_frame is not None:
            self.grammar_backend.reserve_framing_tokens(
                self._answer_frame.framing_token_ids()
            )
        self.think_excluded_token_ids = self._get_think_excluded_token_ids(
            reasoning_parser, tokenizer
        )
        self.max_think_tokens = envs.SGLANG_MAX_THINK_TOKENS.get()
        self.enable_token_filter = self.enable_strict_thinking and (
            self.think_excluded_token_ids is not None or self.max_think_tokens >= 0
        )
        if (
            self.enable_token_filter
            and not self.grammar_backend.is_support_token_filter
        ):
            raise ValueError(
                "Strict reasoning format requested but the grammar backend does not "
                "support token filtering. Use a grammar backend that supports token "
                "filtering (e.g., xgrammar) or disable strict reasoning mode."
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
            think_end_ids=self.think_end_ids,
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

    def get_request_grammar_key(
        self,
        key: Tuple[str, str],
        *,
        input_ids: Sequence[int],
        thinking_budget: Optional[int],
    ) -> Tuple[str, str]:
        key_type, key_string = key
        if not self._is_framed(key_type):
            return key
        if self._answer_frame.ends_in_open_answer_block(input_ids):
            return (f"{key_type}:{_OPEN_ANSWER_VARIANT}", key_string)
        # Mirrors the strict wrapper, which applies a request budget only while filtering.
        if not self.enable_token_filter or thinking_budget is None:
            return key
        budget = max(thinking_budget, -1)
        if budget == max(self.max_think_tokens, -1):
            return key
        return (f"{key_type}:{_THINK_BUDGET_VARIANT}{budget}", key_string)

    def _is_framed(self, key_type: str) -> bool:
        return (
            self._answer_frame is not None
            and key_type in self._answer_frame.framed_key_types
        )

    def get_cached_or_future_value(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> Tuple[BaseGrammarObject | Future[BaseGrammarObject], bool]:
        if key in self._budget_variant_keys:
            self._budget_variant_keys.move_to_end(key)
        return super().get_cached_or_future_value(key, require_reasoning)

    def set_cache(self, key: Tuple[str, str], value: BaseGrammarObject):
        super().set_cache(key, value)
        if _THINK_BUDGET_VARIANT not in key[0]:
            return
        self._budget_variant_keys[key] = None
        self._budget_variant_keys.move_to_end(key)
        while len(self._budget_variant_keys) > _BUDGET_VARIANT_CACHE_SIZE:
            evicted, _ = self._budget_variant_keys.popitem(last=False)
            self.cache.pop(evicted, None)

    def reset(self):
        super().reset()
        self._budget_variant_keys.clear()

    def _init_value_dispatch(
        self, key: Tuple[str, str], reasoning: bool
    ) -> Optional[BaseGrammarObject]:
        key_type, _, variant = key[0].partition(":")
        key = (key_type, key[1])
        ret = self.grammar_backend._init_value_dispatch(key, reasoning)
        if ret is None or isinstance(ret, InvalidGrammarObject):
            return ret
        if key[0] == "full_assistant_ebnf":
            return ret
        if self._is_framed(key[0]):
            # The frame delimits thinking by message kind, so it owns the whole
            # turn; strict thinking only caps the thinking message.
            framed = self._frame_answer(ret, variant)
            if framed is not None:
                return framed
        if (
            not self.enable_strict_thinking
            and key[0] == "structural_tag"
            and self._answer_frame is not None
            and self._answer_frame.owns_turn(key[1])
        ):
            return ret
        return self._make_grammar_object(ret, reasoning)

    def _frame_answer(
        self, grammar: BaseGrammarObject, variant: str
    ) -> Optional[BaseGrammarObject]:
        in_answer_block = variant == _OPEN_ANSWER_VARIANT
        if variant.startswith(_THINK_BUDGET_VARIANT):
            max_think_tokens = int(variant.removeprefix(_THINK_BUDGET_VARIANT))
        elif self.enable_strict_thinking:
            max_think_tokens = self.max_think_tokens
        else:
            max_think_tokens = -1
        return self.grammar_backend.compile_framed_grammar(
            grammar,
            lambda content: self._answer_frame.structural_tag(
                content,
                in_answer_block=in_answer_block,
                max_think_tokens=max_think_tokens,
            ),
        )
