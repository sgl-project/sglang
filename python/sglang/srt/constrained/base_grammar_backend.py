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
"""The baseclass of a backend for grammar-guided constrained decoding."""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class BaseGrammarObject:

    def __init__(self):
        self._finished = False

    def accept_token(self, token: int) -> None:
        """
        Accept a token in the grammar.
        """
        raise NotImplementedError()

    def rollback(self, k: int):
        raise NotImplementedError()

    def is_terminated(self):
        return False

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        raise NotImplementedError()

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        raise NotImplementedError()

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        raise NotImplementedError()

    def copy(self) -> "BaseGrammarObject":
        return self

    @property
    def finished(self):
        return self._finished

    @finished.setter
    def finished(self, finished):
        self._finished = finished

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        """
        Try to jump forward in the grammar.

        Returns:
            A jump forward helper which may be used in `jump_forward_str_state`.
            None if the jump forward is not possible.
        """
        raise NotImplementedError()

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        """
        Jump forward for the grammar.

        Returns:
            A tuple of the jump forward string and the next state of the grammar
            (which can be used in `jump_and_retokenize` if needed).
        """
        raise NotImplementedError()

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ) -> None:
        """
        Jump forward occurs, and update the grammar state if needed.
        """
        raise NotImplementedError()


INVALID_GRAMMAR_OBJ = BaseGrammarObject()


@dataclass
class CacheEntry:
    value: BaseGrammarObject
    event: Event


class BaseGrammarBackend:
    def __init__(self):
        self.executor = ThreadPoolExecutor()
        self.cache: Dict[Tuple[str, str], CacheEntry] = {}

    def _not_supported(self, key_type: str, key_string: str) -> None:
        logger.warning(f"Skip unsupported {key_type=}, {key_string=}")

    def dispatch_fallback(
        self, key_type: str, key_string: str
    ) -> Optional[BaseGrammarObject]:
        """
        This function should not be reached in any case.
        """
        raise ValueError(f"Invalid key_type: {key_type}={key_string}")

    def dispatch_json(self, key_string: str) -> Optional[BaseGrammarObject]:
        return self._not_supported("json", key_string)

    def dispatch_regex(self, key_string: str) -> Optional[BaseGrammarObject]:
        return self._not_supported("regex", key_string)

    def dispatch_ebnf(self, key_string: str) -> Optional[BaseGrammarObject]:
        return self._not_supported("ebnf", key_string)

    def dispatch_structural_tag(self, key_string: str) -> Optional[BaseGrammarObject]:
        return self._not_supported("structural_tag", key_string)

    def _init_value_dispatch(self, key: Tuple[str, str]) -> Optional[BaseGrammarObject]:
        key_type, key_string = key
        if key_type == "json":
            return self.dispatch_json(key_string)
        elif key_type == "regex":
            return self.dispatch_regex(key_string)
        elif key_type == "ebnf":
            return self.dispatch_ebnf(key_string)
        elif key_type == "structural_tag":
            return self.dispatch_structural_tag(key_string)
        elif key_type == "structural_pattern":
            return self.dispatch_structural_pattern(key_string)
        else:
            return self.dispatch_fallback(key_type, key_string)

    def get_cached_or_future_value(
        self, key: Tuple[str, str]
    ) -> Optional[BaseGrammarObject]:
        value = self.cache.get(key)
        if value:
            return value.copy(), True
        value = self.executor.submit(self._init_value_dispatch, key)
        return value, False

    def set_cache(self, key: Tuple[str, str], value: BaseGrammarObject):
        self.cache[key] = value

    def reset(self):
        self.cache.clear()


def create_grammar_backend(
    server_args: ServerArgs, tokenizer, vocab_size: int
) -> Optional[BaseGrammarBackend]:
    if server_args.grammar_backend == "outlines":
        from sglang.srt.constrained.outlines_backend import OutlinesGrammarBackend

        grammar_backend = OutlinesGrammarBackend(
            tokenizer,
            whitespace_pattern=server_args.constrained_json_whitespace_pattern,
        )
    elif server_args.grammar_backend == "xgrammar":
        from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend

        grammar_backend = XGrammarGrammarBackend(tokenizer, vocab_size=vocab_size)
    elif server_args.grammar_backend == "llguidance":
        from sglang.srt.constrained.llguidance_backend import GuidanceBackend

        grammar_backend = GuidanceBackend(
            tokenizer=tokenizer,
            whitespace_pattern=server_args.constrained_json_whitespace_pattern,
        )
    elif server_args.grammar_backend == "none":
        return None
    else:
        raise ValueError(f"Invalid grammar backend: {server_args.grammar_backend}")

    if server_args.reasoning_parser and hasattr(tokenizer, "think_end_id"):
        from sglang.srt.constrained.reasoner_grammar_backend import (
            ReasonerGrammarBackend,
        )

        grammar_backend = ReasonerGrammarBackend(
            grammar_backend, tokenizer.think_end_id
        )

    return grammar_backend
