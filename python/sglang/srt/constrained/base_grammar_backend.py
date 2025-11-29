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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Event
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class GrammarStats:
    compilation_time: Optional[float] = None
    schema_count: Optional[int] = None
    ebnf_size: Optional[int] = None
    is_cache_hit: bool = False
    is_grammar_aborted: bool = False
    tree_traversal_time: List[float] = field(default_factory=list)
    dispatch_type: Optional[str] = None
    num_timeout: int = 0


class BaseGrammarObject:

    def __init__(self):
        self._finished = False
        self.grammar_stats = None
        self.current_token = None

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
        s = time.perf_counter()
        key_type, key_string = key
        if key_type == "json":
            grammar = self.dispatch_json(key_string)
        elif key_type == "regex":
            grammar = self.dispatch_regex(key_string)
        elif key_type == "ebnf":
            grammar = self.dispatch_ebnf(key_string)
        elif key_type == "structural_tag":
            grammar = self.dispatch_structural_tag(key_string)
        elif key_type == "structural_pattern":
            grammar = self.dispatch_structural_pattern(key_string)
        elif key_type == "structural_pattern_v2":
            grammar = self.dispatch_structural_pattern_v2(key_string)
        else:
            grammar = self.dispatch_fallback(key_type, key_string)

        if grammar is not None and grammar.grammar_stats is not None:
            grammar.grammar_stats.compilation_time = time.perf_counter() - s
        return grammar

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


GRAMMAR_BACKEND_REGISTRY = {}


def register_grammar_backend(name, init_func):
    GRAMMAR_BACKEND_REGISTRY[name] = init_func


def create_grammar_backend(
    server_args: ServerArgs,
    tokenizer,
    vocab_size: int,
    eos_token_ids: Optional[set] = None,
) -> Optional[BaseGrammarBackend]:
    name = server_args.grammar_backend

    # Custom grammar backend has the highest priority
    if name in GRAMMAR_BACKEND_REGISTRY:
        return GRAMMAR_BACKEND_REGISTRY[name](
            server_args, tokenizer, vocab_size, eos_token_ids
        )

    # Default grammar backends
    if name == "outlines":
        from sglang.srt.constrained.outlines_backend import OutlinesGrammarBackend

        grammar_backend = OutlinesGrammarBackend(
            tokenizer,
            whitespace_pattern=server_args.constrained_json_whitespace_pattern,
        )
    elif name == "xgrammar":
        from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend

        # Convert Set[int] to List[int] if needed
        eos_list = list(eos_token_ids) if eos_token_ids else None

        grammar_backend = XGrammarGrammarBackend(
            tokenizer,
            vocab_size=vocab_size,
            model_eos_token_ids=eos_list,
            any_whitespace=not server_args.constrained_json_disable_any_whitespace,
        )
    elif name == "llguidance":
        from sglang.srt.constrained.llguidance_backend import GuidanceBackend

        grammar_backend = GuidanceBackend(
            tokenizer=tokenizer,
            any_whitespace=not server_args.constrained_json_disable_any_whitespace,
            whitespace_pattern=server_args.constrained_json_whitespace_pattern,
        )
    elif name == "none":
        return None
    else:
        raise ValueError(f"Invalid grammar backend: {name}")

    if server_args.reasoning_parser and hasattr(tokenizer, "think_end_id"):
        from sglang.srt.constrained.reasoner_grammar_backend import (
            ReasonerGrammarBackend,
        )

        grammar_backend = ReasonerGrammarBackend(
            grammar_backend, tokenizer.think_end_id
        )

    return grammar_backend
