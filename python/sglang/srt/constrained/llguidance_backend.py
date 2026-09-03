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
"""Constrained decoding with llguidance backend."""

import json
import logging
from functools import cache
from typing import Iterable, List, NamedTuple, Optional, Tuple, Union

import torch
from llguidance import LLExecutor, LLMatcher, LLTokenizer, StructTag, grammar_from
from llguidance.hf import from_tokenizer
from llguidance.torch import (
    allocate_token_bitmask,
    apply_token_bitmask_inplace,
    fill_next_token_bitmask,
    fill_next_token_bitmask_par,
    fill_next_token_bitmask_par_with_draft_tokens,
)
from transformers import PreTrainedTokenizerFast

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    GrammarRow,
    InvalidGrammarObject,
    register_vocab_mask_buffer,
)
from sglang.srt.constrained.utils import is_legacy_structural_tag
from sglang.srt.utils import get_int_env_var

logger = logging.getLogger(__name__)
_LLGUIDANCE_LOG_LEVEL = get_int_env_var("LLGUIDANCE_LOG_LEVEL", 1)


class GrammarDraftRow(NamedTuple):
    """Grammar, destination block, and tokens for a draft-chain mask fill."""

    base_row: int
    grammar: "GuidanceGrammar"
    draft_tokens: List[int]


@cache
def _get_or_init_mask_executor() -> LLExecutor:
    return LLExecutor()


def fill_token_bitmask_with_draft_tokens(
    entries: List[GrammarDraftRow],
    vocab_mask: torch.Tensor,
) -> None:
    """Fill speculative draft-chain masks with llguidance's native kernel.

    Each matcher is advanced over its legal draft prefix and rolled back by the
    number of consumed parser tokens before returning. Rows after the first
    illegal token retain the caller's all-allow value. Correctness requires
    matcher rollback to restore parser-token state exactly. Matcher errors retain
    the native executor's behavior; this wrapper adds no per-matcher Python probe.
    """
    if not entries:
        return
    matchers = [(e.grammar.ll_matcher, e.base_row, e.draft_tokens) for e in entries]
    fill_next_token_bitmask_par_with_draft_tokens(
        _get_or_init_mask_executor(), matchers, vocab_mask
    )


def fill_token_bitmask_batched(
    entries: List[GrammarRow],
    vocab_mask: torch.Tensor,
) -> None:
    """Fill regular-decode mask rows with llguidance's native kernel."""
    if not entries:
        return
    matchers = [(e.grammar.ll_matcher, e.row) for e in entries]
    fill_next_token_bitmask_par(_get_or_init_mask_executor(), matchers, vocab_mask)


def _normalize_eos_token_ids(
    eos_token_ids: Optional[Union[int, Iterable[int]]],
) -> Optional[Union[int, List[int]]]:
    if eos_token_ids is None or isinstance(eos_token_ids, int):
        return eos_token_ids
    return list(eos_token_ids)


def _create_llguidance_tokenizer(
    tokenizer,
    n_vocab: Optional[int],
    eos_token: Optional[Union[int, List[int]]],
) -> LLTokenizer:
    if isinstance(tokenizer, PreTrainedTokenizerFast):
        backend_tokenizer = tokenizer.backend_tokenizer
        if backend_tokenizer.padding is None and backend_tokenizer.truncation is None:
            return LLTokenizer(
                backend_tokenizer.to_str(),
                n_vocab=n_vocab,
                eos_token=(tokenizer.eos_token_id if eos_token is None else eos_token),
            )
    return from_tokenizer(tokenizer, n_vocab, eos_token=eos_token)


class GuidanceGrammar(BaseGrammarObject):

    def __init__(
        self,
        llguidance_tokenizer: LLTokenizer,
        serialized_grammar: str,
        *,
        ll_matcher: Optional[LLMatcher] = None,
    ):
        super().__init__()
        self.llguidance_tokenizer = llguidance_tokenizer
        self.serialized_grammar = serialized_grammar

        # A request copy reuses the cached template's compiled matcher.
        self.ll_matcher = (
            ll_matcher
            if ll_matcher is not None
            else LLMatcher(
                self.llguidance_tokenizer,
                self.serialized_grammar,
                log_level=_LLGUIDANCE_LOG_LEVEL,
            )
        )
        self._check_err()

        self.eos_tokens = set(self.llguidance_tokenizer.eos_tokens)

    def accept_token(self, token: int):
        if self.finished:
            return
        if self.ll_matcher.is_stopped() and token in self.eos_tokens:
            self.finished = True
            return
        self.ll_matcher.consume_token(token)
        self._check_err()

    def rollback(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            return
        if self.finished:
            self.finished = False
            # EOS token after stop isn't tracked in ll_matcher
            num_tokens -= 1
        self.ll_matcher.rollback(num_tokens)
        self._check_err()

    def is_terminated(self):
        return self.finished

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        fill_next_token_bitmask(self.ll_matcher, vocab_mask, idx)
        self._check_err()

    @staticmethod
    def fill_vocab_mask_batched(
        entries: List[GrammarRow], vocab_mask: torch.Tensor
    ) -> None:
        """Use the native fill when every entry is a plain llguidance grammar."""
        if all(isinstance(entry.grammar, GuidanceGrammar) for entry in entries):
            fill_token_bitmask_batched(entries, vocab_mask)
            return
        BaseGrammarObject.fill_vocab_mask_batched(entries, vocab_mask)

    @staticmethod
    def reset_vocab_mask(vocab_mask: torch.Tensor) -> None:
        if vocab_mask.dtype != torch.int32:
            raise TypeError(
                f"llguidance requires a packed int32 mask, got {vocab_mask.dtype}"
            )
        vocab_mask.fill_(-1)

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, self.llguidance_tokenizer.vocab_size)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        apply_token_bitmask_inplace(logits, vocab_mask)

    def copy(self):
        # Cache templates are pristine, so cloning their matcher creates a fresh
        # request grammar without recompiling the serialized grammar.
        return GuidanceGrammar(
            llguidance_tokenizer=self.llguidance_tokenizer,
            serialized_grammar=self.serialized_grammar,
            ll_matcher=self.ll_matcher.deep_copy(),
        )

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        ff_tokens = self.ll_matcher.compute_ff_tokens()
        if ff_tokens:
            return ff_tokens, ""
        else:
            return None

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        return "", -1

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        pass

    def _check_err(self) -> None:
        if self.ll_matcher.is_error():
            raise ValueError(self.ll_matcher.get_error())


class GuidanceBackend(BaseGrammarBackend):

    def __init__(
        self,
        tokenizer,
        any_whitespace: bool = True,
        whitespace_pattern: Optional[str] = None,
        n_vocab: Optional[int] = None,
        eos_token_ids: Optional[Union[int, Iterable[int]]] = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.any_whitespace = any_whitespace
        self.whitespace_pattern = whitespace_pattern
        self.llguidance_tokenizer = _create_llguidance_tokenizer(
            self.tokenizer,
            n_vocab,
            eos_token=_normalize_eos_token_ids(eos_token_ids),
        )
        # Initialize the shared executor here so the first batched mask fill
        # does not pay its one-time setup cost on the request path.
        _get_or_init_mask_executor()

    def initialize_vocab_mask_buffer(
        self,
        name: str,
        vocab_size: int,
        max_rows: int,
        device,
    ) -> torch.Tensor:
        vocab_mask = allocate_token_bitmask(
            max_rows, self.llguidance_tokenizer.vocab_size
        )
        return register_vocab_mask_buffer(name, vocab_mask, max_rows)

    def _from_serialized(self, serialized_grammar) -> BaseGrammarObject:
        try:
            return GuidanceGrammar(
                llguidance_tokenizer=self.llguidance_tokenizer,
                serialized_grammar=serialized_grammar,
            )
        except Exception as e:
            logger.error(f"Hit invalid grammar: {serialized_grammar=}, {e=}")
            return InvalidGrammarObject(str(e))

    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        try:
            serialized_grammar = LLMatcher.grammar_from_json_schema(
                key_string,
                defaults={
                    "whitespace_flexible": self.any_whitespace,
                    "whitespace_pattern": self.whitespace_pattern,
                },
            )
        except Exception as e:
            logger.error(f"Hit invalid json_schema: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))
        return self._from_serialized(serialized_grammar)

    def dispatch_regex(self, key_string: str) -> BaseGrammarObject:
        serialized_grammar = grammar_from("regex", key_string)
        return self._from_serialized(serialized_grammar)

    def dispatch_ebnf(self, key_string: str) -> BaseGrammarObject:
        try:
            serialized_grammar = grammar_from("ebnf", key_string)
            return self._from_serialized(serialized_grammar)
        except ValueError as e:
            logger.error(f"Hit invalid ebnf: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))

    def dispatch_structural_tag(self, key_string: str) -> BaseGrammarObject:
        try:
            structural_tag = json.loads(key_string)
            assert is_legacy_structural_tag(structural_tag)
            # Pair each structure with a trigger that prefixes its own
            # ``begin`` — StructTag asserts begin.startswith(trigger), and
            # detectors with per-tool triggers (e.g. Inkling's
            # <|message_model|>{name}<|content_invoke_tool_json|>) emit a
            # distinct trigger per tool, so triggers[0] matches only one of
            # them and multi-tool grammars fail to compile.
            triggers = structural_tag["triggers"]
            tags = [
                StructTag(
                    begin=structure["begin"],
                    grammar=structure["schema"],
                    end=structure["end"],
                    trigger=next(
                        (t for t in triggers if structure["begin"].startswith(t)),
                        triggers[0],
                    ),
                )
                for structure in structural_tag["structures"]
            ]
            g = StructTag.to_grammar(tags)
            return self._from_serialized(g)
        except Exception as e:
            logger.error(f"Hit invalid structural_tag: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))
