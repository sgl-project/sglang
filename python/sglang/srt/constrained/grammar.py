"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Cache for the compressed finite state machine."""
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Tuple, Union

import torch

from sglang.srt.constrained.outlines_cache import OutlinesCache, RegexGuide
from sglang.srt.constrained.outlines_jump_forward import (
    OutlinesJumpCache,
    OutlinesJumpForwardMap,
)
from sglang.srt.constrained.xgrammar_cache import XGrammarCache, XGrammarJumpCache

logger = logging.getLogger(__name__)


class JumpHelper:

    def __init__(
        self, data: Union[List, str] = "", state: int = -1, suffix_ids=[]
    ) -> None:
        self.data: Union[List, str] = data
        self.state: int = state
        self.suffix_ids: List[int] = suffix_ids

    def can_jump(self):
        return len(self.data) > 0


class Grammar:
    pass


class OutlinesGrammar:
    def __init__(
        self,
        grammar: Tuple[RegexGuide, int],
        jump_foward_map: Union[OutlinesJumpForwardMap, None],
    ) -> None:
        self.grammar = grammar
        self.jump_foward_map = jump_foward_map

    def accept_token(self, token: int):
        guide, state = self.grammar
        self.grammar = guide, guide.get_next_state(state, token)

    def try_jump_forward(self, tokenizer) -> JumpHelper:
        _, state = self.grammar
        jump_forward_bytes = self.jump_foward_map.jump_forward_byte(state)
        if jump_forward_bytes is None or len(jump_forward_bytes) == 0:
            return None

        # preprocess the jump forward string
        suffix_bytes = []
        continuation_range = range(0x80, 0xC0)
        cur_state = state
        while (
            len(jump_forward_bytes) and jump_forward_bytes[0][0] in continuation_range
        ):
            # continuation bytes
            byte_edge = jump_forward_bytes.pop(0)
            suffix_bytes.append(byte_edge[0])
            cur_state = byte_edge[1]

        suffix_tokens = [f"<0x{hex(b)[2:].upper()}>" for b in suffix_bytes]
        suffix_ids = tokenizer.convert_tokens_to_ids(suffix_tokens)
        return JumpHelper(suffix_ids, cur_state, suffix_bytes)

    def jump_forward_str_state(self, helper: JumpHelper) -> Tuple[str, int]:
        return self.jump_foward_map.jump_forward_symbol(helper.state)

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        self.grammar = self.grammar[0], next_state

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, vocab_size: int):
        guide, state = self.grammar
        vocab_mask.fill_(1)
        vocab_mask[guide.get_next_instruction(state).tokens] = 0


class GrammarBackend:

    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        whitespace_patterns: bool,
        allow_jump_forward: bool,
        backend: str,
    ):
        self.executor = ThreadPoolExecutor()
        self.backend = backend

        if backend == "xgrammar":
            self.grammar_cache = XGrammarCache(tokenizer, vocab_size)
            self.jump_cache = XGrammarJumpCache() if allow_jump_forward else None
        else:
            assert backend == "outlines"
            self.grammar_cache = OutlinesCache(
                tokenizer,
                whitespace_pattern=whitespace_patterns,
            )
            self.jump_cache = OutlinesJumpCache() if allow_jump_forward else None

    def _query(self, key: Tuple[str, str]) -> Grammar:
        if isinstance(self.grammar_cache, XGrammarCache):
            return Grammar(self.grammar_cache.query(key), self.jump_cache)
        else:
            guide, regex = self.grammar_cache.query(key)
            jump_foward_map = self.jump_cache.query(regex) if self.jump_cache else None
            return OutlinesGrammar((guide, 0), jump_foward_map)

    def query(self, key: Tuple[str, str]) -> Future:
        return self.executor.submit(self._query, key)

    def reset(self):
        self.grammar_cache.reset()
        self.jump_cache.reset()
