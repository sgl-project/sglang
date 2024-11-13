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
    OutlinesJumpForwardCache,
    OutlinesJumpForwardMap,
)

logger = logging.getLogger(__name__)


class JumpHelper:

    def __init__(self, suffix_ids, cur_state: int = -1) -> None:
        self.suffix_ids = suffix_ids
        self.cur_state = cur_state


class Grammar:
    pass


class OutlinesGrammar(Grammar):
    def __init__(
        self,
        guide: RegexGuide,
        state: int,
        jump_forward_map: Union[OutlinesJumpForwardMap, None],
    ) -> None:
        self.guide = guide
        self.state = state
        self.jump_forward_map = jump_forward_map

    def accept_token(self, token: int):
        self.state = self.guide.get_next_state(self.state, token)

    def try_jump_forward(self, tokenizer) -> JumpHelper:
        jump_forward_bytes = self.jump_forward_map.jump_forward_byte(self.state)
        if jump_forward_bytes is None or len(jump_forward_bytes) <= 1:
            return None

        # preprocess the jump forward string
        suffix_bytes = []
        continuation_range = range(0x80, 0xC0)
        cur_state = self.state
        while (
            len(jump_forward_bytes) and jump_forward_bytes[0][0] in continuation_range
        ):
            # continuation bytes
            byte_edge = jump_forward_bytes.pop(0)
            suffix_bytes.append(byte_edge[0])
            cur_state = byte_edge[1]

        suffix_tokens = [f"<0x{hex(b)[2:].upper()}>" for b in suffix_bytes]
        suffix_ids = tokenizer.convert_tokens_to_ids(suffix_tokens)
        return JumpHelper(suffix_ids=suffix_ids, cur_state=cur_state)

    def jump_forward_str_state(self, helper: JumpHelper) -> Tuple[str, int]:
        return self.jump_forward_map.jump_forward_symbol(helper.cur_state)

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        self.state = next_state

    def fill_vocab_mask(self, vocab_mask: torch.Tensor):
        vocab_mask.fill_(1)
        vocab_mask[self.guide.get_next_instruction(self.state).tokens] = 0


class GrammarBackend:

    def __init__(
        self,
        tokenizer,
        whitespace_patterns: bool,
        allow_jump_forward: bool,
        backend: str,
    ):
        self.executor = ThreadPoolExecutor()
        self.backend = backend
        self.grammar_cache = OutlinesCache(
            tokenizer,
            whitespace_pattern=whitespace_patterns,
        )
        self.jump_forward_cache = (
            OutlinesJumpForwardCache() if allow_jump_forward else None
        )

    def _query(self, key: Tuple[str, str]):
        guide, regex = self.grammar_cache.query(key)
        jump_forward_map = (
            self.jump_forward_cache.query(regex) if self.jump_forward_cache else None
        )
        return OutlinesGrammar(guide, 0, jump_forward_map)

    def query(self, key: Tuple[str, str]) -> Future:
        return self.executor.submit(self._query, key)

    def reset(self):
        self.grammar_cache.reset()
        if self.jump_forward_cache:
            self.jump_forward_cache.reset()
