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
from typing import List, Optional, Tuple, Union

import torch

from sglang.srt.constrained import GrammarMatcher, RegexGuide
from sglang.srt.constrained.bnf_cache import BNFCache
from sglang.srt.constrained.fsm_cache import FSMCache
from sglang.srt.constrained.jump_forward import JumpForwardCache, JumpForwardMap

# from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5


class XGrammarJump:
    pass


class JumpHelper:
    data: Union[List, str]
    state: int
    suffix_ids: List[int]

    def __init__(
        self, data: Union[List, str] = "", state: int = -1, suffix_ids=[]
    ) -> None:
        self.data = data
        self.state = state
        self.suffix_ids = suffix_ids

    def can_jump(self):
        return len(self.data) > 0


class Grammar:
    grammar: Union[GrammarMatcher, Tuple[RegexGuide, int]]
    jump_map: Union[XGrammarJump, JumpForwardMap, None]

    def __init__(
        self,
        grammar: Union[GrammarMatcher, Tuple[RegexGuide, int]],
        jump_map: Union[XGrammarJump, JumpForwardMap, None],
    ) -> None:
        self.grammar = grammar
        self.jump_map = jump_map

    def accept_token(self, token: int):
        if isinstance(self.grammar, GrammarMatcher):
            assert self.grammar.accept_token(token)
        else:
            guide, state = self.grammar
            self.grammar = guide, guide.get_next_state(state, token)

    def try_jump(self, tokenizer) -> JumpHelper:
        if isinstance(self.jump_map, XGrammarJump):
            assert isinstance(self.grammar, GrammarMatcher)
            return JumpHelper(self.grammar.find_jump_forward_string())
        elif isinstance(self.jump_map, JumpForwardMap):
            assert isinstance(self.grammar, Tuple)

            _, state = self.grammar
            jump_forward_bytes = self.jump_map.jump_forward_byte(state)
            if jump_forward_bytes is None or len(jump_forward_bytes) == 0:
                return JumpHelper()  # can't jump

            # preprocess the jump forward string
            suffix_bytes = []
            continuation_range = range(0x80, 0xC0)
            cur_state = state
            while (
                len(jump_forward_bytes)
                and jump_forward_bytes[0][0] in continuation_range
            ):
                # continuation bytes
                byte_edge = jump_forward_bytes.pop(0)
                suffix_bytes.append(byte_edge[0])
                cur_state = byte_edge[1]

            suffix_tokens = [f"<0x{hex(b)[2:].upper()}>" for b in suffix_bytes]
            suffix_ids = tokenizer.convert_tokens_to_ids(suffix_tokens)
            return JumpHelper(suffix_ids, cur_state, suffix_bytes)
        else:
            return JumpHelper()  # can't jump

    def jump_forward_str_state(self, helper: JumpHelper) -> Tuple[str, int]:
        if isinstance(helper.data, str):
            return helper.data, -1
        else:
            assert isinstance(self.jump_map, JumpForwardMap)
            return self.jump_map.jump_forward_symbol(helper.state)

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        if isinstance(self.grammar, GrammarMatcher):
            k = 0
            for i, old_id in enumerate(old_output_ids):
                if old_id == new_output_ids[i]:
                    k = i + 1
                else:
                    break

            # rollback to the last token that is the same
            if k < len(old_output_ids):
                self.grammar.rollback(len(old_output_ids) - k)

            for i in range(k, len(new_output_ids)):
                assert self.grammar.accept_token(new_output_ids[i])
        else:
            self.grammar = self.grammar[0], next_state

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, vocab_size: int):
        if isinstance(self.grammar, GrammarMatcher):
            # Note that this bitmask is a bitset, not bool
            bitmask = self.grammar.find_next_token_bitmask()
            # Mask the tokens that are not allowed
            vocab_mask[
                self.grammar.get_rejected_tokens_from_bitmask(bitmask, vocab_size)
            ] = 1
        else:
            guide, state = self.grammar
            vocab_mask.fill_(1)
            vocab_mask[guide.get_next_instruction(state).tokens] = 0


class GrammarCache:
    grammar_cache: Union[BNFCache, FSMCache]
    jump_cache: Union[XGrammarJump, JumpForwardCache, None]

    def __init__(
        self,
        tokenizer_path,
        tokenizer_args_dict,
        skip_tokenizer_init=False,
        whitespace_patterns=None,
        backend=None,
        allow_jump=False,
    ):
        if backend == "xgrammar":
            self.grammar_cache = BNFCache(
                tokenizer_path=tokenizer_path,
                tokenizer_args_dict=tokenizer_args_dict,
                skip_tokenizer_init=skip_tokenizer_init,
                whitespace_patterns=whitespace_patterns,
            )
            self.jump_cache = XGrammarJump() if allow_jump else None
        else:
            assert backend == "outlines"
            self.grammar_cache = FSMCache(
                tokenizer_path=tokenizer_path,
                tokenizer_args_dict=tokenizer_args_dict,
                skip_tokenizer_init=skip_tokenizer_init,
                constrained_json_whitespace_pattern=whitespace_patterns,
                enable=True,
            )
            self.jump_cache = JumpForwardCache() if allow_jump else None

    def query(self, key: Tuple[str, str], vocab_size: int) -> Grammar:
        if isinstance(self.grammar_cache, BNFCache):
            assert not isinstance(self.jump_cache, JumpForwardCache)
            return Grammar(self.grammar_cache.query(key, vocab_size), self.jump_cache)
        else:
            jump_map = None
            guide, regex = self.grammar_cache.query(key)
            if isinstance(self.jump_cache, JumpForwardCache):
                jump_map = self.jump_cache.query(regex)
            return Grammar((guide, 0), jump_map)

    def reset(self):
        if isinstance(self.grammar_cache, FSMCache):
            self.grammar_cache.reset()
        if isinstance(self.jump_cache, JumpForwardCache):
            self.jump_cache.reset()
