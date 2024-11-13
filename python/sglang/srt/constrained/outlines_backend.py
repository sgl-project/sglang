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

"""Constrained decoding with outlines backend."""

import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import torch
from interegular import InvalidSyntax, parse_pattern
from outlines.fsm.guide import RegexGuide
from outlines.models.transformers import TransformerTokenizer
from pydantic import BaseModel

from sglang.srt.constrained.base_tool_cache import BaseToolCache
from sglang.srt.constrained.outlines_jump_forward import (
    OutlinesJumpForwardCache,
    OutlinesJumpForwardMap,
)

logger = logging.getLogger(__name__)


try:
    from outlines.fsm.json_schema import build_regex_from_object
except ImportError:
    # Since outlines 0.0.32, build_regex_from_object is replaced by build_regex_from_schema,
    # which only accepts string schema as input.
    from outlines.fsm.json_schema import build_regex_from_schema

    def build_regex_from_object(
        object: Union[str, BaseModel, Dict], whitespace_pattern: Optional[str] = None
    ):
        if isinstance(object, type(BaseModel)):
            schema = json.dumps(object.model_json_schema())
        elif isinstance(object, Dict):
            schema = json.dumps(object)
        else:
            schema = object
        return build_regex_from_schema(schema, whitespace_pattern)


class OutlinesGrammar:
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

    def try_jump_forward(self, tokenizer) -> Optional[Tuple]:
        if not self.jump_forward_map:
            return None

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
        return suffix_ids, cur_state

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        _, cur_state = helper
        return self.jump_forward_map.jump_forward_symbol(cur_state)

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        self.state = next_state

    def fill_vocab_mask(self, vocab_mask: torch.Tensor):
        vocab_mask.fill_(1)
        vocab_mask[self.guide.get_next_instruction(self.state).tokens] = 0


class OutlinesGrammarBackend:
    def __init__(
        self,
        tokenizer,
        whitespace_patterns: bool,
        allow_jump_forward: bool,
    ):
        self.executor = ThreadPoolExecutor()
        self.grammar_cache = OutlinesCache(
            tokenizer,
            whitespace_pattern=whitespace_patterns,
        )
        self.jump_forward_cache = (
            OutlinesJumpForwardCache() if allow_jump_forward else None
        )

    def _query(self, key: Tuple[str, str]) -> OutlinesGrammar:
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


class OutlinesCache(BaseToolCache):
    def __init__(
        self,
        tokenizer,
        whitespace_pattern=None,
    ):
        super().__init__(enable=True)

        try:
            self.outlines_tokenizer = TransformerTokenizer(tokenizer)
        except AttributeError:
            # FIXME: tmp fix for chatglm2 & chatglm3 (pad_token_id=0)
            origin_pad_token_id = tokenizer.pad_token_id

            def fset(self, value):
                self._value = value

            type(tokenizer).pad_token_id = property(
                fget=type(tokenizer).pad_token_id.fget, fset=fset
            )
            self.outlines_tokenizer = TransformerTokenizer(tokenizer)
            self.outlines_tokenizer.tokenizer.pad_token_id = origin_pad_token_id
            self.outlines_tokenizer.pad_token_id = origin_pad_token_id
            self.outlines_tokenizer.pad_token = (
                self.outlines_tokenizer.tokenizer.pad_token
            )
            self.outlines_tokenizer.vocabulary = (
                self.outlines_tokenizer.tokenizer.get_vocab()
            )
        self.whitespace_pattern = whitespace_pattern

    def init_value(self, key):
        key_type, key_string = key
        if key_type == "json":
            try:
                regex = build_regex_from_object(
                    key_string,
                    whitespace_pattern=self.whitespace_pattern,
                )
            except NotImplementedError as e:
                logger.warning(
                    f"skip invalid json schema: json_schema={key_string}, {e=}"
                )
                return None, key_string
        elif key_type == "regex":
            regex = key_string
        else:
            raise ValueError(f"Invalid key_type: {key_type}")
        try:
            parse_pattern(regex)
        except InvalidSyntax as e:
            logger.warning(f"skip invalid regex guide: {regex=}, {e=}")
            return None, regex

        ret = RegexGuide(regex, self.outlines_tokenizer), regex
        return ret

    def _query(self, key: Tuple[str, str]):
        guide, regex = self.grammar_cache.query(key)
        jump_forward_map = (
            self.jump_forward_cache.query(regex) if self.jump_forward_cache else None
        )
        return OutlinesGrammar(guide, 0, jump_forward_map)
