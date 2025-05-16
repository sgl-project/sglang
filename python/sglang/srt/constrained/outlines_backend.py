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
"""Constrained decoding with outlines backend."""

import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import interegular
import torch
from outlines.fsm.guide import RegexGuide
from outlines.models.transformers import TransformerTokenizer
from pydantic import BaseModel

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
)
from sglang.srt.constrained.outlines_jump_forward import OutlinesJumpForwardMap

try:
    from outlines.fsm.json_schema import build_regex_from_schema
except ImportError:
    from outlines_core.fsm.json_schema import build_regex_from_schema


logger = logging.getLogger(__name__)


class OutlinesGrammar(BaseGrammarObject):
    def __init__(
        self,
        guide: RegexGuide,
        jump_forward_map: Union[OutlinesJumpForwardMap, None],
    ) -> None:
        super().__init__()
        self.guide = guide
        self.jump_forward_map = jump_forward_map
        self.state = 0
        self.finished = False

    def accept_token(self, token: int):
        self.state = self.guide.get_next_state(self.state, token)

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        tokens = torch.tensor(
            self.guide.get_next_instruction(self.state).tokens, dtype=torch.int64
        ).to(vocab_mask.device, non_blocking=True)
        vocab_mask = vocab_mask[idx]
        vocab_mask.fill_(1)
        vocab_mask.scatter_(0, tokens, torch.zeros_like(tokens, dtype=torch.bool))

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor):
        logits.masked_fill_(vocab_mask, float("-inf"))

    def copy(self):
        return OutlinesGrammar(self.guide, self.jump_forward_map)

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


class OutlinesGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        whitespace_pattern: bool,
    ):
        super().__init__()

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

    def _compile_regex(self, regex: str) -> Optional[OutlinesGrammar]:
        try:
            if hasattr(RegexGuide, "from_regex"):
                # outlines >= 0.1.1
                guide = RegexGuide.from_regex(regex, self.outlines_tokenizer)
            else:
                # outlines <= 0.0.46
                guide = RegexGuide(regex, self.outlines_tokenizer)
        except interegular.patterns.InvalidSyntax as e:
            logger.warning(f"skip invalid regex schema: {regex=}, {e=}")
            return None

        jump_forward_map = None
        return OutlinesGrammar(guide, jump_forward_map)

    def dispatch_ebnf(self, key_string: str):
        return super().dispatch_ebnf(key_string)

    def dispatch_structural_tag(self, key_string: str):
        return super().dispatch_structural_tag(key_string)

    def dispatch_json(self, key_string: str):
        try:
            regex = build_regex_from_object(
                key_string,
                whitespace_pattern=self.whitespace_pattern,
            )
        except (NotImplementedError, json.decoder.JSONDecodeError, ValueError) as e:
            logger.warning(f"Skip invalid json_schema: {key_string=}, {e=}")
            return None
        return self._compile_regex(regex)

    def dispatch_regex(self, key_string: str):
        return self._compile_regex(key_string)


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
