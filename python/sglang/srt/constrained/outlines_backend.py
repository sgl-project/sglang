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
from typing import Dict, Optional, Union

import interegular
import torch
from outlines.fsm.guide import RegexGuide
from outlines.models.transformers import TransformerTokenizer
from pydantic import BaseModel

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)
from sglang.srt.constrained.json_schema_validation import (
    validate_outlines_json_schema,
)

try:
    from outlines.fsm.json_schema import build_regex_from_schema
except ImportError:
    from outlines_core.fsm.json_schema import build_regex_from_schema


logger = logging.getLogger(__name__)


class OutlinesGrammar(BaseGrammarObject):
    def __init__(self, guide: RegexGuide) -> None:
        super().__init__()
        self.guide = guide
        self.state = 0

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
        return OutlinesGrammar(self.guide)


class OutlinesGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        whitespace_pattern: str | None,
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

    def _compile_regex(self, regex: str) -> BaseGrammarObject:
        try:
            if hasattr(RegexGuide, "from_regex"):
                # outlines >= 0.1.1
                guide = RegexGuide.from_regex(regex, self.outlines_tokenizer)
            else:
                # outlines <= 0.0.46
                guide = RegexGuide(regex, self.outlines_tokenizer)
        except interegular.patterns.InvalidSyntax as e:
            logger.error(f"Hit invalid regex schema: {regex=}, {e=}")
            return InvalidGrammarObject(str(e))

        return OutlinesGrammar(guide)

    def dispatch_ebnf(self, key_string: str):
        return super().dispatch_ebnf(key_string)

    def dispatch_structural_tag(self, key_string: str):
        return super().dispatch_structural_tag(key_string)

    def dispatch_json(self, key_string: str):
        try:
            validate_outlines_json_schema(json.loads(key_string))
            regex = build_regex_from_object(
                key_string,
                whitespace_pattern=self.whitespace_pattern,
            )
        except (
            NotImplementedError,
            json.decoder.JSONDecodeError,
            ValueError,
        ) as e:
            logger.error(f"Hit invalid json_schema: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))
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
