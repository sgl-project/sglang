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

from interegular import InvalidSyntax, parse_pattern
from outlines.fsm.guide import RegexGuide
from outlines.models.transformers import TransformerTokenizer

from sglang.srt.constrained import build_regex_from_object
from sglang.srt.constrained.base_tool_cache import BaseToolCache

logger = logging.getLogger(__name__)


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
