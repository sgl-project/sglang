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

from outlines.fsm.json_schema import build_regex_from_schema

from sglang.srt.constrained import RegexGuide, TransformerTokenizer
from sglang.srt.constrained.base_tool_cache import BaseToolCache


class FSMCache(BaseToolCache):
    def __init__(
        self,
        tokenizer_path,
        tokenizer_args_dict,
        enable=True,
        skip_tokenizer_init=False,
        json_schema_mode=False,
    ):
        super().__init__(enable=enable)

        self.json_schema_mode = json_schema_mode

        if (
            skip_tokenizer_init
            or tokenizer_path.endswith(".json")
            or tokenizer_path.endswith(".model")
        ):
            # Do not support TiktokenTokenizer or SentencePieceTokenizer
            return

        from importlib.metadata import version

        if version("outlines") >= "0.0.35":
            from transformers import AutoTokenizer

            tokenizer_args_dict.setdefault("padding_side", "left")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, **tokenizer_args_dict
            )
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
        else:
            self.outlines_tokenizer = TransformerTokenizer(
                tokenizer_path, **tokenizer_args_dict
            )

    def init_value(self, value):
        if self.json_schema_mode:
            regex = build_regex_from_schema(value, whitespace_pattern=r"[\n\t ]*")
            return RegexGuide(regex, self.outlines_tokenizer), regex
        else:
            return RegexGuide(value, self.outlines_tokenizer)
