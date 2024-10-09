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

from transformers import AutoTokenizer
from xgrammar import BuiltinGrammar, GrammarStateMatcher

from sglang.srt.constrained.base_tool_cache import BaseToolCache


class BNFCache(BaseToolCache):
    def __init__(
        self,
        tokenizer_path,
        tokenizer_args_dict,
        skip_tokenizer_init=False,
        enable=True,
    ):
        super().__init__(enable=enable)
        if skip_tokenizer_init:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, **tokenizer_args_dict
        )

    def init_value(self, key):
        key_type, key_string = key

        if key_type == "json":
            grammar = BuiltinGrammar.json_schema(key_string)
        elif key_type == "regex":
            assert False, "Not supported by xgrammar yet"
        else:
            raise ValueError(f"Invalid key_type: {key_type}")

        return grammar

    def query(self, key):
        grammar = super().query(key)
        return GrammarStateMatcher(grammar, self.tokenizer)


# class BNFCache(BaseToolCache):
#     def __init__(
#         self,
#         tokenizer_path,
#         tokenizer_args_dict,
#         enable=True,
#         skip_tokenizer_init=False,
#         constrained_json_whitespace_pattern=None,
#     ):
#         super().__init__(enable=enable)

#         if skip_tokenizer_init:
#             return

#         tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_args_dict)
#         self.tokenizer = tokenizer
#         self.constrained_json_whitespace_pattern = constrained_json_whitespace_pattern

#     def init_value(self, key):
#         key_type, key_string = key
#         if key_type == "json":
#             grammar = BuiltinGrammar.json_schema(key_string)
#         elif key_type == "regex":
#             assert False, "Not supported yet"
#         else:
#             raise ValueError(f"Invalid key_type: {key_type}")

#         return GrammarStateMatcher(grammar, self.tokenizer)
