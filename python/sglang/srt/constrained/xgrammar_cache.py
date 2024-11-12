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

from typing import Tuple

from transformers import AutoTokenizer

try:
    from xgrammar import CachedGrammarCompiler, CompiledGrammar, GrammarMatcher
except ImportError as e:

    class Dummy:
        pass

    GrammarMatcher = Dummy
    CompiledGrammar = Dummy
    CachedGrammarCompiler = Dummy


MAX_ROLLBACK_TOKENS = 10


class XGrammarJumpCache:
    """A dummy class."""

    def reset(self):
        pass


class XGrammarBackend:
    def __init__(
        self,
        tokenizer_path,
        tokenizer_args_dict,
        skip_tokenizer_init=False,
        whitespace_patterns=None,
    ):
        # TODO(dark): how to deal with whitespace_patterns and skip_tokenizer_init
        if skip_tokenizer_init:
            return

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_args_dict)
        self.grammar_cache: CachedGrammarCompiler = CachedGrammarCompiler(
            tokenizer_or_vocab=tokenizer
        )

    def get_context(self, key: Tuple[str, str]) -> CompiledGrammar:
        key_type, key_string = key
        if key_type == "json":
            return self.grammar_cache.get_compiled_grammar_for_json_schema(key_string)
        elif key_type == "regex":
            raise ValueError("regex hasn't been supported by xgrammar yet")
        else:
            raise ValueError(f"Invalid key_type: {key_type}")

    def query(self, key: Tuple[str, str], vocab_size: int) -> GrammarMatcher:
        ctx = self.get_context(key)
        return GrammarMatcher(
            ctx, max_rollback_tokens=MAX_ROLLBACK_TOKENS, mask_vocab_size=vocab_size
        )

    def reset(self):
        self.grammar_cache.clear()
