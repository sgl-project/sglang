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
from xgrammar import (
    GrammarMatcher,
    GrammarMatcherInitContext,
    GrammarMatcherInitContextCache,
)

MAX_ROLLBACK_STEPS = 10


class BNFCache:
    def __init__(
        self,
        tokenizer_path,
        tokenizer_args_dict,
        skip_tokenizer_init=False,
        enable=True,
    ):
        # TODO(dark): determine how to handle with `skip_tokenizer_init`

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, **tokenizer_args_dict
        )
        self.grammar_cache = GrammarMatcherInitContextCache(
            tokenizer_or_vocab=self.tokenizer
        )

    def get_context(self, key) -> GrammarMatcherInitContext:
        key_type, key_string = key

        if key_type == "json":
            return self.grammar_cache.get_init_context_for_json_schema(key_string)
        elif key_type == "regex":
            raise ValueError(f"regex hasn't been supported by xgrammar yet")
        else:
            raise ValueError(f"Invalid key_type: {key_type}")

    def query(self, key) -> GrammarMatcher:
        ctx = self.get_context(key)
        return GrammarMatcher(ctx, max_rollback_steps=MAX_ROLLBACK_STEPS)
