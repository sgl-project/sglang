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

"""Constrained decoding with xgrammar backend."""


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

from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Tuple

import torch

try:
    from xgrammar import CachedGrammarCompiler, CompiledGrammar, GrammarMatcher
except ImportError as e:
    import_error = e

    class Dummy:
        pass

    GrammarMatcher = CompiledGrammar = CachedGrammarCompiler = Dummy


MAX_ROLLBACK_TOKENS = 10


class XGrammarGrammar:

    def __init__(self, matcher: GrammarMatcher, vocab_size: int) -> None:
        self.matcher = matcher
        self.vocab_size = vocab_size

    def accept_token(self, token: int):
        assert self.matcher.accept_token(token)

    def try_jump_forward(self, tokenizer) -> Tuple[List[int], str]:
        return [], self.matcher.find_jump_forward_string()

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        _, data = helper
        return data, -1

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        k = 0
        for i, old_id in enumerate(old_output_ids):
            if old_id == new_output_ids[i]:
                k = i + 1
            else:
                break

        # rollback to the last token that is the same
        if k < len(old_output_ids):
            self.matcher.rollback(len(old_output_ids) - k)

        for i in range(k, len(new_output_ids)):
            assert self.matcher.accept_token(new_output_ids[i])

    def fill_vocab_mask(self, vocab_mask: torch.Tensor):
        # Note that this bitmask is a bitset, not bool
        bitmask = self.matcher.get_next_token_bitmask()
        # Mask the tokens that are not allowed
        vocab_mask[
            self.matcher.get_rejected_tokens_from_bitmask(bitmask, self.vocab_size)
        ] = 1


class XGrammarGrammarBackend:
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
    ):
        self.executor = ThreadPoolExecutor()
        self.grammar_cache = XGrammarCache(tokenizer)
        self.vocab_size = vocab_size

    def _query(self, key: Tuple[str, str]) -> XGrammarGrammar:
        return XGrammarGrammar(self.grammar_cache.query(key), self.vocab_size)

    def query(self, key: Tuple[str, str]) -> Future:
        return self.executor.submit(self._query, key)

    def reset(self):
        self.grammar_cache.reset()


class XGrammarCache:
    def __init__(self, tokenizer, vocab_size: int):
        self.grammar_cache = CachedGrammarCompiler(tokenizer_or_vocab=tokenizer)
        self.vocab_size = vocab_size

    def get_context(self, key: Tuple[str, str]) -> CompiledGrammar:
        key_type, key_string = key
        if key_type == "json":
            return self.grammar_cache.get_compiled_grammar_for_json_schema(key_string)
        elif key_type == "regex":
            raise ValueError("regex hasn't been supported by xgrammar yet")
        else:
            raise ValueError(f"Invalid key_type: {key_type}")

    def query(self, key: Tuple[str, str]) -> GrammarMatcher:
        ctx = self.get_context(key)
        return GrammarMatcher(
            ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            mask_vocab_size=self.vocab_size,
        )

    def reset(self):
        self.grammar_cache.clear()
