# Copyright 2026 SGLang Team
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

from typing import Dict, Sequence, Tuple


class TokenTextBoundary:
    """Match a byte delimiter across tokens, retaining bytes after its first end."""

    def __init__(
        self, pattern: bytes, vocab: Sequence[bytes], byte_tokens: Dict[bytes, int]
    ):
        self.pattern = pattern
        self.first_byte = pattern[:1]
        self.vocab = vocab
        self.byte_tokens = byte_tokens
        self.prefixes = tuple(pattern[:i] for i in range(len(pattern)))
        self._closing_tokens: Dict[int, Tuple[Tuple[int, Tuple[int, ...]], ...]] = {}

    def advance(self, matched: int, token: int):
        piece = self.vocab[token]
        if matched == 0 and self.first_byte not in piece:
            return 0, None
        text = self.prefixes[matched] + piece
        end = text.find(self.pattern)
        if end >= 0:
            return 0, self.encode_bytes(text[end + len(self.pattern) :])
        for length in range(min(len(text), len(self.pattern) - 1), 0, -1):
            if text.endswith(self.prefixes[length]):
                return length, None
        return 0, None

    def encode_bytes(self, text: bytes) -> Tuple[int, ...]:
        # A byte-level vocabulary covers each individual byte. Greedy pieces
        # preserve incomplete UTF-8 as well as long whitespace tokens.
        result = []
        while text:
            for length in range(len(text), 0, -1):
                token = self.byte_tokens.get(text[:length])
                if token is not None:
                    result.append(token)
                    text = text[length:]
                    break
        return tuple(result)

    def closing_tokens(self, matched: int):
        if matched not in self._closing_tokens:
            prefix = self.prefixes[matched]
            result = []
            for token, piece in enumerate(self.vocab):
                text = prefix + piece
                end = text.find(self.pattern)
                if end >= 0:
                    suffix = self.encode_bytes(text[end + len(self.pattern) :])
                    if suffix:
                        result.append((token, suffix))
            self._closing_tokens[matched] = tuple(result)
        return self._closing_tokens[matched]
