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
"""Incremental matching of a token sequence against a generated token stream.

A marker such as a reasoning model's think-end token may span several tokens
and arrive across several decode steps, so matching has to survive both. The
match length is carried by the caller, which lets one matcher serve many
independent streams.
"""

from typing import List, Sequence


def build_border_table(pattern: Sequence[int]) -> List[int]:
    """Longest proper prefix that is also a suffix, per prefix length (KMP)."""
    border = [0] * len(pattern)
    k = 0
    for i in range(1, len(pattern)):
        while k > 0 and pattern[i] != pattern[k]:
            k = border[k - 1]
        if pattern[i] == pattern[k]:
            k += 1
        border[i] = k
    return border


class TokenSequenceMatcher:
    """Matches a fixed token sequence incrementally.

    Restarting a failed match from position 0 would miss an occurrence whenever
    the pattern overlaps itself: with pattern [A, A, B] the stream A A A B ends
    on a complete match, but a naive restart drops the second A and never
    reports it. The border table gives the longest prefix still alive after a
    mismatch, so every occurrence is found regardless of the pattern's shape.
    """

    def __init__(self, pattern: Sequence[int]):
        if not pattern:
            raise ValueError("Token sequence matcher needs a non-empty pattern.")
        self.pattern = list(pattern)
        self._border = build_border_table(self.pattern)

    def __len__(self) -> int:
        return len(self.pattern)

    def next_token(self, match_len: int) -> int:
        """The token that would extend a match of the given length."""
        return self.pattern[match_len]

    def advance(self, match_len: int, token: int) -> int:
        """Match length after `token`, or len(pattern) when it completes."""
        while match_len > 0 and token != self.pattern[match_len]:
            match_len = self._border[match_len - 1]
        if token == self.pattern[match_len]:
            match_len += 1
        return match_len
