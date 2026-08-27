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

from typing import Sequence


class TokenSequenceMatcher:
    def __init__(self, pattern: Sequence[int]):
        if not pattern:
            raise ValueError("pattern must contain at least one token")
        self.pattern = tuple(pattern)
        self.prefix_lengths = self._build_prefix_lengths()

    def _build_prefix_lengths(self) -> tuple[int, ...]:
        prefix_lengths = [0] * len(self.pattern)
        matched = 0
        for index in range(1, len(self.pattern)):
            while matched > 0 and self.pattern[index] != self.pattern[matched]:
                matched = prefix_lengths[matched - 1]
            if self.pattern[index] == self.pattern[matched]:
                matched += 1
            prefix_lengths[index] = matched
        return tuple(prefix_lengths)

    def __len__(self) -> int:
        return len(self.pattern)

    def advance(self, matched: int, token: int) -> int:
        while matched > 0 and token != self.pattern[matched]:
            matched = self.prefix_lengths[matched - 1]
        if token == self.pattern[matched]:
            matched += 1
        return matched
