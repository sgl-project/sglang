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
"""Token layouts of the tensors handed across layer communication boundaries."""

from enum import Enum, auto
from typing import FrozenSet, Mapping

import msgspec


class TokenAxis(Enum):
    """A parallel dimension across which ranks hold different tokens."""

    ATTN_DP = auto()
    ATTN_CP = auto()
    # Each attention-TP rank holds a slice of its group's tokens.
    ATTN_TP_SCATTER = auto()


class Layout(msgspec.Struct, frozen=True):
    """The token axes a rank's rows are sharded over. Single-rank axes are left
    out, so two layouts are equal exactly when every rank holds the same tokens."""

    sharded: FrozenSet[TokenAxis]

    @classmethod
    def sharded_over(
        cls, *axes: TokenAxis, axis_sizes: Mapping[TokenAxis, int]
    ) -> "Layout":
        return cls(frozenset(axis for axis in axes if axis_sizes[axis] > 1))
