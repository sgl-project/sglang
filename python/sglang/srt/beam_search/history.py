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
"""Backpointer DAG for beam search history.

The authoritative token history of every beam is an append-only tree of
(parent, token) nodes. Reparenting a beam is attaching its next node under
another beam's leaf, O(1) per step with zero copying; sequences are only
materialized at group finish.
"""

from __future__ import annotations

from typing import List, Optional

import msgspec


class BeamNode(msgspec.Struct):
    """One generated token; the chain of parents is the sequence prefix."""

    token: int
    parent: Optional[BeamNode] = None


def materialize_tokens(leaf: Optional[BeamNode]) -> List[int]:
    """Walk leaf -> root and return the token sequence in generation order."""
    tokens: List[int] = []
    node = leaf
    while node is not None:
        tokens.append(node.token)
        node = node.parent
    tokens.reverse()
    return tokens
