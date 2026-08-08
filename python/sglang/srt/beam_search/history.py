"""Backpointer DAG for beam search history.

The authoritative token history of every beam is an append-only tree of
(parent, token) nodes. Reparenting a beam is attaching its next node under
another beam's leaf, O(1) per step with zero copying; sequences are only
materialized at group finish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BeamNode:
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


def tail_tokens(leaf: Optional[BeamNode], num_tokens: int) -> List[int]:
    """Up to the last num_tokens tokens ending at leaf, in order. Currently
    exercised only by tests; for a future bounded stop-string check."""
    tokens: List[int] = []
    node = leaf
    while node is not None and len(tokens) < num_tokens:
        tokens.append(node.token)
        node = node.parent
    tokens.reverse()
    return tokens
