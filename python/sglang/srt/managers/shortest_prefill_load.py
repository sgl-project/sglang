"""Load-scaled shortest-prefill-first: a class boundary and a prefill token budget
that follow the engine's prefill backlog.

The backlog B is the uncached prompt tokens of the waiting requests plus what a
continuing chunked prefill has left, per DP attention rank. Requests below the
class boundary go first, shortest first, and may join a continuing chunk; the rest
follow in arrival order and never join one. The boundary stays at the chunk size
C until the backlog passes the decay start, then halves every K chunks of
backlog, so under overload the order tends to first-come-first-served. The token
budget caps each prefill batch at min_ratio * C * 2 ** (B / (D * C)) prefill
tokens, with no cap once it reaches C. The flags are in units of C, so the policy
keeps its shape when the chunk size changes.
"""

import math
from typing import Optional

import msgspec


class ShortestPrefillLoadScaling(msgspec.Struct, frozen=True, kw_only=True):
    # All in tokens: the flags' chunk units times the per-rank chunk size.
    chunk_tokens: int
    halving_tokens: float
    decay_start_tokens: float
    # 0 when the budget is off.
    budget_min_tokens: float
    budget_doubling_tokens: float
    # The backlog from which the budget reaches the chunk size and stops capping.
    budget_full_backlog: float

    @classmethod
    def resolve(
        cls,
        *,
        chunked_prefill_size: Optional[int],
        halving_chunks: float,
        decay_start_chunks: Optional[float],
        budget_min_ratio: Optional[float],
        budget_doubling_chunks: Optional[float],
    ) -> Optional["ShortestPrefillLoadScaling"]:
        """None when off. Unset values derive from K: the decay starts at 2 * K
        chunks, the budget's minimum ratio is 1/8, and it doubles every K chunks."""
        if not halving_chunks:
            return None
        if not chunked_prefill_size:
            # The scheduler can still turn chunked prefill off for some models.
            raise ValueError(
                "--shortest-prefill-boundary-halving-chunks requires chunked prefill"
            )
        if decay_start_chunks is None:
            decay_start_chunks = 2 * halving_chunks
        if budget_min_ratio is None:
            budget_min_ratio = 1 / 8
        if budget_doubling_chunks is None:
            budget_doubling_chunks = halving_chunks
        chunk = chunked_prefill_size
        return cls(
            chunk_tokens=chunk,
            halving_tokens=halving_chunks * chunk,
            decay_start_tokens=decay_start_chunks * chunk,
            budget_min_tokens=budget_min_ratio * chunk,
            budget_doubling_tokens=budget_doubling_chunks * chunk,
            budget_full_backlog=(
                budget_doubling_chunks * chunk * math.log2(1 / budget_min_ratio)
                if budget_min_ratio
                else 0.0
            ),
        )

    def class_boundary(self, backlog: int) -> float:
        """Uncached size from which a request goes by arrival."""
        excess = max(0, backlog - self.decay_start_tokens)
        return self.chunk_tokens * 2.0 ** (-excess / self.halving_tokens)

    def token_budget(
        self, backlog: int, *, page_size: int, align: Optional[int]
    ) -> Optional[int]:
        """Prefill tokens one batch may carry, or None for no cap. Rounded down to
        the page and the chunk alignment, at least one of them: a chunk below the
        alignment truncates to zero tokens."""
        if backlog >= self.budget_full_backlog:
            return None
        budget = int(
            self.budget_min_tokens * 2.0 ** (backlog / self.budget_doubling_tokens)
        )
        unit = page_size if align is None else math.lcm(page_size, align)
        return max(unit, budget // unit * unit)
