"""What an allocation takes from each sub-pool."""

from __future__ import annotations

import dataclasses
from typing import Iterable, Mapping, Optional


@dataclasses.dataclass(frozen=True)
class KVDemand:
    """What an allocation takes: tokens of the token sub-pool, tokens of each
    window sub-pool by its name, and state slots.

    Admission and the take need every window named (`checked`): a window left
    out is an error, not a window that takes nothing, so no window's share is
    guessed. An eviction request may name only the sub-pools it is short of
    (`completed` fills the rest with zero)."""

    full_tokens: int = 0
    window_tokens: Mapping[str, int] = dataclasses.field(default_factory=dict)
    state_slots: int = 0

    @classmethod
    def same_tokens(cls, allocator, tokens: int, *, state_slots: int = 0) -> KVDemand:
        """Every token sub-pool takes ``tokens``: the new tokens of an extend
        or decode step, which every window keeps."""
        return cls(
            tokens,
            {name: tokens for name in allocator.window_pool_names()},
            state_slots,
        )

    def checked(self, window_pool_names: Iterable[str]) -> KVDemand:
        """This demand, once it names exactly the windows listed."""
        names = set(window_pool_names)
        if set(self.window_tokens) != names:
            raise ValueError(
                f"a demand must name every window pool: it names "
                f"{sorted(self.window_tokens)}, the allocator keeps {sorted(names)}"
            )
        if self.full_tokens < 0 or self.state_slots < 0:
            raise ValueError(f"negative demand {self}")
        if any(t < 0 for t in self.window_tokens.values()):
            raise ValueError(f"negative window demand {self}")
        return self

    def completed(self, window_pool_names: Iterable[str]) -> KVDemand:
        """This demand with every listed window named, those it leaves out
        taking nothing; a window it names that is not listed is an error."""
        names = tuple(window_pool_names)
        unknown = set(self.window_tokens) - set(names)
        if unknown:
            raise ValueError(
                f"no window pools {sorted(unknown)}; the allocator keeps "
                f"{sorted(names)}"
            )
        windows = {name: int(self.window_tokens.get(name, 0)) for name in names}
        return dataclasses.replace(self, window_tokens=windows).checked(names)

    def is_empty(self) -> bool:
        return not (
            self.full_tokens > 0
            or self.state_slots > 0
            or any(t > 0 for t in self.window_tokens.values())
        )


def estimate_window_tokens(
    extend_input_len: int,
    max_new_tokens: int,
    *,
    sliding_window_size: Optional[int],
    page_size: int,
    allocation_limit: Optional[int] = None,
    host_hit_length: int = 0,
) -> int:
    """Peak window reservation for one prefill/decode request."""
    if sliding_window_size is None or sliding_window_size <= 0:
        reserved = extend_input_len + max_new_tokens + page_size
    else:
        allocated = (
            extend_input_len
            if allocation_limit is None
            else min(extend_input_len, allocation_limit)
        )
        allocated_tail = max(allocated - sliding_window_size, 0)
        # With a roughly two-window SWA pool, a cached prefix can already lock
        # one window. Charging another whole window for a short resume can then
        # block admission forever on an idle pool. Reserve only its uncached
        # tail plus decode headroom, capped at the window.
        # Including the extension also keeps the reservation large enough for
        # this pass's prefill allocation.
        reserved = (
            allocated_tail
            + min(extend_input_len + max_new_tokens, sliding_window_size)
            + page_size
        )
    if host_hit_length > 0:
        reserved += -(-host_hit_length // page_size) * page_size
    return reserved
