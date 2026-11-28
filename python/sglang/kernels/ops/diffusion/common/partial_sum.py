# SPDX-License-Identifier: Apache-2.0
"""A two-rank row-parallel output left unreduced for the kernel that reads it.

The all-reduce after a row-parallel layer on two ranks leaves, at every
element, the BF16 rounding of the two ranks' partial sums added in fp32. An
epilogue kernel that loads its input through ``load_partial_sum`` with
``HAS_PEER`` set reads both partials and rounds their sum the same way, so it
writes the bytes it would have written after the all-reduce, and the separate
reduce pass and its transfer are gone.
"""

from typing import NamedTuple

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


class PartialSum(NamedTuple):
    """This rank's and the peer's partial sums of a row-parallel layer's output,
    before the all-reduce; ``theirs`` may be mapped from the peer's device. Both
    are contiguous and have the output's shape."""

    mine: torch.Tensor
    theirs: torch.Tensor

    def reduce(self) -> torch.Tensor:
        """The all-reduced output."""
        return self.mine + self.theirs


def split_partials(x: "torch.Tensor | PartialSum"):
    """``(mine, theirs)`` for a ``PartialSum``, ``(x, None)`` for a tensor."""
    if isinstance(x, PartialSum):
        return x.mine, x.theirs
    return x, None


@triton.jit
def load_partial_sum(ptrs, peer_ptrs, mask, HAS_PEER: tl.constexpr):
    """The value at ``ptrs``; with ``HAS_PEER``, the sum of the partials at
    ``ptrs`` and ``peer_ptrs`` rounded as a two-rank all-reduce leaves it."""
    x = tl.load(ptrs, mask=mask, other=0.0)
    if HAS_PEER:
        y = tl.load(peer_ptrs, mask=mask, other=0.0)
        x = (x.to(tl.float32) + y.to(tl.float32)).to(x.dtype)
    return x
