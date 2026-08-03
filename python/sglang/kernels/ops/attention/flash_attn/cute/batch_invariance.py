"""Process-wide opt-in to batch-invariant attention.

The forward kernel splits the KV blocks of a query tile into a masked band and
an unmasked remainder, and the boundary is derived from ``seqlen_q`` (see
``BlockInfo.get_n_block_min_causal_local_mask``). The two paths round
differently, so a row's output depends on how many query tokens share the
forward pass -- scoring a context in one prefill and scoring it again as a
cache-hit extend disagree. Callers that need reproducible logprobs turn this on
and give up the unmasked fast path.

This is a process-wide switch rather than a call argument so the flag does not
have to thread through the autograd entry points; it must be set before the
first kernel compile, and it participates in the forward compile cache key.
"""

from __future__ import annotations

_batch_invariant = False


def set_batch_invariant(enabled: bool) -> None:
    global _batch_invariant
    _batch_invariant = bool(enabled)


def is_batch_invariant() -> bool:
    return _batch_invariant
