"""Cross-rank size-agreement logging for CP/DP collectives (row-overwrite case).

The CP gather destinations are sized from EACH RANK'S OWN attn_cp_metadata
(``torch.empty(max_len * cp_size, ...)``), but the collective writes the SUM
of the per-rank contributions into every rank's buffer. torch's
all_gather_into_tensor shape check is local (out == world * in), so a
metadata divergence between the CP pair -- e.g. extend lens disagreeing
after the two ranks' pools/radix trees drift -- passes every local check and
the smaller allocation overflows with gathered bf16 tokens.

The same locally-sized-destination pattern exists on the DP side
(``_dp_gather_via_all_reduce``'s memcpy into the global dp buffer, the
gatherv/reduce_scatterv pair whose per-rank sizes come from
``get_dp_global_num_tokens()``, and the persistent local dp buffer used as
a reduce-scatter output), so those sites log the same way.

SGLANG_CP_SIZE_LOG=1 emits one HOST-ONLY log line per collective: no device
work, no extra collective, so the racing-writer timing is untouched (unlike
every sync-based discriminator). Lines look like::

    [cp-size] zigzag seq=42 rank=0/2 max_len=2048 local_len=2048 \
x_rows=2048 out_rows=4096 ctx=2048,2092

Pair the lines across the ranks of a group by (tag, occurrence order); any
differing field on a paired line is the overflow precondition met. The seq
counter is shared across all tags in lockstep, so a seq mismatch on a paired
line (or a per-tag count mismatch across ranks) means the two ranks took
different code paths -- itself a finding. ``cp_size_pair.py`` at the
workspace root does the pairing, the cross-rank diff, and the single-rank
overflow invariants (start+num>buf_rows; max(sizes)>out_rows).

See docs/cc_read/pcp_cp_strategy_gap_analysis.md (§24).
"""

import hashlib
import itertools
import logging
import os
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

# One counter for every instrumented site: collectives run in lockstep across
# the ranks of a group, so paired lines should carry equal seqs.
_gather_seq = itertools.count()


def _ext_key(forward_batch: Any) -> str:
    """Compact identity of the batch that sized this collective.

    The extend-lens tuple is the divergence vector: identical on a healthy
    CP pair, and the field most likely to differ when the two ranks' cache
    state has drifted. Emitted space-free so the line stays greppable as
    whitespace-separated ``k=v`` tokens.
    """
    ext = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if ext is None:
        return "none"
    ext = tuple(int(v) for v in ext)
    if len(ext) <= 8:
        return ",".join(str(v) for v in ext) if ext else "empty"
    return f"n{len(ext)}:" + hashlib.md5(repr(ext).encode()).hexdigest()[:12]


def log_cp_size_event(
    tag: str,
    rank: int,
    size: int,
    fields: Mapping[str, Any],
    forward_batch: Any = None,
    ctx: Optional[str] = None,
) -> None:
    """Emit one ``[cp-size] tag seq=N rank=R/S k=v ... ctx=...`` line.

    ``rank``/``size`` are the emitting rank's coordinates in whatever group
    runs the collective (CP pair, DP group, ...); ``fields`` become the
    ``k=v`` tokens the pairing script diffs. Keep every value space-free.
    """
    if os.getenv("SGLANG_CP_SIZE_LOG", "0") != "1":
        return
    if ctx is None:
        ctx = _ext_key(forward_batch)
    body = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(
        "[cp-size] %s seq=%d rank=%d/%d %s ctx=%s",
        tag,
        next(_gather_seq),
        rank,
        size,
        body,
        ctx,
    )


def log_cp_collective(
    tag: str,
    cp_rank: int,
    cp_size: int,
    *,
    max_len: int,
    local_len: int,
    x_rows: int,
    out_rows: int,
    ctx: Optional[str] = None,
    forward_batch: Any = None,
) -> None:
    """Gather-site convenience wrapper (destination rows = max_len * cp_size)."""
    log_cp_size_event(
        tag,
        cp_rank,
        cp_size,
        {
            "max_len": max_len,
            "local_len": local_len,
            "x_rows": x_rows,
            "out_rows": out_rows,
        },
        forward_batch=forward_batch,
        ctx=ctx,
    )
