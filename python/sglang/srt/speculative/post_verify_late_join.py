"""Late join of the post-verify side-stream state commit, with a guard on every other reader and writer.

With SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH the Mamba state commit replays on a side stream, and EAGLE v2 joins
it right after the draft extend. SGLANG_OPT_POST_VERIFY_LATE_JOIN=1 joins it before the next verify or prefill
instead, so it overlaps the draft extend, scheduling and the next draft, and orders every reader or writer of state the
pending commit may touch outside that window after it: state-slot hand-out and row binding, HiCache loads and backups,
retraction, a loop pass that launches nothing, control requests and cache flushes. The values each kernel reads and
writes are those of the early join; only their ordering against the commit moves.

The guard covers aggregated serving at TP1 and PP1 with the overlap scheduler and the Mamba extra buffer, without the
unified memory pool or int8 Mamba checkpoints; elsewhere a warning, and the early join stays. Once active, the
post-verify graph pool is also presized (as SGLANG_OPT_POST_VERIFY_POOL_PRESIZE), and the side-stream commit graph runs
the tiled commit at 2 warps (each element's fold order is the tiling's, so the values do not change).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_graphs = None  # the _PostVerifyGraphs whose commit joins late, once activated
_wait_counts: dict = {}
_logged: set = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.warning(msg, *args)


def late_join_refusal() -> Optional[str]:
    """Why the guarded late join cannot run in this process (None: it can). The guard covers the served shape."""
    from sglang.srt.runtime_context import (
        get_disagg,
        get_exec,
        get_memory,
        get_parallel,
        get_schedule,
    )

    mode = get_disagg().disaggregation_mode
    if mode != "null":
        return f"disaggregation_mode={mode} (retraction backups and transfers are unguarded)"
    if get_parallel().pp_size != 1 or get_parallel().tp_size != 1:
        return f"tp={get_parallel().tp_size} pp={get_parallel().pp_size} (the guard covers TP1, PP1)"
    if get_schedule().disable_overlap_schedule:
        return "the overlap scheduler is off"
    if get_memory().enable_unified_memory:
        return "the unified memory pool (its slot allocator is unguarded)"
    if get_exec().mamba.enable_int8_mamba_checkpoint:
        return "int8 mamba checkpoints (a finish-time store reads the active state)"
    if not get_exec().mamba.enable_mamba_extra_buffer:
        return "no mamba extra buffer (a finish-time settle writes the cursors)"
    return None


def activate(graphs) -> None:
    global _graphs
    _graphs = graphs
    logger.info(
        "Post-verify state commit joins late, before the next verify or prefill, with its guard"
    )


def active() -> bool:
    """Whether the late join was activated in this process."""
    return _graphs is not None


def refused(reason: str) -> None:
    _warn_once(
        "refused",
        "SGLANG_OPT_POST_VERIFY_LATE_JOIN refused: %s; the commit joins early",
        reason,
    )


def _pending_event():
    if _graphs is None or not _graphs._commit_pending:
        return None
    return _graphs._commit_done


def _pending_marker():
    """A fresh event on the side stream that completes with the pending commit, or None when none is pending."""
    if _pending_event() is None:
        return None
    import torch

    marker = torch.cuda.Event()
    marker.record(_graphs._side_stream)
    return marker


class FreeListGuard:
    """Which slots of a FIFO free list (taken from the front, freed to the back) a pending commit may still write.

    By host-side positions only: from ``dirty_from`` to the back lie all slots freed since the first free under a
    pending commit (a finished row's slot is freed before its commit lands); ``marker`` completes once every commit
    pending at those frees has. A slot handed out from that run waits on ``marker`` first.
    """

    def __init__(self):
        self.dirty_from = None
        self.marker = None

    def reset(self) -> None:
        self.dirty_from = self.marker = None

    def freed(self, *, free_len_before: int) -> None:
        marker = _pending_marker()
        if marker is None:
            return
        if self.dirty_from is None:
            self.dirty_from = free_len_before
        self.marker = marker

    def taken(self, count: int) -> Optional[int]:
        """Before ``count`` slots leave the front: the index among them of the first dirty one, or None."""
        if self.dirty_from is None:
            return None
        if self.marker.query():
            self.reset()
            return None
        first = self.dirty_from
        if count <= first:
            self.dirty_from = first - count
            return None
        self.dirty_from = 0
        return first

    def handing_out(self, *, first_dirty: Optional[int], index: int) -> None:
        """Before slot ``index`` of a take whose first dirty slot is ``first_dirty`` goes to a caller."""
        if first_dirty is None or index < first_dirty or self.marker is None:
            return
        import torch

        torch.cuda.current_stream().wait_event(self.marker)
        _note("slot_alloc")


def _note(site: str) -> None:
    n = _wait_counts.get(site, 0) + 1
    _wait_counts[site] = n
    if n & (n - 1) == 0:  # 1, 2, 4, 8, ...
        logger.debug(
            "post-verify late join: %s waited on the pending commit (%d times)", site, n
        )


def wait(site: str, stream=None) -> bool:
    """Order ``stream`` (default: the current stream) after the pending commit; True when one was pending."""
    event = _pending_event()
    if event is None:
        return False
    if stream is None:
        import torch

        stream = torch.cuda.current_stream()
    stream.wait_event(event)
    _note(site)
    return True


def sync(site: str) -> bool:
    """Wait on the host for the pending commit, which then needs no join; True when one was pending."""
    event = _pending_event()
    if event is None:
        return False
    event.synchronize()
    _graphs._commit_pending = False
    _note(site)
    return True
