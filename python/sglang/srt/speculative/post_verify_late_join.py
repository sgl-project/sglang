"""Late join of the post-verify side-stream state commit, with a guard on every other reader and writer.

With SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH the Mamba state commit replays on a side stream, and EAGLE v2 joins
it right after the draft extend (or, under SGLANG_PLB_COMMIT_LATE_JOIN, before the next verify or prefill without a
guard). SGLANG_ENABLE_POST_VERIFY_LATE_JOIN=1 joins it before the next verify or prefill, so it overlaps the draft
extend, scheduling and the next draft, and orders every reader or writer of state the pending commit may touch
outside that window after it: state-slot hand-out and row binding, HiCache loads and backups, retraction, a loop pass
that launches nothing, control requests and cache flushes. The values each kernel reads and writes are those of the
early join; only their ordering against the commit moves.

The guard covers aggregated serving at TP1 and PP1 with the overlap scheduler and the Mamba extra buffer, without the
unified memory pool or int8 Mamba checkpoints; elsewhere a warning, and the early join stays.

Once the late join is active (not refused), also:
* the post-verify graph pool takes, at its first capture, the sampling scratch of the largest batch the verify graphs
  allow and frees it at once, after returning the allocator's unused cached blocks to the device; every later graph
  key splits that segment instead of adding segments the private pool never returns;
* the side-stream commit graph runs the tiled commit at 2 warps (each element's fold order is the tiling's, so the
  values do not change).
"""

import contextlib
import logging
from typing import Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

LATE_JOIN = envs.SGLANG_ENABLE_POST_VERIFY_LATE_JOIN.get()
# The side-stream commit graph's tiled-commit warps (the forward's stays at the kernel default).
SIDE_COMMIT_WARPS = 2
# Arbitrary headroom for the post-verify body's large temporaries besides its two sampling buffers.
_PRESIZE_MARGIN = 64 << 20

_graphs = None  # the _PostVerifyGraphs whose commit joins late, once activated
_wait_counts: dict = {}
_logged: set = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.warning(msg, *args)


def max_graph_keys() -> int:
    """Post-verify graph sets (keys) kept before a new key runs eager."""
    value = envs.SGLANG_POST_VERIFY_MAX_GRAPH_KEYS.get()
    if value < 1:
        raise ValueError(
            f"SGLANG_POST_VERIFY_MAX_GRAPH_KEYS must be a positive integer, got {value}"
        )
    return value


# ------------------------------------------------------------------------------------------------ graph pool
def presize_bytes(
    *, max_requests: int, tokens_per_request: int, vocab: int, free_bytes: int
) -> int:
    """The post-verify body's peak scratch at the largest key: two fp32 [requests * tokens, vocab] sampling buffers
    live at once, plus a margin. 0 when the device has less than that plus 1 GiB free: the pool then grows per key."""
    nbytes = 2 * max_requests * tokens_per_request * vocab * 4 + _PRESIZE_MARGIN
    if free_bytes < nbytes + (1 << 30):
        _warn_once(
            "presize",
            "post-verify graph pool not presized: %.2f GiB free for %.2f GiB",
            free_bytes / 2**30,
            nbytes / 2**30,
        )
        return 0
    logger.debug(
        "post-verify graph pool presized to %.2f GiB (%d requests x %d tokens, vocab %d)",
        nbytes / 2**30,
        max_requests,
        tokens_per_request,
        vocab,
    )
    return nbytes


def release_cached_blocks() -> None:
    """Return the caching allocator's unused cached blocks to the device, so the presize's free-memory reading counts
    them. Call outside any capture: empty_cache frees no general-pool block while a capture is underway."""
    import torch

    torch.cuda.empty_cache()


def presize_pool(nbytes: int, *, device) -> None:
    """Inside the pool's first capture, before its body: one segment of ``nbytes``, freed at once to the pool."""
    import torch

    scratch = torch.empty(nbytes, dtype=torch.uint8, device=device)
    del scratch


def side_commit_tiling():
    """Context for running or capturing the side-stream commit: the tiled commit at SIDE_COMMIT_WARPS once the late
    join is active (activation precedes the first warmup and capture)."""
    if _graphs is None:
        return contextlib.nullcontext()
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
        tiled_commit_warps,
    )

    return tiled_commit_warps(SIDE_COMMIT_WARPS)


# ------------------------------------------------------------------------------------------------ late join
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


def refused(reason: str) -> None:
    _warn_once(
        "refused",
        "SGLANG_ENABLE_POST_VERIFY_LATE_JOIN refused: %s; the commit joins early",
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
