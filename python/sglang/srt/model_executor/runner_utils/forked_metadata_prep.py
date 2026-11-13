"""The DSpark draft graph's in-graph metadata prep forked onto a side stream
(SGLANG_OPT_DSV41_DSPARK_DRAFT_PREP_FORK): embedding and the first layer's
pre-attention run meanwhile, and the first draft attention joins it before it
stores KV or reads the attention metadata."""

from __future__ import annotations

from typing import Callable, Optional

import torch

_pending_join: Optional[torch.cuda.Event] = None


def fork_in_graph_metadata_prep(
    *, stream: torch.cuda.Stream, done: torch.cuda.Event, prep: Callable[[], None]
) -> None:
    global _pending_join
    assert _pending_join is None, "previous forked metadata prep was never joined"
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prep()
        done.record()
    _pending_join = done


def join_in_graph_metadata_prep() -> bool:
    """Make the current stream wait for a pending forked prep; True if one was."""
    global _pending_join
    if _pending_join is None:
        return False
    torch.cuda.current_stream().wait_event(_pending_join)
    _pending_join = None
    return True
