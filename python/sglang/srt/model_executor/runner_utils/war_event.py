"""WAR read-done event utilities for CUDA graph runners."""

from enum import Enum, auto
from typing import Optional

import torch

from sglang.srt.utils import is_cuda


# Whether and where the WAR read-done record lands for a replay.
class WarReadDonePolicy(Enum):
    # This forward mode or algorithm does not publish a read-done event.
    NONE = auto()
    # Snapshot backends finish all shared reads before launch.
    PRE_REPLAY = auto()
    # Captured metadata initialization finishes all shared reads.
    IN_GRAPH = auto()
    # Captured-metadata replays keep reading shared buffers throughout the graph.
    POST_REPLAY = auto()


def make_war_read_done_event(device_module) -> Optional[torch.cuda.Event]:
    """Create a persistent external event for CUDA graph capture."""
    if not is_cuda():
        return None
    try:
        return device_module.Event(external=True)
    except TypeError:
        return None
