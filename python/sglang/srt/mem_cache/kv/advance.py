"""Effects of a request advancing past a state checkpoint boundary."""

from typing import Optional, Tuple, Union

import msgspec


class TrackingAdvance(msgspec.Struct, frozen=True, kw_only=True):
    """The completed forward's checkpoint and the next forward's destination.

    An omitted boundary preserves the latest checkpoint; None invalidates it
    before an in-place write. Boundaries come from the completed batch snapshot.
    """

    boundary: Union[int, None, msgspec.UnsetType] = msgspec.UNSET
    last_position: Optional[int] = None
    next_position: Optional[int] = None
    preserve_previous: bool = False
    release_positions: Tuple[int, ...] = ()


def advance_tracking(storage, req, update: TrackingAdvance) -> None:
    kv = req.kv
    for position in update.release_positions:
        if not kv.mamba_ping_pong_empty[position]:
            storage.release_tracking_slot(req, position)
    if update.preserve_previous:
        kv.mamba_prev_track_seqlen = kv.mamba_last_track_seqlen
    if update.boundary is not msgspec.UNSET:
        kv.mamba_last_track_seqlen = update.boundary
    if update.last_position is not None:
        kv.mamba_last_track_idx = update.last_position
    if update.next_position is not None:
        kv.mamba_next_track_idx = update.next_position
