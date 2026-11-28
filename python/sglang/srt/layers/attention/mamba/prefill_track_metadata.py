"""Host-side row selection for prefill state snapshots.

Slot IDs deliberately stay on the device: unified pools translate virtual IDs
before these row indices gather physical source and destination slots.
"""

from dataclasses import dataclass
from itertools import accumulate


@dataclass
class PrefillTrackPlan:
    tracked_rows: list[int]
    final_rows: list[int]
    unaligned_rows: list[int]
    h_rows: list[int]
    h_src: list[int]
    recompute_rows: list[int]
    recompute_end_locs: list[int]
    chunk_indices: list[int]


def build_prefill_track_plan(
    mask: list[bool],
    track_lens: list[int],
    extend_lens: list[int],
    prefix_lens: list[int],
    chunk_size: int,
    *,
    mamba2: bool,
) -> PrefillTrackPlan:
    """Use the backend's actual chunk size, including Mamba2's flat grid."""
    assert len(mask) == len(track_lens) == len(extend_lens) == len(prefix_lens)
    starts = list(accumulate(extend_lens, initial=0))
    h_offsets = list(
        accumulate(((n + chunk_size - 1) // chunk_size for n in extend_lens), initial=0)
    )
    plan = PrefillTrackPlan([], [], [], [], [], [], [], [-1] * len(mask))
    for row, track in enumerate(mask):
        if not track:
            continue
        plan.tracked_rows.append(row)
        length = track_lens[row] - prefix_lens[row]
        if length % chunk_size == 0:
            plan.final_rows.append(row)
            continue
        chunk = length // chunk_size
        plan.unaligned_rows.append(row)
        plan.chunk_indices[row] = chunk
        end = starts[row] + chunk * chunk_size
        if mamba2 and end % chunk_size:
            plan.recompute_rows.append(row)
            plan.recompute_end_locs.append(end)
        else:
            plan.h_rows.append(row)
            plan.h_src.append(end // chunk_size if mamba2 else h_offsets[row] + chunk)
    return plan
