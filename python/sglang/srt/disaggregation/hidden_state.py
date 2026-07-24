from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class PDHiddenReqState:
    def __init__(self):
        self.meta: Optional[dict] = None
        self.src_indices: Optional[List[int]] = None
        self.dst_indices: Optional[List[int]] = None
        self.written: Optional[List[bool]] = None
        self.capture_layer_ids: Optional[List[int]] = None
        self.current_src_indices: Optional[List[int]] = None
        self.current_start: Optional[int] = None
        self.current_row_len: int = 0
        self.current_is_last: bool = False
        self.owner_direct_sent: bool = False


_pd_hidden_req_states = weakref.WeakKeyDictionary()


def get_pd_hidden_req_state(req: "Req") -> PDHiddenReqState:
    state = _pd_hidden_req_states.get(req)
    if state is None:
        state = PDHiddenReqState()
        _pd_hidden_req_states[req] = state
    return state


def get_pd_hidden_capture_layer_ids(reqs: List["Req"]) -> Optional[List[int]]:
    """Return the per-batch PD hidden capture layers requested by any req."""
    for req in reqs:
        layer_ids = get_pd_hidden_req_state(req).capture_layer_ids
        if layer_ids:
            return [int(x) for x in layer_ids]
    return None
