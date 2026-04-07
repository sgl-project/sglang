import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass
class SpectreDraftState:
    req_id: str
    spec_cnt: int
    req_object: Optional["Req"]
    location: str = "waiting_queue"
    target_origin_input_ids: Optional[List[int]] = None
    last_prefix_length: int = 0
    last_output_length: int = 0
    last_updated_time: float = field(default_factory=time.time)
    created_time: float = field(default_factory=time.time)
    timeout_threshold: float = 30.0  # seconds


class SpectreDraftStateManager:
    def __init__(self, timeout_threshold: float = 60.0):
        self.active_draft_states: Dict[str, SpectreDraftState] = {}
        self._lock = threading.Lock()
        self.time_out_cycle = 200
        self.timeout_threshold = timeout_threshold

    def get_state(self, req_id: str) -> Optional[SpectreDraftState]:
        with self._lock:
            return self.active_draft_states.get(req_id)

    def set_state(self, req_id: str, state: SpectreDraftState):
        with self._lock:
            self.active_draft_states[req_id] = state

    def delete_state(self, req_id: str):
        with self._lock:
            if req_id in self.active_draft_states:
                del self.active_draft_states[req_id]

    def exists(self, req_id: str) -> bool:
        with self._lock:
            return req_id in self.active_draft_states

    def get(self, req_id: str) -> Optional[SpectreDraftState]:
        return self.get_state(req_id)

    def set(self, req_id: str, state: SpectreDraftState) -> None:
        self.set_state(req_id, state)

    def delete(self, req_id: str) -> bool:
        with self._lock:
            if req_id in self.active_draft_states:
                del self.active_draft_states[req_id]
                return True
            return False

    def update_location(self, req_id: str, location: str) -> bool:
        with self._lock:
            if req_id in self.active_draft_states:
                self.active_draft_states[req_id].location = location
                self.active_draft_states[req_id].last_updated_time = time.time()
                return True
            return False

    def update_spec_cnt(self, req_id: str, spec_cnt: int) -> bool:
        with self._lock:
            if req_id in self.active_draft_states:
                self.active_draft_states[req_id].spec_cnt = spec_cnt
                self.active_draft_states[req_id].last_updated_time = time.time()
                return True
            return False

    def touch(self, req_id: str) -> bool:
        with self._lock:
            if req_id in self.active_draft_states:
                self.active_draft_states[req_id].last_updated_time = time.time()
                return True
            return False

    def create_state(
        self,
        req_id: str,
        spec_cnt: int,
        req_object: "Req",
        location: str = "waiting_queue",
        prefix_length: int = 0,
        output_length: int = 0,
    ) -> SpectreDraftState:
        now = time.time()
        state = SpectreDraftState(
            req_id=req_id,
            spec_cnt=spec_cnt,
            req_object=req_object,
            location=location,
            last_prefix_length=prefix_length,
            last_output_length=output_length,
            last_updated_time=now,
            created_time=now,
            timeout_threshold=self.timeout_threshold,
        )

        with self._lock:
            self.active_draft_states[req_id] = state

        return state

    def cleanup_stale_states(self, timeout: Optional[float] = None) -> List[str]:
        if timeout is None:
            timeout = self.timeout_threshold

        current_time = time.time()
        to_remove = []

        with self._lock:
            for req_id, state in list(self.active_draft_states.items()):
                idle_time = current_time - state.last_updated_time
                if idle_time > timeout:
                    to_remove.append(req_id)

        for req_id in to_remove:
            state = self.get_state(req_id)

        return to_remove

    def get_all_rids(self) -> List[str]:
        with self._lock:
            return list(self.active_draft_states.keys())

    def size(self) -> int:
        with self._lock:
            return len(self.active_draft_states)

    def clear(self) -> int:
        with self._lock:
            count = len(self.active_draft_states)
            self.active_draft_states.clear()
            return count
