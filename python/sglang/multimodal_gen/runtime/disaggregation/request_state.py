# SPDX-License-Identifier: Apache-2.0
"""Request state machine for disaggregated diffusion pipelines."""

import enum
import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class RequestState(enum.Enum):
    """Lifecycle states for a disagg pipeline request.

    *_WAITING: request queued, awaiting a free buffer slot.
    *_RUNNING: request dispatched to a specific instance.
    """

    PENDING = "pending"
    ENCODER_WAITING = "encoder_waiting"
    ENCODER_RUNNING = "encoder_running"
    ENCODER_DONE = "encoder_done"
    DENOISING_WAITING = "denoising_waiting"
    DENOISING_RUNNING = "denoising_running"
    DENOISING_DONE = "denoising_done"
    DECODER_WAITING = "decoder_waiting"
    DECODER_RUNNING = "decoder_running"
    DONE = "done"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


_TERMINAL_STATES = {RequestState.DONE, RequestState.FAILED, RequestState.TIMED_OUT}
_ACTIVE_STATES = set(RequestState) - _TERMINAL_STATES

# Normal (non-failure) transitions.  FAILED and TIMED_OUT are handled
# separately in transition() — any active state can reach them.
_VALID_TRANSITIONS: dict[RequestState, set[RequestState]] = {
    RequestState.PENDING: {RequestState.ENCODER_WAITING, RequestState.ENCODER_RUNNING},
    RequestState.ENCODER_WAITING: {RequestState.ENCODER_RUNNING},
    RequestState.ENCODER_RUNNING: {RequestState.ENCODER_DONE},
    RequestState.ENCODER_DONE: {
        RequestState.DENOISING_WAITING,
        RequestState.DENOISING_RUNNING,
    },
    RequestState.DENOISING_WAITING: {RequestState.DENOISING_RUNNING},
    RequestState.DENOISING_RUNNING: {RequestState.DENOISING_DONE},
    RequestState.DENOISING_DONE: {
        RequestState.DECODER_WAITING,
        RequestState.DECODER_RUNNING,
    },
    RequestState.DECODER_WAITING: {RequestState.DECODER_RUNNING},
    RequestState.DECODER_RUNNING: {RequestState.DONE},
}


@dataclass
class RequestRecord:
    request_id: str
    state: RequestState = RequestState.PENDING
    submit_time: float = field(default_factory=time.monotonic)
    last_transition_time: float = field(default_factory=time.monotonic)
    encoder_instance: int | None = None
    denoiser_instance: int | None = None
    decoder_instance: int | None = None
    error: str | None = None

    def elapsed_s(self) -> float:
        return time.monotonic() - self.submit_time

    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES


class RequestTracker:
    """Thread-safe tracker for request state machines."""

    def __init__(self):
        self._lock = threading.Lock()
        self._requests: dict[str, RequestRecord] = {}

    def submit(self, request_id: str) -> RequestRecord:
        with self._lock:
            if request_id in self._requests:
                raise ValueError(f"Duplicate request_id: {request_id}")
            record = RequestRecord(request_id=request_id)
            self._requests[request_id] = record
            return record

    def transition(
        self,
        request_id: str,
        new_state: RequestState,
        *,
        error: str | None = None,
        encoder_instance: int | None = None,
        denoiser_instance: int | None = None,
        decoder_instance: int | None = None,
    ) -> RequestRecord:
        with self._lock:
            record = self._requests.get(request_id)
            if record is None:
                raise ValueError(f"Unknown request_id: {request_id}")

            old_state = record.state

            if new_state in _TERMINAL_STATES and new_state != RequestState.DONE:
                # FAILED / TIMED_OUT: allowed from any active state
                if old_state not in _ACTIVE_STATES:
                    raise ValueError(
                        f"Cannot transition {request_id} from terminal state "
                        f"{old_state.value} to {new_state.value}"
                    )
            elif new_state not in _VALID_TRANSITIONS.get(old_state, set()):
                raise ValueError(
                    f"Invalid transition for {request_id}: "
                    f"{old_state.value} -> {new_state.value}"
                )

            record.state = new_state
            record.last_transition_time = time.monotonic()
            if error is not None:
                record.error = error
            if encoder_instance is not None:
                record.encoder_instance = encoder_instance
            if denoiser_instance is not None:
                record.denoiser_instance = denoiser_instance
            if decoder_instance is not None:
                record.decoder_instance = decoder_instance

            logger.debug(
                "Request %s: %s -> %s", request_id, old_state.value, new_state.value
            )
            return record

    def get(self, request_id: str) -> RequestRecord | None:
        with self._lock:
            return self._requests.get(request_id)

    def remove(self, request_id: str) -> RequestRecord | None:
        with self._lock:
            return self._requests.pop(request_id, None)

    def find_timed_out(self, timeout_s: float) -> list[str]:
        now = time.monotonic()
        with self._lock:
            return [
                r.request_id
                for r in self._requests.values()
                if r.state in _ACTIVE_STATES and (now - r.submit_time) > timeout_s
            ]

    def snapshot(self) -> dict:
        with self._lock:
            state_counts = {}
            for r in self._requests.values():
                state_counts[r.state.value] = state_counts.get(r.state.value, 0) + 1
            return {
                "total": len(self._requests),
                "active": sum(
                    1 for r in self._requests.values() if not r.is_terminal()
                ),
                "by_state": state_counts,
            }
