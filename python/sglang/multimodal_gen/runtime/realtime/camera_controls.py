# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from sglang.multimodal_gen.runtime.realtime.condition_events import (
    ControlSignal,
    ControlStateSamplingQueue,
    ControlStateTransition,
)

CameraActionNormalizer = Callable[[list[Any]], list[str]]
CameraActionValidator = Callable[[Any], list[list[str]]]


def _identity_actions(actions: list[Any]) -> list[str]:
    return list(actions)


class RealtimeCameraControlState:
    """Session-local camera-control buffer shared by realtime model adapters.

    Camera controls arrive in two shapes:

    1. Script mode: ``list[list[str]]`` where each item is one output-frame's
       held actions. The script is consumed once from a FIFO and padded with
       neutral ``[]`` frames after it runs out.
    2. State mode: timestamped transitions such as "W is currently held".
       ``ControlStateSamplingQueue`` expands that continuous state into the next
       chunk and can pulse a short key press for a minimum number of frames.

    The two modes are intentionally exclusive. A new script clears state mode,
    and new state transitions clear script mode, so adapters never merge two
    camera timelines accidentally. ``sample_camera_actions`` returns ``None``
    only when no control should be sent; otherwise it returns exactly
    ``chunk_size`` frames, with ``[]`` meaning neutral/no-op for that frame.
    """

    def __init__(
        self,
        *,
        min_pulse_items: int = 1,
        script_maxlen: int = 512,
        max_transitions: int = 512,
        normalize_state_actions: CameraActionNormalizer = _identity_actions,
    ) -> None:
        self.camera_state = ControlStateSamplingQueue(
            default_item=[],
            min_pulse_items=min_pulse_items,
            max_transitions=max_transitions,
        )
        self.camera_script_queue: deque[ControlSignal] = deque(maxlen=script_maxlen)
        self.latest_sampled_event_id: int | None = None
        self._normalize_state_actions = normalize_state_actions

    def clear(self) -> None:
        """Reset all camera controls owned by this realtime session."""
        self.camera_state.clear()
        self.camera_script_queue.clear()
        self.latest_sampled_event_id = None

    def receive_camera_script(
        self,
        camera_actions: list[list[str]],
        *,
        event_id: int | None = None,
    ) -> None:
        """Replace active controls with a finite per-frame script."""
        self.camera_script_queue.clear()
        self.camera_state.clear()
        for actions in camera_actions:
            self.camera_script_queue.append(
                ControlSignal(
                    kind="camera_actions",
                    payload=list(actions),
                    seq_id=event_id,
                )
            )

    def receive_camera_state_transitions(
        self,
        transitions: list[ControlStateTransition],
    ) -> None:
        """Replace the script with continuous state transitions."""
        self.camera_script_queue.clear()
        self.camera_state.push_many(transitions)

    def receive_camera_actions(
        self,
        camera_actions: list[list[str]],
        *,
        event_id: int | None = None,
    ) -> None:
        self.receive_camera_script(camera_actions, event_id=event_id)

    def receive_camera_state(
        self,
        actions: list[str],
        *,
        event_id: int | None = None,
        timestamp_ms: int | None = None,
    ) -> None:
        self.receive_camera_state_transitions(
            [
                self._camera_state_transition(
                    actions,
                    event_id=event_id,
                    timestamp_ms=timestamp_ms,
                )
            ]
        )

    def receive_camera_event_payload(
        self,
        payload: Any,
        *,
        event_id: int | None,
        validate_camera_actions: CameraActionValidator,
    ) -> str:
        """Parse an external camera event and install it as script or state."""
        if isinstance(payload, dict) and payload.get("mode") == "state":
            transitions = self._camera_transitions_from_event_payload(
                payload,
                event_id=event_id,
            )
            self.receive_camera_state_transitions(transitions)
            return f"kind=camera_actions, mode=state, transitions={len(transitions)}"

        camera_actions = validate_camera_actions(payload)
        self.receive_camera_script(camera_actions, event_id=event_id)
        return f"kind=camera_actions, mode=script, frames={len(camera_actions)}"

    def sample_camera_actions(self, chunk_size: int) -> list[list[str]] | None:
        """Return the next chunk-sized camera action window.

        Script mode has priority because it represents an explicit finite
        timeline. State mode is sampled only when no script is pending.
        """
        if self.camera_script_queue:
            return self._sample_camera_script(chunk_size)
        action_list = self.camera_state.sample_chunk(chunk_size)
        if action_list is None:
            return None
        self.latest_sampled_event_id = self.camera_state.latest_sampled_seq_id()
        return [list(actions) for actions in action_list]

    def _sample_camera_script(self, chunk_size: int) -> list[list[str]]:
        chunk: list[list[str]] = []
        latest_event_id = self.latest_sampled_event_id
        while self.camera_script_queue and len(chunk) < chunk_size:
            signal = self.camera_script_queue.popleft()
            chunk.append(list(signal.payload))
            latest_event_id = signal.seq_id
        while len(chunk) < chunk_size:
            chunk.append([])
        self.latest_sampled_event_id = latest_event_id
        return chunk

    def _camera_state_transition(
        self,
        actions: list[Any],
        *,
        event_id: int | None,
        timestamp_ms: int | None,
    ) -> ControlStateTransition:
        return ControlStateTransition(
            payload=self._normalize_state_actions(actions),
            seq_id=event_id,
            timestamp_ms=timestamp_ms,
        )

    def _camera_transitions_from_event_payload(
        self,
        payload: dict[str, Any],
        *,
        event_id: int | None,
    ) -> list[ControlStateTransition]:
        transitions = payload.get("transitions")
        if not isinstance(transitions, list):
            raise ValueError("camera_actions state payload requires transitions")
        result = []
        for transition in transitions:
            if not isinstance(transition, dict):
                raise ValueError("camera_actions transition must be a map")
            actions = transition.get("actions")
            if not isinstance(actions, list):
                raise ValueError("camera_actions transition actions must be a list")
            timestamp_ms = transition.get("client_ts_ms")
            if timestamp_ms is not None:
                timestamp_ms = int(timestamp_ms)
            result.append(
                self._camera_state_transition(
                    actions,
                    event_id=event_id,
                    timestamp_ms=timestamp_ms,
                )
            )
        return result
