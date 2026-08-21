# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sglang.multimodal_gen.runtime.realtime.control_signals import (
    ControlScriptQueue,
    ControlStateQueue,
    ControlStateTransition,
    parse_control_event_payload,
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
       ``ControlStateQueue`` expands that continuous state into the next
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
        # stores state-mode control signals
        self.camera_state_queue = ControlStateQueue(
            default_item=[],
            min_pulse_items=min_pulse_items,
            max_transitions=max_transitions,
        )
        # stores script-mode control signals
        # script-mode signals take precedence over state-mode signals, see sample_camera_actions
        self.camera_script_queue = ControlScriptQueue(
            "camera_actions",
            max_events=script_maxlen,
            default_item=[],
        )
        self.latest_sampled_event_id: int | None = None
        self._normalize_state_actions = normalize_state_actions

    def clear(self) -> None:
        """Reset all camera controls owned by this realtime session."""
        self.camera_state_queue.clear()
        self.camera_script_queue.clear()
        self.latest_sampled_event_id = None

    def receive_camera_action_script(
        self,
        camera_actions: list[list[str]],
        *,
        event_id: int | None = None,
    ) -> None:
        """Replace active controls with a finite per-frame script."""
        self.camera_state_queue.clear()
        self.camera_script_queue.push_script(
            [list(actions) for actions in camera_actions],
            event_id=event_id,
        )

    def receive_camera_state_transitions(
        self,
        transitions: list[ControlStateTransition],
    ) -> None:
        """Replace the script with continuous state transitions."""
        self.camera_script_queue.clear()
        self.camera_state_queue.push_many(transitions)

    def receive_camera_state(
        self,
        actions: list[str],
        *,
        event_id: int | None = None,
        timestamp_ms: int | None = None,
    ) -> None:
        self.receive_camera_state_transitions(
            [
                ControlStateTransition(
                    payload=self._normalize_state_actions(actions),
                    seq_id=event_id,
                    timestamp_ms=timestamp_ms,
                )
            ]
        )

    def receive_camera_control_event_payload(
        self,
        payload: Any,
        *,
        event_id: int | None,
        validate_camera_actions: CameraActionValidator,
    ) -> str:
        """Parse an external camera event (from endpoint) and install it as script or state."""
        parsed = parse_control_event_payload(
            payload,
            event_id=event_id,
            kind="camera_actions",
            normalize_state_payload=self._normalize_state_actions,
            validate_script_payload=validate_camera_actions,
        )
        if parsed.mode == "state":
            transitions = parsed.payload
            self.receive_camera_state_transitions(transitions)
            return f"kind=camera_actions, mode=state, transitions={len(transitions)}"

        camera_actions = parsed.payload
        self.receive_camera_action_script(camera_actions, event_id=event_id)
        return f"kind=camera_actions, mode=script, frames={len(camera_actions)}"

    def sample_camera_actions(self, chunk_size: int) -> list[list[str]] | None:
        """Core method, return the next chunk-sized camera action window.

        Script mode has priority because it represents an explicit finite
        timeline. State mode is sampled only when no script is pending.
        """
        # Script mode wins: it is an explicit finite timeline and should not be
        # merged with held-key state from the live control path.
        if self.camera_script_queue.has_script():
            return self._sample_camera_script(chunk_size)
        # State mode is the WebUI path: held controls persist across chunks until
        # a new transition changes the current state.
        action_list = self.camera_state_queue.sample_chunk(chunk_size)
        if action_list is None:
            return None
        self.latest_sampled_event_id = self.camera_state_queue.latest_sampled_seq_id()
        return [list(actions) for actions in action_list]

    def _sample_camera_script(self, chunk_size: int) -> list[list[str]]:
        chunk = self.camera_script_queue.sample_script(chunk_size)
        chunk = [list(actions) for actions in chunk]
        self.latest_sampled_event_id = self.camera_script_queue.last_sampled_seq_id()
        return chunk
