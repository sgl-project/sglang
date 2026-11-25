# SPDX-License-Identifier: Apache-2.0
"""Controller-side action buffering. All state belongs to one control loop."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ActionChunkBuffer:
    """Single control-loop owner. Inference may run while pop() consumes the old plan."""

    session_id: str
    step: int = 0
    _actions: np.ndarray | None = field(default=None, init=False, repr=False)
    _request: dict | None = field(default=None, init=False, repr=False)
    _prefix_len: int = field(default=0, init=False, repr=False)

    def request(self, deadline_unix_ms: int, prefix_steps: int) -> dict:
        if deadline_unix_ms <= time.time_ns() // 1_000_000:
            raise TimeoutError("Action deadline elapsed")
        if type(prefix_steps) is not int or prefix_steps < 0:
            raise ValueError("prefix_steps must be a nonnegative integer")
        self._request = dict(
            request_id=uuid.uuid4().hex,
            session_id=self.session_id,
            start_step=self.step,
            deadline_unix_ms=deadline_unix_ms,
        )
        prefix = [] if self._actions is None else self._actions[:prefix_steps].tolist()
        self._prefix_len = len(prefix)
        return {**self._request, "action_prefix": prefix}

    def accept(self, response: dict) -> bool:
        tag = response.get("realtime")
        if self._request is None or tag != self._request:
            return False
        self._request = None
        if time.time_ns() // 1_000_000 >= tag["deadline_unix_ms"]:
            return False
        try:
            rows = np.asarray(response["data"][0]["action"]["values"], dtype=np.float32)
        except (KeyError, IndexError, ValueError, TypeError):
            return False
        offset = self.step - tag["start_step"]
        if (
            offset > self._prefix_len
            or rows.ndim != 2
            or rows.shape[1] == 0
            or len(rows) < self._prefix_len
            or (self._actions is not None and rows.shape[1] != self._actions.shape[1])
            or not 0 <= offset < len(rows)
            or not np.isfinite(rows).all()
        ):
            return False
        replacement = rows[offset:].copy()
        remaining_prefix = self._prefix_len - offset
        if remaining_prefix:
            replacement[:remaining_prefix] = self._actions[:remaining_prefix]
        self._actions = replacement
        return True

    def pop(self) -> np.ndarray | None:
        if self._actions is None or len(self._actions) == 0:
            return None
        action, self._actions = self._actions[0], self._actions[1:]
        self.step += 1
        return action
