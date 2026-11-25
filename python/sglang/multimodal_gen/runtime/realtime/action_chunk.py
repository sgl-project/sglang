# SPDX-License-Identifier: Apache-2.0
"""Timing and handoff of controller-space action chunks."""

from __future__ import annotations

import time

import numpy as np


class ActionChunkConflict(ValueError):
    pass


class ActionDeadlineExceeded(TimeoutError):
    pass


def remaining_ms(deadline_ms: int) -> int:
    remaining = deadline_ms - time.time_ns() // 1_000_000
    if remaining <= 0:
        raise ActionDeadlineExceeded("Action deadline elapsed")
    return remaining


def parse_chunk_request(value, config):
    if not isinstance(value, dict):
        raise ValueError("realtime must be an object")
    deadline = value.get("deadline_unix_ms")
    if type(deadline) is not int or not 0 < deadline < 2**63:
        raise ValueError("realtime.deadline_unix_ms must be a positive integer")
    remaining_ms(deadline)
    session = value.get("session_id")
    if not isinstance(session, str) or not session or len(session) > 256:
        raise ValueError("realtime.session_id must contain 1 to 256 characters")
    step = value.get("start_step")
    if type(step) is not int or step < 0:
        raise ValueError("realtime.start_step must be a nonnegative control step")
    prefix = value.get("action_prefix")
    if prefix is not None and not isinstance(prefix, (list, tuple, np.ndarray)):
        raise ValueError("action_prefix must be an array of controller commands")
    if prefix is not None and np.asarray(prefix).ndim == 0:
        raise ValueError("action_prefix must be a matrix")
    if prefix is not None and len(prefix):
        if not getattr(config, "supports_action_prefix", False):
            raise ValueError("This policy does not support action-prefix conditioning")
        prefix = np.asarray(prefix, dtype=np.float32)
        if (
            prefix.ndim != 2
            or prefix.shape[1] != config.action_dim
            or len(prefix) > config.action_horizon
            or not np.isfinite(prefix).all()
        ):
            raise ValueError(
                "action_prefix must be finite controller-space rows [steps, action_dim] within the policy horizon"
            )
    else:
        prefix = None
    request_id = value.get("request_id")
    if request_id is not None and (
        not isinstance(request_id, str) or not 1 <= len(request_id) <= 256
    ):
        raise ValueError("Invalid realtime.request_id")
    return {
        "request_id": request_id,
        "deadline_unix_ms": deadline,
        "session_id": session,
        "start_step": step,
        "action_prefix": prefix,
    }
