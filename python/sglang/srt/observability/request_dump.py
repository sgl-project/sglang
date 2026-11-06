"""The request dump file: requests the Python ``TokenizerManager`` served,
pickled with the launch state. ``--crash-dump-folder`` writes one before a
crash, and the ``dump_requests_folder`` logging setting writes one every
``dump_requests_threshold`` requests; the dump readers such as
``scripts/playground/replay_request_dump.py`` load either."""

from __future__ import annotations

import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Any, List, Optional, Tuple

from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.watermarking.core import (
    redact_watermark_command_line,
    redact_watermark_secrets,
)

logger = logging.getLogger(__name__)


def build_request_dump(
    requests: List[Tuple],
    server_args: Any,
    resolved_config: Optional[dict],
    launch_command: Optional[str] = None,
) -> dict:
    """``requests`` are ``(request object, output dict, created time, finished
    time)`` tuples."""
    data = {
        "server_args": server_args,
        "config_updates": redact_watermark_secrets(get_context().overrides_log()),
        "resolved_config": resolved_config,
        "requests": redact_watermark_secrets(requests),
    }
    if launch_command is not None:
        data["launch_command"] = launch_command
    return data


def save_request_dump(filename: str, data: dict) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        try:
            pickle.dump(data, f)
        except Exception as e:
            # When the server is launched with --trust-remote-code,
            # server_args sometimes fails to pickle. Retry without
            # server_args so the request data still gets persisted.
            logger.error(
                f"Failed to pickle dump with server_args: {e!r}; "
                "retrying without server_args"
            )
            f.seek(0)
            f.truncate()
            data["server_args"] = None
            # The snapshot copies the same object field by field.
            data["resolved_config"] = None
            pickle.dump(data, f)


def write_crash_dump(
    folder: str,
    hostname: str,
    requests: List[Tuple],
    server_args: Any,
    resolved_config: Optional[dict],
    watermark_config: Any,
) -> str:
    """Write the crash dump to ``<folder>/<hostname>/crash_dump_<time>.pkl``
    and return the path."""
    filename = os.path.join(
        folder,
        hostname,
        f"crash_dump_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl",
    )
    data = build_request_dump(
        requests,
        server_args,
        resolved_config,
        launch_command=redact_watermark_command_line(sys.argv, watermark_config),
    )
    save_request_dump(filename, data)
    return filename
