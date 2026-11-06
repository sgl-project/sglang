# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Control-file protocol between the snapshot controller and the captured engine.

The controller starts the engine as a child process and then hands it to CRIU,
so after a restore neither side is the other's parent and no shared memory
survives. Both sides coordinate only through files in ``<artifact>/control``:

- the controller owns ``release.json`` (resume, plus the listen address to use)
  and ``abort``;
- the engine owns ``scheduler.json`` (released state plus the canary it
  sampled), ``ready.json``, ``resumed.json`` (what the restored engine observed)
  and ``error.json``.

Every wait carries a budget and every engine failure is reported through
``error.json``, so a controller that dies cannot leave the engine parked
forever, and a controller waiting on a broken engine reads the reason instead
of a timeout.
"""

import json
import time

import msgspec

from sglang.srt.engine_snapshot.errors import SnapshotRuntimeFailure
from sglang.srt.engine_snapshot.manifest import SnapshotCanary, write_json_atomic

CONTROL_DIRNAME = "control"

# Controller -> engine.
RELEASE = "release.json"
ABORT = "abort"

# Engine -> controller.
SCHEDULER = "scheduler.json"
READY = "ready.json"
RESUMED = "resumed.json"
ERROR = "error.json"

# How long an engine may stay parked waiting for the controller. The budget is
# spent, not measured against a clock, so a long create-to-restore gap does not
# consume it.
DEFAULT_PARK_SECONDS = 1800.0

_POLL_SECONDS = 0.05


class ReleaseInfo(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """Permission to resume; ``host``/``port`` override the captured address."""

    host: str | None = None
    port: int | None = None


class SchedulerInfo(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """Written by the scheduler once weights, KV memory and the reload are proven."""

    gpu_uuid: str
    canary: SnapshotCanary


class ResumedInfo(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """What the restored scheduler observed when it re-ran the canary."""

    token_id: int
    logprob: float


class EngineInfo(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """Written by the HTTP process once both barriers are engaged."""

    gpu_uuid: str
    model_path: str
    host: str
    port: int


class ErrorInfo(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """Engine-side failure, surfaced to the controller's wait loop."""

    error: str


def write_release(control_dir, host=None, port=None):
    """Let the engine resume, optionally on a different listen address."""
    write_json_atomic(
        control_dir / RELEASE,
        msgspec.to_builtins(ReleaseInfo(host=host, port=port)),
        overwrite=True,
    )


def write_abort(control_dir):
    """Ask a parked engine to stop waiting instead of being killed."""
    (control_dir / ABORT).touch()


def write_error(control_dir, error):
    """Publish an engine-side failure for the controller to read."""
    message = str(error) or type(error).__name__
    write_json_atomic(
        control_dir / ERROR, msgspec.to_builtins(ErrorInfo(message)), overwrite=True
    )


def read_error(control_dir):
    """Return the engine's failure message, or ``None`` when it has not failed."""
    path = control_dir / ERROR
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return f"engine wrote an unreadable {ERROR}"
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"]
    return f"engine wrote a malformed {ERROR}"


def read_json(control_dir, name, type_):
    """Read a control file, reporting missing and malformed files distinctly."""
    path = control_dir / name
    try:
        payload = path.read_bytes()
    except FileNotFoundError as error:
        raise SnapshotRuntimeFailure(f"engine did not write {name}") from error
    try:
        return msgspec.json.decode(payload, type=type_)
    except msgspec.ValidationError as error:
        raise SnapshotRuntimeFailure(f"invalid {name}: {error}") from error


def clear_handshake(control_dir):
    """Drop a previous handshake so a fresh restore cannot read stale state."""
    for name in (RELEASE, ABORT, RESUMED, ERROR):
        (control_dir / name).unlink(missing_ok=True)


def wait_for(control_dir, name, timeout_seconds=DEFAULT_PARK_SECONDS):
    """Wait until ``name`` exists, the controller aborts, or the budget runs out.

    An engine failure recorded in ``error.json`` ends the wait as well, with
    the engine's message instead of a timeout.
    """
    remaining = float(timeout_seconds)
    while not (control_dir / name).is_file():
        if (control_dir / ABORT).exists():
            raise SnapshotRuntimeFailure("controller aborted the snapshot handshake")
        if message := read_error(control_dir):
            raise SnapshotRuntimeFailure(message)
        if remaining <= 0:
            raise SnapshotRuntimeFailure(
                f"controller did not write {name} within {timeout_seconds:g}s"
            )
        delay = min(_POLL_SECONDS, remaining)
        time.sleep(delay)
        remaining -= delay


def wait_and_read(control_dir, name, type_, timeout_seconds=DEFAULT_PARK_SECONDS):
    """Wait for a control file, then decode it."""
    wait_for(control_dir, name, timeout_seconds)
    return read_json(control_dir, name, type_)
