# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Allocator-history forensics for CUDA-graph buffer lifetime debugging.

When ``SGLANG_MEM_FORENSICS_DIR`` names a directory, the model runner starts
CUDA caching-allocator history recording (with Python allocation stacks)
during initialization, and each scheduler writes one
``torch.cuda.memory._snapshot`` pickle tagged ``ready`` when it reaches its
event loop, after every CUDA graph capture has completed. Failure handlers
may call :func:`maybe_dump_memory_forensics` with their own tag for a second
snapshot. Mapping a corrupted tensor's ``data_ptr`` onto the ``ready``
snapshot names the allocation site of a block a captured graph recorded.

Recording is started once per process and left active. The allocator keeps
one process-wide history recorder, and stopping it discards the recorded
events (live blocks keep the stacks they were allocated with), while
changing ``max_entries`` on a recorder that already holds events leaves its
ring buffer misordered. The CUDA graph runner's capture profiler
(``--enable-profile-cuda-graph``) therefore records through
:func:`start_memory_history` and :func:`stop_memory_history`, which keep the
forensics configuration and history untouched while forensics owns the
recorder (its capture snapshot then uses that configuration). Another feature that stops recording itself (the ``MEM`` activity
of the torch profiler) still clears the events: a dump that finds no history
entries writes the snapshot anyway, since its live blocks still map
addresses to allocation stacks, re-arms recording with the configured
parameters, and leaves the tag unconsumed, so the next request for that tag
writes a snapshot whose history starts at the re-arm.
:func:`maybe_start_memory_forensics` and :func:`maybe_dump_memory_forensics`
return without touching ``torch.cuda`` unless the directory variable is set,
and dump failures are logged and never raised, so the original failure path
of a caller is preserved even on a faulted CUDA context. The profiler hooks
call ``torch.cuda`` as the profiler itself did and propagate its errors.
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
import time

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_started = False
_max_entries = 0
_dumped_tags: set[str] = set()


def memory_forensics_enabled() -> bool:
    return bool(envs.SGLANG_MEM_FORENSICS_DIR.get())


def _rank_label() -> str:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return str(torch.distributed.get_rank())
    except Exception:
        pass
    return "na"


def _start_recording() -> None:
    # Always the max_entries recording started with: changing it on a
    # recorder that holds events leaves the ring buffer misordered.
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=_max_entries,
    )


def _owns_recorder() -> bool:
    return _started and memory_forensics_enabled()


def _snapshot_has_history(snapshot) -> bool:
    return any(snapshot.get("device_traces") or [])


def maybe_start_memory_forensics() -> None:
    """Begin allocator-history recording once per process when enabled."""
    global _started, _max_entries
    if not memory_forensics_enabled():
        return
    with _lock:
        if _started:
            return
        try:
            if not torch.cuda.is_available():
                return
            _max_entries = envs.SGLANG_MEM_FORENSICS_MAX_ENTRIES.get()
            _start_recording()
        except Exception:
            logger.exception("Memory forensics recording failed to start")
            return
        _started = True
        logger.info(
            "Memory forensics recording started (dir=%s, max_entries=%d)",
            envs.SGLANG_MEM_FORENSICS_DIR.get(),
            _max_entries,
        )


def start_memory_history() -> None:
    """Start allocator-history recording for another profiler.

    While memory forensics owns the recorder, keep (or re-arm) its own
    configuration instead of the default one, so its history stays ordered.
    """
    if _owns_recorder():
        _start_recording()
    else:
        torch.cuda.memory._record_memory_history()


def stop_memory_history() -> None:
    """Stop recording begun by :func:`start_memory_history`; while memory
    forensics owns the recorder, leave it and its history in place."""
    if _owns_recorder():
        return
    torch.cuda.memory._record_memory_history(enabled=None)


def maybe_dump_memory_forensics(tag: str) -> None:
    """Write one snapshot pickle per (process, tag).

    The snapshot is written to a temporary file and moved into place
    atomically; the tag is consumed only after a successful write of a
    snapshot with history, so a transient failure does not suppress a later
    retry. A snapshot without history entries is still written, recording
    is re-armed and the tag stays unconsumed (see the module docstring).
    File names embed the distributed rank, the PID, and a nanosecond
    timestamp, so concurrent data-parallel replicas, restarts, and multiple
    servers sharing the directory cannot collide.
    """
    if not _started:
        return
    out_dir = envs.SGLANG_MEM_FORENSICS_DIR.get()
    if not out_dir:
        return
    with _lock:
        if tag in _dumped_tags:
            return
        try:
            path = os.path.join(
                out_dir,
                f"mem-forensics-{tag}-rank{_rank_label()}"
                f"-pid{os.getpid()}-{time.time_ns()}.pickle",
            )
            snapshot = torch.cuda.memory._snapshot()
            has_history = _snapshot_has_history(snapshot)
            if not has_history:
                # Another memory profiler stopped the process-wide recorder
                # and its events are gone, but live blocks keep their
                # allocation stacks: re-arm, and write this snapshot; the
                # next request for this tag carries history again.
                logger.warning(
                    "Memory forensics snapshot %r has no allocator history "
                    "(another memory profiler stopped recording); writing it "
                    "with live-block stacks only and re-arming recording.",
                    tag,
                )
                try:
                    _start_recording()
                except Exception:
                    logger.exception("Memory forensics recording failed to re-arm")
            os.makedirs(out_dir, exist_ok=True)
            tmp_path = path + ".tmp"
            try:
                with open(tmp_path, "wb") as file:
                    pickle.dump(snapshot, file)
                os.replace(tmp_path, path)
            except BaseException:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise
        except Exception:
            logger.exception("Memory forensics dump failed for tag %r", tag)
            return
        if has_history:
            _dumped_tags.add(tag)
        logger.info("Memory forensics snapshot written: %s", path)
