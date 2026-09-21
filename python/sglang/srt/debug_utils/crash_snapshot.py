"""CPU-only crash snapshot for asynchronous CUDA faults.

SGLang already collects crash diagnostics, but each of them answers a different
question than the one an asynchronous fault asks:

* CUDA coredumps (``SGLANG_CUDA_COREDUMP``) capture GPU state and need cuda-gdb.
* py-spy dumps (``SGLANG_PYSPY_DUMP_BEFORE_CRASH``) capture where the threads
  were, which for an asynchronous fault is a *later* synchronization point.
* ``--crash-dump-folder`` captures recent requests, for replay.

None of them records which operation was in flight when the sticky error was
raised. That is what makes an illegal memory access hard to attribute: the
failing copy or kernel launch returns normally, and the error only surfaces at an
unrelated frame (PyTorch warns that "the stacktrace below might be incorrect").

With ``SGLANG_DEBUG_CRASH_SNAPSHOT=1`` the last
``SGLANG_DEBUG_CRASH_SNAPSHOT_SIZE`` host-pool KV transfers are kept in a bounded
ring, and the ring plus CPU-only scheduler counters is written to
``<crash_dump_folder>/<hostname>/crash_snapshot_<timestamp>_<pid>.json`` when the
scheduler handles a crash. The folder follows ``--crash-dump-folder`` so the
artifact lands next to the request dump and the device coredumps.

Nothing here touches the GPU. By the time the fault is handled the CUDA context
is usually already unusable -- ``torch.cuda.synchronize()``, ``Tensor.cpu()``,
``torch.empty(device="cuda")``, ``torch.cuda.mem_get_info()`` and
``Stream.query()`` all raise -- so the artifact has to be assembled from Python
state alone.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

DEFAULT_SIZE = 128
DEFAULT_FOLDER = "/tmp/sglang_crash_snapshot"
# A crash dump must stay small enough to attach to an issue.
MAX_TRACEBACK_CHARS = 16 * 1024

_records: deque = deque(maxlen=DEFAULT_SIZE)
_written_path: Optional[str] = None


def is_enabled() -> bool:
    return envs.SGLANG_DEBUG_CRASH_SNAPSHOT.get()


def _ring() -> deque:
    """The ring, resized if the configured capacity changed."""
    global _records
    capacity = max(1, envs.SGLANG_DEBUG_CRASH_SNAPSHOT_SIZE.get())
    if _records.maxlen != capacity:
        _records = deque(_records, maxlen=capacity)
    return _records


def record(kind: str, fields: Dict[str, Any]) -> None:
    """Append one CPU-only event; a no-op unless the snapshot is enabled."""
    if not is_enabled():
        return
    try:
        entry = {"kind": kind, "t": time.time(), "pid": os.getpid()}
        entry.update(fields)
        _ring().append(entry)
    except Exception as e:  # pragma: no cover - never break the caller
        logger.warning(f"crash_snapshot.record failed: {type(e).__name__}: {e}")


def records() -> List[Dict[str, Any]]:
    """A copy of the ring, oldest first, bounded by the configured size."""
    return list(_ring())


def reset() -> None:
    """Drop the recorded events and the once-per-process write flag."""
    global _written_path
    _records.clear()
    _written_path = None


def resolve_folder(server_args: Optional[Any] = None) -> str:
    """Dump folder: ``--crash-dump-folder`` if set, else a temp directory."""
    folder = None
    try:
        from sglang.srt.runtime_context import get_observability

        folder = get_observability().crash_dump_folder
    except Exception:  # pragma: no cover - configuration not published yet
        pass
    if not folder:
        folder = getattr(server_args, "crash_dump_folder", None)
    return folder or DEFAULT_FOLDER


def _harvest(target: Any, name: str, getter) -> Any:
    try:
        return getter()
    except Exception:  # pragma: no cover - attributes vary by build/backend
        return None


def scheduler_context(scheduler: Optional[Any]) -> Dict[str, Any]:
    """CPU-only scheduler counters, so the artifact says what was in flight."""
    if scheduler is None:
        return {}
    batch = _harvest(scheduler, "running_batch", lambda: scheduler.running_batch)
    fields = {
        "forward_ct": _harvest(scheduler, "forward_ct", lambda: scheduler.forward_ct),
        "num_running_reqs": _harvest(
            scheduler, "running_batch.reqs", lambda: len(batch.reqs)
        ),
        "waiting_queue_len": _harvest(
            scheduler, "waiting_queue", lambda: len(scheduler.waiting_queue)
        ),
        "max_total_num_tokens": _harvest(
            scheduler, "max_total_num_tokens", lambda: scheduler.max_total_num_tokens
        ),
        "tp_rank": _harvest(scheduler, "tp_rank", lambda: scheduler.tp_rank),
        "pp_rank": _harvest(scheduler, "pp_rank", lambda: scheduler.pp_rank),
    }
    return {name: value for name, value in fields.items() if value is not None}


def dump_crash_snapshot(
    reason: str,
    *,
    server_args: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    traceback_text: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Write the ring plus CPU-only context; returns the path, or None.

    Written at most once per process, like the crash-dump path in the tokenizer
    manager. Never raises: a diagnostic must not mask the crash it describes.
    """
    global _written_path
    if not is_enabled():
        return None
    if _written_path is not None:
        return _written_path

    try:
        hostname = socket.gethostname()
        directory = os.path.join(resolve_folder(server_args), hostname)
        os.makedirs(directory, exist_ok=True)

        payload: Dict[str, Any] = {
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "hostname": hostname,
            "pid": os.getpid(),
            "scheduler": scheduler_context(scheduler),
            "recent_host_pool_io": records(),
            "extra": extra or {},
        }
        if traceback_text:
            payload["traceback"] = traceback_text[-MAX_TRACEBACK_CHARS:]

        filename = (
            f"crash_snapshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            f"_{os.getpid()}.json"
        )
        path = os.path.join(directory, filename)
        with open(path, "w") as handle:
            # default=str: an unexpected extra must not lose the whole dump.
            json.dump(payload, handle, indent=2, default=str)
        _written_path = path
        logger.error(
            "Wrote crash snapshot (%s) with %d recent host-pool transfer(s): %s",
            reason,
            len(payload["recent_host_pool_io"]),
            path,
        )
        return path
    except Exception as e:
        logger.error(f"Failed to write crash snapshot: {type(e).__name__}: {e}")
        return None
