# SPDX-License-Identifier: Apache-2.0
"""Read checkpoint files into the OS page cache while the workers start.

Startup reads every resident weight once, and on a network volume one reader is
far below what the storage serves to many. The launcher starts helper processes
as soon as the server arguments name the model; they read the checkpoint in
parallel while the workers are still importing, so the loaders later find the
pages already resident.

This module is also the helper's entry point and must stay importable without
the rest of sglang: helpers run it as a plain script, without site-packages.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable

_CHUNK_BYTES = 32 << 20
# One process reads a network volume far below what it serves to several:
# measured on Modal B200 hosts, 8 x 4 was the fastest split of 32 readers.
_PROCESSES = 8
_THREADS_PER_PROCESS = 4
# Arbitrary; leaves room for the workers' own host memory while they load.
_MAX_PAGE_CACHE_FRACTION = 0.8


def _pipeline_root(
    model_path: str,
    *,
    revision: str | None,
    model_subfolder: str | None,
    model_variant: str | None,
) -> str | None:
    """The local directory whose model_index.json the pipeline will load, if known.

    A repository may keep several pipelines in partitions, and the pipeline
    class picks one. Only a choice the arguments spell out is followed: reading
    another partition would fill memory with weights nobody loads.
    """
    root = model_path
    if not os.path.isdir(root):
        # Only the launcher gets here; the helper script never needs site-packages.
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            root = snapshot_download(root, revision=revision, local_files_only=True)
        except (LocalEntryNotFoundError, OSError, ValueError):
            return None

    def has_index(directory: str) -> bool:
        return os.path.isfile(os.path.join(directory, "model_index.json"))

    if model_subfolder is not None:
        chosen = os.path.join(root, model_subfolder)
        return chosen if has_index(chosen) else None
    partitions = [
        name for name in sorted(os.listdir(root)) if has_index(os.path.join(root, name))
    ]
    if not partitions:
        return root if has_index(root) else None
    named = [
        name for name in partitions if name.lower() == (model_variant or "").lower()
    ]
    return os.path.join(root, named[0]) if len(named) == 1 else None


def pipeline_weight_files(
    model_path: str,
    *,
    revision: str | None,
    model_subfolder: str | None,
    model_variant: str | None,
    component_overrides: dict[str, str],
) -> list[str]:
    """Local safetensors files of the components the pipeline index declares.

    An approximation the loaders refine. It never downloads, and a component
    whose override is not a local path is left out.
    """
    from sglang.multimodal_gen.runtime.managers.memory_managers.component_loading_order import (
        _list_component_safetensors_files,
    )

    root = _pipeline_root(
        model_path,
        revision=revision,
        model_subfolder=model_subfolder,
        model_variant=model_variant,
    )
    if root is None:
        return []
    with open(os.path.join(root, "model_index.json")) as f:
        model_index = json.load(f)
    files: list[str] = []
    for name, spec in model_index.items():
        if name.startswith("_") or not isinstance(spec, list) or spec[0] is None:
            continue
        component = component_overrides.get(name) or os.path.join(root, name)
        files.extend(
            _list_component_safetensors_files(component)
            # Some components keep their weights one folder down.
            or sorted(glob.glob(os.path.join(component, "*", "*.safetensors")))
        )
    return files


def _host_memory_headroom_bytes() -> int | None:
    """Memory the page cache can grow into: MemAvailable, capped by the cgroup."""
    available = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
        with open("/sys/fs/cgroup/memory.max") as f:
            limit = f.read().strip()
        with open("/sys/fs/cgroup/memory.current") as f:
            current = int(f.read().strip())
    except (OSError, ValueError):
        return available
    if available is not None and limit != "max":
        available = min(available, int(limit) - current)
    return available


def _chunks(files: list[tuple[str, int]]) -> list[tuple[str, int, int]]:
    """(path, offset, length) pieces of at most _CHUNK_BYTES covering every file."""
    return [
        (path, offset, min(_CHUNK_BYTES, size - offset))
        for path, size in files
        for offset in range(0, size, _CHUNK_BYTES)
    ]


def start_weight_prefetch(
    paths: list[str],
    *,
    on_done: Callable[[int, float], None] | None = None,
) -> int:
    """Start helper processes that read ``paths`` once, and return the bytes planned.

    Returns 0 without starting anything when there is nothing to read or the
    files would not fit in the page cache. ``on_done(bytes_read, seconds)`` runs
    on a background thread after the last helper exits.
    """
    # Hub caches link names to shared blobs; read each blob once.
    files = [
        (path, os.path.getsize(path))
        for path in sorted(set(map(os.path.realpath, paths)))
    ]
    total = sum(size for _, size in files)
    headroom = _host_memory_headroom_bytes()
    if (
        total == 0
        or headroom is None
        or total > _MAX_PAGE_CACHE_FRACTION * headroom
        or not hasattr(os, "preadv")
    ):
        return 0
    fd, plan_path = tempfile.mkstemp(prefix="sglang-weight-prefetch-", suffix=".json")
    with os.fdopen(fd, "w") as plan:
        json.dump(files, plan)
    helpers = [
        subprocess.Popen(
            [
                sys.executable,
                "-S",
                os.path.abspath(__file__),
                str(index),
                str(_PROCESSES),
                str(_THREADS_PER_PROCESS),
                plan_path,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for index in range(_PROCESSES)
    ]
    start = time.monotonic()

    def reap() -> None:
        bytes_read = 0
        for helper in helpers:
            output, _ = helper.communicate()
            bytes_read += int(output.strip() or 0)
        os.unlink(plan_path)
        if on_done is not None:
            on_done(bytes_read, time.monotonic() - start)

    threading.Thread(target=reap, name="weight-prefetch-reap", daemon=True).start()
    return total


def _read_share(
    chunks: list[tuple[str, int, int]], index: int, stride: int, threads: int
) -> int:
    """Read every ``stride``-th chunk from ``index`` and return the bytes read."""
    cursor = iter(chunks[index::stride])
    lock = threading.Lock()
    fds: dict[str, int] = {}
    bytes_read = 0

    def worker() -> None:
        nonlocal bytes_read
        view = memoryview(bytearray(_CHUNK_BYTES))
        while True:
            with lock:
                job = next(cursor, None)
                if job is None:
                    return
                path, offset, length = job
                fd = fds.get(path)
                if fd is None:
                    fd = fds[path] = os.open(path, os.O_RDONLY)
            done = 0
            while done < length:
                got = os.preadv(fd, [view[done:length]], offset + done)
                if got <= 0:
                    break
                done += got
            with lock:
                bytes_read += done

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    for fd in fds.values():
        os.close(fd)
    return bytes_read


def _exit_with_parent() -> None:
    """Ask Linux to kill this helper when the launcher that started it dies.

    A copy of ``runtime.utils.process.kill_itself_when_parent_died``, which the
    helper cannot import: it runs without site-packages.
    """
    try:
        import ctypes
        import signal

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except OSError:
        pass


def _main(argv: list[str]) -> None:
    index, stride, threads = map(int, argv[:3])
    plan_path = argv[3]
    _exit_with_parent()
    with open(plan_path) as plan:
        files = [(path, size) for path, size in json.load(plan)]
    print(_read_share(_chunks(files), index, stride, threads), flush=True)


if __name__ == "__main__":
    _main(sys.argv[1:])
