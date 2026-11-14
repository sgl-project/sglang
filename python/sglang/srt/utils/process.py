# Copyright 2023-2024 SGLang Team
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
"""Process lifecycle helpers: killing process trees, ulimits, CPU affinity."""

from __future__ import annotations

import ctypes
import itertools
import logging
import os
import resource
import shutil
import signal
import sys
import time
from typing import Optional

import psutil

from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

__all__ = [
    "kill_process_tree",
    "set_ulimit",
    "delete_directory",
    "set_gpu_proc_affinity",
    "kill_itself_when_parent_died",
]


def _still_holding_resources(procs):
    """Procs still holding GPU context, pinned memory or fds.

    A zombie has already had its resources freed by the kernel (only the exit
    status lingers), so it counts as gone; NoSuchProcess / OSError (see
    _wait_for_reap_or_raise) mean the same.
    """
    alive = []
    for p in procs:
        try:
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                alive.append(p)
        except (psutil.NoSuchProcess, OSError):
            pass
    return alive


def _wait_for_reap_or_raise(procs, wait_timeout: float) -> None:
    """Wait for `procs` to exit; warn at ~10s, raise on `wait_timeout`.

    SIGKILL is asynchronous -- children hold GPU context, pinned memory and
    fds until the kernel reaps them. Raise on timeout so a stuck process
    surfaces instead of leaving a latent race.

    Polls /proc via is_running()/status() rather than psutil.wait_procs, whose
    os.pidfd_open path (used for non-child procs) raises OSError(EINVAL) against
    a just-killed process on some kernels and aborts the whole wait.
    """
    warn_at = min(10.0, wait_timeout / 2)
    deadline = time.monotonic() + wait_timeout
    warn_deadline = time.monotonic() + warn_at
    warned = False
    while True:
        alive = _still_holding_resources(procs)
        if not alive:
            return
        now = time.monotonic()
        if now >= deadline:
            raise RuntimeError(
                f"kill_process_tree: {len(alive)} process(es) not reaped within "
                f"{wait_timeout}s after SIGKILL; pids={[p.pid for p in alive]}"
            )
        if not warned and now >= warn_deadline:
            logger.warning(
                "kill_process_tree: %d process(es) still alive after %.1fs SIGKILL; "
                "continuing to wait up to %.1fs total. pids=%s",
                len(alive),
                warn_at,
                wait_timeout,
                [p.pid for p in alive],
            )
            warned = True
        time.sleep(0.1)


def kill_process_tree(
    parent_pid,
    include_parent: bool = True,
    skip_pid: int = None,
    wait_timeout: Optional[float] = 60,
):
    """Kill the process and all its child processes.

    `wait_timeout` (seconds) blocks until every killed process is reaped and
    raises `RuntimeError` on timeout. SIGKILL only queues the teardown, so
    returning without waiting leaves the GPU context, the pinned host memory
    and the ports held for seconds; pass `None` only where blocking is
    unacceptable, such as a `__del__`. The `parent_pid == os.getpid()` branch
    calls `sys.exit(0)` and cannot wait for itself -- use
    `include_parent=False` if child reap must finish first.
    """
    logger.info(
        f"kill_process_tree called: parent_pid={parent_pid}, "
        f"include_parent={include_parent}, pid={os.getpid()}"
    )

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
        children = itself.children(recursive=True)
    except psutil.NoSuchProcess:
        return

    killed = []
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
            killed.append(child)
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
            killed.append(itself)
        except psutil.NoSuchProcess:
            pass

    if wait_timeout is not None and killed:
        _wait_for_reap_or_raise(killed, wait_timeout)


def set_ulimit(target_soft_limit=65535):
    # number of open files
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(f"Fail to set RLIMIT_NOFILE: {e}")

    # stack size
    resource_type = resource.RLIMIT_STACK
    current_soft, current_hard = resource.getrlimit(resource_type)
    target_soft_limit_stack_size = 1024 * target_soft_limit
    if current_soft < target_soft_limit_stack_size:
        try:
            resource.setrlimit(
                resource_type, (target_soft_limit_stack_size, current_hard)
            )
        except ValueError as e:
            logger.warning(f"Fail to set RLIMIT_STACK: {e}")


def delete_directory(dirpath):
    try:
        # This will remove the directory and all its contents
        shutil.rmtree(dirpath)
    except OSError as e:
        logger.warning("Failed to delete directory %s: %s", dirpath, e.strerror)


def set_gpu_proc_affinity(gpu_id: int):
    # current process
    pid = os.getpid()
    p = psutil.Process(pid)

    parallel = get_parallel()
    nnodes_per_tp_group = max(parallel.nnodes // parallel.pp_size, 1)
    tp_size_per_node = parallel.tp_size // nnodes_per_tp_group

    # total physical cores
    total_pcores = psutil.cpu_count(logical=False)
    # physical cores per TP (N.B. more Cores than GPUs on node)
    num_cores_bind = total_pcores // tp_size_per_node

    # able to handle multiple DP per node
    start_cpu_id = (gpu_id * num_cores_bind) % total_pcores
    end_cpu_id = start_cpu_id + num_cores_bind

    if psutil.cpu_count() != psutil.cpu_count(logical=False):
        # HT on
        lower_cpu_ids = [id for id in range(start_cpu_id, end_cpu_id)]
        upper_cpu_ids = [id + total_pcores for id in range(start_cpu_id, end_cpu_id)]
        bind_cpu_ids = list(itertools.chain(lower_cpu_ids, upper_cpu_ids))
    else:
        # HT off
        bind_cpu_ids = [id for id in range(start_cpu_id, end_cpu_id)]

    # set cpu_affinity to current process
    p.cpu_affinity(bind_cpu_ids)
    logger.info(f"Process {pid} gpu_id {gpu_id} is running on CPUs: {p.cpu_affinity()}")


def kill_itself_when_parent_died():
    if sys.platform == "linux":
        # sigkill this process when parent worker manager dies
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    elif sys.platform == "darwin":
        # macOS has no PR_SET_PDEATHSIG equivalent; the MLX backend provides a
        # kqueue-based watchdog that SIGKILLs this worker once it is orphaned.
        from sglang.srt.hardware_backend.mlx.parent_watchdog import (
            start_parent_death_watcher,
        )

        start_parent_death_watcher()
    else:
        logger.warning(
            "kill_itself_when_parent_died is only supported on linux and macOS."
        )
