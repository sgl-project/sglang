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

import asyncio
import ctypes
import logging
import os
import resource
import shutil
import signal
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
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
    "disable_transparent_hugepages",
]

_PR_SET_THP_DISABLE = 41


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


def _thread_siblings(cpu: int) -> str:
    """The kernel's list of the threads sharing ``cpu``'s core, ``cpu`` itself when unknown."""
    try:
        with open(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        ) as f:
            return f.read().strip()
    except OSError:
        return str(cpu)


def physical_cores(cpus) -> list[list[int]]:
    """``cpus`` grouped by physical core, a core's threads in one list."""
    cores = {}
    for cpu in sorted(cpus):
        cores.setdefault(_thread_siblings(cpu), []).append(cpu)
    return sorted(cores.values())


def _share(cores: list[list[int]], rank: int, num_ranks: int) -> list[int]:
    """Rank ``rank``'s slice of ``cores`` split ``num_ranks`` ways, every thread of its cores."""
    # physical cores per TP (N.B. more Cores than GPUs on node)
    num_cores_bind = max(len(cores) // num_ranks, 1)
    # able to handle multiple DP per node
    start = (rank * num_cores_bind) % len(cores)
    return sorted(c for core in cores[start : start + num_cores_bind] for c in core)


def allocate_cpus(starts: dict[int, list[list[int]]]) -> dict[int, list[int]]:
    """Each scheduler's CPUs, by GPU, from the physical cores it may use.

    Every scheduler first gets one core of its own where the masks allow it (a
    matching); the rest are claimed in turns, each turn taking the free core the fewest
    others may use. Those counts are then placed as runs of free cores, the schedulers
    with the fewest cores to choose from first; when runs cannot hold them, the claims
    stand as they are. A scheduler left without a core shares all of its own."""
    cores = {gpu_id: [tuple(core) for core in own] for gpu_id, own in starts.items()}
    demand = Counter(core for own in cores.values() for core in own)
    order = sorted(cores, key=lambda gpu_id: (len(cores[gpu_id]), gpu_id))

    owner = {}

    def claim(gpu_id, seen):
        # the cores the fewest others may use first, as the turns below
        for core in sorted(cores[gpu_id], key=lambda core: (demand[core], core)):
            if core not in seen:
                seen.add(core)
                if core not in owner or claim(owner[core], seen):
                    owner[core] = gpu_id
                    return True
        return False

    for gpu_id in order:
        claim(gpu_id, set())
    claims = {gpu_id: set() for gpu_id in cores}
    for core, gpu_id in owner.items():
        claims[gpu_id].add(core)
    free = set(demand) - set(owner)
    claimed = True
    while claimed:
        claimed = False
        for gpu_id, own in cores.items():
            choices = free.intersection(own)
            if choices:
                core = min(choices, key=lambda core: (demand[core], core))
                free.remove(core)
                claims[gpu_id].add(core)
                claimed = True

    def cpus_of(own):
        return sorted(cpu for core in own for cpu in core)

    plan = {}
    taken = set()
    for i, gpu_id in enumerate(order):
        count = len(claims[gpu_id])
        if count == 0:
            plan[gpu_id] = cpus_of(cores[gpu_id])
            continue
        options = sorted(core for core in cores[gpu_id] if core not in taken)
        if len(options) < count:
            return {
                gpu_id: cpus_of(claims[gpu_id]) if claims[gpu_id] else cpus_of(own)
                for gpu_id, own in cores.items()
            }
        later = Counter(core for other in order[i + 1 :] for core in cores[other])
        first = min(
            range(len(options) - count + 1),
            key=lambda s: (sum(later[core] for core in options[s : s + count]), s),
        )
        run = options[first : first + count]
        taken.update(run)
        plan[gpu_id] = cpus_of(run)
    return plan


def plan_gpu_proc_affinity(server_args, gpu_ids) -> dict[int, list[int]]:
    """The CPUs each local scheduler binds to under ``SGLANG_SET_CPU_AFFINITY``, by GPU.

    A scheduler starts from the launcher's CPUs, narrowed to its GPU's NUMA node when
    NUMA binding places it there; ``allocate_cpus`` splits those starts. Run in the
    launcher, before any scheduler narrows the visible devices."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_SET_CPU_AFFINITY.get():
        return {}
    from sglang.srt.utils import numa_utils
    from sglang.srt.utils.device import maybe_reindex_device_id

    launcher_cpus = set(os.sched_getaffinity(0))
    starts = {}
    for gpu_id in gpu_ids:
        # the device id configure_subprocess looks the NUMA node up with
        with maybe_reindex_device_id(gpu_id) as device_id:
            cpus = numa_utils.numa_bound_cpus(server_args, device_id, launcher_cpus)
        starts[gpu_id] = physical_cores(cpus)
    return allocate_cpus(starts)


class CpuShareBoard:
    """Hands each scheduler on a host its ``allocate_cpus`` share once every scheduler
    the launcher placed on that host has reported the physical cores it may use. For
    launchers that place schedulers on hosts they cannot inspect (Ray)."""

    def __init__(self):
        self._expected = {}
        self._starts = defaultdict(dict)
        self._plans = {}
        self._ready = defaultdict(asyncio.Event)

    def _plan_when_complete(self, host):
        if (
            host not in self._plans
            and host in self._expected
            and len(self._starts[host]) >= self._expected[host]
        ):
            self._plans[host] = allocate_cpus(self._starts[host])
            self._ready[host].set()

    async def expect(self, counts: dict):
        """How many schedulers the launcher placed on each host."""
        self._expected.update(counts)
        for host in counts:
            self._plan_when_complete(host)

    async def register(self, host: str, gpu_id: int, cores, timeout: float = 300.0):
        """This scheduler's CPUs, or None when its host's schedulers are not all known
        within ``timeout`` seconds."""
        self._starts[host][gpu_id] = cores
        self._plan_when_complete(host)
        try:
            await asyncio.wait_for(self._ready[host].wait(), timeout)
        except asyncio.TimeoutError:
            return None
        return self._plans[host].get(gpu_id)


@contextmanager
def gpu_proc_affinity_env(plan: dict[int, list[int]], gpu_id: int):
    """Hand a scheduler spawned inside this block the CPUs ``plan`` gives ``gpu_id``."""
    from sglang.srt.environ import envs

    cpus = plan.get(gpu_id)
    if cpus is None:
        yield
        return
    with envs.SGLANG_GPU_PROC_CPUS.override(",".join(map(str, cpus))):
        yield


def set_gpu_proc_affinity(gpu_id: int, server_args=None):
    from sglang.srt.environ import envs

    # current process
    pid = os.getpid()
    p = psutil.Process(pid)

    planned = envs.SGLANG_GPU_PROC_CPUS.get()
    if planned:
        bind_cpu_ids = [int(c) for c in planned.split(",")]
    else:
        # A launcher without a plan (e.g. Ray): split the CPUs this process may use (a
        # container's CPU slice may hold only part of the host) among the node's ranks,
        # within the NUMA node the binding after this one moves it to.
        cpus = set(p.cpu_affinity())
        if server_args is not None:
            from sglang.srt.utils import numa_utils

            cpus = numa_utils.numa_bound_cpus(server_args, gpu_id, cpus)
        parallel = get_parallel()
        nnodes_per_tp_group = max(parallel.nnodes // parallel.pp_size, 1)
        tp_size_per_node = parallel.tp_size // nnodes_per_tp_group
        bind_cpu_ids = _share(physical_cores(cpus), gpu_id, tp_size_per_node)

    # set cpu_affinity to current process
    try:
        p.cpu_affinity(bind_cpu_ids)
    except (psutil.AccessDenied, PermissionError) as e:
        # as for a NUMA bind the container denies
        if envs.SGLANG_CRASH_ON_NUMA_BIND_FAILURE.get():
            raise
        logger.warning(
            f"Process {pid} gpu_id {gpu_id} may not set its CPU affinity ({e}); "
            f"keeping CPUs {p.cpu_affinity()}"
        )
        return
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


def disable_transparent_hugepages() -> bool:
    if sys.platform != "linux":
        logger.warning("Transparent huge pages can only be disabled on Linux.")
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        args = [ctypes.c_ulong(value) for value in (1, 0, 0, 0)]
        if libc.prctl(_PR_SET_THP_DISABLE, *args) != 0:
            logger.warning(
                "prctl(PR_SET_THP_DISABLE) failed: %s",
                os.strerror(ctypes.get_errno()),
            )
            return False
    except (OSError, AttributeError) as e:
        logger.warning(
            "Cannot disable transparent huge pages: prctl(PR_SET_THP_DISABLE) is "
            "unavailable in libc (%s). THP stays enabled for this process.",
            e,
        )
        return False
    return True
