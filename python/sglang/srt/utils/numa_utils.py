import ctypes
import glob
import logging
import math
import multiprocessing
import os
import random
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_cpu_ids_by_node, is_cuda

_is_cuda = is_cuda()

logger = logging.getLogger(__name__)


@contextmanager
def configure_subprocess(server_args: ServerArgs, gpu_id: int):
    if envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = get_numa_node_if_available(server_args, gpu_id)
        if numa_node is not None:
            # _numactl_cpu_mem_args returns None (warn/raise) on empty CPU intersection (#26983).
            numactl_args = _numactl_cpu_mem_args(numa_node, gpu_id)
            if numactl_args is not None:
                # Verify numactl can actually apply the binding before we exec it
                # in front of the interpreter; relax the memory policy if not.
                numactl_args, probe_err = _probe_numactl_args(numactl_args)
                if numactl_args is None:
                    # numactl could not apply even a CPU-only binding (e.g.
                    # set_mempolicy(2)/sched_setaffinity(2) blocked by seccomp,
                    # which the read-only get_mempolicy(2) probe in
                    # _can_set_mempolicy cannot detect). Reuse #26983's failure
                    # semantics (warn-and-continue, or raise when
                    # SGLANG_CRASH_ON_NUMA_BIND_FAILURE) with an explicit reason
                    # carrying the captured stderr: the CPU intersection already
                    # succeeded here, so the default "no CPU cores allowed"
                    # message would mislead operators toward the wrong cause.
                    probe_suffix = f": {probe_err}" if probe_err else ""
                    _handle_numa_bind_failure(
                        numa_node,
                        reason=(
                            f"numactl could not apply NUMA binding for node "
                            f"{numa_node} (e.g. set_mempolicy/sched_setaffinity "
                            f"blocked by seccomp, or cpuset rejects the policy)"
                            f"{probe_suffix}; skipping NUMA binding for GPU {gpu_id}."
                        ),
                    )
                    yield
                    return
                executable, debug_str = _create_numactl_executable(
                    numactl_args=numactl_args
                )
                debug_str += (
                    f", logical_gpu_id={gpu_id}, "
                    f"physical_gpu_id={_get_nvml_device_index(gpu_id)}, "
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
                )
                with _mp_set_executable(executable=executable, debug_str=debug_str):
                    yield
                    return
    yield


def _create_numactl_executable(numactl_args: str):
    old_executable = os.fsdecode(multiprocessing.spawn.get_executable())
    script = f'''#!/bin/sh
exec numactl {numactl_args} {old_executable} "$@"'''
    path = Path(
        f"/tmp/sglang_temp_file_{time.time()}_{random.randrange(0, 10000000)}.sh"
    )
    path.write_text(script)
    path.chmod(0o777)
    return str(path), f"{script=}"


@contextmanager
def _mp_set_executable(executable: str, debug_str: str):
    start_method = multiprocessing.get_start_method()
    assert start_method == "spawn", f"{start_method=}"

    old_executable = os.fsdecode(multiprocessing.spawn.get_executable())
    multiprocessing.spawn.set_executable(executable)
    logger.debug(f"mp.set_executable {old_executable} -> {executable} ({debug_str})")
    try:
        yield
    finally:
        assert (
            os.fsdecode(multiprocessing.spawn.get_executable()) == executable
        ), f"{multiprocessing.spawn.get_executable()=}"
        multiprocessing.spawn.set_executable(old_executable)
        logger.debug(f"mp.set_executable revert to {old_executable}")


def _get_nvml_device_index(device_id: int) -> int:
    # _get_nvml_device_index is an internal PyTorch helper, so fall back to
    # device_id directly if the helper is unavailable.
    get_nvml_device_index = getattr(torch.cuda, "_get_nvml_device_index", None)
    if get_nvml_device_index is None:
        logger.warning(
            "torch.cuda._get_nvml_device_index is unavailable; falling back to "
            f"device_id={device_id} as the NVML device index. This may select "
            "the wrong physical GPU when CUDA_VISIBLE_DEVICES reorders devices "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})."
        )
        return device_id
    return get_nvml_device_index(device_id)


def get_numa_node_if_available(server_args: ServerArgs, gpu_id: int) -> Optional[int]:
    """
    Returns the NUMA node for the given GPU id. If it is not set in the server_args, it will try to query the NUMA node for the GPU.
    If the NUMA node is not available, has already been configured externally, or the user lacks permission to set NUMA affinity, it will return None.

    Args:
        server_args: The server arguments.
        gpu_id: The GPU id.

    Returns:
        The NUMA node for the given GPU id or None if it is not available.
    """
    if server_args.numa_node is not None:
        return server_args.numa_node[gpu_id]
    if _is_numa_available():
        queried_numa_node = _query_numa_node_for_gpu(gpu_id)
        if len(queried_numa_node) == 0:
            return None
        if len(queried_numa_node) > 1:
            local_worker_index = (
                gpu_id - server_args.base_gpu_id
            ) // server_args.gpu_id_step
            selected_node = queried_numa_node[
                local_worker_index % len(queried_numa_node)
            ]
            logger.warning(
                f"Multiple NUMA nodes found for GPU {gpu_id}: "
                f"{queried_numa_node}. Selecting node {selected_node}."
            )
            return selected_node
        return queried_numa_node[0]
    return None


def get_libnuma():
    libnuma = None

    for libnuma_so in ["libnuma.so", "libnuma.so.1"]:
        try:
            libnuma = ctypes.CDLL(libnuma_so)
        except OSError as e:
            logger.debug(f"{e}")
            libnuma = None
        if libnuma is not None:
            break
    return libnuma


def numa_bind_to_node(node: int):
    libnuma = get_libnuma()

    if libnuma is None or libnuma.numa_available() < 0:
        logger.warning("numa not available on this system, skip bind action")
        return

    node_cpus = _node_cpus(node)
    if node_cpus:
        allowed_cpus = os.sched_getaffinity(0)
        target_cpus = node_cpus & allowed_cpus
        if not target_cpus:
            _handle_numa_bind_failure(node, allowed_cpus)
            return
        os.sched_setaffinity(0, target_cpus)
    else:
        libnuma.numa_run_on_node(ctypes.c_int(node))
    libnuma.numa_set_preferred(ctypes.c_int(node))


class _Bitmask(ctypes.Structure):
    _fields_ = [("size", ctypes.c_ulong), ("maskp", ctypes.POINTER(ctypes.c_ulong))]


def _node_cpus(node: int) -> set:
    libnuma = get_libnuma()
    if libnuma is None or libnuma.numa_available() < 0:
        return set()
    libnuma.numa_allocate_cpumask.restype = ctypes.POINTER(_Bitmask)
    libnuma.numa_node_to_cpus.argtypes = [ctypes.c_int, ctypes.POINTER(_Bitmask)]
    libnuma.numa_node_to_cpus.restype = ctypes.c_int
    libnuma.numa_bitmask_isbitset.argtypes = [ctypes.POINTER(_Bitmask), ctypes.c_uint]
    libnuma.numa_bitmask_isbitset.restype = ctypes.c_int
    libnuma.numa_bitmask_free.argtypes = [ctypes.POINTER(_Bitmask)]
    mask = libnuma.numa_allocate_cpumask()
    try:
        if libnuma.numa_node_to_cpus(node, mask) != 0:
            return set()
        return {
            i
            for i in range(mask.contents.size)
            if libnuma.numa_bitmask_isbitset(mask, i)
        }
    finally:
        libnuma.numa_bitmask_free(mask)


def _numactl_cpu_mem_args(node: int, gpu_id: int) -> Optional[str]:
    node_cpus = _node_cpus(node)
    if not node_cpus:
        return f"--cpunodebind={node} --membind={node}"
    allowed_cpus = os.sched_getaffinity(0)
    target_cpus = node_cpus & allowed_cpus
    if not target_cpus:
        _handle_numa_bind_failure(node, allowed_cpus, gpu_id)
        return None
    if target_cpus == node_cpus:
        return f"--cpunodebind={node} --membind={node}"
    cpu_list = ",".join(str(c) for c in sorted(target_cpus))
    return f"--physcpubind={cpu_list} --membind={node}"


def _strip_memory_args(numactl_args: str) -> str:
    """Return ``numactl_args`` with the ``--membind`` segment removed, keeping
    only the CPU binding (``--cpunodebind`` / ``--physcpubind``)."""
    return " ".join(
        token for token in numactl_args.split() if not token.startswith("--membind")
    )


def _probe_numactl_args(numactl_args: str) -> tuple[Optional[str], str]:
    """Dry-run ``numactl <args> true`` and fall back to a weaker binding when the
    kernel rejects the strongest one.

    ``configure_subprocess`` applies NUMA binding by exec-ing ``numactl`` in front
    of the Python interpreter (see ``_create_numactl_executable``), so a binding
    that ``numactl`` refuses kills the worker before Python starts, with no
    traceback. ``_can_set_mempolicy`` only probes ``get_mempolicy(2)`` (read),
    which does not catch ``set_mempolicy(2)`` being denied (e.g. by a seccomp
    profile) or a ``--membind`` that the cpuset rejects with ``EINVAL``.

    To avoid that silent crash we probe the requested args and progressively relax
    the *memory* policy while keeping the CPU binding intact::

        --membind=N  ->  --preferred=N  ->  drop the memory segment

    Returns ``(args, last_stderr)``: ``args`` is the strongest binding that
    actually runs, or ``None`` if even CPU-only fails (or ``numactl`` is missing /
    errors out); ``last_stderr`` is the rejection reason numactl printed for the
    strongest binding that was rejected (empty on success), so the caller can
    surface it on the total-failure path.
    """

    def _probe(args: str):
        """Run ``numactl <args> true``; return ``(succeeded, stderr_text)``."""
        try:
            proc = subprocess.run(
                ["numactl", *args.split(), "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                logger.debug(f"numactl probe for {args!r} rejected: {stderr!r}")
            return proc.returncode == 0, stderr
        except Exception as e:
            # Missing numactl, timeout, etc. Treat as "this binding does not work".
            logger.debug(f"numactl probe for {args!r} failed: {e}")
            return False, str(e)

    def _suffix(err: str) -> str:
        return f": {err}" if err else ""

    # 1. Strongest binding: exactly what was requested.
    ok, last_err = _probe(numactl_args)
    if ok:
        return numactl_args, ""

    # 2. Relax a hard --membind=N to a soft --preferred=N. The memory segment here
    #    is always a single node, which maps cleanly onto --preferred (single-node
    #    only). MPOL_PREFERRED is a hint and can succeed where MPOL_BIND is denied.
    if "--membind=" in numactl_args:
        preferred_args = numactl_args.replace("--membind=", "--preferred=")
        ok, _ = _probe(preferred_args)
        if ok:
            logger.warning(
                f"numactl rejected hard memory binding ({numactl_args!r})"
                f"{_suffix(last_err)}; falling back to soft preferred policy "
                f"({preferred_args!r})."
            )
            return preferred_args, ""

    # 3. Drop the memory segment entirely, keep only the CPU binding.
    cpu_only_args = _strip_memory_args(numactl_args)
    if cpu_only_args and cpu_only_args != numactl_args:
        ok, cpu_err = _probe(cpu_only_args)
        if ok:
            logger.warning(
                f"numactl rejected memory binding ({numactl_args!r})"
                f"{_suffix(last_err)}; falling back to CPU-only binding "
                f"({cpu_only_args!r})."
            )
            return cpu_only_args, ""
        last_err = cpu_err

    # 4. Nothing worked.
    return None, last_err


def _handle_numa_bind_failure(
    node: int,
    allowed_cpus=None,
    gpu_id: Optional[int] = None,
    *,
    reason: Optional[str] = None,
) -> None:
    """Emit the NUMA-bind failure warning, or raise it when
    ``SGLANG_CRASH_ON_NUMA_BIND_FAILURE`` is set.

    Two call modes:
      * ``reason is None`` (default): the failure is an empty CPU intersection,
        so the message reports ``allowed_cpus`` (which must be provided).
      * ``reason`` provided: the failure is something else (e.g. numactl rejected
        the binding at runtime); the caller supplies the exact message and
        ``allowed_cpus`` / ``gpu_id`` are not needed.
    """
    if reason is None:
        gpu_str = f" for GPU {gpu_id}" if gpu_id is not None else ""
        reason = (
            f"NUMA node {node} has no CPU cores allowed by the current affinity "
            f"{sorted(allowed_cpus)}, skipping NUMA binding{gpu_str}."
        )
    logger.warning(reason)
    if envs.SGLANG_CRASH_ON_NUMA_BIND_FAILURE.get():
        raise RuntimeError(reason)


def _can_set_mempolicy() -> bool:
    """Check if the process has permission to use NUMA memory policy syscalls."""
    try:
        libnuma = get_libnuma()
        if libnuma is None or libnuma.numa_available() < 0:
            return False
        mode = ctypes.c_int()
        ret = libnuma.get_mempolicy(
            ctypes.byref(mode), None, ctypes.c_ulong(0), None, ctypes.c_ulong(0)
        )
        return ret == 0
    except Exception:
        return False


def _is_numa_available() -> bool:
    """
    Check if NUMA is available and not already configured externally.
    """
    if not _is_cuda:
        return False

    # Check if this is a numa system.
    if not os.path.isdir("/sys/devices/system/node/node1"):
        return False

    if not shutil.which("numactl") and envs.SGLANG_NUMA_BIND_V2.get():
        logger.debug(
            "numactl command not found, skipping NUMA node configuration for GPU. Install numactl (e.g., apt-get install numactl) to enable automatic NUMA binding."
        )
        return False

    if not _can_set_mempolicy():
        logger.warning(
            "User lacks permission to set NUMA affinity, skipping NUMA node configuration for GPU. If using docker, try adding --cap-add SYS_NICE to your docker run command."
        )
        return False

    return True


def _query_numa_node_for_gpu(device_id: int):
    """
    Get the NUMA node affinity list for a GPU device.

    Args:
        device_id: CUDA logical device index (post-CUDA_VISIBLE_DEVICES).
    Returns:
        List of NUMA node IDs that have affinity with the device.
    """
    try:
        import pynvml
    except ModuleNotFoundError:
        logger.warning("pynvml not installed, skipping NUMA node configuration for GPU")
        return []

    try:
        pynvml.nvmlInit()

        # device_id is a CUDA logical index. Convert it to the corresponding
        # NVML index so reordered CUDA_VISIBLE_DEVICES maps to the right GPU.
        # _get_nvml_device_index takes CUDA_VISIBLE_DEVICES into account.
        nvml_device_id = _get_nvml_device_index(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_device_id)
        numa_node_count = len(glob.glob("/sys/devices/system/node/node[0-9]*"))

        c_ulong_bits = ctypes.sizeof(ctypes.c_ulong) * 8
        node_set_size = max(1, math.ceil(numa_node_count / c_ulong_bits))
        node_set = pynvml.nvmlDeviceGetMemoryAffinity(
            handle,
            node_set_size,
            pynvml.NVML_AFFINITY_SCOPE_NODE,
        )

        # Decode the bitmask into a list of NUMA node IDs
        numa_nodes = []
        for node_id in range(numa_node_count):
            mask_array_index = node_id // c_ulong_bits
            mask_bit_index = node_id % c_ulong_bits
            if node_set[mask_array_index] & (1 << mask_bit_index):
                numa_nodes.append(node_id)
        return numa_nodes
    except pynvml.NVMLError as e:
        logger.warning(
            f"NVML error querying memory affinity for GPU {device_id}: {e}, skipping NUMA node configuration for GPU"
        )
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass  # Ignore shutdown errors


def init_threads_binding(
    *,
    tp_rank: int,
    tp_size: int,
):
    omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
    cpu_ids_by_node = get_cpu_ids_by_node()
    n_numa_node = len(cpu_ids_by_node)
    if omp_cpuids == "all":
        assert tp_size <= n_numa_node, (
            f"SGLANG_CPU_OMP_THREADS_BIND is not set, in this case, "
            f"tp_size {tp_size} should be smaller than or equal to number of numa node on the machine {n_numa_node}. "
            f"If you need tp_size to be larger than number of numa node, please set the CPU cores for each tp rank via SGLANG_CPU_OMP_THREADS_BIND explicitly. "
            f"For example, on a machine with 2 numa nodes, where core 0-31 are on numa node 0 and core 32-63 are on numa node 1, "
            f"it is suggested to use -tp 2 and bind tp rank 0 to core 0-31 and tp rank 1 to core 32-63. "
            f"This is the default behavior if SGLANG_CPU_OMP_THREADS_BIND is not set and it is the same as setting SGLANG_CPU_OMP_THREADS_BIND=0-31|32-63. "
            f"If you do need tp_size to be larger than the number of numa nodes, you could set SGLANG_CPU_OMP_THREADS_BIND explicitly for example SGLANG_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|48-63 and run with -tp 4. "
            f"If you don't want each tp rank to use all the cores on one numa node, you could set for example SGLANG_CPU_OMP_THREADS_BIND=0-15|32-47 and run with -tp 2."
        )
        if tp_size < n_numa_node:
            logger.warning(
                f"Detected the current machine has {n_numa_node} numa nodes available, but tp_size is set to {tp_size}, so only {tp_size} numa nodes are used."
            )
        local_omp_cpuid = cpu_ids_by_node[tp_rank]
    else:
        threads_bind_list = omp_cpuids.split("|")
        assert tp_size == len(threads_bind_list), (
            f"SGLANG_CPU_OMP_THREADS_BIND setting must be aligned with TP size parameter ({tp_size}). "
            f"Please double check your settings."
        )
        local_omp_cpuid = threads_bind_list[tp_rank]
        if tp_size > n_numa_node:
            logger.warning(
                f"TP size ({tp_size})is larger than numa node number ({n_numa_node}), "
                f"in this case the available memory amount of each rank cannot be determined in prior. "
                f"Please set proper `--max-total-tokens` to avoid the out-of-memory error."
            )
    return local_omp_cpuid
