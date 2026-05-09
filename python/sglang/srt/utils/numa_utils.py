import ctypes
import glob
import logging
import math
import multiprocessing
import os
import random
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import psutil

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
            numactl_args = f"--cpunodebind={numa_node} --membind={numa_node}"
            executable, debug_str = _create_numactl_executable(
                numactl_args=numactl_args
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
            # get_numa_node_for_gpu could return multiple nodes, we use the first one for now.
            # I don't think there any hardware configs that would have more than one.
            logger.warning(
                f"Multiple NUMA nodes found for GPU {gpu_id}: {queried_numa_node}. Using the first one."
            )
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
    else:
        libnuma.numa_run_on_node(ctypes.c_int(node))
        libnuma.numa_set_preferred(ctypes.c_int(node))


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

    # Check if affinity is already constrained
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_affinity = process.cpu_affinity()
    all_cpus = list(range(psutil.cpu_count()))
    constrained_affinity = cpu_affinity != all_cpus
    if constrained_affinity:
        logger.warning(
            "NUMA affinity is already constrained for process, skipping NUMA node configuration for GPU. Remove your constraints to allow automatic configuration."
        )
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
        device_id: GPU device index.
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

        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
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
    tp_rank,
    tp_size,
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
