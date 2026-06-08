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
import torch

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_cuda

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
