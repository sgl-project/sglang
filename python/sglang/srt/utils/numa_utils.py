import logging
import multiprocessing
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@contextmanager
def configure_subprocess(server_args: ServerArgs, gpu_id: int):
    if (
        numa_nodes := server_args.numa_node
    ) is not None and envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = numa_nodes[gpu_id]
        numactl_args = f"--cpunodebind={numa_node} --membind={numa_node}"
        executable, debug_str = _create_numactl_executable(numactl_args=numactl_args)
        with _mp_set_executable(executable=executable, debug_str=debug_str):
            yield
    else:
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
    logger.info(f"mp.set_executable {old_executable} -> {executable} ({debug_str})")
    try:
        yield
    finally:
        assert (
            os.fsdecode(multiprocessing.spawn.get_executable()) == executable
        ), f"{multiprocessing.spawn.get_executable()=}"
        multiprocessing.spawn.set_executable(old_executable)
        logger.info(f"mp.set_executable revert to {old_executable}")
