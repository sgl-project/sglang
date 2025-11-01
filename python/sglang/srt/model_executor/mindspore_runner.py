# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""ms_runner launch MindSpore distributed modules."""

import multiprocessing as mp
import os
import sys
from pathlib import Path

import mindspore as ms
import torch
from mindspore._c_expression import GroupOptions
from mindspore.communication import create_group

from sglang.srt.distributed.parallel_state import _groups


class _Tmp:
    def __init__(self):
        self.sched_p = None

    def set_sched_process(self, p):
        self.sched_p = p

    def __del__(self):
        if self.sched_p:
            self.sched_p.kill()


_tmp = _Tmp()


def _get_host_and_ip(distributed_init_method):
    try:
        _, ip_str, port_str = distributed_init_method.split(":")
        ip = ip_str.split("/")[-1]
        port = int(port_str)
    except Exception as e:
        raise RuntimeError(
            "Cannot get host and port information from %s, error: %s!"
            % (distributed_init_method, str(e))
        )

    return ip, port


def run_scheduler_init(rank, local_rank, world_size, master_addr, master_port):
    with open(str(Path() / "schedule.log"), "w") as scheduler_f:
        # For Python outputs.
        sys.stdout = scheduler_f
        sys.stderr = scheduler_f
        # For C++ outputs.
        os.dup2(scheduler_f.fileno(), 1)
        os.dup2(scheduler_f.fileno(), 2)
        os.environ["DEVICE_ID"] = str(local_rank)
        os.environ["MS_WORKER_NUM"] = str(world_size)
        os.environ["MS_ROLE"] = "MS_SCHED"
        os.environ["MS_NODE_ID"] = str(rank)
        os.environ["MS_SCHED_HOST"] = str(master_addr)
        os.environ["MS_SCHED_PORT"] = str(master_port)
        # This function is blocked until the whole cluster exits.
        ms.communication.init()


def set_ms_parallel_env(rank, local_rank, world_size, init_method):
    master_addr, master_port = _get_host_and_ip(init_method)
    # change port avoiding port conflicts with torch
    master_port = master_port + 35 if master_port < 65500 else master_port - 35
    if not os.getenv("MS_ROLE"):
        if rank == 0:
            # Create a subprocess for scheduler of MindSpore, just for internal collaboration, not for collective communication
            sched_p = mp.Process(
                target=run_scheduler_init,
                args=(rank, local_rank, world_size, master_addr, master_port),
            )
            sched_p.start()
            global _tmp
            _tmp.set_sched_process(sched_p)

        os.environ["DEVICE_ID"] = str(local_rank)
        os.environ["MS_WORKER_NUM"] = str(world_size)
        os.environ["MS_ROLE"] = "MS_WORKER"
        os.environ["MS_NODE_ID"] = str(rank)
        os.environ["MS_SCHED_HOST"] = str(master_addr)
        os.environ["MS_SCHED_PORT"] = str(master_port)


def reuse_hccl_comm():
    for group_name, group in _groups.items():
        # Torch ProcessGroupHccl
        device_group = group().device_group
        hccl_comm_handle = device_group._get_backend(torch.device("npu")).get_hccl_comm(
            group().local_rank
        )
        print(
            f"MindSpore reuse torch group: {device_group}, group_name: {group_name}, local rank: {group().local_rank},"
            f"hccl communicator handle: {hex(hccl_comm_handle)}",
            flush=True,
        )
        # Create MS communication group by hccl comm handle to reuse Torch group.
        group_options = GroupOptions()
        group_options.hccl_config = {"hccl_comm": hccl_comm_handle}
        create_group(group_name, group().ranks, group_options)


def init_ms_distributed(world_size, rank, local_rank, server_args, port):
    if server_args.dist_init_addr:
        dist_init_method = f"tcp://{server_args.dist_init_addr}"
    else:
        dist_init_method = f"tcp://{server_args.host}:{port}"
    set_ms_parallel_env(rank, local_rank, world_size, dist_init_method)

    ms.set_context(infer_boost="on", jit_level="O0")
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    ms.set_device("Ascend", local_rank)
    ms.communication.init("hccl")
    # After distributed job is initialized, reuse hccl comms for MindSpore.
    reuse_hccl_comm()
