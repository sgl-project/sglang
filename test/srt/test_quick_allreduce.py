import multiprocessing
import os
import random
import socket
import unittest
from typing import Any

import ray
import torch
import torch.distributed as dist

from sglang.srt import _custom_ops as ops
from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.communication_op import (  # noqa
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    qr_rocm_arch_available,
)
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
)
from sglang.test.test_utils import CustomTestCase


class TestQuickreduceVariableInput(CustomTestCase):
    """
    When the tensor parallelism is set to 4 or 8, frequent changes
    in the input shape can cause QuickReduce to hang (this issue
    has been observed with the gpt_oss model).
    """

    tp_size = [4, 8]

    def qr_variable_input(self, rank, world_size):
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        qr_max_size = None  # MB
        _ptr = ops.init_custom_qr(rank, world_size, qr_max_size)
        ranks = []
        for i in range(world_size):
            ranks.append(i)
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=rank,
            world_size=world_size,
        )
        cpu_group = torch.distributed.new_group(ranks, backend="nccl")

        handle = ops.qr_get_handle(_ptr)
        world_size = dist.get_world_size(group=cpu_group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=cpu_group)
        ops.qr_open_handles(_ptr, handles)

        num = 1
        s1 = 1024
        while num < 50000:  # 50000 is sufficient to identify issues.
            dtype = torch.float16
            if num % 2 == 0:
                s2 = 1024
                inp1 = torch.zeros(
                    (s1, s2), dtype=dtype, device=torch.cuda.current_device()
                )
            else:
                s2 = 2048
                inp1 = torch.ones(
                    (s1, s2), dtype=dtype, device=torch.cuda.current_device()
                )
            result = torch.empty_like(inp1)
            # FP = 0 INT8 = 1 INT6 = 2 INT4 = 3 NONE = 4
            ops.qr_all_reduce(_ptr, inp1, result, 3, cast_bf2half=True)
            try:
                if inp1[0, 0] == 0:
                    assert torch.all(result == 0)
                else:
                    assert torch.all(result == world_size)
            except AssertionError:
                print("Assertion failed! Allreduce results are incorrect.")
                raise
            num += 1

    @unittest.skipIf(
        not qr_rocm_arch_available(),
        "Only test Quick AllReduce on ROCm architectures >= gfx94*",
    )
    def test_custom_quick_allreduce_variable_input(self, tp_size):
        world_size = tp_size
        if world_size > torch.cuda.device_count():
            return

        multiprocessing.set_start_method("spawn", force=True)
        # 60s is enough
        timeout = 60
        processes = []
        for rank in range(tp_size):
            p = multiprocessing.Process(
                target=self.qr_variable_input, args=(rank, tp_size)
            )
            p.start()
            processes.append((rank, p))
        for rank, p in processes:
            p.join(timeout=timeout)
            if p.is_alive():
                for r, proc in processes:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join()
                raise RuntimeError(
                    f"QuickReduce hang detected after {timeout} seconds!"
                )


if __name__ == "__main__":
    unittest.main()
