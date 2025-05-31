import multiprocessing
import os
import random
import socket
import unittest
from typing import Any

import torch
import torch.distributed as dist

from sglang.srt.distributed import (
    init_distributed_environment,
    set_custom_all_reduce,
    set_quick_all_reduce,
)
from sglang.srt.distributed.communication_op import (  # noqa
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
)
from sglang.test.test_utils import CustomTestCase


def get_open_port() -> int:
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int,
    cls: Any,
    test_target: Any,
) -> None:

    distributed_init_port = get_open_port()

    processes = []

    for rank in range(world_size):
        p = multiprocessing.Process(
            target=test_target, args=(world_size, rank, distributed_init_port)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


class TestCustomAllReduce:
    @classmethod
    def setUpClass(cls):
        random.seed(42)
        # 1MB to 32MB
        cls.test_sizes = [4096, 32768, 262144, 2097152, 16777216, 33554432]
        cls.world_sizes = [2, 4, 8]
        cls.test_loop = 10

    def test_graph_allreduce(self):
        for world_size in self.world_sizes:
            if world_size > torch.cuda.device_count():
                continue
            multi_process_parallel(world_size, self, self.graph_allreduce)

    def test_eager_allreduce(self):
        for world_size in self.world_sizes:
            if world_size > torch.cuda.device_count():
                continue
            multi_process_parallel(world_size, self, self.eager_allreduce)

    def graph_allreduce(self, world_size, rank, distributed_init_port):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["QUICK_ALL_REDUCE_LEVEL"] = "1"
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        distributed_init_method = f"tcp://localhost:{distributed_init_port}"
        set_custom_all_reduce(True)
        set_quick_all_reduce(True)
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        group = get_tensor_model_parallel_group().device_group

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.cuda.synchronize()
        del data

        for sz in self.test_sizes:
            for dtype in [torch.float16, torch.bfloat16]:
                for _ in range(self.test_loop):
                    with graph_capture() as graph_capture_context:
                        # use integers so result matches NCCL exactly
                        inp1 = torch.randint(
                            1,
                            16,
                            (sz,),
                            dtype=dtype,
                            device=torch.cuda.current_device(),
                        )
                        inp2 = torch.randint(
                            1,
                            16,
                            (sz,),
                            dtype=dtype,
                            device=torch.cuda.current_device(),
                        )
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(
                            graph, stream=graph_capture_context.stream
                        ):
                            out1 = tensor_model_parallel_all_reduce(inp1)
                            # the input buffer is immediately modified to test
                            # synchronization
                            dist.all_reduce(inp1, group=group)
                            out2 = tensor_model_parallel_all_reduce(inp2)
                            dist.all_reduce(inp2, group=group)
                    graph.replay()
                    for inp, out in [[inp1, out1], [inp2, out2]]:
                        try:
                            torch.testing.assert_close(out, inp, rtol=1e-2, atol=1.0)
                        except AssertionError as e:
                            print("Max abs diff:", (out - inp).abs().max())
                            print(
                                "Max rel diff:",
                                ((out - inp).abs() / inp.abs().clamp(min=1e-5)).max(),
                            )
                            raise

    def eager_allreduce(self, world_size, rank, distributed_init_port):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["QUICK_ALL_REDUCE_LEVEL"] = "1"

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        set_custom_all_reduce(False)
        set_quick_all_reduce(True)
        distributed_init_method = f"tcp://localhost:{distributed_init_port}"
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        group = get_tensor_model_parallel_group().device_group

        for sz in self.test_sizes:
            for dtype in [torch.float16, torch.bfloat16]:
                for _ in range(self.test_loop):
                    inp1 = torch.randint(
                        1, 16, (sz,), dtype=dtype, device=torch.cuda.current_device()
                    )
                    out1 = tensor_model_parallel_all_reduce(inp1)
                    dist.all_reduce(inp1, group=group)
                    try:
                        torch.testing.assert_close(out1, inp1, rtol=1e-2, atol=1.0)
                    except AssertionError as e:
                        print("Max abs diff:", (out1 - inp1).abs().max())
                        print(
                            "Max rel diff:",
                            ((out1 - inp1).abs() / inp1.abs().clamp(min=1e-5)).max(),
                        )
                        raise


if __name__ == "__main__":
    unittest.main()
