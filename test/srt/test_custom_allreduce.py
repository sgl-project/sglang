import os
import random
import socket
import unittest
from typing import Any

import ray
import torch
import torch.distributed as dist

from sglang.srt.distributed import init_distributed_environment
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

    # Using ray helps debugging the error when it failed
    # as compared to multiprocessing.
    # NOTE: We need to set working_dir for distributed tests,
    # otherwise we may get import errors on ray workers

    ray.init(log_to_driver=True)

    distributed_init_port = get_open_port()
    refs = []
    for rank in range(world_size):
        refs.append(test_target.remote(cls, world_size, rank, distributed_init_port))
    ray.get(refs)

    ray.shutdown()


class TestCustomAllReduce(CustomTestCase):
    TEST_SIZES = [
        512,
        4096,
        32768,
        262144,
        2097152,
        16777216,
        33554432,
    ]  # 512B...32MB
    WORLD_SIZES = [2, 4, 6, 8]
    TEST_LOOP = 10

    @classmethod
    def setUpClass(cls):
        random.seed(42)  # keep the deterministic seed

    def test_graph_allreduce(self):
        for world_size in self.WORLD_SIZES:
            if world_size > torch.cuda.device_count():
                continue
            multi_process_parallel(world_size, self, self.graph_allreduce)

    def test_eager_allreduce(self):
        for world_size in self.WORLD_SIZES:
            if world_size > torch.cuda.device_count():
                continue
            multi_process_parallel(world_size, self, self.eager_allreduce)

    @ray.remote(num_gpus=1, max_calls=1)
    def graph_allreduce(self, world_size, rank, distributed_init_port):
        del os.environ["CUDA_VISIBLE_DEVICES"]
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        distributed_init_method = f"tcp://localhost:{distributed_init_port}"
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

        for sz in self.TEST_SIZES:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                for _ in range(self.TEST_LOOP):
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
                    torch.testing.assert_close(out1, inp1)
                    torch.testing.assert_close(out2, inp2)

    @ray.remote(num_gpus=1, max_calls=1)
    def eager_allreduce(self, world_size, rank, distributed_init_port):
        del os.environ["CUDA_VISIBLE_DEVICES"]
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        distributed_init_method = f"tcp://localhost:{distributed_init_port}"
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        group = get_tensor_model_parallel_group().device_group

        for sz in self.TEST_SIZES:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                for _ in range(self.TEST_LOOP):
                    inp1 = torch.randint(
                        1, 16, (sz,), dtype=dtype, device=torch.cuda.current_device()
                    )
                    out1 = tensor_model_parallel_all_reduce(inp1)
                    dist.all_reduce(inp1, group=group)
                    torch.testing.assert_close(out1, inp1)


if __name__ == "__main__":
    unittest.main()
