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
from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    qr_rocm_arch_available,
)
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
)
from sglang.test.test_utils import CustomTestCase

torch.manual_seed(42)
random.seed(44)  # keep the deterministic seed


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
    world_size: int, cls: Any, test_target: Any, quant_mode: str
) -> None:

    # Using ray helps debugging the error when it failed
    # as compared to multiprocessing.
    # NOTE: We need to set working_dir for distributed tests,
    # otherwise we may get import errors on ray workers

    ray.init(log_to_driver=True)

    distributed_init_port = get_open_port()
    refs = []
    for rank in range(world_size):
        refs.append(
            test_target.remote(cls, world_size, rank, distributed_init_port, quant_mode)
        )
    ray.get(refs)

    ray.shutdown()


class TestQuickAllReduce(CustomTestCase):
    TEST_SIZES = [
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        16 * 1024 * 1024,
        32 * 1024 * 1024,
    ]
    TEST_LOOP = 5
    # Too many configurations can lead to a test grid that is too large
    # The tp takes too long to boot,let's just choose 4 out of 12 configurations
    # WORLD_SIZES = [2, 4, 8]
    # QUANT_MODE = ["FP", "INT8", "INT6", "INT4"]
    QUANT_MODE_WORLD_SIZE_PART = [["FP", 8], ["INT4", 4], ["INT8", 2], ["INT6", 2]]

    @unittest.skipIf(
        not qr_rocm_arch_available(),
        "Only test Quick AllReduce on ROCm architectures >= gfx94*",
    )
    def test_graph_allreduce(self):
        for quant_mode_world_size_part in self.QUANT_MODE_WORLD_SIZE_PART:
            quant_mode = quant_mode_world_size_part[0]
            world_size = quant_mode_world_size_part[1]
            if world_size > torch.cuda.device_count():
                continue
            multi_process_parallel(world_size, self, self.graph_allreduce, quant_mode)

    @unittest.skipIf(
        not qr_rocm_arch_available(),
        "Only test Quick AllReduce on ROCm architectures >= gfx94*",
    )
    def test_eager_allreduce(self):
        for quant_mode_world_size_part in self.QUANT_MODE_WORLD_SIZE_PART:
            quant_mode = quant_mode_world_size_part[0]
            world_size = quant_mode_world_size_part[1]
            if world_size > torch.cuda.device_count():
                continue
            multi_process_parallel(world_size, self, self.eager_allreduce, quant_mode)

    @ray.remote(num_gpus=1, max_calls=1)
    def graph_allreduce(self, world_size, rank, distributed_init_port, quant_mode):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = quant_mode
        os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "0"
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
            for dtype in [torch.float16, torch.bfloat16]:
                for _ in range(self.TEST_LOOP):
                    with graph_capture() as graph_capture_context:
                        # use integers so result matches NCCL exactly
                        inp1 = torch.randint(
                            1,
                            23,
                            (sz,),
                            dtype=dtype,
                            device=torch.cuda.current_device(),
                        )
                        inp2 = torch.randint(
                            -23,
                            1,
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
                    atol = 1.25 * world_size
                    rtol = 0.5 * world_size
                    for inp, out in [[inp1, out1], [inp2, out2]]:
                        torch.testing.assert_close(out, inp, atol=atol, rtol=rtol)
                        # try:
                        #     torch.testing.assert_close(out, inp, atol=atol, rtol=rtol)
                        # except AssertionError as e:
                        #     print("Max abs diff:", (out - inp).abs().max())
                        #     print("Max rel diff:", ((out - inp).abs() / inp.abs().clamp(min=1e-5)).max())

    @ray.remote(num_gpus=1, max_calls=1)
    def eager_allreduce(self, world_size, rank, distributed_init_port, quant_mode):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = quant_mode
        os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "0"
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
            for dtype in [torch.float16, torch.bfloat16]:
                for _ in range(self.TEST_LOOP):
                    inp1 = torch.randint(
                        1,
                        23,
                        (sz,),
                        dtype=dtype,
                        device=torch.cuda.current_device(),
                    )
                    out1 = tensor_model_parallel_all_reduce(inp1)
                    dist.all_reduce(inp1, group=group)
                    atol = 1.25 * world_size
                    rtol = 0.5 * world_size
                    torch.testing.assert_close(out1, inp1, atol=atol, rtol=rtol)
                    # try:
                    #     torch.testing.assert_close(out1, inp1, atol=atol, rtol=rtol)
                    # except AssertionError as e:
                    #     print("Max abs diff:", (out1 - inp1).abs().max())
                    #     print("Max rel diff:", ((out1 - inp1).abs() / inp1.abs().clamp(min=1e-5)).max())


if __name__ == "__main__":
    unittest.main()
