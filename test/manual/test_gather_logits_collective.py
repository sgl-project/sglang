"""Multi-GPU parity test for the point-to-point logits gather (#3365).

Checks that ``tensor_model_parallel_gather`` (``GroupCoordinator.gather``)
reconstructs the same tensor as all-gather on the destination rank (identical
token ids), returns ``None`` elsewhere, and works both eager and inside a CUDA
graph, across world sizes / shapes / dtypes.

Run: python -m pytest test/manual/test_gather_logits_collective.py -s
"""

import os
import socket
import unittest
from typing import Any

import ray
import torch

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
)
from sglang.test.test_utils import CustomTestCase


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def multi_process_parallel(world_size: int, cls: Any, test_target: Any) -> None:
    ray.init(log_to_driver=True)
    distributed_init_port = get_open_port()
    refs = []
    for rank in range(world_size):
        refs.append(test_target.remote(cls, world_size, rank, distributed_init_port))
    ray.get(refs)
    ray.shutdown()


# [n_tokens, vocab_shard_per_rank] logits-like shapes plus a couple of others.
TEST_SHAPES = [(1, 4096), (3, 8192), (7, 2048)]
TEST_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


class TestGatherLogitsCollective(CustomTestCase):
    def test_gather_matches_all_gather(self):
        world_size = torch.cuda.device_count()
        if world_size < 2:
            self.skipTest("Need at least 2 GPUs")
        # Cap to 4 to keep the test quick even on 8-GPU hosts.
        world_size = min(world_size, 4)
        multi_process_parallel(world_size, self, self.gather_worker)

    @ray.remote(num_gpus=1, max_calls=1)
    def gather_worker(self, world_size, rank, distributed_init_port):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://localhost:{distributed_init_port}",
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        group = get_tensor_model_parallel_group().device_group

        # Warmup so the NCCL communicator is initialized before graph capture.
        warm = torch.zeros(1, device=device)
        torch.distributed.all_reduce(warm, group=group)
        torch.cuda.synchronize()

        for shape in TEST_SHAPES:
            for dtype in TEST_DTYPES:
                torch.manual_seed(1234 + rank)
                local = torch.randn(shape, dtype=dtype, device=device) + rank

                # Reference: the all-gather path every rank currently uses.
                ref_full = tensor_model_parallel_all_gather(local.clone(), dim=-1)

                # ---- eager gather ----
                gathered = tensor_model_parallel_gather(local.clone(), dst=0, dim=-1)
                if rank == 0:
                    assert gathered is not None
                    torch.testing.assert_close(gathered, ref_full)
                    # token ids from gather must equal token ids from all-gather
                    assert torch.equal(
                        gathered.float().argmax(-1), ref_full.float().argmax(-1)
                    )
                else:
                    assert gathered is None, f"rank {rank} expected None, got a tensor"

                # ---- gather inside a CUDA graph ----
                static_in = local.clone()
                # Warmup capture stream (NCCL P2P must be primed before capture).
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        tensor_model_parallel_gather(static_in, dst=0, dim=-1)
                torch.cuda.current_stream().wait_stream(s)

                g = torch.cuda.CUDAGraph()
                with graph_capture() as gc:
                    with torch.cuda.graph(g, stream=gc.stream):
                        graph_out = tensor_model_parallel_gather(
                            static_in, dst=0, dim=-1
                        )
                # Change the input and replay: rank 0's captured output buffer
                # must reflect the new data.
                new_local = torch.randn(shape, dtype=dtype, device=device) + rank + 5
                static_in.copy_(new_local)
                new_ref = tensor_model_parallel_all_gather(new_local.clone(), dim=-1)
                g.replay()
                torch.cuda.synchronize()
                if rank == 0:
                    assert graph_out is not None
                    torch.testing.assert_close(graph_out, new_ref)
                else:
                    assert graph_out is None


if __name__ == "__main__":
    unittest.main()
