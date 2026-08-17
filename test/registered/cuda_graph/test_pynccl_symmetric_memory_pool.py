"""Regression test for rank-divergent symmetric-memory CUDA graph capture."""

from __future__ import annotations

import os
import subprocess

import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="2-gpu-large")


def _worker_main() -> None:
    os.environ["SGLANG_SYMM_MEM_PREALLOC_GB_SIZE"] = "1"

    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        defer_symmetric_memory_graph_registration,
        get_nccl_mem_pool,
        set_graph_pool_id,
        set_use_dedicated_symmetric_memory_graph_pool,
        use_symmetric_memory,
    )
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from sglang.srt.runtime_context import get_context

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    config_override = get_context().override_server_args(enable_symm_mem=True)
    config_override.install()
    set_custom_all_reduce(False)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        enable_symm_mem=True,
    )
    tp_group = get_tp_group()

    # Establish one registered 16 MiB segment, then leave a 1 MiB allocation
    # live only on rank 0. The next eager-pool allocation consequently has a
    # different offset on rank 0, matching the production failure's allocator
    # precondition without loading a model.
    with use_symmetric_memory(tp_group):
        segment_anchor = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    with use_symmetric_memory(tp_group):
        rank_zero_blocker = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    if rank != 0:
        del rank_zero_blocker

    # Warm pynccl before capture with a rank-symmetric eager allocation.
    with use_symmetric_memory(tp_group):
        warmup = torch.full((256,), rank + 1, dtype=torch.float32, device="cuda")
    del warmup
    torch.cuda.synchronize()
    dist.barrier(group=tp_group.cpu_group)

    graph_pool = torch.cuda.graph_pool_handle()
    capture_stream = torch.cuda.Stream()
    static_input = torch.full(
        (64 * 1024 * 1024,), rank + 1, dtype=torch.uint8, device="cuda"
    )

    graphs = []
    all_graph_outputs = []
    with tp_group.graph_capture(stream=capture_stream):
        set_use_dedicated_symmetric_memory_graph_pool(True)
        set_graph_pool_id(graph_pool)
        for _ in range(3):
            with defer_symmetric_memory_graph_registration(tp_group):
                prime_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    prime_graph, pool=graph_pool, stream=capture_stream
                ):
                    with use_symmetric_memory(tp_group):
                        prime_outputs = [
                            torch.empty_like(static_input) for _ in range(4)
                        ]
                    for prime_output in prime_outputs:
                        prime_output.copy_(static_input)
                        tp_group.all_reduce(prime_output)
            del prime_output, prime_outputs
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=graph_pool, stream=capture_stream):
                with use_symmetric_memory(tp_group):
                    graph_outputs = [torch.empty_like(static_input) for _ in range(4)]
                for graph_output in graph_outputs:
                    graph_output.copy_(static_input)
                    tp_group.all_reduce(graph_output)
            del prime_graph
            graphs.append(graph)
            all_graph_outputs.append(graph_outputs)

    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()

    with use_symmetric_memory(tp_group):
        eager_output = torch.full((1024,), rank + 1, dtype=torch.float32, device="cuda")
    tp_group.all_reduce(eager_output)
    torch.cuda.synchronize()

    data_ptr = all_graph_outputs[0][0].untyped_storage().data_ptr()
    relative_offset = None
    for segment in get_nccl_mem_pool(graph_capture=True).snapshot():
        segment_base = segment["address"]
        if segment_base <= data_ptr < segment_base + segment["total_size"]:
            relative_offset = data_ptr - segment_base
            break
    if relative_offset is None:
        raise AssertionError("captured buffer is not in the active symmetric pool")

    offsets = [None] * world_size
    dist.all_gather_object(offsets, relative_offset, group=tp_group.cpu_group)
    if len(set(offsets)) != 1:
        raise AssertionError(f"captured symmetric-memory offsets diverged: {offsets}")
    expected_sum = world_size * (world_size + 1) // 2
    if not all(
        torch.all(graph_output == expected_sum)
        for graph_outputs in all_graph_outputs
        for graph_output in graph_outputs
    ):
        raise AssertionError("captured graph output mismatch")
    if not torch.all(eager_output == expected_sum):
        raise AssertionError("post-capture eager all-reduce output mismatch")

    set_graph_pool_id(None)
    set_use_dedicated_symmetric_memory_graph_pool(False)

    del segment_anchor
    if rank == 0:
        del rank_zero_blocker
    destroy_model_parallel()
    destroy_distributed_environment()
    config_override.restore()


class TestPyNcclSymmetricMemoryPool(CustomTestCase):
    def test_cuda_graph_uses_rank_deterministic_pool(self) -> None:
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        result = subprocess.run(
            ["torchrun", "--standalone", "--nproc-per-node=2", __file__],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _worker_main()
    else:
        import unittest

        unittest.main()
