import inspect
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.kernels.ops.kvcache.shared_kv_publication import (
    compile_shared_kv_publication,
    shared_kv_publish,
)
from sglang.srt.mem_cache.shared_kv.synchronization import SharedWritePublisher
from sglang.srt.mem_cache.shared_kv.vmm import create_rank_major_shared_tensor
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-c", runner_config="4-gpu-b200")

PORT = 29827
DIRECT_PORT = 29828


def _init_distributed(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import get_parallel

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        attention_context_model_parallel_size=world_size,
    )
    return get_parallel().attn_cp_group


def _destroy_distributed() -> None:
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()


def _run_direct_ptx_publication(rank: int, world_size: int, port: int) -> None:
    cp_group = _init_distributed(rank, world_size, port)
    flags = create_rank_major_shared_tensor(
        (world_size,), dtype=torch.int32, cpu_group=cp_group.cpu_group
    )
    flags.local_view.zero_()
    torch.cuda.synchronize()
    dist.barrier(group=cp_group.cpu_group)
    peer_ptrs = torch.tensor(
        [
            flags.global_view.data_ptr() + peer * flags.aligned_bytes_per_rank
            for peer in range(world_size)
        ],
        dtype=torch.int64,
        device=f"cuda:{rank}",
    )
    epoch = torch.zeros((1,), dtype=torch.int32, device=f"cuda:{rank}")
    if rank == 0:
        torch.cuda._sleep(10_000_000)

    shared_kv_publish(flags.global_view, peer_ptrs, epoch, rank, world_size)
    torch.cuda.synchronize()

    assert flags.local_view[:world_size].tolist() == [1] * world_size
    dist.barrier(group=cp_group.cpu_group)
    flags.close()
    _destroy_distributed()


def _run_delayed_device_publication(rank: int, world_size: int, port: int) -> None:
    cp_group = _init_distributed(rank, world_size, port)
    payload = create_rank_major_shared_tensor(
        (1,), dtype=torch.int32, cpu_group=cp_group.cpu_group
    )
    payload.local_view.zero_()
    torch.cuda.synchronize()
    dist.barrier(group=cp_group.cpu_group)
    publisher = SharedWritePublisher(cp_group)
    if rank == 0:
        torch.cuda._sleep(10_000_000)
        payload.local_view[0] = 73
    publisher.publish()
    torch.cuda.synchronize()
    assert payload.global_view[0].item() == 73
    dist.barrier(group=cp_group.cpu_group)
    actual_flags = publisher._flags.local_view[:world_size].tolist()
    assert actual_flags == [1] * world_size, (rank, actual_flags)
    dist.barrier(group=cp_group.cpu_group)

    for _ in range(16):
        publisher.publish()
    torch.cuda.synchronize()
    dist.barrier(group=cp_group.cpu_group)
    assert publisher._flags.local_view[:world_size].tolist() == [17] * world_size
    dist.barrier(group=cp_group.cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        publisher.publish()
    start_epoch = publisher._epoch.item()
    for _ in range(17):
        graph.replay()
    torch.cuda.synchronize()
    assert publisher._epoch.item() == start_epoch + 17
    assert (
        publisher._flags.local_view[:world_size].tolist()
        == [start_epoch + 17] * world_size
    )
    dist.barrier(group=cp_group.cpu_group)

    publisher.close()
    publisher.close()
    payload.close()
    _destroy_distributed()


class TestSharedWritePublisher(CustomTestCase):
    def test_publication_jit_compiles(self):
        if not torch.cuda.is_available():
            self.skipTest("publication JIT compile test needs CUDA")
        compile_shared_kv_publication(2)

    def test_direct_ptx_publication_waits_for_every_rank(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("device publication test needs at least two GPUs")
        mp.spawn(
            _run_direct_ptx_publication,
            args=(2, DIRECT_PORT),
            nprocs=2,
            join=True,
        )

    def test_constructor_has_no_runtime_disable_or_strategy_switch(self):
        parameters = inspect.signature(SharedWritePublisher).parameters

        self.assertEqual(list(parameters), ["attention_cp_group"])
        self.assertFalse(
            {"enabled", "async_op", "strategy", "signal"}.intersection(parameters)
        )

    def test_delayed_writer_and_graph_replays_are_published(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("device publication test needs at least two GPUs")
        mp.spawn(
            _run_delayed_device_publication,
            args=(2, PORT),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
