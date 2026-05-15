import multiprocessing
import os
import unittest

import pytest
import torch

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    update_environment_variables,
)
from sglang.srt.distributed.device_communicators.pyhccl import PyHcclCommunicator
from sglang.srt.distributed.parallel_state import (
    get_world_group,
    init_distributed_environment,
)
from sglang.test.test_utils import CustomTestCase


def distributed_run(fn, world_size):
    number_of_processes = world_size
    processes: list[multiprocessing.Process] = []
    for i in range(number_of_processes):
        env: dict[str, str] = {}
        env["RANK"] = str(i)
        env["LOCAL_RANK"] = str(i)
        env["WORLD_SIZE"] = str(number_of_processes)
        env["LOCAL_WORLD_SIZE"] = str(number_of_processes)
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = "12345"
        p = multiprocessing.Process(target=fn, args=(env,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def worker_fn_wrapper(fn):
    # `multiprocessing.Process` cannot accept environment variables directly
    # so we need to pass the environment variables as arguments
    # and update the environment variables in the function

    def wrapped_fn(env):
        update_environment_variables(env)
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = os.environ["LOCAL_RANK"]
        ip = env["MASTER_ADDR"]
        port = env["MASTER_PORT"]

        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://{ip}:{port}",
            backend="hccl",
        )
        fn()

    return wrapped_fn


@worker_fn_wrapper
def worker_fn():
    pyhccl_comm = PyHcclCommunicator(
        get_world_group().cpu_group, device=get_world_group().device
    )
    tensor = torch.ones(16, 1024, 1024, dtype=torch.float32).npu(pyhccl_comm.rank)
    tensor = pyhccl_comm.all_reduce(tensor)
    torch.npu.synchronize()

    assert torch.all(tensor == pyhccl_comm.world_size).cpu().item()


@worker_fn_wrapper
def multiple_allreduce_worker_fn():
    device = torch.device(f"npu:{torch.distributed.get_rank()}")
    groups = [
        torch.distributed.new_group(ranks=[0, 1], backend="gloo"),
        torch.distributed.new_group(ranks=[2, 3], backend="gloo"),
    ]
    group = groups[0] if torch.distributed.get_rank() in [0, 1] else groups[1]
    pyhccl_comm = PyHcclCommunicator(group=group, device=device)
    tensor = torch.ones(16, 1024, 1024, dtype=torch.float32, device=device)
    # two groups can communicate independently
    if torch.distributed.get_rank() in [0, 1]:
        tensor = pyhccl_comm.all_reduce(tensor)
        tensor = pyhccl_comm.all_reduce(tensor)
        torch.npu.synchronize()
        assert torch.all(tensor == 4).cpu().item()
    else:
        tensor = pyhccl_comm.all_reduce(tensor)
        torch.npu.synchronize()
        assert torch.all(tensor == 2).cpu().item()


@worker_fn_wrapper
def broadcast_worker_fn():
    # Test broadcast for every root rank.
    # Essentially this is an all-gather operation.
    pyhccl_comm = PyHcclCommunicator(
        get_world_group().cpu_group, device=get_world_group().device
    )
    recv_tensors = [
        torch.empty(16, 1024, 1024, dtype=torch.float32, device=pyhccl_comm.device)
        for i in range(pyhccl_comm.world_size)
    ]
    recv_tensors[pyhccl_comm.rank] = (
        torch.ones(16, 1024, 1024, dtype=torch.float32, device=pyhccl_comm.device)
        * pyhccl_comm.rank
    )

    for i in range(pyhccl_comm.world_size):
        pyhccl_comm.broadcast(recv_tensors[i], src=i)
        # the broadcast op might be launched in a different stream
        # need to synchronize to make sure the tensor is ready
        torch.npu.synchronize()
        assert torch.all(recv_tensors[i] == i).cpu().item()


@worker_fn_wrapper
def all_gather_worker_fn():
    pyhccl_comm = PyHcclCommunicator(
        get_world_group().cpu_group, device=get_world_group().device
    )

    rank = pyhccl_comm.rank
    world_size = pyhccl_comm.world_size
    device = f"npu:{pyhccl_comm.rank}"

    num_elems = 1000
    tensor = (
        torch.arange(num_elems, dtype=torch.float32, device=device) + rank * num_elems
    )
    result = torch.zeros(num_elems * world_size, dtype=torch.float32, device=device)

    expected = torch.cat(
        [
            torch.arange(num_elems, dtype=torch.float32) + r * num_elems
            for r in range(world_size)
        ]
    ).to(device)

    pyhccl_comm.all_gather(result, tensor)
    torch.npu.synchronize()
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)


@worker_fn_wrapper
def reduce_scatter_worker_fn():
    pyhccl_comm = PyHcclCommunicator(
        get_world_group().cpu_group, device=get_world_group().device
    )

    rank = pyhccl_comm.rank
    world_size = pyhccl_comm.world_size
    device = f"npu:{pyhccl_comm.rank}"

    num_elems = 1000
    tensor = (
        torch.arange(num_elems, dtype=torch.float32, device=device) + rank * num_elems
    )
    assert num_elems % world_size == 0
    result = torch.zeros(num_elems // world_size, dtype=torch.float32, device=device)

    # Calculate expected result for this rank's chunk
    scattered_size = num_elems // world_size
    all_tensors = [
        torch.arange(num_elems, dtype=torch.float32) + r * num_elems
        for r in range(world_size)
    ]
    expected = sum(
        tensor[rank * scattered_size : (rank + 1) * scattered_size]
        for tensor in all_tensors
    ).to(device)

    pyhccl_comm.reduce_scatter(tensor, result)

    torch.npu.synchronize()
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)


@worker_fn_wrapper
def sub_comm_worker_fn():
    pyhccl_comm = PyHcclCommunicator(
        get_world_group().cpu_group, device=get_world_group().device
    )

    device = f"npu:{pyhccl_comm.rank}"
    tensor = torch.ones(16, 1024, 1024, dtype=torch.float32, device=device)
    rank_list = [0, 1]
    sub_comm = pyhccl_comm.create_subcomm(rank_list)

    tensor = pyhccl_comm.all_reduce(in_tensor=tensor, comm=sub_comm)
    torch.npu.synchronize()
    if torch.distributed.get_rank() in rank_list:
        assert torch.all(tensor == len(rank_list)).cpu().item()
    else:
        assert torch.all(tensor == 1).cpu().item()


class TestAscendPyhccl(CustomTestCase):
    @pytest.mark.skipif(
        torch.npu.device_count() < 2, reason="Need at least 2 NPUs to run the test."
    )
    def test_pyhccl(self):
        distributed_run(worker_fn, 2)

    @pytest.mark.skipif(
        torch.npu.device_count() < 4, reason="Need at least 4 NPUs to run the test."
    )
    def test_pyhccl_multiple_allreduce(self):
        distributed_run(multiple_allreduce_worker_fn, 4)

    @pytest.mark.skipif(
        torch.npu.device_count() < 4, reason="Need at least 4 NPUs to run the test."
    )
    def test_pyhccl_broadcast(self):
        distributed_run(broadcast_worker_fn, 4)

    @pytest.mark.skipif(
        torch.npu.device_count() < 2, reason="Need at least 2 NPUs to run the test."
    )
    def test_pyhccl_all_gather(self):
        distributed_run(all_gather_worker_fn, 2)

    @pytest.mark.skipif(
        torch.npu.device_count() < 2, reason="Need at least 2 NPUs to run the test."
    )
    def test_pyhccl_reduce_scatter(self):
        distributed_run(reduce_scatter_worker_fn, 2)

    @pytest.mark.skipif(
        torch.npu.device_count() < 2, reason="Need at least 2 NPUs to run the test."
    )
    def test_pyhccl_sub_comm(self):
        distributed_run(sub_comm_worker_fn, 2)


if __name__ == "__main__":
    unittest.main()
