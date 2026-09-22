"""Broadcast chains must preserve rank agreement across changed-input replays."""

import os
from contextlib import nullcontext

import pytest
import torch

from sglang.srt.distributed import parallel_state as ps
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.fixture(scope="module")
def world():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world_size, world_rank=rank)
    yield world_size, local_rank
    ps.destroy_distributed_environment()


@pytest.fixture(scope="module", params=["full", "pairs"])
def group(world, request):
    world_size, local_rank = world
    group_size = world_size if request.param == "full" else 2
    groups = [
        list(range(start, start + group_size))
        for start in range(0, world_size, group_size)
    ]
    coordinator = ps.init_model_parallel_group(
        groups,
        local_rank,
        "nccl",
        use_pynccl=True,
        use_custom_allreduce=False,
        group_name="broadcast_test",
    )
    assert coordinator.pynccl_comm.available
    yield coordinator
    coordinator.destroy()


@pytest.mark.parametrize("dtype", [torch.int64, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 16, 129])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("communicator", ["enabled", "disabled", "missing"])
def test_broadcast_chain(group, dtype, num_tokens, use_graph, communicator):
    input_ = torch.full((num_tokens,), group.rank_in_group, dtype=dtype, device="cuda")

    def forward():
        values = input_.clone()
        outputs = []
        for step in range(7):
            values = values + group.rank_in_group + step
            assert group.broadcast(values, src=step % group.world_size) is values
            outputs.append(values.clone())
        return outputs

    comm = group.pynccl_comm
    if communicator == "missing":
        group.pynccl_comm = None
    try:
        context = group.graph_capture() if use_graph else nullcontext()
        with context as capture:
            state = (
                comm.change_state(enable=communicator == "enabled")
                if communicator != "missing"
                else nullcontext()
            )
            with state:
                for _ in range(3):
                    forward()
                if use_graph:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture.stream):
                        outputs = forward()
                else:
                    outputs = None
                torch.cuda.synchronize()
                for base in (1, 17, 33):
                    input_.fill_(base + group.rank_in_group)
                    if use_graph:
                        graph.replay()
                    else:
                        outputs = forward()
                    expected = base
                    for step, output in enumerate(outputs):
                        expected += step % group.world_size + step
                        torch.testing.assert_close(
                            output, torch.full_like(output, expected), atol=0, rtol=0
                        )
    finally:
        group.pynccl_comm = comm


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4))
