"""Four-rank selection equivalence, including graph replay and padded shards."""

import atexit
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.speculative.dspark.sharded_greedy import sharded_greedy_step
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-gb300")


@cache_once
def group():
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(list(range(world)), rank, backend="nccl")
    atexit.register(dist.destroy_process_group)
    torch.cuda.set_stream(torch.cuda.Stream())
    return ps._WORLD


@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("width,last", [(32320, 32320), (8192, 17), (8, 0)])
@pytest.mark.parametrize("case", ["random", "tie", "nan", "inf"])
def test_sharded_selection_graph(m, width, last, case):
    g = group()
    rank = g.rank_in_group
    real = last if rank == g.world_size - 1 else width
    # A slice of a block's logits has a non-contiguous row stride.
    storage = torch.randn(m, 5, width, device="cuda")
    base = storage[:, 2]
    bias = torch.randn(m, real, device="cuda", dtype=torch.bfloat16)

    def candidate():
        return sharded_greedy_step(bias, base, group=g, vocab_start=rank * width)

    for _ in range(3):
        candidate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = candidate()
    for replay in range(4):
        storage.normal_()
        bias.normal_()
        if case == "tie":
            storage.zero_()
            bias.zero_()
        if case == "nan" and real:
            base[:, replay % real] = float("nan")
        if case == "inf":
            storage.fill_(-float("inf"))
            if replay % 2 and real:
                base[:, replay % real] = float("inf")
        graph.replay()
        # Independent reference: gather complete, correctly padded FP32 logits.
        local = torch.full((m, width), -float("inf"), device="cuda")
        local[:, :real] = base[:, :real] + bias.float()
        full = g.all_gather(local, dim=-1)
        ref = full[:, : (g.world_size - 1) * width + last].argmax(-1)
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
