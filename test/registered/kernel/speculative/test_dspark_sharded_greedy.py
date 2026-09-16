"""Four-rank selection equivalence, including graph replay and padded shards."""

import atexit
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.speculative.dspark.sharded_greedy import sharded_greedy_step
from sglang.srt.distributed.device_communicators.vocab_gather import (
    NVLinkVocabGather,
    make_vocab_gather,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@cache_once
def group():
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(list(range(world)), rank, backend="nccl")
    atexit.register(dist.destroy_process_group)
    torch.cuda.set_stream(torch.cuda.Stream())
    return GroupCoordinator(
        group_ranks=[list(range(world))],
        local_rank=rank,
        torch_distributed_backend="nccl",
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=True,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="sharded_greedy_test",
    )


_NVLINK = pytest.mark.parametrize("nvlink", [False, True])
_ROWS = pytest.mark.parametrize("m", [1, 4, 64])
_SHARDS = pytest.mark.parametrize("width,last", [(32320, 32320), (8192, 17), (8, 0)])


def _replay_and_check(m, width, last, nvlink, perturb):
    """Capture the 4-rank selection graph, then replay it under `perturb`."""
    g = group()
    transport = make_vocab_gather(
        g, local_width=width, prefer_nvlink=nvlink, symm_rows=0
    )
    if nvlink and not isinstance(transport, NVLinkVocabGather):
        pytest.skip("this TP group has no multicast plane")
    rank = g.rank_in_group
    real = last if rank == g.world_size - 1 else width
    # A slice of a block's logits has a non-contiguous row stride.
    storage = torch.randn(m, 5, width, device="cuda")
    base = storage[:, 2]
    bias = torch.randn(m, real, device="cuda", dtype=torch.bfloat16)

    def candidate():
        return sharded_greedy_step(
            bias,
            base,
            group=g,
            vocab_start=rank * width,
            gather=transport.gather_stacked,
        )

    for _ in range(3):
        candidate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = candidate()
    for replay in range(4):
        storage.normal_()
        bias.normal_()
        perturb(replay, storage, base, bias, real)
        graph.replay()
        # Independent reference: gather complete, correctly padded FP32 logits.
        local = torch.full((m, width), -float("inf"), device="cuda")
        local[:, :real] = base[:, :real] + bias.float()
        full = g.all_gather(local, dim=-1)
        ref = full[:, : (g.world_size - 1) * width + last].argmax(-1)
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


def _random(replay, storage, base, bias, real):
    pass


def _tie(replay, storage, base, bias, real):
    storage.zero_()
    bias.zero_()


def _nan(replay, storage, base, bias, real):
    if real:
        base[:, replay % real] = float("nan")


def _inf(replay, storage, base, bias, real):
    storage.fill_(-float("inf"))
    if replay % 2 and real:
        base[:, replay % real] = float("inf")


@_NVLINK
@_ROWS
@_SHARDS
def test_random_logits(m, width, last, nvlink):
    _replay_and_check(m, width, last, nvlink, _random)


@_NVLINK
@_ROWS
@_SHARDS
def test_ties_resolve_to_the_lowest_index(m, width, last, nvlink):
    _replay_and_check(m, width, last, nvlink, _tie)


@_NVLINK
@_ROWS
@_SHARDS
def test_nan_propagates(m, width, last, nvlink):
    _replay_and_check(m, width, last, nvlink, _nan)


@_NVLINK
@_ROWS
@_SHARDS
def test_infinities(m, width, last, nvlink):
    _replay_and_check(m, width, last, nvlink, _inf)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
