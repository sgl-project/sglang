"""``VocabGather``: the TP vocab-parallel row gather behind the DSpark draft head.

On a TP group built the way the server builds it (custom all-reduce on, so
``ca_comm`` is a CustomAllReduceV2), ``make_vocab_gather`` must pick the NVLink
gather, and every path of that gather (push kernel, pull kernel into the
symmetric-memory output, NCCL past its capacity) must match the NCCL
``all_gather(dim=-1)`` reference, hand back a result that does not alias the
shared output, and replay correctly inside a CUDA graph.

Usage::

    python test/registered/kernels/ops/communication/test_vocab_gather.py --num-gpu 4
"""

from __future__ import annotations

import atexit
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.srt.distributed.device_communicators.vocab_gather import (
    NcclVocabGather,
    NVLinkVocabGather,
    make_vocab_gather,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

LOCAL_WIDTH = 32320  # DeepSeek-V4.1's 129280-entry vocab over TP4
SYMM_ROWS = 256


def _device() -> torch.device:
    return torch.device("cuda", int(os.environ["LOCAL_RANK"]))


@cache_once
def _tp_group() -> GroupCoordinator:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    atexit.register(dist.destroy_process_group)
    return GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=local_rank,
        torch_distributed_backend="nccl",
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=True,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="vocab_gather_test",
    )


@cache_once
def _gathers():
    tp = _tp_group()
    nvlink = make_vocab_gather(tp, local_width=LOCAL_WIDTH, symm_rows=SYMM_ROWS)
    if not isinstance(nvlink, NVLinkVocabGather):
        pytest.skip("this TP group has no multicast plane")
    nccl = make_vocab_gather(tp, local_width=LOCAL_WIDTH, prefer_nvlink=False)
    return nvlink, nccl


def _rows(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=_device()).manual_seed(seed + dist.get_rank())
    x = torch.randn(rows, LOCAL_WIDTH, device=_device(), generator=gen)
    x[:, ::97] = 0.0  # exact zeros: the push kernel's Lamport sentinel path
    return x


def _sync() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def _check(nvlink: NVLinkVocabGather, nccl: NcclVocabGather, x: torch.Tensor) -> None:
    ref = nccl(x)
    got = nvlink(x)
    _sync()
    assert got.shape == ref.shape
    assert bool((got == ref).all())  # -0.0 == 0.0, so the sentinel flip is invisible
    if nvlink.pull_out is not None:
        lo = nvlink.pull_out.data_ptr()
        hi = lo + nvlink.pull_out.numel() * nvlink.pull_out.element_size()
        assert not (lo <= got.data_ptr() < hi), "result aliases the shared pull output"


@pytest.mark.parametrize("rows", [1, 3, 6, 25, 64, 200, SYMM_ROWS + 1])
def test_matches_nccl(rows: int) -> None:
    nvlink, nccl = _gathers()
    _check(nvlink, nccl, _rows(rows, seed=rows))


def test_consecutive_pull_results_survive() -> None:
    nvlink, nccl = _gathers()
    a, b = _rows(64, seed=1), _rows(64, seed=2)
    ra, rb = nvlink(a), nvlink(b)
    _sync()
    assert bool((ra == nccl(a)).all()) and bool((rb == nccl(b)).all())


@pytest.mark.parametrize("rows", [1, 64])
def test_cuda_graph_replay(rows: int) -> None:
    nvlink, nccl = _gathers()
    x = _rows(rows, seed=100 + rows)
    nvlink(x)  # eager once: JIT load
    _sync()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _sync()
        with torch.cuda.graph(graph, stream=stream):
            out = nvlink(x)
    _sync()
    for it in range(3):
        x.copy_(_rows(rows, seed=200 + it))
        ref = nccl(x)
        _sync()
        graph.replay()
        _sync()
        assert bool((out == ref).all())


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
