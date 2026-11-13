"""Correctness test for the PCIe-IPC vocab gather (FlashInfer PCIe-IPC all-gather).

The gather is a pure copy, so it must be bit-identical to the NCCL gather it
replaces, in both the side-by-side (``__call__``) and rank-major
(``gather_stacked``) layouts, eagerly and under CUDA-graph replay, and slices
the workspace cannot take must come back from the NCCL fallback unchanged.

Usage::

    python test/registered/kernels/ops/communication/test_pcie_ipc_vocab_gather.py
    python test/registered/kernels/ops/communication/test_pcie_ipc_vocab_gather.py --num-gpu 4
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.srt.distributed.device_communicators.vocab_gather import (
    NcclVocabGather,
    PcieIpcVocabGather,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

MAX_ROWS = 16
# DeepSeek-V4's per-rank vocab shard at TP4, and a narrow one.
WIDTHS = [1032, 32320]
DTYPES = [torch.float32, torch.bfloat16]


@cache_once
def _world():
    flashinfer_comm = pytest.importorskip("flashinfer.comm")
    if not hasattr(flashinfer_comm, "PcieIpcAllGatherWorkspace"):
        pytest.skip("this FlashInfer has no PCIe-IPC all-gather")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    get_parallel().override_permanently(world_group=ps._WORLD)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    return ps._WORLD


@cache_once
def _gather(width: int, dtype: torch.dtype) -> PcieIpcVocabGather:
    group = _world()
    return PcieIpcVocabGather(
        group=group,
        local_width=width,
        dtype=dtype,
        max_rows=MAX_ROWS,
        fallback=NcclVocabGather(group),
    )


def _local(rows: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    # Distinct per rank and per call, including NaN/inf/-0.0 bit patterns
    # that a numeric (non-copy) path would not preserve.
    x = torch.randn(rows, width, device="cuda") * (dist.get_rank() + 1)
    x[:, 0] = float("nan")
    x[:, -1] = -0.0
    if width > 2:
        x[:, 1] = float("inf")
    return x.to(dtype)


def _nccl_stacked(x: torch.Tensor) -> torch.Tensor:
    """Reference rank-major gather, independent of the code under test."""
    out = torch.empty(
        (dist.get_world_size() * x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(out, x, group=_world().device_group)
    return out


def _nccl(x: torch.Tensor) -> torch.Tensor:
    """Reference side-by-side gather, as ``all_gather(dim=-1)`` returns it."""
    rows, width = x.shape
    return (
        _nccl_stacked(x)
        .view(dist.get_world_size(), rows, width)
        .movedim(0, 1)
        .reshape(rows, -1)
    )


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("rows", [1, 2, 3, 7, MAX_ROWS])
def test_matches_nccl(rows: int, width: int, dtype: torch.dtype) -> None:
    pcie_ipc = _gather(width, dtype)
    x = _local(rows, width, dtype)
    _assert_bitwise_equal(pcie_ipc(x), _nccl(x))
    _assert_bitwise_equal(pcie_ipc.gather_stacked(x), _nccl_stacked(x))


@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_graph_replay_matches_nccl(dtype: torch.dtype) -> None:
    """Warm up and capture on a side stream, then replay with new inputs and check
    each replay against NCCL, with an eager call on the default stream between
    replays. The workspace's stream rebind is a host-side call made by the eager
    calls when the current stream changes; it is not captured in the graph and
    replay does not repeat it."""
    width = WIDTHS[-1]
    pcie_ipc = _gather(width, dtype)
    static_in = {r: _local(r, width, dtype) for r in (1, 4)}
    static_out = {}
    graphs = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for rows, x in static_in.items():
            pcie_ipc(x)  # warm up on the capture stream
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                static_out[rows] = (pcie_ipc(x), pcie_ipc.gather_stacked(x))
            graphs[rows] = graph
    torch.cuda.current_stream().wait_stream(stream)

    for _ in range(3):
        for rows, x in static_in.items():
            x.copy_(_local(rows, width, dtype))
            dist.barrier()
            graphs[rows].replay()
            side, stacked = static_out[rows]
            _assert_bitwise_equal(side, _nccl(x))
            _assert_bitwise_equal(stacked, _nccl_stacked(x))
        # an eager call between replays moves the binding back
        x = _local(2, width, dtype)
        _assert_bitwise_equal(pcie_ipc(x), _nccl(x))


@pytest.mark.parametrize("dtype", DTYPES)
def test_earlier_result_survives_the_next_gather(dtype: torch.dtype) -> None:
    """A returned gather is not overwritten by the next call into the workspace."""
    width = WIDTHS[-1]
    pcie_ipc = _gather(width, dtype)
    x1 = _local(1, width, dtype)
    side, stacked = pcie_ipc(x1), pcie_ipc.gather_stacked(x1)
    x2 = _local(1, width, dtype)
    pcie_ipc(x2)
    pcie_ipc.gather_stacked(x2)
    _assert_bitwise_equal(side, _nccl(x1))
    _assert_bitwise_equal(stacked, _nccl_stacked(x1))


@pytest.mark.parametrize("dtype", DTYPES)
def test_strided_and_misaligned_slices_match_nccl(dtype: torch.dtype) -> None:
    width = WIDTHS[-1]
    pcie_ipc = _gather(width, dtype)
    for x in (
        _local(3, 2 * width, dtype)[:, ::2],  # not contiguous
        _local(1, 2 * width + 1, dtype)[0, 1:].view(2, width),  # not 16-byte aligned
    ):
        _assert_bitwise_equal(pcie_ipc(x), _nccl(x.contiguous()))
        _assert_bitwise_equal(pcie_ipc.gather_stacked(x), _nccl_stacked(x.contiguous()))


def test_falls_back_to_nccl_for_slices_the_workspace_cannot_take() -> None:
    width, dtype = WIDTHS[-1], torch.float32
    pcie_ipc = _gather(width, dtype)
    for x in (
        _local(MAX_ROWS + 1, width, dtype),  # past the workspace
        _local(2 * MAX_ROWS, width // 2, dtype),  # fits by elements, past max_rows
        _local(2, width, torch.float16),  # not the workspace dtype
    ):
        _assert_bitwise_equal(pcie_ipc(x), _nccl(x))
        _assert_bitwise_equal(pcie_ipc.gather_stacked(x), _nccl_stacked(x))


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4))
