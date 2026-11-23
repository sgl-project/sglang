"""Muse Spark TP projections: column-slice GEMM + multicast all-gather, row split.

``gemm_ag`` must equal every rank's column-slice GEMM concatenated (the gather only
moves bytes), eager and over CUDA graph replays, and be as accurate as the full GEMM.
``row_split`` must equal every rank's row-block GEMM stacked, padded tail dropped.

Usage::

    python test/registered/kernels/ops/communication/test_muse_spark_gemm_ag.py
"""

from __future__ import annotations

import atexit
import logging
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication.muse_spark_all_gather import gemm_ag_supported
from sglang.srt.distributed.device_communicators.symm_workspace import (
    WorkspaceRequest,
    request_symm_workspace,
)
from sglang.srt.layers.communication.muse_spark_gemm_ag import gemm_ag
from sglang.srt.layers.muse_spark.fused.glue import row_split
from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=120, stage="extra-b", runner_config="8-gpu-h200")

K = 1024
SLOT_BYTES = 512 * 4096 * 2  # the largest gathered block below
# (M, N): small-projection bound, then the SM90 TP4 / TP8 shapes up to MAX_TOKENS
CASES = [(1, 2048), (8, 8192), (9, 4096), (64, 2048), (256, 8192), (512, 4096)]


@cache_once
def _init_world() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(int(os.environ["WORLD_SIZE"]))),
        local_rank=local_rank,
        backend="nccl",
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    return coord.cpu_group


@cache_once
def _nccl_group() -> dist.ProcessGroup:
    _init_world()
    return dist.new_group(backend="nccl", device_id=_device())


@cache_once
def _lease():
    request = WorkspaceRequest(owner="muse_spark_gemm_ag", push_slot_bytes=SLOT_BYTES)
    lease = request_symm_workspace(_init_world(), _device(), request)
    return lease if lease is not None and lease.has_multicast else None


def _device() -> torch.device:
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


def _randn(*shape: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    # same seed on every rank: the projection input and weight are replicated
    g = torch.Generator(device=_device()).manual_seed(seed)
    x = torch.randn(shape, generator=g, device=_device()) * scale
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("m,n", CASES)
@torch.inference_mode()
def test_gemm_ag(m: int, n: int, use_graph: bool) -> None:
    lease = _lease()
    if lease is None:
        pytest.skip("gemm_ag needs a multicast symmetric workspace")
    world = lease.world_size
    x, w = _randn(m, K, seed=1), _randn(n, K, seed=2, scale=K**-0.5)
    if not gemm_ag_supported(lease, x, w):
        on_sm90 = torch.cuda.get_device_capability() == (9, 0)
        assert not (on_sm90 and world in (4, 8)), "SM90 TP4 / TP8 take this shape"
        pytest.skip("shape not routed to gemm_ag on this GPU / world size")
    n_local = n // world

    def project() -> tuple[torch.Tensor, torch.Tensor]:
        # every rank's slice through the same GEMM dispatcher, concatenated
        slices = [w[r * n_local : (r + 1) * n_local] for r in range(world)]
        ref = torch.cat([bf16_gemm_dispatch(x, s, None) for s in slices], dim=1)
        return gemm_ag(lease, x, w), ref

    run = project
    if use_graph:
        project()  # allocate the workspace, compile and pick GEMMs before capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            outs = project()

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            graph.replay()
            return outs

    for it in range(4):  # replays reuse the push slots across phases
        x.copy_(_randn(m, K, seed=10 + it))
        out, ref = run()
        # bitwise up to the sign of zero (the Lamport marker)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)
    # the full GEMM is the stock projection: only its tiling differs
    exact = x.double() @ w.double().T
    err = lambda y: (y.double() - exact).abs().mean().item()  # noqa: E731
    assert err(out) <= 1.1 * err(bf16_gemm_dispatch(x, w, None))


@pytest.mark.parametrize("num_tokens", [1024, 1024 + 3, 4099])
@torch.inference_mode()
def test_row_split(num_tokens: int) -> None:
    _init_world()
    world, rank = dist.get_world_size(), dist.get_rank()
    group = SimpleNamespace(
        world_size=world,
        rank_in_group=rank,
        all_gather_into_tensor=lambda out, inp: dist.all_gather_into_tensor(
            out, inp, group=_nccl_group()
        ),
    )
    x, w = _randn(num_tokens, K, seed=3), _randn(2048, K, seed=4, scale=K**-0.5)
    rows = -(-num_tokens // world)
    out = row_split(group, x, lambda xr: bf16_gemm_dispatch(xr, w, None))
    blocks = [bf16_gemm_dispatch(x[r * rows :][:rows], w, None) for r in range(world)]
    torch.testing.assert_close(out, torch.cat(blocks), atol=0, rtol=0)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4, 8))
