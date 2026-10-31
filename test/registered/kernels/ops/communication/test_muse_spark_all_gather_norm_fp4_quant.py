"""Muse Spark pooler gather fused with the pre-expert RMSNorm + static NVFP4 quant.

The fused kernel must stay bitwise equal, on every rank, to the push-plane all-gather
followed by fused_norm_fp4_quant_static: when rows outnumber the work blocks, when the
gathered rows fill the push slot, across both push-plane phases with fresh data, and when
replayed from a CUDA graph as served.

Run with ``python <this file>``; it relaunches under torchrun on 4 and 8 GPUs.
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
from sglang.kernels.ops.communication.muse_spark_all_gather import (
    all_gather,
    all_gather_norm_fp4_quant,
)
from sglang.kernels.ops.layernorm.muse_spark_rmsnorm_nvfp4_quant import (
    fused_norm_fp4_quant_static,
)
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=90, stage="extra-b", runner_config="4-gpu-b200")

N = 4096
EPS = 1e-6
PUSH_BYTES = 8 * 1024 * 1024
# The kernel launches min(rows, PUSH_BLOCKS) blocks: 1 row leaves counters past the grid to
# block 0, 64 rows give one row per block, 65 and 200 loop rows per block (both reduction
# buffers), and 1024 rows of N bf16 fill PUSH_BYTES.
PUSH_BLOCKS = 64
ROWS = [1, 64, 65, 200, 1024]
# Each round launches three push-plane kernels, so every launch alternates the phase half;
# four rounds of fresh data put each kernel on both halves with new payloads.
ROUNDS = 4


@cache_once
def _init():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    comm = CustomAllReduceV2(
        coord.cpu_group,
        torch.device(f"cuda:{local_rank}"),
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=PUSH_BYTES,
        max_push_blocks=PUSH_BLOCKS,
    )
    assert not comm.disabled
    register_comm_cleanup(comm)
    return coord, comm


def _assert_all_ranks(failures: list[str], context: str) -> None:
    """Raise on every rank if any rank failed, so no rank is left waiting in a collective."""
    per_rank = [None] * dist.get_world_size()
    dist.all_gather_object(per_rank, failures)
    report = [f"rank {r}: {f}" for r, fs in enumerate(per_rank) for f in fs]
    assert not report, f"{context}: " + "; ".join(report)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


def _local_rows(coord, rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed * 64 + coord.rank)
    local = torch.randn(rows, N // coord.world_size, device="cuda", generator=g) * 2
    local[:, ::97] = 0  # exact zeros travel through the push plane's +0 marker remap
    return local.bfloat16()


def _run_step(comm, local, weight, global_scale):
    """The fused op with and without the gathered rows, then the unfused all-gather."""
    rows, fp4, sf = all_gather_norm_fp4_quant(
        comm, local, weight, EPS, global_scale, write_rows=True
    )
    _, fp4_no_rows, sf_no_rows = all_gather_norm_fp4_quant(
        comm, local, weight, EPS, global_scale, write_rows=False
    )
    ref_rows = torch.empty(local.shape[0], N, dtype=torch.bfloat16, device="cuda")
    all_gather(comm, local, ref_rows)
    return rows, fp4, sf, fp4_no_rows, sf_no_rows, ref_rows


@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("rows", ROWS)
def test_matches_all_gather_then_norm_quant(rows: int, use_graph: bool):
    coord, comm = _init()
    if not comm.has_multicast:
        pytest.skip("needs a multicast-capable push plane")
    shared = torch.Generator(device="cuda").manual_seed(rows)
    weight = (1 + 0.1 * torch.randn(N, device="cuda", generator=shared)).bfloat16()
    global_scale = torch.tensor([448.0 * 6.0 / 3.5], device="cuda")
    local = _local_rows(coord, rows, seed=0)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        outs = _run_step(comm, local, weight, global_scale)
        if use_graph:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with comm.capture():
                with torch.cuda.graph(graph, stream=side):
                    outs = _run_step(comm, local, weight, global_scale)
    torch.cuda.synchronize()

    for rnd in range(ROUNDS):
        local.copy_(_local_rows(coord, rows, seed=rnd + 1))
        if use_graph:
            graph.replay()
        else:
            outs = _run_step(comm, local, weight, global_scale)
        torch.cuda.synchronize()
        got_rows, fp4, sf, fp4_no_rows, sf_no_rows, ref_rows = outs
        ref_fp4, ref_sf = fused_norm_fp4_quant_static(
            ref_rows, weight, EPS, global_scale
        )
        torch.cuda.synchronize()
        checks = (
            ("rows", got_rows, ref_rows),
            ("fp4", fp4, ref_fp4),
            ("sf", sf, ref_sf),
            ("fp4 (no rows)", fp4_no_rows, ref_fp4),
            ("sf (no rows)", sf_no_rows, ref_sf),
        )
        failures = [
            f"{name}: {int((_bits(a) != _bits(b)).sum())} bytes differ"
            for name, a, b in checks
            if not torch.equal(_bits(a), _bits(b))
        ]
        _assert_all_ranks(failures, f"round {rnd}")
        # Every rank quantizes the same gathered rows, so the outputs must agree across ranks.
        peers = [torch.empty_like(fp4) for _ in range(coord.world_size)]
        dist.all_gather(peers, fp4, group=coord.device_group)
        _assert_all_ranks(
            [
                f"fp4 differs from rank {r}"
                for r, p in enumerate(peers)
                if not torch.equal(p, fp4)
            ],
            f"round {rnd} cross-rank",
        )


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
