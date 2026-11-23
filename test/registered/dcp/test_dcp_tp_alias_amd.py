"""ROCm TP8/DCP8: the collectives DCP issues, eager and replayed from a CUDA graph.

On ROCm, full-width DCP reuses the TP coordinator unless TP carries QuickAllReduce.
Each torchrun worker builds TP8/DCP8 through ``initialize_model_parallel``, checks
which coordinator DCP got, and compares the DCP collectives (all-gather,
all-gather-into-tensor, reduce-scatter along a dim, all-to-all, FP32 and BF16
all-reduce) with exact references, once eagerly and once from a replayed graph.

This file doubles as the torchrun worker script.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=90, suite="base-c-test-large-8-gpu-amd-mi35x")

WORLD = 8
ROWS = 4
COLS = 64


def _inputs(rank: int, step: int, device: torch.device) -> dict[str, torch.Tensor]:
    # Small integers keep every sum exact in bf16.
    value = rank + 1 + step
    # Chunk j (sent to rank j) holds rank * 100 + j * 10 + [0, ROWS).
    idx = torch.arange(WORLD * ROWS, device=device, dtype=torch.float32)
    a2a = rank * 100 + idx.div(ROWS, rounding_mode="floor") * 10 + idx.remainder(ROWS)
    return {
        "gather": torch.full((ROWS, COLS), value, dtype=torch.bfloat16, device=device),
        "gather_bytes": torch.full((COLS,), value, dtype=torch.uint8, device=device),
        "scatter": torch.arange(WORLD * ROWS * COLS, device=device, dtype=torch.float32)
        .reshape(WORLD * ROWS, COLS)
        .remainder(7)
        + value,
        "a2a": a2a + step,
        "sum_fp32": torch.full((ROWS, COLS), value, dtype=torch.float32, device=device),
        "sum_bf16": torch.full(
            (ROWS, COLS), value, dtype=torch.bfloat16, device=device
        ),
    }


def _expected(rank: int, step: int, device: torch.device) -> dict[str, torch.Tensor]:
    per_rank = [_inputs(r, step, device) for r in range(WORLD)]
    scatter = sum(p["scatter"] for p in per_rank)
    a2a = torch.cat(
        [per_rank[src]["a2a"].view(WORLD, ROWS)[rank] for src in range(WORLD)]
    )
    return {
        "gather": torch.cat([p["gather"] for p in per_rank], dim=0),
        "gather_bytes": torch.cat([p["gather_bytes"] for p in per_rank]),
        "scatter": scatter[rank * ROWS : (rank + 1) * ROWS],
        "a2a": a2a,
        "sum_fp32": sum(p["sum_fp32"] for p in per_rank),
        "sum_bf16": sum(p["sum_bf16"].float() for p in per_rank).to(torch.bfloat16),
    }


def _run_collectives(dcp, inp: dict[str, torch.Tensor], out: dict[str, torch.Tensor]):
    out["gather"] = dcp.all_gather(inp["gather"], dim=0)
    dcp.all_gather_into_tensor(out["gather_bytes"], inp["gather_bytes"])
    out["scatter"] = dcp.reduce_scatter_along_dim(inp["scatter"], dim=0)
    dcp.all_to_all_single(out["a2a"], inp["a2a"])
    out["sum_fp32"] = dcp.all_reduce(inp["sum_fp32"].clone())
    out["sum_bf16"] = dcp.all_reduce(inp["sum_bf16"].clone())


def _check(mode: str, out: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]):
    for name, ref in expected.items():
        if not torch.equal(out[name], ref):
            raise AssertionError(f"{mode}: DCP {name} mismatch")


def _worker(custom_all_reduce: bool) -> None:
    import sglang.srt.distributed.parallel_state as ps
    from sglang.test.test_utils import publish_build_topology

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    ps.set_custom_all_reduce(custom_all_reduce)
    ps.init_distributed_environment(
        world_size=WORLD, rank=rank, local_rank=local_rank, backend="nccl"
    )
    publish_build_topology(tp_size=WORLD, dcp_size=WORLD, world_rank=rank)
    ps.initialize_model_parallel()
    tp, dcp = ps._TP, ps._DCP

    if tp.qr_comm is None:
        assert dcp is tp, "full-width DCP must reuse the QR-free TP coordinator"
    else:
        assert dcp is not tp, "DCP must not share a TP coordinator with QuickAllReduce"
        assert dcp.qr_comm is None
    probe = torch.ones(ROWS, COLS, dtype=torch.bfloat16, device=device)
    method = dcp._resolve_outplace_all_reduce_method(probe)
    assert method != "qr", "DCP all-reduce must never take QuickAllReduce"

    with torch.inference_mode():
        eager: dict[str, torch.Tensor] = {
            "gather_bytes": torch.empty(WORLD * COLS, dtype=torch.uint8, device=device),
            "a2a": torch.empty(WORLD * ROWS, dtype=torch.float32, device=device),
        }
        _run_collectives(dcp, _inputs(rank, 0, device), eager)
        torch.cuda.synchronize()
        _check("eager", eager, _expected(rank, 0, device))

        static_inp = _inputs(rank, 0, device)
        graph_out: dict[str, torch.Tensor] = {
            "gather_bytes": torch.empty_like(eager["gather_bytes"]),
            "a2a": torch.empty_like(eager["a2a"]),
        }
        graph = torch.cuda.CUDAGraph()
        with ps.graph_capture() as ctx:
            _run_collectives(
                dcp, static_inp, graph_out
            )  # warm up on the capture stream
            with torch.cuda.graph(graph, stream=ctx.stream):
                _run_collectives(dcp, static_inp, graph_out)
        for step in (1, 2):
            for name, value in _inputs(rank, step, device).items():
                static_inp[name].copy_(value)
            graph.replay()
            torch.cuda.synchronize()
            _check(f"graph replay {step}", graph_out, _expected(rank, step, device))

    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()
    if rank == 0:
        print(
            f"ALL PASSED (custom_all_reduce={custom_all_reduce}, dcp_is_tp={dcp is tp})"
        )


@pytest.mark.skipif(not is_hip(), reason="DCP reuses the TP coordinator on ROCm only")
@pytest.mark.parametrize("custom_all_reduce", [False, True])
def test_dcp8_collectives_eager_and_graph(custom_all_reduce: bool) -> None:
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"needs {WORLD} GPUs")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={WORLD}",
        __file__,
        "--custom-all-reduce" if custom_all_reduce else "--no-custom-all-reduce",
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0 and "ALL PASSED" in result.stdout, result.stdout


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _worker(custom_all_reduce="--custom-all-reduce" in sys.argv)
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
