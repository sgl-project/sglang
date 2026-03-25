"""
Correctness test for the JIT custom all-reduce (v2) kernel.

The test compares the JIT custom all-reduce output against NCCL all-reduce
for various tensor sizes and dtypes, in both eager and CUDA-graph modes.

Usage:
    python -m pytest test_jit_custom_all_reduce.py -v

This file doubles as the torchrun worker script.  The test class launches
    torchrun --nproc_per_node=N <this_file>
and asserts that all worker processes exit successfully.
"""

from __future__ import annotations

import itertools
import logging
import os
import subprocess
import sys
from typing import Optional

import pytest
import torch
import torch.distributed as dist
from tqdm import tqdm

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.all_reduce import AllReduceAlgo
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    suite="stage-b-kernel-unit-1-gpu-large",
    disabled="requires multi-GPU distributed setup",
)
register_cuda_ci(
    est_time=120,
    suite="nightly-kernel-1-gpu",
    nightly=True,
    disabled="requires multi-GPU distributed setup",
)

# ---------------------------------------------------------------------------
# Test parameters (shared between test class and worker)
# ---------------------------------------------------------------------------

TEST_SIZES = [
    16,
    32,
    512,
    1024,
    1024 + 16,  # weird case
    4 * 1024,
    32 * 1024,
    256 * 1024,
    2 * 1024 * 1024,  # 2M elements
    4 * 1024 * 1024,  # 4M elements
]
TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHOTS = [
    AllReduceAlgo.ONE_SHOT_PULL,
    AllReduceAlgo.ONE_SHOT_PUSH,
    AllReduceAlgo.TWO_SHOT_PULL,
]
USE_GRAPH_OPTIONS = [True, False]
TEST_CONFIG = itertools.product(TEST_SIZES, TEST_DTYPES, SHOTS, USE_GRAPH_OPTIONS)
TEST_LAYERS = 2
TEST_LOOP = 16

# ---------------------------------------------------------------------------
# Test class (runs via pytest, launches torchrun subprocesses)
# ---------------------------------------------------------------------------


def run_torchrun(nproc: int, timeout: int = 300) -> None:
    """Launch this script as a torchrun worker and assert success."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        __file__,
    ]
    os.environ["DISABLE_PBAR"] = "1"
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"torchrun (nproc={nproc}) failed with rc={result.returncode}\n"
        f"{result.stdout}"
    )


@pytest.mark.parametrize("nproc", [2, 3, 4, 5, 6, 7, 8])
def test_custom_allreduce(nproc: int) -> None:
    device_count = torch.cuda.device_count()
    if device_count < nproc:
        pytest.skip(
            f"Requires at least {nproc} GPUs, but only {device_count} available"
        )
    run_torchrun(nproc)


# ---------------------------------------------------------------------------
# Worker logic (executed by each torchrun process)
# ---------------------------------------------------------------------------


def init_distributed():
    """Initialize distributed groups via torchrun env vars.

    Returns (rank, device, cpu_group, nccl_group, comm).
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = local_rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )

    cpu_group = coord.cpu_group
    nccl_group = coord.device_group
    assert nccl_group is not None

    max_size = max(TEST_SIZES) * 4
    comm = CustomAllReduceV2(cpu_group, device, max_size, max_size)
    if comm.disabled:
        raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")

    return rank, device, cpu_group, nccl_group, comm


@torch.inference_mode()
def worker_test(
    device: torch.device,
    nccl_group: dist.ProcessGroup,
    comm: CustomAllReduceV2,
    size: int,
    dtype: torch.dtype,
    use_graph: bool,
    algo: AllReduceAlgo,
) -> Optional[RuntimeError]:
    comm.override_algo = algo

    def get_run_graph_fn():
        graph = torch.cuda.CUDAGraph()
        graph_inp = torch.zeros((TEST_LAYERS, size), dtype=dtype, device=device)
        out_jits = []
        with comm.capture():
            with torch.cuda.graph(graph):
                for i in range(TEST_LAYERS):
                    out_jits.append(comm.custom_all_reduce(graph_inp[i]))
                out_jit = torch.stack(out_jits)
        torch.cuda.synchronize()

        def run_graph(x: torch.Tensor) -> torch.Tensor:
            graph_inp.copy_(x)
            graph.replay()
            return out_jit.clone()

        return run_graph

    def get_run_eager_fn():
        def run_eager(x: torch.Tensor) -> torch.Tensor:
            eager_inp = x.clone()
            out_eagers = []
            for i in range(TEST_LAYERS):
                out_eagers.append(comm.custom_all_reduce(eager_inp[i]))
                torch.cuda.synchronize()
            return torch.stack(out_eagers)

        return run_eager

    run_fn = get_run_graph_fn() if use_graph else get_run_eager_fn()
    num_errors = 0
    for _ in range(TEST_LOOP):
        # NOTE: 15 * 8 < 128, which is the precision limit for bf16
        inp = torch.randint(0, 16, (TEST_LAYERS, size), dtype=dtype, device=device)
        assert comm.should_custom_ar(inp[0])
        out_ref = inp.clone()
        dist.all_reduce(out_ref, group=nccl_group)
        out_jit = run_fn(inp)
        num_errors += not torch.all(out_jit == out_ref)
        torch.cuda.synchronize()
        nccl_group.barrier().wait()
    if num_errors > 0:
        return RuntimeError(
            f"Test failed for {size=}, {dtype=}, {algo=}, "
            f"{use_graph=} with {num_errors} errors. "
        )
    return None


def worker_main() -> None:
    """Entry point for each torchrun worker process."""
    rank, device, cpu_group, nccl_group, comm = init_distributed()
    world_size = dist.get_world_size()

    torch.cuda.set_stream(torch.cuda.Stream())

    logging.disable(logging.INFO)  # Suppress internal logging for cleaner test output
    items = list(enumerate(TEST_CONFIG))
    disable_pbar = os.environ.get("DISABLE_PBAR", "0") == "1" or rank != 0
    pbar = tqdm(items, desc=f"Testing {world_size} GPUs", disable=disable_pbar)
    for i, (size, dtype, algo, use_graph) in pbar:
        error = worker_test(device, nccl_group, comm, size, dtype, use_graph, algo)
        if error is not None:
            print(
                f"Worker {rank} failed for {size=}, {dtype=}, "
                f"{algo=}, {use_graph=}, iteration={i}\n"
                f"Error: {error}"
            )
        # communicate the result to rank 0 for logging
        result = torch.tensor([int(error is not None)], device=device)
        dist.all_reduce(result, group=cpu_group)
        failed = bool(result.item())
        if failed:
            raise RuntimeError(
                f"Test failed on rank {rank} for config: "
                f"{size=}, {dtype=}, {algo=}, {use_graph=}"
            )

    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        worker_main()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
