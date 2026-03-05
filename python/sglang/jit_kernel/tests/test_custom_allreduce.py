"""
Correctness test for the JIT custom all-reduce (v2) kernel.

The test compares the JIT custom all-reduce output against NCCL all-reduce
for various tensor sizes and dtypes, in both eager and CUDA-graph modes.

Usage:
    python -m pytest test_jit_custom_allreduce.py -v

This file doubles as the torchrun worker script.  The test class launches
    torchrun --nproc_per_node=N <this_file>
and asserts that all worker processes exit successfully.
"""

from __future__ import annotations

import itertools
import logging
import os
import subprocess
from typing import TYPE_CHECKING

import pytest
import torch
import triton
from tqdm import tqdm

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

# ---------------------------------------------------------------------------
# Test parameters (shared between test class and worker)
# ---------------------------------------------------------------------------

TEST_SIZES = [
    512,
    4096,
    32768,
    262144,  # 256K elements
    2097152,  # 2M elements
    16777216,  # 16M elements
]
TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHOTS = [1, 2]
USE_GRAPH_OPTIONS = [False, True]
TEST_CONFIG = itertools.product(TEST_SIZES, TEST_DTYPES, SHOTS, USE_GRAPH_OPTIONS)
TEST_LOOP = 16

# ---------------------------------------------------------------------------
# Test class (runs via pytest, launches torchrun subprocesses)
# ---------------------------------------------------------------------------


def _run_torchrun(nproc: int, timeout: int = 300) -> None:
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
    _run_torchrun(nproc)


# ---------------------------------------------------------------------------
# Worker logic (executed by each torchrun process)
# ---------------------------------------------------------------------------


def _init_distributed():
    """Initialize distributed groups via torchrun env vars.

    Returns (rank, device, cpu_group, nccl_group, comm).
    """
    import torch.distributed as dist

    import sglang.srt.distributed.parallel_state as ps
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

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

    # Warmup NCCL to ensure the communicator is ready.
    warmup = torch.zeros(1, device=device)
    dist.all_reduce(warmup, group=nccl_group)
    torch.cuda.synchronize()
    del warmup

    max_size = 32 * 1024 * 1024  # 32MB
    comm = CustomAllReduceV2(group=cpu_group, device=device, max_size=max_size)
    if comm.disabled:
        raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")

    return rank, device, cpu_group, nccl_group, comm


def _worker_test(
    device: torch.device,
    nccl_group: torch.distributed.ProcessGroup,
    comm: CustomAllReduceV2,
    size: int,
    dtype: torch.dtype,
    use_graph: bool,
    shot: int,
) -> None:
    import torch.distributed as dist

    inp = torch.randint(0, 16, (TEST_LOOP, size), dtype=dtype, device=device)
    # NOTE: 16 * 8 <= 128, the max precision that can be exactly represented in BF16
    if not comm.should_custom_ar(inp):
        return
    comm.override_shot(shot)

    # Build a reference tensor via NCCL.
    ref = inp.clone()
    dist.all_reduce(ref, group=nccl_group)

    # Capture the JIT custom all-reduce in a CUDA graph.
    torch.cuda.synchronize()
    out_jits = []
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with comm.capture():
            with torch.cuda.graph(graph):
                for i in range(TEST_LOOP):
                    out_jits.append(comm.custom_all_reduce(inp[i]))
        torch.cuda.synchronize()
        # Replay and verify.
        graph.replay()
    else:
        for i in range(TEST_LOOP):
            out_jits.append(comm.custom_all_reduce(inp[i]))
    torch.cuda.synchronize()
    out_jit = torch.stack(out_jits)
    triton.testing.assert_close(out_jit, ref)  # should be no error


def worker_main() -> None:
    """Entry point for each torchrun worker process."""
    import torch.distributed as dist

    rank, device, cpu_group, nccl_group, comm = _init_distributed()
    world_size = dist.get_world_size()

    torch.cuda.set_stream(torch.cuda.Stream())

    logging.disable(logging.INFO)  # Suppress internal logging for cleaner test output
    items = list(enumerate(TEST_CONFIG))
    disable_pbar = os.environ.get("DISABLE_PBAR", "0") == "1" or rank != 0
    pbar = tqdm(items, desc=f"Testing {world_size} GPUs", disable=disable_pbar)
    for i, (size, dtype, shot, use_graph) in pbar:
        try:
            _worker_test(device, nccl_group, comm, size, dtype, use_graph, shot)
        except Exception as e:
            raise RuntimeError(
                f"Worker {rank} failed for size={size}, dtype={dtype}, "
                f"shot={shot}, use_graph={use_graph}, iteration={i}"
            ) from e

    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    worker_main()
