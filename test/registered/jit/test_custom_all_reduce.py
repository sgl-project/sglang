"""Correctness test for the JIT custom all-reduce (v2) kernel.

Compares the JIT custom all-reduce output against NCCL all-reduce for a sweep
of tensor sizes, dtypes, and algorithms, in both eager and CUDA-graph modes.

Usage::

    # Run the test on the default world sizes (2, 4, 8 GPUs):
    python tests/test_custom_all_reduce.py
    # Pick a specific world size (or comma-separated list), e.g. the rarer
    # odd / non-power-of-two counts that the default sweep skips:
    python tests/test_custom_all_reduce.py --num-gpu 3
    python tests/test_custom_all_reduce.py --num-gpu 2,4,6,8
    # Extra pytest args (forwarded to each torchrun worker):
    python tests/test_custom_all_reduce.py -k bfloat16
"""

from __future__ import annotations

import atexit
import itertools
import logging
import multiprocessing
import os
from multiprocessing.context import SpawnProcess
from typing import List

import pytest
import torch
import torch.distributed as dist
import triton

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.all_reduce import (
    AllReduceAlgo,
    _jit_custom_all_reduce_pull_module,
    _jit_custom_all_reduce_push_module,
)
from sglang.jit_kernel.mp import register_comm_cleanup
from sglang.jit_kernel.tests.utils import multigpu_pytest_main
from sglang.jit_kernel.utils import cache_once, get_ci_test_range
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300,
    suite="base-b-kernel-unit-8-gpu-h200",
)
register_cuda_ci(
    est_time=300,
    suite="nightly-kernel-8-gpu-h200",
    nightly=True,
)

# ---------------------------------------------------------------------------
# Test parameters
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
    2 * 1024 * 1024,
    4 * 1024 * 1024,
]
TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
TEST_ALGOS = [
    AllReduceAlgo.ONE_SHOT_PULL,
    AllReduceAlgo.ONE_SHOT_PUSH,
    AllReduceAlgo.TWO_SHOT_PULL,
]
USE_GRAPH_OPTIONS = [False, True]
TEST_LAYERS = 4
TEST_LOOP = 16

TEST_SIZES = get_ci_test_range(TEST_SIZES, [16, 1024, 32 * 1024, 2 * 1024 * 1024])
TEST_DTYPES = get_ci_test_range(TEST_DTYPES, [torch.bfloat16])

# ---------------------------------------------------------------------------
# Parallel JIT precompile (outer process, before any torchrun child starts)
# ---------------------------------------------------------------------------


def _compile_one(dtype: torch.dtype, world_size: int) -> None:
    """Compile both (push, pull) variants for a single (dtype, world_size).

    Top-level so it survives ``spawn`` pickling. Compiled artifacts are
    cached on disk by ``tvm_ffi``; torchrun children will reuse them.
    """
    _jit_custom_all_reduce_pull_module(dtype, world_size)
    _jit_custom_all_reduce_push_module(dtype, world_size)


def _precompile_kernels(num_gpus: List[int]) -> None:
    """Fan out one process per (dtype, world_size) to warm the JIT cache.

    Without this, every torchrun child serial-compiles its kernels on first
    use, multiplying the wall-clock cost of the run by ~(#dtypes * #ranks).
    """
    ctx = multiprocessing.get_context("spawn")
    procs: list[tuple[torch.dtype, int, SpawnProcess]] = []
    for dtype, world_size in itertools.product(TEST_DTYPES, num_gpus):
        p = ctx.Process(target=_compile_one, args=(dtype, world_size))
        p.start()
        procs.append((dtype, world_size, p))
    for dtype, world_size, p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"Custom-all-reduce precompile failed for "
                f"{dtype=} {world_size=} (exit {p.exitcode})"
            )


# ---------------------------------------------------------------------------
# Per-rank distributed setup (run once per torchrun worker)
# ---------------------------------------------------------------------------


@cache_once
def _init_cpu_group_once() -> dist.ProcessGroup:
    """Initialize gloo world group + cuda device for this rank."""
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    cpu_group = coord.cpu_group
    assert isinstance(cpu_group, dist.ProcessGroup)
    # Suppress chatty internal logging for cleaner test output.
    logging.disable(logging.INFO)
    # Use a non-default stream (mirrors prior behavior).
    torch.cuda.set_stream(torch.cuda.Stream())
    return cpu_group


@cache_once
def _init_nccl_group_once() -> dist.ProcessGroup:
    _init_cpu_group_once()
    coord = ps._WORLD
    assert coord is not None and coord.device_group is not None
    return coord.device_group


@cache_once
def _init_comm_once() -> CustomAllReduceV2:
    cpu_group = _init_cpu_group_once()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    max_size = max(TEST_SIZES) * max(
        torch.tensor([], dtype=d).element_size() for d in TEST_DTYPES
    )
    comm = CustomAllReduceV2(cpu_group, device, max_size, max_size)
    if comm.disabled:
        raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")
    register_comm_cleanup(comm)
    return comm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_graph", USE_GRAPH_OPTIONS)
@pytest.mark.parametrize("algo", TEST_ALGOS)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("size", TEST_SIZES)
@torch.inference_mode()
def test_custom_all_reduce(
    size: int,
    dtype: torch.dtype,
    algo: AllReduceAlgo,
    use_graph: bool,
) -> None:
    nccl_group = _init_nccl_group_once()
    comm = _init_comm_once()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    comm.override_algo = algo

    if use_graph:
        graph = torch.cuda.CUDAGraph()
        graph_inp = torch.zeros((TEST_LAYERS, size), dtype=dtype, device=device)
        outs: list[torch.Tensor] = []
        with comm.capture():
            with torch.cuda.graph(graph):
                for i in range(TEST_LAYERS):
                    outs.append(comm.custom_all_reduce(graph_inp[i]))
                out_jit_stack = torch.stack(outs)
        torch.cuda.synchronize()

        def run(x: torch.Tensor) -> torch.Tensor:
            graph_inp.copy_(x)
            graph.replay()
            return out_jit_stack.clone()

    else:

        def run(x: torch.Tensor) -> torch.Tensor:
            eager_inp = x.clone()
            outs = []
            for i in range(TEST_LAYERS):
                outs.append(comm.custom_all_reduce(eager_inp[i]))
                torch.cuda.synchronize()
            return torch.stack(outs)

    for _ in range(TEST_LOOP):
        # NOTE: 15 * 8 < 128, which is the precision limit for bf16
        inp = torch.randint(0, 16, (TEST_LAYERS, size), dtype=dtype, device=device)
        assert comm.should_custom_ar(inp[0])
        out_ref = inp.clone()
        dist.all_reduce(out_ref, group=nccl_group)
        out_jit = run(inp)
        # Exact equality, since values are small integers within bf16 precision.
        triton.testing.assert_close(out_ref, out_jit, atol=0, rtol=0)


if __name__ == "__main__":
    # Only sweep the common world sizes (2, 4, 8) by default: testing every
    # count in 2..8 serially overruns the per-file CI time budget, and 3/5/6/7
    # are rare in practice. Use --num-gpu to exercise them explicitly.
    multigpu_pytest_main(
        __name__,
        __file__,
        num_gpus=(2, 4, 8),
        pre_launch_fn=_precompile_kernels,
    )
