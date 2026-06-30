"""
Correctness test for ``GroupCoordinator.reduce_scatter_along_dim``.

The test compares the ``reduce_scatter_along_dim`` output against PyTorch's
native ``dist.reduce_scatter_tensor`` for various tensor shapes, dims, and
dtypes, exercising both positive and negative dim indexing.

Usage:
    python -m pytest test_reduce_scatter_along_dim.py -v

This file doubles as the torchrun worker script.  The test class launches
    torchrun --nproc_per_node=N <this_file>
and asserts that all worker processes exit successfully.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional, Tuple

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-unit",
    runner_config="8-gpu-h200",
)

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

# (head_num, batch_size, head_dim) shapes
TEST_SHAPES = [
    (8, 16, 32),
    (16, 64, 128),
    (4, 1024, 512),
    (16, 3, 128),
    (64, 5, 512),
    (128, 7, 512),
]


# For each shape we test several dim values (both positive and negative)
def _dims_for_shape(shape: Tuple[int, ...]) -> List[int]:
    ndim = len(shape)
    pos_dims = list(range(ndim))
    neg_dims = [-d - 1 for d in range(ndim)]
    return pos_dims + neg_dims


TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
TEST_LOOP = 8


# ---------------------------------------------------------------------------
# Helpers for multiprocess launch (shared between test and worker)
# ---------------------------------------------------------------------------


def multiprocess_test(file: str, nproc: int, timeout: int = 120) -> None:
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        file,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"torchrun (nproc={nproc}) timed out after {timeout}s\n{e.stdout}"
        ) from e

    assert result.returncode == 0, (
        f"torchrun (nproc={nproc}) failed with rc={result.returncode}\n"
        f"{result.stdout}"
    )


def multiprocess_main(file: str, main_fn) -> None:
    if "LOCAL_RANK" in os.environ:
        main_fn()
    else:
        sys.exit(pytest.main([file, "-v", "-s"]))


# ---------------------------------------------------------------------------
# Test class (runs via pytest, launches torchrun subprocesses)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nproc", [2, 4, 8])
def test_reduce_scatter_along_dim(nproc: int) -> None:
    device_count = torch.cuda.device_count()
    if device_count < nproc:
        pytest.skip(
            f"Requires at least {nproc} GPUs, but only {device_count} available"
        )
    multiprocess_test(__file__, nproc)


# ---------------------------------------------------------------------------
# Worker logic (executed by each torchrun process)
# ---------------------------------------------------------------------------


def init_distributed():
    """Initialize distributed groups via torchrun env vars."""
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

    return rank, device, cpu_group, nccl_group, coord


def _reference_reduce_scatter_along_dim(
    input_: torch.Tensor,
    dim: int,
    world_size: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Reference implementation using torch.distributed.reduce_scatter_tensor."""
    if dim < 0:
        dim += input_.dim()

    # Move target dim to position 0 and make contiguous
    input_tensor = input_.movedim(dim, 0).contiguous()

    assert input_tensor.shape[0] % world_size == 0
    chunk_size = input_tensor.shape[0] // world_size
    output_shape = (chunk_size,) + input_tensor.shape[1:]

    output_tensor = torch.empty(
        output_shape,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    dist.reduce_scatter_tensor(output_tensor, input_tensor, group=group)

    # Move dim back
    return output_tensor.movedim(0, dim)


@torch.inference_mode()
def worker_test(
    device: torch.device,
    nccl_group: dist.ProcessGroup,
    coord: ps.GroupCoordinator,
    shape: Tuple[int, ...],
    dim: int,
    dtype: torch.dtype,
    world_size: int,
) -> Optional[RuntimeError]:
    """Run a single (shape, dim, dtype) configuration and compare against reference."""
    for _ in range(TEST_LOOP):
        inp = torch.randint(0, 16, shape, dtype=dtype, device=device)

        # Our implementation
        out = coord.reduce_scatter_along_dim(inp, dim=dim)

        # Reference
        ref = _reference_reduce_scatter_along_dim(inp, dim, world_size, nccl_group)

        if not torch.all(out == ref):
            return RuntimeError(f"Mismatch for shape={shape}, dim={dim}, dtype={dtype}")
    return None


def worker_main() -> None:
    """Entry point for each torchrun worker process."""
    rank, device, cpu_group, nccl_group, coord = init_distributed()
    world_size = coord.world_size

    torch.cuda.set_stream(torch.cuda.Stream())

    errors: List[str] = []
    for shape in TEST_SHAPES:
        # Only test dims where shape[dim] is divisible by world_size
        for dim in _dims_for_shape(shape):
            actual_dim = dim if dim >= 0 else dim + len(shape)
            if shape[actual_dim] % world_size != 0:
                continue

            for dtype in TEST_DTYPES:
                error = worker_test(
                    device, nccl_group, coord, shape, dim, dtype, world_size
                )
                if error is not None:
                    errors.append(str(error))

                # Synchronize across ranks – if any rank fails, all fail
                result = torch.tensor([int(error is not None)], device="cpu")
                dist.all_reduce(result, group=cpu_group)
                if result.item():
                    raise RuntimeError(
                        f"Rank {rank} failed for shape={shape}, dim={dim}, "
                        f"dtype={dtype}. Errors: {'; '.join(errors)}"
                    )

    dist.destroy_process_group()


if __name__ == "__main__":
    multiprocess_main(__file__, worker_main)
