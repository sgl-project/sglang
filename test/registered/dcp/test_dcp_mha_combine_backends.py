"""Equivalence test for the MHA/GQA DCP post-attention merge backends.

``DcpAttnComm.combine_mha`` dispatches the MHA/GQA reduction on
``--dcp-comm-backend``; before it existed the Triton backend hardcoded ``ag_rs``
and silently ignored ``a2a``/``fi_a2a``. The two patterns are supposed to compute
the same LSE-weighted merge by different routes (all-gather LSE + all-reduce +
head slice, versus all-to-all of head partials + a local Triton combine), so this
pins them to each other on the MHA convention: fp32 partials and natural-log LSE.

Usage:
    python -m pytest test_dcp_mha_combine_backends.py -v

This file doubles as the torchrun worker script.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Tuple

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="extra-b",
    runner_config="8-gpu-h200",
)

# (batch_tokens, heads_per_rank, head_dim); the merge widens heads by world_size.
TEST_SHAPES: List[Tuple[int, int, int]] = [
    (4, 2, 64),
    (16, 4, 128),
    (128, 8, 128),
    (3, 1, 512),
]


def multiprocess_test(file: str, nproc: int, timeout: int = 240) -> None:
    cmd = ["torchrun", f"--nproc_per_node={nproc}", file]
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


@pytest.mark.parametrize("nproc", [2, 4, 8])
def test_dcp_mha_combine_backends(nproc: int) -> None:
    device_count = torch.cuda.device_count()
    if device_count < nproc:
        pytest.skip(
            f"Requires at least {nproc} GPUs, but only {device_count} available"
        )
    multiprocess_test(__file__, nproc)


def init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    return local_rank, device, coord


@torch.inference_mode()
def worker_main() -> None:
    from sglang.srt.layers.dcp import cp_lse_ag_out_rs_mha, dcp_a2a_lse_reduce

    rank, device, coord = init_distributed()
    world_size = coord.world_size
    torch.cuda.set_stream(torch.cuda.Stream())

    for tokens, heads_per_rank, head_dim in TEST_SHAPES:
        widened_heads = heads_per_rank * world_size
        # Every rank must merge the same partials, so seed identically and let
        # the rank offset produce each rank's distinct contribution.
        generator = torch.Generator(device=device).manual_seed(1234)
        partials = torch.randn(
            world_size,
            tokens,
            widened_heads,
            head_dim,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        lses = (
            torch.randn(
                world_size,
                tokens,
                widened_heads,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            * 3.0
        )

        ag_rs_out = cp_lse_ag_out_rs_mha(partials[rank], lses[rank], coord)
        a2a_out = dcp_a2a_lse_reduce(
            partials[rank].contiguous(),
            lses[rank].contiguous(),
            coord,
            is_lse_base_on_e=True,
        )

        assert ag_rs_out.shape == a2a_out.shape, (
            f"rank {rank}: shape mismatch for "
            f"({tokens}, {heads_per_rank}, {head_dim}): "
            f"ag_rs {tuple(ag_rs_out.shape)} vs a2a {tuple(a2a_out.shape)}"
        )
        assert ag_rs_out.shape == (tokens, heads_per_rank, head_dim), (
            f"rank {rank}: merge must return this rank's head shard, got "
            f"{tuple(ag_rs_out.shape)}"
        )
        torch.testing.assert_close(
            a2a_out.float(),
            ag_rs_out.float(),
            rtol=2e-3,
            atol=2e-3,
            msg=lambda m: (
                f"rank {rank}: ag_rs and a2a disagree for "
                f"({tokens}, {heads_per_rank}, {head_dim})\n{m}"
            ),
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        worker_main()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
