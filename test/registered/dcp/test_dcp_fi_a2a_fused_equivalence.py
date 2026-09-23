"""
Equivalence test for the ``fi_a2a_fused`` DCP comm backend.

``fi_a2a`` exchanges head partials with FlashInfer's MNNVL all-to-all and then
merges them with SGLang's Triton LSE kernel. ``fi_a2a_fused`` hands both halves
to FlashInfer's fused kernel (flashinfer #4929). The two must agree: this test
feeds both paths identical partials and compares their outputs.

The fused op is unreleased (targets FlashInfer 0.7.0), so the whole module skips
when it is not importable -- see ``helix/scripts/fi4929_overlay/`` in the work
tracker for the source overlay that provides it.

Requirements: all ranks in one NVLink/LSA domain, NCCL >= 2.29 with device
communicator support, and fp16/bf16 partials.

Usage:
    python -m pytest test_dcp_fi_a2a_fused_equivalence.py -v

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
    est_time=240,
    stage="extra-b",
    runner_config="8-gpu-b200",
)

# (batch, heads_per_rank, head_dim). head_dim * itemsize must be 16-byte
# aligned; 512 is the DeepSeek/Kimi MLA latent rank, the shape that matters.
TEST_SHAPES: List[Tuple[int, int, int]] = [
    (1, 2, 512),
    (1, 16, 512),
    (8, 16, 512),
    (64, 8, 512),
]
TEST_DTYPES = [torch.bfloat16, torch.float16]
TEST_LOOP = 4

# The fused kernel reduces in fp32 but accumulates in a different order from the
# Triton combine, so compare at the tolerance FlashInfer's own test uses.
RTOL, ATOL = 1e-2, 1e-3


def multiprocess_test(file: str, nproc: int, timeout: int = 600) -> None:
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
        f"torchrun (nproc={nproc}) failed with rc={result.returncode}\n{result.stdout}"
    )


def _fused_available() -> bool:
    try:
        from flashinfer.comm import (  # noqa: F401
            decode_cp_a2a_lse_reduce,
            decode_cp_a2a_lse_reduce_create_workspace,
        )
    except ImportError:
        return False
    return True


@pytest.mark.parametrize("nproc", [2, 4, 8])
def test_fi_a2a_fused_matches_fi_a2a(nproc: int) -> None:
    if not _fused_available():
        pytest.skip("FlashInfer fused DCP LSE reduce (#4929) not available")
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
def _run_shape(
    device, coord, workspace, shape, dtype, is_lse_base_on_e, seed
) -> str | None:
    """Compare the fused kernel against an all-gather + local reference merge."""
    from flashinfer.comm import decode_cp_a2a_lse_reduce

    from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton

    B, H_per_rank, D = shape
    N = coord.world_size
    rank = coord.rank_in_group

    # Seed per (rank, iteration): ranks must differ or the merge is trivial, and
    # iterations must differ or the repeat loop re-tests one input.
    torch.manual_seed(seed * 1000 + rank)
    partial_o = torch.randn(B, H_per_rank, N, D, dtype=dtype, device=device)
    partial_lse = torch.randn(B, H_per_rank, N, dtype=torch.float32, device=device)

    fused = decode_cp_a2a_lse_reduce(
        partial_o,
        partial_lse,
        workspace,
        rank,
        N,
        lse_mode="basee" if is_lse_base_on_e else "base2",
    )

    # Reference: reconstruct the all-to-all with an all-gather, take the column
    # addressed to this rank, then merge with the Triton kernel fi_a2a uses.
    gathered_o = [torch.empty_like(partial_o) for _ in range(N)]
    gathered_lse = [torch.empty_like(partial_lse) for _ in range(N)]
    dist.all_gather(gathered_o, partial_o, group=coord.device_group)
    dist.all_gather(gathered_lse, partial_lse, group=coord.device_group)

    # gathered_*[src][..., rank, :] is what src sent to us.
    recv_o = torch.stack([g[:, :, rank, :] for g in gathered_o], dim=0)
    recv_lse = torch.stack([g[:, :, rank] for g in gathered_lse], dim=0)

    ref, _ = dcp_lse_combine_triton(recv_o, recv_lse, is_lse_base_on_e=is_lse_base_on_e)

    if not torch.allclose(fused.float(), ref.float(), rtol=RTOL, atol=ATOL):
        diff = (fused.float() - ref.float()).abs().max().item()
        return (
            f"shape={shape} dtype={dtype} base_e={is_lse_base_on_e}: "
            f"max abs diff {diff}"
        )
    return None


def worker_main() -> None:
    from flashinfer.comm import decode_cp_a2a_lse_reduce_create_workspace

    rank, device, coord = init_distributed()
    N = coord.world_size

    max_b = max(s[0] for s in TEST_SHAPES)
    max_h = max(s[1] for s in TEST_SHAPES)
    max_d = max(s[2] for s in TEST_SHAPES)

    for dtype in TEST_DTYPES:
        # A workspace is dtype-specific and bound to one stream; build a fresh
        # one per dtype rather than reusing across them.
        workspace = decode_cp_a2a_lse_reduce_create_workspace(
            max_tokens=max_b,
            local_heads=max_h,
            cp_size=N,
            head_dim=max_d,
            dtype=dtype,
            group=coord.device_group,
        )
        dist.barrier(group=coord.device_group)

        for shape in TEST_SHAPES:
            for is_lse_base_on_e in (False, True):
                for it in range(TEST_LOOP):
                    error = _run_shape(
                        device, coord, workspace, shape, dtype, is_lse_base_on_e, it
                    )
                    # If any rank mismatches, every rank must fail together --
                    # a one-rank raise would hang the others in the next collective.
                    flag = torch.tensor([int(error is not None)], device="cpu")
                    dist.all_reduce(flag, group=coord.cpu_group)
                    if flag.item():
                        raise RuntimeError(
                            f"Rank {rank} fused/reference mismatch: {error}"
                        )

        del workspace
        torch.cuda.synchronize(device)

    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        worker_main()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
