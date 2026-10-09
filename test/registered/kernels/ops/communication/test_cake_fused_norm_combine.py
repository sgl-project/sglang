"""Cake fused residual + two-track RMSNorm + eight-peer combine through sglang.kernels.

Checks the registry resolution (no GPU), the in-process ``supports_*``
admission (world size, hidden size, dtype, missing FlashInfer module) and,
under ``torchrun`` with exactly 8 ranks on sm_100a / sm_103a with CUDA IPC,
parity of the fused launch with a pure-torch reference. The 8-rank test skips
with the reason otherwise.

Usage::

    python test/registered/kernels/ops/communication/test_cake_fused_norm_combine.py
"""

from __future__ import annotations

import atexit
import os
from typing import Optional

import pytest
import torch
import torch.distributed as dist

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import communication as cake_comm
from sglang.kernels.ops.communication.cake import (
    cake_fused_norm_combine,
    cake_fused_norm_combine_create_workspace,
    cake_fused_norm_combine_destroy_workspace,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=120, stage="nightly", runner_config="8-gpu-b200")

OPS = (
    "communication.fused_norm_combine",
    "communication.fused_norm_combine_create_workspace",
    "communication.fused_norm_combine_destroy_workspace",
)
T = cake_comm.NC_TRACKS
H = cake_comm.NC_HIDDEN


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.communication:")


def _module_available() -> bool:
    return cake_comm.flashinfer_module_available(
        cake_comm.FI_NC_MODULE, cake_comm.FI_NC_JIT_MODULE
    )


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.cuda.current_device())
    if not _module_available():
        pytest.skip(
            "installed FlashInfer lacks flashinfer.comm.cake_fused_norm_combine"
        )
    if torch.cuda.get_device_capability(device) not in cake_comm.ARCHS:
        pytest.skip("Cake fused norm-combine is built for sm_100a / sm_103a")
    return device


def test_supports_rejects_bad_inputs(monkeypatch):
    device = _cuda_or_skip()
    x = torch.empty(32, T, H, device=device, dtype=torch.bfloat16)
    assert cake_comm.supports_fused_norm_combine(x, world_size=8)
    assert cake_comm.supports_fused_norm_combine(x, world_size=8, max_tokens=32)
    assert not cake_comm.supports_fused_norm_combine(x, world_size=8, max_tokens=16)
    assert not cake_comm.supports_fused_norm_combine(x, world_size=4)
    assert not cake_comm.supports_fused_norm_combine(x, world_size=12)
    assert not cake_comm.supports_fused_norm_combine(x[:, :, :2048], world_size=8)
    assert not cake_comm.supports_fused_norm_combine(x[:, :1], world_size=8)
    assert not cake_comm.supports_fused_norm_combine(x.half(), world_size=8)
    assert not cake_comm.supports_fused_norm_combine(x.view(32, T * H), world_size=8)
    assert not cake_comm.supports_fused_norm_combine(x.cpu(), world_size=8)
    monkeypatch.setattr(cake_comm, "flashinfer_module_available", lambda *a: False)
    assert not cake_comm.supports_fused_norm_combine(x, world_size=8)


# ---------------------------------------------------------------------------
# Eight-rank parity (torchrun workers only)
# ---------------------------------------------------------------------------


def _dist_group() -> Optional[dist.ProcessGroup]:
    if dist.is_available() and dist.is_initialized():
        return dist.group.WORLD
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        return None
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl")
    atexit.register(dist.destroy_process_group)
    return dist.group.WORLD


def _reference(x, residual, weight, eps, group):
    """Residual add, per-track RMSNorm, track mean, rank-ordered BF16 sum."""
    res = (x.float() + residual.float()).to(torch.bfloat16)
    rf = res.float()
    norm = (rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + eps)) * weight.float()
    norm_bf16 = norm.to(torch.bfloat16)
    contribution = norm_bf16.float().mean(dim=1).to(torch.bfloat16)  # [T, H]
    world = dist.get_world_size(group)
    parts = [torch.empty_like(contribution) for _ in range(world)]
    dist.all_gather(parts, contribution, group=group)
    acc = parts[0].float()
    for part in parts[1:]:
        acc = (acc + part.float()).to(torch.bfloat16).float()
    return norm_bf16, res, acc


@pytest.mark.parametrize("tokens", [8, 300, 1100])
def test_fused_norm_combine_matches_reference(tokens):
    device = _cuda_or_skip()
    group = _dist_group()
    if group is None or dist.get_world_size(group) != cake_comm.NC_WORLD_SIZE:
        pytest.skip("needs 8 ranks with CUDA IPC (one node)")
    rank = dist.get_rank(group)
    max_tokens = 1100
    workspace = cake_fused_norm_combine_create_workspace(
        rank=rank,
        world_size=cake_comm.NC_WORLD_SIZE,
        max_tokens=max_tokens,
        group=group,
        device=device,
    )
    try:
        torch.manual_seed(500 + rank)
        x = torch.randn(tokens, T, H, device=device, dtype=torch.bfloat16)
        residual = torch.randn(tokens, T, H, device=device, dtype=torch.bfloat16)
        torch.manual_seed(5)  # replicated weight
        weight = (torch.rand(T, H, device=device) + 0.5).to(torch.bfloat16)
        eps = 1e-6
        assert cake_comm.supports_fused_norm_combine(
            x, world_size=cake_comm.NC_WORLD_SIZE, max_tokens=max_tokens
        )
        norm_out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        collective_out = torch.empty(tokens, H, device=device, dtype=torch.bfloat16)
        cake_fused_norm_combine(
            x,
            residual,
            weight,
            norm_out=norm_out,
            residual_out=residual_out,
            collective_out=collective_out,
            workspace=workspace,
            epsilon=eps,
        )
        torch.cuda.synchronize()
        norm_ref, res_ref, coll_ref = _reference(x, residual, weight, eps, group)
        torch.testing.assert_close(
            residual_out.float(), res_ref.float(), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            norm_out.float(), norm_ref.float(), atol=1e-2, rtol=1e-2
        )
        # Eight sequential BF16 roundings in the kernel vs. the reference's
        # rank-ordered emulation: allow two BF16 ulps of the summed magnitude.
        torch.testing.assert_close(
            collective_out.float(), coll_ref, atol=2e-2, rtol=2e-2
        )
        # Replicated result on every rank.
        gathered = [
            torch.empty_like(collective_out) for _ in range(cake_comm.NC_WORLD_SIZE)
        ]
        dist.all_gather(gathered, collective_out, group=group)
        for other in gathered[1:]:
            assert torch.equal(gathered[0], other)
        dist.barrier(group=group)
    finally:
        torch.cuda.synchronize()
        cake_fused_norm_combine_destroy_workspace(workspace)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(8,))
