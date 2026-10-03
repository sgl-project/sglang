"""Cake Kimi-K3 TP12 fused LatentMoE tail through sglang.kernels.

Checks the registry resolution (no GPU), the in-process ``supports_*``
admission (world size 12, latent / hidden sizes, dtype, missing FlashInfer
module) and, under ``torchrun`` with exactly 12 ranks of a GB200 / GB300 NVL72
NVLink domain, parity of the prepared runner and the one-shot function with a
pure-torch reference plus rank invariance of the output. The 12-rank test
skips with the reason otherwise.

Usage::

    python test/registered/kernels/ops/communication/test_cake_kimi_k3_tp12_tail.py --num-gpu 12
"""

from __future__ import annotations

import atexit
import os
from typing import Optional

import pytest
import torch
import torch.distributed as dist

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import communication_kimi_k3_tp12_tail as cake_kimi
from sglang.kernels.ops.communication.cake import (
    cake_create_kimi_k3_tp12_tail_workspace,
    cake_kimi_k3_tp12_tail,
    cake_prepare_kimi_k3_tp12_tail,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

# Single-GPU parts only; the 12-rank parity test self-skips on this runner.
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "communication.kimi_k3_tp12_tail",
    "communication.prepare_kimi_k3_tp12_tail",
    "communication.create_kimi_k3_tp12_tail_workspace",
)
HIDDEN = cake_kimi.HIDDEN
LATENT = cake_kimi.LATENT
RMS_EPS = 1e-5


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith(
        "sglang.kernels.cake_kernels.communication_kimi_k3_tp12_tail:"
    )


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.cuda.current_device())
    if not cake_kimi.flashinfer_module_available(
        cake_kimi.FI_MODULE, cake_kimi.FI_BACKEND_MODULE
    ):
        pytest.skip("installed FlashInfer lacks flashinfer.kimi_k3_tp12_tail")
    if torch.cuda.get_device_capability(device) not in cake_kimi.ARCHS:
        pytest.skip("Cake Kimi-K3 TP12 tail is built for sm_100a / sm_103a")
    return device


def test_supports_rejects_bad_inputs(monkeypatch):
    device = _cuda_or_skip()
    routed = torch.empty(16, LATENT, device=device, dtype=torch.bfloat16)
    shared = torch.empty(16, HIDDEN, device=device, dtype=torch.bfloat16)
    assert cake_kimi.supports_kimi_k3_tp12_tail(routed, shared, world_size=12)
    assert cake_kimi.supports_kimi_k3_tp12_tail(
        routed, shared, world_size=12, max_tokens=16
    )
    assert not cake_kimi.supports_kimi_k3_tp12_tail(
        routed, shared, world_size=12, max_tokens=8
    )
    assert not cake_kimi.supports_kimi_k3_tp12_tail(routed, shared, world_size=8)
    assert not cake_kimi.supports_kimi_k3_tp12_tail(routed, shared, world_size=16)
    assert not cake_kimi.supports_kimi_k3_tp12_tail(
        routed[:, :2048], shared, world_size=12
    )
    assert not cake_kimi.supports_kimi_k3_tp12_tail(
        routed, shared[:, :4096], world_size=12
    )
    assert not cake_kimi.supports_kimi_k3_tp12_tail(routed, shared[:8], world_size=12)
    assert not cake_kimi.supports_kimi_k3_tp12_tail(
        routed.half(), shared.half(), world_size=12
    )
    assert not cake_kimi.supports_kimi_k3_tp12_tail(
        routed.cpu(), shared.cpu(), world_size=12
    )
    monkeypatch.setattr(cake_kimi, "flashinfer_module_available", lambda *a: False)
    assert not cake_kimi.supports_kimi_k3_tp12_tail(routed, shared, world_size=12)


# ---------------------------------------------------------------------------
# Twelve-rank parity (torchrun workers only; MNNVL fabric memory)
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


def _reference(routed, shared, norm_w, up_w, group):
    """FP32 partial sums, one BF16 rounding of the latent, KimiRMSNorm, one final rounding."""
    world = dist.get_world_size(group)
    rs = [torch.empty_like(routed) for _ in range(world)]
    ss = [torch.empty_like(shared) for _ in range(world)]
    dist.all_gather(rs, routed, group=group)
    dist.all_gather(ss, shared, group=group)
    routed_sum = torch.stack([t.float() for t in rs]).sum(0).to(torch.bfloat16)
    shared_sum = torch.stack([t.float() for t in ss]).sum(0)
    xf = routed_sum.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    y = norm_w * xf.to(torch.bfloat16)
    gemm = y.float() @ up_w.float().t()
    return (gemm + shared_sum).to(torch.bfloat16)


def _rank_invariant(out: torch.Tensor, group: dist.ProcessGroup) -> bool:
    outs = [torch.empty_like(out) for _ in range(dist.get_world_size(group))]
    dist.all_gather(outs, out, group=group)
    return all(torch.equal(outs[0], t) for t in outs[1:])


@pytest.mark.parametrize("tokens", [1, 8, 64, 300])
def test_tail_matches_reference_on_twelve_ranks(tokens):
    device = _cuda_or_skip()
    group = _dist_group()
    if group is None or dist.get_world_size(group) != cake_kimi.WORLD_SIZE:
        pytest.skip("needs 12 ranks with MNNVL (GB200 / GB300 NVL72)")
    rank = dist.get_rank(group)
    max_tokens = 300
    workspace = cake_create_kimi_k3_tp12_tail_workspace(
        rank=rank, max_tokens=max_tokens, group=group, device=device
    )
    try:
        torch.manual_seed(900 + rank)
        routed = torch.randn(tokens, LATENT, device=device, dtype=torch.bfloat16)
        shared = torch.randn(tokens, HIDDEN, device=device, dtype=torch.bfloat16)
        torch.manual_seed(9)  # replicated weights
        norm_w = (torch.rand(LATENT, device=device) + 0.5).to(torch.bfloat16)
        up_w = (torch.randn(HIDDEN, LATENT, device=device) * 0.02).to(torch.bfloat16)
        out = torch.full(
            (tokens, HIDDEN), float("nan"), device=device, dtype=torch.bfloat16
        )
        assert cake_kimi.supports_kimi_k3_tp12_tail(
            routed, shared, world_size=cake_kimi.WORLD_SIZE, max_tokens=max_tokens
        )
        expected = _reference(routed, shared, norm_w, up_w, group)

        runner = cake_prepare_kimi_k3_tp12_tail(
            routed, shared, norm_w, up_w, out, workspace=workspace
        )
        result = runner()
        torch.cuda.synchronize()
        assert result is out
        assert torch.isfinite(out.float()).all()
        torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)
        assert _rank_invariant(out, group)
        first = out.clone()

        # Second launch of the same runner (Lamport buffers rotate) is bitwise stable.
        out.fill_(float("nan"))
        runner()
        torch.cuda.synchronize()
        assert torch.equal(out, first)

        # One-shot function path.
        out.fill_(float("nan"))
        cake_kimi_k3_tp12_tail(routed, shared, norm_w, up_w, out, workspace=workspace)
        torch.cuda.synchronize()
        assert torch.equal(out, first)
        dist.barrier(group=group)
    finally:
        torch.cuda.synchronize()
        workspace.destroy()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(12,))
