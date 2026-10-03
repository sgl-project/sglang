"""Cake MoE all-reduce union / finalize + all-reduce through sglang.kernels.

Checks the registry resolution (no GPU), the in-process ``supports_*``
admission (world size, hidden size, dtype, missing FlashInfer JIT module) and,
under ``torchrun`` with 2, 4 or 8 ranks on sm_100a / sm_103a, parity of the
``backend="cake"`` launches on a TRT-LLM IPC workspace with a pure-torch
reference. The multi-rank tests skip with the reason otherwise.

Usage::

    python test/registered/kernels/ops/communication/test_cake_moe_allreduce_fusion.py
    python test/registered/kernels/ops/communication/test_cake_moe_allreduce_fusion.py --num-gpu 4
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
    cake_trtllm_moe_allreduce_fusion,
    cake_trtllm_moe_finalize_allreduce_fusion,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=240, stage="nightly", runner_config="8-gpu-b200")

OPS = (
    "communication.trtllm_moe_allreduce_fusion",
    "communication.trtllm_moe_finalize_allreduce_fusion",
)
HIDDEN = cake_comm.MOE_AR_HIDDEN
MAX_TOKENS = 128
EPS = 1e-5


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.communication:")


def _modules_available() -> bool:
    return cake_comm.flashinfer_module_available(
        cake_comm.FI_AR_MODULE,
        cake_comm.FI_MOE_AR_JIT_MODULE,
        cake_comm.FI_MOE_FINALIZE_JIT_MODULE,
    )


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.cuda.current_device())
    if not _modules_available():
        pytest.skip("installed FlashInfer lacks the Cake MoE all-reduce JIT modules")
    if torch.cuda.get_device_capability(device) not in cake_comm.ARCHS:
        pytest.skip("Cake MoE all-reduce is built for sm_100a / sm_103a")
    return device


def test_supports_rejects_bad_inputs(monkeypatch):
    device = _cuda_or_skip()
    tok = torch.empty(16, HIDDEN, device=device, dtype=torch.bfloat16)
    for ws in cake_comm.MOE_AR_WORLD_SIZES:
        assert cake_comm.supports_trtllm_moe_allreduce_fusion(
            tok, world_size=ws, hidden_dim=HIDDEN
        )
        assert cake_comm.supports_trtllm_moe_finalize_allreduce_fusion(
            tok, world_size=ws
        )
    assert cake_comm.supports_trtllm_moe_allreduce_fusion(
        tok.half(), world_size=2, hidden_dim=HIDDEN
    )
    assert not cake_comm.supports_trtllm_moe_allreduce_fusion(
        tok, world_size=3, hidden_dim=HIDDEN
    )
    assert not cake_comm.supports_trtllm_moe_allreduce_fusion(
        tok, world_size=16, hidden_dim=HIDDEN
    )
    assert not cake_comm.supports_trtllm_moe_allreduce_fusion(
        tok, world_size=8, hidden_dim=4096
    )
    assert not cake_comm.supports_trtllm_moe_allreduce_fusion(
        tok[:, :4096], world_size=8, hidden_dim=4096
    )
    assert not cake_comm.supports_trtllm_moe_allreduce_fusion(
        tok.float(), world_size=8, hidden_dim=HIDDEN
    )
    assert not cake_comm.supports_trtllm_moe_allreduce_fusion(
        tok.cpu(), world_size=8, hidden_dim=HIDDEN
    )
    assert not cake_comm.supports_trtllm_moe_finalize_allreduce_fusion(
        tok, world_size=6
    )
    assert not cake_comm.supports_trtllm_moe_finalize_allreduce_fusion(
        tok[:, :4096], world_size=8
    )
    assert not cake_comm.supports_trtllm_moe_finalize_allreduce_fusion(
        tok.float(), world_size=8
    )
    assert not cake_comm.supports_trtllm_moe_finalize_allreduce_fusion(
        tok.t().contiguous().t(), world_size=8
    )
    monkeypatch.setattr(cake_comm, "flashinfer_module_available", lambda *a: False)
    assert not cake_comm.supports_trtllm_moe_allreduce_fusion(
        tok, world_size=8, hidden_dim=HIDDEN
    )
    assert not cake_comm.supports_trtllm_moe_finalize_allreduce_fusion(
        tok, world_size=8
    )


# ---------------------------------------------------------------------------
# Multi-rank parity (torchrun workers only; TRT-LLM IPC workspace)
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


_WORKSPACE = {}


def _ipc_workspace(group: dist.ProcessGroup):
    """One TRT-LLM IPC workspace per torchrun worker, destroyed at exit."""
    key = id(group)
    if key not in _WORKSPACE:
        from flashinfer import comm as fi_comm

        handles, ptrs = fi_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
            dist.get_rank(group),
            dist.get_world_size(group),
            MAX_TOKENS,
            HIDDEN,
            group=group,
        )

        def _destroy():
            torch.cuda.synchronize()
            fi_comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                handles, group=group
            )

        atexit.register(_destroy)
        _WORKSPACE[key] = ptrs
    return _WORKSPACE[key]


def _multi_rank_setup():
    device = _cuda_or_skip()
    group = _dist_group()
    if group is None or dist.get_world_size(group) not in cake_comm.MOE_AR_WORLD_SIZES:
        pytest.skip("needs 2/4/8 ranks with a TRT-LLM IPC workspace (one node)")
    return device, group, _ipc_workspace(group)


def _bounded(shape, dtype, device, gen, scale=1.0):
    return ((torch.rand(shape, device=device, generator=gen) * 2 - 1) * scale).to(dtype)


def _rank_ordered_sum(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    parts = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, local, group=group)
    acc = parts[0].float()
    for part in parts[1:]:
        acc = (acc + part.float()).to(local.dtype).float()
    return acc


def _rmsnorm(x_f32: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return (
        x_f32 * torch.rsqrt(x_f32.square().mean(-1, keepdim=True) + EPS) * gamma.float()
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("token_num", [1, 64])
def test_moe_allreduce_union_matches_reference(dtype, token_num):
    device, group, workspace_ptrs = _multi_rank_setup()
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    experts = 4
    gen = torch.Generator(device=device).manual_seed(0xCA4E + world * 1000 + rank)
    expert_input = _bounded((experts, token_num, HIDDEN), dtype, device, gen)
    expert_scale = _bounded((experts, token_num), torch.float32, device, gen)
    token_input = _bounded((token_num, HIDDEN), dtype, device, gen)
    residual_in = _bounded((token_num, HIDDEN), dtype, device, gen)
    gamma = (_bounded((HIDDEN,), dtype, device, gen, scale=0.125) + 1).contiguous()
    assert cake_comm.supports_trtllm_moe_allreduce_fusion(
        token_input, world_size=world, hidden_dim=HIDDEN
    )
    moe_out = torch.empty_like(residual_in)
    residual_out = torch.empty_like(residual_in)
    norm_out = torch.empty_like(residual_in)
    cake_trtllm_moe_allreduce_fusion(
        world,
        rank,
        token_num,
        HIDDEN,
        workspace_ptrs,
        False,
        residual_in,
        gamma,
        EPS,
        1.0,
        experts,
        expert_scale,
        expert_input,
        token_input,
        None,
        moe_out,
        residual_out,
        norm_out,
        None,
        None,
    )
    torch.cuda.synchronize()

    local = torch.zeros_like(token_input)
    for e in range(experts):
        contribution = (expert_input[e].float() * expert_scale[e].float()[:, None]).to(
            dtype
        )
        local = (local.float() + contribution.float()).to(dtype)
    local = (local.float() + token_input.float()).to(dtype)
    ar_ref = _rank_ordered_sum(local, group)
    residual_ref = (ar_ref + residual_in.float()).to(dtype)
    norm_ref = _rmsnorm(residual_ref.float(), gamma).to(dtype)
    torch.testing.assert_close(moe_out.float(), ar_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        residual_out.float(), residual_ref.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(norm_out.float(), norm_ref.float(), atol=1e-2, rtol=1e-2)
    dist.barrier(group=group)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("token_num", [1, 64])
def test_moe_finalize_allreduce_matches_reference(dtype, token_num):
    device, group, workspace_ptrs = _multi_rank_setup()
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    top_k = 2
    gen = torch.Generator(device=device).manual_seed(0xF1A1 + world * 1000 + rank)
    permuted_rows = token_num * top_k + 8  # padded expert buffer
    allreduce_in = _bounded((permuted_rows, HIDDEN), dtype, device, gen)
    perm = torch.randperm(permuted_rows, device=device, generator=gen)[
        : token_num * top_k
    ]
    expanded_idx = perm.to(torch.int32).view(token_num, top_k).contiguous()
    expert_scale = torch.rand((token_num, top_k), device=device, generator=gen).to(
        dtype
    )
    shared = _bounded((token_num, HIDDEN), dtype, device, gen)
    residual_in = _bounded((token_num, HIDDEN), dtype, device, gen)
    norm_weight = (
        _bounded((HIDDEN,), dtype, device, gen, scale=0.125) + 1
    ).contiguous()
    assert cake_comm.supports_trtllm_moe_finalize_allreduce_fusion(
        residual_in, world_size=world
    )
    residual_out = torch.empty_like(residual_in)
    norm_out = torch.empty_like(residual_in)
    cake_trtllm_moe_finalize_allreduce_fusion(
        allreduce_in,
        residual_in,
        norm_weight,
        expanded_idx,
        norm_out,
        residual_out,
        None,
        None,
        workspace_ptrs,
        False,
        rank,
        world,
        EPS,
        shared,
        expert_scale,
        None,
    )
    torch.cuda.synchronize()

    rows = allreduce_in.float()[expanded_idx.long()]  # [T, top_k, H]
    local = (rows * expert_scale.float()[:, :, None]).sum(1) + shared.float()
    ar_ref = _rank_ordered_sum(local.to(dtype), group)
    residual_ref = (ar_ref + residual_in.float()).to(dtype)
    norm_ref = _rmsnorm(residual_ref.float(), norm_weight).to(dtype)
    # The finalize accumulates the top-k rows in FP32 before the all-reduce;
    # the reference rounds once per rank, so allow two 16-bit ulps.
    torch.testing.assert_close(
        residual_out.float(), residual_ref.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(norm_out.float(), norm_ref.float(), atol=2e-2, rtol=2e-2)
    dist.barrier(group=group)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4, 8))
