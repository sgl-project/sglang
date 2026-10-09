"""Cake MoE expert-parallel all-to-all through sglang.kernels.

Covers the functional ``backend="cake"`` all-to-all (workspace sizing,
initialize, dispatch, combine, sanitize) and the ``flashinfer.moe_ep``
``CakeAlltoAll`` factory. Checks the registry resolution (no GPU); the
in-process ``supports_*`` admission (device architecture, missing Cake
backend); a single-GPU dispatch/combine round trip with ``ep_size`` 1 and 2
(every rank's slice of the workspace lives on one device, as in FlashInfer's
own single-GPU tests, so no NVSHMEM is needed); and a ``CakeAlltoAll``
construction smoke test that needs ``torchrun`` with 2, 4 or 8 ranks and
skips with the reason otherwise.

Usage::

    python test/registered/kernels/ops/communication/test_cake_moe_a2a.py
    python test/registered/kernels/ops/communication/test_cake_moe_a2a.py --num-gpu 2
"""

from __future__ import annotations

import atexit
import os
from typing import Optional

import pytest
import torch
import torch.distributed as dist

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import communication_moe_a2a as cake_a2a
from sglang.kernels.ops.communication.cake import (
    cake_moe_a2a_combine,
    cake_moe_a2a_dispatch,
    cake_moe_a2a_get_workspace_size_per_rank,
    cake_moe_a2a_initialize,
    cake_moe_a2a_sanitize_expert_ids,
    cake_moe_ep_alltoall,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "communication.moe_a2a_get_workspace_size_per_rank",
    "communication.moe_a2a_initialize",
    "communication.moe_a2a_dispatch",
    "communication.moe_a2a_combine",
    "communication.moe_a2a_sanitize_expert_ids",
    "communication.moe_ep_alltoall",
)


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.communication_moe_a2a:")


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda", torch.cuda.current_device())


def _cake_or_skip() -> torch.device:
    device = _cuda_or_skip()
    if not cake_a2a._moe_a2a_cake_backend_available():
        pytest.skip("installed FlashInfer lacks the Cake MoE all-to-all backend")
    if torch.cuda.get_device_capability(device) not in cake_a2a.ARCHS:
        pytest.skip("Cake MoE all-to-all is built for sm_100a / sm_103a")
    return device


def test_supports_checks_device_and_backend(monkeypatch):
    device = _cuda_or_skip()
    expected = (
        cake_a2a._moe_a2a_cake_backend_available()
        and torch.cuda.get_device_capability(device) in cake_a2a.ARCHS
    )
    assert cake_a2a.supports_moe_a2a() is expected
    assert cake_a2a.supports_moe_a2a(device) is expected
    assert cake_a2a.supports_moe_a2a(device.index) is expected
    assert not cake_a2a.supports_moe_a2a(torch.device("cpu"))
    assert not cake_a2a.supports_moe_ep_alltoall(torch.device("cpu"))
    probe = cake_a2a._moe_a2a_cake_backend_available  # the lru_cache-wrapped probe
    monkeypatch.setattr(cake_a2a, "_moe_a2a_cake_backend_available", lambda: False)
    assert not cake_a2a.supports_moe_a2a(device)
    assert not cake_a2a.supports_moe_ep_alltoall(device)
    # With the FlashInfer module reported missing, the real (uncached) probe is False.
    monkeypatch.setattr(cake_a2a, "flashinfer_module_available", lambda *a: False)
    probe.cache_clear()
    try:
        assert not probe.__wrapped__()
    finally:
        probe.cache_clear()


# ---------------------------------------------------------------------------
# Single-GPU round trip (all ranks' workspace slices on one device)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ep_size", [1, 2])
@pytest.mark.parametrize("hidden", [256, 1024])
def test_dispatch_combine_round_trip_single_gpu(ep_size, hidden):
    device = _cake_or_skip()
    assert cake_a2a.supports_moe_a2a(device)
    torch.manual_seed(0)
    num_tokens, top_k, num_experts = 32, 1, 4 * ep_size
    payload = torch.randn(
        ep_size * num_tokens, hidden, device=device, dtype=torch.bfloat16
    )
    experts = torch.randint(
        0, num_experts, (ep_size * num_tokens, top_k), device=device, dtype=torch.int32
    )
    payload_bytes = hidden * payload.element_size()
    ids_bytes = top_k * experts.element_size()
    size = cake_moe_a2a_get_workspace_size_per_rank(
        ep_size, num_tokens, payload_bytes + ids_bytes, payload_bytes
    )
    assert isinstance(size, int) and size > 0
    workspace = torch.zeros(ep_size, size, dtype=torch.uint8, device=device)
    metainfo = [
        cake_moe_a2a_initialize(workspace, rank, ep_size, num_tokens)
        for rank in range(ep_size)
    ]
    torch.cuda.synchronize()

    streams = [torch.cuda.Stream() for _ in range(ep_size)]
    recv, recv_ids, offsets = [], [], []
    for rank in range(ep_size):
        with torch.cuda.stream(streams[rank]):
            sl = slice(rank * num_tokens, (rank + 1) * num_tokens)
            out, offset, _ = cake_moe_a2a_dispatch(
                experts[sl].contiguous(),
                [payload[sl].contiguous(), experts[sl].contiguous()],
                workspace,
                metainfo[rank],
                num_tokens,
                rank,
                ep_size,
                top_k,
                num_experts,
            )
            recv.append(out[0])
            recv_ids.append(out[1])
            offsets.append(offset)
    torch.cuda.synchronize()

    # Experts are owned contiguously: expert e lives on rank e // (num_experts / ep_size).
    experts_per_rank = num_experts // ep_size
    target_rank = experts[:, 0] // experts_per_rank
    for rank in range(ep_size):
        got = recv[rank].flatten(end_dim=1)
        got = got[got.any(dim=1)]
        ref = payload[target_rank == rank]
        got, _ = torch.sort(got.float(), dim=0)
        ref, _ = torch.sort(ref.float(), dim=0)
        torch.testing.assert_close(got, ref, atol=0, rtol=0)

    combined = [None] * ep_size
    for rank in range(ep_size):
        with torch.cuda.stream(streams[rank]):
            combined[rank] = cake_moe_a2a_combine(
                recv[rank].clone(),
                num_tokens,
                workspace,
                metainfo[rank],
                num_tokens,
                rank,
                ep_size,
                top_k,
                offsets[rank],
            )
    torch.cuda.synchronize()
    for rank in range(ep_size):
        sl = slice(rank * num_tokens, (rank + 1) * num_tokens)
        # Identity "expert": combine returns each token's own payload bitwise.
        torch.testing.assert_close(combined[rank], payload[sl], atol=0, rtol=0)

    # Sanitize the received expert-id payload: slots that no token was routed
    # into become invalid_expert_id; routed slots keep their (local) experts.
    for rank in range(ep_size):
        ids = recv_ids[rank]
        cake_moe_a2a_sanitize_expert_ids(ids, workspace, metainfo[rank], rank, -1)
        torch.cuda.synchronize()
        routed_here = int((target_rank == rank).sum())
        valid = ids[ids != -1]
        assert valid.numel() == routed_here * top_k
        assert bool(((valid // experts_per_rank) == rank).all())


# ---------------------------------------------------------------------------
# moe_ep CakeAlltoAll (torchrun workers only)
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


def test_moe_ep_cake_alltoall_round_trip():
    device = _cake_or_skip()
    group = _dist_group()
    if group is None or dist.get_world_size(group) not in (2, 4, 8):
        pytest.skip("needs 2/4/8 ranks with NVSHMEM / NVLink one-sided memory")
    if not cake_a2a.supports_moe_ep_alltoall(device):
        pytest.skip("installed FlashInfer lacks flashinfer.moe_ep CakeAlltoAll")
    from flashinfer.moe_ep import BootstrapConfig, MoEEpCommParams

    rank, world = dist.get_rank(group), dist.get_world_size(group)
    # top_k=1 so the identity-expert round trip returns each token once;
    # combine sums the expert outputs of all top-k target ranks.
    hidden, top_k, tokens = 1024, 1, 16
    params = MoEEpCommParams(
        num_experts=8 * world,
        top_k=top_k,
        max_tokens_per_rank=tokens,
        hidden_size=hidden,
        dtype=torch.bfloat16,
    )
    bootstrap = BootstrapConfig(
        world_size=world, rank=rank, process_group=group, device=device.index
    )
    comm = cake_moe_ep_alltoall(bootstrap, params)
    try:
        assert type(comm).is_platform_supported()
        assert comm.alltoall_backend == "cake"
        torch.manual_seed(300 + rank)
        hidden_states = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
        topk_ids = torch.stack(
            [
                torch.randperm(params.num_experts, device=device)[:top_k]
                for _ in range(tokens)
            ]
        ).to(torch.int32)
        topk_weights = torch.full((tokens, top_k), 1.0 / top_k, device=device)
        result = comm.dispatch(hidden_states, topk_ids, topk_weights)
        assert result.hidden_states.shape[-1] == hidden
        assert result.topk_ids.shape[-1] == top_k
        # Identity expert with top_k=1: combine hands each token back unchanged.
        combined = comm.combine(result.hidden_states)
        torch.cuda.synchronize()
        assert tuple(combined.shape) == (tokens, hidden)
        torch.testing.assert_close(
            combined.float(), hidden_states.float(), atol=1e-2, rtol=1e-2
        )
        dist.barrier(group=group)
    finally:
        torch.cuda.synchronize()
        comm.destroy()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(1, 2, 4, 8))
