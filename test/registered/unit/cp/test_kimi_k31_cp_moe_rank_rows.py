"""Multi-GPU test of the Kimi-K3.1 latent MoE's output contract under CP-TP group
sharing (``KimiK3MoE(..., cp_shard_rows=True)``): on rows gathered rank-major
over the attention-CP group (the TP group), it returns only this rank's CP rows
of the complete output, through the fused front (reduce-scatter) or the
fallback (complete output, then this rank's rows).

The module is real (router gate, shared experts, latent down/up projections and
norm, the merged fused-front weight, TP collectives); only the routed experts
are a deterministic stand-in for the quantized runner, returning a TP-partial
sum as the real ones do.

Usage::

    python test/registered/unit/cp/test_kimi_k31_cp_moe_rank_rows.py
"""

from __future__ import annotations

import atexit
import logging
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.srt.models import kimi_k3
from sglang.srt.runtime_context import SpawnRanks, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.parallel_groups import publish

register_cuda_ci(est_time=60, stage="base-b", runner_config="2-gpu-large")

HIDDEN, LATENT, EXPERTS = 256, 64, 8


class _RoutedExperts(torch.nn.Module):
    """A TP-partial routed sum: every rank contributes its share."""

    should_fuse_routed_scaling_factor_in_topk = False
    quant_method = None  # not the mxfp4 runner: a dense bf16 front contract
    moe_runner_config = SimpleNamespace(activation="situ")

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, routed_input, topk_output):
        return torch.tanh(routed_input) / dist.get_world_size()


class _TopK(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, hidden_states, router_logits):
        return None


@cache_once
def _moe() -> kimi_k3.KimiK3MoE:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    publish(
        ServerArgs(model_path="dummy", tp_size=world_size),
        role="test",
        ranks=SpawnRanks(world_rank=local_rank),
    )
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(
        world_group=ps._WORLD,
        tp_group=ps._WORLD,
        # CP-TP group sharing: attention CP spans the TP group.
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_tp_group=SimpleNamespace(rank_in_group=0, world_size=1),
        attn_cp_group=ps._WORLD,
        attn_cp_size=world_size,
        attn_cp_rank=local_rank,
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        moe_intermediate_size=128,
        moe_renormalize=True,
        routed_scaling_factor=1.0,
        num_shared_experts=1,
        routed_expert_hidden_size=LATENT,
        num_experts=EXPERTS,
        n_routed_experts=EXPERTS,
        topk_method="noaux_tc",
        num_experts_per_token=2,
        # K3.1's activation: it reads the fused front's strided fp32 gate_up.
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        num_expert_group=1,
        topk_group=1,
        moe_router_activation_func="sigmoid",
        rms_norm_eps=1e-6,
        latent_moe_use_norm=True,
    )
    torch.set_default_dtype(torch.bfloat16)
    with (
        torch.device("cuda"),
        patch.object(kimi_k3, "get_moe_impl_class", return_value=_RoutedExperts),
        patch.object(kimi_k3, "TopK", _TopK),
    ):
        moe = kimi_k3.KimiK3MoE(config)
    torch.set_default_dtype(torch.float32)
    g = torch.Generator(device="cuda").manual_seed(1000 + local_rank)
    for param in moe.parameters():
        param.data.copy_(
            torch.randn(param.shape, generator=g, device="cuda", dtype=torch.float32)
            * 0.05
        )
    if moe.routed_expert_norm is not None:
        moe.routed_expert_norm.weight.data.fill_(1.0)
    moe._merge_front_weights()
    return moe


def _input(num_tokens):
    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    return torch.randn(
        num_tokens, HIDDEN, generator=g, device="cuda", dtype=torch.float32
    ).to(torch.bfloat16)


def _fallback(moe):
    """The same module with the fused front off (the unfused front)."""
    return patch.dict(moe.__dict__, {"_eligible_for_fused_front": False})


@pytest.mark.parametrize("rows_per_rank", [1, 8, 333])
def test_fused_and_fallback_return_this_ranks_rows(rows_per_rank):
    moe = _moe()
    assert moe._eligible_for_fused_front, "the fused front must be exercised"
    rank, size = dist.get_rank(), dist.get_world_size()
    x = _input(rows_per_rank * size)
    full = moe(x)
    with _fallback(moe):
        full_unfused = moe(x)
        cp_fallback = moe(x, cp_shard_rows=True)
    cp_fused = moe(x, cp_shard_rows=True)

    # The ordinary contract: every row, complete, on either front.
    assert full.shape == (rows_per_rank * size, HIDDEN)
    torch.testing.assert_close(full, full_unfused, rtol=2e-2, atol=2e-2)
    # The CP contract: only this rank's rows, every one valid.
    expected = full.tensor_split(size)[rank]
    for out in (cp_fused, cp_fallback):
        assert out.shape == (rows_per_rank, HIDDEN)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


def test_empty_batch_returns_empty_rows():
    moe = _moe()
    x = _input(0)
    assert moe(x, cp_shard_rows=True).shape == (0, HIDDEN)
    with _fallback(moe):
        assert moe(x, cp_shard_rows=True).shape == (0, HIDDEN)


def test_rejects_unequal_shards():
    moe = _moe()
    with pytest.raises(AssertionError):
        moe(_input(dist.get_world_size() * 4 + 1), cp_shard_rows=True)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=[2])
