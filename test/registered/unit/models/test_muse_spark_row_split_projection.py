"""Muse Spark's prefill row split of the pooler and unpooler reproduces the replicated MoE."""

import time
import unittest
from datetime import timedelta
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    maybe_stub_sgl_kernel,
)

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import muse_spark
from sglang.srt.models.muse_spark import MuseSpark1xMoE, MuseSpark1xRouter
from sglang.srt.runtime_context import get_forward, get_parallel

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# The pooler narrows the FFN input for the experts and the unpooler widens it back.
_HIDDEN, _EXPERT_HIDDEN, _EXPERTS, _TOP_K = 8, 4, 8, 2
_MIN_TOKENS = muse_spark._ROW_SPLIT_MIN_TOKENS
# A rank that desyncs must fail the test, not stall CI on a hung collective.
_COLLECTIVE_TIMEOUT = timedelta(seconds=60)
_JOIN_TIMEOUT_S = 180


class _GlooTpGroup:
    """The GroupCoordinator collectives the MoE calls, over the gloo world,
    recording each so the test sees which path a pass took."""

    def __init__(self):
        self.world_size = dist.get_world_size()
        self.rank_in_group = dist.get_rank()
        self.calls = []

    def all_reduce(self, input_):
        self.calls.append("all_reduce")
        output = input_.clone()
        dist.all_reduce(output)
        return output

    def all_gather_into_tensor(self, output, input_):
        self.calls.append("all_gather")
        dist.all_gather_into_tensor(output, input_)

    def reduce_scatter_tensor(self, output, input_):
        self.calls.append("reduce_scatter")
        dist.reduce_scatter_tensor(output, input_)


class _EpLocalExperts:
    """This rank's experts: a partial sum over the routes to its own experts,
    which the TP group then sums. Route membership stands in for the route
    weights, so every value stays a small integer."""

    def __init__(self, weights, *, rank, world_size):
        per_rank = _EXPERTS // world_size
        self.local = range(rank * per_rank, (rank + 1) * per_rank)
        self.weights = weights
        self.post_expert_norm = None

    def __call__(self, hidden, topk_weights, topk_ids, post_expert_norm, **_kwargs):
        out = torch.zeros_like(hidden)
        for expert in self.local:
            routed = (topk_ids == expert).any(dim=-1, keepdim=True)
            out += routed * (hidden @ self.weights[expert].T)
        return out


def _ints(shape, generator) -> torch.Tensor:
    # With these widths every product and partial sum stays within 256, exact
    # in bf16, so any reduction order or GEMM blocking gives the same bits.
    return torch.randint(-1, 2, shape, generator=generator).to(torch.bfloat16)


def _linear(weight: torch.Tensor) -> nn.Linear:
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight = nn.Parameter(weight, requires_grad=False)
    return linear


def _moe(*, rank: int, world_size: int) -> MuseSpark1xMoE:
    # Only the attributes the TP forward reads; the constructor needs a model.
    # One seed on every rank, since the projections and the router are replicated.
    g = torch.Generator().manual_seed(0)
    moe = MuseSpark1xMoE.__new__(MuseSpark1xMoE)
    nn.Module.__init__(moe)
    moe.layer_idx = 0
    moe.tp_size = world_size
    moe.moe_tp_size = 1
    moe.top_k = _TOP_K
    moe.route_scale = 1.0
    moe.router_multiplier = None
    moe.route_eps = 1e-15
    moe.router_in_fp32 = True
    moe.nvfp4_checkpoint = False
    moe.finalize_ar_workspace = None
    moe.unpool_gather_boundary = None
    moe.row_split_supported = True
    moe.gate = MuseSpark1xRouter(_HIDDEN, _EXPERTS, dtype=torch.float32)
    moe.gate.weight.copy_(torch.randn((_EXPERTS, _HIDDEN), generator=g))
    moe.pre_expert_proj = _linear(_ints((_EXPERT_HIDDEN, _HIDDEN), g))
    moe.post_expert_proj = _linear(_ints((_HIDDEN, _EXPERT_HIDDEN), g))
    moe.pre_expert_norm = None
    moe.experts = _EpLocalExperts(
        _ints((_EXPERTS, _EXPERT_HIDDEN, _EXPERT_HIDDEN), g),
        rank=rank,
        world_size=world_size,
    )
    return moe


def _forward(moe, hidden, *, rank, world_size, forward_mode):
    group = _GlooTpGroup()
    forward_batch = Mock(spec=ForwardBatch, forward_mode=forward_mode)
    with get_parallel().override(tp_size=world_size, tp_rank=rank, tp_group=group):
        out = moe._forward_tp(hidden, None, forward_batch)
    return out, group.calls


def _check_split_matches_replicated(*, rank, world_size, num_tokens):
    moe = _moe(rank=rank, world_size=world_size)
    hidden = _ints((num_tokens, _HIDDEN), torch.Generator().manual_seed(num_tokens))
    split, split_calls = _forward(
        moe, hidden, rank=rank, world_size=world_size, forward_mode=ForwardMode.EXTEND
    )
    # A DFLASH verify batch reaches the same row count inside decode CUDA graphs.
    replicated, replicated_calls = _forward(
        moe,
        hidden,
        rank=rank,
        world_size=world_size,
        forward_mode=ForwardMode.TARGET_VERIFY,
    )
    assert split_calls == ["all_gather", "reduce_scatter", "all_gather"], split_calls
    assert replicated_calls == ["all_reduce"], replicated_calls
    assert torch.equal(split, replicated)


def _worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=_COLLECTIVE_TIMEOUT,
    )
    try:
        # CUDA-only pieces swapped at their boundary: the bf16 GEMM op (F.linear
        # is its CPU route), the symmetric workspace (None, as without multicast)
        # and the device capability (an H200).
        with (
            patch.object(muse_spark, "bf16_gemm_dispatch", F.linear),
            patch.object(muse_spark, "muse_workspace", return_value=None),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            # A multiple of TP, and one whose last rank needs padding rows.
            for num_tokens in (_MIN_TOKENS, _MIN_TOKENS + 1):
                _check_split_matches_replicated(
                    rank=rank, world_size=world_size, num_tokens=num_tokens
                )
    finally:
        dist.destroy_process_group()


def _spawn(world_size: int) -> list:
    ctx = mp.get_context("spawn")
    port = find_available_port(29500)
    procs = [
        ctx.Process(target=_worker, args=(rank, world_size, port))
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + _JOIN_TIMEOUT_S
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join()
    return [p.exitcode for p in procs]


class TestMuseSparkRowSplitProjection(CustomTestCase):
    def test_prefill_split_matches_replicated(self):
        """Through the TP forward, an EXTEND pass from the threshold up gathers
        the pooler's row blocks and turns the unpooler's all-reduce into a
        reduce-scatter and an all-gather; a TARGET_VERIFY pass of the same size
        stays replicated. Both must agree bit for bit, padded M included. The
        inputs are small integers, so this proves the split, padding and gather
        plumbing, not bit-exactness of the GPU kernels."""
        for world_size in (2, 4):
            with self.subTest(world_size=world_size):
                self.assertEqual(_spawn(world_size), [0] * world_size)

    def test_split_skips_sp_and_short_passes(self):
        """Under layernorm SP each rank already holds only its own rows, so
        splitting them again would drop tokens; short passes stay replicated."""
        moe = _moe(rank=0, world_size=2)
        self.assertTrue(moe._uses_row_split(ForwardMode.EXTEND, _MIN_TOKENS))
        self.assertFalse(moe._uses_row_split(ForwardMode.EXTEND, _MIN_TOKENS - 1))
        with get_forward().scoped(sp_active=True):
            self.assertFalse(moe._uses_row_split(ForwardMode.EXTEND, _MIN_TOKENS))


if __name__ == "__main__":
    unittest.main()
