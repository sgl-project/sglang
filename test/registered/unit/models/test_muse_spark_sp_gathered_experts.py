"""Muse Spark layernorm SP without an A2A backend reproduces the replicated MoE."""

import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    run_distributed_test,
)

maybe_stub_sgl_kernel()

from sglang.srt.layers import layernorm_sp
from sglang.srt.models.muse_spark import MuseSpark1xMoE
from sglang.srt.runtime_context import get_parallel

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_HIDDEN, _EXPERTS, _TOP_K = 32, 8, 3


class _GlooTpGroup:
    """The GroupCoordinator collectives layernorm SP calls, over a gloo group."""

    def __init__(self, group):
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.rank_in_group = dist.get_rank(group)

    def all_gather_into_tensor(self, output, input_):
        dist.all_gather_into_tensor(output, input_, group=self.group)

    def reduce_scatter_tensor(self, output, input_):
        dist.reduce_scatter_tensor(output, input_, group=self.group)


class _EpLocalExperts:
    """This rank's experts, row-local like the Marlin runner: a partial sum over
    the routes to its own experts, which the TP group then sums."""

    def __init__(self, weights, *, rank, world_size):
        per_rank = _EXPERTS // world_size
        self.local = range(rank * per_rank, (rank + 1) * per_rank)
        self.weights = weights
        self.post_expert_norm = object()
        self.rows_seen = []

    def __call__(self, hidden, topk_weights, topk_ids, post_expert_norm):
        assert post_expert_norm is self.post_expert_norm
        # The experts must get what the router emits: contiguous int32 ids and
        # FP32 weights, not strided views of the packed gather.
        assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
        assert topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
        self.rows_seen.append(hidden.shape[0])
        out = torch.zeros_like(hidden)
        for expert in self.local:
            gain = (topk_weights * (topk_ids == expert)).sum(-1, keepdim=True)
            out += gain * (hidden @ self.weights[expert].T)
        return out


def _pre_expert_norm(hidden):
    # Exact and not idempotent, so a skipped or repeated norm changes the output.
    return hidden * 2


def _sp_moe(experts) -> MuseSpark1xMoE:
    # Only the attributes _forward_sp_experts reads; the constructor needs a model.
    moe = MuseSpark1xMoE.__new__(MuseSpark1xMoE)
    nn.Module.__init__(moe)
    moe.pre_expert_norm = _pre_expert_norm
    moe.use_megamoe = False
    moe.sp_dispatcher = None
    moe.experts = experts
    return moe


def _routes(num_tokens):
    # Small integers and weights in eighths: every product and partial sum is
    # exact in FP32, so any reduction order gives the same bits.
    g = torch.Generator().manual_seed(num_tokens)
    hidden = torch.randint(-3, 4, (num_tokens, _HIDDEN), generator=g).float()
    ids = torch.stack(
        [torch.randperm(_EXPERTS, generator=g)[:_TOP_K] for _ in range(num_tokens)]
    ).to(torch.int32)
    weights = torch.randint(1, 8, (num_tokens, _TOP_K), generator=g).float() / 8
    return hidden, weights, ids


def _check_matches_replicated(*, tp_group, num_tokens):
    rank, world_size = tp_group.rank_in_group, tp_group.world_size
    hidden, weights, ids = _routes(num_tokens)
    g = torch.Generator().manual_seed(7)
    expert_weights = torch.randint(
        -3, 4, (_EXPERTS, _HIDDEN, _HIDDEN), generator=g
    ).float()

    replicated_experts = _EpLocalExperts(
        expert_weights, rank=rank, world_size=world_size
    )
    replicated = replicated_experts(
        _pre_expert_norm(hidden), weights, ids, replicated_experts.post_expert_norm
    )
    dist.all_reduce(replicated, group=tp_group.group)

    experts = _EpLocalExperts(expert_weights, rank=rank, world_size=world_size)
    local_hidden = layernorm_sp.sp_entry_scatter(hidden)
    shard = _sp_moe(experts)._forward_sp_experts(
        local_hidden,
        layernorm_sp.shard_token_rows(weights),
        layernorm_sp.shard_token_rows(ids),
        None,
    )
    padded = -(-num_tokens // world_size) * world_size
    assert experts.rows_seen == [padded], experts.rows_seen
    assert shard.shape == local_hidden.shape
    gathered = layernorm_sp.sp_exit_gather(shard, num_tokens=num_tokens)
    assert torch.equal(gathered, replicated)


def _check_topk_gather_is_lossless(*, tp_group):
    rank, world_size = tp_group.rank_in_group, tp_group.world_size
    g = torch.Generator().manual_seed(world_size)
    weights = torch.randn((4 * world_size, _TOP_K), generator=g)
    ids = torch.randint(0, _EXPERTS, weights.shape, generator=g, dtype=torch.int32)
    rows = slice(4 * rank, 4 * (rank + 1))
    got_weights, got_ids = layernorm_sp.all_gather_topk_rows(
        topk_weights=weights[rows], topk_ids=ids[rows]
    )
    assert torch.equal(got_weights.view(torch.int32), weights.view(torch.int32))
    assert torch.equal(got_ids, ids)


def _worker(rank: int) -> None:
    # The timeout turns a desynchronized collective into an error that
    # run_distributed_test reports, rather than a hung CI job.
    tp_group = _GlooTpGroup(
        dist.new_group(backend="gloo", timeout=timedelta(seconds=60))
    )
    with get_parallel().override(
        tp_size=tp_group.world_size, tp_rank=rank, tp_group=tp_group
    ):
        # A multiple of TP, one that needs entry-scatter padding, and fewer
        # tokens than ranks.
        world_size = tp_group.world_size
        for num_tokens in (8 * world_size, 8 * world_size + 3, 1):
            _check_matches_replicated(tp_group=tp_group, num_tokens=num_tokens)
        _check_topk_gather_is_lossless(tp_group=tp_group)


class TestMuseSparkGatheredExperts(CustomTestCase):
    def test_sp_without_a2a_matches_the_replicated_moe(self):
        """Each rank runs its experts on the all-gathered tokens and
        reduce-scatters the partial sums; after the exit gather this must equal
        the replicated MoE's all-reduced output bit for bit, padding included."""
        for world_size in (2, 4):
            with self.subTest(world_size=world_size):
                run_distributed_test(_worker, world_size=world_size, backend="gloo")


if __name__ == "__main__":
    unittest.main()
