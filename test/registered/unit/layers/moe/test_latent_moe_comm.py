"""Latent-width MoE exchange vs the hidden-width dataflow on 16 simulated ranks
(2 DP groups x 8 attention-TP ranks, EP16) with thread-backed collectives."""

import contextlib
import threading
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import latent_moe_comm as lmc
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

DP_SIZE, ATTN_TP_SIZE = 2, 8
WORLD = DP_SIZE * ATTN_TP_SIZE
HIDDEN, LATENT, NUM_EXPERTS, TOP_K = 64, 16, 32, 4
EXPERT_INTER, SHARED_INTER = 24, 48


class _SimGroup:
    """Rendezvous for one process group; ranks run as threads."""

    def __init__(self, size):
        self.barrier = threading.Barrier(size, timeout=60)
        self.slots = [None] * size

    def exchange(self, rank, value):
        self.slots[rank] = value
        self.barrier.wait()
        values = list(self.slots)
        self.barrier.wait()
        return values


class _SimNcclGroup:
    """GroupCoordinator collectives with NCCL shape rules; records calls."""

    def __init__(self, group, rank, calls):
        self.group, self.rank, self.calls = group, rank, calls

    def _sum(self, x):
        return torch.stack(self.group.exchange(self.rank, x)).sum(0)

    def all_gather_into_tensor(self, output, input):
        self.calls.add("all_gather_into_tensor")
        values = self.group.exchange(self.rank, input)
        assert len({v.shape for v in values}) == 1, "uneven all_gather_into_tensor"
        output.copy_(torch.cat(values))

    def all_gatherv(self, input, sizes, output):
        self.calls.add("all_gatherv")
        values = self.group.exchange(self.rank, input)
        assert [v.shape[0] for v in values] == sizes
        output.copy_(torch.cat(values))

    def reduce_scatter_tensor(self, output, input):
        self.calls.add("reduce_scatter_tensor")
        total, rows = self._sum(input), output.shape[0]
        assert total.shape[0] == rows * len(self.group.slots), "uneven reduce_scatter"
        output.copy_(total[self.rank * rows : (self.rank + 1) * rows])

    def reduce_scatterv(self, input, output, sizes):
        self.calls.add("reduce_scatterv")
        start = sum(sizes[: self.rank])
        output.copy_(self._sum(input)[start : start + sizes[self.rank]])

    def all_reduce(self, x):
        return self._sum(x)


def _make_weights(*, dtype, with_shared, seed):
    g = torch.Generator().manual_seed(seed)

    def randn(*shape, scale):
        return (torch.randn(*shape, generator=g, dtype=torch.float64) * scale).to(dtype)

    weights = {
        "gate": randn(NUM_EXPERTS, HIDDEN, scale=HIDDEN**-0.5),
        "down": randn(LATENT, HIDDEN, scale=HIDDEN**-0.5),
        "up": randn(HIDDEN, LATENT, scale=LATENT**-0.5),
        "w_up": randn(NUM_EXPERTS, EXPERT_INTER, LATENT, scale=LATENT**-0.5),
        "w_down": randn(NUM_EXPERTS, LATENT, EXPERT_INTER, scale=EXPERT_INTER**-0.5),
        "norm": (1 + torch.rand(LATENT, generator=g, dtype=torch.float64)).to(dtype),
    }
    if with_shared:
        weights["shared_up"] = randn(SHARED_INTER, HIDDEN, scale=HIDDEN**-0.5)
        weights["shared_down"] = randn(HIDDEN, SHARED_INTER, scale=SHARED_INTER**-0.5)
    return weights


def _route(weights, h):
    """Sigmoid top-k, computed per row so it is batching-invariant."""
    scores = (h.double()[:, None, :] * weights["gate"].double()[None]).sum(-1).sigmoid()
    topk_weights, topk_ids = scores.topk(TOP_K)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    return topk_weights.float(), topk_ids.int()


def _linear(x, w):
    return (x.double() @ w.double().t()).to(x.dtype)


def _rms_norm(x, w):
    xd = x.double()
    return (xd * (xd.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt() * w.double()).to(
        x.dtype
    )


def _local_experts(weights, ep_rank, latent, topk_weights, topk_ids):
    per_rank = NUM_EXPERTS // WORLD
    out = torch.zeros(latent.shape, dtype=torch.float64)
    for expert in range(ep_rank * per_rank, (ep_rank + 1) * per_rank):
        rows, slot = (topk_ids == expert).nonzero(as_tuple=True)
        act = _linear(latent[rows], weights["w_up"][expert]).double().relu().square()
        expert_out = _linear(act.to(latent.dtype), weights["w_down"][expert])
        out.index_add_(0, rows, expert_out.double() * topk_weights[rows, slot, None])
    return out.to(latent.dtype)


def _shared_partial(weights, h, *, shard, num_shards):
    cols = slice(
        shard * SHARED_INTER // num_shards, (shard + 1) * SHARED_INTER // num_shards
    )
    act = _linear(h, weights["shared_up"][cols]).double().relu().square()
    return _linear(act.to(h.dtype), weights["shared_down"][:, cols])


def _hidden_width_rank(rank, tp_group, weights, dp_hidden, record, latent_norm):
    """Today's dataflow: gather hidden, route every row, TP16 shared expert.
    Without a latent norm (Nemotron-H) the up projection runs on partials and
    the hidden-width sum completes them; with one (Kimi K3) the latent sum
    must complete before the norm."""
    gathered = torch.cat(dp_hidden)
    topk_weights, topk_ids = _route(weights, gathered)
    record["topk_ids"] = topk_ids
    latent = _local_experts(
        weights, rank, _linear(gathered, weights["down"]), topk_weights, topk_ids
    )
    if latent_norm:
        latent = _rms_norm(tp_group.all_reduce(latent), weights["norm"])
        routed = _linear(latent, weights["up"])
        output = routed
        if "shared_up" in weights:
            output = output + tp_group.all_reduce(
                _shared_partial(weights, gathered, shard=rank, num_shards=WORLD)
            )
    else:
        partial = _linear(latent, weights["up"])
        if "shared_up" in weights:
            partial = partial + _shared_partial(
                weights, gathered, shard=rank, num_shards=WORLD
            )
        output = tp_group.all_reduce(partial)
    start = sum(h.shape[0] for h in dp_hidden[: rank // ATTN_TP_SIZE])
    return output[start : start + dp_hidden[rank // ATTN_TP_SIZE].shape[0]]


def _latent_rank(rank, weights, dp_hidden, latent_norm):
    def up_proj(latent, addend):
        if latent_norm:
            latent = _rms_norm(latent, weights["norm"])
        return addend + _linear(latent, weights["up"])

    shared = None
    if "shared_up" in weights:
        shard = rank % ATTN_TP_SIZE
        shared = lambda h: _shared_partial(  # noqa: E731
            weights, h, shard=shard, num_shards=ATTN_TP_SIZE
        )

    return lmc.latent_moe_forward(
        dp_hidden[rank // ATTN_TP_SIZE],
        layer_id=0,
        gate=lambda h: h,
        topk=lambda h, _: StandardTopKOutput(*_route(weights, h), None),
        down_proj=lambda h: _linear(h, weights["down"]),
        experts=lambda latent, out: _local_experts(
            weights, rank, latent, out.topk_weights, out.topk_ids
        ),
        up_proj=up_proj,
        shared_experts=shared,
        alt_stream=None,
    )


def _run_world(fn):
    tp_group = _SimGroup(WORLD)
    attn_groups = [_SimGroup(ATTN_TP_SIZE) for _ in range(DP_SIZE)]
    outputs, records, errors = [None] * WORLD, [{} for _ in range(WORLD)], []

    def run(rank):
        try:
            outputs[rank] = fn(
                rank, tp_group, attn_groups[rank // ATTN_TP_SIZE], records[rank]
            )
        except BaseException as e:  # unblock peers waiting in a collective
            errors.append(e)
            for group in (tp_group, *attn_groups):
                group.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(WORLD)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return outputs, records


def _run_latent(weights, dp_hidden, *, max_len_padding, calls, latent_norm=False):
    state = threading.local()

    def rank_fn(rank, tp_group, attn_group, record):
        state.parallel = types.SimpleNamespace(
            tp_size=WORLD,
            tp_rank=rank,
            attn_tp_size=ATTN_TP_SIZE,
            attn_tp_rank=rank % ATTN_TP_SIZE,
            attn_tp_group=_SimNcclGroup(attn_group, rank % ATTN_TP_SIZE, set()),
        )
        state.tp_group = _SimNcclGroup(tp_group, rank, calls)
        state.capturer = types.SimpleNamespace(
            capture=lambda layer_id, topk_indices: record.__setitem__(
                "topk_ids", topk_indices
            )
        )
        return _latent_rank(rank, weights, dp_hidden, latent_norm)

    with (
        patch.object(lmc, "get_parallel", lambda: state.parallel),
        patch.object(lmc, "get_tp_group", lambda: state.tp_group),
        patch.object(lmc, "get_global_experts_capturer", lambda: state.capturer),
        patch.object(lmc, "is_dp_max_padding", return_value=max_len_padding),
        patch.object(
            lmc,
            "get_dp_global_num_tokens",
            return_value=[h.shape[0] for h in dp_hidden],
        ),
        patch.object(
            lmc, "use_symmetric_memory", lambda *a, **k: contextlib.nullcontext()
        ),
        patch.object(lmc, "is_allocation_symmetric", return_value=False),
    ):
        return _run_world(rank_fn)


def _run_hidden_width(weights, dp_hidden, latent_norm=False):
    def rank_fn(rank, tp_group, _, record):
        group = _SimNcclGroup(tp_group, rank, set())
        return _hidden_width_rank(rank, group, weights, dp_hidden, record, latent_norm)

    return _run_world(rank_fn)


def _dp_hidden(dp_num_tokens, *, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(n, HIDDEN, generator=g, dtype=torch.float64).to(dtype)
        for n in dp_num_tokens
    ]


class TestLatentMoEForward(CustomTestCase):
    def _compare(
        self,
        dp_num_tokens,
        *,
        max_len_padding,
        with_shared=True,
        latent_norm=False,
        seed=0,
    ):
        weights = _make_weights(dtype=torch.float64, with_shared=with_shared, seed=seed)
        dp_hidden = _dp_hidden(dp_num_tokens, dtype=torch.float64, seed=seed + 1)
        calls = set()
        new_out, new_rec = _run_latent(
            weights,
            dp_hidden,
            max_len_padding=max_len_padding,
            calls=calls,
            latent_norm=latent_norm,
        )
        ref_out, ref_rec = _run_hidden_width(weights, dp_hidden, latent_norm)
        for rank in range(WORLD):
            # Same routing, and gathered ids keep the DP-gathered row layout.
            self.assertTrue(
                torch.equal(new_rec[rank]["topk_ids"], ref_rec[rank]["topk_ids"])
            )
            torch.testing.assert_close(
                new_out[rank], ref_out[rank], rtol=1e-10, atol=1e-10
            )
        return calls

    def test_max_len_uses_static_collectives(self):
        calls = self._compare([32, 32], max_len_padding=True)
        self.assertEqual(calls, {"all_gather_into_tensor", "reduce_scatter_tensor"})

    def test_sum_len_uneven_uses_variable_collectives(self):
        # DP1's ranks own one row each.
        calls = self._compare([48, 8], max_len_padding=False, seed=3)
        self.assertEqual(calls, {"all_gatherv", "reduce_scatterv"})

    def test_sum_len_even_uses_static_collectives(self):
        calls = self._compare([16, 16], max_len_padding=False, seed=4)
        self.assertEqual(calls, {"all_gather_into_tensor", "reduce_scatter_tensor"})

    def test_idle_dp_group(self):
        # An idle DP group still joins the collectives with zero rows.
        calls = self._compare([16, 0], max_len_padding=False, seed=6)
        self.assertEqual(calls, {"all_gatherv", "reduce_scatterv"})

    def test_without_shared_expert(self):
        self._compare([16, 24], max_len_padding=False, with_shared=False, seed=5)

    def test_latent_norm_reads_complete_rows(self):
        # Kimi K3 norms the latent sum: the reduce-scatter must hand the norm
        # whole rows, as the default path's latent all-reduce does.
        self._compare([24, 16], max_len_padding=False, latent_norm=True, seed=9)
        self._compare([32, 32], max_len_padding=True, latent_norm=True, seed=10)

    def test_bf16_error_matches_hidden_width_path(self):
        weights = _make_weights(dtype=torch.bfloat16, with_shared=True, seed=7)
        dp_hidden = _dp_hidden([32, 16], dtype=torch.bfloat16, seed=8)
        new_out, _ = _run_latent(weights, dp_hidden, max_len_padding=False, calls=set())
        ref_out, _ = _run_hidden_width(weights, dp_hidden)
        exact, _ = _run_hidden_width(
            {k: v.double() for k, v in weights.items()}, [h.double() for h in dp_hidden]
        )
        for rank in range(0, WORLD, ATTN_TP_SIZE):
            scale = exact[rank].abs().max()
            ref_err = (ref_out[rank].double() - exact[rank]).abs().max() / scale
            new_err = (new_out[rank].double() - exact[rank]).abs().max() / scale
            self.assertLess(new_err, max(2 * ref_err, 2**-7))


class TestDispatchPayload(CustomTestCase):
    def test_round_trip_is_bit_exact(self):
        g = torch.Generator().manual_seed(0)
        for dtype in (torch.bfloat16, torch.float32):
            latent = torch.randn(5, LATENT, generator=g).to(dtype)
            topk_ids = torch.randint(0, 512, (5, TOP_K), generator=g).int()
            topk_weights = torch.rand(5, TOP_K, generator=g)
            topk_weights[0] = torch.tensor([-0.0, 1e-40, float("nan"), 3.0])
            packed = lmc.pack_dispatch_payload(
                latent=latent, topk_weights=topk_weights, topk_ids=topk_ids
            )
            out_latent, out_weights, out_ids = lmc.unpack_dispatch_payload(
                packed, latent_dim=LATENT, top_k=TOP_K
            )
            self.assertTrue(torch.equal(out_latent, latent))
            self.assertTrue(torch.equal(out_ids, topk_ids))
            self.assertTrue(
                torch.equal(
                    out_weights.view(torch.int32), topk_weights.view(torch.int32)
                )
            )


if __name__ == "__main__":
    unittest.main()
