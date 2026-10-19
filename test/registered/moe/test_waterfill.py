"""Regression tests pinning the Waterfill algorithm (``--enable-waterfill``).

Waterfill routes DeepSeek's always-on shared expert as an extra top-k slot
(8 routed -> 9) and sends it to an EP rank with headroom instead of always to
the token's home rank. It is layered on top of the generic MoE code (TopK,
DeepEP dispatch, DeepseekV2MoE's epilogue), and refactors of that code have
changed its behavior before without touching ``waterfill.py``. These tests pin
the contract end to end so that such changes fail here:

1. **TopK layout.** With Waterfill the router is asked for the routed experts
   only (``top_k - num_fused_shared_experts``, no shared slot, STANDARD output)
   and the balancer appends the shared column afterwards.
2. **ID remap.** DeepEP's per-rank layout gains one shared slot per rank:
   routed ``e -> e + e // (E / W)``, the shared slot of rank ``r`` is
   ``r * (E / W + 1) + E / W``. Routed weights are copied unchanged; rows with no
   valid expert (padding) get ``-1`` and weight 0 in the shared column.
3. **Rank selection.** Static mode: the shared slot is sampled over all ranks in
   proportion to their headroom below the water level (ranks at or above it get
   none); the home rank is preferred 11:10. Dynamic mode restricts the choice to
   the home rank and the ranks the token's routed experts already visit. Small
   batches (< ``MIN_BATCH_FOR_BALANCE``) keep the shared expert local.
4. **Shared weight.** The shared slot carries ``1/routed_scaling_factor`` when
   ``DeepseekV2MoE.forward_deepep`` scales the combined MoE output afterwards
   and ``1.0`` when the router already folded the factor into the routed
   weights (``should_fuse_routed_scaling_factor_in_topk`` quant methods), so
   the shared expert always contributes exactly once.
5. **End to end.** ``DeepseekV2MoE.forward_deepep`` under Waterfill, with the
   real TopK and a linear stand-in for the experts, equals
   ``routed_scaling_factor * sum_i w_i E_i(x) + S(x)`` in both scaling modes.

The Triton kernels are GPU-only, so the whole file needs an accelerator.
"""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.deepep_waterfill_kernels import (
    LOCAL_SHARED_MARKER,
    WaterfillDispatchPlan,
    materialize_waterfill_dispatch_fused,
)
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.waterfill import (
    WaterfillBalancer,
    expand_topk_with_shared_expert,
)
from sglang.srt.model_executor.model_runner_components import moe_ep_setup
from sglang.srt.models import deepseek_v2 as deepseek_v2_module
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

DEVICE = get_device()
RSF = 2.5
NUM_EXPERTS = 256
TOPK = 8
EP_SIZE = 8
EP_RANK = 3
OLD_EPR = NUM_EXPERTS // EP_SIZE  # 32 routed experts per rank
NEW_EPR = OLD_EPR + 1  # + the rank's shared slot
N_GROUP, TOPK_GROUP = 8, 4


def _shared_slot(rank):
    return rank * NEW_EPR + OLD_EPR


def _remap_routed(ids):
    """The DeepEP per-rank-slot remap for routed ids (-1 stays -1)."""
    return torch.where(ids >= 0, ids + ids // OLD_EPR, ids)


def _random_topk(num_tokens, *, seed=0, padded_rows=()):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(NUM_EXPERTS, generator=g)[:TOPK] for _ in range(num_tokens)]
    ).to(torch.int32)
    weights = torch.rand(num_tokens, TOPK, generator=g)
    weights = weights / weights.sum(-1, keepdim=True)
    for r in padded_rows:
        ids[r] = -1
    return ids.to(DEVICE), weights.to(DEVICE)


def _ids_to_ranks(ids):
    return torch.where(ids >= 0, ids // OLD_EPR, torch.full_like(ids, -1))


def _exec_stub(enable_waterfill=True):
    return SimpleNamespace(
        moe=SimpleNamespace(
            enable_waterfill=enable_waterfill,
            ep_num_redundant_experts=0,
            ep_dispatch_algorithm=None,
        )
    )


def _build_topk(*, fold):
    """A DeepSeek-V3-style TopK as DeepseekV2MoE builds it under Waterfill."""
    with patch.object(topk_module, "get_exec", return_value=_exec_stub()):
        return TopK(
            top_k=TOPK + 1,
            layer_id=0,
            use_grouped_topk=True,
            topk_group=TOPK_GROUP,
            num_expert_group=N_GROUP,
            renormalize=True,
            num_fused_shared_experts=1,
            scoring_func="sigmoid",
            correction_bias=torch.zeros(NUM_EXPERTS, device=DEVICE),
            routed_scaling_factor=RSF,
            apply_routed_scaling_factor_on_output=fold,
            output_format=TopKOutputFormat.BYPASSED,  # must be overridden
        )


def _prepare(module, *, rank=EP_RANK):
    """Attach the WaterfillBalancer the way ModelRunner does."""
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(n_routed_experts=NUM_EXPERTS)
    )
    model = torch.nn.ModuleList([module]) if isinstance(module, TopK) else module
    with (
        patch.object(moe_ep_setup, "get_exec", return_value=_exec_stub()),
        patch.object(
            moe_ep_setup,
            "get_parallel",
            return_value=SimpleNamespace(moe_ep_size=EP_SIZE, moe_ep_rank=rank),
        ),
    ):
        moe_ep_setup.prepare_moe_topk(model=model, model_config=model_config)
    return module


def _make_balancer(*, fold=False) -> WaterfillBalancer:
    """The balancer prepare_moe_topk builds for a TopK (post-MoE scaling by
    default, so the shared slot carries 1/rsf)."""
    return _prepare(_build_topk(fold=fold)).waterfill_balancer


def _check_layout(test, ids, weights, out_ids, out_weights, shared_weight):
    """Contract 2: layout of the expanded [N, topk+1] tensors."""
    n = ids.shape[0]
    test.assertEqual(out_ids.shape, (n, TOPK + 1))
    test.assertEqual(out_weights.shape, (n, TOPK + 1))
    test.assertEqual(out_ids.dtype, ids.dtype)
    test.assertEqual(out_weights.dtype, weights.dtype)
    torch.testing.assert_close(out_ids[:, :TOPK], _remap_routed(ids))
    valid = ids >= 0
    torch.testing.assert_close(
        out_weights[:, :TOPK], torch.where(valid, weights, torch.zeros_like(weights))
    )
    has_valid = valid.any(dim=1)
    shared_ids = out_ids[:, TOPK]
    # Padded rows: marker id and zero weight; live rows: some rank's shared slot
    # with exactly the configured weight.
    test.assertTrue(torch.all(shared_ids[~has_valid] == LOCAL_SHARED_MARKER))
    test.assertTrue(torch.all(out_weights[~has_valid, TOPK] == 0))
    live = shared_ids[has_valid]
    test.assertTrue(torch.all((live - OLD_EPR) % NEW_EPR == 0))
    test.assertTrue(torch.all((live >= 0) & (live < EP_SIZE * NEW_EPR)))
    torch.testing.assert_close(
        out_weights[has_valid, TOPK],
        torch.full_like(out_weights[has_valid, TOPK], shared_weight),
    )
    return (live - OLD_EPR) // NEW_EPR  # chosen rank per live row


class TestWaterfillExpansion(CustomTestCase):
    """Contracts 2 and 3 on the two expansion paths."""

    def test_local_expansion_layout_and_home_rank(self):
        ids, weights = _random_topk(300, padded_rows=(0, 17, 299))
        out_ids, out_weights = expand_topk_with_shared_expert(
            ids, weights, NUM_EXPERTS, EP_SIZE, EP_RANK, 1.0 / RSF
        )
        chosen = _check_layout(self, ids, weights, out_ids, out_weights, 1.0 / RSF)
        # The local path never moves the shared expert off the home rank.
        self.assertTrue(torch.all(chosen == EP_RANK))

    def test_fused_expansion_layout_matches_local_path(self):
        ids, weights = _random_topk(1000, seed=1, padded_rows=(5, 6))
        rank_load = torch.full((EP_SIZE,), 1000, dtype=torch.int64, device=DEVICE)
        out_ids, out_weights = materialize_waterfill_dispatch_fused(
            ids,
            weights,
            rank_load,
            NUM_EXPERTS,
            EP_SIZE,
            EP_RANK,
            1.0,
            allow_all_ranks=True,
            target_total=0,
        )
        _check_layout(self, ids, weights, out_ids, out_weights, 1.0)
        local_ids, local_weights = expand_topk_with_shared_expert(
            ids, weights, NUM_EXPERTS, EP_SIZE, EP_RANK, 1.0
        )
        # Routed columns are identical on both paths; only the shared rank differs.
        torch.testing.assert_close(out_ids[:, :TOPK], local_ids[:, :TOPK])
        torch.testing.assert_close(out_weights, local_weights)

    def test_fused_expansion_is_deterministic(self):
        ids, weights = _random_topk(2048, seed=2)
        rank_load = torch.tensor(
            [100, 2000, 300, 400, 500, 600, 700, 800], dtype=torch.int64, device=DEVICE
        )
        a = materialize_waterfill_dispatch_fused(
            ids,
            weights,
            rank_load,
            NUM_EXPERTS,
            EP_SIZE,
            EP_RANK,
            0.4,
            allow_all_ranks=True,
        )
        b = materialize_waterfill_dispatch_fused(
            ids,
            weights,
            rank_load,
            NUM_EXPERTS,
            EP_SIZE,
            EP_RANK,
            0.4,
            allow_all_ranks=True,
        )
        torch.testing.assert_close(a[0], b[0])
        torch.testing.assert_close(a[1], b[1])

    def test_static_waterfill_fills_headroom_below_water_level(self):
        """Static mode samples the shared rank in proportion to
        ``target_total - rank_load`` (home rank 11:10 preferred); ranks at or above
        the water level receive nothing."""
        num_tokens = 8192
        ids, weights = _random_topk(num_tokens, seed=3)
        # Routed load per rank as the balancer would count it; rank 1 is already
        # above the water level, rank 5 is far below it.
        rank_load = torch.tensor(
            [8192, 20000, 8192, 8192, 8192, 1000, 8192, 8192],
            dtype=torch.int64,
            device=DEVICE,
        )
        out_ids, _ = materialize_waterfill_dispatch_fused(
            ids,
            weights,
            rank_load,
            NUM_EXPERTS,
            EP_SIZE,
            EP_RANK,
            0.4,
            allow_all_ranks=True,
            target_total=0,
        )
        chosen = (out_ids[:, TOPK] - OLD_EPR) // NEW_EPR
        counts = torch.bincount(chosen, minlength=EP_SIZE).double()

        total = int(rank_load.sum())
        target_total = (total + total // TOPK + EP_SIZE - 1) // EP_SIZE
        headroom = (target_total - rank_load).clamp(min=0).double()
        pref = torch.ones(EP_SIZE, dtype=torch.float64, device=DEVICE) * (10 / 11)
        pref[EP_RANK] = 1.0
        expected = headroom * pref
        expected = expected / expected.sum() * num_tokens

        self.assertEqual(int(counts[1]), 0, "a rank above the water level got tokens")
        self.assertGreater(int(counts[5]), int(counts[0]))
        # The hash-based sampler is deterministic, not i.i.d.; allow 15% slack
        # on every rank that is expected to receive a meaningful share.
        for r in range(EP_SIZE):
            if expected[r] < 50:
                continue
            self.assertLess(
                abs(counts[r] - expected[r]) / expected[r],
                0.15,
                f"rank {r}: got {int(counts[r])}, expected ~{int(expected[r])}",
            )

    def test_dynamic_waterfill_only_uses_home_and_visited_ranks(self):
        """Dynamic mode with ``allow_all_ranks=False`` may only place the shared
        expert on the home rank or on a rank one of the routed experts goes to."""
        ids, weights = _random_topk(4096, seed=4)
        rank_load = torch.tensor(
            [5000, 100, 5000, 5000, 5000, 5000, 100, 5000],
            dtype=torch.int64,
            device=DEVICE,
        )
        out_ids, _ = materialize_waterfill_dispatch_fused(
            ids,
            weights,
            rank_load,
            NUM_EXPERTS,
            EP_SIZE,
            EP_RANK,
            0.4,
            allow_all_ranks=False,
            target_total=6000,
        )
        chosen = (out_ids[:, TOPK] - OLD_EPR) // NEW_EPR
        visited = _ids_to_ranks(ids)
        allowed = (visited == chosen[:, None]).any(dim=1) | (chosen == EP_RANK)
        self.assertTrue(bool(allowed.all()))
        # With two ranks nearly empty, a visible share must move off the home rank.
        self.assertLess(int((chosen == EP_RANK).sum()), ids.shape[0])


class TestWaterfillBalancer(CustomTestCase):
    """Contract 3 at the balancer level (static mode, single process)."""

    def test_count_local_routed_matches_bincount(self):
        ids, _ = _random_topk(777, seed=5, padded_rows=(1, 2, 3))
        counts = _make_balancer().count_local_routed(ids)
        ranks = _ids_to_ranks(ids)
        expected = torch.bincount(ranks[ranks >= 0], minlength=EP_SIZE).to(torch.int64)
        torch.testing.assert_close(counts, expected)

    def test_low_batch_keeps_shared_expert_local(self):
        balancer = _make_balancer()
        self.assertTrue(balancer.use_static_waterfill)
        n = WaterfillBalancer.MIN_BATCH_FOR_BALANCE - 1
        ids, weights = _random_topk(n, seed=6)
        out = balancer.expand_topk(
            topk_module.StandardTopKOutput(weights, ids, router_logits=None), n
        )
        chosen = _check_layout(
            self, ids, weights, out.topk_ids, out.topk_weights, 1.0 / RSF
        )
        self.assertTrue(torch.all(chosen == EP_RANK))

    def test_balanced_batch_moves_shared_expert_off_a_loaded_home_rank(self):
        balancer = _make_balancer()
        n = 4096
        # Every token routes to the home rank's experts only: the home rank is
        # the hot spot, so Waterfill must spread the shared expert elsewhere.
        g = torch.Generator(device="cpu").manual_seed(7)
        ids = (
            (
                EP_RANK * OLD_EPR
                + torch.stack(
                    [torch.randperm(OLD_EPR, generator=g)[:TOPK] for _ in range(n)]
                )
            )
            .to(torch.int32)
            .to(DEVICE)
        )
        weights = torch.full((n, TOPK), 1.0 / TOPK, device=DEVICE)
        out = balancer.expand_topk(
            topk_module.StandardTopKOutput(weights, ids, router_logits=None), n
        )
        chosen = _check_layout(
            self, ids, weights, out.topk_ids, out.topk_weights, 1.0 / RSF
        )
        self.assertEqual(int((chosen == EP_RANK).sum()), 0)
        counts = torch.bincount(chosen, minlength=EP_SIZE)
        # The seven idle ranks share the load evenly (within 10%).
        others = counts[torch.arange(EP_SIZE, device=DEVICE) != EP_RANK].double()
        self.assertLess(float((others.max() - others.min()) / others.mean()), 0.10)

    def test_empty_batch(self):
        balancer = _make_balancer()
        ids = torch.empty(0, TOPK, dtype=torch.int32, device=DEVICE)
        weights = torch.empty(0, TOPK, dtype=torch.float32, device=DEVICE)
        out = balancer.expand_topk(
            topk_module.StandardTopKOutput(weights, ids, router_logits=None), 0
        )
        self.assertEqual(out.topk_ids.shape, (0, TOPK + 1))
        self.assertEqual(out.topk_weights.shape, (0, TOPK + 1))

    def test_dynamic_plan_water_level(self):
        """Dynamic mode's water level: effective load + one shared slot per token,
        averaged over the ranks; the all-ranks shortcut only when no rank is above."""
        balancer = _make_balancer()
        routed = torch.tensor([800] * EP_SIZE, dtype=torch.int64, device=DEVICE)
        tokens = torch.tensor([100] * EP_SIZE, dtype=torch.int64, device=DEVICE)
        plan = balancer._build_dynamic_dispatch_plan(routed, tokens, TOPK)
        self.assertIsInstance(plan, WaterfillDispatchPlan)
        torch.testing.assert_close(plan.rank_load, routed + tokens)
        # total effective 7200 + total tokens 800 (=6400/8) -> 8000 / 8 ranks.
        self.assertEqual(plan.target_total, 1000)
        self.assertTrue(plan.allow_all_ranks)

        routed[1] = 3000
        plan = balancer._build_dynamic_dispatch_plan(routed, tokens, TOPK)
        self.assertFalse(plan.allow_all_ranks)


@contextmanager
def _topk_runtime():
    """The runtime state TopK.forward reads in a single-process test."""
    with (
        patch.object(topk_module, "_use_aiter", False),
        patch.object(topk_module, "get_exec", return_value=_exec_stub()),
        patch.object(
            topk_module,
            "get_parallel",
            return_value=SimpleNamespace(
                tp_group=None, moe_ep_size=EP_SIZE, moe_ep_rank=EP_RANK
            ),
        ),
        patch.object(topk_module, "is_allocation_symmetric", return_value=False),
        patch.object(
            topk_module,
            "use_symmetric_memory",
            side_effect=lambda *_args, **_kwargs: nullcontext(),
        ),
    ):
        yield


class TestWaterfillTopK(CustomTestCase):
    """Contracts 1 and 4: the TopK module under Waterfill."""

    def _build_topk(self, *, fold):
        return _build_topk(fold=fold)

    def _prepare(self, topk):
        return _prepare(topk)

    def test_router_is_asked_for_routed_experts_only(self):
        topk = self._build_topk(fold=False)
        self.assertTrue(topk.enable_waterfill)
        self.assertEqual(topk.topk_config.top_k, TOPK)
        self.assertEqual(topk.topk_config.num_fused_shared_experts, 0)
        self.assertEqual(topk.topk_config.output_format, TopKOutputFormat.STANDARD)
        self.assertIsNone(topk.waterfill_balancer)

    def test_forward_requires_prepare(self):
        topk = self._build_topk(fold=False)
        hidden = torch.zeros(4, 16, device=DEVICE)
        logits = torch.randn(4, NUM_EXPERTS, device=DEVICE)
        with (
            _topk_runtime(),
            self.assertRaisesRegex(RuntimeError, "prepared by ModelRunner"),
        ):
            topk(hidden, logits)

    def test_shared_weight_follows_where_rsf_is_applied(self):
        # Post-MoE scaling: the epilogue multiplies the combined output by rsf,
        # the shared slot carries 1/rsf. Folded in top-k: nothing scales the
        # output afterwards, the shared slot carries 1.0.
        self.assertAlmostEqual(
            self._prepare(
                self._build_topk(fold=False)
            ).waterfill_balancer.shared_weight,
            1.0 / RSF,
        )
        self.assertAlmostEqual(
            self._prepare(self._build_topk(fold=True)).waterfill_balancer.shared_weight,
            1.0,
        )

    def test_forward_appends_shared_column_after_routing(self):
        torch.manual_seed(0)
        for fold in (False, True):
            with self.subTest(fold=fold):
                topk = self._prepare(self._build_topk(fold=fold))
                n = 512
                hidden = torch.zeros(n, 16, device=DEVICE, dtype=torch.bfloat16)
                logits = torch.randn(n, NUM_EXPERTS, device=DEVICE)
                with _topk_runtime():
                    out = topk(hidden, logits)
                self.assertEqual(out.topk_ids.shape, (n, TOPK + 1))
                routed_ids = out.topk_ids[:, :TOPK]
                routed_w = out.topk_weights[:, :TOPK]
                # Routed ids are in the per-rank-slot layout and unique per row.
                self.assertTrue(torch.all((routed_ids + 1) % NEW_EPR != 0))
                self.assertEqual(
                    int(routed_ids.sort(dim=1).values.diff(dim=1).eq(0).sum()), 0
                )
                torch.testing.assert_close(
                    routed_w.sum(-1),
                    torch.full((n,), RSF if fold else 1.0, device=DEVICE),
                    atol=1e-4,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    out.topk_weights[:, TOPK],
                    torch.full((n,), 1.0 if fold else 1.0 / RSF, device=DEVICE),
                )
                shared = out.topk_ids[:, TOPK]
                self.assertTrue(torch.all((shared - OLD_EPR) % NEW_EPR == 0))


class _StubGate(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.e_score_correction_bias = torch.nn.Parameter(
            torch.zeros(NUM_EXPERTS, device=DEVICE), requires_grad=False
        )
        self.e_score_correction_bias_vl = None

    def forward(self, hidden_states, forward_batch=None):
        return self.logits[: hidden_states.shape[0]]


class _LinearExperts(torch.nn.Module):
    """Stand-in for FusedMoE on the DeepEP path: expert ``i`` (in the per-rank-slot
    layout) multiplies its input by ``scale[i]``; the output is the top-k weighted
    sum, exactly what dispatch + grouped GEMM + combine compute."""

    def __init__(self, *, num_experts, top_k, num_fused_shared_experts, fold, **_):
        super().__init__()
        self.kwargs = dict(
            num_experts=num_experts,
            top_k=top_k,
            num_fused_shared_experts=num_fused_shared_experts,
        )
        self.should_fuse_routed_scaling_factor_in_topk = fold
        routed_scale = torch.arange(1, NUM_EXPERTS + 1, device=DEVICE).float() / 64
        self.shared_scale = 7.0
        scale = torch.empty(NUM_EXPERTS + EP_SIZE, device=DEVICE)
        for r in range(EP_SIZE):
            scale[r * NEW_EPR : r * NEW_EPR + OLD_EPR] = routed_scale[
                r * OLD_EPR : (r + 1) * OLD_EPR
            ]
            scale[_shared_slot(r)] = self.shared_scale
        self.routed_scale = routed_scale
        self.scale = scale

    def forward(self, hidden_states, topk_output):
        ids, weights = topk_output.topk_ids, topk_output.topk_weights
        valid = ids >= 0
        per_slot = torch.where(
            valid, self.scale[ids.clamp(min=0)], torch.zeros_like(weights)
        )
        coeff = (weights * per_slot).sum(-1, keepdim=True)
        return hidden_states * coeff.to(hidden_states.dtype)


class TestWaterfillDeepseekMoE(CustomTestCase):
    """Contract 5: the whole DeepSeek MoE block under Waterfill."""

    def _build(self, *, fold, logits):
        config = SimpleNamespace(
            routed_scaling_factor=RSF,
            n_shared_experts=1,
            n_routed_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            hidden_act="silu",
            hidden_size=64,
            moe_intermediate_size=64,
            norm_topk_prob=True,
            n_group=N_GROUP,
            topk_group=TOPK_GROUP,
            scoring_func="sigmoid",
            num_hash_layers=0,
            vocab_size=16,
        )
        a2a = SimpleNamespace(
            **{
                name: (lambda v=(name == "is_deepep"): v)
                for name in (
                    "is_deepep",
                    "is_mooncake",
                    "is_nixl",
                    "is_mori",
                    "is_ascend_fuseep",
                    "is_flashinfer",
                    "is_deepep_v2",
                    "is_pplx",
                    "is_megamoe",
                    "is_flashinfer_megamoe",
                    "is_none",
                    "is_standard",
                )
            }
        )
        module = deepseek_v2_module
        with (
            patch.object(module, "_use_aiter", False),
            patch.object(topk_module, "_use_aiter", False),
            patch.object(
                module,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=1, moe_ep_size=EP_SIZE),
            ),
            patch.object(module, "get_exec", return_value=_exec_stub()),
            patch.object(topk_module, "get_exec", return_value=_exec_stub()),
            patch.object(
                module, "is_shared_experts_fusion_disabled", return_value=False
            ),
            patch.object(module, "has_per_rank_fused_shared_slots", return_value=True),
            patch.object(module, "get_moe_a2a_backend", return_value=a2a),
            patch.object(module, "MoEGate", return_value=_StubGate(logits)),
            patch.object(
                module,
                "get_moe_impl_class",
                return_value=lambda **kwargs: _LinearExperts(fold=fold, **kwargs),
            ),
            patch.object(
                module,
                "SboFlags",
                SimpleNamespace(fuse_shared_experts_inside_sbo=lambda: False),
            ),
        ):
            moe = module.DeepseekV2MoE(config=config, layer_id=0)
        return _prepare(moe)

    def _reference(self, hidden, logits, experts):
        """rsf * sum_i w_i E_i(x) + S(x), with the routed experts and their
        normalized weights taken from the same DeepSeek router the block uses
        (so near-tied scores cannot make the two sides pick different experts)."""
        weights, ids = topk_module.biased_grouped_topk_gpu(
            hidden,
            logits,
            torch.zeros(NUM_EXPERTS, device=DEVICE),
            topk=TOPK,
            renormalize=True,
            num_expert_group=N_GROUP,
            topk_group=TOPK_GROUP,
            num_fused_shared_experts=0,
            routed_scaling_factor=RSF,
            apply_routed_scaling_factor_on_output=False,
        )
        torch.testing.assert_close(
            weights.sum(-1), torch.ones(hidden.shape[0], device=DEVICE)
        )
        coeff = (weights * experts.routed_scale[ids.long()]).sum(-1, keepdim=True)
        return hidden * (RSF * coeff + experts.shared_scale).to(hidden.dtype)

    def test_forward_deepep_counts_routed_rsf_times_and_shared_once(self):
        torch.manual_seed(0)
        forward_batch = SimpleNamespace(moe_num_token_non_padded=lambda: None)
        for fold in (False, True):
            for n in (16, 1024):  # local expansion and fused Waterfill
                with self.subTest(fold=fold, num_tokens=n):
                    hidden = torch.randn(n, 64, device=DEVICE)
                    logits = torch.randn(n, NUM_EXPERTS, device=DEVICE)
                    moe = self._build(fold=fold, logits=logits)
                    self.assertEqual(moe.num_fused_shared_experts, 1)
                    self.assertEqual(
                        moe.experts.kwargs["num_experts"], NUM_EXPERTS + EP_SIZE
                    )
                    self.assertEqual(moe.experts.kwargs["top_k"], TOPK + 1)
                    self.assertFalse(hasattr(moe, "shared_experts"))
                    self.assertEqual(moe.topk.topk_config.top_k, TOPK)
                    self.assertEqual(
                        moe.topk.topk_config.apply_routed_scaling_factor_on_output, fold
                    )
                    # No EPLB metadata in this process: the TopK runs without a
                    # logical->physical remap, as it does with EPLB disabled.
                    with (
                        _topk_runtime(),
                        patch.object(
                            deepseek_v2_module.ExpertLocationDispatchInfo,
                            "init_new",
                            return_value=None,
                        ),
                    ):
                        out = moe.forward_deepep(hidden, forward_batch)
                    expected = self._reference(hidden, logits, moe.experts)
                    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
