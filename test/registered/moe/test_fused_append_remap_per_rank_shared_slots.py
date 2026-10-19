"""Unit tests for fused append + per-rank shared-slot remap.

Covers ``fused_append_remap_shared_experts_deepep``, the single Triton launch
form of ``remap_topk_for_per_rank_shared_slots()`` on the per-rank shared-slot
path, and ``select_experts`` on that path with the real routers. The kernels are
GPU-only (Triton), so these tests are skipped when no accelerator is present.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    fused_append_remap_shared_experts_deepep,
)
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    biased_grouped_topk_impl,
    remap_topk_for_per_rank_shared_slots,
    select_experts,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils import get_device, is_cuda
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")


def _reference_append_remap(
    topk_ids, topk_weights, s, scale_factor, shared_id_base, num_local_routed
):
    """Pure-torch golden reference mirroring the kernel's documented contract.

    Routed IDs:   e -> e + (e // num_local_routed) * s
    Shared IDs:   shared_id_base + arange(s)
    Routed wgt:   passthrough
    Shared wgt:   scale_factor
    """
    m, k = topk_ids.shape
    out_ids = torch.empty((m, k + s), dtype=topk_ids.dtype, device=topk_ids.device)
    out_w = torch.empty(
        (m, k + s), dtype=topk_weights.dtype, device=topk_weights.device
    )
    out_ids[:, :k] = topk_ids + (topk_ids // num_local_routed) * s
    out_w[:, :k] = topk_weights
    shared = shared_id_base + torch.arange(s, device=topk_ids.device)
    out_ids[:, k:] = shared.to(topk_ids.dtype)
    out_w[:, k:] = scale_factor
    return out_ids, out_w


@unittest.skipUnless(
    torch.cuda.is_available(), "fused append+remap kernel requires a GPU"
)
class TestFusedAppendRemapPerRankSharedSlots(CustomTestCase):
    # (m, k, num_physical_routed, ep_size, ep_rank, num_fused_shared_experts).
    # Includes non-power-of-two k and num_fused_shared_experts (DeepSeek-V4 routes
    # top-6): the kernel blocks over next_power_of_2 and masks, so these must work.
    CASES = [
        (1, 8, 256, 8, 0, 1),
        (4, 8, 256, 8, 7, 1),
        (17, 8, 264, 8, 3, 1),
        (128, 16, 128, 4, 2, 2),
        (1, 6, 258, 6, 0, 1),  # DSV4: k=6 (non-pow2), s=1
        (13, 6, 258, 6, 5, 1),  # DSV4: k=6 (non-pow2), non-zero ep_rank
        (32, 6, 264, 4, 2, 3),  # non-pow2 k=6 and non-pow2 s=3 together
    ]

    def _make_inputs(self, m, k, num_physical_routed, ids_dtype=torch.int64):
        device = get_device()
        g = torch.Generator(device="cpu").manual_seed(m * 1000 + k * 7 + 1)
        topk_ids = torch.randint(
            0, num_physical_routed, (m, k), generator=g, dtype=ids_dtype
        ).to(device)
        topk_weights = torch.rand((m, k), generator=g, dtype=torch.float32).to(device)
        return topk_ids, topk_weights

    @staticmethod
    def _shared_id_base(num_physical_routed, ep_size, ep_rank, s):
        num_local_routed = num_physical_routed // ep_size
        num_local_experts = num_local_routed + s
        return ep_rank * num_local_experts + num_local_routed, num_local_routed

    def test_matches_golden_reference(self):
        """Kernel output equals the documented routed-remap + shared-append math."""
        for m, k, npr, ep_size, ep_rank, s in self.CASES:
            with self.subTest(m=m, k=k, npr=npr, ep_rank=ep_rank, s=s):
                shared_id_base, num_local_routed = self._shared_id_base(
                    npr, ep_size, ep_rank, s
                )
                scale_factor = 1.0
                topk_ids, topk_weights = self._make_inputs(m, k, npr)

                got_ids, got_w = fused_append_remap_shared_experts_deepep(
                    topk_ids,
                    topk_weights,
                    s,
                    scale_factor,
                    shared_id_base,
                    num_local_routed,
                )
                exp_ids, exp_w = _reference_append_remap(
                    topk_ids,
                    topk_weights,
                    s,
                    scale_factor,
                    shared_id_base,
                    num_local_routed,
                )

                self.assertEqual(tuple(got_ids.shape), (m, k + s))
                self.assertTrue(torch.equal(got_ids, exp_ids))
                self.assertTrue(torch.allclose(got_w, exp_w))

    def test_no_routed_shared_collision_across_ranks(self):
        """Remapped routed ids never land on any rank's shared slots (S > 1).

        Independent of the gap-insertion math the kernel/eager path use: the
        per-rank layout is, by definition, ep_size contiguous blocks of width
        num_local_experts == num_local_routed + S, each block being
        [num_local_routed routed ids ... S shared ids]. So a physical routed id
        ``e`` must map to ``rank * num_local_experts + local`` where
        ``rank = e // num_local_routed`` and ``local = e % num_local_routed`` --
        derived from the block layout, not from ``e + (e // nlr) * S``.

        This is the regression guard for the S > 1 bug: the old
        ``e + e // num_local_routed`` shifts by a single slot, so e.g. the first
        routed id of rank 1 (e == num_local_routed) mapped to
        ``num_local_routed + 1``, colliding with rank 0's shared slots when
        S > 1. The check asserts (a) the kernel matches the block-derived ids and
        (b) no remapped routed id intersects the shared-slot id set of ANY rank.
        """
        # Every config here uses S > 1 and spans all ep_size ranks (npr == m*k
        # feeds each physical routed id exactly once) so rank boundaries are hit.
        # (m, k, num_physical_routed, ep_size, num_fused_shared_experts).
        CASES = [
            (44, 6, 264, 4, 3),  # DSV4-shaped: non-pow2 k=6, non-pow2 S=3
            (32, 8, 256, 8, 2),  # pow2 k, S=2
            (43, 6, 258, 6, 4),  # non-pow2 npr/rank boundaries, S=4
        ]
        for m, k, npr, ep_size, s in CASES:
            with self.subTest(m=m, k=k, npr=npr, ep_size=ep_size, s=s):
                self.assertEqual(m * k, npr)  # cover each physical id once
                num_local_routed = npr // ep_size
                num_local_experts = num_local_routed + s
                device = get_device()

                # Feed every physical routed id [0, npr) through the kernel.
                all_ids = torch.arange(npr, device=device, dtype=torch.int64).view(m, k)
                weights = torch.ones((m, k), dtype=torch.float32, device=device)
                # shared_id_base / ep_rank only affect the appended shared columns,
                # not the routed remap under test; ep_rank 0 is fine here.
                shared_id_base = num_local_routed
                got_ids, _ = fused_append_remap_shared_experts_deepep(
                    all_ids, weights, s, 1.0, shared_id_base, num_local_routed
                )
                routed_out = got_ids[:, :k].reshape(-1)

                # (a) Independent block-derived expectation.
                e = torch.arange(npr, device=device, dtype=torch.int64)
                rank = e // num_local_routed
                local = e % num_local_routed
                expected = rank * num_local_experts + local
                self.assertTrue(torch.equal(routed_out, expected))

                # (b) No remapped routed id hits any rank's shared slots.
                shared_slots = set()
                for r in range(ep_size):
                    base = r * num_local_experts + num_local_routed
                    shared_slots.update(range(base, base + s))
                routed_set = set(routed_out.tolist())
                self.assertEqual(routed_set & shared_slots, set())
                # Routed ids stay unique and inside the global id space.
                self.assertEqual(len(routed_set), npr)
                self.assertLess(max(routed_set), ep_size * num_local_experts)

    def test_equivalence_with_eager_remap(self):
        """Fused and eager append agree for folding and nonfolding routes."""
        rsf = 2.5
        for folded, scale_factor in ((False, 1.0 / rsf), (True, 1.0)):
            for m, k, npr, ep_size, ep_rank, s in self.CASES:
                with self.subTest(
                    m=m, k=k, npr=npr, ep_rank=ep_rank, s=s, folded=folded
                ):
                    shared_id_base, num_local_routed = self._shared_id_base(
                        npr, ep_size, ep_rank, s
                    )
                    topk_ids, topk_weights = self._make_inputs(m, k, npr)

                    fused_ids, fused_w = fused_append_remap_shared_experts_deepep(
                        topk_ids.clone(),
                        topk_weights.clone(),
                        s,
                        scale_factor,
                        shared_id_base,
                        num_local_routed,
                    )

                    with get_parallel().override(
                        moe_ep_size=ep_size, moe_ep_rank=ep_rank
                    ):
                        eager_ids, eager_w = remap_topk_for_per_rank_shared_slots(
                            topk_ids.clone(),
                            topk_weights.clone(),
                            s,
                            npr,
                            TopKConfig(
                                top_k=k,
                                num_shared_topk_slots=s,
                                routed_scaling_factor=rsf,
                                apply_routed_scaling_factor_on_output=folded,
                            ),
                        )

                    self.assertTrue(torch.equal(fused_ids, eager_ids))
                    self.assertTrue(torch.allclose(fused_w, eager_w))

    def test_shared_weight_is_one_on_aiter_path(self):
        """On the aiter path the always-on shared expert must contribute 1.0x."""
        m, k, npr, ep_size, ep_rank, s = 8, 8, 256, 8, 1, 1
        shared_id_base, num_local_routed = self._shared_id_base(
            npr, ep_size, ep_rank, s
        )
        topk_ids, topk_weights = self._make_inputs(m, k, npr)

        _, got_w = fused_append_remap_shared_experts_deepep(
            topk_ids, topk_weights, s, 1.0, shared_id_base, num_local_routed
        )
        self.assertTrue(torch.all(got_w[:, -s:] == 1.0))

    def test_pad_fold_matches_separate_fill(self):
        """HAS_PADDING fold == separate padded-fill(0) then append+remap.

        The fusion folds the padded-topk_ids fill into this kernel: rows
        >= num_token_non_padded get pad_fill_id in every routed slot. With
        pad_fill_id=0 this is bit-identical to the previous path that filled the
        padded region with 0 (topk_ids=0 -> remap 0 + 0//nlr = 0) via a separate
        _fill_padded_rows launch before append+remap ran.
        """
        for m, k, npr, ep_size, ep_rank, s in self.CASES:
            for n_valid in (0, max(m // 2, 1), m):
                with self.subTest(m=m, k=k, ep_rank=ep_rank, s=s, n_valid=n_valid):
                    shared_id_base, num_local_routed = self._shared_id_base(
                        npr, ep_size, ep_rank, s
                    )
                    topk_ids, topk_weights = self._make_inputs(m, k, npr)

                    # Baseline: pre-fill padded rows to 0, no fold.
                    base_ids = topk_ids.clone()
                    base_ids[n_valid:] = 0
                    exp_ids, exp_w = fused_append_remap_shared_experts_deepep(
                        base_ids,
                        topk_weights.clone(),
                        s,
                        1.0,
                        shared_id_base,
                        num_local_routed,
                    )

                    # Fused: fold the fill (no pre-fill), pad_fill_id=0.
                    ntnp = torch.tensor(
                        [n_valid], dtype=torch.int32, device=topk_ids.device
                    )
                    got_ids, got_w = fused_append_remap_shared_experts_deepep(
                        topk_ids.clone(),
                        topk_weights.clone(),
                        s,
                        1.0,
                        shared_id_base,
                        num_local_routed,
                        num_token_non_padded=ntnp,
                        pad_fill_id=0,
                    )

                    self.assertTrue(torch.equal(got_ids, exp_ids))
                    self.assertTrue(torch.allclose(got_w, exp_w))

    def test_no_shared_experts_is_noop(self):
        """s == 0 returns the inputs untouched (no kernel launch)."""
        topk_ids, topk_weights = self._make_inputs(4, 8, 256)
        got_ids, got_w = fused_append_remap_shared_experts_deepep(
            topk_ids, topk_weights, 0, 1.0, 0, 32
        )
        self.assertTrue(torch.equal(got_ids, topk_ids))
        self.assertTrue(torch.equal(got_w, topk_weights))


@unittest.skipUnless(is_cuda(), "covers the CUDA routers")
class TestSelectExpertsPerRankSharedSlots(CustomTestCase):
    """Shared slots must not alter routed selection or normalization."""

    NUM_TOKENS, TOP_K, EP_SIZE, EP_RANK = 32, 8, 8, 3
    # (num_experts, num_expert_group, topk_group, scoring_func, biased); each row
    # reaches a different router.
    ROUTERS = [
        (256, 8, 4, "sigmoid", True),  # DeepSeek-V3 grouped routing
        (384, 8, 4, "sigmoid", True),  # grouped, outside FlashInfer's shapes
        (64, 8, 3, "softmax", False),  # grouped, no correction bias
        (256, None, None, "sqrtsoftplus", True),  # DeepSeek-V4 routing
        (128, None, None, "softmax", False),
    ]

    def _route(
        self, *, num_fused_shared_experts, logits, bias, router, renormalize=True
    ):
        _, num_expert_group, topk_group, scoring_func, _ = router
        topk_config = TopKConfig(
            top_k=self.TOP_K + num_fused_shared_experts,
            renormalize=renormalize,
            use_grouped_topk=num_expert_group is not None,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            num_shared_topk_slots=num_fused_shared_experts,
            correction_bias=bias,
            scoring_func=scoring_func,
            routed_scaling_factor=2.5,
            allow_routed_experts_capture=False,
        )
        with (
            patch.object(
                topk_module,
                "has_per_rank_fused_shared_slots",
                side_effect=lambda num_fused: num_fused > 0,
            ),
            get_parallel().override(moe_ep_size=self.EP_SIZE, moe_ep_rank=self.EP_RANK),
        ):
            out = select_experts(
                torch.empty(logits.shape[0], 16, device=logits.device),
                logits,
                topk_config,
            )
        return out.topk_ids.long(), out.topk_weights

    def test_routes_like_no_fusion(self):
        k = self.TOP_K
        for router in self.ROUTERS:
            num_experts, _, _, _, biased = router
            with self.subTest(router=router):
                g = torch.Generator(device="cpu").manual_seed(num_experts)
                logits = torch.randn(self.NUM_TOKENS, num_experts, generator=g)
                bias = 0.1 * torch.randn(num_experts, generator=g) if biased else None
                logits = logits.to(get_device())
                bias = bias.to(get_device()) if biased else None
                ref_ids, ref_weights = self._route(
                    num_fused_shared_experts=0, logits=logits, bias=bias, router=router
                )
                ids, weights = self._route(
                    num_fused_shared_experts=1, logits=logits, bias=bias, router=router
                )

                self.assertEqual(ids.shape, (self.NUM_TOKENS, k + 1))
                torch.testing.assert_close(
                    weights[:, :k].sum(-1), torch.ones_like(weights[:, 0])
                )
                # Rank r's routed block starts at r * (num_local_routed + 1).
                num_local_routed = num_experts // self.EP_SIZE
                expected_ids = ref_ids + ref_ids // num_local_routed
                order, ref_order = ids[:, :k].argsort(-1), expected_ids.argsort(-1)
                self.assertTrue(
                    torch.equal(
                        ids[:, :k].gather(-1, order), expected_ids.gather(-1, ref_order)
                    )
                )
                torch.testing.assert_close(
                    weights[:, :k].gather(-1, order), ref_weights.gather(-1, ref_order)
                )
                home_shared_id = (
                    self.EP_RANK * (num_local_routed + 1) + num_local_routed
                )
                self.assertTrue(torch.all(ids[:, k] == home_shared_id))

    def test_unnormalized_global_shared_contribution_is_one(self):
        """Raw routed scores must not multiply the global shared contribution."""
        logits = torch.zeros(4, 256, device=get_device())
        rsf = 2.5
        with get_context().override_server_args(
            moe_a2a_backend="none", moe_runner_backend="triton"
        ):
            for folded in (False, True):
                with self.subTest(folded=folded):
                    config = TopKConfig(
                        top_k=9,
                        use_grouped_topk=True,
                        num_expert_group=8,
                        topk_group=4,
                        renormalize=False,
                        num_shared_topk_slots=1,
                        correction_bias=torch.zeros(256, device=get_device()),
                        scoring_func="sigmoid",
                        routed_scaling_factor=rsf,
                        apply_routed_scaling_factor_on_output=folded,
                        allow_routed_experts_capture=False,
                    )
                    output = select_experts(logits, logits, config)
                    torch.testing.assert_close(
                        output.topk_weights[:, :8],
                        torch.full_like(
                            output.topk_weights[:, :8], 0.5 * (rsf if folded else 1.0)
                        ),
                    )
                    shared = output.topk_weights[:, -1] * (1.0 if folded else rsf)
                    torch.testing.assert_close(shared, torch.ones_like(shared))

    def test_unnormalized_weights_stay_raw(self):
        """Unnormalized routing must preserve raw selected sigmoid scores."""
        k, num_experts = self.TOP_K, 256
        router = (num_experts, 8, 4, "sigmoid", True)
        g = torch.Generator(device="cpu").manual_seed(0)
        logits = torch.randn(self.NUM_TOKENS, num_experts, generator=g).to(get_device())
        bias = (0.1 * torch.randn(num_experts, generator=g)).to(get_device())
        _, ref_ids = biased_grouped_topk_impl(
            logits,
            logits,
            bias,
            topk=k,
            renormalize=False,
            num_expert_group=8,
            topk_group=4,
        )
        num_local_routed = num_experts // self.EP_SIZE
        for num_fused_shared_experts in (0, 1):
            with self.subTest(num_fused_shared_experts=num_fused_shared_experts):
                ids, weights = self._route(
                    num_fused_shared_experts=num_fused_shared_experts,
                    logits=logits,
                    bias=bias,
                    router=router,
                    renormalize=False,
                )
                routed_ids = ids[:, :k]
                if num_fused_shared_experts:
                    routed_ids = routed_ids - routed_ids // (num_local_routed + 1)
                self.assertTrue(
                    torch.equal(
                        routed_ids.sort(-1).values, ref_ids.long().sort(-1).values
                    )
                )
                torch.testing.assert_close(
                    weights[:, :k], logits.sigmoid().gather(-1, routed_ids)
                )


if __name__ == "__main__":
    unittest.main()
