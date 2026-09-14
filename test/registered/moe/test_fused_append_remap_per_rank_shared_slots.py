"""Unit tests for the two GPU shared-expert top-k paths.

``fused_append_remap_shared_experts_deepep`` collapses
``fused_append_shared_experts()`` + ``remap_topk_for_per_rank_shared_slots()``
into one Triton launch; the aiter grouped-topk path instead pre-fills the
shared columns of a persistent buffer. Both are GPU-only.

The aiter path is split in two: ``TestAiterGroupedTopkSharedFuse`` needs a real
AITER GPU (ROCm only), while ``TestAiterGroupedTopkFusionContract`` stubs the
kernel so the host-side contract runs on every GPU runner.
"""

import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    fused_append_remap_shared_experts_deepep,
    fused_append_shared_experts,
)
from sglang.srt.layers.moe import aiter_topk as aiter_topk_module
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    _use_aiter,
    biased_grouped_topk_gpu,
    remap_topk_for_per_rank_shared_slots,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_device
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

    def test_equivalence_with_eager_append_then_remap(self):
        """Fused kernel == append shared experts + per-rank shared-slot remap.

        The eager remap overwrites the shared weight: 1.0 on the aiter/HIP path
        (routed_scaling_factor is pre-folded into the routed topk weights), else
        1/routed_scaling_factor. The fused kernel is invoked with that same value
        so the two paths stay bit-identical (ids match regardless of scaling).
        """
        rsf = 2.5
        scale_factor = 1.0 if _use_aiter else 1.0 / rsf
        for m, k, npr, ep_size, ep_rank, s in self.CASES:
            with self.subTest(m=m, k=k, npr=npr, ep_rank=ep_rank, s=s):
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

                with get_parallel().override(moe_ep_size=ep_size, moe_ep_rank=ep_rank):
                    eager_ids, eager_w = fused_append_shared_experts(
                        topk_ids.clone(),
                        topk_weights.clone(),
                        s,
                        scale_factor,
                        npr,  # shared-expert base id (overwritten by the remap)
                    )
                    eager_ids, eager_w = remap_topk_for_per_rank_shared_slots(
                        eager_ids,
                        eager_w,
                        s,
                        npr,
                        TopKConfig(
                            top_k=k,
                            num_fused_shared_experts=s,
                            routed_scaling_factor=rsf,
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


@unittest.skipUnless(
    torch.cuda.is_available() and _use_aiter,
    "the persistent shared-column buffer only exists on the aiter grouped-topk path",
)
class TestAiterGroupedTopkSharedFuse(CustomTestCase):
    """Kimi K2.5 routing must remain bit-identical without the append kernel."""

    # (num_tokens, num_experts, num_expert_group, topk_group, topk_routed,
    #  n_shared, routed_scaling_factor).
    CASES = [
        # Kimi K2.5: 384 routed experts, one group, top-8, one shared.
        (1, 384, 1, 1, 8, 1, 2.827),
        (37, 384, 1, 1, 8, 1, 2.827),
        (512, 384, 1, 1, 8, 1, 2.827),
        (8192, 384, 1, 1, 8, 1, 2.827),
    ]

    def setUp(self):
        super().setUp()
        # The buffer and the token budget are module-level caches; a stale entry
        # from another test would decide this one's path.
        aiter_topk_module._aiter_topk_fuse_shared_bufs.clear()
        aiter_topk_module._aiter_topk_fuse_shared_max_tokens_cache = None
        self.configs = {}

    def _make_inputs(self, num_tokens, num_experts):
        device = get_device()
        g = torch.Generator(device="cpu").manual_seed(num_tokens * 31 + num_experts)
        hidden_states = torch.randn(
            (num_tokens, 16), generator=g, dtype=torch.bfloat16
        ).to(device)
        gating_output = torch.randn(
            (num_tokens, num_experts), generator=g, dtype=torch.bfloat16
        ).to(device)
        correction_bias = torch.randn(
            (num_experts,), generator=g, dtype=torch.float32
        ).to(device)
        return hidden_states, gating_output, correction_bias

    def _topk(
        self, inputs, num_expert_group, topk_group, topk_routed, n_shared, rsf, factor
    ):
        hidden_states, gating_output, correction_bias = inputs
        key = (
            id(correction_bias),
            num_expert_group,
            topk_group,
            topk_routed,
            n_shared,
            rsf,
            factor,
        )
        if key not in self.configs:
            self.configs[key] = TopKConfig(
                top_k=topk_routed + n_shared,
                use_grouped_topk=True,
                renormalize=True,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=n_shared,
                correction_bias=correction_bias,
                routed_scaling_factor=rsf,
                fused_shared_experts_scaling_factor=factor,
            )
        with (
            patch.object(topk_module, "capture_routed_experts_if_allowed"),
            patch.object(topk_module, "get_global_expert_distribution_recorder"),
        ):
            output = aiter_topk_module.try_select_experts(
                hidden_states, gating_output, self.configs[key]
            )
        if output is not None:
            return output.topk_weights, output.topk_ids
        return biased_grouped_topk_gpu(
            hidden_states,
            gating_output,
            correction_bias,
            topk_routed,
            renormalize=True,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            num_fused_shared_experts=n_shared,
            routed_scaling_factor=rsf,
            apply_routed_scaling_factor_on_output=False,
        )

    def test_prepopulated_buffer_matches_plain_append(self):
        # None and 1 are the two spellings models use for an unscaled shared
        # expert; both must land on the same weight.
        for factor in (None, 1, 0.25):
            for case in self.CASES:
                num_tokens, num_experts, groups, topk_group, topk_routed, s, rsf = case
                with self.subTest(num_tokens=num_tokens, rsf=rsf, factor=factor):
                    inputs = self._make_inputs(num_tokens, num_experts)

                    with get_parallel().override(moe_ep_size=1):
                        fused_w, fused_ids = self._topk(
                            inputs, groups, topk_group, topk_routed, s, rsf, factor
                        )
                        # The return aliases the persistent buffer, which the
                        # next case overwrites.
                        fused_w, fused_ids = fused_w.clone(), fused_ids.clone()

                        # A zero budget puts every batch over the buffer, which
                        # is exactly the fallback select_experts appends to.
                        with patch.object(
                            aiter_topk_module,
                            "_get_aiter_topk_fuse_shared_max_tokens",
                            return_value=0,
                        ):
                            plain_w, plain_ids = self._topk(
                                inputs, groups, topk_group, topk_routed, s, rsf, factor
                            )

                    self.assertEqual(tuple(plain_ids.shape), (num_tokens, topk_routed))
                    plain_ids, plain_w = fused_append_shared_experts(
                        plain_ids,
                        plain_w,
                        s,
                        1.0 if factor is None else factor,
                        num_experts,  # shared-expert base id
                    )

                    self.assertEqual(
                        tuple(fused_ids.shape), (num_tokens, topk_routed + s)
                    )
                    self.assertTrue(torch.equal(fused_ids, plain_ids))
                    self.assertTrue(torch.equal(fused_w, plain_w))

    def test_kimi_select_experts_skips_append_and_preserves_padding(self):
        inputs = self._make_inputs(37, 384)
        hidden, logits, bias = inputs
        config = TopKConfig(
            top_k=9,
            use_grouped_topk=True,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            num_fused_shared_experts=1,
            correction_bias=bias,
            routed_scaling_factor=2.827,
            fused_shared_experts_scaling_factor=0.25,
        )
        append_path = "sglang.kernels.ops.moe.fused_moe_triton_kernels.fused_append_shared_experts"
        with (
            get_parallel().override(moe_ep_size=1),
            patch(append_path, wraps=fused_append_shared_experts) as append,
            patch.object(topk_module, "capture_routed_experts_if_allowed") as capture,
            patch.object(topk_module, "get_global_expert_distribution_recorder"),
        ):
            fused = topk_module.select_experts(hidden, logits, config)
            append.assert_not_called()
            self.assertEqual(capture.call_args.args[2].shape[1], 8)
            expected_ids = fused.topk_ids.clone()
            expected_w = fused.topk_weights.clone()
            valid = torch.tensor([17], dtype=torch.int32, device=logits.device)
            padded = topk_module.select_experts(
                hidden,
                logits,
                config,
                num_token_non_padded=valid,
            )
            append.assert_called_once()
            self.assertTrue(torch.equal(padded.topk_ids[:17], expected_ids[:17]))
            self.assertTrue(torch.equal(padded.topk_weights[:17], expected_w[:17]))
            self.assertEqual(torch.count_nonzero(padded.topk_weights[17:]).item(), 0)
            append.reset_mock()
            reused = topk_module.select_experts(hidden, logits, config)
            append.assert_not_called()
            self.assertTrue(torch.equal(reused.topk_ids, expected_ids))
            self.assertTrue(torch.equal(reused.topk_weights, expected_w))

    def test_kimi_graph_replay(self):
        inputs = self._make_inputs(37, 384)
        with get_parallel().override(moe_ep_size=1):
            self._topk(inputs, 1, 1, 8, 1, 2.827, 1)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                weights, ids = self._topk(inputs, 1, 1, 8, 1, 2.827, 1)
            for _ in range(3):
                inputs[1].normal_()
                graph.replay()
                got_w, got_ids = weights.clone(), ids.clone()
                with patch.object(
                    aiter_topk_module,
                    "_get_aiter_topk_fuse_shared_max_tokens",
                    return_value=0,
                ):
                    ref_w, ref_ids = self._topk(inputs, 1, 1, 8, 1, 2.827, 1)
                ref_ids, ref_w = fused_append_shared_experts(ref_ids, ref_w, 1, 1, 384)
                self.assertTrue(torch.equal(got_ids, ref_ids))
                self.assertTrue(torch.equal(got_w, ref_w))

    def test_fusion_eligibility_fallbacks(self):
        hidden, logits, bias = self._make_inputs(37, 384)
        config = TopKConfig(
            top_k=9,
            use_grouped_topk=True,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            num_fused_shared_experts=1,
            correction_bias=bias,
        )
        with get_parallel().override(moe_ep_size=1):
            for kwargs in (
                {"num_token_non_padded": torch.tensor([17], device=logits.device)},
                {"expert_location_dispatch_info": object()},
            ):
                with self.subTest(kwargs=kwargs):
                    self.assertIsNone(
                        aiter_topk_module.try_select_experts(
                            hidden, logits, config, **kwargs
                        )
                    )
            for setting in (
                "SGLANG_SIMULATE_UNIFORM_EXPERTS",
                "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
                "SGLANG_K3_RADIX4_TOPK",
            ):
                with (
                    self.subTest(setting=setting),
                    patch.object(
                        getattr(aiter_topk_module.envs, setting),
                        "get",
                        return_value=True,
                    ),
                ):
                    self.assertIsNone(
                        aiter_topk_module.try_select_experts(hidden, logits, config)
                    )
            with patch.object(
                aiter_topk_module, "has_per_rank_fused_shared_slots", return_value=True
            ):
                self.assertIsNone(
                    aiter_topk_module.try_select_experts(hidden, logits, config)
                )
        self.assertFalse(aiter_topk_module._aiter_topk_fuse_shared_bufs)

    def test_expert_parallelism_takes_the_plain_path(self):
        """EP retains the existing append path."""
        num_tokens, num_experts, groups, topk_group, topk_routed, s, rsf = self.CASES[1]
        inputs = self._make_inputs(num_tokens, num_experts)
        with get_parallel().override(moe_ep_size=2):
            _, topk_ids = self._topk(inputs, groups, topk_group, topk_routed, s, rsf, 1)
        self.assertEqual(tuple(topk_ids.shape), (num_tokens, topk_routed))

    def test_uninitialized_expert_parallel_group_falls_back(self):
        """Reaching this path before the MoE EP group exists must drop to the
        plain append rather than raise out of the parallel-state accessor."""
        num_tokens, num_experts, groups, topk_group, topk_routed, s, rsf = self.CASES[0]
        inputs = self._make_inputs(num_tokens, num_experts)
        with patch.object(
            aiter_topk_module,
            "get_parallel",
            side_effect=AssertionError(
                "expert model parallel group is not initialized"
            ),
        ):
            _, topk_ids = self._topk(inputs, groups, topk_group, topk_routed, s, rsf, 1)
        self.assertEqual(tuple(topk_ids.shape), (num_tokens, topk_routed))


class _FakeAiterBiasedGroupedTopk:
    """Stand-in for ``aiter.biased_grouped_topk`` with the same call contract.

    The real kernel exists only on ROCm. What the fusion needs from it is
    narrow and reproducible in torch: write ``topk_weights``/``topk_ids`` in
    place, honoring the row stride of the (strided) views it is handed. The
    stub records every call so the hand-off can be asserted, and writes through
    ``copy_`` so a caller that passed contiguous scratch instead of a view into
    the persistent buffer would surface as lost shared columns.
    """

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        renormalize,
        routed_scaling_factor,
    ):
        self.calls.append(
            {
                "bias_dtype": correction_bias.dtype,
                "weights_shape": tuple(topk_weights.shape),
                "ids_shape": tuple(topk_ids.shape),
                "weights_row_stride": topk_weights.stride(0),
                "ids_row_stride": topk_ids.stride(0),
                "weights_ptr": topk_weights.data_ptr(),
                "ids_ptr": topk_ids.data_ptr(),
                "num_expert_group": num_expert_group,
                "topk_group": topk_group,
                "renormalize": renormalize,
                "routed_scaling_factor": routed_scaling_factor,
            }
        )
        k = topk_weights.shape[1]
        scores = gating_output.float() + correction_bias.float()
        values, indices = torch.topk(scores, k, dim=-1)
        if renormalize:
            values = values / values.sum(dim=-1, keepdim=True)
        topk_ids.copy_(indices.to(topk_ids.dtype))
        topk_weights.copy_((values * routed_scaling_factor).to(topk_weights.dtype))


@unittest.skipUnless(
    torch.cuda.is_available(), "the fusion allocates its buffers on the accelerator"
)
class TestAiterGroupedTopkFusionContract(CustomTestCase):
    """Backend-independent coverage of the aiter grouped-topk fusion.

    ``TestAiterGroupedTopkSharedFuse`` checks the numerics against the real
    AITER kernel and can only run on ROCm. Everything else about the fusion --
    which configs are eligible, how the persistent buffer is laid out and
    reused, what reaches the kernel, and that ``select_experts`` prefers the
    fusion -- is plain torch/Python and is asserted here against a stubbed
    kernel, so it runs on every GPU runner instead of on AMD alone.
    """

    NUM_EXPERTS = 384
    TOPK_ROUTED = 8
    N_SHARED = 1

    def setUp(self):
        super().setUp()
        # Module-level caches: a stale entry from another test would decide
        # this one's path.
        aiter_topk_module._aiter_topk_fuse_shared_bufs.clear()
        aiter_topk_module._aiter_topk_fuse_shared_max_tokens_cache = None
        self.addCleanup(aiter_topk_module._aiter_topk_fuse_shared_bufs.clear)
        self.fake_kernel = _FakeAiterBiasedGroupedTopk()
        fake_aiter = types.ModuleType("aiter")
        fake_aiter.biased_grouped_topk = self.fake_kernel
        # The fusion imports aiter lazily, so swapping the module entry covers
        # both "aiter absent" (NVIDIA) and "aiter present" (ROCm).
        for patcher in (
            patch.dict(sys.modules, {"aiter": fake_aiter}),
            patch.object(topk_module, "capture_routed_experts_if_allowed"),
            patch.object(topk_module, "get_global_expert_distribution_recorder"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _config(self, **overrides):
        kwargs = dict(
            top_k=self.TOPK_ROUTED + self.N_SHARED,
            use_grouped_topk=True,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            num_fused_shared_experts=self.N_SHARED,
            correction_bias=torch.randn(
                (self.NUM_EXPERTS,), dtype=torch.float32, device=get_device()
            ),
            routed_scaling_factor=2.827,
            fused_shared_experts_scaling_factor=1.0,
        )
        kwargs.update(overrides)
        return TopKConfig(**kwargs)

    def _inputs(self, num_tokens=37):
        device = get_device()
        hidden_states = torch.randn(
            (num_tokens, 16), dtype=torch.bfloat16, device=device
        )
        router_logits = torch.randn(
            (num_tokens, self.NUM_EXPERTS), dtype=torch.bfloat16, device=device
        )
        return hidden_states, router_logits

    def _fuse(self, config, num_tokens=37, max_tokens=4096, **kwargs):
        hidden_states, router_logits = self._inputs(num_tokens)
        with patch.object(
            aiter_topk_module,
            "_get_aiter_topk_fuse_shared_max_tokens",
            return_value=max_tokens,
        ):
            return aiter_topk_module.try_select_experts(
                hidden_states, router_logits, config, **kwargs
            )

    def test_ineligible_configs_fall_back_without_touching_aiter(self):
        """Every guard in the eligibility predicate returns None, allocates no
        buffer, and never reaches the kernel."""
        config_cases = {
            "not_grouped_topk": {"use_grouped_topk": False},
            "no_correction_bias": {"correction_bias": None},
            "no_shared_experts": {"num_fused_shared_experts": 0, "top_k": 8},
            "no_expert_group": {"num_expert_group": None},
            "no_topk_group": {"topk_group": None},
            "scaling_on_output": {"apply_routed_scaling_factor_on_output": True},
        }
        for name, overrides in config_cases.items():
            with self.subTest(case=name), get_parallel().override(moe_ep_size=1):
                self.assertIsNone(self._fuse(self._config(**overrides)))

        call_cases = {
            "padded_tokens": {
                "num_token_non_padded": torch.tensor([17], device=get_device())
            },
            "expert_location_dispatch": {"expert_location_dispatch_info": object()},
        }
        for name, kwargs in call_cases.items():
            with self.subTest(case=name), get_parallel().override(moe_ep_size=1):
                self.assertIsNone(self._fuse(self._config(), **kwargs))

        with self.subTest(case="expert_parallel"):
            with get_parallel().override(moe_ep_size=2):
                self.assertIsNone(self._fuse(self._config()))

        with self.subTest(case="parallel_state_uninitialized"):
            with patch.object(
                aiter_topk_module,
                "get_parallel",
                side_effect=AssertionError("not initialized"),
            ):
                self.assertIsNone(self._fuse(self._config()))

        with self.subTest(case="per_rank_shared_slots"):
            with (
                get_parallel().override(moe_ep_size=1),
                patch.object(
                    aiter_topk_module,
                    "has_per_rank_fused_shared_slots",
                    return_value=True,
                ),
            ):
                self.assertIsNone(self._fuse(self._config()))

        for setting in (
            "SGLANG_SIMULATE_UNIFORM_EXPERTS",
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
            "SGLANG_K3_RADIX4_TOPK",
        ):
            with self.subTest(case=setting):
                with (
                    get_parallel().override(moe_ep_size=1),
                    patch.object(
                        getattr(aiter_topk_module.envs, setting),
                        "get",
                        return_value=True,
                    ),
                ):
                    self.assertIsNone(self._fuse(self._config()))

        with self.subTest(case="batch_over_buffer"):
            with get_parallel().override(moe_ep_size=1):
                self.assertIsNone(
                    self._fuse(self._config(), num_tokens=33, max_tokens=32)
                )

        self.assertEqual(self.fake_kernel.calls, [])
        self.assertFalse(aiter_topk_module._aiter_topk_fuse_shared_bufs)

    def test_shared_columns_are_prefilled_and_survive_the_kernel(self):
        """The returned rows carry the routed kernel output plus the prefilled
        shared columns, and the kernel only ever sees the routed sub-view."""
        # None and 1 are the two spellings of an unscaled shared expert.
        for factor in (None, 1, 0.25):
            with self.subTest(factor=factor):
                self.fake_kernel.calls.clear()
                aiter_topk_module._aiter_topk_fuse_shared_bufs.clear()
                config = self._config(fused_shared_experts_scaling_factor=factor)
                with get_parallel().override(moe_ep_size=1):
                    output = self._fuse(config, num_tokens=37)

                total = self.TOPK_ROUTED + self.N_SHARED
                self.assertEqual(tuple(output.topk_ids.shape), (37, total))
                self.assertEqual(tuple(output.topk_weights.shape), (37, total))

                expected_shared_ids = torch.arange(
                    self.NUM_EXPERTS,
                    self.NUM_EXPERTS + self.N_SHARED,
                    dtype=output.topk_ids.dtype,
                    device=output.topk_ids.device,
                ).expand(37, self.N_SHARED)
                self.assertTrue(
                    torch.equal(
                        output.topk_ids[:, self.TOPK_ROUTED :], expected_shared_ids
                    )
                )
                expected_weight = 1.0 if factor is None else float(factor)
                self.assertTrue(
                    torch.all(
                        output.topk_weights[:, self.TOPK_ROUTED :] == expected_weight
                    )
                )
                self.assertTrue(
                    torch.all(output.topk_ids[:, : self.TOPK_ROUTED] < self.NUM_EXPERTS)
                )

                # The kernel gets a strided view whose row stride spans the
                # shared columns; that stride is what removes the append kernel.
                (call,) = self.fake_kernel.calls
                self.assertEqual(call["weights_shape"], (37, self.TOPK_ROUTED))
                self.assertEqual(call["ids_shape"], (37, self.TOPK_ROUTED))
                self.assertEqual(call["weights_row_stride"], total)
                self.assertEqual(call["ids_row_stride"], total)
                self.assertEqual(call["weights_ptr"], output.topk_weights.data_ptr())
                self.assertEqual(call["ids_ptr"], output.topk_ids.data_ptr())

    def test_kernel_receives_the_routing_arguments(self):
        """Group config and renormalize pass through, a missing routed scaling
        factor becomes 1.0, and the bias is cast to the router logits dtype."""
        with get_parallel().override(moe_ep_size=1):
            self._fuse(
                self._config(num_expert_group=8, topk_group=4, renormalize=False)
            )
            self._fuse(self._config(routed_scaling_factor=None))

        first, second = self.fake_kernel.calls
        self.assertEqual(first["num_expert_group"], 8)
        self.assertEqual(first["topk_group"], 4)
        self.assertFalse(first["renormalize"])
        self.assertEqual(first["routed_scaling_factor"], 2.827)
        self.assertEqual(first["bias_dtype"], torch.bfloat16)
        self.assertEqual(second["routed_scaling_factor"], 1.0)

    def test_buffer_is_persistent_per_config_and_isolated_across_layers(self):
        """Addresses stay fixed across calls (the CUDA-graph replay
        requirement) and two layers never share one buffer."""
        config = self._config()
        other_layer = self._config()
        with get_parallel().override(moe_ep_size=1):
            first = self._fuse(config, num_tokens=37)
            second = self._fuse(config, num_tokens=64)
            third = self._fuse(other_layer, num_tokens=37)

        self.assertEqual(first.topk_ids.data_ptr(), second.topk_ids.data_ptr())
        self.assertEqual(first.topk_weights.data_ptr(), second.topk_weights.data_ptr())
        self.assertNotEqual(first.topk_ids.data_ptr(), third.topk_ids.data_ptr())
        self.assertNotEqual(
            first.topk_weights.data_ptr(), third.topk_weights.data_ptr()
        )
        self.assertEqual(len(aiter_topk_module._aiter_topk_fuse_shared_bufs), 2)

        # The larger batch reuses the same buffer, so the shared columns of the
        # rows it newly exposes must still hold the prefilled values.
        self.assertTrue(torch.all(second.topk_weights[:, self.TOPK_ROUTED :] == 1.0))
        self.assertTrue(
            torch.all(second.topk_ids[:, self.TOPK_ROUTED :] == self.NUM_EXPERTS)
        )

    def test_token_budget_tracks_the_prefill_configuration(self):
        """Buffer height follows chunked prefill, with the 8192 floor, the
        memory cap, and no caching before the schedule config is published."""
        cap = aiter_topk_module._AITER_TOPK_FUSE_SHARED_MAX_TOKENS_CAP

        def budget(chunked_prefill_size, max_prefill_tokens):
            aiter_topk_module._aiter_topk_fuse_shared_max_tokens_cache = None
            schedule = types.SimpleNamespace(
                chunked_prefill_size=chunked_prefill_size,
                max_prefill_tokens=max_prefill_tokens,
            )
            with patch(
                "sglang.srt.runtime_context.get_schedule", return_value=schedule
            ):
                return aiter_topk_module._get_aiter_topk_fuse_shared_max_tokens()

        self.assertEqual(budget(16384, 32768), 32768)
        self.assertEqual(budget(2048, 0), 8192)  # floor for tiny configs
        self.assertEqual(budget(0, 0), cap)  # chunked prefill disabled
        self.assertEqual(budget(cap * 4, cap * 4), cap)  # memory cap
        # Unpublished config (unit test / offline init): return the cap without
        # caching it, so a later call still picks up the real budget.
        aiter_topk_module._aiter_topk_fuse_shared_max_tokens_cache = None
        with patch(
            "sglang.srt.runtime_context.get_schedule",
            side_effect=ValueError("schedule config not published"),
        ):
            self.assertEqual(
                aiter_topk_module._get_aiter_topk_fuse_shared_max_tokens(), cap
            )
        self.assertIsNone(aiter_topk_module._aiter_topk_fuse_shared_max_tokens_cache)
        self.assertEqual(budget(16384, 16384), 16384)

    def test_select_experts_prefers_the_fusion_on_aiter_builds(self):
        """``select_experts`` returns the fused output when the fusion applies,
        and otherwise falls through to the ordinary routing path."""
        hidden_states, router_logits = self._inputs(37)
        config = self._config()
        total = self.TOPK_ROUTED + self.N_SHARED
        with (
            patch.object(topk_module, "_is_hip", True),
            patch.object(topk_module, "_use_aiter", True),
            patch.object(
                aiter_topk_module,
                "_get_aiter_topk_fuse_shared_max_tokens",
                return_value=4096,
            ),
            get_parallel().override(moe_ep_size=1),
        ):
            fused = topk_module.select_experts(hidden_states, router_logits, config)
            self.assertEqual(len(self.fake_kernel.calls), 1)
            self.assertEqual(tuple(fused.topk_ids.shape), (37, total))

            # Fall-through: the ordinary routing gate is stubbed because the
            # real one resolves to aiter symbols that a non-ROCm build never
            # imported; the shared-expert append after it is backend-neutral.
            routed = (
                torch.rand((37, self.TOPK_ROUTED), device=router_logits.device),
                torch.randint(
                    0,
                    self.NUM_EXPERTS,
                    (37, self.TOPK_ROUTED),
                    dtype=torch.int32,
                    device=router_logits.device,
                ),
            )
            with (
                patch.object(
                    aiter_topk_module, "try_select_experts", return_value=None
                ) as attempted,
                patch.object(
                    topk_module, "biased_grouped_topk", return_value=routed
                ) as ordinary_gate,
            ):
                fallback = topk_module.select_experts(
                    hidden_states, router_logits, config
                )
            attempted.assert_called_once()
            ordinary_gate.assert_called_once()
            self.assertEqual(len(self.fake_kernel.calls), 1)
            self.assertEqual(tuple(fallback.topk_ids.shape), (37, total))
            self.assertTrue(
                torch.equal(fallback.topk_ids[:, : self.TOPK_ROUTED], routed[1])
            )


if __name__ == "__main__":
    unittest.main()
