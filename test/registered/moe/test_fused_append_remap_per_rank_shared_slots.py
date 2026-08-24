"""Unit tests for fused append + per-rank shared-slot remap.

Covers ``fused_append_remap_shared_experts_deepep``, which collapses
``fused_append_shared_experts()`` followed by
``remap_topk_for_per_rank_shared_slots()`` into a
single Triton launch on the per-rank shared-slot path. The kernel is GPU-only
(Triton), so these tests are skipped when no accelerator is present.
"""

import unittest

import torch

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    fused_append_remap_shared_experts_deepep,
    fused_append_shared_experts,
)
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    _use_aiter,
    remap_topk_for_per_rank_shared_slots,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")
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


if __name__ == "__main__":
    unittest.main()
