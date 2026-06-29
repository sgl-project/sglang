"""Unit tests for the fused append + DeepEP-remap shared-experts Triton kernel.

Covers ``fused_append_remap_shared_experts_deepep``, which collapses
``fused_append_shared_experts()`` followed by ``_remap_topk_for_deepep()`` into a
single Triton launch on the aiter/DeepEP-class path. The kernel is GPU-only
(Triton), so these tests are skipped when no accelerator is present.
"""

import unittest

import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
    fused_append_remap_shared_experts_deepep,
    fused_append_shared_experts,
)
from sglang.srt.layers.moe.topk import TopKConfig, _remap_topk_for_deepep, _use_aiter
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

    Routed IDs:   e -> e + e // num_local_routed
    Shared IDs:   shared_id_base + arange(s)
    Routed wgt:   passthrough
    Shared wgt:   scale_factor
    """
    m, k = topk_ids.shape
    out_ids = torch.empty((m, k + s), dtype=topk_ids.dtype, device=topk_ids.device)
    out_w = torch.empty(
        (m, k + s), dtype=topk_weights.dtype, device=topk_weights.device
    )
    out_ids[:, :k] = topk_ids + topk_ids // num_local_routed
    out_w[:, :k] = topk_weights
    shared = shared_id_base + torch.arange(s, device=topk_ids.device)
    out_ids[:, k:] = shared.to(topk_ids.dtype)
    out_w[:, k:] = scale_factor
    return out_ids, out_w


@unittest.skipUnless(
    torch.cuda.is_available(), "fused append+remap kernel requires a GPU"
)
class TestFusedAppendRemapDeepEP(CustomTestCase):
    # (m, k, num_physical_routed, ep_size, ep_rank, num_fused_shared_experts).
    # k and num_fused_shared_experts are kept powers of two (tl.arange constraint).
    CASES = [
        (1, 8, 256, 8, 0, 1),
        (4, 8, 256, 8, 7, 1),
        (17, 8, 264, 8, 3, 1),
        (128, 16, 128, 4, 2, 2),
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

    def test_equivalence_with_eager_append_then_remap(self):
        """Fused kernel == fused_append_shared_experts() + _remap_topk_for_deepep().

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
                    eager_ids, eager_w = _remap_topk_for_deepep(
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
