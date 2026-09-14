"""Unit tests for ``fused_append_shared_experts`` with a non-power-of-two topk.

``_fused_append_shared_experts_kernel`` previously indexed the routed and shared
lanes with ``tl.arange(0, K)`` / ``tl.arange(0, S)``, which Triton only accepts
for power-of-two ranges. DeepSeek-V4 routes top-6 (K=6), so enabling shared
experts fusion (``--enforce-shared-experts-fusion``) crashed with
``ValueError: arange's range must be a power of 2``. The kernel now iterates
over ``next_power_of_2`` blocks with masking, so these tests specifically cover
non-power-of-two K and S. The kernel is GPU-only (Triton), so the tests are
skipped when no accelerator is present.
"""

import unittest

import torch

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    fused_append_shared_experts,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")


def _reference_append(topk_ids, topk_weights, s, scale_factor, n_base):
    """Pure-torch golden reference: routed lanes pass through, shared lanes are
    appended as ``n_base + arange(s)`` with weight ``scale_factor``."""
    m, k = topk_ids.shape
    out_ids = torch.empty((m, k + s), dtype=topk_ids.dtype, device=topk_ids.device)
    out_w = torch.empty(
        (m, k + s), dtype=topk_weights.dtype, device=topk_weights.device
    )
    out_ids[:, :k] = topk_ids
    out_w[:, :k] = topk_weights
    shared = n_base + torch.arange(s, device=topk_ids.device)
    out_ids[:, k:] = shared.to(topk_ids.dtype)
    out_w[:, k:] = scale_factor
    return out_ids, out_w


@unittest.skipUnless(
    torch.cuda.is_available(), "fused_append_shared_experts kernel requires a GPU"
)
class TestFusedAppendSharedExpertsTop6(CustomTestCase):
    # (m, k, s). k and/or s are deliberately NON power-of-two -- the case that
    # used to crash. k=6 is the DeepSeek-V4 top-6 routing width.
    CASES = [
        (1, 6, 1),  # DSV4 top-6, single shared expert (the original crash)
        (4, 6, 1),
        (17, 6, 1),
        (128, 6, 1),
        (8, 6, 3),  # non-pow2 K and non-pow2 S together
        (33, 5, 2),  # non-pow2 K and non-pow2 S, odd M
        (4, 8, 1),  # power-of-two K still correct (regression guard)
    ]

    N_BASE = 256  # shared-expert base id (num routed experts)

    def _make_inputs(self, m, k, ids_dtype=torch.int64):
        device = get_device()
        g = torch.Generator(device="cpu").manual_seed(m * 1000 + k * 7 + 1)
        topk_ids = torch.randint(
            0, self.N_BASE, (m, k), generator=g, dtype=ids_dtype
        ).to(device)
        topk_weights = torch.rand((m, k), generator=g, dtype=torch.float32).to(device)
        return topk_ids, topk_weights

    def test_matches_golden_reference(self):
        """Kernel output equals routed-passthrough + shared-append, incl. K=6."""
        scale_factor = 0.5
        for m, k, s in self.CASES:
            with self.subTest(m=m, k=k, s=s):
                topk_ids, topk_weights = self._make_inputs(m, k)

                got_ids, got_w = fused_append_shared_experts(
                    topk_ids.clone(),
                    topk_weights.clone(),
                    s,
                    scale_factor,
                    N=self.N_BASE,
                )
                exp_ids, exp_w = _reference_append(
                    topk_ids, topk_weights, s, scale_factor, self.N_BASE
                )

                self.assertEqual(tuple(got_ids.shape), (m, k + s))
                self.assertEqual(tuple(got_w.shape), (m, k + s))
                self.assertTrue(torch.equal(got_ids, exp_ids))
                self.assertTrue(torch.allclose(got_w, exp_w))

    def test_routed_lanes_unmodified(self):
        """The first K columns must be the original routed ids/weights verbatim."""
        m, k, s = 16, 6, 1
        topk_ids, topk_weights = self._make_inputs(m, k)
        got_ids, got_w = fused_append_shared_experts(
            topk_ids.clone(), topk_weights.clone(), s, 1.0, N=self.N_BASE
        )
        self.assertTrue(torch.equal(got_ids[:, :k], topk_ids))
        self.assertTrue(torch.allclose(got_w[:, :k], topk_weights))

    def test_no_shared_experts_is_noop(self):
        """s == 0 returns the inputs untouched (no kernel launch)."""
        topk_ids, topk_weights = self._make_inputs(4, 6)
        got_ids, got_w = fused_append_shared_experts(
            topk_ids, topk_weights, 0, 1.0, N=self.N_BASE
        )
        self.assertTrue(torch.equal(got_ids, topk_ids))
        self.assertTrue(torch.equal(got_w, topk_weights))


if __name__ == "__main__":
    unittest.main()
