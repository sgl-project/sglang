"""Dropping other EP ranks' routes must stay bit-identical to computing them.

Under expert parallelism without an all-to-all, routes to non-local experts carry
id -1. The Triton path drops them at alignment, quantizes only routed rows, and
the top-k combine skips their slots instead of summing zeros written for them.
Adding +0.0 to an fp32 sum that starts at +0.0 is exact, so the whole layer is
compared bitwise against the path that still computes those routes.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

E_GLOBAL, EP, TOPK, HIDDEN, INTER, BLOCK = 256, 8, 8, 2048, 512, [128, 128]
E = E_GLOBAL // EP
NUM_TOKENS = (1, 33, 300, 2048)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMoeSkipNonlocalRoutes(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.weights = {
            False: (
                (torch.randn(E, 2 * INTER, HIDDEN, device="cuda") / HIDDEN**0.5).to(
                    torch.bfloat16
                ),
                (torch.randn(E, HIDDEN, INTER, device="cuda") / INTER**0.5).to(
                    torch.bfloat16
                ),
                None,
                None,
            ),
            True: (
                torch.randn(E, 2 * INTER, HIDDEN, device="cuda").to(
                    torch.float8_e4m3fn
                ),
                torch.randn(E, HIDDEN, INTER, device="cuda").to(torch.float8_e4m3fn),
                torch.rand(E, 2 * INTER // 128, HIDDEN // 128, device="cuda") / 32
                + 0.01,
                torch.rand(E, HIDDEN // 128, INTER // 128, device="cuda") / 16 + 0.02,
            ),
        }

    def _layer(self, m, fp8, routed_scaling_factor, skip, defer_sum=False):
        from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe

        torch.manual_seed(m)
        hidden = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        global_ids = torch.rand(m, E_GLOBAL, device="cuda").topk(TOPK, dim=-1).indices
        # This rank owns experts [0, E); every other route is -1.
        topk_ids = torch.where(global_ids < E, global_ids, -1).to(torch.int32)
        topk_weights = torch.rand(m, TOPK, device="cuda").softmax(-1)
        w1, w2, w1_s, w2_s = self.weights[fp8]
        gate = fused_moe._skip_nonlocal_routes if skip else (lambda **kw: False)
        with mock.patch.object(fused_moe, "_skip_nonlocal_routes", gate):
            return fused_moe.fused_experts_impl(
                hidden,
                w1,
                w2,
                topk_weights,
                topk_ids,
                use_fp8_w8a8=fp8,
                w1_scale=w1_s,
                w2_scale=w2_s,
                block_shape=BLOCK if fp8 else None,
                routed_scaling_factor=routed_scaling_factor,
                filter_expert=True,
                defer_sum=defer_sum,
            )

    def test_skipping_nonlocal_routes_is_exact(self):
        with (
            get_context().override_server_args(enable_deterministic_inference=False),
            get_parallel().override(tp_group=None),
        ):
            for fp8 in (True, False):
                for m in NUM_TOKENS:
                    for rsf in (None, 2.5):
                        with self.subTest(fp8=fp8, m=m, routed_scaling_factor=rsf):
                            skip = self._layer(m, fp8, rsf, skip=True)
                            ref = self._layer(m, fp8, rsf, skip=False)
                            self.assertFalse(skip.isnan().any())
                            self.assertTrue(torch.equal(skip, ref))

    def test_defer_sum_keeps_nonlocal_routes(self):
        # The caller sums these slots itself, so none of them may be left unwritten.
        with (
            get_context().override_server_args(enable_deterministic_inference=False),
            get_parallel().override(tp_group=None),
        ):
            skip = self._layer(2048, False, None, skip=True, defer_sum=True)
            ref = self._layer(2048, False, None, skip=False, defer_sum=True)
        self.assertTrue(torch.equal(skip, ref))


if __name__ == "__main__":
    unittest.main()
