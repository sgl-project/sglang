import os
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(
    torch.version.hip
    and torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    "gfx950 (MI35x) only",
)
class TestSmallMRouterGfx950(CustomTestCase):
    def _ints(self, *shape, bound):
        return torch.randint(-bound, bound + 1, shape, device="cuda").bfloat16()

    def test_router(self):
        from sglang.kernels.ops.moe import smallm_router_gfx950 as R
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
            fused_append_shared_experts_with_weights,
        )
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

        torch.manual_seed(0)
        # multiples of 2^-6 times small ints: every dot product is exact in fp32, so the gate GEMM rounding
        # is order-independent and the bf16 logits carry many exact ties (lowest expert id must win)
        wg, ws = self._ints(512, 4096, bound=1) / 64, self._ints(1, 4096, bound=1) / 64
        for tok in (1, 3, 4, 8):
            x = self._ints(tok, 4096, bound=2)
            self.assertTrue(R.smallm_router_supported(x))
            w, ids = R.smallm_router(x, wg, ws, 0.5)
            logits = (x.double() @ wg.double().T).bfloat16()
            ref_w, ref_ids = moe_fused_gate(logits, None, 10, scoring_func="softmax")
            ref_ids, ref_w = fused_append_shared_experts_with_weights(
                ref_ids,
                ref_w,
                None,
                1,
                N=512,
                fuse_gate=True,
                hidden_states=x,
                gate_weight=ws,
                scale=0.5,
            )
            self.assertTrue(torch.equal(ids, ref_ids), f"tok={tok}")
            torch.testing.assert_close(w, ref_w, atol=1e-5, rtol=1e-5)

        for bad in (
            self._ints(9, 4096, bound=2),
            self._ints(41, 4096, bound=2),
            x.half(),
            x[:, :2048],
        ):
            self.assertFalse(R.smallm_router_supported(bad))
        os.environ["SGLANG_ROCM_SMALLM_ROUTER"] = "0"
        try:
            self.assertFalse(R.smallm_router_supported(x))
        finally:
            del os.environ["SGLANG_ROCM_SMALLM_ROUTER"]

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gw, gids = R.smallm_router(x, wg, ws, 0.5)
        g.replay()
        g.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(gids, ids) and torch.equal(gw, w))


if __name__ == "__main__":
    unittest.main()
