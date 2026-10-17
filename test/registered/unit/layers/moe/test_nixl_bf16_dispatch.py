"""CPU regressions for NIXL BF16 MoE dispatch selection."""

import unittest
from unittest.mock import patch

from sglang.srt.layers.moe.ep_moe import layer as ep_moe
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNixlBf16Dispatch(CustomTestCase):
    def test_bf16_nixl_dispatch_uses_fused_moe(self):
        with (
            patch.object(ep_moe.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
            patch.object(
                ep_moe, "get_moe_a2a_backend", return_value=MoeA2ABackend.NIXL
            ),
            patch.object(
                ep_moe.envs.SGLANG_NIXL_EP_BF16_DISPATCH,
                "get",
                return_value=True,
            ),
        ):
            self.assertTrue(ep_moe._uses_fused_moe_for_nixl_bf16_dispatch())

    def test_fp8_nixl_dispatch_keeps_legacy_path(self):
        with (
            patch.object(ep_moe.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
            patch.object(
                ep_moe, "get_moe_a2a_backend", return_value=MoeA2ABackend.NIXL
            ),
            patch.object(
                ep_moe.envs.SGLANG_NIXL_EP_BF16_DISPATCH,
                "get",
                return_value=False,
            ),
        ):
            self.assertFalse(ep_moe._uses_fused_moe_for_nixl_bf16_dispatch())


if __name__ == "__main__":
    unittest.main()
