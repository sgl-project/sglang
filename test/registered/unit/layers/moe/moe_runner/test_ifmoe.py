"""Unit coverage for the IFMOE fused MoE runner backend integration.

Mirrors `python/sglang/srt/layers/moe/moe_runner/ifmoe/`. These tests cover
the *integration wiring* of the new backend (enum / backend selection /
server-arg / quant-info dataclass / side-effect registration import) and
deliberately do NOT execute the CUDA kernel: the kernel JIT-compiles a large
.cu on first use and its numerical correctness is validated at the
server/E2E tier (see docs/ifmoe_vs_triton_e2e_report — MMLU N=256 parity)
and by isolated NCU profiling, not in a fast CPU unit test.
"""

import unittest

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    CustomTestCase = unittest.TestCase

    def register_cpu_ci(*args, **kwargs):
        pass


register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestIFMoeBackendWiring(CustomTestCase):
    def test_enum_and_is_ifmoe(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        self.assertTrue(hasattr(MoeRunnerBackend, "IFMOE"))
        ifmoe = MoeRunnerBackend.IFMOE
        self.assertTrue(ifmoe.is_ifmoe())
        # A different backend must not report as ifmoe.
        self.assertFalse(MoeRunnerBackend.TRITON.is_ifmoe())

    def test_server_arg_choice_accepts_ifmoe(self):
        from sglang.srt.server_args import MOE_RUNNER_BACKEND_CHOICES

        self.assertIn("ifmoe", MOE_RUNNER_BACKEND_CHOICES)

    def test_side_effect_registration_import(self):
        # runner.py constructs the ifmoe backend via this side-effect import;
        # it must succeed without a CUDA device (JIT is lazy, not at import).
        from sglang.srt.layers.moe.moe_runner import ifmoe

        self.assertTrue(hasattr(ifmoe, "__file__"))

    def test_quant_info_dataclass(self):
        import dataclasses

        from sglang.srt.layers.moe.moe_runner.base import MoeQuantInfo
        from sglang.srt.layers.moe.moe_runner.ifmoe.quant_info import (
            IFMoeQuantInfo,
        )

        self.assertTrue(issubclass(IFMoeQuantInfo, MoeQuantInfo))
        self.assertTrue(dataclasses.is_dataclass(IFMoeQuantInfo))
        field_names = {f.name for f in dataclasses.fields(IFMoeQuantInfo)}
        for required in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        ):
            self.assertIn(required, field_names)


if __name__ == "__main__":
    unittest.main()
