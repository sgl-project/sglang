"""On ROCm, --enable-deterministic-inference runs FP8 experts through the Triton MoE
runner instead of AITER. MXFP4 experts, MXFP8 experts and INT4 weights keep their
runner, and normal mode is unchanged."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers.quantization import fp8 as fp8_module
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _method(is_fp4_expert=False, use_mxfp8=False):
    method = Fp8MoEMethod.__new__(Fp8MoEMethod)
    method.is_fp4_expert = is_fp4_expert
    method.use_mxfp8 = use_mxfp8
    return method


def _exec(deterministic):
    return SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
    )


class TestDeterministicFp8MoeRunner(CustomTestCase):
    def test_fp8_experts_on_triton_only_in_deterministic_mode(self):
        # (deterministic, is_fp4_expert, use_mxfp8, hip_int4, expected)
        for det, fp4, mxfp8, int4, expected in [
            (True, False, False, False, True),
            (False, False, False, False, False),
            (True, True, False, False, False),
            (True, False, True, False, False),
            (True, False, False, True, False),
        ]:
            with (
                self.subTest(det=det, fp4=fp4, mxfp8=mxfp8, int4=int4),
                mock.patch(
                    "sglang.srt.runtime_context.get_exec", return_value=_exec(det)
                ),
                mock.patch.object(fp8_module, "_use_hip_int4", int4),
            ):
                self.assertEqual(
                    _method(fp4, mxfp8)._deterministic_fp8_experts_on_triton(),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
