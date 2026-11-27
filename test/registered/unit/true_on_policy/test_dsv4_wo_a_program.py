"""TOP DeepSeek-V4: serving and trainer read one decision for the wo_a program (FP8 GEMM or BF16 dequant)."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.true_on_policy.dsv4 import fp8_wo_a_program

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _program(*, flag, sm90, ue8m0, jit=True):
    """``flag`` None leaves SGLANG_OPT_FP8_WO_A_GEMM unset (its default is on)."""
    with (
        patch.dict(os.environ),
        patch("sglang.srt.layers.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0", ue8m0),
        patch("sglang.srt.layers.deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM", jit),
        patch(
            "sglang.srt.runtime_context.get_platform",
            return_value=SimpleNamespace(is_sm90=sm90),
        ),
    ):
        envs.SGLANG_OPT_FP8_WO_A_GEMM.clear()
        if flag is not None:
            envs.SGLANG_OPT_FP8_WO_A_GEMM.set(flag)
        return fp8_wo_a_program()


class TestFP8WoAProgram(CustomTestCase):
    def test_blackwell_default_runs_fp8(self):
        self.assertTrue(_program(flag=None, sm90=False, ue8m0=True))

    def test_disabling_the_gemm_selects_bf16_on_any_architecture(self):
        self.assertFalse(_program(flag=False, sm90=False, ue8m0=True))
        self.assertFalse(_program(flag=False, sm90=True, ue8m0=False))

    def test_sm90_needs_an_explicit_opt_in(self):
        self.assertFalse(_program(flag=None, sm90=True, ue8m0=False))
        self.assertTrue(_program(flag=True, sm90=True, ue8m0=False))

    def test_sm90_without_jit_deepgemm_dequantizes(self):
        self.assertFalse(_program(flag=True, sm90=True, ue8m0=False, jit=False))


if __name__ == "__main__":
    unittest.main()
