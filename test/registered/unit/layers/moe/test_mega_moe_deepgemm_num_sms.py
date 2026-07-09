"""Unit tests for DeepGEMM MegaMoE SM-count selection."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.environ import envs
from sglang.srt.layers.moe import mega_moe
from sglang.test.test_utils import CustomTestCase


class FakeDeepGemm:
    def __init__(self, num_sms: int):
        self._num_sms = num_sms
        self.set_num_sms_calls = []

    def get_num_sms(self) -> int:
        return self._num_sms

    def set_num_sms(self, num_sms: int) -> None:
        self.set_num_sms_calls.append(num_sms)
        self._num_sms = num_sms


class TestMegaMoEDeepGemmNumSms(CustomTestCase):
    def test_reserves_even_sms_and_restores_original(self):
        deep_gemm = FakeDeepGemm(num_sms=152)

        with (
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_SMS.override(0),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(2),
        ):
            with mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm) as num_sms:
                self.assertEqual(num_sms, 150)
                self.assertEqual(deep_gemm.get_num_sms(), 150)

        self.assertEqual(deep_gemm.get_num_sms(), 152)
        self.assertEqual(deep_gemm.set_num_sms_calls, [150, 152])

    def test_rounds_reserved_sms_down_to_even_count(self):
        deep_gemm = FakeDeepGemm(num_sms=148)

        with (
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_SMS.override(0),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(1),
        ):
            with mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm) as num_sms:
                self.assertEqual(num_sms, 146)

        self.assertEqual(deep_gemm.set_num_sms_calls, [146, 148])

    def test_explicit_num_sms_wins_over_reserved_sms(self):
        deep_gemm = FakeDeepGemm(num_sms=152)

        with (
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_SMS.override(144),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(2),
        ):
            with mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm) as num_sms:
                self.assertEqual(num_sms, 144)

        self.assertEqual(deep_gemm.set_num_sms_calls, [144, 152])

    def test_explicit_num_sms_is_rounded_down_to_even_count(self):
        deep_gemm = FakeDeepGemm(num_sms=152)

        with (
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_SMS.override(147),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(0),
        ):
            with mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm) as num_sms:
                self.assertEqual(num_sms, 146)

        self.assertEqual(deep_gemm.set_num_sms_calls, [146, 152])

    def test_does_not_choose_one_sm_for_clustered_kernel(self):
        deep_gemm = FakeDeepGemm(num_sms=148)

        with (
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_SMS.override(0),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(200),
        ):
            with mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm) as num_sms:
                self.assertEqual(num_sms, 2)

        self.assertEqual(deep_gemm.set_num_sms_calls, [2, 148])

    def test_zero_reserve_leaves_deepgemm_unchanged(self):
        deep_gemm = FakeDeepGemm(num_sms=148)

        with (
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_SMS.override(0),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(0),
        ):
            with mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm) as num_sms:
                self.assertEqual(num_sms, 148)

        self.assertEqual(deep_gemm.set_num_sms_calls, [])


if __name__ == "__main__":
    unittest.main()
