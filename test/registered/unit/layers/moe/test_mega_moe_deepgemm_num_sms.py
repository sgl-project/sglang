"""Unit tests for Blackwell DeepGEMM MegaMoE SM-count selection."""

import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.moe import mega_moe  # noqa: E402


class FakeDeepGemm:
    def __init__(self, num_sms: int):
        self._num_sms = num_sms
        self.set_num_sms_calls = []

    def get_num_sms(self) -> int:
        return self._num_sms

    def set_num_sms(self, num_sms: int) -> None:
        self.set_num_sms_calls.append(num_sms)
        self._num_sms = num_sms


class TestMegaMoEDeepGemmNumSms(unittest.TestCase):
    def test_blackwell_reserves_two_sms_and_restores(self):
        deep_gemm = FakeDeepGemm(num_sms=152)

        with (
            mock.patch.object(mega_moe, "_device_sm", 103),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(2),
            mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm),
        ):
            self.assertEqual(deep_gemm.get_num_sms(), 150)

        self.assertEqual(deep_gemm.get_num_sms(), 152)
        self.assertEqual(deep_gemm.set_num_sms_calls, [150, 152])

    def test_reserve_selection(self):
        for reserved_num_sms, expected_num_sms in ((1, 150), (200, 2)):
            deep_gemm = FakeDeepGemm(num_sms=152)
            with (
                self.subTest(reserved_num_sms=reserved_num_sms),
                mock.patch.object(mega_moe, "_device_sm", 103),
                envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(
                    reserved_num_sms
                ),
                mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm),
            ):
                self.assertEqual(deep_gemm.get_num_sms(), expected_num_sms)

    def test_zero_reserve_keeps_all_sms(self):
        deep_gemm = FakeDeepGemm(num_sms=152)

        with (
            mock.patch.object(mega_moe, "_device_sm", 103),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(0),
            mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm),
        ):
            self.assertEqual(deep_gemm.get_num_sms(), 152)

        self.assertEqual(deep_gemm.set_num_sms_calls, [])

    def test_restores_count_when_forward_raises(self):
        deep_gemm = FakeDeepGemm(num_sms=152)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with (
                mock.patch.object(mega_moe, "_device_sm", 103),
                envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(2),
                mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm),
            ):
                raise RuntimeError("boom")

        self.assertEqual(deep_gemm.get_num_sms(), 152)
        self.assertEqual(deep_gemm.set_num_sms_calls, [150, 152])

    def test_sm90_keeps_full_device_count(self):
        deep_gemm = FakeDeepGemm(num_sms=132)

        with (
            mock.patch.object(mega_moe, "_device_sm", 90),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(2),
            mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm),
        ):
            self.assertEqual(deep_gemm.get_num_sms(), 132)

        self.assertEqual(deep_gemm.set_num_sms_calls, [])


if __name__ == "__main__":
    unittest.main()
