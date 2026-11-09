"""SGLANG_K3_TARGET_DENSE_FP8 resolves to an FP8 scope; auto is wide on gfx95
only, and off with the AMD monokernel."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import k3_rocm_dense_fp8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestResolveScope(CustomTestCase):
    def test_auto_is_wide_on_gfx95(self):
        self.assertEqual(
            k3_rocm_dense_fp8.resolve_scope("auto", gfx95=True, amd_monokernel=False),
            "wide",
        )

    def test_auto_is_off_elsewhere(self):
        self.assertEqual(
            k3_rocm_dense_fp8.resolve_scope("auto", gfx95=False, amd_monokernel=False),
            "off",
        )

    def test_auto_is_off_with_the_amd_monokernel(self):
        # The monokernel reads the MoE front and KDA projections as bf16.
        self.assertEqual(
            k3_rocm_dense_fp8.resolve_scope("auto", gfx95=True, amd_monokernel=True),
            "off",
        )

    def test_explicit_values_pass_through(self):
        for value in ("off", "front", "wide"):
            for gfx95 in (True, False):
                for amd_monokernel in (True, False):
                    self.assertEqual(
                        k3_rocm_dense_fp8.resolve_scope(value, gfx95, amd_monokernel),
                        value,
                    )

    def test_value_is_case_insensitive(self):
        self.assertEqual(
            k3_rocm_dense_fp8.resolve_scope("Wide", gfx95=False, amd_monokernel=False),
            "wide",
        )

    def test_unknown_value_raises(self):
        with self.assertRaisesRegex(ValueError, "auto\\|off\\|front\\|wide"):
            k3_rocm_dense_fp8.resolve_scope("all", gfx95=True, amd_monokernel=False)


class TestScopeFromEnv(CustomTestCase):
    def _scope(
        self, gfx95: bool, amd_monokernel: bool = False
    ) -> tuple[str, bool, bool]:
        exec_config = SimpleNamespace(
            kernel=SimpleNamespace(enable_amd_monokernel=amd_monokernel)
        )
        with (
            mock.patch.object(
                k3_rocm_dense_fp8, "is_gfx95_supported", return_value=gfx95
            ),
            mock.patch.object(k3_rocm_dense_fp8, "get_exec", return_value=exec_config),
        ):
            return (
                k3_rocm_dense_fp8.scope(),
                k3_rocm_dense_fp8.front_enabled(),
                k3_rocm_dense_fp8.qkvg_enabled(),
            )

    def test_default_is_auto(self):
        self.assertEqual(envs.SGLANG_K3_TARGET_DENSE_FP8.default, "auto")

    def test_unset_enables_wide_on_gfx95_only(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("SGLANG_K3_TARGET_DENSE_FP8", None)
            self.assertEqual(self._scope(gfx95=True), ("wide", True, True))
            self.assertEqual(self._scope(gfx95=False), ("off", False, False))

    def test_unset_is_off_with_the_amd_monokernel(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("SGLANG_K3_TARGET_DENSE_FP8", None)
            self.assertEqual(
                self._scope(gfx95=True, amd_monokernel=True), ("off", False, False)
            )

    def test_explicit_front_applies_with_the_amd_monokernel(self):
        with envs.SGLANG_K3_TARGET_DENSE_FP8.override("front"):
            self.assertEqual(
                self._scope(gfx95=True, amd_monokernel=True), ("front", True, False)
            )

    def test_explicit_off_disables_on_gfx95(self):
        with envs.SGLANG_K3_TARGET_DENSE_FP8.override("off"):
            self.assertEqual(self._scope(gfx95=True), ("off", False, False))

    def test_front_keeps_qkvg_bf16(self):
        with envs.SGLANG_K3_TARGET_DENSE_FP8.override("front"):
            self.assertEqual(self._scope(gfx95=True), ("front", True, False))


if __name__ == "__main__":
    unittest.main()
