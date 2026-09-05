"""Unit tests for OpAuto policy (default-off, cold skip, sticky demote)."""

from unittest.mock import patch

from sglang.kernels.opauto import (
    can_use_or_demote,
    enable_opauto,
    get_policy,
    get_state,
    is_enabled,
    set_cold_skip_jit,
    should_prefer_native_aot_fallback,
    should_skip_cold_jit,
)
from sglang.kernels.opauto.state import BackendStatus
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOpAutoPolicy(CustomTestCase):
    def setUp(self) -> None:
        enable_opauto(False)
        set_cold_skip_jit(True)
        get_state().clear()
        get_policy()._loaded = False

    def tearDown(self) -> None:
        enable_opauto(False)
        get_state().clear()

    def test_disabled_matches_static_order(self):
        self.assertFalse(is_enabled())
        cands = [KernelBackend.JIT, KernelBackend.AOT, KernelBackend.TORCH]
        self.assertEqual(get_policy().pick("layernorm.rmsnorm", cands), cands)

    def test_disabled_probe_passthrough(self):
        calls = {"n": 0}

        def probe():
            calls["n"] += 1
            return True

        self.assertTrue(can_use_or_demote("op.x", probe))
        self.assertEqual(calls["n"], 1)
        self.assertFalse(get_state().is_failed("op.x", "jit"))

    def test_cold_skip_jit_on_pre_ampere(self):
        enable_opauto(True)
        set_cold_skip_jit(True)
        with patch(
            "sglang.kernels.jit.utils.arch.is_pre_ampere_cuda",
            return_value=True,
        ):
            self.assertTrue(should_skip_cold_jit("diffusion.qknorm"))
            cands = [KernelBackend.JIT, KernelBackend.AOT]
            picked = get_policy().pick("diffusion.qknorm", cands)
            self.assertNotIn(KernelBackend.JIT, picked)
            self.assertIn(KernelBackend.AOT, picked)

    def test_sticky_demote(self):
        enable_opauto(True)
        set_cold_skip_jit(False)

        def boom():
            raise RuntimeError("compile failed")

        self.assertFalse(can_use_or_demote("op.y", boom, backend="jit"))
        self.assertTrue(get_state().is_failed("op.y", "jit"))
        self.assertEqual(
            get_state().get("op.y", "jit").status, BackendStatus.FAILED
        )

        # Second call must not re-run probe.
        calls = {"n": 0}

        def probe():
            calls["n"] += 1
            return True

        self.assertFalse(can_use_or_demote("op.y", probe, backend="jit"))
        self.assertEqual(calls["n"], 0)

    def test_native_aot_fallback_pre_ampere(self):
        enable_opauto(True)
        set_cold_skip_jit(True)
        with patch(
            "sglang.kernels.jit.utils.arch.is_pre_ampere_cuda",
            return_value=True,
        ):
            self.assertTrue(should_prefer_native_aot_fallback("layernorm.rmsnorm"))

    def test_native_aot_fallback_off_when_disabled(self):
        enable_opauto(False)
        with patch(
            "sglang.kernels.jit.utils.arch.is_pre_ampere_cuda",
            return_value=True,
        ):
            self.assertFalse(should_prefer_native_aot_fallback("layernorm.rmsnorm"))
