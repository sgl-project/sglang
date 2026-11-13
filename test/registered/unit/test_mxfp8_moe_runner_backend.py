"""Check the MoE backend quantization constraints and the SAIL A2A admission
boundary."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import contextlib
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_override_base import ResolvedView
from sglang.srt.arg_groups.overrides import (
    _a2a_backend_overrides,
    _moe_runner_backend_quant_constraints,
    declare_resolution,
)
from sglang.srt.arg_groups.platform_hook import handle_sail_backends
from sglang.srt.environ import envs
from sglang.srt.runtime_context import override_platform
from sglang.test.test_utils import CustomTestCase


@contextlib.contextmanager
def _platform(*, is_hip: bool, is_gfx95: bool = False):
    """gfx95 is read straight from utils.device, not through the platform
    probes, so it is patched where overrides.py imported it."""
    with override_platform(is_hip=is_hip, is_npu=False):
        with patch(
            "sglang.srt.arg_groups.overrides.is_gfx95_supported",
            return_value=is_gfx95,
        ):
            yield


def _resolve(backend: str, quantization: str = "mxfp8") -> str:
    """Run the pass and return the backend it resolves to. The pass declares
    only what it changes, so an empty declaration means the request survived.

    moe_a2a_backend is "none" throughout: the pass reads it to pick the default
    (flashinfer_megamoe a2a forces a matching runner), and that interaction is
    out of scope here.
    """
    view = ResolvedView(
        SimpleNamespace(
            moe_runner_backend=backend,
            quantization=quantization,
            moe_a2a_backend="none",
        )
    )
    declared = _moe_runner_backend_quant_constraints(view)
    return declared.get("moe_runner_backend", backend)


class TestMxfp8MoeRunnerBackend(CustomTestCase):
    def test_rocm_honors_explicit_triton(self):
        # triton is the only mxfp8 runner ROCm can use: every other entry in
        # MXFP8_MOE_RUNNER_BACKEND_CHOICES is CUDA-only, and flashinfer_trtllm's
        # MoE apply path imports flashinfer, which is absent on ROCm.
        for is_gfx95 in (True, False):
            with self.subTest(is_gfx95=is_gfx95):
                with _platform(is_hip=True, is_gfx95=is_gfx95):
                    self.assertEqual(_resolve("triton"), "triton")

    def test_cuda_still_rejects_triton(self):
        with _platform(is_hip=False):
            self.assertEqual(_resolve("triton"), "flashinfer_trtllm")

    def test_auto_default_is_unchanged(self):
        # The fix widens `allowed`, not the default: "auto" must resolve exactly
        # as it did before on every platform.
        cases = (
            (dict(is_hip=True, is_gfx95=True), "triton"),
            (dict(is_hip=True, is_gfx95=False), "flashinfer_trtllm"),
            (dict(is_hip=False, is_gfx95=False), "flashinfer_trtllm"),
        )
        for facts, expected in cases:
            with self.subTest(**facts):
                with _platform(**facts):
                    self.assertEqual(_resolve("auto"), expected)

    def test_unsupported_backend_is_still_overridden(self):
        # The widened `allowed` must not turn the gate into a pass-through: a
        # name outside MXFP8_MOE_RUNNER_BACKEND_CHOICES still falls to the
        # platform default.
        with _platform(is_hip=True, is_gfx95=False):
            self.assertEqual(_resolve("marlin"), "flashinfer_trtllm")

    def test_non_mxfp8_quantization_is_untouched(self):
        with _platform(is_hip=True, is_gfx95=False):
            self.assertEqual(_resolve("triton", quantization="fp8"), "triton")


class TestSailMoeA2aBackend(CustomTestCase):
    """An emulated CUDA capability must not let SAIL pick a MegaMoE it cannot run.

    The refusal lives in ``handle_sail_backends``, not in the platform-agnostic
    ``_a2a_backend_overrides`` pass, so the cases below drive the hook the same
    way the resolution pipeline does: run the A2A pass, then the hook.
    """

    def setUp(self):
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        # The hook switches both DeepGEMM toggles on and gates its MXFP4 W4A16
        # checks off these vars; keep all four out of the process-global env.
        for var in (
            envs.SGLANG_SAIL_DEEPGEMM_DENSE,
            envs.SGLANG_SAIL_DEEPGEMM_MOE,
            envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
            envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA,
        ):
            stack.enter_context(var.override(False))

    @staticmethod
    def _run_hook(backend, waterfill=False):
        args = SimpleNamespace(
            moe_a2a_backend=backend,
            moe_runner_backend="auto",
            enable_waterfill=waterfill,
        )
        declared = _a2a_backend_overrides(ResolvedView(args))
        if declared:
            declare_resolution(args, "_a2a_backend_overrides", **declared)
        # The hook only reads MLA afterwards to tweak prefill CUDA graphs, which
        # needs a real checkpoint; that branch is not under test here.
        with patch(
            "sglang.srt.arg_groups.platform_hook.use_mla_backend",
            return_value=False,
        ):
            handle_sail_backends(args)
        return args

    def test_sail_rejects_megamoe_with_or_without_waterfill(self):
        # Waterfill keeps megamoe in _a2a_backend_overrides, so only this hook
        # can stop it reaching a communicator SAIL cannot drive.
        with override_platform(is_sail=True):
            for waterfill in (False, True):
                with (
                    self.subTest(waterfill=waterfill),
                    self.assertRaisesRegex(ValueError, "SAIL.*MegaMoE"),
                ):
                    self._run_hook("megamoe", waterfill)

    def test_sail_preserves_supported_backends_and_waterfill_resolution(self):
        cases = (
            ("none", False, "none"),
            ("deepep", False, "deepep"),
            ("none", True, "deepep"),
            ("deepep", True, "deepep"),
        )
        with override_platform(is_sail=True):
            for backend, waterfill, expected in cases:
                with self.subTest(backend=backend, waterfill=waterfill):
                    args = self._run_hook(backend, waterfill)
                    # The record keeps what the operator typed; the pass declares
                    # the waterfill rewrite and the hook must accept it.
                    self.assertEqual(args.moe_a2a_backend, backend)
                    self.assertEqual(ResolvedView(args).moe_a2a_backend, expected)

    def test_other_platforms_keep_megamoe(self):
        with override_platform(is_sail=False):
            args = SimpleNamespace(
                moe_a2a_backend="megamoe",
                moe_runner_backend="auto",
                enable_waterfill=True,
            )
            handle_sail_backends(args)
            self.assertEqual(ResolvedView(args).moe_a2a_backend, "megamoe")


if __name__ == "__main__":
    unittest.main()
