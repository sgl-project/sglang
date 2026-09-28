"""Unit tests for the mxfp8 moe_runner_backend constraints in
srt/arg_groups/overrides.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import (
    ResolvedView,
    _moe_runner_backend_quant_constraints,
)
from sglang.srt.runtime_context import override_platform
from sglang.test.test_utils import CustomTestCase


@contextlib.contextmanager
def _platform(*, is_hip: bool, is_gfx95: bool = False):
    """gfx95 is read straight from utils.common, not through the platform
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


if __name__ == "__main__":
    unittest.main()
