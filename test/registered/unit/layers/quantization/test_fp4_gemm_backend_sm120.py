"""Unit test for the NVFP4 GEMM backend that ``auto`` resolves to on SM120.

FlashInfer ships a dense NVFP4 GEMM built for SM120/SM121 (``mm_fp4`` backend
``"b12x"``) and prefers it in its own ``"auto"`` order there. SGLang's
``initialize_fp4_gemm_config`` mapped ``auto`` to ``flashinfer_cutlass`` on
every non-SM100 Blackwell part, so the faster kernel was never selected and no
CLI choice could ask for it. These tests pin the resolution table so a later
edit cannot silently send SM120 back to the CUTLASS kernel, and check that the
new choice maps to the FlashInfer API name. They mock the device probes so they
run on CPU CI.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.layers.quantization.fp4_utils as fp4_utils
import sglang.srt.model_executor.runner.flashinfer_autotune as autotune
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.test.test_utils import CustomTestCase


def _resolve(requested, *, capability, is_sm100=False, cuda=True):
    exec_ns = SimpleNamespace(kernel=SimpleNamespace(fp4_gemm_runner_backend=requested))
    platform_ns = SimpleNamespace(is_sm100=is_sm100)
    fp4_utils.FP4_GEMM_RUNNER_BACKEND = None
    try:
        with (
            patch.object(fp4_utils, "get_exec", return_value=exec_ns),
            patch.object(fp4_utils, "get_platform", return_value=platform_ns),
            patch.object(fp4_utils, "is_cuda", return_value=cuda),
            patch.object(fp4_utils, "get_device_capability", return_value=capability),
        ):
            fp4_utils.initialize_fp4_gemm_config()
            return fp4_utils.get_fp4_gemm_runner_backend()
    finally:
        fp4_utils.FP4_GEMM_RUNNER_BACKEND = None


class TestFp4GemmBackendAuto(CustomTestCase):
    def test_sm120_auto_selects_b12x(self):
        self.assertEqual(
            _resolve("auto", capability=(12, 0)), Fp4GemmRunnerBackend.FLASHINFER_B12X
        )

    def test_sm121_auto_selects_b12x(self):
        self.assertEqual(
            _resolve("auto", capability=(12, 1)), Fp4GemmRunnerBackend.FLASHINFER_B12X
        )

    def test_sm100_auto_keeps_cutedsl(self):
        self.assertEqual(
            _resolve("auto", capability=(10, 0), is_sm100=True),
            Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL,
        )

    def test_sm90_auto_keeps_marlin(self):
        self.assertEqual(
            _resolve("auto", capability=(9, 0)), Fp4GemmRunnerBackend.MARLIN
        )

    def test_explicit_choice_wins_on_sm120(self):
        self.assertEqual(
            _resolve("flashinfer_cutlass", capability=(12, 0)),
            Fp4GemmRunnerBackend.FLASHINFER_CUTLASS,
        )

    def test_b12x_maps_to_flashinfer_api_name(self):
        self.assertEqual(
            Fp4GemmRunnerBackend.FLASHINFER_B12X.get_flashinfer_backend(), "b12x"
        )

    def test_b12x_predicate(self):
        self.assertTrue(Fp4GemmRunnerBackend.FLASHINFER_B12X.is_flashinfer_b12x())
        self.assertFalse(Fp4GemmRunnerBackend.FLASHINFER_CUTLASS.is_flashinfer_b12x())


def _autotune_gate(backend, *, quantization="modelopt_fp4"):
    """Run ``should_run_flashinfer_autotune`` with only the FP4 GEMM term live.

    The MoE backend is ``triton`` and the model is NVFP4, so the MoE and FP8
    terms are both False and the result is the FP4 GEMM term alone.
    """
    exec_ns = SimpleNamespace(
        kernel=SimpleNamespace(disable_flashinfer_autotune=False),
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
        moe=SimpleNamespace(moe_runner_backend="triton", moe_a2a_backend="none"),
    )
    runner = SimpleNamespace(
        device="cuda",
        model_config=SimpleNamespace(quantization=quantization),
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
        is_draft_worker=False,
    )
    fp4_utils.FP4_GEMM_RUNNER_BACKEND = backend
    try:
        with (
            patch.object(autotune, "get_exec", return_value=exec_ns),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        ):
            return autotune.should_run_flashinfer_autotune(runner)
    finally:
        fp4_utils.FP4_GEMM_RUNNER_BACKEND = None


class TestFp4AutotuneGate(CustomTestCase):
    def test_b12x_keeps_the_fp4_autotune_pass(self):
        self.assertTrue(_autotune_gate(Fp4GemmRunnerBackend.FLASHINFER_B12X))

    def test_cutlass_still_autotunes(self):
        self.assertTrue(_autotune_gate(Fp4GemmRunnerBackend.FLASHINFER_CUTLASS))

    def test_marlin_does_not_autotune(self):
        self.assertFalse(_autotune_gate(Fp4GemmRunnerBackend.MARLIN))


if __name__ == "__main__":
    unittest.main()
