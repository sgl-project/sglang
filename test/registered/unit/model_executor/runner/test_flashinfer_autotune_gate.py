"""FlashInfer autotune gate for NVFP4 GEMMs.

compressed-tensors W4A4 NVFP4 linears call the same FlashInfer ``fp4_gemm`` as
modelopt NVFP4 checkpoints, so the gate must enable FP4 GEMM tactic autotune
for them too; otherwise every shape runs FlashInfer's default tactic.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
)
from sglang.srt.model_executor.runner import flashinfer_autotune as autotune
from sglang.test.test_utils import CustomTestCase


def _model_with_schemes(*schemes) -> torch.nn.Module:
    model = torch.nn.Module()
    for index, scheme in enumerate(schemes):
        layer = torch.nn.Module()
        layer.scheme = scheme
        model.add_module(f"layer{index}", layer)
    return model


def _exec_config() -> SimpleNamespace:
    return SimpleNamespace(
        kernel=SimpleNamespace(disable_flashinfer_autotune=False),
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
        moe=SimpleNamespace(moe_runner_backend="auto", moe_a2a_backend="none"),
    )


def _model_runner(quantization: str, model) -> SimpleNamespace:
    return SimpleNamespace(
        device="cuda",
        model_config=SimpleNamespace(quantization=quantization),
        model=model,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
        is_draft_worker=False,
    )


class TestCompressedTensorsNvfp4Detection(CustomTestCase):
    def test_no_model(self):
        self.assertFalse(autotune._model_uses_compressed_tensors_nvfp4(None))

    def test_model_without_schemes(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        self.assertFalse(autotune._model_uses_compressed_tensors_nvfp4(model))

    def test_other_compressed_tensors_scheme(self):
        model = _model_with_schemes(object())
        self.assertFalse(autotune._model_uses_compressed_tensors_nvfp4(model))

    def test_nvfp4_scheme_in_any_layer(self):
        model = _model_with_schemes(object(), CompressedTensorsW4A4Fp4())
        self.assertTrue(autotune._model_uses_compressed_tensors_nvfp4(model))


class TestFp4GemmAutotuneGate(CustomTestCase):
    def _should_run(self, quantization: str, model, *, cutlass: bool = True) -> bool:
        fp4_backend = Mock()
        fp4_backend.is_flashinfer_cutlass.return_value = cutlass
        fp4_backend.is_flashinfer_cutedsl.return_value = False
        with (
            patch.object(autotune, "get_exec", return_value=_exec_config()),
            patch(
                "sglang.srt.layers.quantization.fp4_utils.get_fp4_gemm_runner_backend",
                return_value=fp4_backend,
            ),
            patch("torch.cuda.get_device_capability", return_value=(12, 1)),
        ):
            return autotune.should_run_flashinfer_autotune(
                _model_runner(quantization, model)
            )

    def test_compressed_tensors_nvfp4_runs_autotune(self):
        model = _model_with_schemes(CompressedTensorsW4A4Fp4())
        self.assertTrue(self._should_run("compressed-tensors", model))

    def test_compressed_tensors_without_nvfp4_skips_autotune(self):
        model = _model_with_schemes(object())
        self.assertFalse(self._should_run("compressed-tensors", model))

    def test_compressed_tensors_nvfp4_needs_flashinfer_fp4_backend(self):
        model = _model_with_schemes(CompressedTensorsW4A4Fp4())
        self.assertFalse(self._should_run("compressed-tensors", model, cutlass=False))

    def test_modelopt_fp4_still_runs_autotune(self):
        self.assertTrue(self._should_run("modelopt_fp4", None))


if __name__ == "__main__":
    unittest.main()
