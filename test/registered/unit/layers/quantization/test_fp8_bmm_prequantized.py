"""A pre-quantized activation must not be rescaled twice or lose its output dtype."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrequantizedBmm(CustomTestCase):
    def test_scale_dtype_shape_and_bias(self):
        # Isolate this adapter from GPU-only module imports; the dependency
        # boundaries execute an equivalent dequantized matmul on CPU.
        path = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/srt/layers/quantization/fp8_utils.py"
        )
        tree = ast.parse(path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "apply_fp8_linear_bmm_flashinfer"
        )

        def quantize(x, scale, repeat_scale=False):
            return (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale

        def bmm(a, b, scale_a, scale_b, out_dtype):
            return ((a.float() * scale_a) @ (b.float() * scale_b)).to(out_dtype)

        namespace = dict(
            torch=torch,
            Optional=Optional,
            static_quant_fp8=quantize,
            flashinfer_bmm_fp8=bmm,
        )
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        apply = namespace[function.name]
        values = torch.arange(-12, 12, dtype=torch.float32).reshape(2, 3, 4)
        weight = (
            torch.arange(-10, 10, dtype=torch.float32)
            .reshape(4, 5)
            .to(torch.float8_e4m3fn)
        )
        scale_a, scale_b = torch.tensor([0.25]), torch.tensor([0.5])
        bias = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        for dtype, override in (
            (torch.float8_e4m3fn, torch.float16),
            (torch.float8_e4m3fn, torch.bfloat16),
            (torch.float8_e4m3fn, None),
            (torch.bfloat16, torch.float16),
            (torch.float16, torch.bfloat16),
        ):
            with self.subTest(dtype=dtype, override=override):
                x = values.to(dtype)
                if dtype == torch.float8_e4m3fn:
                    reference_input = x.float() * scale_a
                    output_dtype = override or torch.bfloat16
                else:
                    reference_input = quantize(x, scale_a)[0].float() * scale_a
                    output_dtype = dtype
                expected = (
                    reference_input.reshape(-1, 4) @ (weight.float() * scale_b)
                ).to(output_dtype)
                case_bias = bias.to(output_dtype)
                expected = (expected + case_bias).reshape(2, 3, 5)
                actual = apply(
                    x,
                    weight,
                    scale_b,
                    scale_a,
                    case_bias,
                    pre_quant_output_dtype=override,
                )
                self.assertEqual(actual.dtype, expected.dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_modelopt_bmm_linear_takes_prequantized_input(self):
        # GDN_PREFILL_NORM_FP8 hands out_proj E4M3 bytes at its static input scale; the
        # ModelOpt FlashInfer bmm path must pass them through and return the layer dtype.
        from sglang.srt.layers.quantization import fp8_utils
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptFp8LinearMethod,
        )

        def bmm(a, b, scale_a, scale_b, out_dtype):
            return ((a.float() * scale_a) @ (b.float() * scale_b)).to(out_dtype)

        def requantize(*args, **kwargs):
            raise AssertionError("the pre-quantized input was quantized again")

        method = object.__new__(ModelOptFp8LinearMethod)
        method.use_marlin = method.use_sm120_fp8 = False
        weight = torch.arange(-10, 10, dtype=torch.float32).reshape(4, 5)
        layer = SimpleNamespace(
            use_flashinfer_bmm=True,
            weight=weight.to(torch.float8_e4m3fn),
            weight_scale=torch.tensor([0.5]),
            input_scale=torch.tensor([0.25]),
            orig_dtype=torch.bfloat16,
        )
        x = torch.arange(-12, 12, dtype=torch.float32).reshape(6, 4)
        x = x.to(torch.float8_e4m3fn)
        with (
            patch.object(fp8_utils, "flashinfer_bmm_fp8", bmm, create=True),
            patch.object(fp8_utils, "static_quant_fp8", requantize),
        ):
            actual = method.apply(layer, x)
        expected = ((x.float() * 0.25) @ (weight * 0.5)).to(torch.bfloat16)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
