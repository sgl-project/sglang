# SPDX-License-Identifier: Apache-2.0
"""FLUX3 packed MLP inputs must preserve SwiGLU rounding through FP8 GEMM."""

import unittest
from itertools import product
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import BitExactFusionGate
from sglang.multimodal_gen.runtime.models.dits.flux3 import Flux3Fp8RowwiseLinear
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestFlux3SwiGLUFp8(CustomTestCase):
    def setUp(self):
        super().setUp()
        # Unsupported fallback layouts can disable the existing SwiGLU gate;
        # isolate that real state machine so other model tests are unaffected.
        self.enterContext(
            patch(
                "sglang.multimodal_gen.runtime.models.dits.flux3._SWIGLU",
                BitExactFusionGate("FLUX3 test SwiGLU", per_signature=True),
            )
        )

    @staticmethod
    def _output(result, tuple_output):
        if tuple_output:
            output, bias = result
            assert bias is None
            return output
        return result

    def _reference(self, layer, packed, tuple_output):
        gate, value = packed.chunk(2, dim=-1)
        return self._output(layer(F.silu(gate) * value), tuple_output)

    def _assert_matches(self, layer, packed, tuple_output):
        actual = self._output(layer.forward_swiglu(packed), tuple_output)
        self.assertTrue(
            torch.equal(actual, self._reference(layer, packed, tuple_output))
        )
        return actual

    def _cuda_inputs(self, hidden, batch=2, length=17):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
            self.skipTest("requires NVIDIA SM89+")
        generator = torch.Generator(device="cuda").manual_seed(42)
        weight = torch.randn(64, hidden, device="cuda", generator=generator).to(
            torch.float8_e4m3fn
        )
        scales = torch.rand(64, device="cuda", generator=generator)
        storage = torch.randn(
            batch,
            length,
            3 * hidden,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        return weight, scales, storage[..., hidden:]

    def test_cpu_fallback(self):
        """Unsupported devices must retain the original quantized linear result."""
        generator = torch.Generator().manual_seed(42)
        packed = torch.randn(2, 17, 64, generator=generator).bfloat16()
        weight = torch.randn(16, 32, generator=generator).to(torch.float8_e4m3fn)
        scales = torch.rand(16, generator=generator)

        # CPU lacks FP8 scaled_mm: emulate only the GEMM dependency, retaining
        # the real SwiGLU, padding, quantization and output reshaping paths.
        def scaled_mm(a, b, scale_a, scale_b, *, out_dtype, use_fast_accum):
            self.assertTrue(use_fast_accum)
            return ((a.float() * scale_a) @ (b.float() * scale_b)).to(out_dtype)

        with patch("torch._scaled_mm", scaled_mm), torch.inference_mode():
            for tuple_output in (False, True):
                with self.subTest(tuple_output=tuple_output):
                    layer = Flux3Fp8RowwiseLinear(weight, scales, tuple_output)
                    actual = self._assert_matches(layer, packed, tuple_output)
                    self.assertEqual(actual.shape, (2, 17, 16))

    @torch.inference_mode()
    def test_strided_inputs(self):
        """Strided rows and row padding preserve the unfused result."""
        for hidden, (batch, length), tuple_output in product(
            (3072, 9216), ((1, 1), (2, 17), (1, 340)), (False, True)
        ):
            with self.subTest(
                hidden=hidden, batch=batch, length=length, tuple_output=tuple_output
            ):
                weight, scales, packed = self._cuda_inputs(hidden, batch, length)
                layer = Flux3Fp8RowwiseLinear(weight, scales, tuple_output)
                self._assert_matches(layer, packed, tuple_output)

    @torch.inference_mode()
    def test_graph_replay(self):
        """Graph replay reads updated strided inputs for both output formats."""
        weight, scales, packed = self._cuda_inputs(3072)
        for tuple_output in (False, True):
            with self.subTest(tuple_output=tuple_output):
                layer = Flux3Fp8RowwiseLinear(weight, scales, tuple_output)
                self._assert_matches(layer, packed, tuple_output)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = self._output(layer.forward_swiglu(packed), tuple_output)
                packed.neg_()
                graph.replay()
                self.assertTrue(
                    torch.equal(captured, self._reference(layer, packed, tuple_output))
                )

    @torch.inference_mode()
    def test_transposed_fallback(self):
        """Non-flattenable batch/sequence strides preserve output ordering."""
        for hidden in (3072, 9216):
            with self.subTest(hidden=hidden):
                weight, scales, packed = self._cuda_inputs(hidden, batch=17, length=2)
                layer = Flux3Fp8RowwiseLinear(weight, scales, False)
                self._assert_matches(layer, packed.transpose(0, 1), False)


if __name__ == "__main__":
    unittest.main()
