"""Native BF16 intermediate rounding and current-input replay regressions."""

import unittest

import torch

from sglang.kernels.ops.activation import bf16_gelu_tanh_mul_lut
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, stage="stage-b", runner_config="1-gpu-xpu")


@unittest.skipUnless(torch.xpu.is_available(), "requires XPU")
class TestBf16GeluTanhMulLut(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from sgl_kernel import gelu_tanh_and_mul

        cls.native = staticmethod(gelu_tanh_and_mul)
        cls.values = (
            torch.arange(65536, device="xpu", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        cls.table = gelu_tanh_and_mul(
            torch.cat([cls.values, torch.ones_like(cls.values)]).unsqueeze(0)
        ).flatten()

    def assert_native_bits(self, actual, expected):
        same = (actual.view(torch.int16) == expected.view(torch.int16)) | (
            actual.isnan() & expected.isnan()
        )
        self.assertTrue(same.all().item())

    def test_compile_and_graph_read_current_inputs(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        # Exercise the public cold import on the first compiled invocation.
        compiled = torch.compile(
            bf16_gelu_tanh_mul_lut, backend=backend, fullgraph=True
        )
        for offset in (0, 1, 32):
            with self.subTest(offset=offset):
                x = torch.randn(7 * 12288 + offset, device="xpu", dtype=torch.bfloat16)[
                    offset:
                ].view(7, 12288)
                self.assert_native_bits(compiled(x, self.table), self.native(x.clone()))
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    output = bf16_gelu_tanh_mul_lut(x, self.table)
                for scale in (1.0, -0.5):
                    x.mul_(scale)
                    expected = self.native(x.clone())
                    output.fill_(float("nan"))
                    graph.replay()
                    self.assert_native_bits(output, expected)
        self.assertTrue(graphs)

    def test_native_bf16_patterns_and_signed_zero(self):
        ups = torch.tensor(
            [
                -float("inf"),
                -10,
                -1,
                -0.25,
                -0.0,
                0.0,
                0.001,
                0.125,
                1,
                10,
                float("inf"),
                float("nan"),
            ],
            device="xpu",
            dtype=torch.bfloat16,
        )
        x = torch.cat(
            [self.values[None, :].expand(len(ups), -1), ups[:, None].expand(-1, 65536)],
            dim=-1,
        ).contiguous()
        self.assert_native_bits(bf16_gelu_tanh_mul_lut(x, self.table), self.native(x))

    def test_index_limit_rejected_without_allocation(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            input = torch.empty((4096, 524288), device="xpu", dtype=torch.bfloat16)
            table = torch.empty(65536, device="xpu", dtype=torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "INT_MAX"):
                bf16_gelu_tanh_mul_lut(input, table)


if __name__ == "__main__":
    unittest.main()
