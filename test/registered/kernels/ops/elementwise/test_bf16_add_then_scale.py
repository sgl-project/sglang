"""Preserve the BF16 add boundary and current operands on compiled replay."""

import unittest

import torch

from sglang.kernels.ops.elementwise import bf16_add_then_scale
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, stage="stage-b", runner_config="1-gpu-xpu")


@unittest.skipUnless(torch.xpu.is_available(), "requires XPU")
class TestBf16AddThenScale(CustomTestCase):
    def assert_bits_or_nan(self, actual, expected):
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(torch.isnan(actual), torch.isnan(expected)))
        valid = ~torch.isnan(expected)
        self.assertTrue(
            torch.equal(
                actual.view(torch.int16)[valid], expected.view(torch.int16)[valid]
            )
        )

    def test_add_rounding_all_patterns_and_special_values(self):
        with torch.no_grad():
            # This case distinguishes two BF16 operations from a single final
            # cast, even with entirely finite, exactly representable operands.
            x = torch.ones(1, 1, device="xpu", dtype=torch.bfloat16)
            y = torch.full_like(x, 1 / 256)
            scale = torch.tensor([1.5], device="xpu", dtype=torch.bfloat16)
            eager = (x + y) * scale
            no_add_rounding = ((x.float() + y.float()) * scale.float()).to(x.dtype)
            self.assertFalse(torch.equal(eager, no_add_rounding))
            self.assert_bits_or_nan(bf16_add_then_scale(x, y, scale), eager)

            x = (
                torch.arange(65536, device="xpu", dtype=torch.int32)
                .to(torch.int16)
                .view(torch.bfloat16)
                .view(256, 256)
            )
            for residual in (0.0, -0.0, 1.0, float("inf"), float("nan")):
                y = torch.full_like(x, residual)
                for scalar in (
                    0.0,
                    -0.0,
                    1.5,
                    -1.0,
                    2.0**-133,
                    float("inf"),
                    -float("inf"),
                    float("nan"),
                ):
                    with self.subTest(residual=residual, scalar=scalar):
                        scale.fill_(scalar)
                        self.assert_bits_or_nan(
                            bf16_add_then_scale(x, y, scale), (x + y) * scale
                        )
            scale.fill_(-1)
            self.assert_bits_or_nan(bf16_add_then_scale(x, -x, scale), (x + -x) * scale)

    def test_compile_and_replay_read_changed_scalar_storage(self):
        graphs = []

        def backend(graph, _):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(bf16_add_then_scale, backend=backend, fullgraph=True)
        with torch.no_grad():
            for offset in (0, 1, 32):
                x = torch.randn(7 * 1536 + offset, device="xpu", dtype=torch.bfloat16)[
                    offset:
                ].view(7, 1536)
                y = torch.randn_like(x)
                scale = torch.full(
                    (1 + offset,), 1.5, device="xpu", dtype=torch.bfloat16
                )[offset:]
                self.assert_bits_or_nan(compiled(x, y, scale), (x + y) * scale)
                bf16_add_then_scale(x, y, scale)
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    output = bf16_add_then_scale(x, y, scale)
                scalar_pointer = scale.data_ptr()
                for scalar in (-0.5, 0.0, 1.75):
                    x.mul_(0.5)
                    y.add_(0.125)
                    scale.fill_(scalar)
                    self.assertEqual(scale.data_ptr(), scalar_pointer)
                    output.fill_(float("nan"))
                    graph.replay()
                    torch.xpu.synchronize()
                    self.assert_bits_or_nan(output, (x + y) * scale)
            self.assertTrue(graphs)
            self.assertTrue(
                all("triton_kernel_wrapper" in str(graph.graph) for graph in graphs)
            )
        torch._dynamo.reset()

    def test_contract_rejects_broadcast_dtype_and_grad_changes(self):
        x = torch.zeros(2, 1536, device="xpu", dtype=torch.bfloat16)
        y = torch.zeros_like(x)
        scale = torch.ones(1, device="xpu", dtype=torch.bfloat16)
        for a, b, scalar in (
            (x, y[:, :1], scale),
            (x, y, scale.expand(1536)),
            (x, y, scale.float()),
            (x[:, ::2], y[:, ::2], scale),
            (x.detach().requires_grad_(True), y, scale),
        ):
            with self.assertRaises(ValueError):
                bf16_add_then_scale(a, b, scalar)


if __name__ == "__main__":
    unittest.main()
