
import unittest
import torch
from sglang.srt.models.deepseek_v4 import _apply_wo_a_bf16_matmul
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
import unittest
import torch
from sglang.test.test_utils import CustomTestCase
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci




register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "The prefill WO-A path targets Blackwell",
)
class TestPrefillWoA(CustomTestCase):
    def test_exact_output_and_contiguous_layout(self):
        torch.manual_seed(911)
        weight = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16)
        for rows in (4096, 4097, 65536):
            with self.subTest(rows=rows):
                # Match the attention backend's 64 padded heads, 16 local heads.
                backing = torch.randn(
                    rows, 64, 512, device="cuda", dtype=torch.bfloat16
                )
                x = backing[:, :16].view(rows, 2, 4096)
                expected = torch.einsum("tgd,grd->tgr", x, weight)
                actual = _apply_wo_a_bf16_matmul(
                    x, weight, is_decode=False, is_prefill=True
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.flatten(1).data_ptr(), actual.data_ptr())














@unittest.skipUnless(is_hip() and is_gfx95_supported(), "requires gfx950")
class TestWoABf16Prefill(CustomTestCase):
    def setUp(self):
        from sglang.srt.models.deepseek_v4 import _apply_wo_a_bf16_matmul
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.project = _apply_wo_a_bf16_matmul
        torch.manual_seed(39186)

    def operands(self, rows, *, strided=False, dtype=torch.bfloat16, width=4096):
        x = torch.randn(rows, 4 if strided else 2, width, device="cuda", dtype=dtype)
        if strided:
            x = x[:, 1:3]
        w = torch.randn(2, 1024, width, device="cuda", dtype=dtype) * 0.015625
        return x, w

    def test_prefill_and_mutable_graph(self):
        for rows, strided in (
            (4096, False),
            (4097, True),
            (65536, False),
        ):
            with self.subTest(rows=rows, strided=strided):
                x, w = self.operands(rows, strided=strided)
                y = self.project(x, w, is_decode=False, is_prefill=True)
                self.assertTrue(y.is_contiguous())
                torch.testing.assert_close(
                    y, torch.einsum("tgd,grd->tgr", x, w), atol=0, rtol=0
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    y = self.project(x, w, is_decode=False, is_prefill=True)
                for _ in range(2):
                    x.normal_()
                    w.normal_(std=0.015625)
                    graph.replay()
                    torch.testing.assert_close(
                        y, torch.einsum("tgd,grd->tgr", x, w), atol=0, rtol=0
                    )
                del graph, x, w, y

    def test_decode_verify_and_mutable_graph(self):

        for rows in (1, 2, 8, 129):
            with self.subTest(rows=rows):
                x, w = self.operands(rows, strided=rows == 8)
                kwargs = dict(is_decode=True, is_target_verify=rows > 1)
                self.project(x, w, **kwargs)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    y = self.project(x, w, **kwargs)
                for _ in range(2):
                    x.normal_()
                    w.normal_(std=0.015625)
                    graph.replay()
                    ref = torch.einsum("tgd,grd->tgr", x, w)
                    self.assertTrue(y.is_contiguous())
                    if rows > 8:
                        torch.testing.assert_close(y, ref, atol=0, rtol=0)
                    else:
                        error = (y.float() - ref.float()).square().mean()
                        self.assertLess(
                            (error / ref.float().square().mean()).sqrt().item(), 1e-4
                        )


# backend-specific: gfx950 BF16 GEMV and small-batch projection use HIP kernels.
register_amd_ci(est_time=25, suite="stage-b-kernel-test-1-gpu-amd-mi35x")

if __name__ == "__main__":
    unittest.main()
