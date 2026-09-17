"""gfx950 BF16 WO-A dispatch, numerical equivalence and graph replay."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=45, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "requires gfx950")
class TestWoABf16Prefill(unittest.TestCase):
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
            (16384, False),
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
        from sglang.srt.models import deepseek_v4 as model

        for rows in (1, 2, 8, 129, 192, 256, 384):
            with self.subTest(rows=rows):
                x, w = self.operands(rows, strided=rows == 8)
                kwargs = dict(is_decode=True, is_target_verify=rows > 1)
                name = "wo_a_bf16_gemv" if rows == 1 else "wo_a_bf16_small_batch"
                if rows <= 8:
                    with patch.object(
                        model, name, wraps=getattr(model, name)
                    ) as kernel:
                        self.project(x, w, **kwargs)
                        kernel.assert_called_once()
                else:
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


if __name__ == "__main__":
    unittest.main()
