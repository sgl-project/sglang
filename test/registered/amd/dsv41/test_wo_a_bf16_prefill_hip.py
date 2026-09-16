"""gfx950 BF16 WO-A prefill layout, numerical equivalence and graph replay."""

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

    def test_fallbacks(self):
        for reason in (
            "empty",
            "small",
            "large",
            "dtype",
            "width",
            "mode",
            "device",
            "weight_stride",
        ):
            with self.subTest(reason=reason):
                rows = {"empty": 0, "small": 4095, "large": 65537}.get(reason, 4096)
                x, w = self.operands(
                    rows,
                    dtype=torch.float32 if reason == "dtype" else torch.bfloat16,
                    width=512 if reason == "width" else 4096,
                )
                if reason == "weight_stride":
                    w = w.transpose(1, 2).contiguous().transpose(1, 2)
                with (
                    patch(
                        "sglang.srt.models.deepseek_v4._is_gfx95_supported",
                        reason != "device",
                    ),
                    patch(
                        "torch.bmm",
                        side_effect=AssertionError("unexpected direct-output dispatch"),
                    ),
                ):
                    y = self.project(x, w, is_decode=False, is_prefill=reason != "mode")
                torch.testing.assert_close(
                    y, torch.einsum("tgd,grd->tgr", x, w), atol=0, rtol=0
                )


if __name__ == "__main__":
    unittest.main()
