"""gfx950 standalone post dispatch, AITER equivalence and mutable graph replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "requires gfx950")
class TestMhcPostSplitH(unittest.TestCase):
    def setUp(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

        self.layer = SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v41"), hc_mult=4
        )
        self.run_post = lambda *args: DeepseekV4DecoderLayer.hc_post(self.layer, *args)
        for setting in (
            envs.SGLANG_OPT_HIP_MHC_POST_SPLIT_H.override(True),
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(False),
            envs.SGLANG_OPT_USE_FLASHINFER_MHC.override(False),
        ):
            setting.__enter__()
            self.addCleanup(setting.__exit__, None, None, None)
        torch.manual_seed(39186)

    def operands(self, rows, width=5120, dtype=torch.bfloat16):
        return (
            torch.randn(rows, width, device="cuda", dtype=dtype),
            torch.randn(rows, 4, width, device="cuda", dtype=dtype),
            torch.sigmoid(torch.randn(rows, 4, device="cuda")),
            torch.softmax(torch.randn(rows, 4, 4, device="cuda"), -1),
        )

    def test_model_dispatch_and_graph_replay(self):
        from aiter.ops.mhc import mhc_post

        from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h

        for rows in (768, 1024, 1025, 1536, 2048, 4096):
            with self.subTest(rows=rows):
                args = self.operands(rows)
                reference = torch.empty_like(args[1])
                with patch(
                    "sglang.srt.models.deepseek_v4.mhc_post_split_h",
                    wraps=mhc_post_split_h,
                ) as split:
                    actual = self.run_post(*args)
                    split.assert_called_once()
                    self.assertEqual(split.call_args.kwargs["block_size"], 2048)
                mhc_post(reference, *args)
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = self.run_post(*args)
                for _ in range(2):
                    args[0].normal_()
                    args[1].normal_()
                    args[2].uniform_()
                    args[3].uniform_()
                    graph.replay()
                    mhc_post(reference, *args)
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
