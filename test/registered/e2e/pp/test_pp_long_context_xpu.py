"""Long-context pipeline parallelism on Intel XPU (PP=2, Llama-3.1-8B-Instruct).

Many-chunk prefill up to ~120K tokens across an uneven layer split, with PP
communication on a dedicated stream and decode on the full XPU graph.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.xpu.pp_long_context_xpu_base import PPLongContextXPUServerBase

register_xpu_ci(est_time=240, suite="nightly-kernel-main-test-2-gpu-xpu", nightly=True)


@unittest.skipUnless(torch.xpu.is_available(), "needs Intel XPU")
class TestXPUPPLongContext(PPLongContextXPUServerBase):
    other_args = [
        "--chunked-prefill-size",
        "4096",
        "--cuda-graph-backend-decode",
        "full",
    ]
    env = {"SGLANG_PP_LAYER_PARTITION": "15,17", "SGLANG_PP_COMM_OVERLAP": "1"}

    def test_comm_overlap_stream_is_used(self):
        self.assertIn(
            "PP tensor communication overlap enabled on a dedicated xpu stream",
            self.server_log(),
        )

    def test_passkey_retrieval_across_many_chunks(self):
        # 120K tokens is ~30 chunks, each crossing the stage boundary.
        for num_tokens, passkey in ((32768, 61442), (120000, 27819)):
            with self.subTest(num_tokens=num_tokens):
                out = self.assert_passkey(num_tokens, passkey)
                self.assertEqual(out["meta_info"]["prompt_tokens"], num_tokens)


if __name__ == "__main__":
    unittest.main()
