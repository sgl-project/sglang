"""PP dynamic chunking on Intel XPU (PP=2, Llama-3.1-8B-Instruct).

Covers the chunk schedule the PP long-context guide documents and the fallback
when the startup latency probe runs out of memory.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.xpu.pp_long_context_xpu_base import PPLongContextXPUServerBase

register_xpu_ci(est_time=420, suite="nightly-kernel-main-test-2-gpu-xpu", nightly=True)


@unittest.skipUnless(torch.xpu.is_available(), "needs Intel XPU")
class TestXPUPPDynamicChunking(PPLongContextXPUServerBase):
    """Dynamic chunking from a 16K initial chunk.

    Its 20K-token probes take most of the memory the KV pool leaves free, and the PP
    communicator is built only after them, so the server must release that memory
    before it serves.
    """

    BASE = 16384
    other_args = ["--chunked-prefill-size", str(BASE), "--enable-dynamic-chunking"]

    def test_chunks_shrink_as_the_prefix_grows(self):
        self.assertRegex(
            self.server_log(), r"\[PP Dynamic Chunk\] \[PP0\] Predictor ready"
        )
        before = len(self.pp0_chunk_sizes())
        self.assert_passkey(65536, 40913)
        sizes = self.pp0_chunk_sizes()[before:]
        # The last chunk is the remainder; every predicted chunk before it is
        # no larger than the previous one and no smaller than a quarter of the base.
        predicted = sizes[:-1]
        self.assertEqual(predicted[0], self.BASE, sizes)
        self.assertEqual(predicted, sorted(predicted, reverse=True), sizes)
        self.assertLess(predicted[-1], self.BASE, sizes)
        self.assertGreaterEqual(min(predicted), self.BASE // 4, sizes)
        self.assertEqual(sum(sizes), 65536, sizes)


@unittest.skipUnless(torch.xpu.is_available(), "needs Intel XPU")
class TestXPUPPDynamicChunkingProbeFallback(PPLongContextXPUServerBase):
    """A probe that runs out of memory disables dynamic chunking, not the server.

    On a 24 GB card a 0.85 static fraction leaves PP0 too little headroom for the
    15K-token probe; the server must still come up and serve. A full 12K chunk does
    not fit at this fraction either, so the request stays below one chunk.
    """

    mem_fraction_static = "0.85"
    other_args = ["--chunked-prefill-size", "12288", "--enable-dynamic-chunking"]

    def test_server_keeps_serving_after_the_probe_fails(self):
        if "Failed to profile prefill latency" not in self.server_log():
            self.skipTest("the probe fit in memory on this card")
        self.assertIn("Dynamic chunking will be disabled", self.server_log())
        self.assert_passkey(2048, 73104)


if __name__ == "__main__":
    unittest.main()
