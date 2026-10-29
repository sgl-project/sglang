"""The torch top-k/top-p/min-p sampler captures in an XPU graph and replays like eager."""

import unittest

import torch

from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")


class TestSamplerGraphCaptureXPU(CustomTestCase):
    def setUp(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        torch.manual_seed(0)
        batch, vocab = 4, 3072
        self.probs = torch.softmax(torch.randn(batch, vocab, device="xpu"), dim=-1)
        self.top_ks = torch.full((batch,), 50, device="xpu", dtype=torch.int32)
        self.top_ps = torch.full((batch,), 0.8, device="xpu")
        self.min_ps = torch.full((batch,), 0.05, device="xpu")
        self.seeds = torch.arange(batch, device="xpu", dtype=torch.int64) + 1234
        self.positions = torch.arange(batch, device="xpu", dtype=torch.int64) + 7

    def assert_replay_matches_eager(self, sample):
        eager = sample()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            captured = sample()
        graph.replay()
        torch.xpu.synchronize()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)

    def test_seeded_top_k_top_p(self):
        self.assert_replay_matches_eager(
            lambda: top_k_top_p_min_p_sampling_from_probs_torch(
                self.probs,
                self.top_ks,
                self.top_ps,
                self.min_ps,
                False,
                self.seeds,
                self.positions,
            )
        )

    def test_min_p(self):
        # min_p is only valid without a seed, and torch.multinomial is not
        # deterministic, so compare the filtered weights, not the draw.
        def filtered_weights():
            return top_k_top_p_min_p_sampling_from_probs_torch(
                self.probs,
                self.top_ks,
                self.top_ps,
                self.min_ps,
                True,
                None,
                self.positions,
                return_filtered_probs=True,
            )[1]

        self.assert_replay_matches_eager(filtered_weights)


if __name__ == "__main__":
    unittest.main()
