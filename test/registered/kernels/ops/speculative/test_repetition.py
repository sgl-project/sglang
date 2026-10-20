import unittest

import torch

from sglang.kernels.ops.speculative.repetition import override_repeating_draft_tokens
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRepetitionCandidates(CustomTestCase):
    def test_live_lengths_and_inactive_rows_on_replay(self):
        """Graph replay uses live positions and preserves inactive rows and padding."""
        patterns = torch.tensor(
            [[3, 100, 11, 22, 33], [1, 80, 55, 0, 0], [0, 0, 0, 0, 0]], device="cuda"
        )
        lengths = torch.tensor([151, 120, 9], dtype=torch.int32, device="cuda")
        storage = torch.full((3, 9), -7, dtype=torch.int64, device="cuda")
        tokens = storage[:, :7]
        override_repeating_draft_tokens(tokens, lengths, patterns)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            override_repeating_draft_tokens(tokens, lengths, patterns)
        for prefix, expected in [
            (151, [22, 33, 11, 22, 33, 11, 22]),
            (152, [33, 11, 22, 33, 11, 22, 33]),
        ]:
            lengths[0] = prefix
            graph.replay()
            self.assertEqual(tokens[0].tolist(), expected)
            self.assertEqual(tokens[1].tolist(), [55] * 7)
            self.assertEqual(tokens[2].tolist(), [-7] * 7)
            self.assertTrue(torch.all(storage[:, 7:] == -7).item())


if __name__ == "__main__":
    unittest.main()
