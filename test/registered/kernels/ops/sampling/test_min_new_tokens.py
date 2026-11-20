import unittest

import torch

from sglang.kernels.ops.sampling.min_new_tokens import apply_min_new_tokens_penalties
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMinNewTokens(unittest.TestCase):
    def _inputs(self):
        ids = torch.tensor([[0, 2, 7, 0], [3, 0, 0, 0], [0, 0, 0, 0]], device="cuda")
        penalties = torch.tensor(
            [
                [-float("inf"), -float("inf"), -float("inf"), 0],
                [-float("inf"), 0, 0, 0],
                [0, 0, 0, 0],
            ],
            device="cuda",
        )
        lengths = torch.tensor([[0], [2], [0]], dtype=torch.int32, device="cuda")
        minimum = torch.tensor([[1], [2], [1]], dtype=torch.int32, device="cuda")
        return ids, penalties, lengths, minimum

    def test_ragged_stops_dtypes_and_strided_logits(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            for strided in (False, True):
                with self.subTest(dtype=dtype, strided=strided):
                    logits = torch.ones(
                        3, 32 if strided else 16, device="cuda", dtype=dtype
                    )
                    if strided:
                        logits = logits[:, ::2]
                    expected = logits.clone()
                    expected[0, [0, 2, 7]] = -float("inf")
                    apply_min_new_tokens_penalties(logits, *self._inputs())
                    torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    def test_cuda_graph_replay_uses_current_lengths(self):
        args = self._inputs()
        logits = torch.ones(3, 16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            apply_min_new_tokens_penalties(logits, *args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            apply_min_new_tokens_penalties(logits, *args)
        for count in (0, 1, 2, 3):
            logits.fill_(1)
            args[2].fill_(count)
            graph.replay()
            expected = torch.ones_like(logits)
            if count < 1:
                expected[0, [0, 2, 7]] = -float("inf")
            if count < 2:
                expected[1, 3] = -float("inf")
            torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    def test_nonfinite_logits_preserve_additive_semantics(self):
        logits = torch.zeros(3, 16, device="cuda")
        logits[0, 0] = float("inf")
        logits[0, 2] = float("nan")
        logits[0, 4] = -float("inf")
        dense = torch.zeros_like(logits)
        dense[0, [0, 2, 7]] = -float("inf")
        expected = logits + dense
        apply_min_new_tokens_penalties(logits, *self._inputs())
        torch.testing.assert_close(logits, expected, rtol=0, atol=0, equal_nan=True)

        logits = torch.ones(3, 16, device="cuda", dtype=torch.float64)
        logits[0, 7] = 1e300
        expected = logits.clone()
        expected[0, [0, 2, 7]] = -float("inf")
        apply_min_new_tokens_penalties(logits, *self._inputs())
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    def test_empty_and_multiblock_stop_lists(self):
        for count in (0, 313):
            logits = torch.ones(7, 512, device="cuda")
            ids = torch.arange(count, device="cuda").expand(7, -1).contiguous()
            penalties = torch.full((7, count), -float("inf"), device="cuda")
            lengths = torch.zeros(7, 1, dtype=torch.int32, device="cuda")
            minimum = torch.ones_like(lengths)
            expected = logits.clone()
            expected[:, :count] = -float("inf")
            apply_min_new_tokens_penalties(logits, ids, penalties, lengths, minimum)
            torch.testing.assert_close(logits, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
