"""Split convolution stores preserve varlen outputs, state, and graph replay."""

import unittest
from itertools import accumulate

import torch

from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestSplitConvOutput(CustomTestCase):
    def test_varlen_planes_preserve_state_and_replay_new_inputs(self):
        # Unequal sequence lengths distinguish the full output-plane pitch
        # from a request's length. H=128 also straddles a 256-channel CTA.
        for width, block_table in ((3, False), (4, True)):
            with self.subTest(width=width, block_table=block_table):
                self._check_split_output(width, block_table)

    def _check_split_output(self, width, block_table):
        torch.manual_seed(44)
        lengths, hidden = [1, 2, 7, 65], 128
        tokens, dim = sum(lengths), 3 * hidden
        # The fused QKV/beta/gate projection leaves a row-strided QKV view.
        x = torch.randn(tokens, dim + 272, device="cuda", dtype=torch.bfloat16)
        x = x[:, :dim]
        # GLM KDA uses FP32 convolution weights without a bias.
        weight_dtype = torch.float32 if width == 4 else x.dtype
        weight = torch.randn(dim, width, device="cuda", dtype=weight_dtype)
        bias = torch.randn(dim, device="cuda", dtype=x.dtype) if width == 3 else None
        seed = torch.randn(6, width - 1, dim, device="cuda", dtype=x.dtype)
        states = [seed.clone().transpose(-1, -2) for _ in range(2)]
        cu = torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int64)
        slots = torch.tensor([4, 2, 1, 3], device="cuda", dtype=torch.int32)
        initial = torch.tensor([False, True, False, True], device="cuda")
        table = None
        if block_table:
            table = torch.tensor(
                [(i, j) for i, n in enumerate(lengths) for j in range((n + 7) // 8)]
                + [(-1, 0)] * 3,
                device="cuda",
                dtype=torch.int32,
            )

        def run(split):
            output = causal_conv1d_fn(
                x.T,
                weight,
                bias,
                states[int(split)],
                cu,
                None if block_table else lengths,
                slots,
                initial,
                block_table=table,
                output_split_size=hidden if split else 0,
            )
            return output.unbind(0) if split else output.T.split(hidden, dim=-1)

        expected, actual = run(False), run(True)
        for before, after in zip(expected, actual):
            self.assertTrue(after.is_contiguous())
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        torch.testing.assert_close(states[1], states[0], rtol=0, atol=0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run(True)
        x.mul_(0.25)
        for state in states:
            state.copy_(seed.transpose(-1, -2))
        expected = run(False)
        graph.replay()
        for before, after in zip(expected, actual):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        torch.testing.assert_close(states[1], states[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
