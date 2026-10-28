"""Clover coreset parity, request isolation and graph replay contracts."""

import unittest

import torch

from sglang.kernels.ops.moe.clover_coreset_triton import (
    clover_block_coreset_triton,
)
from sglang.srt.models.clover_moe import _block_coreset_reference, block_coreset
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (9, 10),
    "Clover coreset requires SM90 or SM100-family CUDA",
)
class TestCloverCoreset(CustomTestCase):
    def test_parity_and_request_isolation(self):
        torch.manual_seed(42)
        dtype, size, top_p = torch.bfloat16, 32, 0.9
        # Strided inputs exercise row and expert stride handling.
        logits = torch.randn(2 * size, 512, device="cuda", dtype=dtype)[:, ::2]
        before = logits.clone()
        actual = block_coreset(logits, size, 8, top_p)
        expected = _block_coreset_reference(logits, size, 8, top_p)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        separate = torch.cat(
            [
                clover_block_coreset_triton(part, size, 8, top_p)
                for part in logits.split(size)
            ]
        )
        torch.testing.assert_close(actual, separate, atol=0, rtol=0)
        torch.testing.assert_close(logits, before, atol=0, rtol=0)

    def test_ties_and_threshold_boundary(self):
        logits = torch.zeros(32, 256, device="cuda", dtype=torch.bfloat16)
        actual = block_coreset(logits, 32, 8, 0.5)
        # Eight equally weighted experts, strict cumulative boundary: retain four.
        self.assertTrue(torch.isfinite(actual[:, :4]).all().item())
        self.assertTrue(torch.isneginf(actual[:, 4:]).all().item())
        for top_p in (0.0, 1.0):
            torch.testing.assert_close(
                block_coreset(logits, 32, 8, top_p),
                _block_coreset_reference(logits, 32, 8, top_p),
                atol=0,
                rtol=0,
            )

    def test_graph_replay_uses_new_input(self):
        logits = torch.zeros(32, 256, device="cuda", dtype=torch.bfloat16)
        logits[:, :8] = 10
        for _ in range(3):
            block_coreset(logits, 32, 8, 0.9)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = block_coreset(logits, 32, 8, 0.9)
        logits.zero_()
        logits[:, 64:72] = 10
        graph.replay()
        torch.testing.assert_close(
            output, _block_coreset_reference(logits, 32, 8, 0.9), atol=0, rtol=0
        )

    def test_unsupported_shape_falls_back(self):
        logits = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(
            block_coreset(logits, 3, 4, 0.9),
            _block_coreset_reference(logits, 3, 4, 0.9),
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
