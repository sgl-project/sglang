"""The attention-residual bank reuses one grow-only buffer per stream across
forward passes, and aggregates exactly as a bank allocated per forward."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import attn_residual
from sglang.srt.layers.attn_residual import AttnResidual
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

HIDDEN = 1024
BLOCKS = 3


class _Norm(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(
            (1 + 0.1 * torch.randn(HIDDEN, generator=g)).cuda().bfloat16()
        )
        self.variance_epsilon = 1e-6

    def forward(self, x):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (x32 * self.weight.float()).to(x.dtype)


class _Proj(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(
            (torch.randn(1, HIDDEN, generator=g) * HIDDEN**-0.5).cuda().bfloat16()
        )


def _run(bank, hidden, points):
    """points: (score projection, norm, writes a bank row) per aggregation."""
    prefix, outputs = None, []
    for proj, norm, writes in points:
        normed, prefix = bank.forward(hidden, prefix, proj, norm, norm, write=writes)
        outputs.append(normed)
        hidden = torch.tanh(normed)
        if writes:
            prefix = None
    return outputs


class TestAttnResidualBankReuse(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs a GPU")
        cls.points = [(_Proj(i), _Norm(i), i % 2 == 0) for i in range(2 * BLOCKS)]

    def setUp(self):
        patcher = patch.object(attn_residual, "_BANK_SCRATCH", {})
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _hidden(tokens, seed=0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(
            tokens, HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16
        )

    def test_banks_share_one_buffer_until_a_larger_step(self):
        large = AttnResidual(self._hidden(64), BLOCKS).block_residual
        for tokens in (16, 64, 1):
            bank = AttnResidual(self._hidden(tokens), BLOCKS).block_residual
            self.assertEqual(bank.shape, (tokens, BLOCKS, HIDDEN))
            self.assertTrue(bank.is_contiguous())
            self.assertEqual(bank.data_ptr(), large.data_ptr())
        grown = AttnResidual(self._hidden(65), BLOCKS).block_residual
        self.assertEqual(grown.shape, (65, BLOCKS, HIDDEN))
        self.assertEqual(len(attn_residual._BANK_SCRATCH), 1)
        stored = next(iter(attn_residual._BANK_SCRATCH.values()))
        self.assertEqual(stored.data_ptr(), grown.data_ptr())

    def test_no_shared_storage_across_streams_or_without_reuse(self):
        shared = AttnResidual(self._hidden(8), BLOCKS).block_residual
        own = AttnResidual(self._hidden(8), BLOCKS, reuse=False).block_residual
        self.assertNotEqual(own.data_ptr(), shared.data_ptr())
        with torch.cuda.stream(torch.cuda.Stream()):
            other = AttnResidual(self._hidden(8), BLOCKS).block_residual
        self.assertNotEqual(other.data_ptr(), shared.data_ptr())

    def test_graph_capture_allocates_its_own_bank(self):
        hidden = self._hidden(8)
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            AttnResidual(hidden, BLOCKS).block_residual.fill_(1)
        self.assertEqual(attn_residual._BANK_SCRATCH, {})

    def test_reused_bank_aggregates_like_a_fresh_one(self):
        AttnResidual(self._hidden(32), BLOCKS).block_residual.fill_(float("nan"))
        hidden = self._hidden(7, seed=1)
        reused = _run(AttnResidual(hidden, BLOCKS), hidden, self.points)
        fresh = _run(AttnResidual(hidden, BLOCKS, reuse=False), hidden, self.points)
        for got, expected in zip(reused, fresh):
            self.assertFalse(got.isnan().any())
            torch.testing.assert_close(got, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
