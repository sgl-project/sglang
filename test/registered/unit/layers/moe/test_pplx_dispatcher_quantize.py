"""GPU test for the pplx dispatcher's FP8 wire quantization."""

import unittest

import torch

from sglang.srt.layers.moe.token_dispatcher import pplx as pplx_module
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(is_cuda(), "the pplx dispatcher runs on NVIDIA CUDA only")
class TestPplxFp8Quantize(CustomTestCase):
    def test_fp8_dispatch_quantizes_into_fp8_with_float32_block_scales(self):
        """An FP8 dispatch must quantize the hidden states into an fp8 payload
        with float32 scales per 128-column block (the pplx wire contract)."""
        # The constructor needs pplx-kernels and a process group; _quantize
        # only reads use_fp8.
        impl = pplx_module._PplxDispatcherImpl.__new__(pplx_module._PplxDispatcherImpl)
        impl.use_fp8 = True
        num_tokens, hidden_size = 4, 256
        block = pplx_module._FP8_BLOCK_SIZE
        hidden_states = torch.randn(
            num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
        )

        x_q, x_s = impl._quantize(hidden_states)

        self.assertEqual(x_q.dtype.itemsize, 1)
        self.assertEqual(tuple(x_q.shape), (num_tokens, hidden_size))
        self.assertEqual(x_s.dtype, torch.float32)
        self.assertEqual(tuple(x_s.shape), (num_tokens, hidden_size // block))

        blocks = hidden_states.float().view(num_tokens, -1, block)
        dequant = x_q.float().view(num_tokens, -1, block) * x_s.unsqueeze(-1)
        # fp8 e4m3 keeps 3 mantissa bits: at most 1/16 relative error per value.
        tolerance = blocks.abs().amax(dim=-1, keepdim=True) / 16
        self.assertTrue(bool(((dequant - blocks).abs() <= tolerance).all()))


if __name__ == "__main__":
    unittest.main()
