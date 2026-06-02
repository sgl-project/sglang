import unittest

import torch
from utils import parametrize, precision

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

torch.manual_seed(1234)


def apply_token_bitmask_inplace_torch(logits, bitmask):
    """Reference implementation from xgrammar_backend.py."""
    vocab_size = logits.shape[-1]
    token_ids = torch.arange(vocab_size, dtype=torch.int32)
    word_idx = token_ids // 32
    bit_idx = token_ids % 32
    words = bitmask[:, word_idx].to(torch.int32)
    allowed = ((words >> bit_idx) & 1).to(torch.bool)
    logits.masked_fill_(~allowed, float("-inf"))


class TestGrammar(CustomTestCase):

    @parametrize(
        batch_size=[1, 4, 16],
        vocab_size=[10, 128, 32000, 32001],
        dtype=[torch.float32, torch.float16, torch.bfloat16],
    )
    def test_apply_token_bitmask_inplace(self, batch_size, vocab_size, dtype):
        logits = torch.randn(batch_size, vocab_size, dtype=dtype)
        bitmask_words = (vocab_size + 31) // 32
        bitmask = torch.randint(
            -(2**31), 2**31, (batch_size, bitmask_words), dtype=torch.int32
        )

        logits_ref = logits.clone()
        apply_token_bitmask_inplace_torch(logits_ref, bitmask)

        logits_kernel = logits.clone()
        torch.ops.sgl_kernel.apply_token_bitmask_inplace_cpu(logits_kernel, bitmask)

        atol = rtol = precision[logits_ref.dtype]
        torch.testing.assert_close(logits_kernel, logits_ref, atol=atol, rtol=rtol)

    def test_basic_correctness(self):
        """Simple test matching the CUDA test pattern."""
        neginf = float("-inf")
        bool_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
        logits = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            dtype=torch.float32,
        )
        expected = torch.where(bool_mask, logits, neginf)

        logits_test = logits.unsqueeze(0).clone()
        bitmask = torch.tensor([[0b1010101010]], dtype=torch.int32)
        torch.ops.sgl_kernel.apply_token_bitmask_inplace_cpu(logits_test, bitmask)
        torch.testing.assert_close(logits_test.squeeze(0), expected)

    def test_all_masked(self):
        """All tokens masked (bitmask = 0)."""
        batch_size, vocab_size = 2, 64
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
        bitmask = torch.zeros(batch_size, vocab_size // 32, dtype=torch.int32)

        torch.ops.sgl_kernel.apply_token_bitmask_inplace_cpu(logits, bitmask)
        self.assertTrue(torch.all(logits == float("-inf")))

    def test_none_masked(self):
        """No tokens masked (bitmask all 1s)."""
        batch_size, vocab_size = 2, 64
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
        logits_ref = logits.clone()
        bitmask = torch.full((batch_size, vocab_size // 32), -1, dtype=torch.int32)

        torch.ops.sgl_kernel.apply_token_bitmask_inplace_cpu(logits, bitmask)
        torch.testing.assert_close(logits, logits_ref)


if __name__ == "__main__":
    unittest.main()
