import unittest

import torch

from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.test.test_utils import CustomTestCase

torch.manual_seed(0)

try:
    from torchembed._triton import fused_rope_forward  # noqa: F401

    _torchembed_available = True
except ImportError:
    _torchembed_available = False

_CASES = [
    (64, 64, 32, 8000, True, torch.bfloat16, 32, 32, 1, 1),
    (128, 128, 4096, 10000, True, torch.bfloat16, 2, 512, 4, 2),
    (128, 128, 2048, 10000, True, torch.bfloat16, 2, 512, 32, 8),
    (128, 128, 2048, 10000, False, torch.bfloat16, 2, 512, 32, 8),
    (256, 128, 4096, 10000, True, torch.bfloat16, 3, 39, 4, 2),
    (128, 64, 2048, 10000, True, torch.bfloat16, 2, 512, 16, 4),
]


@unittest.skipIf(
    not torch.cuda.is_available() or not _torchembed_available,
    reason="requires CUDA and torchembed",
)
class TestRotaryEmbeddingTorchembed(CustomTestCase):
    def _run_case(
        self,
        head_size: int,
        rotary_dim: int,
        max_pos: int,
        base: int,
        is_neox: bool,
        dtype: torch.dtype,
        batch_size: int,
        seq_len: int,
        num_q: int,
        num_kv: int,
    ) -> None:
        rope = RotaryEmbedding(
            head_size, rotary_dim, max_pos, base, is_neox, dtype
        ).to("cuda")

        pos_ids = torch.arange(seq_len, device="cuda").repeat(batch_size)
        query = torch.randn(
            batch_size * seq_len, num_q * head_size, dtype=dtype, device="cuda"
        )
        key = torch.randn(
            batch_size * seq_len, num_kv * head_size, dtype=dtype, device="cuda"
        )

        q_ref, k_ref = rope.forward_native(pos_ids, query.clone(), key.clone())
        q_cuda, k_cuda = rope.forward_cuda(pos_ids, query.clone(), key.clone())

        torch.testing.assert_close(q_ref, q_cuda, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k_ref, k_cuda, atol=1e-2, rtol=1e-2)

    def test_all_cases(self) -> None:
        for case in _CASES:
            with self.subTest(case=case):
                self._run_case(*case)


@unittest.skipIf(
    not torch.cuda.is_available() or not _torchembed_available,
    reason="requires CUDA and torchembed",
)
class TestRotaryEmbeddingGrad(CustomTestCase):
    def test_gradient_flow(self):
        head_size, rotary_dim, max_pos = 64, 64, 128
        batch_size, seq_len = 2, 16
        num_q, num_kv = 4, 2
        dtype = torch.float32

        rope = RotaryEmbedding(
            head_size, rotary_dim, max_pos, 10000, True, dtype
        ).to("cuda")

        pos_ids = torch.arange(seq_len, device="cuda").repeat(batch_size)
        q = torch.randn(
            batch_size * seq_len, num_q * head_size, dtype=dtype, device="cuda",
            requires_grad=True,
        )
        k = torch.randn(
            batch_size * seq_len, num_kv * head_size, dtype=dtype, device="cuda",
            requires_grad=True,
        )

        q_out, k_out = rope.forward_cuda(pos_ids, q, k)
        loss = q_out.sum() + k_out.sum()
        loss.backward()

        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertFalse(torch.isnan(q.grad).any())
        self.assertFalse(torch.isnan(k.grad).any())


if __name__ == "__main__":
    unittest.main()
