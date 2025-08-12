import unittest

import torch

from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.utils import get_bool_env_var, is_hip
from sglang.test.test_utils import CustomTestCase

torch.manual_seed(0)

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


_CASES = [
    (64, 64, 32, 8000, True, torch.bfloat16, "cuda", 32, 32, 1, 1),
    (256, 128, 4096, 10000, True, torch.bfloat16, "cuda", 2, 512, 4, 2),
    (512, 128, 311, 10000, True, torch.bfloat16, "cuda", 3, 39, 4, 2),
    (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 32, 8),
    (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 16, 4),
    (512, 128, 311, 10000, False, torch.bfloat16, "cuda", 3, 39, 4, 2),
]


@unittest.skipIf(_use_aiter, reason="SGLANG_USE_AITER=1 will not use vllm path.")
class TestRotaryEmbeddingNative(CustomTestCase):
    # Compare RotaryEmbedding.forward_hip() to forward_native().
    def _run_case(
        self,
        head_size: int,
        rotary_dim: int,
        max_pos: int,
        base: int,
        is_neox: bool,
        dtype: torch.dtype,
        device: str,
        batch_size: int,
        seq_len: int,
        num_q: int,
        num_kv: int,
    ) -> None:
        rope_ref = RotaryEmbedding(
            head_size, rotary_dim, max_pos, base, is_neox, dtype
        ).to(device)
        rope_hip = RotaryEmbedding(
            head_size, rotary_dim, max_pos, base, is_neox, dtype
        ).to(device)

        pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
        query = torch.randn(
            batch_size * seq_len, num_q * head_size, dtype=dtype, device=device
        )
        key = torch.randn(
            batch_size * seq_len, num_kv * head_size, dtype=dtype, device=device
        )

        q_ref, k_ref = rope_ref.forward_native(pos_ids, query.clone(), key.clone())
        q_hip, k_hip = rope_hip.forward_hip(pos_ids, query.clone(), key.clone())

        torch.testing.assert_close(q_ref, q_hip, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k_ref, k_hip, atol=1e-2, rtol=1e-2)

    def test_all_cases(self) -> None:
        """Drive over the full parameter matrix using subTest()."""
        for case in _CASES:
            with self.subTest(case=case):
                self._run_case(*case)


@unittest.skipIf(not _use_aiter, reason="Requires AMD GPU plus SGLANG_USE_AITER=1")
class TestRotaryEmbeddingAITer(CustomTestCase):
    @staticmethod
    def _run_case_aiter(
        head_size: int,
        rotary_dim: int,
        max_pos: int,
        base: int,
        is_neox: bool,
        dtype: torch.dtype,
        device: str,
        batch_size: int,
        seq_len: int,
        num_q: int,
        num_kv: int,
    ) -> None:
        from aiter.rotary_embedding import RotaryEmbedding as AiterRotaryEmbedding

        rope_ref = AiterRotaryEmbedding(
            head_size, rotary_dim, max_pos, base, is_neox, dtype
        ).to(device)
        rope_hip = AiterRotaryEmbedding(
            head_size, rotary_dim, max_pos, base, is_neox, dtype
        ).to(device)

        pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
        query = torch.randn(
            batch_size * seq_len, num_q * head_size, dtype=dtype, device=device
        )
        key = torch.randn(
            batch_size * seq_len, num_kv * head_size, dtype=dtype, device=device
        )

        q_ref, k_ref = rope_ref.forward_native(pos_ids, query.clone(), key.clone())
        q_hip, k_hip = rope_hip.forward_hip(pos_ids, query.clone(), key.clone())

        torch.testing.assert_close(q_ref, q_hip, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k_ref, k_hip, atol=1e-2, rtol=1e-2)

    def test_all_cases(self) -> None:
        for case in _CASES:
            with self.subTest(case=case):
                self._run_case_aiter(*case)


if __name__ == "__main__":
    unittest.main()
