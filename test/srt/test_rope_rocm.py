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

    def test_ops_equivalence_basic(self) -> None:
        import aiter as ops
        from aiter.rotary_embedding import RotaryEmbedding as AiterRotaryEmbedding

        (
            head_size,
            rotary_dim,
            max_pos,
            base,
            is_neox,
            dtype,
            device,
            bs,
            seq_len,
            num_q,
            num_kv,
        ) = (
            128,
            64,
            2048,
            10000,
            True,
            torch.bfloat16,
            "cuda",
            2,
            32,
            4,
            2,
        )

        rope = AiterRotaryEmbedding(
            head_size, rotary_dim, max_pos, base, is_neox, dtype
        ).to(device)

        positions = torch.arange(seq_len, device=device).repeat(bs)
        num_tokens = positions.numel()

        q2d = torch.randn(num_tokens, num_q * head_size, dtype=dtype, device=device)
        k2d = torch.randn(num_tokens, num_kv * head_size, dtype=dtype, device=device)

        q_ref, k_ref = rope.forward_hip(positions.clone(), q2d.clone(), k2d.clone())

        q_sbhd = q2d.view(1, num_tokens, num_q, head_size)
        k_sbhd = k2d.view(1, num_tokens, num_kv, head_size)

        cos = rope.cos_cache.to(device=device, dtype=dtype)
        sin = rope.sin_cache.to(device=device, dtype=dtype)
        pos_b_s = positions.view(1, num_tokens)
        rotate_style = 0 if is_neox else 1
        ops.rope_cached_positions_2c_fwd_inplace(
            q_sbhd,
            k_sbhd,
            cos,
            sin,
            pos_b_s,
            rotate_style,
            reuse_freqs_front_part=True,
            nope_first=False,
        )

        self.assertTrue(q_ref.shape == q2d.shape)
        self.assertTrue(k_ref.shape == k2d.shape)
        torch.testing.assert_close(q_ref, q_sbhd.view_as(q2d), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k_ref, k_sbhd.view_as(k2d), atol=1e-2, rtol=1e-2)

    def test_ops_equivalence_nope_first(self) -> None:
        import aiter as ops
        from aiter.rotary_embedding import RotaryEmbedding as AiterRotaryEmbedding

        (
            head_size,
            rotary_dim,
            max_pos,
            base,
            is_neox,
            dtype,
            device,
            bs,
            seq_len,
            num_q,
            num_kv,
        ) = (
            128,
            64,
            2048,
            10000,
            True,
            torch.bfloat16,
            "cuda",
            1,
            16,
            2,
            2,
        )

        rope = AiterRotaryEmbedding(
            head_size, rotary_dim, max_pos, base, is_neox, dtype
        ).to(device)

        positions = torch.arange(seq_len, device=device).repeat(bs)
        num_tokens = positions.numel()

        q2d = torch.randn(num_tokens, num_q * head_size, dtype=dtype, device=device)
        k2d = torch.randn(num_tokens, num_kv * head_size, dtype=dtype, device=device)

        q_ref, k_ref = rope.forward_hip(
            positions.clone(), q2d.clone(), k2d.clone(), is_nope_first=True
        )

        q_sbhd = q2d.view(1, num_tokens, num_q, head_size)
        k_sbhd = k2d.view(1, num_tokens, num_kv, head_size)

        cos = rope.cos_cache.to(device=device, dtype=dtype)
        sin = rope.sin_cache.to(device=device, dtype=dtype)
        pos_b_s = positions.view(1, num_tokens)
        rotate_style = 0 if is_neox else 1

        q_rot = q_sbhd[..., -rotary_dim:]
        k_rot = k_sbhd[..., -rotary_dim:]
        ops.rope_cached_positions_2c_fwd_inplace(
            q_rot,
            k_rot,
            cos,
            sin,
            pos_b_s,
            rotate_style,
            reuse_freqs_front_part=True,
            nope_first=True,
        )

        torch.testing.assert_close(q_ref, q_sbhd.view_as(q2d), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k_ref, k_sbhd.view_as(k2d), atol=1e-2, rtol=1e-2)

    def test_sglang_rotary_embedding_forward_hip_matches_native(self) -> None:
        from sglang.srt.layers.rotary_embedding import (
            RotaryEmbedding as SglRotaryEmbedding,
        )

        (
            head_size,
            rotary_dim,
            max_pos,
            base,
            is_neox,
            dtype,
            device,
            bs,
            seq_len,
            num_q,
            num_kv,
        ) = (
            128,
            64,
            2048,
            10000,
            True,
            torch.bfloat16,
            "cuda",
            2,
            64,
            4,
            2,
        )

        rope = SglRotaryEmbedding(
            head_size, rotary_dim, max_pos, base, is_neox, dtype
        ).to(device)

        positions = torch.arange(seq_len, device=device).repeat(bs)
        q = torch.randn(bs * seq_len, num_q * head_size, dtype=dtype, device=device)
        k = torch.randn(bs * seq_len, num_kv * head_size, dtype=dtype, device=device)

        q_ref, k_ref = rope.forward_native(positions.clone(), q.clone(), k.clone())
        q_hip, k_hip = rope.forward_hip(positions.clone(), q.clone(), k.clone())

        torch.testing.assert_close(q_ref, q_hip, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k_ref, k_hip, atol=1e-2, rtol=1e-2)

    def test_llama3_rotary_embedding_forward_hip_matches_native(self) -> None:
        from sglang.srt.layers.rotary_embedding import get_rope as sgl_get_rope

        (
            head_size,
            rotary_dim,
            max_pos,
            base,
            is_neox,
            dtype,
            device,
            bs,
            seq_len,
            num_q,
            num_kv,
        ) = (
            128,
            128,
            2048,
            10000,
            True,
            torch.bfloat16,
            "cuda",
            2,
            64,
            4,
            2,
        )

        rope = sgl_get_rope(
            head_size,
            rotary_dim,
            max_pos,
            base,
            is_neox,
            rope_scaling={
                "rope_type": "llama3",
                "factor": 1.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 1.0,
                "original_max_position_embeddings": max_pos,
            },
            dtype=dtype,
        ).to(device)

        positions = torch.arange(seq_len, device=device).repeat(bs)
        q = torch.randn(bs * seq_len, num_q * head_size, dtype=dtype, device=device)
        k = torch.randn(bs * seq_len, num_kv * head_size, dtype=dtype, device=device)

        q_ref, k_ref = rope.forward_native(positions.clone(), q.clone(), k.clone())
        q_hip, k_hip = rope.forward_hip(positions.clone(), q.clone(), k.clone())

        torch.testing.assert_close(q_ref, q_hip, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k_ref, k_hip, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
