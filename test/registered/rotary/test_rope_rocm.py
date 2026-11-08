import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.layers.rotary_embedding import base as rotary_base
from sglang.srt.layers.rotary_embedding import rope_variant
from sglang.srt.utils import get_bool_env_var, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=3, suite="stage-a-test-1-gpu-small-amd")

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


@unittest.skipUnless(_is_hip, "Requires AMD GPU")
class TestRotaryEmbeddingNative(CustomTestCase):
    # SGLang's fallback remains in use by Llama when AITER is enabled.
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
        q_hip, k_hip = rope_hip(pos_ids, query.clone(), key.clone())
        self.assertEqual(rope_hip.cos_sin_cache.dtype, torch.float32)

        # FP32 cache storage must preserve the existing query-dtype arithmetic.
        from sgl_kernel import rotary_embedding

        q_same, k_same = query.clone(), key.clone()
        rotary_embedding(
            pos_ids,
            q_same,
            k_same,
            head_size,
            rope_hip.cos_sin_cache.to(dtype),
            is_neox,
        )
        torch.testing.assert_close(q_hip, q_same, atol=0, rtol=0)
        torch.testing.assert_close(k_hip, k_same, atol=0, rtol=0)

        torch.testing.assert_close(q_ref, q_hip, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k_ref, k_hip, atol=1e-2, rtol=1e-2)

    def test_all_cases(self) -> None:
        """Drive over the full parameter matrix using subTest()."""
        cases = _CASES + [
            (128, 64, 32, 10000, True, torch.float16, "cuda", 1, 32, 4, 2),
            (128, 64, 32, 10000, False, torch.float16, "cuda", 1, 32, 4, 2),
            (128, 64, 32, 10000, True, torch.float32, "cuda", 1, 32, 4, 2),
        ]
        for case in cases:
            with self.subTest(case=case):
                self._run_case(*case)


@unittest.skipIf(not _use_aiter, reason="Requires AMD GPU plus SGLANG_USE_AITER=1")
class TestRotaryEmbeddingAITer(CustomTestCase):
    # NOTE: Slightly relaxed tolerance (2e-2 vs 1e-2) for AITER RoPE kernel.
    # Minor precision differences under investigation.
    # See: https://github.com/sgl-project/sglang/pull/15318

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

        torch.testing.assert_close(q_ref, q_hip, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k_ref, k_hip, atol=2e-2, rtol=2e-2)

    def test_all_cases(self) -> None:
        for case in _CASES:
            with self.subTest(case=case):
                self._run_case_aiter(*case)


@unittest.skipUnless(
    rope_variant._use_aiter_rope_gfx1250,
    "Requires gfx1250 plus SGLANG_USE_AITER=1",
)
class TestDeepseekScalingRopeAiterGfx1250(CustomTestCase):
    # Shapes follow the MLA callers: q_pe is the rope slice of q, k_pe the rope
    # slice of the latent KV row.
    def _run_case(self, is_neox: bool, num_tokens: int, num_heads: int) -> None:
        rope = rope_variant.DeepseekScalingRotaryEmbedding(
            64,
            64,
            4096,
            50000,
            is_neox,
            32.0,
            torch.bfloat16,
            mscale=1.0,
            mscale_all_dim=1.0,
            device="cuda",
        ).to("cuda")
        positions = torch.randint(0, 4096, (num_tokens,), device="cuda")
        q = torch.randn(num_tokens, num_heads, 192, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(num_tokens, 1, 576, dtype=torch.bfloat16, device="cuda")
        q_pe, k_pe = q[..., 128:], kv[..., 512:]

        q_ref, k_ref = rope.forward_native(
            positions, q_pe.contiguous(), k_pe.contiguous()
        )
        q_out, k_out = rope(positions, q_pe, k_pe)

        self.assertEqual(q_out.data_ptr(), q_pe.data_ptr())
        self.assertEqual(k_out.data_ptr(), k_pe.data_ptr())
        # forward_native adds a leading dim on the neox path.
        torch.testing.assert_close(
            q[..., 128:], q_ref.reshape(q_pe.shape), atol=3e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            kv[..., 512:], k_ref.reshape(k_pe.shape), atol=3e-2, rtol=2e-2
        )

    def test_matches_native(self) -> None:
        for is_neox in (False, True):
            for num_tokens, num_heads in ((1, 16), (311, 8), (2048, 64)):
                with self.subTest(is_neox=is_neox, num_tokens=num_tokens):
                    self._run_case(is_neox, num_tokens, num_heads)

    def test_rebuilds_after_cache_growth(self) -> None:
        rope = self._small_rope()
        q = torch.randn(4, 16, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(4, 1, 64, dtype=torch.bfloat16, device="cuda")
        rope(torch.arange(4, device="cuda"), q.clone(), k.clone())

        old_cache = rope.cos_sin_cache
        rope._ensure_cos_sin_cache_length(1024)
        self.assertIsNot(rope.cos_sin_cache, old_cache)
        positions = torch.tensor([0, 255, 600, 1000], device="cuda")
        q_ref, k_ref = rope.forward_native(positions, q.clone(), k.clone())
        q_out, k_out = rope(positions, q.clone(), k.clone())
        self.assertIs(rope._aiter_cos_sin[0], rope.cos_sin_cache)
        torch.testing.assert_close(q_out, q_ref, atol=3e-2, rtol=2e-2)
        torch.testing.assert_close(k_out, k_ref, atol=3e-2, rtol=2e-2)

    def _small_rope(self) -> rope_variant.DeepseekScalingRotaryEmbedding:
        return rope_variant.DeepseekScalingRotaryEmbedding(
            64,
            64,
            256,
            50000,
            False,
            1.0,
            torch.bfloat16,
            mscale=1.0,
            mscale_all_dim=1.0,
            device="cuda",
        ).to("cuda")

    def test_rebuilds_on_dtype_change(self) -> None:
        rope = self._small_rope()
        positions = torch.arange(4, device="cuda")
        for dtype in (torch.bfloat16, torch.float16, torch.bfloat16):
            q = torch.randn(4, 16, 64, dtype=dtype, device="cuda")
            k = torch.randn(4, 1, 64, dtype=dtype, device="cuda")
            q_ref, k_ref = rope.forward_native(positions, q.clone(), k.clone())
            q_out, k_out = rope(positions, q.clone(), k.clone())
            self.assertEqual(rope._aiter_cos_sin[1].dtype, dtype)
            torch.testing.assert_close(q_out, q_ref, atol=3e-2, rtol=2e-2)
            torch.testing.assert_close(k_out, k_ref, atol=3e-2, rtol=2e-2)

    def test_offsets_use_native(self) -> None:
        rope = self._small_rope()
        positions = torch.arange(4, device="cuda")
        q = torch.randn(4, 16, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(4, 1, 64, dtype=torch.bfloat16, device="cuda")
        offsets = torch.full((4,), 7, device="cuda")
        q_ref, k_ref = rope.forward_native(positions, q.clone(), k.clone(), offsets)
        q_out, k_out = rope(positions, q.clone(), k.clone(), offsets)
        torch.testing.assert_close(q_out, q_ref, atol=3e-2, rtol=2e-2)
        torch.testing.assert_close(k_out, k_ref, atol=3e-2, rtol=2e-2)

    def test_key_passed_as_its_own_query(self) -> None:
        # IndexerKPool's K-only path passes the rope slice of the key twice.
        rope = self._small_rope()
        positions = torch.arange(4, device="cuda")
        key = torch.randn(4, 1, 128, dtype=torch.bfloat16, device="cuda")
        k_rope = key[..., :64]
        k_ref = rope.forward_native(positions, k_rope.clone(), k_rope.clone())[1]
        rope(positions, k_rope, k_rope)
        torch.testing.assert_close(key[..., :64], k_ref, atol=3e-2, rtol=2e-2)

    def test_force_native_keeps_native(self) -> None:
        exec_ctx = SimpleNamespace(
            deterministic=SimpleNamespace(rl_on_policy_target="fsdp")
        )
        with (
            patch.object(rotary_base, "publish_role", return_value="rollout"),
            patch.object(rotary_base, "get_exec", return_value=exec_ctx),
        ):
            rope = self._small_rope()
        self.assertEqual(rope._forward_method, rope.forward_native)


if __name__ == "__main__":
    unittest.main()
