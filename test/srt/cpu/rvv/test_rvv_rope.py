"""Unit tests for RVV rotary embedding kernels."""

import unittest

import torch

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

from .rvv_utils import has_sgl_kernel_op, helper_non_contiguous, precision

torch.manual_seed(1234)


@unittest.skipUnless(
    has_sgl_kernel_op("rotary_embedding_cpu"),
    "sgl_kernel rotary_embedding_cpu not available (non-RISC-V build)",
)
class TestRVVRopeCore(CustomTestCase):
    """Test suite for RVV RoPE kernel compatibility checks."""

    # (head_size, rotary_dim, max_pos, base, is_neox, dtype, device, batch, seq, q_heads, kv_heads)
    test_config = [
        (64, 64, 32, 8000, True, torch.bfloat16, "cpu", 32, 32, 1, 1),
        (64, 64, 32, 8000, True, torch.float16, "cpu", 32, 32, 1, 1),  # FP16 neox
        (256, 128, 4096, 10000, True, torch.bfloat16, "cpu", 2, 512, 32, 8),
        (512, 128, 311, 10000, True, torch.bfloat16, "cpu", 3, 39, 4, 2),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cpu", 2, 512, 32, 8),
        (128, 128, 2048, 10000, False, torch.float16, "cpu", 2, 512, 32, 8),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cpu", 2, 512, 16, 4),
        (512, 128, 311, 10000, False, torch.bfloat16, "cpu", 3, 39, 4, 2),
        # Non-neox tail case for partial-vector handling.
        (128, 96, 2048, 10000, False, torch.bfloat16, "cpu", 2, 64, 8, 2),
        (128, 96, 2048, 10000, False, torch.float16, "cpu", 2, 64, 8, 2),
    ]

    def _run_rope(
        self,
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        batch_size,
        seq_len,
        num_q_heads,
        num_kv_heads,
        dims,
        is_neox_style,
        device,
        dtype,
    ):
        """Run one RoPE case for either 2D or 4D tensor layout."""
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        torch.manual_seed(100)
        rope_ref = RotaryEmbedding(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
        ).to(device)
        pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
        query = torch.randn(
            batch_size * seq_len,
            num_q_heads * head_size,
            dtype=dtype,
            device=device,
        )
        key = torch.randn(
            batch_size * seq_len,
            num_kv_heads * head_size,
            dtype=dtype,
            device=device,
        )
        if dims == 3:
            # 3D layout requires num_kv_heads == 1.
            query = query.view(batch_size * seq_len, num_q_heads, head_size)
            key = key.view(batch_size * seq_len, num_kv_heads, head_size)
        elif dims == 4:
            query = query.view(batch_size, seq_len, num_q_heads, head_size)
            key = key.view(batch_size, seq_len, num_kv_heads, head_size)
        query_ref, key_ref = query.clone(), key.clone()
        query_cpu, key_cpu = query.clone(), key.clone()

        query_ref_out, key_ref_out = rope_ref.forward_native(
            pos_ids, query_ref, key_ref
        )
        query_cpu_out, key_cpu_out = torch.ops.sgl_kernel.rotary_embedding_cpu(
            pos_ids,
            query_cpu,
            key_cpu,
            rope_ref.head_size,
            rope_ref.cos_sin_cache.to(query.dtype),
            rope_ref.is_neox_style,
        )
        atol = rtol = precision["rotary_embedding"][dtype]
        torch.testing.assert_close(query_ref_out, query_cpu_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(key_ref_out, key_cpu_out, atol=atol, rtol=rtol)

    def test_rope_2d(self):
        """2D layout with varied head sizes, rotary dims, and neox/non-neox styles."""
        for cfg in self.test_config:
            hs, rd, mp, base, neox, dt, dev, bs, sl, qh, kvh = cfg
            with self.subTest(
                head_size=hs, rotary_dim=rd, is_neox=neox, batch=bs, seq=sl
            ):
                self._run_rope(
                    head_size=hs,
                    rotary_dim=rd,
                    max_position_embeddings=mp,
                    base=base,
                    batch_size=bs,
                    seq_len=sl,
                    num_q_heads=qh,
                    num_kv_heads=kvh,
                    dims=2,
                    is_neox_style=neox,
                    device=dev,
                    dtype=dt,
                )

    def test_rope_4d(self):
        """4D layout adds a batch dimension; must produce identical output to 2D."""
        for cfg in self.test_config:
            hs, rd, mp, base, neox, dt, dev, bs, sl, qh, kvh = cfg
            with self.subTest(
                head_size=hs, rotary_dim=rd, is_neox=neox, batch=bs, seq=sl
            ):
                self._run_rope(
                    head_size=hs,
                    rotary_dim=rd,
                    max_position_embeddings=mp,
                    base=base,
                    batch_size=bs,
                    seq_len=sl,
                    num_q_heads=qh,
                    num_kv_heads=kvh,
                    dims=4,
                    is_neox_style=neox,
                    device=dev,
                    dtype=dt,
                )

    def test_rope_2d_non_contiguous(self):
        """Non-contiguous Q/K must not read wrong positions due to stride errors."""
        head_size, rotary_dim, max_pos = 128, 128, 2048
        base, is_neox = 10000, False
        device = "cpu"
        batch_size, seq_len, num_q_heads, num_kv_heads = 4, 32, 8, 2

        for dtype in [torch.bfloat16, torch.float16]:
            with self.subTest(dtype=dtype):
                set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
                torch.manual_seed(100)
                rope_ref = RotaryEmbedding(
                    head_size, rotary_dim, max_pos, base, is_neox, dtype
                ).to(device)
                pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)

                query = helper_non_contiguous(
                    torch.randn(
                        batch_size * seq_len, num_q_heads * head_size, dtype=dtype
                    )
                )
                key = helper_non_contiguous(
                    torch.randn(
                        batch_size * seq_len, num_kv_heads * head_size, dtype=dtype
                    )
                )
                self.assertFalse(query.is_contiguous())

                query_ref_out, key_ref_out = rope_ref.forward_native(
                    pos_ids, query.clone(), key.clone()
                )
                query_out, key_out = torch.ops.sgl_kernel.rotary_embedding_cpu(
                    pos_ids,
                    query.clone(),
                    key.clone(),
                    rope_ref.head_size,
                    rope_ref.cos_sin_cache.to(dtype),
                    rope_ref.is_neox_style,
                )
                atol = rtol = precision["rotary_embedding"][dtype]
                torch.testing.assert_close(
                    query_ref_out, query_out, atol=atol, rtol=rtol
                )
                torch.testing.assert_close(key_ref_out, key_out, atol=atol, rtol=rtol)


@unittest.skipUnless(
    has_sgl_kernel_op("rotary_embedding_cpu"),
    "sgl_kernel rotary_embedding_cpu not available (non-RISC-V build)",
)
class TestRVVRope3D(TestRVVRopeCore):
    """Test suite for the 3D RoPE kernel path (num_kv_heads == 1, non-neox only)."""

    # 3D requires num_kv_heads == 1.
    test_config_3d = [
        (128, 64, 2048, 10000, False, torch.bfloat16, "cpu", 2, 32, 8, 1),
        (64, 64, 512, 8000, False, torch.float16, "cpu", 4, 16, 4, 1),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cpu", 2, 64, 16, 1),
        # Tail case for partial-vector handling.
        (128, 48, 2048, 10000, False, torch.bfloat16, "cpu", 2, 32, 4, 1),
        (128, 48, 2048, 10000, False, torch.float16, "cpu", 2, 32, 4, 1),
    ]

    def test_rope_3d(self):
        """3D layout (num_kv_heads=1) uses a different kernel path; must match 2D reference."""
        for cfg in self.test_config_3d:
            hs, rd, mp, base, neox, dt, dev, bs, sl, qh, kvh = cfg
            with self.subTest(head_size=hs, rotary_dim=rd, batch=bs, seq=sl, dtype=dt):
                self._run_rope(
                    head_size=hs,
                    rotary_dim=rd,
                    max_position_embeddings=mp,
                    base=base,
                    batch_size=bs,
                    seq_len=sl,
                    num_q_heads=qh,
                    num_kv_heads=kvh,
                    dims=3,
                    is_neox_style=neox,
                    device=dev,
                    dtype=dt,
                )

    def test_rope_3d_neox_raises(self):
        """3D + neox=True is unsupported; must raise rather than silently produce wrong output."""
        with self.assertRaises(RuntimeError):
            self._run_rope(
                head_size=128,
                rotary_dim=64,
                max_position_embeddings=2048,
                base=10000,
                batch_size=2,
                seq_len=32,
                num_q_heads=8,
                num_kv_heads=1,
                dims=3,
                is_neox_style=True,
                device="cpu",
                dtype=torch.bfloat16,
            )


if __name__ == "__main__":
    unittest.main()
