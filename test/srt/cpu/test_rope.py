import unittest

import sgl_kernel
import torch
from utils import precision

from sglang.srt.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    RotaryEmbedding,
)
from sglang.test.test_utils import CustomTestCase

torch.manual_seed(0)


class TestROPE(CustomTestCase):
    def test_deepseek_v2_rope(self):
        num_head = 16
        seq_len = 1024
        q_head_dim = 192
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        max_pos = 256
        k_dim = 576
        rotary_dim = 64
        is_neox_style = False

        # Create cos_sin_cache
        freqs = torch.rand(max_pos, qk_rope_head_dim // 2)
        cos = freqs.cos() * 0.7
        sin = freqs.sin() * 0.7
        cos_sin_cache = torch.cat((cos, sin), dim=-1).to(torch.bfloat16)
        positions = torch.randint(0, max_pos, (seq_len,))

        rope = DeepseekScalingRotaryEmbedding(
            qk_rope_head_dim,
            rotary_dim,
            max_pos,
            16,  # not used since cos_sin_cache is provided
            is_neox_style,
            1.0,
            torch.bfloat16,
            device="cpu",
        )
        rope.register_buffer("cos_sin_cache", cos_sin_cache)

        for dtype in [torch.bfloat16]:
            enable_autocast = True

            with torch.no_grad(), torch.amp.autocast("cpu", enabled=enable_autocast):
                q = torch.randn(seq_len, num_head, q_head_dim, dtype=dtype)
                q_clone = q.clone()
                k = torch.randn(seq_len, 1, k_dim, dtype=dtype)
                k_clone = k.clone()
                _, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
                _, q_pe_clone = q_clone.split(
                    [qk_nope_head_dim, qk_rope_head_dim], dim=-1
                )
                k_pe = k[:, :, k_dim - qk_rope_head_dim :]
                k_pe_clone = k_clone[:, :, k_dim - qk_rope_head_dim :]

                # ref kernel
                q_pe, k_pe = rope.forward_native(
                    query=q_pe,
                    key=k_pe,
                    positions=positions,
                )

                # fused rope kernel
                q_pe_clone, k_pe_clone = torch.ops.sgl_kernel.rotary_embedding_cpu(
                    positions,
                    q_pe_clone,
                    k_pe_clone,
                    rope.head_size,
                    cos_sin_cache,
                    False,
                )

                atol = rtol = precision[q_pe.dtype]
                torch.testing.assert_close(q_pe, q_pe_clone, atol=atol, rtol=rtol)
                torch.testing.assert_close(k_pe, k_pe_clone, atol=atol, rtol=rtol)
                torch.testing.assert_close(k_pe, k_pe_clone)

    def test_origin_rope(self):
        def single_test(
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: int,
            is_neox_style: bool,
            dtype: torch.dtype,
            device: str,
            batch_size: int,
            seq_len: int,
            num_q_heads: int,
            num_kv_heads: int,
        ):
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
            torch.testing.assert_close(
                query_ref_out, query_cpu_out, atol=1e-2, rtol=1e-2
            )
            torch.testing.assert_close(key_ref_out, key_cpu_out, atol=1e-2, rtol=1e-2)

        test_config = [
            (64, 64, 32, 8000, True, torch.bfloat16, "cpu", 32, 32, 1, 1),
            (256, 128, 4096, 10000, True, torch.bfloat16, "cpu", 2, 512, 32, 8),
            (512, 128, 311, 10000, True, torch.bfloat16, "cpu", 3, 39, 4, 2),
            (128, 128, 2048, 10000, False, torch.bfloat16, "cpu", 2, 512, 32, 8),
            (128, 128, 2048, 10000, False, torch.bfloat16, "cpu", 2, 512, 16, 4),
            (512, 128, 311, 10000, False, torch.bfloat16, "cpu", 3, 39, 4, 2),
        ]

        for (
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
            device,
            batch_size,
            seq_len,
            num_q_heads,
            num_kv_heads,
        ) in test_config:
            single_test(
                head_size,
                rotary_dim,
                max_position_embeddings,
                base,
                is_neox_style,
                dtype,
                device,
                batch_size,
                seq_len,
                num_q_heads,
                num_kv_heads,
            )


if __name__ == "__main__":
    unittest.main()
