import unittest

import sgl_kernel
import torch
from utils import precision

from sglang.srt.layers.rotary_embedding import DeepseekScalingRotaryEmbedding
from sglang.test.test_utils import CustomTestCase


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
                q_pe_clone, k_pe_clone = (
                    torch.ops.sgl_kernel.rotary_position_embedding_cpu(
                        positions, q_pe_clone, k_pe_clone, cos_sin_cache
                    )
                )

                atol = rtol = precision[q_pe.dtype]
                self.assertTrue(torch.allclose(q_pe, q_pe_clone, atol=atol, rtol=rtol))
                self.assertTrue(torch.allclose(k_pe, k_pe_clone, atol=atol, rtol=rtol))
                torch.testing.assert_close(k_pe, k_pe_clone)


if __name__ == "__main__":
    unittest.main()
