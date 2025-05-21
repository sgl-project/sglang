import unittest

import torch
from sgl_kernel.common_ops import (
    rotary_position_embedding_cpu as rotary_position_embedding,
)
from utils import _rotate_gptj, _rotate_neox, precision

from sglang.test.test_utils import CustomTestCase


class TestROPE(CustomTestCase):

    def _forward_ref(
        self,
        positions,
        query,
        key,
        cos_sin_cache,
        rotary_dim,
        head_size,
        is_neox_style,
        offsets=None,
    ):
        query_rot = query[..., :rotary_dim]
        key_rot = key[..., :rotary_dim]
        if rotary_dim < head_size:
            query_pass = query[..., rotary_dim:]
            key_pass = key[..., rotary_dim:]

        cos_sin = cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if is_neox_style:
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if rotary_dim < head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key

    def test_deepseek_v2_rope(self):
        num_head = 16
        seq_len = 1024
        q_head_dim = 192
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        max_pos = 256
        k_dim = 576

        # Create cos_sin_cache
        freqs = torch.rand(max_pos, qk_rope_head_dim // 2)
        cos = freqs.cos() * 0.7
        sin = freqs.sin() * 0.7
        cos_sin_cache = torch.cat((cos, sin), dim=-1).to(torch.bfloat16)
        positions = torch.randint(0, max_pos, (seq_len,))

        for dtype in [torch.bfloat16]:
            enable_autocast = True

            with torch.no_grad(), torch.cpu.amp.autocast(enabled=enable_autocast):
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
                q_pe, k_pe = self._forward_ref(
                    positions, q_pe, k_pe, cos_sin_cache, 64, 64, False
                )

                # fused rope kernel
                q_pe_clone, k_pe_clone = rotary_position_embedding(
                    positions, q_pe_clone, k_pe_clone, cos_sin_cache
                )

                atol = rtol = precision[q_pe.dtype]
                self.assertTrue(torch.allclose(q_pe, q_pe_clone, atol=atol, rtol=rtol))
                self.assertTrue(torch.allclose(k_pe, k_pe_clone, atol=atol, rtol=rtol))
                torch.testing.assert_close(k_pe, k_pe_clone)


if __name__ == "__main__":
    unittest.main()
