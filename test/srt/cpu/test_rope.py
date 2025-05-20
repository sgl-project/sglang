import unittest

import torch
from sgl_kernel.common_ops import (
    rotary_position_embedding_cpu as rotary_position_embedding,
)

from sglang.test.test_utils import CustomTestCase


class TestROPE(CustomTestCase):
    def _rotate_neox(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _rotate_gptj(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)

    def _forward_ref(self, positions, query, key, cos_sin_cache, offsets=None):
        self.rotary_dim = 64
        self.head_size = 64
        self.is_neox_style = False
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        cos_sin = cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = self._rotate_neox if self.is_neox_style else self._rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
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
                q_pe, k_pe = self._forward_ref(positions, q_pe, k_pe, cos_sin_cache)

                # fused rope kernel
                q_pe_clone, k_pe_clone = rotary_position_embedding(
                    positions, q_pe_clone, k_pe_clone, cos_sin_cache
                )

                torch.testing.assert_close(q_pe, q_pe_clone)
                torch.testing.assert_close(k_pe, k_pe_clone)


if __name__ == "__main__":
    unittest.main()
