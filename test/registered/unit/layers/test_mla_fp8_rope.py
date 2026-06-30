import unittest

import torch

from sglang.srt.layers.attention.utils import mla_quantize_and_rope_for_fp8
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _apply_rope(x, pos_ids, cos_sin_cache, is_neox):
    cos_sin = cos_sin_cache.index_select(0, pos_ids)
    cos, sin = cos_sin.chunk(2, dim=-1)
    if x.dim() == 2:
        return apply_rotary_emb(x.unsqueeze(1), cos, sin, is_neox).squeeze(1)
    return apply_rotary_emb(x, cos, sin, is_neox)


class TestMlaFp8Rope(CustomTestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_mismatched_q_k_lengths_use_separate_k_positions(self):
        device = torch.device("cuda")
        dtype = torch.bfloat16
        fp8_dtype = torch.float8_e4m3fn
        q_len = 2
        k_len = 5
        num_heads = 3
        kv_lora_rank = 4
        qk_rope_head_dim = 8

        freqs = (
            torch.arange(10, dtype=torch.float32, device=device).unsqueeze(1)
            * torch.arange(
                1, qk_rope_head_dim // 2 + 1, dtype=torch.float32, device=device
            ).unsqueeze(0)
            / 100.0
        )
        cos_sin_cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).to(
            dtype
        )
        q_pos_ids = torch.tensor([4, 7], dtype=torch.int64, device=device)
        k_pos_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)

        for is_neox in (True, False):
            with self.subTest(is_neox=is_neox):
                torch.manual_seed(0)
                q_nope = torch.randn(
                    q_len, num_heads, kv_lora_rank, dtype=dtype, device=device
                )
                q_rope = torch.randn(
                    q_len, num_heads, qk_rope_head_dim, dtype=dtype, device=device
                )
                k_nope = torch.randn(k_len, kv_lora_rank, dtype=dtype, device=device)
                k_rope = torch.randn(
                    k_len, qk_rope_head_dim, dtype=dtype, device=device
                )

                q_out, k_nope_out, k_rope_out = mla_quantize_and_rope_for_fp8(
                    q_nope,
                    q_rope,
                    k_nope,
                    k_rope,
                    q_pos_ids,
                    cos_sin_cache,
                    is_neox,
                    kv_lora_rank,
                    qk_rope_head_dim,
                    k_pos_ids=k_pos_ids,
                )

                expected_q = torch.empty(
                    q_len,
                    num_heads,
                    kv_lora_rank + qk_rope_head_dim,
                    dtype=fp8_dtype,
                    device=device,
                )
                expected_q[..., :kv_lora_rank] = q_nope.to(fp8_dtype)
                expected_q[..., kv_lora_rank:] = _apply_rope(
                    q_rope, q_pos_ids, cos_sin_cache, is_neox
                ).to(fp8_dtype)
                expected_k_nope = k_nope.to(fp8_dtype)
                expected_k_rope = _apply_rope(
                    k_rope, k_pos_ids, cos_sin_cache, is_neox
                ).to(fp8_dtype)

                torch.testing.assert_close(
                    q_out[..., :kv_lora_rank].float(),
                    expected_q[..., :kv_lora_rank].float(),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    k_nope_out.float(), expected_k_nope.float(), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    q_out[..., kv_lora_rank:].float(),
                    expected_q[..., kv_lora_rank:].float(),
                    rtol=0,
                    atol=0.25,
                )
                torch.testing.assert_close(
                    k_rope_out.float(), expected_k_rope.float(), rtol=0, atol=0.25
                )


if __name__ == "__main__":
    unittest.main()
