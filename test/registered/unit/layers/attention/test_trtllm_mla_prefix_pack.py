"""A TRT-LLM MLA packed prefix chunk must not be quantized twice or lose its KV scales."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    _quantize_fp8_qkv,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


class TestTRTLLMMLAPrefixPack(CustomTestCase):
    def test_packed_prefix_matches_unfused_quantization(self):
        num_tokens, num_heads, qk_nope, qk_rope, v_head = 257, 12, 128, 64, 128
        # Power-of-two scales make k * (1 / s) and k / s round identically.
        for k_scale, v_scale in ((1.0, 1.0), (2.0, 4.0)):
            with self.subTest(k_scale=k_scale, v_scale=v_scale):
                torch.manual_seed(7)
                # Strided views, as sliced from kv_b_proj's output.
                kv = torch.randn(
                    (num_tokens, num_heads, qk_nope + v_head),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                k_nope, v = kv[..., :qk_nope], kv[..., qk_nope:]
                k_pe = torch.randn(
                    (num_tokens, 1, qk_rope), device="cuda", dtype=torch.bfloat16
                )
                q = torch.randn(
                    (num_tokens, num_heads, qk_nope + qk_rope),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                layer = SimpleNamespace(
                    k_scale_float=k_scale,
                    v_scale_float=v_scale,
                    k_scale=torch.tensor([k_scale], device="cuda"),
                    v_scale=torch.tensor([v_scale], device="cuda"),
                )
                backend = TRTLLMMLABackend.__new__(TRTLLMMLABackend)

                packed_k, packed_v = backend.pack_prefix_chunk_kv(
                    k_nope, k_pe, v, layer=layer
                )
                k = torch.cat((k_nope, k_pe.expand(-1, num_heads, -1)), dim=-1)
                expected = _quantize_fp8_qkv(q, k, v, layer)
                actual = _quantize_fp8_qkv(q, packed_k, packed_v, layer)

                self.assertIs(actual[1], packed_k)
                self.assertIs(actual[2], packed_v)
                self.assertEqual(actual[3:], (k_scale, v_scale))
                self.assertEqual(expected[3:], (k_scale, v_scale))
                for got, want in zip(actual[:3], expected[:3]):
                    torch.testing.assert_close(
                        got.float(), want.float(), rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
