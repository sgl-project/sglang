"""Small audio attention, SDPA fallback and Transformers integration."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.kernels.ops.attention.mimo_local_attention import mimo_local_attention
from sglang.srt.models.mimo_audio import (
    _mimo_local_attention_forward,
    _register_mimo_local_attention,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestMiMoLocalAttention(CustomTestCase):
    def test_shapes_strides_causality_and_graph(self):
        for dtype in [torch.bfloat16, torch.float16]:
            for batch, heads, tokens in [
                (3, 7, 1),
                (3, 7, 2),
                (3, 7, 3),
                (32, 64, 4),
                (1875, 64, 4),
            ]:
                for causal in [False, True]:
                    for token_major in [False, True]:
                        with self.subTest(
                            dtype=dtype,
                            shape=(batch, heads, tokens),
                            causal=causal,
                            token_major=token_major,
                        ):
                            if token_major:
                                q, k, v = [
                                    torch.randn(
                                        batch,
                                        tokens,
                                        heads,
                                        16,
                                        device="cuda",
                                        dtype=dtype,
                                    ).transpose(1, 2)
                                    for _ in range(3)
                                ]
                            else:
                                q, k, v = [
                                    torch.randn(
                                        batch,
                                        heads,
                                        tokens,
                                        16,
                                        device="cuda",
                                        dtype=dtype,
                                    )
                                    for _ in range(3)
                                ]
                            mimo_local_attention(q, k, v, is_causal=causal)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                actual = mimo_local_attention(q, k, v, is_causal=causal)
                            for factor in [1.0, -1.0, 0.0]:
                                q.mul_(factor)
                                k.neg_()
                                graph.replay()
                                scores = q.float() @ k.float().transpose(-1, -2) * 0.25
                                if causal:
                                    allowed = torch.ones(
                                        tokens, tokens, device="cuda", dtype=torch.bool
                                    ).tril()
                                    scores.masked_fill_(~allowed, -float("inf"))
                                probabilities = torch.exp(
                                    scores - scores.amax(dim=-1, keepdim=True)
                                )
                                denominator = probabilities.sum(dim=-1, keepdim=True)
                                expected = (
                                    (
                                        (probabilities.to(dtype).float() @ v.float())
                                        / denominator
                                    )
                                    .transpose(1, 2)
                                    .to(dtype)
                                )
                                torch.testing.assert_close(
                                    actual, expected, rtol=0.02, atol=0.02
                                )
                                error = (
                                    (actual.float() - expected.float())
                                    .square()
                                    .mean()
                                    .sqrt()
                                )
                                reference = (
                                    expected.float()
                                    .square()
                                    .mean()
                                    .sqrt()
                                    .clamp_min(1e-6)
                                )
                                self.assertLess(float(error / reference), 0.001)
                                self.assertTrue(actual.is_contiguous())
                                if dtype == torch.bfloat16 and heads == 64:
                                    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                                        cudnn = F.scaled_dot_product_attention(
                                            q, k, v, is_causal=causal
                                        ).transpose(1, 2)
                                    difference = actual.float() - cudnn.float()
                                    relative_error = (
                                        difference.square().mean().sqrt() / reference
                                    )
                                    self.assertLess(float(relative_error), 0.0001)

    def test_sdpa_mask_and_dimension_fallback(self):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        module = SimpleNamespace(num_key_value_groups=1, is_causal=True)
        for tokens, dim, masked in [(4, 16, True), (5, 16, False), (4, 32, False)]:
            q, k, v = [
                torch.randn(3, 7, tokens, dim, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            mask = (
                torch.ones(tokens, tokens, device="cuda", dtype=torch.bool).tril()
                if masked
                else None
            )
            actual = _mimo_local_attention_forward(
                module, q, k, v, mask, is_causal=False
            )[0]
            expected = sdpa_attention_forward(module, q, k, v, mask, is_causal=False)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_transformers_local_model_and_mask_registration(self):
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

        config = Qwen2Config(
            vocab_size=32,
            hidden_size=112,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=7,
            num_key_value_heads=7,
            head_dim=16,
        )
        config._attn_implementation = "sdpa"
        control = Qwen2Model(config).cuda().bfloat16().eval()
        candidate_config = copy.deepcopy(config)
        candidate_config._attn_implementation = _register_mimo_local_attention()
        candidate = Qwen2Model(candidate_config).cuda().bfloat16().eval()
        candidate.load_state_dict(control.state_dict())
        x = torch.randn(3, 4, 112, device="cuda", dtype=torch.bfloat16)
        for causal in [False, True]:
            with (
                torch.no_grad(),
                patch(
                    "sglang.kernels.ops.attention.mimo_local_attention.mimo_local_attention",
                    wraps=mimo_local_attention,
                ) as kernel,
            ):
                actual = candidate(inputs_embeds=x, is_causal=causal).last_hidden_state
                if torch.cuda.get_device_capability()[0] == 9:
                    self.assertEqual(kernel.call_count, 2)
                expected = control(inputs_embeds=x, is_causal=causal).last_hidden_state
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.03)


if __name__ == "__main__":
    unittest.main()
