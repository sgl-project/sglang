import unittest

import torch

from sglang.srt.models.iquest_q1 import _apply_learned_sink
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def torch_reference(query, sink_key, attn_output, lse, scale):
    group = query.shape[1] // sink_key.shape[0]
    sink = sink_key.to(query.dtype).repeat_interleave(group, dim=0)
    sink_logit = (query.float() * sink.float()).sum(-1) * scale
    factor = torch.sigmoid(lse.float() - sink_logit)
    return (attn_output.float() * factor.unsqueeze(-1)).to(attn_output.dtype)


class TestIQuestQ1LearnedSink(CustomTestCase):
    def test_matches_torch_across_layouts_and_dtypes(self):
        torch.manual_seed(17)
        # (tokens, heads, kv_heads, head_dim): empty batch, row-tail mask, model GQA,
        # MHA with head_dim 80, the non-128 reduction path, the 64-row prefill tile.
        shapes = (
            (0, 6, 1, 128),
            (7, 6, 1, 128),
            (9, 48, 8, 128),
            (37, 16, 16, 80),
            (17, 8, 1, 256),
            (16384, 6, 1, 128),
        )
        cases = [
            (shape, dtype, strided)
            for shape in shapes
            for dtype, strided in (
                (torch.float32, False),
                (torch.bfloat16, False),
                (torch.bfloat16, True),
            )
        ]
        for shape, dtype, strided in cases:
            with self.subTest(shape=shape, dtype=dtype, strided=strided):
                tokens, heads, kv_heads, head_dim = shape
                step = 2 if strided else 1
                q = torch.randn(
                    tokens, heads, head_dim * step, device="cuda", dtype=dtype
                )[..., ::step]
                output = torch.randn_like(q)
                if strided:
                    output = torch.randn(
                        heads, tokens, head_dim * step, device="cuda", dtype=dtype
                    ).transpose(0, 1)[..., ::step]
                sink = torch.randn(kv_heads, head_dim * step, device="cuda")[:, ::step]
                lse = torch.randn(heads, tokens, device="cuda").T
                if not strided:
                    lse = lse.contiguous()
                original = output.clone()
                scale = head_dim**-0.5
                expected = torch_reference(q, sink, output, lse, scale)
                actual = _apply_learned_sink(q, sink, output, lse, scale)
                rtol = 3e-5 if dtype == torch.float32 else torch.finfo(dtype).eps
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=5e-7)
                torch.testing.assert_close(output, original, rtol=0, atol=0)
                self.assertEqual(actual.shape, q.shape)
                self.assertEqual(actual.dtype, output.dtype)
                self.assertTrue(actual.is_contiguous())

    def test_extreme_lse(self):
        q = torch.ones(5, 6, 128, device="cuda")
        sink = torch.zeros(1, 128, device="cuda")
        output = torch.randn_like(q)
        lse = (
            torch.tensor([-float("inf"), -1000, 0, 1000, float("inf")], device="cuda")
            .unsqueeze(-1)
            .expand(5, 6)
        )
        actual = _apply_learned_sink(q, sink, output, lse, 128**-0.5)
        factors = torch.tensor([0, 0, 0.5, 1, 1], device="cuda")
        torch.testing.assert_close(
            actual, output * factors[:, None, None], rtol=0, atol=0
        )

    def test_torch_compile_fullgraph_preserves_outputs(self):
        torch.manual_seed(41)
        compiled = torch.compile(_apply_learned_sink, fullgraph=True)
        for tokens in (1, 8):
            with self.subTest(tokens=tokens):
                q = torch.randn(tokens, 6, 128, device="cuda", dtype=torch.bfloat16)
                sink = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
                output = torch.randn_like(q)
                lse = torch.randn(tokens, 6, device="cuda")
                for _ in range(2):
                    q.normal_()
                    sink.normal_()
                    output.normal_()
                    lse.normal_()
                    expected = _apply_learned_sink(q, sink, output, lse, 128**-0.5)
                    actual = compiled(q, sink, output, lse, 128**-0.5)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
