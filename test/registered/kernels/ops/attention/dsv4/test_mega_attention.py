import unittest

import torch

from sglang.kernels.ops.attention.dsv4.mega_attention import (
    flashmla_mega_decode,
    flashmla_mega_prefill,
    pack_mega_query,
)
from sglang.kernels.ops.attention.dsv4.torch_quant import (
    dequantize_k_cache_v41,
    quantize_k_cache_v41,
)
from sglang.srt.layers.attention.dsv4.mega_attention import (
    prepare_mega_wo_a,
    project_mega_attention,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.dsv41_kv_quant_reference import (
    dequantize_k_cache_v41_fp4,
    quantize_k_cache_v41_fp4,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def unpack_output(data, scale):
    tokens, groups, hidden = data.shape
    sf = scale.contiguous().view(torch.uint8).float()
    values = data.float().view(tokens, groups, hidden // 32, 32)
    values *= torch.exp2(sf - 127).unsqueeze(-1)
    return (
        values.view(tokens, groups, 16, 8, 32)
        .transpose(2, 3)
        .reshape(tokens, groups, hidden)
    )


def paged_view(raw, page_size, bytes_per_token):
    return raw[:, : page_size * bytes_per_token].view(
        raw.shape[0], page_size, 1, bytes_per_token
    )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "MegaAttention requires SM100/SM103",
)
class TestMegaAttention(CustomTestCase):
    def test_query_packing_zeroes_padded_heads(self):
        for heads in (8, 16, 32, 64, 128):
            with self.subTest(heads=heads):
                q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
                padded = 64 if heads <= 64 else 128
                expected = q.new_zeros(3, padded, 512)
                expected[:, :heads] = q
                expected = (
                    expected.view(3, padded, 32, 16)
                    .transpose(1, 2)
                    .contiguous()
                    .view(3, padded, 512)
                )
                torch.testing.assert_close(
                    pack_mega_query(q, padded), expected, rtol=0, atol=0
                )

    def test_nvfp4_decode_prefill_and_wo_projection(self):
        for heads in (16, 64):
            for use_fp4 in (False, True):
                with self.subTest(heads=heads, fp4=use_fp4):
                    self._check_pipeline(heads, use_fp4)

    def _check_pipeline(self, heads, use_fp4):
        torch.manual_seed(456)
        q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
        packed = pack_mega_query(q, 64)
        positions = torch.tensor([0, 7, 127], device="cuda", dtype=torch.int32)
        angle = (
            torch.arange(256, device="cuda")[:, None]
            / torch.arange(1, 33, device="cuda")[None]
        )
        table = torch.cat((angle.cos(), angle.sin()), dim=-1).float()
        main = quantize_k_cache_v41(
            torch.randn(1, 128, 512, device="cuda", dtype=torch.bfloat16)
        )
        latent = torch.randn(1, 64, 512, device="cuda", dtype=torch.bfloat16)
        extra = (
            quantize_k_cache_v41_fp4(latent)
            if use_fp4
            else quantize_k_cache_v41(latent)
        )
        extra_dequant = (
            dequantize_k_cache_v41_fp4(extra, 64)
            if use_fp4
            else dequantize_k_cache_v41(extra, 64)
        )
        indices = torch.arange(128, device="cuda", dtype=torch.int32).repeat(3, 1)
        extra_indices = torch.arange(64, device="cuda", dtype=torch.int32).repeat(3, 1)
        lens = torch.tensor([128, 96, 1], device="cuda", dtype=torch.int32)
        extra_lens = torch.tensor([64, 32, 0], device="cuda", dtype=torch.int32)
        sink = torch.randn(64, device="cuda", dtype=torch.float32)
        kwargs = dict(
            softmax_scale=512**-0.5,
            attn_sink=sink,
            topk_length=lens,
            extra_kv=paged_view(extra, 64, 288 if use_fp4 else 528),
            extra_indices=extra_indices,
            extra_topk_length=extra_lens,
        )
        cache = paged_view(main, 128, 528)
        decoded = flashmla_mega_decode(
            packed, cache, indices, positions, table, **kwargs
        )
        kv = torch.cat(
            (
                dequantize_k_cache_v41(main, 128).flatten(0, 1),
                extra_dequant.flatten(0, 1),
            )
        ).unsqueeze(1)
        main_idx = indices.masked_fill(
            torch.arange(128, device="cuda")[None] >= lens[:, None], -1
        )
        extra_idx = (extra_indices + 128).masked_fill(
            torch.arange(64, device="cuda")[None] >= extra_lens[:, None], -1
        )
        combined = torch.cat((main_idx, extra_idx), dim=-1).unsqueeze(1)
        prefilled = flashmla_mega_prefill(
            packed,
            kv,
            combined,
            positions,
            table,
            softmax_scale=512**-0.5,
            attn_sink=sink,
        )
        torch.testing.assert_close(
            unpack_output(*decoded), unpack_output(*prefilled), rtol=0.05, atol=0.01
        )

        groups = heads // 8
        weight = torch.randn(groups, 64, 4096, device="cuda", dtype=torch.bfloat16)
        weight_fp8, weight_scale = prepare_mega_wo_a(weight)
        actual = project_mega_attention(
            decoded.data, decoded.scale, weight_fp8, weight_scale
        )
        expected = torch.einsum(
            "tgd,grd->tgr", unpack_output(*decoded)[:, :groups], weight.float()
        )
        error = (
            actual.float() - expected
        ).square().mean().sqrt() / expected.square().mean().sqrt()
        self.assertLess(error.item(), 0.04)

        def run():
            out = flashmla_mega_decode(
                pack_mega_query(q, 64), cache, indices, positions, table, **kwargs
            )
            return project_mega_attention(out.data, out.scale, weight_fp8, weight_scale)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        q.mul_(0.5)
        positions.add_(13)
        graph.replay()
        torch.testing.assert_close(captured, run(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
