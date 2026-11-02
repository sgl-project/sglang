"""ROCm BF16 DSA warmup must cover splits used by later query lengths."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip(), "ROCm TileLang required")
class TestTileLangDSAPrecompile(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.attention.dsa import tilelang_kernel

        self.kernels = tilelang_kernel
        self.kernels._precompile_bf16_mla.cache_clear()
        self.addCleanup(self.kernels._precompile_bf16_mla.cache_clear)

    def test_precompiled_splits_cover_selector(self):
        # Include padded top-k values for which ni is not a power of two.
        for block_I, threads, cu, block_per_cu in (
            (64, 256, 256, 2),
            (32, 128, 304, 1),
        ):
            for topk in (64, 128, 192, 320, 1152, 2048, 2112, 2176):
                with (
                    self.subTest(block_I=block_I, topk=topk),
                    patch.object(
                        self.kernels, "sparse_mla_fwd_decode_partial"
                    ) as partial,
                    patch.object(
                        self.kernels, "sparse_mla_fwd_decode_combine"
                    ) as combine,
                ):
                    ni = topk // block_I
                    self.kernels._precompile_bf16_mla(
                        8, 512, 64, topk, 0.0625, block_I, threads
                    )
                    selected = {
                        self.kernels._pick_inner_iter(seq, ni, cu, block_per_cu)
                        for seq in range(1, 1025)
                    }
                    compiled = [c.kwargs["inner_iter"] for c in partial.call_args_list]
                    self.assertCountEqual(compiled, selected)
                    self.assertCountEqual(
                        [c.args[2] for c in combine.call_args_list],
                        [(ni // inner_iter) * block_I for inner_iter in selected],
                    )
                    for call in partial.call_args_list:
                        self.assertEqual(call.args, (8, 512, 64, topk))
                        self.assertEqual(call.kwargs["sm_scale"], 0.0625)
                        self.assertEqual(call.kwargs["block_I"], block_I)
                        self.assertEqual(call.kwargs["threads"], threads)

    def test_cache_distinguishes_model_parameters(self):
        config = dict(
            num_heads=8,
            dim=512,
            tail_dim=64,
            topk=2048,
            sm_scale=0.0625,
            block_I=64,
            threads=256,
        )
        with (
            patch.object(self.kernels, "sparse_mla_fwd_decode_partial") as partial,
            patch.object(self.kernels, "sparse_mla_fwd_decode_combine") as combine,
        ):
            self.kernels._precompile_bf16_mla(**config)
            self.assertEqual(partial.call_count, 6)
            self.assertEqual(combine.call_count, 6)
            self.kernels._precompile_bf16_mla(**config)
            self.assertEqual(partial.call_count, 6)
            self.assertEqual(combine.call_count, 6)

            for parameter, value in (
                ("num_heads", 16),
                ("dim", 256),
                ("tail_dim", 0),
                ("sm_scale", 0.125),
                ("threads", 128),
            ):
                with self.subTest(parameter=parameter):
                    partial.reset_mock()
                    combine.reset_mock()
                    self.kernels._precompile_bf16_mla(**{**config, parameter: value})
                    self.assertEqual(partial.call_count, 6)
                    self.assertEqual(combine.call_count, 6)


@unittest.skipUnless(
    is_hip() and is_gfx95_supported() and torch.cuda.is_available(),
    "gfx950 GPU required",
)
class TestTileLangDSAPrecompileGPU(CustomTestCase):
    def test_query_length_changes_do_not_compile(self):
        from tilelang.jit import JITImpl

        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            _precompile_bf16_mla,
            tilelang_sparse_fwd,
        )

        _precompile_bf16_mla.cache_clear()
        self.addCleanup(_precompile_bf16_mla.cache_clear)
        torch.manual_seed(42)
        for d_v, tail_dim, topk in ((512, 64, 2048), (256, 0, 2112)):
            with self.subTest(d_v=d_v, tail_dim=tail_dim, topk=topk):
                dim, heads = d_v + tail_dim, 8
                scale = dim**-0.5
                q = torch.randn(1, heads, dim, device="cuda", dtype=torch.bfloat16)
                kv = torch.randn(topk, 1, dim, device="cuda", dtype=torch.bfloat16)
                indices = torch.arange(topk, device="cuda", dtype=torch.int32)
                indices[-61:] = -1
                indices = indices.view(1, 1, topk)
                scores = (
                    torch.einsum("thd,kd->thk", q.float(), kv[:, 0].float()) * scale
                )
                scores[..., -61:] = -torch.inf
                expected = torch.einsum(
                    "thk,kd->thd", scores.softmax(-1), kv[:, 0, :d_v].float()
                ).to(torch.bfloat16)

                # One decode-sized warmup must also prepare intermediate prefill splits.
                actual = tilelang_sparse_fwd(q, kv, indices, scale, d_v=d_v)
                torch.testing.assert_close(
                    actual.squeeze(0), expected, atol=0.04, rtol=0.04
                )
                with patch.object(
                    JITImpl,
                    "compile",
                    side_effect=AssertionError("Unexpected TileLang JIT cache miss"),
                ):
                    for tokens in (31, 32, 63, 64, 127, 128, 255, 256, 511, 512):
                        query = q.expand(tokens, -1, -1).contiguous()
                        ids = indices.expand(tokens, -1, -1).contiguous()
                        actual = tilelang_sparse_fwd(query, kv, ids, scale, d_v=d_v)
                        torch.testing.assert_close(
                            actual.squeeze(0),
                            expected.expand(tokens, -1, -1),
                            atol=0.04,
                            rtol=0.04,
                        )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = tilelang_sparse_fwd(query, kv, ids, scale, d_v=d_v)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(captured, actual, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
