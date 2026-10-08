"""DSA sparse kernels must accept GLM's zero-RoPE geometry."""

import inspect
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    _infer_dsa_dims,
    dequantize_k_cache,
    dequantize_k_cache_paged,
)
from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase, empty_gpu_cache

# backend-specific: the zero-tail specialization only exists in the HIP TileLang kernels
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestPackedRowInference(CustomTestCase):
    def test_packed_width_maps_to_one_layout(self):
        """A 64-wide RoPE tail adds 128 bytes and a NoPE tile adds 132, so no width is ambiguous."""
        for packed_width, layout in {
            264: (256, 0),
            392: (256, 64),
            528: (512, 0),
            656: (512, 64),
        }.items():
            self.assertEqual(_infer_dsa_dims(packed_width), layout)
        with self.assertRaises(ValueError):
            _infer_dsa_dims(265)

    def test_glm53_dispatch_gate_covers_sparse_attention_envelope(self):
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
            can_use_glm53_triton_sparse_attention,
        )

        base = dict(
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            q_nope_dim=512,
            q_rope_dim=0,
            kv_dim=512,
            d_v=512,
            topk_width=2051,
            dsa_index_topk=2048,
            dsa_index_kpool=4,
            is_gfx95=True,
        )
        for tokens, heads in ((1, 8), (1, 16), (65536, 16), (131072, 8)):
            with self.subTest(tokens=tokens, heads=heads):
                self.assertTrue(
                    can_use_glm53_triton_sparse_attention(
                        **base, num_tokens=tokens, num_heads=heads
                    )
                )
        self.assertTrue(
            can_use_glm53_triton_sparse_attention(**base, num_tokens=None, num_heads=16)
        )

        for override in (
            dict(num_tokens=65537, num_heads=16),
            dict(num_tokens=131073, num_heads=8),
            dict(num_tokens=8192, num_heads=16, topk_width=2048),
            dict(num_tokens=8192, num_heads=16, q_dtype=torch.float32),
            dict(num_tokens=8192, num_heads=16, q_rope_dim=64),
            dict(
                num_tokens=8192,
                num_heads=16,
                topk_width=2112,
                dsa_index_kpool=65,
            ),
        ):
            args = base | override
            self.assertFalse(can_use_glm53_triton_sparse_attention(**args))

    def test_glm53_2051_prune_selects_the_measured_config(self):
        from sglang.kernels.ops.attention.dsa import triton_sparse_mla

        named_args = {
            "topk": 2048 + 4 - 1,
            "q_nope_ptr": torch.empty(1, 16, 512, device="meta"),
            "kv_ptr": torch.empty(1, 1, 512, device="meta"),
        }
        with patch.object(triton_sparse_mla, "_IS_GFX95", True):
            configs = triton_sparse_mla._prune_configs(
                triton_sparse_mla._SPLIT_DIM_CONFIGS,
                named_args,
                USE_FP8_DOT=False,
                H=16,
                D_V=512,
                D_TAIL=0,
            )
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].kwargs["BLOCK_N"], 32)
        self.assertEqual(configs[0].num_warps, 2)
        self.assertEqual(configs[0].num_stages, 3)

    def test_query_strides_are_runtime_arguments(self):
        from sglang.kernels.ops.attention.dsa import (
            triton_sparse_mla,
            triton_sparse_mla_decode,
        )

        for module in (triton_sparse_mla, triton_sparse_mla_decode):
            source = inspect.getsource(module)
            for stride in ("STRIDE_QN_T", "STRIDE_QN_H", "STRIDE_QR_T", "STRIDE_QR_H"):
                self.assertNotIn(f"{stride}: tl.constexpr", source)

    def test_tilelang_remains_an_explicit_tilelang_path(self):
        from sglang.srt.layers.attention.dsa.dsa_backend_kpool import (
            DeepseekSparseAttnBackendKPoolMixin,
        )
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        source = inspect.getsource(DeepseekSparseAttnBackend._forward_tilelang)
        self.assertIn("tilelang_sparse_fwd", source)
        self.assertNotIn("triton_sparse_mla", source)

        topk_indices = torch.empty(1, 2051, device="meta", dtype=torch.int32)
        supported = SimpleNamespace(
            dsa_index_kpool=4, _triton_kpool_tail_supported=True
        )
        DeepseekSparseAttnBackendKPoolMixin._check_kpool_tail_backend(
            supported, topk_indices, "triton", "prefill"
        )
        unsupported = SimpleNamespace(
            dsa_index_kpool=4, _triton_kpool_tail_supported=False
        )
        with self.assertRaises(NotImplementedError):
            DeepseekSparseAttnBackendKPoolMixin._check_kpool_tail_backend(
                unsupported, topk_indices, "triton", "prefill"
            )

    def test_glm53_aiter_sparse_mla_dispatch_bounds(self):
        from sglang.srt.layers.attention.dsa.dsa_backend_kpool import (
            DeepseekSparseAttnBackendKPoolMixin,
        )
        from sglang.srt.layers.attention.dsa_backend import _use_aiter_sparse_mla

        for tokens in (1, 64, 4096):
            self.assertFalse(_use_aiter_sparse_mla("triton", tokens, is_decode=True))
            self.assertFalse(_use_aiter_sparse_mla("triton", tokens, is_decode=False))
        for tokens in (1, 2, 4, 8, 12, 15):
            self.assertFalse(
                _use_aiter_sparse_mla("aiter_sparse_mla", tokens, is_decode=True)
            )
        for tokens in (16, 64, 512, 4096):
            self.assertTrue(
                _use_aiter_sparse_mla("aiter_sparse_mla", tokens, is_decode=True)
            )
        for tokens in (1, 64, 256):
            self.assertTrue(
                _use_aiter_sparse_mla("aiter_sparse_mla", tokens, is_decode=False)
            )
        for tokens in (257, 1024, 65536):
            self.assertFalse(
                _use_aiter_sparse_mla("aiter_sparse_mla", tokens, is_decode=False)
            )

        topk_indices = torch.empty(1, 2051, device="meta", dtype=torch.int32)
        DeepseekSparseAttnBackendKPoolMixin._check_kpool_tail_backend(
            SimpleNamespace(dsa_index_kpool=4, _triton_kpool_tail_supported=True),
            topk_indices,
            "aiter_sparse_mla",
            "decode",
        )
        with self.assertRaises(NotImplementedError):
            DeepseekSparseAttnBackendKPoolMixin._check_kpool_tail_backend(
                SimpleNamespace(dsa_index_kpool=4, _triton_kpool_tail_supported=False),
                topk_indices,
                "aiter_sparse_mla",
                "decode",
            )


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class TestScaledCacheLayouts(CustomTestCase):
    def test_quant_dequant_round_trip_and_paged_gather(self):
        """A 264-byte row must dequantize to 256 columns and gather by page like the 656-byte row."""
        torch.manual_seed(7)
        for dim_nope, dim_rope in ((256, 0), (512, 64)):
            with self.subTest(dim_nope=dim_nope, dim_rope=dim_rope):
                source = torch.randn(
                    8, 1, 1, dim_nope + dim_rope, device="cuda", dtype=torch.bfloat16
                )
                packed = quantize_k_cache(source, dv=dim_nope)
                restored = dequantize_k_cache(packed, dv=dim_nope)
                torch.testing.assert_close(restored, source, atol=0.08, rtol=0.08)

                pages = torch.tensor([7, 1, 1, 4], device="cuda", dtype=torch.int32)
                gathered = dequantize_k_cache_paged(packed, pages)
                expected = restored.view(8, 1, -1)[pages]
                torch.testing.assert_close(gathered, expected, atol=0, rtol=0)


def _torch_sparse_attention(q, kv, indices, scale, d_v):
    rows = indices[:, 0]
    valid = rows >= 0
    selected = kv[rows.clamp_min(0), 0]
    scores = torch.einsum("thd,tkd->thk", q.float(), selected.float()) * scale
    scores.masked_fill_(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(valid[:, None, :], probs, 0)
    return torch.einsum("thk,tkd->thd", probs, selected[..., :d_v].float()).to(
        torch.bfloat16
    )


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "the zero-tail TileLang specialization is compiled for gfx950",
)
class TestTileLangDSAZeroRope(CustomTestCase):
    @staticmethod
    def _bf16_inputs(tokens, heads=16, kv_len=2112, live_topk=2051, all_masked=False):
        torch.manual_seed(7)
        q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(kv_len, 1, 512, device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(2112, device="cuda", dtype=torch.int32)
        indices = indices.remainder(kv_len).view(1, 1, 2112)
        indices = indices.expand(tokens, -1, -1).clone()
        indices[..., live_topk:] = -1
        if live_topk:
            indices[..., : min(64, live_topk)] = 3
        if all_masked:
            indices.fill_(-1)
        return q, kv, indices

    def _assert_matches_torch(self, use_fp8, d_v, d_tail):
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            FP8_DTYPE,
            tilelang_sparse_fwd,
        )

        torch.manual_seed(7)
        tokens, heads, topk = 17, 64, 2112
        dim = d_v + d_tail
        q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(topk, 1, dim, device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(topk, device="cuda", dtype=torch.int32)
        indices = indices.view(1, 1, topk).expand(tokens, -1, -1).clone()
        indices[..., 2051:] = -1  # padded rows must be masked, not gathered from slot 0
        if use_fp8:
            q = q.to(FP8_DTYPE)
            kv = kv.to(FP8_DTYPE)

        scale = 1.0 / math.sqrt(dim)
        expected = _torch_sparse_attention(q, kv, indices, scale, d_v)
        # the HIP combine kernel returns [batch=1, tokens, heads, d_v]
        actual = tilelang_sparse_fwd(q, kv, indices, scale, d_v=d_v).squeeze(0)
        torch.testing.assert_close(
            actual,
            expected,
            atol=0.20 if use_fp8 else 0.04,
            rtol=0.12 if use_fp8 else 0.04,
        )

    def test_bf16_zero_rope_matches_torch(self):
        """Before the fix the BF16 partial kernel emitted zero-extent tail copies and failed to compile."""
        self._assert_matches_torch(use_fp8=False, d_v=256, d_tail=0)

    def test_fp8_zero_rope_matches_torch(self):
        """Before the fix the FP8 partial kernel asserted d_v == 512 and read four NoPE tiles."""
        self._assert_matches_torch(use_fp8=True, d_v=256, d_tail=0)

    def test_tail64_layout_unchanged_by_zero_tail_specialization(self):
        """The 512+64 path was rewritten into has_tail/num_main_tiles branches and must still match."""
        for use_fp8 in (False, True):
            with self.subTest(use_fp8=use_fp8):
                self._assert_matches_torch(use_fp8=use_fp8, d_v=512, d_tail=64)

    def test_glm53_h16_d512_zero_rope_matches_torch(self):
        """Pin the GLM-5.3 TP4 sparse-attention head geometry and padded top-k."""
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            tilelang_sparse_fwd,
        )

        q, kv, indices = self._bf16_inputs(tokens=17)
        scale = 1.0 / math.sqrt(256)
        expected = _torch_sparse_attention(q, kv, indices, scale, d_v=512)
        actual = tilelang_sparse_fwd(q, kv, indices, scale, d_v=512).squeeze(0)
        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)
        repeated = tilelang_sparse_fwd(q, kv, indices, scale, d_v=512).squeeze(0)
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)

    def test_glm53_all_masked_rows_are_finite_zeros(self):
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            tilelang_sparse_fwd,
        )

        q, kv, indices = self._bf16_inputs(tokens=1, all_masked=True)
        actual = tilelang_sparse_fwd(q, kv, indices, 1.0 / math.sqrt(256), d_v=512)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_glm53_zero_rope_cuda_graph_replay(self):
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            tilelang_sparse_fwd,
        )

        q, kv, indices = self._bf16_inputs(tokens=17)
        scale = 1.0 / math.sqrt(256)
        eager = tilelang_sparse_fwd(q, kv, indices, scale, d_v=512)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = tilelang_sparse_fwd(q, kv, indices, scale, d_v=512)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, eager, atol=0, rtol=0)

    def test_glm53_production_grids_are_finite(self):
        """M=8192/16384 must retain the traced one-group partial dispatch."""
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            tilelang_sparse_fwd,
        )

        kv = torch.zeros(2112, 1, 512, device="cuda", dtype=torch.bfloat16)
        base_indices = torch.arange(2112, device="cuda", dtype=torch.int32)
        base_indices[2051:] = -1
        for tokens in (8192, 16384):
            with self.subTest(tokens=tokens):
                q = torch.zeros(tokens, 16, 512, device="cuda", dtype=torch.bfloat16)
                indices = (
                    base_indices.view(1, 1, -1).expand(tokens, -1, -1).contiguous()
                )
                actual = tilelang_sparse_fwd(
                    q, kv, indices, 1.0 / math.sqrt(256), d_v=512
                )
                self.assertEqual(actual.shape, (1, tokens, 16, 512))
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))
                del q, indices, actual
                empty_gpu_cache()


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "the GLM-5.3 Triton specialization is enabled only on gfx950",
)
class TestTritonDSAZeroRope(CustomTestCase):
    @staticmethod
    def _run(q, kv, indices):
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
            triton_sparse_mla_fwd,
        )

        return triton_sparse_mla_fwd(
            q,
            q[..., 512:],
            kv,
            indices,
            1.0 / math.sqrt(256),
            d_v=512,
        )

    @staticmethod
    def _run_decode(q, kv, indices, workspace):
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
            triton_sparse_mla_decode_splitk,
        )

        return triton_sparse_mla_decode_splitk(
            q,
            q[..., 512:],
            kv,
            indices,
            1.0 / math.sqrt(256),
            d_v=512,
            workspace=workspace,
        )

    def test_glm53_matches_torch_with_duplicates_and_padding(self):
        for heads in (8, 16):
            for live_topk in (1, 512, 2048, 2051):
                with self.subTest(heads=heads, live_topk=live_topk):
                    q, kv, indices = TestTileLangDSAZeroRope._bf16_inputs(
                        tokens=17, heads=heads, live_topk=live_topk
                    )
                    expected = _torch_sparse_attention(
                        q, kv, indices, 1.0 / math.sqrt(256), d_v=512
                    )
                    actual = self._run(q, kv, indices).squeeze(0)
                    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)
                    repeated = self._run(q, kv, indices).squeeze(0)
                    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)

    def test_glm53_all_masked_rows_are_finite_zeros(self):
        for heads in (8, 16):
            with self.subTest(heads=heads):
                q, kv, indices = TestTileLangDSAZeroRope._bf16_inputs(
                    tokens=1, heads=heads, all_masked=True
                )
                actual = self._run(q, kv, indices)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_glm53_decode_matches_torch_and_is_deterministic(self):
        for heads in (8, 16):
            for live_topk in (0, 1, 512, 2051):
                with self.subTest(heads=heads, live_topk=live_topk):
                    q, kv, indices = TestTileLangDSAZeroRope._bf16_inputs(
                        tokens=5, heads=heads, live_topk=live_topk
                    )
                    expected = _torch_sparse_attention(
                        q, kv, indices, 1.0 / math.sqrt(256), d_v=512
                    )
                    workspace = []
                    actual = self._run_decode(q, kv, indices, workspace).squeeze(0)
                    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)
                    repeated = self._run_decode(q, kv, indices, workspace).squeeze(0)
                    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)

    def test_glm53_unpadded_kpool_width_matches_torch(self):
        for heads in (8, 16):
            with self.subTest(heads=heads):
                q, kv, indices = TestTileLangDSAZeroRope._bf16_inputs(
                    tokens=17, heads=heads
                )
                indices = indices[..., :2051].contiguous()
                expected = _torch_sparse_attention(
                    q, kv, indices, 1.0 / math.sqrt(256), d_v=512
                )
                prefill = self._run(q, kv, indices).squeeze(0)
                decode = self._run_decode(q, kv, indices, []).squeeze(0)
                torch.testing.assert_close(prefill, expected, atol=0.04, rtol=0.04)
                torch.testing.assert_close(decode, expected, atol=0.04, rtol=0.04)

    def test_glm53_decode_workspace_reuse_and_cuda_graph_replay(self):
        kv = torch.zeros(2112, 1, 512, device="cuda", dtype=torch.bfloat16)
        base_indices = torch.arange(2112, device="cuda", dtype=torch.int32)
        base_indices[2051:] = -1
        for tokens, heads in ((64, 16), (512, 8)):
            with self.subTest(tokens=tokens, heads=heads):
                q = torch.zeros(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
                indices = (
                    base_indices.view(1, 1, -1).expand(tokens, -1, -1).contiguous()
                )
                workspace = []
                eager = self._run_decode(q, kv, indices, workspace)
                torch.cuda.synchronize()
                pointers = [(lse.data_ptr(), acc.data_ptr()) for lse, acc in workspace]

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = self._run_decode(q, kv, indices, workspace)
                graph.replay()
                torch.cuda.synchronize()

                torch.testing.assert_close(captured, eager, atol=0, rtol=0)
                self.assertEqual(
                    [(lse.data_ptr(), acc.data_ptr()) for lse, acc in workspace],
                    pointers,
                )

    def test_glm53_cuda_graph_replay(self):
        for heads in (8, 16):
            with self.subTest(heads=heads):
                q, kv, indices = TestTileLangDSAZeroRope._bf16_inputs(
                    tokens=17, heads=heads
                )
                eager = self._run(q, kv, indices)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = self._run(q, kv, indices)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured, eager, atol=0, rtol=0)

    def test_glm53_mixed_rows_and_physical_padding_graph_replay(self):
        live_counts = (0, 1, 3, 4, 511, 512, 2047, 2048, 2049, 2051, 512, 3, 1)
        physical_tokens = len(live_counts) + 4
        for heads in (8, 16):
            with self.subTest(heads=heads):
                q, kv, indices = TestTileLangDSAZeroRope._bf16_inputs(
                    tokens=physical_tokens, heads=heads
                )
                for row, live_topk in enumerate(live_counts):
                    indices[row, :, live_topk:] = -1
                indices[len(live_counts) :].fill_(-1)

                expected = _torch_sparse_attention(
                    q, kv, indices, 1.0 / math.sqrt(256), d_v=512
                )
                eager = self._run(q, kv, indices).squeeze(0)
                torch.testing.assert_close(eager, expected, atol=0.04, rtol=0.04)
                self.assertTrue(
                    torch.equal(
                        eager[len(live_counts) :],
                        torch.zeros_like(eager[len(live_counts) :]),
                    )
                )

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = self._run(q, kv, indices)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured.squeeze(0), eager, atol=0, rtol=0)

    def test_glm53_production_grids_are_finite(self):
        kv = torch.zeros(2112, 1, 512, device="cuda", dtype=torch.bfloat16)
        base_indices = torch.arange(2112, device="cuda", dtype=torch.int32)
        base_indices[2051:] = -1
        for tokens, heads in ((1, 8), (65536, 16), (131072, 8)):
            with self.subTest(tokens=tokens, heads=heads):
                q = torch.zeros(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
                indices = (
                    base_indices.view(1, 1, -1).expand(tokens, -1, -1).contiguous()
                )
                actual = self._run(q, kv, indices)
                self.assertEqual(actual.shape, (1, tokens, heads, 512))
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))
                del q, indices, actual
                empty_gpu_cache()


def _aiter_sparse_mla_available():
    try:
        from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(
    torch.cuda.is_available()
    and is_hip()
    and is_gfx95_supported()
    and _aiter_sparse_mla_available(),
    "AITER sparse_mla_fwd dispatch is gfx950-only",
)
class TestAiterSparseMlaDispatch(CustomTestCase):
    @staticmethod
    def _aiter(q, kv, indices):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        return DeepseekSparseAttnBackend._forward_aiter_sparse_mla(
            None,
            q_nope=q,
            kv_cache=kv,
            v_head_dim=512,
            page_table_1=indices,
            sm_scale=1.0 / math.sqrt(256),
        )

    @staticmethod
    def _inputs(tokens, heads):
        q, kv, indices = TestTileLangDSAZeroRope._bf16_inputs(
            tokens=tokens, heads=heads
        )
        indices = indices[:, 0, :2051].contiguous()
        indices[0, 100:] = -1
        return q, kv, indices

    def test_glm53_decode_and_prefill_match_triton(self):
        for heads in (8, 16):
            for tokens in (16, 64, 256):
                with self.subTest(heads=heads, tokens=tokens):
                    q, kv, indices = self._inputs(tokens, heads)
                    actual = self._aiter(q, kv, indices)
                    decode = TestTritonDSAZeroRope._run_decode(
                        q, kv, indices.unsqueeze(1), []
                    )
                    prefill = TestTritonDSAZeroRope._run(q, kv, indices.unsqueeze(1))
                    self.assertEqual(actual.shape, decode.shape)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, decode, atol=0.04, rtol=0.04)
                    torch.testing.assert_close(actual, prefill, atol=0.04, rtol=0.04)

    def test_glm53_cuda_graph_replay(self):
        for heads, tokens in ((16, 64), (8, 512)):
            with self.subTest(heads=heads, tokens=tokens):
                q, kv, indices = self._inputs(tokens, heads)
                eager = self._aiter(q, kv, indices)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = self._aiter(q, kv, indices)
                q.copy_(torch.randn_like(q))
                graph.replay()
                torch.cuda.synchronize()
                expected = self._aiter(q, kv, indices)
                torch.testing.assert_close(captured, expected, atol=0, rtol=0)
                self.assertFalse(torch.equal(captured, eager))


if __name__ == "__main__":
    unittest.main()
