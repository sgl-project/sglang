"""Derived policy boundaries for Unlimited OCR v2 DASH attention."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.attention.unlimited_ocr_dash.attend import (  # noqa: E402
    _DEC_PARTIALS,
    _decode_partials,
)
from sglang.kernels.ops.attention.unlimited_ocr_dash.route import (  # noqa: E402
    _want_tiled,
)
from sglang.srt.layers.attention.unlimited_dash_ops.dash_router import (  # noqa: E402
    router_from_sparse_config,
)
from sglang.srt.layers.attention.unlimited_dash_ops.dash_stage0 import (  # noqa: E402
    resolve_stage0,
)
from sglang.srt.layers.attention.unlimited_ocr_v2_backend import (  # noqa: E402
    UnlimitedOCRV2AttnBackend,
    _uses_static_width,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUnlimitedOCRV2Policy(CustomTestCase):
    def test_checkpoint_router_values_override_fallbacks(self):
        fallback = router_from_sparse_config({})
        configured = router_from_sparse_config(
            {
                "dash_infer_topm": 32,
                "dash_stage0": "head_cls",
            }
        )

        self.assertEqual(fallback.topm, 16)
        self.assertEqual(resolve_stage0({}), "indexer")
        self.assertEqual(configured.topm, 32)
        self.assertEqual(resolve_stage0({"dash_stage0": "head_cls"}), "head_cls")

    def test_tiled_router_starts_at_register_width_boundary(self):
        self.assertFalse(_want_tiled(4096))
        self.assertTrue(_want_tiled(8192))

    def test_decode_partials_separate_graph_and_eager_storage(self):
        _DEC_PARTIALS.clear()
        self.addCleanup(_DEC_PARTIALS.clear)
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            graph = _decode_partials(
                2, 4, 2, 8, torch.float32, "cpu", alloc_chunks=4, alloc_rows=4
            )
            graph_ptrs = tuple(t.data_ptr() for t in graph)
            eager = _decode_partials(2, 8, 2, 8, torch.float32, "cpu")

        self.assertTrue(set(graph_ptrs).isdisjoint(t.data_ptr() for t in eager))
        graph_again = _decode_partials(
            2, 4, 2, 8, torch.float32, "cpu", alloc_chunks=4, alloc_rows=4
        )
        self.assertEqual(graph_ptrs, tuple(t.data_ptr() for t in graph_again))

    def test_static_width_is_reserved_for_graph_backed_paths(self):
        cases = (
            (False, False, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        )
        for use_decode_twin, use_prefill_twin, expected in cases:
            with self.subTest(
                use_decode_twin=use_decode_twin,
                use_prefill_twin=use_prefill_twin,
            ):
                self.assertEqual(
                    _uses_static_width(
                        use_decode_twin=use_decode_twin,
                        use_prefill_twin=use_prefill_twin,
                    ),
                    expected,
                )

    def test_eager_growth_preserves_graph_owned_routing_buffers(self):
        backend = object.__new__(UnlimitedOCRV2AttnBackend)
        backend.buf_width = 16
        backend.cg_max_tokens = 4
        backend._pcg_max_tokens = 6
        backend._idx_buf = torch.full((4, 16), -1, dtype=torch.int32)
        backend._bias_buf = torch.zeros((4, 16), dtype=torch.float32)
        backend._p_idx_buf = torch.full((6, 16), -1, dtype=torch.int32)
        backend._p_bias_buf = torch.zeros((6, 16), dtype=torch.float32)
        backend._eager_idx_buf = None
        backend._eager_bias_buf = None
        backend._score_buf = torch.empty((4, 32), dtype=torch.float32)
        backend._score_buf_win = torch.empty((4, 128), dtype=torch.float32)
        backend._allow_mask = torch.empty((4, 32), dtype=torch.int8)
        backend._eager_score_buf = None
        backend._eager_score_buf_win = None
        backend._eager_allow_mask = None
        backend.max_block_static = 32
        backend._cg_ready = True
        backend.device = "cpu"

        graph_objects = (
            backend._idx_buf,
            backend._bias_buf,
            backend._p_idx_buf,
            backend._p_bias_buf,
            backend._score_buf,
            backend._score_buf_win,
            backend._allow_mask,
        )
        graph_ptrs = tuple(t.data_ptr() for t in graph_objects)

        decode = SimpleNamespace(
            num_tokens=4, use_decode_twin=True, use_prefill_twin=False
        )
        prefill = SimpleNamespace(
            num_tokens=6, use_decode_twin=False, use_prefill_twin=True
        )
        eager = SimpleNamespace(
            num_tokens=12, use_decode_twin=False, use_prefill_twin=False
        )

        decode_idx, decode_bias = backend._routing_buffers(decode, "cpu")
        prefill_idx, prefill_bias = backend._routing_buffers(prefill, "cpu")
        self.assertEqual(decode_idx.data_ptr(), graph_ptrs[0])
        self.assertEqual(decode_bias.data_ptr(), graph_ptrs[1])
        self.assertEqual(prefill_idx.data_ptr(), graph_ptrs[2])
        self.assertEqual(prefill_bias.data_ptr(), graph_ptrs[3])

        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            eager_idx, eager_bias = backend._routing_buffers(eager, "cpu")
            eager_score = backend._scores(eager, 32, "cpu")
            eager_score_win = backend._scores_win(eager, "cpu")
            eager_mask = backend._allow_buf(
                SimpleNamespace(
                    num_reqs=12,
                    max_block=32,
                    use_decode_twin=False,
                )
            )
        eager_objects = (backend._eager_idx_buf, backend._eager_bias_buf)
        eager_ptrs = (eager_idx.data_ptr(), eager_bias.data_ptr())
        self.assertGreaterEqual(backend._eager_idx_buf.shape[0], eager.num_tokens)
        self.assertEqual(eager_score.shape, (12, 32))
        self.assertEqual(eager_score_win.shape, (12, 128))
        self.assertEqual(eager_mask.shape, (12, 32))
        self.assertTrue(set(eager_ptrs).isdisjoint(graph_ptrs))
        self.assertIsNot(backend._eager_score_buf, backend._score_buf)
        self.assertIsNot(backend._eager_score_buf_win, backend._score_buf_win)
        self.assertIsNot(backend._eager_allow_mask, backend._allow_mask)
        self.assertEqual(
            tuple(id(t) for t in graph_objects),
            tuple(
                id(t)
                for t in (
                    backend._idx_buf,
                    backend._bias_buf,
                    backend._p_idx_buf,
                    backend._p_bias_buf,
                    backend._score_buf,
                    backend._score_buf_win,
                    backend._allow_mask,
                )
            ),
        )
        self.assertEqual(
            graph_ptrs,
            tuple(
                t.data_ptr()
                for t in (
                    backend._idx_buf,
                    backend._bias_buf,
                    backend._p_idx_buf,
                    backend._p_bias_buf,
                    backend._score_buf,
                    backend._score_buf_win,
                    backend._allow_mask,
                )
            ),
        )

        smaller = SimpleNamespace(
            num_tokens=3, use_decode_twin=False, use_prefill_twin=False
        )
        backend._routing_buffers(smaller, "cpu")
        self.assertIs(backend._eager_idx_buf, eager_objects[0])
        self.assertIs(backend._eager_bias_buf, eager_objects[1])
        self.assertEqual(
            eager_ptrs,
            (
                backend._eager_idx_buf.data_ptr(),
                backend._eager_bias_buf.data_ptr(),
            ),
        )

        for md in (
            SimpleNamespace(num_tokens=5, use_decode_twin=True, use_prefill_twin=False),
            SimpleNamespace(num_tokens=7, use_decode_twin=False, use_prefill_twin=True),
        ):
            with self.subTest(md=md):
                with self.assertRaisesRegex(RuntimeError, "routing capacity"):
                    backend._routing_buffers(md, "cpu")


class TestUnlimitedOCRV2FusedDecodePaths(CustomTestCase):
    """Which step shapes take the fused kernels, and which fall back.

    The kernels themselves are CUDA-only, so these pin the policy: a shape that
    cannot be served by the fixed-address buffers must reach the PyTorch branch,
    because a captured graph would otherwise bake a moving pointer.
    """

    def _cos_sin_backend(self, buf_rows):
        backend = object.__new__(UnlimitedOCRV2AttnBackend)
        backend.model_dtype = torch.float32
        backend._cos_sin_cache = {}
        backend._p_cos_sin_buf = {}
        rope = SimpleNamespace(
            signature=("main",),
            cos_sin=lambda tok, page, dtype: (tok.float(), page.float()),
        )
        backend.rope_main = rope
        backend.rope_index = rope
        backend._cos_sin_buf = {
            "main": (
                torch.zeros(buf_rows, 4),
                torch.zeros(buf_rows, 4),
            ),
            "index": (
                torch.zeros(buf_rows, 4),
                torch.zeros(buf_rows, 4),
            ),
        }
        return backend

    def test_graph_metadata_inputs_skip_eager_decode_length_derivations(self):
        backend = object.__new__(UnlimitedOCRV2AttnBackend)
        backend._cg_ready = True
        backend._pcg_ready = False
        backend.cg_max_tokens = 4
        backend.cg_max_bs = 4
        backend._in_prefill_cg = False
        backend._seq_lens_buf = torch.zeros(4, dtype=torch.int32)
        backend._ctx_lens_buf = torch.zeros(4, dtype=torch.int32)
        backend._bt_buf = torch.zeros(4, 8, dtype=torch.int32)
        backend._qpage_buf = torch.zeros(4, dtype=torch.int32)
        backend._coord_tok_buf = torch.zeros(4, dtype=torch.int64)
        backend._coord_page_buf = torch.zeros(4, dtype=torch.int64)
        backend._allow_mask = torch.zeros(4, 8, dtype=torch.int8)
        backend.max_block_static = 8
        backend.max_context_len = 256
        backend.block_size = 32
        backend.image_bucket_constraint = True
        backend._cos_sin_cache = {}
        backend.forward_metadata = None
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_idle=lambda: False,
                is_decode_or_idle=lambda: True,
            ),
            input_ids=torch.arange(4),
            seq_lens=torch.tensor([8, 7, 6, 1], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([8, 7, 6, 1], dtype=torch.int64),
            out_cache_loc=torch.arange(4, dtype=torch.int64),
        )

        with (
            patch.object(backend, "_index_pages", return_value=8),
            patch.object(backend, "_refresh_cos_sin"),
            patch.object(backend, "_log_first_step"),
            patch.object(backend, "_to_static_i32") as cast,
        ):
            backend._build_metadata(forward_batch, graph_path=True, in_capture=False)

        cast.assert_not_called()
        self.assertEqual(
            backend.forward_metadata.seq_lens.data_ptr(),
            forward_batch.seq_lens.data_ptr(),
        )
        self.assertIsNone(backend.forward_metadata.context_lens)
        self.assertEqual(backend.forward_metadata.max_seq_len, 256)

    def test_cos_sin_fuses_into_the_buffers_when_they_fit(self):
        backend = self._cos_sin_backend(buf_rows=8)
        # An extend step: decode steps refresh cos/sin inside the graph.
        md = SimpleNamespace(
            coords=(torch.arange(4), torch.arange(4)),
            num_tokens=4,
            is_decode=False,
            use_prefill_twin=False,
        )

        with patch(
            "sglang.srt.layers.attention.unlimited_ocr_v2_backend.cos_sin_2d_fused"
        ) as fused:
            fused.return_value = (torch.zeros(4, 4), torch.zeros(4, 4))
            backend._refresh_cos_sin(md)

        self.assertEqual(fused.call_count, 2, "one fused launch per rope branch")
        for which, call in zip(("main", "index"), fused.call_args_list):
            buf_cos, buf_sin = backend._cos_sin_buf[which]
            self.assertIs(call.kwargs["out_cos"], buf_cos)
            self.assertIs(call.kwargs["out_sin"], buf_sin)

    def test_cos_sin_falls_back_when_the_batch_exceeds_the_buffer(self):
        backend = self._cos_sin_backend(buf_rows=2)
        # An extend step: decode steps refresh cos/sin inside the graph.
        md = SimpleNamespace(
            coords=(torch.arange(4), torch.arange(4)),
            num_tokens=4,
            is_decode=False,
            use_prefill_twin=False,
        )

        with patch(
            "sglang.srt.layers.attention.unlimited_ocr_v2_backend.cos_sin_2d_fused"
        ) as fused:
            backend._refresh_cos_sin(md)

        fused.assert_not_called()

    def _allow_backend(self, buf):
        backend = object.__new__(UnlimitedOCRV2AttnBackend)
        backend._img_allow_buf = buf
        backend.local_blocks = 4
        backend.block_size = 32
        backend.device = "cpu"
        return backend

    def test_local_union_fuses_for_one_token_decode(self):
        buf = torch.zeros(4, 64, dtype=torch.int8)
        backend = self._allow_backend(buf)
        md = SimpleNamespace(
            decode_query_len=1,
            num_reqs=4,
            max_block=64,
            query_allow=torch.zeros(4, 64, dtype=torch.int8),
            seq_lens=torch.full((4,), 128, dtype=torch.int32),
        )

        with patch(
            "sglang.srt.layers.attention.unlimited_ocr_v2_backend.local_union_fused",
            return_value=buf,
        ) as fused:
            backend._build_image_allow_decode(md)

        fused.assert_called_once()
        self.assertIs(fused.call_args.kwargs["out"], buf)
        self.assertIs(md.image_allow, buf)

    def test_local_union_falls_back_without_a_large_enough_buffer(self):
        for label, buf, num_reqs, max_block in (
            ("no buffer", None, 4, 64),
            ("too few rows", torch.zeros(2, 64, dtype=torch.int8), 4, 64),
            ("too narrow", torch.zeros(4, 32, dtype=torch.int8), 4, 64),
        ):
            with self.subTest(case=label):
                backend = self._allow_backend(buf)
                md = SimpleNamespace(
                    decode_query_len=1,
                    num_reqs=num_reqs,
                    max_block=max_block,
                    query_allow=torch.zeros(4, max_block, dtype=torch.int8),
                    seq_lens=torch.full((4,), 128, dtype=torch.int32),
                )
                with patch(
                    "sglang.srt.layers.attention.unlimited_ocr_v2_backend.local_union_fused"
                ) as fused:
                    backend._build_image_allow_decode(md)
                fused.assert_not_called()
                self.assertEqual(md.image_allow.shape, (4, max_block))

    def test_local_union_falls_back_for_multi_token_decode(self):
        # A multi-token decode lays out num_reqs * decode_query_len rows.
        buf = torch.zeros(4, 64, dtype=torch.int8)
        backend = self._allow_backend(buf)
        md = SimpleNamespace(
            decode_query_len=2,
            num_reqs=2,
            max_block=64,
            query_allow=torch.zeros(4, 64, dtype=torch.int8),
            seq_lens=torch.full((2,), 128, dtype=torch.int32),
        )

        with patch(
            "sglang.srt.layers.attention.unlimited_ocr_v2_backend.local_union_fused"
        ) as fused:
            backend._build_image_allow_decode(md)

        fused.assert_not_called()
        self.assertEqual(md.image_allow.shape, (4, 64))


if __name__ == "__main__":
    unittest.main()
