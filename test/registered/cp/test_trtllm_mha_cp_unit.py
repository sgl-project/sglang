import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.layers.attention.trtllm_mha_backend as trtllm_mha_backend_module
from sglang.srt.layers.attention.trtllm_mha_backend import (
    TRTLLMHAAttnBackend,
    _compute_per_query_window_geometry,
    _compute_ragged_kv_ranges,
    _gather_ragged_cache_rows,
    _sum_per_query_window_kv_tokens,
)
from sglang.srt.layers.cp.base import CPAttentionBackendKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestComputeRaggedKVRanges(CustomTestCase):
    def test_full_attention_keeps_every_kv_row(self):
        q_lens = torch.tensor([3, 1, 4], dtype=torch.int32)
        kv_lens = torch.tensor([8, 1, 9], dtype=torch.int32)

        starts, effective_lens = _compute_ragged_kv_ranges(
            q_lens, kv_lens, window_left=-1
        )

        torch.testing.assert_close(starts, torch.tensor([0, 0, 0], dtype=torch.int32))
        torch.testing.assert_close(effective_lens, kv_lens)

    def test_sliding_window_keeps_rows_visible_to_uneven_query_chunks(self):
        q_lens = torch.tensor([3, 5, 2, 1], dtype=torch.int32)
        kv_lens = torch.tensor([8, 11, 4, 10], dtype=torch.int32)

        starts, effective_lens = _compute_ragged_kv_ranges(
            q_lens, kv_lens, window_left=2
        )

        # start_i = max(0, kv_len_i - q_len_i - window_left)
        torch.testing.assert_close(
            starts, torch.tensor([3, 4, 0, 7], dtype=torch.int32)
        )
        torch.testing.assert_close(
            effective_lens, torch.tensor([5, 7, 4, 3], dtype=torch.int32)
        )


class TestGatherRaggedCacheRows(CustomTestCase):
    page_size = 3
    head_num = 2
    head_dim = 2
    kv_indices = torch.tensor([5, 0, 3, 2], dtype=torch.int64)

    @classmethod
    def _canonical_nhd_cache(cls):
        return torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)

    @classmethod
    def _expected_rows(cls):
        return torch.tensor(
            [
                [[20, 21], [22, 23]],
                [[0, 1], [2, 3]],
                [[12, 13], [14, 15]],
                [[8, 9], [10, 11]],
            ],
            dtype=torch.float32,
        )

    def _assert_gathered_rows(self, cache):
        actual = _gather_ragged_cache_rows(
            cache,
            self.kv_indices,
            page_size=self.page_size,
            head_num=self.head_num,
            head_dim=self.head_dim,
        )
        torch.testing.assert_close(actual, self._expected_rows(), rtol=0, atol=0)

    def test_gathers_nhd_slot_major_cache(self):
        self._assert_gathered_rows(self._canonical_nhd_cache())

    def test_gathers_hnd_paged_cache(self):
        cache = (
            self._canonical_nhd_cache()
            .reshape(2, self.page_size, self.head_num, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        self._assert_gathered_rows(cache)

    def test_gathers_page_major_paged_cache(self):
        cache = self._canonical_nhd_cache().reshape(
            2, self.page_size, self.head_num, self.head_dim
        )

        self._assert_gathered_rows(cache)

    def test_gathers_production_page_major_cache_layout(self):
        cache = self._canonical_nhd_cache().reshape(
            2, self.page_size, self.head_num, self.head_dim
        )

        actual = _gather_ragged_cache_rows(
            cache,
            self.kv_indices,
            page_size=self.page_size,
            head_num=self.head_num,
            head_dim=self.head_dim,
            cache_layout="page_major_layer_major",
        )

        torch.testing.assert_close(actual, self._expected_rows(), rtol=0, atol=0)


class TestPerQueryWindowGeometry(CustomTestCase):
    def test_expands_each_query_to_its_exact_sliding_window(self):
        q_lens = torch.tensor([2, 3], dtype=torch.int32)
        kv_lens = torch.tensor([2, 7], dtype=torch.int32)
        req_pool_indices = torch.tensor([5, 9], dtype=torch.int32)

        expanded_req_indices, starts, window_lens, cu_seqlens_kv = (
            _compute_per_query_window_geometry(
                q_lens,
                kv_lens,
                req_pool_indices,
                window_left=2,
            )
        )

        torch.testing.assert_close(
            expanded_req_indices,
            torch.tensor([5, 5, 9, 9, 9], dtype=torch.int32),
        )
        torch.testing.assert_close(
            starts, torch.tensor([0, 0, 2, 3, 4], dtype=torch.int32)
        )
        torch.testing.assert_close(
            window_lens, torch.tensor([1, 2, 3, 3, 3], dtype=torch.int32)
        )
        torch.testing.assert_close(
            cu_seqlens_kv,
            torch.tensor([0, 1, 3, 6, 9, 12], dtype=torch.int32),
        )
        self.assertEqual(
            _sum_per_query_window_kv_tokens([2, 3], [2, 7], window_left=2),
            12,
        )

    def test_gather_chain_preserves_exact_windows_after_swa_translation(self):
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.req_to_token = torch.stack(
            [torch.arange(8, dtype=torch.int32), torch.arange(8, 16, dtype=torch.int32)]
        )
        backend.page_size = 1
        k_cache = torch.arange(32 * 192, dtype=torch.float32).view(32, 1, 192)
        v_cache = torch.arange(32 * 128, dtype=torch.float32).view(32, 1, 128)
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=lambda _layer_id: (k_cache, v_cache)
        )
        backend._swa_kv_pool = SimpleNamespace(
            layers_mapping={3: (0, True)},
            translate_loc_from_full_to_swa=lambda indices: indices + 16,
        )
        layer = SimpleNamespace(
            layer_id=3,
            sliding_window_size=2,
            tp_k_head_num=1,
            tp_v_head_num=1,
            qk_head_dim=192,
            v_head_dim=128,
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32)
        )

        class FakeKVIndexKernel:
            def __getitem__(self, _grid):
                def launch(
                    req_to_token,
                    req_indices,
                    lengths,
                    cu_seqlens,
                    starts,
                    output,
                    _stride,
                ):
                    for i in range(req_indices.numel()):
                        out_start = int(cu_seqlens[i])
                        out_end = int(cu_seqlens[i + 1])
                        req = int(req_indices[i])
                        start = int(starts[i])
                        output[out_start:out_end] = req_to_token[
                            req, start : start + int(lengths[i])
                        ]

                return launch

        with patch.object(
            trtllm_mha_backend_module,
            "create_flashinfer_kv_indices_triton",
            FakeKVIndexKernel(),
        ):
            k, v, lengths, cu_seqlens, max_kv_len = (
                backend._gather_per_query_window_ragged_kv(
                    layer=layer,
                    forward_batch=forward_batch,
                    q_lens=torch.tensor([2, 2], dtype=torch.int32),
                    kv_lens=torch.tensor([2, 5], dtype=torch.int32),
                    q_lens_host=[2, 2],
                    kv_lens_host=[2, 5],
                )
            )

        expected_rows = torch.tensor(
            [16, 16, 17, 25, 26, 27, 26, 27, 28], dtype=torch.int64
        )
        torch.testing.assert_close(k, k_cache.index_select(0, expected_rows))
        torch.testing.assert_close(v, v_cache.index_select(0, expected_rows))
        torch.testing.assert_close(
            lengths, torch.tensor([1, 2, 3, 3], dtype=torch.int32)
        )
        torch.testing.assert_close(
            cu_seqlens, torch.tensor([0, 1, 3, 6, 9], dtype=torch.int32)
        )
        self.assertEqual(max_kv_len, 3)


class _ExtendMode:
    def is_target_verify(self):
        return False

    def is_draft_extend_v2(self):
        return False

    def is_context_parallel_extend(self):
        return True


class TestTRTLLMAsymmetricPrefillDispatch(CustomTestCase):
    def _make_backend(self):
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.decode_uses_native_fp4 = False
        backend.data_type = torch.bfloat16
        backend.q_data_type = torch.bfloat16
        backend.is_xqa_impl = False
        backend.page_size = 1
        backend.token_to_kv_pool = Mock()
        backend.forward_metadata = SimpleNamespace(swa_out_cache_loc=None)
        backend._forward_asymmetric_ragged_prefill = Mock(
            return_value=torch.arange(2 * 2 * 128, dtype=torch.bfloat16).view(2, 2, 128)
        )
        return backend

    @staticmethod
    def _make_layer():
        return SimpleNamespace(
            head_dim=192,
            qk_head_dim=192,
            v_head_dim=128,
            tp_q_head_num=2,
            tp_k_head_num=1,
            tp_v_head_num=1,
            layer_id=3,
            is_cross_attention=False,
            sliding_window_size=-1,
            scaling=192**-0.5,
            k_scale=None,
            v_scale=None,
        )

    @staticmethod
    def _make_forward_batch():
        return SimpleNamespace(
            forward_mode=_ExtendMode(),
            out_cache_loc=torch.tensor([7, 8]),
            attn_cp_metadata=object(),
        )

    def test_non_cp_asymmetric_extend_uses_native_ragged_output_width(self):
        backend = self._make_backend()
        layer = self._make_layer()
        forward_batch = self._make_forward_batch()
        q = torch.zeros(2, 2 * 192, dtype=torch.bfloat16)
        k = torch.zeros(2, 1, 192, dtype=torch.bfloat16)
        v = torch.zeros(2, 1, 128, dtype=torch.bfloat16)
        sinks = torch.ones(2, dtype=torch.float32)

        with patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
            return_value=False,
            create=True,
        ):
            output = backend.forward_extend(q, k, v, layer, forward_batch, sinks=sinks)

        backend.token_to_kv_pool.set_kv_buffer.assert_called_once()
        backend._forward_asymmetric_ragged_prefill.assert_called_once()
        ragged_q, ragged_layer, ragged_batch, ragged_sinks = (
            backend._forward_asymmetric_ragged_prefill.call_args.args
        )
        torch.testing.assert_close(ragged_q, q.view(2, 2, 192))
        self.assertIs(ragged_layer, layer)
        self.assertIs(ragged_batch, forward_batch)
        self.assertIs(ragged_sinks, sinks)
        self.assertEqual(output.shape, (2, 2 * 128))

    def test_cp_asymmetric_extend_materializes_post_rope_kv(self):
        backend = self._make_backend()
        layer = self._make_layer()
        forward_batch = self._make_forward_batch()
        q = torch.zeros(2, 2 * 192, dtype=torch.bfloat16)
        k = torch.zeros(2, 1, 192, dtype=torch.bfloat16)
        v = torch.zeros(2, 1, 128, dtype=torch.bfloat16)
        cp_strategy = Mock()

        with (
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
                return_value=True,
                create=True,
            ),
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.get_cp_strategy",
                return_value=cp_strategy,
                create=True,
            ),
        ):
            output = backend.forward_extend(q, k, v, layer, forward_batch)

        cp_strategy.materialize_full_kv.assert_called_once_with(
            forward_batch,
            layer,
            k,
            v,
            swa_loc=None,
        )
        backend.token_to_kv_pool.set_kv_buffer.assert_not_called()
        backend._forward_asymmetric_ragged_prefill.assert_called_once()
        self.assertEqual(output.shape, (2, 2 * 128))

    def test_missing_cp_metadata_falls_back_to_non_cp_cache_write(self):
        backend = self._make_backend()
        layer = self._make_layer()
        forward_batch = self._make_forward_batch()
        forward_batch.attn_cp_metadata = None
        q = torch.zeros(2, 2 * 192, dtype=torch.bfloat16)
        k = torch.zeros(2, 1, 192, dtype=torch.bfloat16)
        v = torch.zeros(2, 1, 128, dtype=torch.bfloat16)

        with (
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.get_cp_strategy"
            ) as get_strategy,
        ):
            output = backend.forward_extend(q, k, v, layer, forward_batch)

        get_strategy.assert_not_called()
        backend.token_to_kv_pool.set_kv_buffer.assert_called_once()
        backend._forward_asymmetric_ragged_prefill.assert_called_once()
        self.assertEqual(output.shape, (2, 2 * 128))

    def test_cp_fp8_bypasses_fused_cache_write_before_kv_gather(self):
        backend = self._make_backend()
        backend.data_type = torch.float8_e4m3fn
        backend._should_use_fused_fp8_path = Mock(return_value=True)
        backend._fused_fp8_qkv_kv_cache = Mock()
        layer = self._make_layer()
        forward_batch = self._make_forward_batch()
        q = torch.zeros(2, 2 * 192, dtype=torch.bfloat16)
        k = torch.zeros(2, 1, 192, dtype=torch.bfloat16)
        v = torch.zeros(2, 1, 128, dtype=torch.bfloat16)
        cp_strategy = Mock()

        with (
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.get_cp_strategy",
                return_value=cp_strategy,
            ),
        ):
            output = backend.forward_extend(q, k, v, layer, forward_batch)

        backend._should_use_fused_fp8_path.assert_called_once()
        backend._fused_fp8_qkv_kv_cache.assert_not_called()
        cp_strategy.materialize_full_kv.assert_called_once()
        self.assertEqual(output.shape, (2, 2 * 128))

    def test_cp_asymmetric_ragged_dispatch_uses_both_host_geometries(self):
        backend = self._make_backend()
        backend.device = torch.device("cpu")
        backend._forward_asymmetric_ragged_prefill = (
            TRTLLMHAAttnBackend._forward_asymmetric_ragged_prefill.__get__(backend)
        )
        backend._run_asymmetric_ragged_attention = Mock(
            side_effect=lambda q_chunk, *_args: q_chunk[..., :128]
        )
        layer = self._make_layer()
        meta = SimpleNamespace(
            total_q_prev_tokens=6,
            actual_seq_q_prev_list=[3, 3],
            actual_seq_q_next_list=[2, 2],
            kv_len_prev_list=[8, 10],
            kv_len_next_list=[14, 18],
            cu_seqlens_q_prev_tensor=torch.tensor([0, 3, 6], dtype=torch.int32),
            cu_seqlens_q_next_tensor=torch.tensor([0, 2, 4], dtype=torch.int32),
            kv_len_prev_tensor=torch.tensor([8, 10], dtype=torch.int32),
            kv_len_next_tensor=torch.tensor([14, 18], dtype=torch.int32),
            max_seqlen_q_prev=3,
            max_seqlen_q_next=2,
        )
        forward_batch = self._make_forward_batch()
        forward_batch.attn_cp_metadata = meta
        q = torch.zeros(10, 2, 192, dtype=torch.bfloat16)
        cp_strategy = Mock()

        def run_attention(q_all, fb, device, attn_fn, attention_backend):
            self.assertIs(fb, forward_batch)
            self.assertEqual(attention_backend, CPAttentionBackendKind.TRTLLM_MHA)
            return torch.cat(
                [
                    attn_fn(
                        q_all[:6],
                        meta.cu_seqlens_q_prev_tensor,
                        meta.kv_len_prev_tensor,
                        meta.max_seqlen_q_prev,
                    ),
                    attn_fn(
                        q_all[6:],
                        meta.cu_seqlens_q_next_tensor,
                        meta.kv_len_next_tensor,
                        meta.max_seqlen_q_next,
                    ),
                ]
            )

        cp_strategy.run_attention.side_effect = run_attention
        with (
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.get_cp_strategy",
                return_value=cp_strategy,
            ),
        ):
            output = backend._forward_asymmetric_ragged_prefill(
                q, layer, forward_batch, None
            )

        self.assertEqual(output.shape, (10, 2, 128))
        calls = backend._run_asymmetric_ragged_attention.call_args_list
        self.assertEqual(calls[0].args[-2:], ([3, 3], [8, 10]))
        self.assertEqual(calls[1].args[-2:], ([2, 2], [14, 18]))

    def test_cp_symmetric_paged_context_uses_both_zigzag_geometries(self):
        backend = self._make_backend()
        backend.device = torch.device("cpu")
        backend.workspace_buffer = torch.empty(1, dtype=torch.uint8)
        backend.max_context_len = 64
        backend._swa_kv_pool = None
        backend._should_use_fused_fp8_path = Mock(return_value=False)
        backend._reshape_paged_kv_cache = Mock(return_value=(object(), object()))
        backend.token_to_kv_pool.get_kv_buffer.return_value = (
            torch.empty(1),
            torch.empty(1),
        )
        backend.forward_metadata = SimpleNamespace(
            swa_out_cache_loc=None,
            swa_page_table=None,
            page_table=torch.zeros(2, 1, dtype=torch.int32),
        )
        layer = self._make_layer()
        layer.head_dim = 128
        layer.qk_head_dim = 128
        layer.v_head_dim = 128
        q = torch.zeros(10, 2 * 128, dtype=torch.bfloat16)
        k = torch.zeros(10, 1, 128, dtype=torch.bfloat16)
        v = torch.zeros(10, 1, 128, dtype=torch.bfloat16)
        meta = SimpleNamespace(
            total_q_prev_tokens=6,
            cu_seqlens_q_prev_tensor=torch.tensor([0, 3, 6], dtype=torch.int32),
            cu_seqlens_q_next_tensor=torch.tensor([0, 2, 4], dtype=torch.int32),
            kv_len_prev_tensor=torch.tensor([8, 10], dtype=torch.int32),
            kv_len_next_tensor=torch.tensor([14, 18], dtype=torch.int32),
            max_seqlen_q_prev=3,
            max_seqlen_q_next=2,
        )
        forward_batch = self._make_forward_batch()
        forward_batch.attn_cp_metadata = meta
        cp_strategy = Mock()

        def run_attention(q_all, fb, device, attn_fn, attention_backend):
            self.assertIs(fb, forward_batch)
            self.assertEqual(attention_backend, CPAttentionBackendKind.TRTLLM_MHA)
            return torch.cat(
                [
                    attn_fn(
                        q_all[:6],
                        meta.cu_seqlens_q_prev_tensor,
                        meta.kv_len_prev_tensor,
                        meta.max_seqlen_q_prev,
                    ),
                    attn_fn(
                        q_all[6:],
                        meta.cu_seqlens_q_next_tensor,
                        meta.kv_len_next_tensor,
                        meta.max_seqlen_q_next,
                    ),
                ]
            )

        cp_strategy.run_attention.side_effect = run_attention
        kernel = Mock(side_effect=lambda **kwargs: kwargs["query"].clone())
        fake_flashinfer = SimpleNamespace(
            prefill=SimpleNamespace(trtllm_batch_context_with_kv_cache=kernel)
        )
        with (
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.get_cp_strategy",
                return_value=cp_strategy,
            ),
            patch.object(
                trtllm_mha_backend_module,
                "flashinfer",
                fake_flashinfer,
                create=True,
            ),
        ):
            output = backend.forward_extend(q, k, v, layer, forward_batch)

        self.assertEqual(output.shape, (10, 2 * 128))
        self.assertEqual(kernel.call_count, 2)
        torch.testing.assert_close(
            kernel.call_args_list[0].kwargs["cum_seq_lens_kv"],
            torch.tensor([0, 8, 18], dtype=torch.int32),
        )
        torch.testing.assert_close(
            kernel.call_args_list[1].kwargs["cum_seq_lens_kv"],
            torch.tensor([0, 14, 32], dtype=torch.int32),
        )

    def test_asymmetric_decode_has_actionable_fa4_error(self):
        backend = self._make_backend()
        with self.assertRaisesRegex(RuntimeError, "--decode-attention-backend fa4"):
            backend.forward_decode(
                torch.empty(1, 2 * 192),
                torch.empty(1, 1, 192),
                torch.empty(1, 1, 128),
                self._make_layer(),
                self._make_forward_batch(),
            )

    def test_ragged_kernel_receives_fp32_attention_sinks(self):
        backend = self._make_backend()
        backend.workspace_buffer = torch.empty(1, dtype=torch.uint8)
        backend._gather_compact_ragged_kv = Mock(
            return_value=(
                torch.zeros(2, 1, 192, dtype=torch.bfloat16),
                torch.zeros(2, 1, 128, dtype=torch.bfloat16),
                torch.tensor([2], dtype=torch.int32),
                torch.tensor([0, 2], dtype=torch.int32),
                2,
            )
        )
        layer = self._make_layer()
        q = torch.zeros(2, 2, 192, dtype=torch.bfloat16)
        sinks = torch.ones(2, dtype=torch.bfloat16)

        def ragged_kernel(**kwargs):
            return kwargs["out"]

        kernel = Mock(side_effect=ragged_kernel)
        fake_flashinfer = SimpleNamespace(
            prefill=SimpleNamespace(trtllm_ragged_attention_deepseek=kernel)
        )
        with patch.object(
            trtllm_mha_backend_module,
            "flashinfer",
            fake_flashinfer,
            create=True,
        ):
            backend._run_asymmetric_ragged_attention(
                q,
                layer,
                self._make_forward_batch(),
                sinks,
                torch.tensor([0, 2], dtype=torch.int32),
                torch.tensor([2], dtype=torch.int32),
                2,
                [2],
                [2],
            )

        self.assertEqual(
            kernel.call_args.kwargs["attention_sinks"].dtype, torch.float32
        )
        self.assertEqual(kernel.call_args.kwargs["backend"], "trtllm-gen")

    def test_long_swa_uses_exact_per_query_windows_with_finite_mask(self):
        backend = self._make_backend()
        backend.workspace_buffer = torch.empty(1, dtype=torch.uint8)
        backend._gather_compact_ragged_kv = Mock()
        backend._gather_per_query_window_ragged_kv = Mock(
            return_value=(
                torch.zeros(6, 1, 192, dtype=torch.bfloat16),
                torch.zeros(6, 1, 128, dtype=torch.bfloat16),
                torch.tensor([3, 3], dtype=torch.int32),
                torch.tensor([0, 3, 6], dtype=torch.int32),
                3,
            )
        )
        layer = self._make_layer()
        layer.sliding_window_size = 2
        q = torch.zeros(2, 2, 192, dtype=torch.bfloat16)

        def ragged_kernel(**kwargs):
            return kwargs["out"]

        kernel = Mock(side_effect=ragged_kernel)
        fake_flashinfer = SimpleNamespace(
            prefill=SimpleNamespace(trtllm_ragged_attention_deepseek=kernel)
        )
        with patch.object(
            trtllm_mha_backend_module,
            "flashinfer",
            fake_flashinfer,
            create=True,
        ):
            backend._run_asymmetric_ragged_attention(
                q,
                layer,
                self._make_forward_batch(),
                None,
                torch.tensor([0, 2], dtype=torch.int32),
                torch.tensor([4], dtype=torch.int32),
                2,
                [2],
                [4],
            )

        backend._gather_compact_ragged_kv.assert_not_called()
        backend._gather_per_query_window_ragged_kv.assert_called_once()
        kwargs = kernel.call_args.kwargs
        self.assertEqual(kwargs["batch_size"], 2)
        self.assertEqual(kwargs["max_q_len"], 1)
        self.assertEqual(kwargs["max_kv_len"], 3)
        self.assertEqual(kwargs["window_left"], 2)
        self.assertEqual(kwargs["backend"], "trtllm-gen")
        torch.testing.assert_close(
            kwargs["cum_seq_lens_q"],
            torch.tensor([0, 1, 2], dtype=torch.int32),
        )

    def test_long_swa_chunks_large_expanded_ragged_batches(self):
        backend = self._make_backend()
        backend.workspace_buffer = torch.empty(1, dtype=torch.uint8)
        effective_lens = torch.tensor([1, 2] + [3] * 125 + [1, 2, 3], dtype=torch.int32)
        cu_seqlens_kv = torch.nn.functional.pad(
            torch.cumsum(effective_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        k = (
            torch.arange(384, dtype=torch.bfloat16)
            .view(384, 1, 1)
            .expand(384, 1, 192)
            .clone()
        )
        v = (
            torch.arange(384, dtype=torch.bfloat16)
            .view(384, 1, 1)
            .expand(384, 1, 128)
            .clone()
        )
        backend._gather_per_query_window_ragged_kv = Mock(
            return_value=(
                k,
                v,
                effective_lens,
                cu_seqlens_kv,
                3,
            )
        )
        layer = self._make_layer()
        layer.sliding_window_size = 2
        # CP alignment pads the model input to a multiple of 2 * cp_size even
        # when a cached-prefix batch is too short to activate CP-v2.
        q = torch.zeros(136, 2, 192, dtype=torch.bfloat16)

        kernel = Mock()

        def ragged_kernel(**kwargs):
            kwargs["out"].fill_(kernel.call_count)
            return kwargs["out"]

        kernel.side_effect = ragged_kernel
        fake_flashinfer = SimpleNamespace(
            prefill=SimpleNamespace(trtllm_ragged_attention_deepseek=kernel)
        )
        with patch.object(
            trtllm_mha_backend_module,
            "flashinfer",
            fake_flashinfer,
            create=True,
        ):
            output = backend._run_asymmetric_ragged_attention(
                q,
                layer,
                self._make_forward_batch(),
                None,
                torch.tensor([0, 127, 130], dtype=torch.int32),
                torch.tensor([127, 3], dtype=torch.int32),
                127,
                [127, 3],
                [127, 3],
            )

        self.assertEqual(kernel.call_count, 2)
        first, second = kernel.call_args_list
        self.assertEqual(first.kwargs["batch_size"], 128)
        self.assertEqual(second.kwargs["batch_size"], 2)
        self.assertEqual(first.kwargs["query"].shape[0], 128)
        self.assertEqual(second.kwargs["query"].shape[0], 2)
        self.assertEqual(first.kwargs["key"].shape[0], 379)
        self.assertEqual(second.kwargs["key"].shape[0], 5)
        torch.testing.assert_close(
            second.kwargs["key"][:, 0, 0],
            torch.arange(379, 384, dtype=torch.bfloat16),
        )
        torch.testing.assert_close(
            second.kwargs["value"][:, 0, 0],
            torch.arange(379, 384, dtype=torch.bfloat16),
        )
        torch.testing.assert_close(
            second.kwargs["seq_lens"],
            torch.tensor([2, 3], dtype=torch.int32),
        )
        torch.testing.assert_close(
            second.kwargs["cum_seq_lens_q"],
            torch.tensor([0, 1, 2], dtype=torch.int32),
        )
        torch.testing.assert_close(
            second.kwargs["cum_seq_lens_kv"],
            torch.tensor([0, 2, 5], dtype=torch.int32),
        )
        torch.testing.assert_close(output[:128], torch.ones_like(output[:128]))
        torch.testing.assert_close(output[128:130], torch.full_like(output[128:130], 2))
        torch.testing.assert_close(output[130:], torch.zeros_like(output[130:]))


if __name__ == "__main__":
    unittest.main()
