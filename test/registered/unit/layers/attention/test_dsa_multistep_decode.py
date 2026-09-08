import unittest
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    DeepseekSparseAttnBackendMTPPrecomputeMixin,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
    _restore_dsa_decode_dp_padding,
    _trim_dsa_decode_dp_padding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSAMultiStepDecode(unittest.TestCase):
    def test_eager_flashmla_scheduler_stays_on_live_query_axis(self):
        flashmla_metadata = object()
        backend = SimpleNamespace(
            speculative_num_draft_tokens=4,
            use_mha=False,
            dsa_decode_impl="flashmla_kv",
            dsa_index_topk=16,
            real_page_size=1,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(64, dtype=torch.int32).view(4, 16)
            ),
            set_dsa_prefill_impl=MagicMock(),
            get_topk_transform_method=MagicMock(),
            _draft_decode_seq_len_offset=MagicMock(return_value=1),
            _cal_indexer_k_start_end=MagicMock(return_value=(None, None)),
            get_device_int32_arange=lambda size: torch.arange(size, dtype=torch.int32),
            _compute_flashmla_metadata=MagicMock(return_value=flashmla_metadata),
            _transform_table_1_to_real=MagicMock(
                return_value=torch.zeros((1, 16), dtype=torch.int32)
            ),
            _build_topk_v2_plan=MagicMock(return_value=None),
        )
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens=torch.tensor([7]),
            seq_lens_cpu=torch.tensor([7]),
            req_pool_indices=torch.tensor([0]),
            spec_info=object(),
            forward_mode=ForwardMode.DECODE,
            seq_lens_sum=7,
        )
        physical_dsa_lengths = torch.tensor([8, 0, 0, 0], dtype=torch.int32)

        with (
            patch(
                "sglang.srt.layers.attention.dsa_backend.pad_dsa_cache_seqlens",
                return_value=physical_dsa_lengths,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.is_cuda",
                return_value=False,
            ),
        ):
            DeepseekSparseAttnBackend.init_forward_metadata(backend, forward_batch)

        flashmla_lengths = backend._compute_flashmla_metadata.call_args.kwargs[
            "cache_seqlens"
        ]
        self.assertEqual(flashmla_lengths.tolist(), [8])
        self.assertIs(backend.forward_metadata.flashmla_metadata, flashmla_metadata)
        self.assertTrue(
            torch.equal(
                backend.forward_metadata.dsa_cache_seqlens_int32,
                physical_dsa_lengths,
            )
        )

    def test_flashmla_decode_trims_physical_lengths_to_live_query_rows(self):
        captured = {}
        flashmla = ModuleType("sgl_kernel.flash_mla")

        def fake_flash_mla_with_kvcache(**kwargs):
            captured.update(kwargs)
            return torch.ones((1, 1, 2, 2)), None

        flashmla.flash_mla_with_kvcache = fake_flash_mla_with_kvcache
        sgl_kernel = ModuleType("sgl_kernel")
        sgl_kernel.flash_mla = flashmla
        backend = SimpleNamespace(
            flashmla_kv_num_q_heads=2,
            real_page_size=64,
            kv_cache_dim=3,
            dsa_kv_cache_store_fp8=True,
            dsa_index_topk=2,
        )
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor([8, 0, 0, 0], dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=torch.empty((1,), dtype=torch.int32),
                num_splits=torch.empty((2,), dtype=torch.int32),
            ),
        )
        layer = SimpleNamespace(tp_q_head_num=2, head_dim=3)

        with patch.dict(
            "sys.modules",
            {
                "sgl_kernel": sgl_kernel,
                "sgl_kernel.flash_mla": flashmla,
            },
        ):
            output = DeepseekSparseAttnBackend._forward_flashmla_kv(
                backend,
                q_all=torch.empty((1, 2, 3)),
                kv_cache=torch.empty((64, 3)),
                v_head_dim=2,
                sm_scale=1.0,
                layer=layer,
                metadata=metadata,
                page_table_1=torch.zeros((1, 2), dtype=torch.int32),
            )

        self.assertEqual(captured["q"].shape[0], 1)
        self.assertEqual(captured["cache_seqlens"].tolist(), [8])
        self.assertEqual(captured["indices"].shape[0], 1)
        self.assertEqual(captured["num_splits"].shape[0], 2)
        self.assertEqual(output.shape, (1, 1, 2, 2))

    def test_flashmla_decode_rejects_scheduler_row_mismatch(self):
        flashmla = ModuleType("sgl_kernel.flash_mla")
        flashmla.flash_mla_with_kvcache = MagicMock()
        sgl_kernel = ModuleType("sgl_kernel")
        sgl_kernel.flash_mla = flashmla
        backend = SimpleNamespace()
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor([8, 0, 0, 0], dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=torch.empty((1,), dtype=torch.int32),
                num_splits=torch.empty((5,), dtype=torch.int32),
            ),
        )

        with (
            patch.dict(
                "sys.modules",
                {
                    "sgl_kernel": sgl_kernel,
                    "sgl_kernel.flash_mla": flashmla,
                },
            ),
            self.assertRaisesRegex(RuntimeError, "q_tokens=1, num_splits=5"),
        ):
            DeepseekSparseAttnBackend._forward_flashmla_kv(
                backend,
                q_all=torch.empty((1, 2, 3)),
                kv_cache=torch.empty((64, 3)),
                v_head_dim=2,
                sm_scale=1.0,
                layer=SimpleNamespace(tp_q_head_num=2, head_dim=3),
                metadata=metadata,
                page_table_1=torch.zeros((1, 2), dtype=torch.int32),
            )

    def test_flashmla_decode_rejects_missing_live_length_row(self):
        flashmla = ModuleType("sgl_kernel.flash_mla")
        flashmla.flash_mla_with_kvcache = MagicMock()
        sgl_kernel = ModuleType("sgl_kernel")
        sgl_kernel.flash_mla = flashmla
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor([8], dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=torch.empty((1,), dtype=torch.int32),
                num_splits=torch.empty((3,), dtype=torch.int32),
            ),
        )

        with (
            patch.dict(
                "sys.modules",
                {
                    "sgl_kernel": sgl_kernel,
                    "sgl_kernel.flash_mla": flashmla,
                },
            ),
            self.assertRaisesRegex(RuntimeError, "q_tokens=2, length_rows=1"),
        ):
            DeepseekSparseAttnBackend._forward_flashmla_kv(
                SimpleNamespace(),
                q_all=torch.empty((2, 2, 3)),
                kv_cache=torch.empty((64, 3)),
                v_head_dim=2,
                sm_scale=1.0,
                layer=SimpleNamespace(tp_q_head_num=2, head_dim=3),
                metadata=metadata,
                page_table_1=torch.zeros((2, 2), dtype=torch.int32),
            )

    def test_draft_children_advance_visible_kv_length(self):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.speculative_num_steps = 3
        spec_info = object()

        backend.speculative_step_id = 0
        self.assertEqual(
            backend._draft_decode_seq_len_offset(ForwardMode.DECODE, spec_info), 1
        )
        backend.speculative_step_id = 1
        self.assertEqual(
            backend._draft_decode_seq_len_offset(ForwardMode.DECODE, spec_info), 2
        )
        self.assertEqual(
            backend._draft_decode_seq_len_offset(ForwardMode.TARGET_VERIFY, spec_info),
            0,
        )
        self.assertEqual(
            backend._draft_decode_seq_len_offset(ForwardMode.DECODE, None), 0
        )

        backend.speculative_num_steps = 0
        self.assertEqual(
            backend._draft_decode_seq_len_offset(ForwardMode.DECODE, spec_info), 0
        )

    def test_decode_precompute_applies_offset_to_all_length_metadata(self):
        req_to_token = torch.arange(32, dtype=torch.int32).view(2, 16)
        backend = SimpleNamespace(
            decode_cuda_graph_metadata={
                2: SimpleNamespace(page_table_1=torch.empty((2, 8)))
            },
            req_to_token=req_to_token,
            dsa_index_topk=6,
            real_page_size=1,
            dsa_decode_impl="fa3",
            _transform_table_1_to_real=MagicMock(),
        )

        with patch(
            "sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute._is_cuda",
            False,
        ):
            metadata = (
                DeepseekSparseAttnBackendMTPPrecomputeMixin._precompute_decode_mode(
                    backend,
                    bs=2,
                    req_pool_indices=torch.tensor([0, 1]),
                    seq_lens=torch.tensor([3, 5]),
                    seq_lens_cpu=torch.tensor([3, 5]),
                    seq_len_offset=2,
                )
            )

        self.assertEqual(metadata.cache_seqlens.tolist(), [5, 7])
        self.assertEqual(metadata.seqlens_expanded.tolist(), [5, 7])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 5, 12])
        self.assertEqual(metadata.dsa_cache_seqlens.tolist(), [5, 6])
        self.assertEqual(metadata.dsa_cu_seqlens_k.tolist(), [0, 5, 11])
        self.assertTrue(torch.equal(metadata.page_indices, req_to_token[:, :8]))

    def test_decode_precompute_rejects_offset_past_page_table_capacity(self):
        backend = SimpleNamespace(
            decode_cuda_graph_metadata={
                1: SimpleNamespace(page_table_1=torch.empty((1, 8)))
            },
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "max_seq_len=7, offset=2, capacity=8",
        ):
            DeepseekSparseAttnBackendMTPPrecomputeMixin._precompute_decode_mode(
                backend,
                bs=1,
                req_pool_indices=torch.tensor([0]),
                seq_lens=torch.tensor([7]),
                seq_lens_cpu=torch.tensor([7]),
                seq_len_offset=2,
            )

    def test_adjust_decode_metadata_reuses_max_prefix_page_table(self):
        page_indices = torch.arange(16, dtype=torch.int32).view(2, 8)
        previous = SimpleNamespace(
            cache_seqlens=torch.tensor([5, 7], dtype=torch.int32),
            page_indices=page_indices,
            real_page_table=None,
            seqlens_expanded_size=2,
            max_len=8,
            max_seqlen_k=8,
        )
        backend = SimpleNamespace(dsa_index_topk=6, dsa_decode_impl="fa3")

        metadata = DeepseekSparseAttnBackendMTPPrecomputeMixin._adjust_decode_precomputed_metadata(
            backend, previous, seq_len_delta=-1
        )

        self.assertEqual(metadata.cache_seqlens.tolist(), [4, 6])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 4, 10])
        self.assertEqual(metadata.dsa_cache_seqlens.tolist(), [4, 6])
        self.assertEqual(metadata.dsa_cu_seqlens_k.tolist(), [0, 4, 10])
        self.assertIs(metadata.page_indices, page_indices)

    def test_graph_replay_precomputes_each_child_offset(self):
        children = []
        for step_id in (0, 1):
            child = MagicMock()
            child.speculative_step_id = step_id
            children.append(child)
        max_prefix_metadata = object()
        first_metadata = object()
        children[1]._precompute_replay_metadata.return_value = max_prefix_metadata
        children[0]._adjust_decode_precomputed_metadata.return_value = first_metadata
        children[1]._adjust_decode_precomputed_metadata.return_value = (
            max_prefix_metadata
        )

        backend = DeepseekSparseAttnMultiStepBackend.__new__(
            DeepseekSparseAttnMultiStepBackend
        )
        backend.speculative_num_steps = 3
        backend.attn_backends = children
        forward_batch = SimpleNamespace(
            batch_size=2,
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([7, 9]),
            seq_lens_cpu=torch.tensor([7, 9]),
            spec_info=object(),
        )

        backend.init_forward_metadata_out_graph(forward_batch, in_capture=False)

        self.assertEqual(
            children[1]._precompute_replay_metadata.call_args.kwargs["seq_len_offset"],
            2,
        )
        children[0]._precompute_replay_metadata.assert_not_called()
        children[0]._adjust_decode_precomputed_metadata.assert_called_once_with(
            max_prefix_metadata, seq_len_delta=-1
        )
        children[1]._adjust_decode_precomputed_metadata.assert_called_once_with(
            max_prefix_metadata, seq_len_delta=0
        )
        children[
            0
        ].init_forward_metadata_replay_cuda_graph_from_precomputed.assert_called_once_with(
            bs=2, precomputed=first_metadata, forward_mode=ForwardMode.DECODE
        )
        children[
            1
        ].init_forward_metadata_replay_cuda_graph_from_precomputed.assert_called_once_with(
            bs=2,
            precomputed=max_prefix_metadata,
            forward_mode=ForwardMode.DECODE,
        )

    def test_frozen_kv_graph_replay_keeps_committed_target_lengths(self):
        children = []
        for step_id in (0, 1):
            child = MagicMock()
            child.speculative_step_id = step_id
            child._precompute_replay_metadata.return_value = object()
            children.append(child)

        backend = DeepseekSparseAttnMultiStepBackend.__new__(
            DeepseekSparseAttnMultiStepBackend
        )
        backend.speculative_num_steps = 3
        backend.attn_backends = children
        forward_batch = SimpleNamespace(
            batch_size=2,
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([7, 9]),
            seq_lens_cpu=torch.tensor([7, 9]),
            spec_info=None,
        )

        backend.init_forward_metadata_out_graph(forward_batch, in_capture=False)

        self.assertEqual(
            children[1]._precompute_replay_metadata.call_args.kwargs["seq_len_offset"],
            0,
        )
        children[0]._precompute_replay_metadata.assert_not_called()
        children[0]._adjust_decode_precomputed_metadata.assert_not_called()
        children[1]._adjust_decode_precomputed_metadata.assert_not_called()
        for child in children:
            child.init_forward_metadata_replay_cuda_graph_from_precomputed.assert_called_once_with(
                bs=2,
                precomputed=children[1]._precompute_replay_metadata.return_value,
                forward_mode=ForwardMode.DECODE,
            )

    def test_trim_and_restore_eager_dp_padding(self):
        q = torch.arange(24).view(4, 2, 3)
        trimmed, padding_rows = _trim_dsa_decode_dp_padding(q, 2)
        self.assertTrue(torch.equal(trimmed, q[:2]))
        self.assertEqual(padding_rows, 2)

        output = torch.ones((2, 1, 2, 4))
        restored = _restore_dsa_decode_dp_padding(output, padding_rows)
        self.assertEqual(restored.shape, (4, 1, 2, 4))
        self.assertTrue(torch.equal(restored[:2], output))
        self.assertTrue(torch.all(restored[2:] == 0))

    def test_trim_rejects_metadata_larger_than_activations(self):
        with self.assertRaisesRegex(RuntimeError, "metadata=3, activations=2"):
            _trim_dsa_decode_dp_padding(torch.empty((2, 4)), 3)

    def test_trtllm_fp8_keeps_physical_rows_until_attention(self):
        metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([8, 12], dtype=torch.int32),
            page_table_1=torch.zeros((2, 12), dtype=torch.int32),
            max_seq_len_k=12,
        )
        stored = {}
        token_pool = SimpleNamespace(
            set_mla_kv_buffer=lambda _layer, loc, k, k_rope: stored.update(
                loc=loc, k=k, k_rope=k_rope
            ),
            get_key_buffer=lambda _layer_id: torch.zeros((24, 3)),
        )
        backend = SimpleNamespace(
            forward_metadata=metadata,
            kv_cache_dtype=torch.float8_e4m3fn,
            token_to_kv_pool=token_pool,
            real_page_size=1,
            kv_cache_dim=3,
            use_fused_topk=False,
            qk_nope_head_dim=2,
            kv_lora_rank=2,
            qk_rope_head_dim=1,
            workspace_buffer=None,
            dsa_index_topk=2,
            _multi_ctas_kv_counter_buffer=None,
            device="cpu",
            num_q_heads=2,
        )
        backend._pad_topk_indices = MethodType(
            DeepseekSparseAttnBackend._pad_topk_indices, backend
        )
        layer = SimpleNamespace(
            layer_id=0,
            is_cross_attention=False,
            tp_q_head_num=2,
            head_dim=3,
            k_scale_float=None,
            scaling=1.0,
        )
        forward_batch = SimpleNamespace(
            positions=torch.arange(4),
            out_cache_loc=torch.arange(4),
            encoder_out_cache_loc=None,
            attn_cp_metadata=None,
        )
        captured = {}

        def fake_quantize(q, q_rope, k, k_rope, positions, *_args):
            captured["quantize_rows"] = (
                q.shape[0],
                q_rope.shape[0],
                k.shape[0],
                k_rope.shape[0],
                positions.shape[0],
            )
            return torch.empty((4, 2, 3)), k, k_rope

        flashinfer = ModuleType("flashinfer")
        flashinfer_decode = ModuleType("flashinfer.decode")

        def fake_decode(**kwargs):
            captured["attention_rows"] = kwargs["query"].shape[0]
            return torch.ones((2, 1, 2, 2), dtype=torch.bfloat16)

        flashinfer_decode.trtllm_batch_decode_with_kv_cache_mla = fake_decode
        flashinfer.decode = flashinfer_decode

        with (
            patch.dict(
                "sys.modules",
                {"flashinfer": flashinfer, "flashinfer.decode": flashinfer_decode},
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.mla_quantize_and_rope_for_fp8",
                side_effect=fake_quantize,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.transform_index_page_table_decode",
                return_value=torch.zeros((2, 2), dtype=torch.int32),
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.dsa_use_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.grow_multi_ctas_kv_counter_buffer_if_needed",
                return_value=None,
            ),
        ):
            output = DeepseekSparseAttnBackend._forward_trtllm(
                backend,
                q=torch.empty((4, 2, 2)),
                k=torch.empty((4, 1, 2)),
                v=torch.empty((4, 1, 2)),
                layer=layer,
                forward_batch=forward_batch,
                seq_lens=metadata.cache_seqlens_int32,
                q_rope=torch.empty((4, 2, 1)),
                k_rope=torch.empty((4, 1, 1)),
                topk_indices=torch.zeros((4, 2), dtype=torch.int32),
                cos_sin_cache=torch.empty((16, 2)),
            )

        self.assertEqual(captured["quantize_rows"], (4, 4, 4, 4, 4))
        self.assertEqual(stored["loc"].shape[0], 4)
        self.assertEqual(stored["k"].shape[0], 4)
        self.assertEqual(captured["attention_rows"], 2)
        self.assertEqual(output.shape[0], 4)
        self.assertTrue(torch.all(output[2:] == 0))

    def test_hisparse_uses_real_request_rows(self):
        metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([8, 12], dtype=torch.int32),
            page_table_1=torch.zeros((2, 12), dtype=torch.int32),
            dsa_cache_seqlens_int32=torch.tensor([8, 12], dtype=torch.int32),
        )
        coordinator = MagicMock()
        coordinator.swap_in_selected_pages.return_value = torch.zeros(
            (2, 2), dtype=torch.int32
        )
        stored = {}
        token_pool = SimpleNamespace(
            set_mla_kv_buffer=lambda _layer, loc, k, k_rope: stored.update(
                loc=loc, k=k, k_rope=k_rope
            ),
            get_key_buffer=lambda _layer_id: torch.zeros((24, 3)),
        )
        backend = SimpleNamespace(
            forward_metadata=metadata,
            dsa_decode_impl="flashmla_kv",
            token_to_kv_pool=token_pool,
            hisparse_coordinator=coordinator,
            use_fused_topk=False,
            _forward_flashmla_kv=MagicMock(return_value=torch.ones((2, 1, 2, 2))),
        )
        backend._pad_topk_indices = MethodType(
            DeepseekSparseAttnBackend._pad_topk_indices, backend
        )
        layer = SimpleNamespace(
            layer_id=0,
            is_cross_attention=False,
            tp_q_head_num=2,
            head_dim=3,
            v_head_dim=2,
            scaling=1.0,
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([4, 5, 6, 7]),
            seq_lens=torch.tensor([8, 12, 1, 1]),
            out_cache_loc=torch.arange(4),
            encoder_out_cache_loc=None,
        )

        with patch(
            "sglang.srt.layers.attention.dsa_backend.concat_mla_absorb_q_general",
            return_value=torch.empty((2, 2, 3)),
        ):
            output = DeepseekSparseAttnBackend.forward_decode(
                backend,
                q=torch.empty((4, 2, 3)),
                k=torch.empty((4, 1, 2)),
                v=torch.empty((4, 1, 2)),
                layer=layer,
                forward_batch=forward_batch,
                topk_indices=torch.zeros((4, 2), dtype=torch.int32),
            )

        args = coordinator.swap_in_selected_pages.call_args.args
        self.assertTrue(torch.equal(args[0], forward_batch.req_pool_indices[:2]))
        self.assertTrue(torch.equal(args[1], forward_batch.seq_lens[:2]))
        self.assertEqual(args[2].shape[0], 2)
        self.assertEqual(stored["loc"].shape[0], 4)
        self.assertEqual(stored["k"].shape[0], 4)
        self.assertEqual(
            backend._forward_flashmla_kv.call_args.kwargs["q_all"].shape[0], 2
        )
        self.assertEqual(output.shape[0], 4)
        self.assertTrue(torch.all(output[2:] == 0))

    def test_aiter_uses_real_batch_size(self):
        metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([8, 12], dtype=torch.int32),
            page_table_1=torch.zeros((2, 12), dtype=torch.int32),
            dsa_cache_seqlens_int32=torch.tensor([8, 12], dtype=torch.int32),
        )
        backend = SimpleNamespace(
            forward_metadata=metadata,
            dsa_decode_impl="aiter",
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda _layer_id: torch.zeros((24, 3))
            ),
            hisparse_coordinator=None,
            use_fused_topk=False,
            _pad_topk_indices=lambda value, rows: value[:rows],
            _forward_aiter=MagicMock(return_value=torch.ones((2, 2, 2))),
        )
        layer = SimpleNamespace(
            layer_id=0,
            is_cross_attention=False,
            tp_q_head_num=2,
            head_dim=3,
            v_head_dim=2,
        )
        forward_batch = SimpleNamespace(batch_size=4)

        with (
            patch(
                "sglang.srt.layers.attention.dsa_backend.transform_index_page_table_decode",
                return_value=torch.zeros((2, 2), dtype=torch.int32),
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend._is_hip",
                True,
            ),
        ):
            output = DeepseekSparseAttnBackend.forward_decode(
                backend,
                q=torch.empty((4, 2, 3)),
                k=None,
                v=None,
                layer=layer,
                forward_batch=forward_batch,
                topk_indices=torch.zeros((4, 2), dtype=torch.int32),
            )

        self.assertEqual(backend._forward_aiter.call_args.kwargs["bs"], 2)
        self.assertEqual(output.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
