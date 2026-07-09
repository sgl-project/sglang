import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.attention.trtllm_mha_backend as trtllm_mha_backend
from sglang.srt.layers.attention.trtllm_mha_backend import (
    TRTLLMHAAttnBackend,
    TRTLLMHAAttnMultiStepDraftBackend,
    TRTLLMMHAMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_utils import TreeMaskMode, resolve_tree_mask_mode
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _make_backend_factory(decode_backend, draft_extend_backend):
    class FakeDraftBackendFactory:
        def __init__(self, *args, **kwargs):
            pass

        def create_decode_backend(self):
            return decode_backend

        def create_draft_extend_backend(self):
            return draft_extend_backend

    return FakeDraftBackendFactory


class TestTRTLLMMHASpecDecode(unittest.TestCase):
    @staticmethod
    def _model_runner():
        return SimpleNamespace(
            model_config=SimpleNamespace(context_len=1024, hidden_size=128),
            kv_cache_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
            page_size=64,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.empty(1, 1024, dtype=torch.int32)
            ),
            device="cpu",
            server_args=SimpleNamespace(
                speculative_eagle_topk=8,
                speculative_num_draft_tokens=32,
            ),
            token_to_kv_pool=object(),
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: object()),
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            sliding_window_size=128,
        )

    def test_backend_caches_untrimmed_swa_capability(self):
        for capability in (False, True):
            with self.subTest(capability=capability), patch.object(
                trtllm_mha_backend.flashinfer.decode,
                "trtllm_gen_supports_untrimmed_swa_spec_dec_tree",
                capability,
                create=True,
            ), patch(
                "sglang.srt.layers.attention.trtllm_mha_backend."
                "FlashInferAttnBackend.__init__",
                return_value=None,
            ), patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.is_sm90_supported",
                return_value=False,
            ), patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.is_sm120_supported",
                return_value=False,
            ):
                backend = TRTLLMHAAttnBackend(self._model_runner())

            self.assertEqual(backend.supports_untrimmed_swa_spec_dec_tree, capability)

    def test_swa_tree_verify_selects_native_or_fallback_metadata(self):
        full_page_table = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        fallback_page_table = torch.tensor([[2, 3]], dtype=torch.int32)
        full_seq_lens = torch.tensor([6], dtype=torch.int32)
        fallback_seq_lens = torch.tensor([4], dtype=torch.int32)

        for capability in (False, True):
            with self.subTest(capability=capability):
                backend = object.__new__(TRTLLMHAAttnBackend)
                backend.data_type = torch.bfloat16
                backend.is_xqa_impl = False
                backend.q_data_type = torch.bfloat16
                backend.page_size = 2
                backend.workspace_buffer = torch.empty(1)
                backend.max_context_len = 16
                backend.sliding_window_size = 4
                backend.speculative_num_draft_tokens = 2
                backend.supports_untrimmed_swa_spec_dec_tree = capability
                backend.forward_metadata = TRTLLMMHAMetadata(
                    cache_seqlens_int32=full_seq_lens,
                    max_seq_len_q=2,
                    swa_page_table=full_page_table,
                    swa_verify_page_table=fallback_page_table,
                    swa_verify_seqlens=fallback_seq_lens,
                    tree_mask=torch.ones(1, 2, 2, dtype=torch.bool),
                )
                cache = torch.zeros(3, 2, 1, 2)
                backend.token_to_kv_pool = SimpleNamespace(
                    get_kv_buffer=lambda _layer_id: (cache, cache)
                )
                layer = SimpleNamespace(
                    layer_id=0,
                    tp_q_head_num=1,
                    tp_k_head_num=1,
                    tp_v_head_num=1,
                    head_dim=2,
                    sliding_window_size=4,
                )
                forward_batch = SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    out_cache_loc=torch.empty(0, dtype=torch.int64),
                )

                with patch.object(
                    backend, "_get_bmm_scales", return_value=(1.0, 1.0)
                ), patch.object(
                    backend, "_get_layer_page_table", return_value=full_page_table
                ), patch.object(
                    backend, "_layer_is_swa", return_value=True
                ), patch.object(
                    trtllm_mha_backend.flashinfer.decode,
                    "trtllm_batch_decode_with_kv_cache",
                    return_value=torch.zeros(2, 1, 2),
                ) as decode:
                    backend.forward_extend(
                        torch.zeros(2, 2),
                        None,
                        None,
                        layer,
                        forward_batch,
                        save_kv_cache=False,
                    )

                kwargs = decode.call_args.kwargs
                expected_page_table = (
                    full_page_table if capability else fallback_page_table
                )
                expected_seq_lens = full_seq_lens if capability else fallback_seq_lens
                expected_max_seq_len = 16 if capability else 8
                self.assertIs(kwargs["block_tables"], expected_page_table)
                self.assertIs(kwargs["seq_lens"], expected_seq_lens)
                self.assertEqual(kwargs["max_seq_len"], expected_max_seq_len)

    def test_native_swa_tree_verify_skips_fallback_trim(self):
        backend = object.__new__(TRTLLMHAAttnBackend)
        backend._swa_kv_pool = object()
        backend.topk = 8
        backend.sliding_window_size = 4
        backend.speculative_num_draft_tokens = 2
        backend.page_size = 2
        metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=torch.tensor([6], dtype=torch.int32),
            max_seq_len_q=2,
            swa_page_table=torch.tensor([[1, 2, 3]], dtype=torch.int32),
        )

        backend.supports_untrimmed_swa_spec_dec_tree = True
        backend._fill_swa_verify_views(metadata)
        self.assertIsNone(metadata.swa_verify_page_table)
        self.assertIsNone(metadata.swa_verify_seqlens)

        backend.supports_untrimmed_swa_spec_dec_tree = False
        backend._fill_swa_verify_views(metadata)
        self.assertIsNotNone(metadata.swa_verify_page_table)
        self.assertIsNotNone(metadata.swa_verify_seqlens)

    def test_tree_spec_backend_reports_qlen_only_mask_mode(self):
        model_runner = SimpleNamespace(
            model_config=SimpleNamespace(context_len=1024, hidden_size=128),
            kv_cache_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
            page_size=64,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.empty(1, 1024, dtype=torch.int32)
            ),
            device="cpu",
            server_args=SimpleNamespace(
                speculative_eagle_topk=8,
                speculative_num_draft_tokens=32,
            ),
            token_to_kv_pool=object(),
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: object()),
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            sliding_window_size=None,
        )

        with patch(
            "sglang.srt.layers.attention.trtllm_mha_backend."
            "FlashInferAttnBackend.__init__",
            return_value=None,
        ), patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.is_sm90_supported",
            return_value=False,
        ), patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.is_sm120_supported",
            return_value=False,
        ):
            backend = TRTLLMHAAttnBackend(model_runner)

        self.assertEqual(backend.tree_mask_mode, TreeMaskMode.QLEN_ONLY)
        self.assertEqual(resolve_tree_mask_mode(backend), TreeMaskMode.QLEN_ONLY)

    def test_worker_resolves_target_tree_mask_mode_after_backend_init(self):
        worker = object.__new__(EagleDraftWorker)
        target_backend = SimpleNamespace(tree_mask_mode=TreeMaskMode.QLEN_ONLY)
        decode_backend = object()
        draft_extend_backend = object()
        worker.server_args = SimpleNamespace()
        worker.draft_runner = SimpleNamespace(attn_backend=object())
        worker.target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=target_backend)
        )
        worker.topk = 8
        worker.speculative_num_steps = 5

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.DraftBackendFactory",
            _make_backend_factory(decode_backend, draft_extend_backend),
        ):
            worker.init_attention_backend()

        self.assertIs(worker.draft_attn_backend, decode_backend)
        self.assertIs(worker.draft_extend_attn_backend, draft_extend_backend)
        self.assertIs(worker.draft_runner.attn_backend, draft_extend_backend)
        self.assertEqual(worker.tree_mask_mode, TreeMaskMode.QLEN_ONLY)

    def test_topk_draft_cuda_graph_metadata_uses_seq_lens_tensor(self):
        bs = 2
        topk = 4
        backend = object.__new__(TRTLLMHAAttnBackend)
        backend.topk = topk
        backend.speculative_step_id = 1
        backend.decode_cuda_graph_metadata = {
            bs: TRTLLMMHAMetadata(
                cache_seqlens_int32=torch.empty(bs * topk, dtype=torch.int32),
                cu_seqlens_k=torch.zeros(bs * topk + 1, dtype=torch.int32),
            )
        }

        backend._apply_cuda_graph_metadata(
            bs=bs,
            req_pool_indices=torch.arange(bs, dtype=torch.int32),
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            spec_info=object(),
        )

        metadata = backend.decode_cuda_graph_metadata[bs]
        expected_seqlens = torch.tensor([12] * topk + [22] * topk, dtype=torch.int32)
        self.assertEqual(
            metadata.cache_seqlens_int32.tolist(), expected_seqlens.tolist()
        )
        self.assertEqual(
            metadata.cu_seqlens_k[1:].tolist(),
            torch.cumsum(expected_seqlens, dim=0, dtype=torch.int32).tolist(),
        )
        self.assertIs(backend.forward_metadata, metadata)

    def test_multistep_draft_falls_back_to_seq_lens_when_cpu_copy_missing(self):
        backend = object.__new__(TRTLLMHAAttnMultiStepDraftBackend)
        backend.topk = 8
        backend.speculative_num_steps = 3
        backend.draft_branch_page_table_buf = torch.empty(1, 1, dtype=torch.int32)
        inner_fb = object()
        step_backends = [SimpleNamespace(calls=[]) for _ in range(2)]
        for step_backend in step_backends:
            step_backend.init_forward_metadata_out_graph = (
                lambda fb, in_capture=False, step_backend=step_backend: (
                    step_backend.calls.append((fb, in_capture))
                )
            )
        backend.attn_backends = step_backends

        forward_batch = SimpleNamespace(
            batch_size=2,
            encoder_lens=torch.tensor([0, 0], dtype=torch.int32),
            seq_lens=torch.tensor([64, 127], dtype=torch.int32),
            seq_lens_cpu=None,
            spec_info=SimpleNamespace(is_draft_input=lambda: True),
        )

        with patch(
            "sglang.srt.model_executor.forward_batch_info.build_inner_fb_view",
            return_value=inner_fb,
        ) as build_inner_fb_view, patch.object(
            backend, "_update_draft_branch_page_tables"
        ) as update_tables:
            backend.init_forward_metadata_out_graph(forward_batch)

        build_inner_fb_view.assert_called_once_with(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
            encoder_lens=forward_batch.encoder_lens,
        )
        update_tables.assert_called_once()
        args, kwargs = update_tables.call_args
        self.assertIs(args[0], forward_batch)
        self.assertEqual(args[1].tolist(), forward_batch.seq_lens.cpu().tolist())
        self.assertIs(kwargs["out"], backend.draft_branch_page_table_buf)
        for step_backend in step_backends:
            self.assertEqual(step_backend.calls, [(inner_fb, False)])


if __name__ == "__main__":
    unittest.main()
