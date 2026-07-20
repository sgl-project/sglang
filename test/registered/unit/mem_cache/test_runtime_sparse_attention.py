import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.radix_attention import (
    AttentionType,
    _should_use_runtime_sparse_attention,
)
from sglang.srt.mem_cache import kv_cache_configurator as configurator_module
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    FlashAttentionAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    SparseConfig,
    SparseCoordinator,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.utils import enable_fused_set_kv_buffer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashAttentionAdaptor(unittest.TestCase):
    def setUp(self):
        self.adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        self.metadata = SimpleNamespace(
            page_table=torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([7, 6], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 7, 13], dtype=torch.int32),
            max_seq_len_k=7,
            scheduler_metadata=torch.ones(1, dtype=torch.int32),
        )
        self.batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([7, 6]),
        )
        self.req_to_token = torch.tensor(
            [[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]]
        )

    def _adapt(self, selected, lengths, *, layer_id=0, prepared=False):
        return self.adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor(selected, dtype=torch.int32),
            valid_lengths=torch.tensor(lengths, dtype=torch.int32),
            sparse_mask=torch.tensor([True, False]),
            current_metadata=self.metadata,
            forward_batch=self.batch,
            req_to_token=self.req_to_token,
            page_size=4,
            layer_id=layer_id,
            metadata_prepared=prepared,
        )

    def test_mixed_batch_is_in_place_across_layers_and_forwards(self):
        pointers = tuple(
            tensor.data_ptr()
            for tensor in (
                self.metadata.page_table,
                self.metadata.cache_seqlens_int32,
                self.metadata.cu_seqlens_k,
            )
        )
        self.adaptor.save_original_metadata(self.metadata)
        self.assertIs(self._adapt([[0, 1], [0, 1]], [2, 2]), self.metadata)
        self.assertEqual(self.metadata.page_table.tolist(), [[0, 2], [1, 3]])
        self.assertEqual(self.metadata.cache_seqlens_int32.tolist(), [7, 6])
        self.assertEqual(self.metadata.cu_seqlens_k.tolist(), [0, 7, 13])
        self.assertEqual(
            tuple(
                tensor.data_ptr()
                for tensor in (
                    self.metadata.page_table,
                    self.metadata.cache_seqlens_int32,
                    self.metadata.cu_seqlens_k,
                )
            ),
            pointers,
        )

        self._adapt([[1, -1], [0, -1]], [1, 1], layer_id=1)
        self.assertEqual(self.metadata.page_table.tolist(), [[2, 2], [1, 3]])
        self.assertEqual(self.metadata.cache_seqlens_int32.tolist(), [3, 6])
        self.assertEqual(self.metadata.cu_seqlens_k.tolist(), [0, 3, 9])

        self.metadata.page_table.copy_(torch.tensor([[0, 2], [1, 3]]))
        self.metadata.cache_seqlens_int32.copy_(torch.tensor([7, 6]))
        self.metadata.cu_seqlens_k.copy_(torch.tensor([0, 7, 13]))
        self.adaptor.save_original_metadata(self.metadata)
        self._adapt([[0, 1], [0, -1]], [2, 1])
        self.assertEqual(self.metadata.cache_seqlens_int32.tolist(), [7, 6])
        self.assertEqual(self.metadata.cu_seqlens_k.tolist(), [0, 7, 13])

        self.metadata.page_table.copy_(torch.tensor([[8, 9], [6, 7]]))
        self._adapt([[8, 9], [6, 7]], [2, 2], prepared=True)
        self.assertEqual(self.metadata.page_table.tolist(), [[8, 9], [6, 7]])


class TestRuntimeSparseBoundaries(unittest.TestCase):
    @staticmethod
    def _batch():
        forward_mode = SimpleNamespace(
            is_decode=Mock(return_value=True),
            is_extend_or_draft_extend_or_mixed=Mock(return_value=False),
            is_mixed=Mock(return_value=False),
            is_split_prefill=Mock(return_value=False),
        )
        return SimpleNamespace(
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([16]),
            forward_mode=forward_mode,
        )

    @patch(
        "sglang.srt.layers.radix_attention.get_tc_piecewise_forward_context",
        return_value=None,
    )
    def test_dense_fallback_boundaries(self, _mock_piecewise_context):
        batch, key = self._batch(), torch.empty(1)
        decoder = SimpleNamespace(
            is_cross_attention=False, attn_type=AttentionType.DECODER
        )
        self.assertTrue(
            _should_use_runtime_sparse_attention(decoder, batch, key, True, object())
        )
        cases = (
            (decoder, key, True, None),
            (decoder, key, False, object()),
            (SimpleNamespace(is_cross_attention=True), key, True, object()),
        )
        for layer, candidate_key, save_kv, coordinator in cases:
            with self.subTest(layer=layer, save_kv=save_kv):
                self.assertFalse(
                    _should_use_runtime_sparse_attention(
                        layer, batch, candidate_key, save_kv, coordinator
                    )
                )

    def test_kv_write_order_and_half_open_layer_range(self):
        with forward_context(
            ForwardContext(attn_backend=None, runtime_sparse_coordinator=object())
        ):
            self.assertFalse(enable_fused_set_kv_buffer(SimpleNamespace()))

        algorithm = Mock()
        coordinator = SparseCoordinator(
            config=SparseConfig(page_size=16, min_sparse_prompt_len=0),
            algorithm=algorithm,
            backend_adaptor=Mock(),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((2, 32), dtype=torch.int32),
                max_context_len=32,
            ),
            token_to_kv_pool=Mock(),
            start_layer=4,
            end_layer=10,
            device=torch.device("cpu"),
        )
        self.assertEqual(coordinator.states.num_layers, 6)
        algorithm.initialize_representation_pool.assert_called_once()

    def test_quest_uses_the_regular_paged_allocator(self):
        configurator = object.__new__(KVCacheConfigurator)
        configurator.server_args = SimpleNamespace(
            disaggregation_mode="null",
            enable_hisparse=True,
            hisparse_config=json.dumps({"algorithm": "quest"}),
            page_size=16,
            dcp_size=1,
        )
        configurator.is_hybrid_swa = False
        configurator.kv_cache_dtype = torch.bfloat16
        configurator.device = "cpu"
        expected = Mock()
        runtime_config = SimpleNamespace(
            disaggregation_mode="null", enable_hisparse=True, page_size=16
        )
        with (
            patch.object(
                configurator_module, "get_disagg", return_value=runtime_config
            ),
            patch.object(
                configurator_module, "get_memory", return_value=runtime_config
            ),
            patch.object(
                configurator_module, "get_schedule", return_value=runtime_config
            ),
            patch.object(
                configurator_module.current_platform,
                "is_out_of_tree",
                return_value=False,
            ),
            patch.object(configurator_module, "_is_npu", False),
            patch.object(
                configurator_module,
                "PagedTokenToKVPoolAllocator",
                return_value=expected,
            ) as paged,
            patch.object(
                configurator_module, "HiSparseTokenToKVPoolAllocator"
            ) as hisparse,
        ):
            actual = configurator._build_token_to_kv_pool_allocator(
                sizes=SimpleNamespace(
                    max_total_num_tokens=64,
                    full_max_total_num_tokens=None,
                    swa_max_total_num_tokens=None,
                ),
                token_to_kv_pool=Mock(),
                is_dsv4_model=False,
                req_to_token_pool=SimpleNamespace(),
                token_to_kv_pool_allocator=None,
            )
        self.assertIs(actual, expected)
        paged.assert_called_once()
        hisparse.assert_not_called()

    def test_coordinator_accepts_two_and_three_item_retrieval_results(self):
        coordinator = object.__new__(SparseCoordinator)
        coordinator._forward_sparse_mask = torch.tensor([True])
        coordinator.algorithm = Mock()
        coordinator.backend_adaptor = Mock(supports_prepared_metadata=True)
        coordinator.backend_adaptor.adapt_for_attn_metadata.return_value = "metadata"
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        )
        coordinator.page_size = 4

        indices = torch.tensor([[3]], dtype=torch.int32)
        lengths = torch.tensor([1], dtype=torch.int32)
        for retrieval_result, prepared in (
            ((indices, lengths), False),
            ((indices, lengths, True), True),
        ):
            with self.subTest(prepared=prepared):
                coordinator.algorithm.retrieve_topk.return_value = retrieval_result
                result = coordinator._handle_sparse_retrieve(
                    torch.empty((1, 1)),
                    SimpleNamespace(layer_id=0),
                    SimpleNamespace(req_pool_indices=torch.tensor([0])),
                    "metadata",
                )
                self.assertEqual(result, "metadata")
                self.assertEqual(
                    coordinator.backend_adaptor.adapt_for_attn_metadata.call_args.kwargs[
                        "metadata_prepared"
                    ],
                    prepared,
                )
