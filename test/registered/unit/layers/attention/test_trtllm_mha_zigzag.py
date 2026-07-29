import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.attention.trtllm_mha_backend as trtllm_mha_backend
from sglang.srt.layers.attention.trtllm_mha_backend import (
    TRTLLMHAAttnBackend,
    TRTLLMMHAMetadata,
)
from sglang.srt.layers.cp.base import CPAttentionBackendKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestTRTLLMMHAZigzagPageTables(CustomTestCase):
    def _backend(self, swa_pool=None):
        backend = object.__new__(TRTLLMHAAttnBackend)
        backend._swa_kv_pool = swa_pool
        return backend

    def _metadata(self):
        return TRTLLMMHAMetadata(
            page_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            swa_page_table=torch.tensor([[30, 31], [40, 41]], dtype=torch.int32),
        )

    def test_builds_prev_then_next_page_tables_for_cp_v2(self):
        backend = self._backend()
        metadata = self._metadata()

        with patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
            return_value=True,
        ):
            backend._build_zigzag_page_tables(metadata, SimpleNamespace())

        self.assertEqual(
            metadata.zigzag_page_table.tolist(),
            [[10, 11], [20, 21], [10, 11], [20, 21]],
        )
        self.assertEqual(
            metadata.zigzag_swa_page_table.tolist(),
            [[30, 31], [40, 41], [30, 31], [40, 41]],
        )

    def test_leaves_zigzag_page_tables_unset_without_cp_v2(self):
        backend = self._backend()
        metadata = self._metadata()

        with patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
            return_value=False,
        ):
            backend._build_zigzag_page_tables(metadata, SimpleNamespace())

        self.assertIsNone(metadata.zigzag_page_table)
        self.assertIsNone(metadata.zigzag_swa_page_table)

    def test_selects_full_or_swa_zigzag_page_table_per_layer(self):
        swa_pool = SimpleNamespace(
            layers_mapping={
                0: (None, False),
                1: (None, True),
            }
        )
        backend = self._backend(swa_pool=swa_pool)
        metadata = self._metadata()
        metadata.zigzag_page_table = torch.tensor([[1], [2]], dtype=torch.int32)
        metadata.zigzag_swa_page_table = torch.tensor([[3], [4]], dtype=torch.int32)
        backend.forward_metadata = metadata

        full_table = backend._get_zigzag_layer_page_table(SimpleNamespace(layer_id=0))
        swa_table = backend._get_zigzag_layer_page_table(SimpleNamespace(layer_id=1))

        self.assertIs(full_table, metadata.zigzag_page_table)
        self.assertIs(swa_table, metadata.zigzag_swa_page_table)

    def _forward_backend(self):
        backend = self._backend()
        backend.decode_uses_native_fp4 = False
        backend.data_type = torch.bfloat16
        backend.q_data_type = torch.bfloat16
        backend.page_size = 1
        backend.device = torch.device("cpu")
        backend.workspace_buffer = torch.empty(1, dtype=torch.uint8)
        backend.max_context_len = 64
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=lambda layer_id: (
                torch.zeros(16, 2, 2, dtype=torch.bfloat16),
                torch.zeros(16, 2, 2, dtype=torch.bfloat16),
            )
        )
        backend.forward_metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=torch.tensor([3, 4], dtype=torch.int32),
            max_seq_len_q=2,
            cu_seqlens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 3, 7], dtype=torch.int32),
            page_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            zigzag_page_table=torch.tensor(
                [[10, 11], [20, 21], [10, 11], [20, 21]],
                dtype=torch.int32,
            ),
        )
        return backend

    def _layer_and_batch(self):
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=2,
            tp_k_head_num=2,
            tp_v_head_num=2,
            head_dim=2,
            scaling=0.5,
            sliding_window_size=-1,
        )
        forward_mode = SimpleNamespace(
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
        )
        forward_batch = SimpleNamespace(
            out_cache_loc=torch.arange(4),
            forward_mode=forward_mode,
        )
        return layer, forward_batch

    def test_forward_extend_selects_combined_zigzag_page_table(self):
        backend = self._forward_backend()
        layer, forward_batch = self._layer_and_batch()
        q = torch.arange(16, dtype=torch.bfloat16).view(4, 4)
        combined_cu_q = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        combined_cache_lens = torch.tensor([3, 4, 3, 4], dtype=torch.int32)
        combined_cu_kv = torch.tensor([0, 3, 7, 10, 14], dtype=torch.int32)

        class _CombinedStrategy:
            def run_attention(
                self,
                query,
                batch,
                device,
                attn_fn,
                attention_backend,
            ):
                self.asserted_attention_backend = attention_backend
                return attn_fn(
                    query,
                    combined_cu_q,
                    combined_cache_lens,
                    1,
                    cu_seqlens_kv=combined_cu_kv,
                    use_zigzag_page_table=True,
                )

        strategy = _CombinedStrategy()
        with (
            patch.object(
                trtllm_mha_backend.flashinfer.prefill,
                "trtllm_batch_context_with_kv_cache",
                side_effect=lambda **kwargs: kwargs["query"],
            ) as context_attn,
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.get_cp_strategy",
                return_value=strategy,
            ),
        ):
            out = backend.forward_extend(
                q,
                None,
                None,
                layer,
                forward_batch,
                save_kv_cache=False,
            )

        self.assertEqual(
            strategy.asserted_attention_backend,
            CPAttentionBackendKind.TRTLLM_MHA,
        )
        call = context_attn.call_args.kwargs
        self.assertTrue(
            torch.equal(
                call["block_tables"],
                backend.forward_metadata.zigzag_page_table,
            )
        )
        self.assertEqual(call["batch_size"], 4)
        self.assertTrue(torch.equal(call["seq_lens"], combined_cache_lens))
        self.assertTrue(torch.equal(call["cum_seq_lens_q"], combined_cu_q))
        self.assertTrue(torch.equal(call["cum_seq_lens_kv"], combined_cu_kv))
        self.assertTrue(torch.equal(call["query"].reshape(4, 4), q))
        self.assertTrue(torch.equal(out, q))

    def test_forward_extend_keeps_original_non_cp_page_table(self):
        backend = self._forward_backend()
        layer, forward_batch = self._layer_and_batch()
        q = torch.arange(16, dtype=torch.bfloat16).view(4, 4)

        with (
            patch.object(
                trtllm_mha_backend.flashinfer.prefill,
                "trtllm_batch_context_with_kv_cache",
                side_effect=lambda **kwargs: kwargs["query"],
            ) as context_attn,
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
                return_value=False,
            ),
        ):
            out = backend.forward_extend(
                q,
                None,
                None,
                layer,
                forward_batch,
                save_kv_cache=False,
            )

        call = context_attn.call_args.kwargs
        self.assertTrue(
            torch.equal(call["block_tables"], backend.forward_metadata.page_table)
        )
        self.assertEqual(call["batch_size"], 2)
        self.assertTrue(
            torch.equal(
                call["seq_lens"],
                backend.forward_metadata.cache_seqlens_int32,
            )
        )
        self.assertTrue(
            torch.equal(
                call["cum_seq_lens_q"],
                backend.forward_metadata.cu_seqlens_q,
            )
        )
        self.assertTrue(
            torch.equal(
                call["cum_seq_lens_kv"],
                backend.forward_metadata.cu_seqlens_k,
            )
        )
        self.assertTrue(torch.equal(out, q))


if __name__ == "__main__":
    unittest.main()
