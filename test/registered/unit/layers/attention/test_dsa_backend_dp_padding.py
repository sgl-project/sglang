import unittest
from contextlib import nullcontext
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DSADcpPageTableCache,
    _restore_trtllm_decode_dp_padding,
    _trim_trtllm_decode_dp_padding,
)
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSADcpPageTableCache(unittest.TestCase):
    def test_shared_topk_layers_reuse_the_local_page_table(self):
        backend = SimpleNamespace(
            dcp_size=2,
            dcp_rank=0,
            dcp_head_groups=1,
            dcp_layer_shares_topk=[False, True, True, False, True],
        )
        cache = DSADcpPageTableCache()
        # Producers rewrite the same top-k tensor in place.
        topk = torch.zeros((1, 4), dtype=torch.int32)
        with patch(
            "sglang.srt.layers.attention.dsa_backend.transform_index_page_table_dcp",
            side_effect=lambda *args, **kwargs: (object(), object()),
        ) as transform:
            tables = [
                DeepseekSparseAttnBackend._dcp_local_page_table(
                    backend, layer_id, topk, topk, cache
                )[0]
                for layer_id in range(5)
            ]
        self.assertEqual(transform.call_count, 2)
        self.assertIs(tables[0], tables[2])
        self.assertIsNot(tables[2], tables[3])
        self.assertIs(tables[3], tables[4])


class TestDSABackendDPPadding(unittest.TestCase):
    def test_occupied_uneven_decode_with_idle_rank_selects_max_len(self):
        with (
            patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
            patch(
                "sglang.srt.layers.dp_attention.dp_gather_width",
                return_value=4,
            ),
        ):
            mode = DpPaddingMode.get_dp_padding_mode(
                is_extend_in_batch=False,
                global_num_tokens=[8, 7, 6, 0],
            )
        self.assertEqual(mode, DpPaddingMode.MAX_LEN)

    def test_trim_and_restore_active_eager_decode_padding(self):
        q_all = torch.arange(8 * 2 * 3).view(8, 2, 3)
        topk_indices = torch.arange(8 * 4, dtype=torch.int32).view(8, 4)

        real_q, real_topk, num_padding_rows = _trim_trtllm_decode_dp_padding(
            q_all,
            topk_indices,
            real_batch_size=6,
        )

        self.assertTrue(torch.equal(real_q, q_all[:6]))
        self.assertTrue(torch.equal(real_topk, topk_indices[:6]))
        self.assertEqual(num_padding_rows, 2)

        real_output = torch.ones((6, 1, 2, 5), dtype=torch.bfloat16)
        output = _restore_trtllm_decode_dp_padding(real_output, num_padding_rows)
        self.assertEqual(output.shape, (8, 1, 2, 5))
        self.assertTrue(torch.equal(output[:6], real_output))
        self.assertTrue(torch.all(output[6:] == 0))

    def test_no_padding_preserves_existing_tensors(self):
        q_all = torch.empty((2, 2, 3))
        topk_indices = torch.empty((2, 8), dtype=torch.int32)

        real_q, real_topk, num_padding_rows = _trim_trtllm_decode_dp_padding(
            q_all,
            topk_indices,
            real_batch_size=2,
        )

        self.assertIs(real_q, q_all)
        self.assertIs(real_topk, topk_indices)
        self.assertEqual(num_padding_rows, 0)
        self.assertIs(
            _restore_trtllm_decode_dp_padding(real_q, num_padding_rows),
            real_q,
        )

    def test_rejects_metadata_larger_than_physical_batch(self):
        with self.assertRaisesRegex(
            AssertionError, "metadata batch size \\(3\\) exceeds q batch size \\(2\\)"
        ):
            _trim_trtllm_decode_dp_padding(
                torch.empty((2, 2, 3)),
                torch.empty((2, 8), dtype=torch.int32),
                real_batch_size=3,
            )

    def test_trtllm_decode_runs_real_rows_then_restores_physical_batch(self):
        self._check_trtllm_decode_padding(dcp_enabled=False)

    def test_dcp_trtllm_decode_pads_lse_with_negative_infinity(self):
        self._check_trtllm_decode_padding(dcp_enabled=True)

    def _check_trtllm_decode_padding(self, *, dcp_enabled):
        metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([8, 12], dtype=torch.int32),
            page_table_1=torch.zeros((2, 12), dtype=torch.int32),
            max_seq_len_k=12,
            dcp_cu_seqlens_q=torch.arange(3, dtype=torch.int32),
            dcp_page_table_cache=DSADcpPageTableCache(),
        )
        backend = SimpleNamespace(
            forward_metadata=metadata,
            kv_cache_dtype=torch.bfloat16,
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda _layer_id: torch.zeros((24, 3))
            ),
            physical_page_size=1,
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
            num_dcp_q_heads=2,
            dcp_enabled=dcp_enabled,
            dcp_size=2,
            dcp_rank=0,
            dcp_head_groups=1,
            dcp_layer_shares_topk=[False],
        )
        backend._pad_topk_indices = MethodType(
            DeepseekSparseAttnBackend._pad_topk_indices, backend
        )
        backend._dcp_local_page_table = MethodType(
            DeepseekSparseAttnBackend._dcp_local_page_table, backend
        )
        backend._pad_trtllm_sparse_page_table = MethodType(
            DeepseekSparseAttnBackend._pad_trtllm_sparse_page_table, backend
        )
        backend._multi_ctas_kv_counter_for = MethodType(
            DeepseekSparseAttnBackend._multi_ctas_kv_counter_for, backend
        )

        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=2,
            head_dim=3,
            k_scale_float=None,
            scaling=1.0,
        )
        forward_batch = SimpleNamespace()
        q = torch.arange(4 * 2 * 3, dtype=torch.float32).view(4, 2, 3)
        topk_indices = torch.arange(4 * 2, dtype=torch.int32).view(4, 2)

        flashinfer = ModuleType("flashinfer")
        flashinfer_decode = ModuleType("flashinfer.decode")
        captured = {}

        def fake_decode(**kwargs):
            captured.update(kwargs)
            return torch.ones((2, 1, 2, 2), dtype=torch.bfloat16)

        def fake_dcp_attention(**kwargs):
            captured.update(kwargs)
            return (
                torch.ones((2, 2, 2), dtype=torch.bfloat16),
                torch.full((2, 2), 7.0),
            )

        flashinfer_decode.trtllm_batch_decode_with_kv_cache_mla = fake_decode
        flashinfer.decode = flashinfer_decode

        def fake_transform(*, page_table, topk_indices, page_size):
            self.assertEqual(page_table.shape[0], 2)
            self.assertEqual(topk_indices.shape[0], 2)
            self.assertEqual(page_size, 1)
            return torch.zeros((2, 2), dtype=torch.int32)

        with (
            patch.dict(
                "sys.modules",
                {
                    "flashinfer": flashinfer,
                    "flashinfer.decode": flashinfer_decode,
                },
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.transform_index_page_table_decode",
                side_effect=fake_transform,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.dsa_use_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.grow_multi_ctas_kv_counter_buffer_if_needed",
                return_value=None,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.transform_index_page_table_dcp",
                side_effect=lambda page_table, *args, **kwargs: (
                    page_table,
                    torch.zeros(page_table.shape[0], dtype=torch.int32),
                ),
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend._dcp_trtllm_sparse_attention",
                side_effect=fake_dcp_attention,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.use_symmetric_memory",
                side_effect=lambda _: nullcontext(),
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.get_parallel",
                return_value=SimpleNamespace(dcp_group=None),
            ),
        ):
            output = DeepseekSparseAttnBackend._forward_trtllm(
                backend,
                q=q,
                k=torch.empty((4, 1, 2)),
                v=torch.empty((4, 1, 2)),
                layer=layer,
                forward_batch=forward_batch,
                seq_lens=metadata.cache_seqlens_int32,
                save_kv_cache=False,
                topk_indices=topk_indices,
            )

        self.assertEqual(captured["query"].shape, (2, 1, 2, 3))
        if dcp_enabled:
            output, lse = output
            self.assertEqual(output.shape, (4, 2, 2))
            self.assertEqual(lse.shape, (4, 2))
            self.assertTrue(torch.all(lse[:2] == 7))
            self.assertTrue(torch.isneginf(lse[2:]).all())
        else:
            self.assertEqual(output.shape, (4, 1, 2, 2))
        self.assertTrue(torch.all(output[:2] == 1))
        self.assertTrue(torch.all(output[2:] == 0))


if __name__ == "__main__":
    unittest.main()
