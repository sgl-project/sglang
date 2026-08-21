import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

with patch.dict(
    sys.modules,
    {
        module: MagicMock()
        for module in (
            "sgl_kernel",
            "sgl_kernel.quantization",
            "sgl_kernel.scalar_type",
        )
    },
):
    from sglang.srt.layers.attention.minicpm import attention_adapter as adapter_module
    from sglang.srt.layers.attention.minicpm.attention_adapter import (
        MiniCPMFlashAttentionAdapter,
        MiniCPMFlashInferAdapter,
    )

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _metadata(rows=1):
    return SimpleNamespace(
        sparse_page_table=torch.zeros((rows, 4), dtype=torch.int32),
        sparse_cache_seqlens_int32=torch.full(
            (rows,),
            4,
            dtype=torch.int32,
        ),
        sparse_cu_seqlens_q=torch.arange(rows + 1, dtype=torch.int32),
        sparse_cu_seqlens_k=torch.arange(
            0,
            (rows + 1) * 4,
            4,
            dtype=torch.int32,
        ),
        sparse_max_seq_len_q=1,
        max_seq_len_q=1,
    )


class TestMiniCPMAttentionAdapter(unittest.TestCase):
    def test_flashattention_adapter_owns_kernel_arguments(self):
        expected = torch.ones(1, 1, 1)
        flash_attn_backend = SimpleNamespace(
            num_splits=4,
            fa_impl_ver=3,
        )
        adapter = MiniCPMFlashAttentionAdapter(flash_attn_backend)
        metadata = _metadata()
        layer = SimpleNamespace(scaling=0.125, logit_cap=0.0)
        k_descale = torch.tensor([[2.0]])
        v_descale = torch.tensor([[4.0]])

        with patch.object(
            adapter_module,
            "flash_attn_with_kvcache",
            return_value=expected,
        ) as kernel:
            result = adapter.forward(
                torch.ones(1, 1, 1),
                torch.ones(4, 1, 1, 1),
                torch.ones(4, 1, 1, 1),
                metadata,
                layer,
                is_prefill=True,
                k_descale=k_descale,
                v_descale=v_descale,
            )

        self.assertIs(result, expected)
        kwargs = kernel.call_args.kwargs
        self.assertIs(kwargs["page_table"], metadata.sparse_page_table)
        self.assertIs(kwargs["k_descale"], k_descale)
        self.assertIs(kwargs["v_descale"], v_descale)
        self.assertEqual(kwargs["num_splits"], 4)
        self.assertEqual(kwargs["ver"], 3)

    def test_flashinfer_prefill_plans_once_and_executes_each_layer(self):
        adapter = MiniCPMFlashInferAdapter.__new__(MiniCPMFlashInferAdapter)
        adapter.prefill_planned = False
        adapter._prepare = Mock()
        adapter.active_rows = torch.tensor([0], dtype=torch.int32)
        adapter.active_kv_indptr = torch.tensor([0, 4], dtype=torch.int32)
        adapter.active_kv_indices = torch.empty(4, dtype=torch.int32)
        expected = torch.ones(1, 1, 1)
        adapter.active_wrapper = SimpleNamespace(forward=Mock(return_value=expected))
        metadata = _metadata()
        layer = SimpleNamespace(
            scaling=0.125,
            logit_cap=0.0,
            k_scale_float=1.0,
            v_scale_float=1.0,
        )

        with patch.object(
            adapter_module,
            "create_flashinfer_kv_indices_triton",
        ) as index_kernel:
            first = adapter.forward(
                torch.ones(1, 1, 1),
                torch.ones(4, 1, 1, 1),
                torch.ones(4, 1, 1, 1),
                metadata,
                layer,
                is_prefill=True,
            )
            second = adapter.forward(
                torch.ones(1, 1, 1),
                torch.ones(4, 1, 1, 1),
                torch.ones(4, 1, 1, 1),
                metadata,
                layer,
                is_prefill=True,
            )

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        adapter._prepare.assert_called_once_with(metadata, is_prefill=True)
        self.assertEqual(index_kernel.__getitem__.return_value.call_count, 2)
        self.assertEqual(adapter.active_wrapper.forward.call_count, 2)

    def test_flashinfer_graph_uses_backend_wrapper_cache(self):
        adapter = MiniCPMFlashInferAdapter.__new__(MiniCPMFlashInferAdapter)
        adapter.device = torch.device("cpu")
        adapter.head_group_num = 2
        adapter.num_qo_heads = 4
        adapter.num_kv_heads = 1
        adapter.head_dim = 16
        adapter.page_size = 1
        adapter.max_kv_tokens_per_row = 4
        adapter.q_dtype = torch.float16
        adapter.kv_dtype = torch.float16
        adapter.kv_indptr = torch.zeros(3, dtype=torch.int32)
        adapter.kv_indices = torch.zeros(8, dtype=torch.int32)
        adapter.kv_last_page_len = torch.ones(2, dtype=torch.int32)
        adapter.rows = torch.arange(2, dtype=torch.int32)
        wrapper = SimpleNamespace(begin_forward=Mock())
        adapter.flashinfer_backend = SimpleNamespace(
            get_cuda_graph_decode_wrappers=Mock(return_value=[wrapper]),
        )
        metadata = _metadata(rows=2)

        adapter.prepare_forward(
            metadata,
            is_prefill=False,
            graph=True,
        )

        adapter.flashinfer_backend.get_cuda_graph_decode_wrappers.assert_called_once_with(
            bs=1,
            num_tokens=2,
        )
        wrapper.begin_forward.assert_called_once()
        self.assertIs(adapter.active_wrapper, wrapper)

    def test_flashinfer_decode_indices_cover_dense_rows(self):
        wrapper = SimpleNamespace(begin_forward=Mock())
        flashinfer_backend = SimpleNamespace(decode_wrappers=[wrapper])
        flashinfer_backend_module = ModuleType(
            "sglang.srt.layers.attention.flashinfer_backend"
        )
        flashinfer_backend_module.FlashInferAttnBackend = Mock(
            return_value=flashinfer_backend
        )
        model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.float16,
            kv_cache_dtype=torch.float16,
            req_to_token_pool=SimpleNamespace(size=1),
        )

        with (
            patch.object(adapter_module, "is_flashinfer_available", return_value=True),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.layers.attention.flashinfer_backend": (
                        flashinfer_backend_module
                    ),
                },
            ),
        ):
            adapter = MiniCPMFlashInferAdapter(
                model_runner,
                head_group_num=2,
                heads_per_group=16,
                head_dim=128,
                page_size=1,
                max_kv_tokens_per_row=7,
            )

        metadata = _metadata(rows=2)
        metadata.sparse_cache_seqlens_int32.fill_(7)
        adapter.prepare_forward(
            metadata,
            is_prefill=False,
            graph=False,
        )

        self.assertEqual(adapter.kv_indices.numel(), 14)
        self.assertEqual(adapter.active_kv_indices.numel(), 14)


if __name__ == "__main__":
    unittest.main()
