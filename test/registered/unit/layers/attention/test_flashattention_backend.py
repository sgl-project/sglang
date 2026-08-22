from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention import flashattention_backend  # noqa: E402
from sglang.srt.layers.attention.flashattention_backend import (  # noqa: E402
    FlashAttentionBackend,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashAttentionBackendDecodeScaling(CustomTestCase):
    def _make_backend(self, fa_impl_ver):
        backend = object.__new__(FlashAttentionBackend)
        backend.fa_impl_ver = fa_impl_ver
        backend.use_mla = False
        backend.kv_cache_is_mxfp8 = False
        backend.kv_cache_dtype_str = "bfloat16"
        backend.kv_cache_dtype = torch.bfloat16
        backend.has_local_attention = False
        backend.attention_chunk_size = None
        backend.topk = 0
        backend.page_size = 1
        backend.use_sliding_window_kv_pool = False
        backend.is_prefill_aware_swa = False
        backend.num_splits = 0
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=lambda _layer_id: (
                torch.zeros(1, 1, 8),
                torch.zeros(1, 1, 8),
            )
        )
        backend.forward_metadata = SimpleNamespace(
            local_attn_metadata=None,
            page_table=torch.zeros(1, 1, dtype=torch.int32),
            cache_seqlens_int32=torch.ones(1, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            max_seq_len_q=1,
            scheduler_metadata=None,
            pa_swa_page_table=None,
        )
        return backend

    def _run_decode(self, fa_impl_ver):
        backend = self._make_backend(fa_impl_ver)
        layer = SimpleNamespace(
            is_cross_attention=False,
            attn_type=None,
            sliding_window_size=None,
            head_dim=8,
            v_head_dim=8,
            tp_q_head_num=1,
            tp_k_head_num=1,
            tp_v_head_num=1,
            k_scale=torch.tensor(2.0),
            v_scale=torch.tensor(3.0),
            layer_id=0,
            scaling=1.0,
            logit_cap=0.0,
        )
        forward_batch = SimpleNamespace(
            batch_size=1,
            spec_info=None,
            _attn_output=None,
        )
        kernel = MagicMock(side_effect=lambda **kwargs: torch.zeros_like(kwargs["q"]))

        with patch.object(flashattention_backend, "flash_attn_with_kvcache", kernel):
            backend.forward_decode(
                torch.ones(1, 8, dtype=torch.float16),
                None,
                None,
                layer,
                forward_batch,
                save_kv_cache=False,
            )

        kernel.assert_called_once()
        return kernel.call_args.kwargs

    def test_fa4_decode_does_not_pass_unsupported_descale_tensors(self):
        kernel_kwargs = self._run_decode(fa_impl_ver=4)

        self.assertNotIn("k_descale", kernel_kwargs)
        self.assertNotIn("v_descale", kernel_kwargs)
        self.assertEqual(kernel_kwargs["q"].dtype, torch.float16)

    def test_fa3_decode_preserves_supported_descale_tensors(self):
        kernel_kwargs = self._run_decode(fa_impl_ver=3)

        self.assertTrue(torch.equal(kernel_kwargs["k_descale"], torch.tensor([[2.0]])))
        self.assertTrue(torch.equal(kernel_kwargs["v_descale"], torch.tensor([[3.0]])))
        self.assertEqual(kernel_kwargs["q"].dtype, torch.bfloat16)


if __name__ == "__main__":
    import unittest

    unittest.main()
