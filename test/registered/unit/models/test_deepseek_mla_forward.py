import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekMLAForwardPolicy(unittest.TestCase):
    def _fuse_rope_for_dsa(self, prefill_backend, decode_backend, kv_cache_dtype):
        server_args = SimpleNamespace(
            dsa_prefill_backend=prefill_backend,
            dsa_decode_backend=decode_backend,
        )
        attn_backend = SimpleNamespace(kv_cache_dtype=kv_cache_dtype)
        fake_self = SimpleNamespace(current_attention_backend="dsa")
        fake_batch = SimpleNamespace()

        with (
            patch.object(forward_mla, "get_global_server_args", lambda: server_args),
            patch.object(forward_mla, "get_attn_backend", lambda: attn_backend),
        ):
            return forward_mla.DeepseekMLAForwardMixin._fuse_rope_for_trtllm_mla(
                fake_self, fake_batch
            )

    def test_flashinfer_sparse_mla_uses_native_bf16_query_path(self):
        self.assertFalse(
            self._fuse_rope_for_dsa(
                "flashinfer_sparse_mla",
                "flashinfer_sparse_mla",
                torch.float8_e4m3fn,
            )
        )

    def test_trtllm_dsa_backend_uses_fp8_rope_fusion(self):
        self.assertTrue(
            self._fuse_rope_for_dsa(
                "trtllm",
                "trtllm",
                torch.float8_e4m3fn,
            )
        )

    def test_non_trtllm_style_dsa_backend_does_not_fuse_rope(self):
        self.assertFalse(
            self._fuse_rope_for_dsa(
                "fa3",
                "fa3",
                torch.float8_e4m3fn,
            )
        )


class TestFlashInferSparseMLAPageTable(unittest.TestCase):
    def test_sparse_mla_lengths_stay_int32(self):
        page_table = torch.tensor([[10, 11, -1], [7, -1, -1]], dtype=torch.int64)

        block_tables, seq_lens = (
            dsa_backend._normalize_flashinfer_sparse_mla_page_table(page_table)
        )

        self.assertEqual(block_tables.dtype, torch.int32)
        self.assertEqual(seq_lens.dtype, torch.int32)
        self.assertEqual(tuple(block_tables.shape), (2, 1, 3))
        self.assertEqual(block_tables.tolist(), [[[10, 11, 0]], [[7, 0, 0]]])
        self.assertEqual(seq_lens.tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
