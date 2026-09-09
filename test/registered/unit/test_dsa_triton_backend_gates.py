"""Construction gates for --dsa-*-backend triton."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, suite="base-a-test")


def _mock_model_runner(*, prefill="tilelang", decode="tilelang"):
    mr = MagicMock()
    mr.device = torch.device("cuda")
    mr.page_size = 1
    mr.server_args.speculative_eagle_topk = 0
    mr.server_args.enable_deterministic_inference = False
    mr.server_args.dsa_topk_backend = "sgl-kernel"
    mr.model_config.hf_config = object()
    mr.model_config.context_len = 32768
    mr.model_config.num_attention_heads = 64
    mr.model_config.qk_nope_head_dim = 512
    mr.model_config.kv_lora_rank = 512
    mr.model_config.qk_rope_head_dim = 64
    mr.token_to_kv_pool.dsa_kv_cache_store_fp8 = True
    mr.token_to_kv_pool.kv_cache_dim = 576
    mr.req_to_token_pool.size = 8
    mr.req_to_token_pool.req_to_token = MagicMock()
    mr.hisparse_coordinator = None
    mr.kv_cache_dtype = torch.float8_e4m3fn
    mr._triton_prefill = prefill
    mr._triton_decode = decode
    return mr


def _mock_exec_kernel(mr):
    exec_mock = MagicMock()
    exec_mock.kernel.dsa_prefill_backend = mr._triton_prefill
    exec_mock.kernel.dsa_decode_backend = mr._triton_decode
    exec_mock.deterministic.enable_deterministic_inference = False
    return exec_mock


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestDSATritonBackendGates(CustomTestCase):
    @patch("sglang.srt.layers.attention.dsa_backend.should_use_dsa_fused_topk", return_value=False)
    @patch("sglang.srt.layers.attention.dsa_backend.get_exec")
    @patch("sglang.srt.layers.attention.dsa_backend.envs")
    @patch("sglang.srt.layers.attention.dsa_backend.get_spec")
    @patch("sglang.srt.layers.attention.dsa_backend.get_parallel")
    @patch("sglang.srt.layers.attention.dsa_backend.get_dsa_index_kpool", return_value=1)
    @patch("sglang.srt.layers.attention.dsa_backend.get_dsa_index_topk", return_value=2048)
    @patch("sglang.srt.layers.attention.dsa_backend.is_deepseek_dsa", return_value=True)
    @patch("sglang.srt.layers.attention.dsa_backend._IS_GFX95", new=True)
    @patch("sglang.srt.layers.attention.dsa_backend._is_hip", new=False)
    def test_cuda_rejects_triton_prefill_backend(
        self,
        _mock_dsa,
        _mock_topk,
        _mock_kpool,
        mock_parallel,
        mock_spec,
        _mock_envs,
        mock_exec,
        _mock_fused_topk,
    ):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        mr = _mock_model_runner(prefill="triton")
        mock_exec.return_value = _mock_exec_kernel(mr)
        mock_parallel.return_value.attn_tp_size = 4
        mock_spec.return_value.speculative_num_draft_tokens = 0
        _mock_envs.SGLANG_DSA_FUSE_TOPK.get.return_value = False

        with self.assertRaisesRegex(ValueError, "gfx950-only"):
            DeepseekSparseAttnBackend(mr)

    @patch("sglang.srt.layers.attention.dsa_backend.should_use_dsa_fused_topk", return_value=False)
    @patch("sglang.srt.layers.attention.dsa_backend.get_exec")
    @patch("sglang.srt.layers.attention.dsa_backend.envs")
    @patch("sglang.srt.layers.attention.dsa_backend.get_spec")
    @patch("sglang.srt.layers.attention.dsa_backend.get_parallel")
    @patch("sglang.srt.layers.attention.dsa_backend.get_dsa_index_kpool", return_value=1)
    @patch("sglang.srt.layers.attention.dsa_backend.get_dsa_index_topk", return_value=2048)
    @patch("sglang.srt.layers.attention.dsa_backend.is_deepseek_dsa", return_value=True)
    @patch("sglang.srt.layers.attention.dsa_backend._IS_GFX95", new=True)
    @patch("sglang.srt.layers.attention.dsa_backend._is_hip", new=False)
    def test_cuda_rejects_triton_decode_backend(
        self,
        _mock_dsa,
        _mock_topk,
        _mock_kpool,
        mock_parallel,
        mock_spec,
        _mock_envs,
        mock_exec,
        _mock_fused_topk,
    ):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        mr = _mock_model_runner(decode="triton")
        mock_exec.return_value = _mock_exec_kernel(mr)
        mock_parallel.return_value.attn_tp_size = 4
        mock_spec.return_value.speculative_num_draft_tokens = 0
        _mock_envs.SGLANG_DSA_FUSE_TOPK.get.return_value = False

        with self.assertRaisesRegex(ValueError, "gfx950-only"):
            DeepseekSparseAttnBackend(mr)


class TestDSATritonShapeGates(CustomTestCase):
    def test_decode_gate_matches_mtp_token_range(self):
        from sglang.srt.layers.attention.dsa_backend import _triton_sparse_mla_decode_ok

        kv = torch.zeros(1, dtype=torch.float8_e4m3fn)
        self.assertTrue(_triton_sparse_mla_decode_ok(kv, 16, 512, 576, 2048, 1))
        self.assertTrue(_triton_sparse_mla_decode_ok(kv, 8, 512, 576, 2048, 84))
        self.assertFalse(_triton_sparse_mla_decode_ok(kv, 16, 512, 576, 2048, 0))
        self.assertFalse(_triton_sparse_mla_decode_ok(kv, 16, 512, 576, 2048, 100))
        self.assertFalse(_triton_sparse_mla_decode_ok(kv, 32, 512, 576, 2048, 12))


if __name__ == "__main__":
    unittest.main()
