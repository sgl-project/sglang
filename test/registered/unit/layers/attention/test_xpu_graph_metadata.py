import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestXPUGraphMetadata(CustomTestCase):
    def test_full_prefill_swa_page_table_with_multi_token_page_size(self):
        """Verify that with page_size > 1, SWA page table indexes raw token slot IDs

        first before integer division, preventing corrupting the lookup with
        premature page division.
        """
        page_size = 16
        max_context_len = 64
        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.attention_chunk_size = None
        backend.use_sliding_window_kv_pool = True
        backend.page_size = page_size
        backend.max_context_len = max_context_len

        # Pool setup: 2 requests, stride indices at 0, 16, 32, 48.
        # Say token slot for (req 0, pos 0) is 64 (page 4),
        # for (req 0, pos 16) is 80 (page 5),
        # for (req 1, pos 0) is 32 (page 2),
        # for (req 1, pos 16) is 48 (page 3).
        req_to_token = torch.zeros((2, max_context_len), dtype=torch.int32)
        req_to_token[0, 0] = 64
        req_to_token[0, 16] = 80
        req_to_token[1, 0] = 32
        req_to_token[1, 16] = 48
        backend.req_to_token = req_to_token

        # SWA translation mapping maps token slot ID -> SWA slot ID.
        # Suppose SWA map maps token slots by adding 1000:
        # slot 64 -> 1064 (SWA page 1064 // 16 = 66)
        # slot 80 -> 1080 (SWA page 1080 // 16 = 67)
        # slot 32 -> 1032 (SWA page 1032 // 16 = 64)
        # slot 48 -> 1048 (SWA page 1048 // 16 = 65)
        # If translate_loc_from_full_to_swa were passed (raw // 16), it would receive
        # 4, 5, 2, 3 instead of 64, 80, 32, 48!
        received_translation_inputs = []

        def mock_translate(token_slots):
            received_translation_inputs.append(token_slots.clone())
            return token_slots + 1000

        backend.token_to_kv_pool = SimpleNamespace(
            translate_loc_from_full_to_swa=mock_translate
        )

        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens=torch.tensor([32, 32], dtype=torch.int32),
            extend_seq_lens=torch.tensor([32, 32], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([32, 32], dtype=torch.int32),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            out_cache_loc=torch.arange(4, dtype=torch.int64),
            positions=torch.arange(4, dtype=torch.int64),
        )

        backend._init_full_cg_prefill_metadata(forward_batch, in_capture=True)

        # Check translation input was raw token slots, NOT page IDs
        self.assertEqual(
            len(received_translation_inputs), 2
        )  # page_table + out_cache_loc
        token_slots_input = received_translation_inputs[0]
        expected_slots = torch.tensor([[64, 80], [32, 48]], dtype=torch.int32)
        self.assertTrue(torch.equal(token_slots_input, expected_slots))

        # Check full page_table: token_slots // page_size
        expected_full_pages = expected_slots // page_size
        self.assertTrue(
            torch.equal(
                backend.full_cg_prefill_metadata.page_table[:, :2],
                expected_full_pages.to(torch.int32),
            )
        )

        # Check SWA page_table: (token_slots + 1000) // page_size
        expected_swa_pages = (expected_slots + 1000) // page_size
        self.assertTrue(
            torch.equal(
                backend.full_cg_prefill_metadata.swa_page_table[:, :2],
                expected_swa_pages.to(torch.int32),
            )
        )

    def test_full_prefill_rejects_attention_chunk_size(self):
        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.attention_chunk_size = 1024
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens=torch.tensor([16], dtype=torch.int32),
        )
        with self.assertRaisesRegex(
            ValueError,
            "Full prefill graph does not support models with attention_chunk_size",
        ):
            backend._init_full_cg_prefill_metadata(forward_batch, in_capture=True)

    def test_prefill_cuda_graph_runner_rejects_attention_chunk_size_under_full_graph(
        self,
    ):
        class _FakeModelRunner:
            device = "cpu"
            is_draft_worker = False
            attention_chunk_size = 512
            attn_backend = SimpleNamespace(
                attention_chunk_size=512,
                dllm_attention=None,
            )
            req_to_token_pool = SimpleNamespace(
                size=16,
                req_to_token=torch.zeros((16, 64), dtype=torch.int32),
            )
            model_config = SimpleNamespace(
                is_multimodal=False,
                hidden_size=64,
                hf_config=SimpleNamespace(),
            )
            model = SimpleNamespace(quant_config=None)
            lora_manager = None
            is_generation = True
            spec_algorithm = SimpleNamespace(
                is_eagle=lambda: False,
                is_dflash_family=lambda: False,
            )
            dtype = torch.float16
            pp_group = SimpleNamespace(is_first_rank=True)
            get_pp_proxy_topk_size = lambda self: 0
            get_pp_proxy_residual_num_blocks = lambda self: 0
            get_pp_proxy_dspark_hidden_size = lambda self: 0
            attn_tp_sequence_sharded = lambda self: False
            attention_layers = []
            mha_companion_layers = []
            moe_layers = []
            moe_fusions = []

        override = get_context().override_server_args(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(
                    backend="full",
                    bs=[1],
                    max_context_size=64,
                    full_prefill_max_req=1,
                )
            ),
        )
        override.install()
        self.addCleanup(override.restore)

        with (
            patch(
                "sglang.srt.model_executor.runner.prefill_cuda_graph_runner.resolve_prefill_backend",
                return_value=SimpleNamespace(),
            ),
            patch(
                "sglang.srt.model_executor.runner.prefill_cuda_graph_runner._is_full_graph_backend",
                return_value=True,
            ),
            self.assertRaisesRegex(
                ValueError,
                "Full prefill CUDA/XPU graph does not support models with attention_chunk_size",
            ),
        ):
            PrefillCudaGraphRunner(_FakeModelRunner())


if __name__ == "__main__":
    unittest.main()
