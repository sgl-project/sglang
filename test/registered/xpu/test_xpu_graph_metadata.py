import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")


class TestXPUGraphMetadata(CustomTestCase):
    def test_full_prefill_swa_page_table_with_multi_token_page_size_and_boundary_masking(
        self,
    ):
        """Verify that with page_size > 1 and variable request lengths:
        1. SWA page table indexes raw token slot IDs first before integer division.
        2. Boundary / inactive token slots (e.g. -1 or unallocated slots when
           seq_len < max_context_len) are safely masked before calling
           translate_loc_from_full_to_swa, preventing UB or out-of-bounds indexing.
        3. Inactive page table slots are cleanly zeroed out.
        """
        page_size = 16
        max_context_len = 64
        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.attention_chunk_size = None
        backend.use_sliding_window_kv_pool = True
        backend.page_size = page_size
        backend.max_context_len = max_context_len

        # Pool setup: 2 requests, stride indices at 0, 16, 32, 48.
        # Req 0 has length 32 (needs 2 pages: pos 0 and 16).
        # Req 1 has length 10 (needs 1 page: pos 0). Pos 16 has -1 (unallocated / dirty).
        req_to_token = torch.full((2, max_context_len), -1, dtype=torch.int32)
        req_to_token[0, 0] = 64  # Page 4
        req_to_token[0, 16] = 80  # Page 5
        req_to_token[1, 0] = 32  # Page 2
        # req_to_token[1, 16] remains -1 (dirty/unallocated)
        backend.req_to_token = req_to_token

        received_translation_inputs = []

        def mock_translate(token_slots):
            # Assert no -1 or negative slot IDs are passed to translation!
            self.assertTrue(
                torch.all(token_slots >= 0), f"Negative slot found: {token_slots}"
            )
            received_translation_inputs.append(token_slots.clone())
            return token_slots + 1000

        backend.token_to_kv_pool = SimpleNamespace(
            translate_loc_from_full_to_swa=mock_translate
        )

        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens=torch.tensor([32, 10], dtype=torch.int32),
            extend_seq_lens=torch.tensor([32, 10], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([32, 10], dtype=torch.int32),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            out_cache_loc=torch.arange(4, dtype=torch.int64),
            positions=torch.arange(4, dtype=torch.int64),
        )

        backend._init_full_cg_prefill_metadata(forward_batch, in_capture=True)

        # Check translation input: unallocated slot (req 1, page 1) was masked to 0
        token_slots_input = received_translation_inputs[0]
        expected_slots = torch.tensor([[64, 80], [32, 0]], dtype=torch.int32)
        self.assertTrue(torch.equal(token_slots_input, expected_slots))

        # Check full page_table: req 1 page 1 is zeroed
        expected_full_pages = torch.tensor([[4, 5], [2, 0]], dtype=torch.int32)
        self.assertTrue(
            torch.equal(
                backend.full_cg_prefill_metadata.page_table[:, :2],
                expected_full_pages,
            )
        )

        # Check SWA page_table: (expected_slots + 1000) // page_size with masked entry zeroed
        # req 0: 1064 // 16 = 66, 1080 // 16 = 67
        # req 1: 1032 // 16 = 64, page 1 masked to 0
        expected_swa_pages = torch.tensor([[66, 67], [64, 0]], dtype=torch.int32)
        self.assertTrue(
            torch.equal(
                backend.full_cg_prefill_metadata.swa_page_table[:, :2],
                expected_swa_pages,
            )
        )

    def test_forward_extend_hybrid_swa_and_irope_rejection(self):
        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.attention_chunk_size = 1024
        backend.dcp_size = 1
        backend.topk = 1

        # Case 1: layer with use_irope=True, but local_attn_metadata is None -> MUST raise
        metadata = SimpleNamespace(local_attn_metadata=None)
        backend.forward_metadata = metadata
        layer_irope = SimpleNamespace(
            use_irope=True,
            is_cross_attention=False,
            attn_type=None,
            sliding_window_size=-1,
        )
        forward_batch = SimpleNamespace(
            batch_size=1,
            out_cache_loc=None,
            encoder_out_cache_loc=None,
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Full prefill graph does not support models with attention_chunk_size and use_irope",
        ):
            backend.forward_extend(
                q=torch.zeros((1, 1, 16)),
                k=None,
                v=None,
                layer=layer_irope,
                forward_batch=forward_batch,
                save_kv_cache=False,
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
                prefill_cuda_graph_default_max_seq_len=None,
            )
            req_to_token_pool = SimpleNamespace(
                size=16,
                req_to_token=torch.zeros((16, 64), dtype=torch.int32),
            )
            model_config = SimpleNamespace(
                is_multimodal=False,
                hidden_size=64,
                hf_config=SimpleNamespace(model_type=None),
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
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(
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

    def test_prefill_cuda_graph_runner_rejects_irope_under_full_graph(self):
        class _FakeLayer:
            use_irope = True

        class _FakeModel:
            quant_config = None

            def modules(self):
                return [_FakeLayer()]

        class _FakeModelRunner:
            device = "cpu"
            is_draft_worker = False
            attention_chunk_size = 512
            attn_backend = SimpleNamespace(
                attention_chunk_size=512,
                dllm_attention=None,
                prefill_cuda_graph_default_max_seq_len=None,
            )
            req_to_token_pool = SimpleNamespace(
                size=16,
                req_to_token=torch.zeros((16, 64), dtype=torch.int32),
            )
            model_config = SimpleNamespace(
                is_multimodal=False,
                hidden_size=64,
                attention_chunk_size=512,
                hf_config=SimpleNamespace(model_type=None),
            )
            model = _FakeModel()
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
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(
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
                "Full prefill CUDA/XPU graph does not support models with "
                "attention_chunk_size or use_irope",
            ),
        ):
            PrefillCudaGraphRunner(_FakeModelRunner())


if __name__ == "__main__":
    unittest.main()
