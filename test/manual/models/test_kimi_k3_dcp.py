"""Manual unit coverage for Kimi-K3 decode context parallel wiring."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    _select_local_dcp_heads_for_autotune,
)
from sglang.srt.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiK3LinearForCausalLM,
)
from sglang.test.test_utils import CustomTestCase


def _metadata_kwargs():
    return {
        "seq_lens": torch.tensor([8]),
        "extend_prefix_lens": torch.tensor([4]),
        "extend_prefix_lens_cpu": torch.tensor([4]),
        "extend_seq_lens": torch.tensor([4]),
        "req_pool_indices": torch.tensor([0]),
        "req_to_token": torch.arange(16).view(1, 16),
        "seq_lens_sum": 8,
        "kv_buffer_shape": torch.Size([16, 1, 576]),
        "kv_cache_dtype": torch.bfloat16,
        "kv_cache_device": torch.device("cpu"),
        "create_chunked_prefix_cache_kv_indices_fn": object(),
    }


class TestKimiK3DCP(CustomTestCase):
    @patch(
        "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla."
        "get_parallel",
    )
    def test_autotune_selects_local_heads_without_dcp_communication(self, get_parallel):
        get_parallel.return_value.attn_dcp_rank = 3
        partials = torch.arange(2 * 8 * 4 * 5).view(2, 8 * 4, 5)

        local = _select_local_dcp_heads_for_autotune(partials, num_local_heads=4)

        torch.testing.assert_close(local, partials[:, 12:16])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dcp8_kv_write_uses_one_eighth_per_rank(self):
        dcp_size = 8
        global_tokens = 1024
        local_tokens = global_tokens // dcp_size
        locations = torch.arange(global_tokens, dtype=torch.int64, device="cuda")
        cache_k_nope = (
            torch.arange(1, global_tokens + 1, dtype=torch.float32, device="cuda")
            .to(torch.bfloat16)
            .view(global_tokens, 1, 1)
        )
        cache_k_rope = -cache_k_nope

        for rank in range(dcp_size):
            kv_buffer = torch.zeros(
                (local_tokens, 1, 2), dtype=torch.bfloat16, device="cuda"
            )
            parallel = SimpleNamespace(
                dcp_enabled=True,
                attn_dcp_size=dcp_size,
                attn_dcp_rank=rank,
            )
            with patch(
                "sglang.kernels.ops.kvcache.mla_buffer.get_parallel",
                return_value=parallel,
            ):
                set_mla_kv_buffer_triton(
                    kv_buffer, locations, cache_k_nope, cache_k_rope
                )

            self.assertEqual(kv_buffer.shape[0] * dcp_size, global_tokens)
            expected = cache_k_nope[rank::dcp_size, 0, 0]
            torch.testing.assert_close(kv_buffer[:, 0, 0], expected)
            torch.testing.assert_close(kv_buffer[:, 0, 1], -expected)

    @patch(
        "sglang.srt.layers.attention.tokenspeed_mla_backend."
        "get_in_autotune_dummy_run",
    )
    @patch(
        "sglang.srt.layers.attention.tokenspeed_mla_backend.get_parallel",
    )
    def test_tokenspeed_skips_decode_kernel_for_autotune_dummy(
        self, get_parallel, in_autotune_dummy_run
    ):
        in_autotune_dummy_run.return_value = True
        get_parallel.return_value.dcp_enabled = True
        backend = object.__new__(TokenspeedMLABackend)
        backend.q_data_type = torch.bfloat16
        q = torch.randn(3, 8 * 512, dtype=torch.bfloat16)
        layer = Mock(tp_q_head_num=8, v_head_dim=512)

        output, lse = backend.forward_decode(
            q=q,
            k=Mock(),
            v=Mock(),
            layer=layer,
            forward_batch=Mock(),
        )

        self.assertEqual(output.shape, (3, 8 * 512))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(lse.shape, (3, 8))
        self.assertEqual(lse.dtype, torch.float32)
        self.assertTrue(torch.count_nonzero(output).item() == 0)
        self.assertTrue(torch.count_nonzero(lse).item() == 0)
        get_parallel.assert_called_once_with()

    @patch(
        "sglang.srt.models.kimi_k3.prepare_decode_context_parallel_metadata",
    )
    def test_text_model_delegates_dcp_metadata_to_planner(self, planner):
        planner.return_value = object()
        kwargs = _metadata_kwargs()
        model = object.__new__(KimiK3LinearForCausalLM)

        result = model.prepare_context_parallel_metadata_for_dcp(**kwargs)

        self.assertIs(result, planner.return_value)
        planner.assert_called_once_with(**kwargs)

    def test_multimodal_wrapper_delegates_dcp_metadata_to_text_model(self):
        expected = object()
        language_model = Mock()
        language_model.prepare_context_parallel_metadata_for_dcp.return_value = expected
        model = object.__new__(KimiK3ForConditionalGeneration)
        object.__setattr__(model, "language_model", language_model)
        kwargs = _metadata_kwargs()

        result = model.prepare_context_parallel_metadata_for_dcp(**kwargs)

        self.assertIs(result, expected)
        language_model.prepare_context_parallel_metadata_for_dcp.assert_called_once_with(
            **kwargs
        )


if __name__ == "__main__":
    unittest.main()
