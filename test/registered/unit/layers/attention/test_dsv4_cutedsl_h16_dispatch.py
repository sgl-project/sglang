import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDSV4CutedslH16Dispatch(CustomTestCase):
    def test_eager_selection_and_fallbacks(self):
        from sglang.srt.layers.attention import deepseek_v4_backend as mod

        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            _original_forward_mode=None,
            tbo_parent_token_range=None,
        )
        backend = SimpleNamespace(
            trtllm_attn=False,
            dsv4_prefill_backend="cutedsl_h16",
            forward_metadata=SimpleNamespace(late_layer_tail=None),
        )
        with (
            mock.patch.multiple(
                mod,
                _is_cuda=False,
                get_platform=mock.Mock(return_value=SimpleNamespace(is_sm120=False)),
                is_cp_active=mock.Mock(return_value=False),
                is_in_breakable_cuda_graph=mock.Mock(return_value=False),
                is_in_tc_piecewise_cuda_graph=mock.Mock(return_value=False),
                get_tc_piecewise_forward_context=mock.Mock(return_value=None),
            ),
            mock.patch.object(
                mod.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL, "get", return_value=False
            ),
        ):

            def select():
                return mod.DeepseekV4AttnBackend._use_sparse_prefill(
                    backend, batch, num_qo_tokens=1
                )

            self.assertTrue(select())
            for target, field, value in (
                (backend, "dsv4_prefill_backend", "auto"),
                (backend, "trtllm_attn", True),
                (backend.forward_metadata, "late_layer_tail", object()),
                (batch, "forward_mode", ForwardMode.DRAFT_EXTEND_V2),
                (batch, "_original_forward_mode", ForwardMode.TARGET_VERIFY),
                (batch, "tbo_parent_token_range", (0, 1)),
                (mod.is_cp_active, "return_value", True),
                (mod.is_in_breakable_cuda_graph, "return_value", True),
                (mod.is_in_tc_piecewise_cuda_graph, "return_value", True),
                (
                    mod.get_tc_piecewise_forward_context,
                    "return_value",
                    SimpleNamespace(full_graph=True),
                ),
            ):
                with self.subTest(field=field), mock.patch.object(target, field, value):
                    self.assertFalse(select())
            with (
                mock.patch.object(mod, "_is_cuda", True),
                mock.patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=True
                ),
            ):
                self.assertFalse(select())

    def test_fixed_c128_width(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _validate_dsv4_cutedsl_h16_combined_topk,
        )
        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            SparsePrefillChunkCache,
        )

        self.assertEqual(_validate_dsv4_cutedsl_h16_combined_topk(49152, 512), 512)
        for context, topk in ((8065 * 128, 512), (49152, 8065)):
            with self.assertRaisesRegex(ValueError, "8192"):
                _validate_dsv4_cutedsl_h16_combined_topk(context, topk)
        cache = object.__new__(SparsePrefillChunkCache)
        indices = torch.zeros((2, 384), dtype=torch.int32)
        lengths = torch.tensor([321, 384], dtype=torch.int32)
        cache.compressed = {
            128: SimpleNamespace(combined_indices=indices, combined_lens=lengths)
        }
        cache.c128_padded_combined_indices = {}
        padded, actual_lengths = cache.get_c128_combined(512)
        self.assertIs(actual_lengths, lengths)
        self.assertIs(cache.get_c128_combined(512)[0], padded)
        torch.testing.assert_close(padded[:, :384], indices)
        self.assertTrue(torch.all(padded[:, 384:] == -1))
        for width in (256, 450):
            with self.assertRaises(ValueError):
                cache.get_c128_combined(width)


if __name__ == "__main__":
    unittest.main()
