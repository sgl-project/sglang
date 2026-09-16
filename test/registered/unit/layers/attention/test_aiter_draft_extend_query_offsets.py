"""AITER eager draft-extend must not reuse earlier prefill query offsets."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import aiter_backend
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend, ForwardMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAiterDraftExtendQueryOffsets(unittest.TestCase):
    def _run_prefill_kernel(self, mode, metadata_offsets):
        backend = AiterAttnBackend.__new__(AiterAttnBackend)
        backend.qo_indptr = torch.tensor([0, 4096, 8070], dtype=torch.int32)
        backend.forward_metadata = ForwardMetadata(
            kv_indptr=torch.tensor([0, 8, 16], dtype=torch.int32),
            kv_indices=torch.arange(16, dtype=torch.int32),
            qo_indptr=metadata_offsets,
            kv_last_page_len=None,
            max_q_len=2,
            max_kv_len=8,
        )
        backend.kv_cache_dtype = torch.bfloat16
        backend.input_dtype = torch.bfloat16
        backend.use_mla = False
        backend._use_unified_verify = False
        backend.kv_cache_is_vectorized_5d = False
        backend.use_triton_unified_attention = False
        kv = torch.zeros((16, 1, 4), dtype=torch.bfloat16)
        backend.token_to_kv_pool = SimpleNamespace(get_kv_buffer=lambda _: (kv, kv))
        layer = SimpleNamespace(
            logit_cap=0.0,
            is_cross_attention=False,
            sliding_window_size=-1,
            tp_q_head_num=1,
            head_dim=4,
            qk_head_dim=4,
            v_head_dim=4,
            layer_id=0,
        )
        batch = SimpleNamespace(
            forward_mode=mode,
            batch_size=2,
            attn_attend_prefix_cache=False,
            out_cache_loc=torch.tensor([12, 13, 14, 15]),
        )
        query_tokens = int(
            metadata_offsets[-1]
            if metadata_offsets is not None
            else backend.qo_indptr[-1]
        )
        q = torch.zeros((query_tokens, 1, 4), dtype=torch.bfloat16)
        with (
            patch.object(aiter_backend, "is_gfx95_supported", return_value=False),
            patch.object(
                aiter_backend,
                "mha_batch_prefill_func",
                return_value=q.clone(),
                create=True,
            ) as attention,
        ):
            result = backend.forward_extend(q, None, None, layer, batch)
        self.assertEqual(result.shape, (query_tokens, 4))
        return attention.call_args.args[3], backend.qo_indptr

    def test_draft_extend_uses_current_query_offsets_after_long_prefill(self):
        # #36915 made eager draft-extend own fresh metadata. The earlier
        # prefill buffer can still describe thousands of rows, while this
        # forward has just two draft tokens per request.
        current = torch.tensor([0, 2, 4], dtype=torch.int32)
        passed, stale = self._run_prefill_kernel(ForwardMode.DRAFT_EXTEND_V2, current)
        torch.testing.assert_close(passed, current)
        self.assertEqual(int(passed[-1]), 4)
        self.assertEqual(int(stale[-1]), 8070)

    def test_normal_prefill_retains_legacy_query_offsets(self):
        passed, expected = self._run_prefill_kernel(ForwardMode.EXTEND, None)
        torch.testing.assert_close(passed, expected)


if __name__ == "__main__":
    unittest.main()
