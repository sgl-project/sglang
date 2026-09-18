"""Check the query boundaries passed to AITER's paged prefill kernel."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import aiter_backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAiterEagerDraftExtend(unittest.TestCase):
    def _check_query_boundaries(self, lengths, use_draft_metadata):
        batch_size = len(lengths)
        expected = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).int()
        backend = object.__new__(aiter_backend.AiterAttnBackend)
        backend.use_mla = False
        backend.use_triton_unified_attention = False
        backend._use_unified_verify = False
        backend.kv_cache_is_vectorized_5d = False
        backend.kv_cache_dtype = backend.input_dtype = torch.float32
        # A preceding long prefill leaves a reusable buffer whose boundaries
        # would write beyond the output of a short draft-extend batch.
        backend.qo_indptr = (
            torch.arange(batch_size + 1, dtype=torch.int32) * 7000
            if use_draft_metadata
            else torch.cat((expected, torch.tensor([999], dtype=torch.int32)))
        )
        backend.forward_metadata = aiter_backend.ForwardMetadata(
            kv_indptr=torch.arange(batch_size + 1, dtype=torch.int32) * 16,
            kv_indices=torch.arange(batch_size * 16, dtype=torch.int32),
            qo_indptr=expected if use_draft_metadata else None,
            kv_last_page_len=None,
            max_q_len=max(lengths),
            max_kv_len=16,
        )
        cache = torch.zeros(batch_size * 16, 1, 4)
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=Mock(return_value=(cache, cache))
        )
        layer = SimpleNamespace(
            layer_id=0,
            head_dim=4,
            qk_head_dim=4,
            v_head_dim=4,
            tp_q_head_num=2,
            tp_k_head_num=1,
            tp_v_head_num=1,
            logit_cap=0.0,
            sliding_window_size=-1,
            is_cross_attention=False,
        )
        batch = SimpleNamespace(
            batch_size=batch_size,
            forward_mode=(
                ForwardMode.DRAFT_EXTEND_V2
                if use_draft_metadata
                else ForwardMode.EXTEND
            ),
            attn_attend_prefix_cache=False,
            out_cache_loc=None,
        )
        q = torch.zeros(sum(lengths), layer.tp_q_head_num * layer.head_dim)
        kv = torch.zeros(sum(lengths), layer.tp_k_head_num * layer.head_dim)

        def check_kernel_inputs(q, k, v, cu_seqlens_q, *args, **kwargs):
            torch.testing.assert_close(cu_seqlens_q, expected)
            self.assertEqual(int(cu_seqlens_q[-1]), q.shape[0])
            return torch.zeros_like(q)

        with (
            patch.object(aiter_backend, "is_gfx95_supported", return_value=False),
            patch.object(
                aiter_backend,
                "mha_batch_prefill_func",
                side_effect=check_kernel_inputs,
                create=True,
            ) as kernel,
        ):
            output = backend.forward_extend(
                q, kv, kv, layer, batch, save_kv_cache=False
            )
        kernel.assert_called_once()
        self.assertEqual(output.shape, q.shape)

    def test_eager_draft_extend_uses_current_batch_boundaries(self):
        for lengths in ([4], [4, 2], [4] * 20):
            with self.subTest(lengths=lengths):
                self._check_query_boundaries(lengths, use_draft_metadata=True)

    def test_regular_prefill_uses_reusable_boundaries(self):
        self._check_query_boundaries([4, 3], use_draft_metadata=False)


if __name__ == "__main__":
    unittest.main()
