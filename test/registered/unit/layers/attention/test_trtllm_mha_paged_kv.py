"""trtllm_mha's paged K/V views keep the pool's slot stride.

``_reshape_paged_kv_cache`` turns a per-layer ``[slots, ...]`` buffer into the
``[pages, heads, page_size, head_dim]`` layout the trtllm kernels take. Its
nvfp4 callers also hand it block-scale rows that arrive FLAT
(``[slots, heads * head_dim // 16]``), so the row has to be shaped before the
pages are split: shaping it afterwards, or splitting with ``.view()``,
re-derives the size-1 page dimension's stride from the row at
``page_size == 1``. On a strided buffer (the unified pool's slots are a whole
entry apart, wider than one row) that stride is then wrong.

An HND pool hands over page-major ``[pages, heads, page_size, head_dim]``
buffers instead, which are already the kernel layout: they must reach the
kernel as themselves, head for head and token for token.

CPU-only.

    python -m pytest test/registered/unit/layers/attention/test_trtllm_mha_paged_kv.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _strided_rows(num_slots, row_shape, slot_stride):
    # Distinct values, so equal values mean equal addresses.
    backing = torch.arange(num_slots * slot_stride, dtype=torch.float32)
    return backing.as_strided(
        (num_slots, *row_shape), (slot_stride, *torch.empty(row_shape).stride())
    )


class TestTRTLLMHAPagedKV(unittest.TestCase):
    def test_graph_variants_keep_decode_prefix_sums_and_verify_row_map_alive(self):
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.device = "cpu"
        backend.max_num_pages = 4
        backend.speculative_num_draft_tokens = 3
        backend.use_sliding_window_kv_pool = False
        backend._swa_kv_pool = None
        backend.expand_encoder_only_verify = True
        backend.page_size = 1
        backend.req_to_token = torch.zeros((4, 4), dtype=torch.int32)
        backend._swa_full_to_swa_mapping = None
        backend.runner_ids = SimpleNamespace(is_translating=False)
        backend.init_cuda_graph_state(4, 12)

        def build(num_tokens, forward_mode):
            backend._build_cuda_graph_metadata(
                2, num_tokens, forward_mode, None, "cpu", ragged_verify_layout=None
            )

        build(2, ForwardMode.DECODE)
        captured = backend.decode_cuda_graph_metadata[2]
        build(2, ForwardMode.DECODE)

        def update(**kwargs):
            lengths = kwargs["seq_lens"].to(torch.int32)
            kwargs["cache_seqlens"].copy_(lengths)
            kwargs["cu_seqlens_k"][1:].copy_(lengths.cumsum(0))

        with patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.update_trtllm_mha_graph_metadata",
            side_effect=update,
        ):
            backend._apply_cuda_graph_metadata(
                2, torch.tensor([0, 1]), torch.tensor([7, 11]), ForwardMode.DECODE, None
            )
        self.assertEqual(
            captured.cu_seqlens_q.data_ptr(),
            backend.decode_cuda_graph_metadata[2].cu_seqlens_q.data_ptr(),
        )
        torch.testing.assert_close(
            captured.cu_seqlens_k, torch.tensor([0, 7, 18], dtype=torch.int32)
        )

        build(6, ForwardMode.TARGET_VERIFY)
        captured_map = backend.target_verify_metadata[2].encoder_row_map
        build(6, ForwardMode.TARGET_VERIFY)
        self.assertEqual(
            captured_map.data_ptr(),
            backend.target_verify_metadata[2].encoder_row_map.data_ptr(),
        )
        torch.testing.assert_close(captured_map, torch.tensor([0, 0, 0, 1, 1, 1]))

    def test_reshape_keeps_slot_stride_for_packed_and_flat_rows(self):
        H, D, pages = 4, 32, 3
        slot_stride = 3 * H * D
        layer = SimpleNamespace(tp_k_head_num=H, tp_v_head_num=H)
        # 16 is the smallest page size trtllm_mha runs at; at 1, `.view()`
        # would give the size-1 page dim the row stride.
        for page_size in (1, 16):
            backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
            backend.page_size = page_size
            # (row shape handed in, head_dim argument): nvfp4 packed K/V rows,
            # then the flat block-scale rows.
            for row_shape, head_dim in (
                ((H, D // 2), D // 2),
                ((H * D // 16,), D // 16),
            ):
                with self.subTest(page_size=page_size, row_shape=row_shape):
                    k = _strided_rows(pages * page_size, row_shape, slot_stride)
                    v = _strided_rows(pages * page_size, row_shape, slot_stride)
                    outs = backend._reshape_paged_kv_cache(k, v, layer, head_dim)
                    for flat, out in zip((k, v), outs):
                        self.assertEqual(
                            tuple(out.shape), (pages, H, page_size, head_dim)
                        )
                        self.assertEqual(out.stride(0), page_size * slot_stride)
                        self.assertEqual(out.stride(2), slot_stride)
                        for token in range(pages * page_size):
                            page, slot = divmod(token, page_size)
                            self.assertTrue(
                                torch.equal(
                                    out[page, :, slot],
                                    flat[token].reshape(H, head_dim),
                                )
                            )

    def test_reshape_hands_the_hnd_pool_buffer_through(self):
        H, D, pages, page_size = 4, 32, 3, 16
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.page_size = page_size
        layer = SimpleNamespace(tp_k_head_num=H, tp_v_head_num=H)
        # Distinct values, so equal values mean equal (page, head, token).
        k = torch.arange(pages * H * page_size * D, dtype=torch.float32).view(
            pages, H, page_size, D
        )
        v = k + 0.5
        outs = backend._reshape_paged_kv_cache(k, v, layer, D)
        for buf, out in zip((k, v), outs):
            self.assertEqual(out.shape, buf.shape)
            self.assertEqual(out.stride(), buf.stride())
            self.assertEqual(out.data_ptr(), buf.data_ptr())
            self.assertTrue(torch.equal(out, buf))


if __name__ == "__main__":
    unittest.main()
