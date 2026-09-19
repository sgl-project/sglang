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
buffers instead; those keep their plain view.

CPU-only.

    python -m pytest test/registered/unit/layers/attention/test_trtllm_mha_paged_kv.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _strided_rows(num_slots, row_shape, slot_stride):
    # Distinct values, so equal values mean equal addresses.
    backing = torch.arange(num_slots * slot_stride, dtype=torch.float32)
    return backing.as_strided(
        (num_slots, *row_shape), (slot_stride, *torch.empty(row_shape).stride())
    )


class TestTRTLLMHAPagedKV(unittest.TestCase):
    def test_reshape_keeps_slot_stride_for_packed_and_flat_rows(self):
        H, D, pages = 4, 32, 3
        slot_stride = 3 * H * D
        layer = SimpleNamespace(tp_k_head_num=H, tp_v_head_num=H)
        # 16 is the smallest page size trtllm_mha runs at; 1 is the case a
        # view gets wrong.
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

    def test_reshape_keeps_the_hnd_pool_view(self):
        H, D, pages, page_size = 4, 32, 3, 16
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.page_size = page_size
        layer = SimpleNamespace(tp_k_head_num=H, tp_v_head_num=H)
        k = torch.arange(pages * H * page_size * D, dtype=torch.float32).view(
            pages, H, page_size, D
        )
        v = k + 1
        outs = backend._reshape_paged_kv_cache(k, v, layer, D)
        for buf, out in zip((k, v), outs):
            want = buf.view(-1, page_size, H, D).permute(0, 2, 1, 3)
            self.assertEqual(out.shape, want.shape)
            self.assertEqual(out.stride(), want.stride())
            self.assertEqual(out.data_ptr(), want.data_ptr())


if __name__ == "__main__":
    unittest.main()
