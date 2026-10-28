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

from sglang.srt.layers.attention import trtllm_mha_backend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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


class TestTRTLLMHAFullAttnKVSplit(CustomTestCase):
    """SGLANG_TRTLLM_MHA_FULL_ATTN_KV_SPLITS scopes the SM count trtllm-gen
    decode reads, which sets its KV splits per (request, KV head) to
    sm_count // (batch * kv_heads) and its grid to that many CTAs."""

    SMS, KV_HEADS, SPLITS = 148, 4, 8

    def _backend(self, *, min_bs=12, max_split_ctas=1 << 20):
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend._full_attn_kv_splits = self.SPLITS
        backend._full_attn_kv_split_min_bs = min_bs
        backend._device_sm_count = self.SMS
        backend._full_attn_kv_split_max_ctas = max_split_ctas
        return backend

    def _flashinfer(self, decode):
        return patch.object(
            trtllm_mha_backend,
            "flashinfer",
            SimpleNamespace(decode=decode),
            create=True,
        )

    def _kv_cache(self):
        k_cache = torch.empty(2, self.KV_HEADS, 16, 8)  # HND pages
        return (k_cache, k_cache)

    def _scope_sm_count(self, *, window_left, batch_size, **backend_kwargs):
        backend = self._backend(**backend_kwargs)
        decode = SimpleNamespace(get_device_sm_count=lambda device: self.SMS)
        with self._flashinfer(decode):
            with backend._full_attn_kv_split_scope(
                window_left=window_left,
                batch_size=batch_size,
                kv_cache=self._kv_cache(),
            ):
                seen = decode.get_device_sm_count(None)
            self.assertEqual(decode.get_device_sm_count(None), self.SMS)
        return seen

    def test_verify_launches_inside_the_scope_sized_by_requests(self):
        """A fixed-q-len verify launches q_len_per_req query rows per request and
        FlashInfer counts requests, so the launch must run inside the scope and
        size it by query rows // q_len_per_req."""
        backend = self._backend()
        backend.decode_seq_len_splits = 1
        backend.q_data_type = torch.bfloat16
        backend.max_context_len = 1024
        backend.workspace_buffer = backend._multi_ctas_kv_counter_buffer = None
        seen = []
        decode = SimpleNamespace(get_device_sm_count=lambda device: self.SMS)

        def launch(**kwargs):
            seen.append(decode.get_device_sm_count(None))
            return kwargs["query"]

        decode.trtllm_batch_decode_with_kv_cache = launch
        bs, q_len_per_req = 32, 8
        with self._flashinfer(decode):
            backend._run_fixed_q_len_decode(
                torch.empty(bs * q_len_per_req, 16, 8),
                self._kv_cache(),
                torch.zeros(bs, 2, dtype=torch.int32),
                torch.ones(bs, dtype=torch.int32),
                bmm1_scale=1.0,
                bmm2_scale=1.0,
                window_left=-1,
                sinks=None,
                q_len_per_req=q_len_per_req,
            )
        self.assertEqual(seen, [self.SPLITS * self.KV_HEADS * bs])

    def test_splits_only_full_attention_batches_flashinfer_runs_unsplit(self):
        # Full attention at bs 32: 4 * 32 request-heads exceed the SMs, so
        # FlashInfer would run one CTA each; the scope makes it 8.
        self.assertEqual(
            self._scope_sm_count(window_left=-1, batch_size=32),
            self.SPLITS * self.KV_HEADS * 32,
        )
        # Sliding-window layers, batches under MIN_BS, and batches FlashInfer
        # already splits as finely keep the device SM count.
        self.assertEqual(self._scope_sm_count(window_left=127, batch_size=32), self.SMS)
        self.assertEqual(self._scope_sm_count(window_left=-1, batch_size=8), self.SMS)
        # At bs 4 FlashInfer already gives 148 // 16 = 9 splits.
        self.assertEqual(
            self._scope_sm_count(window_left=-1, batch_size=4, min_bs=1), self.SMS
        )

    def test_split_ctas_stay_within_the_workspace_and_counters(self):
        # 4 * 32 base CTAs and room for 640 split CTAs: 5 splits, not 8.
        self.assertEqual(
            self._scope_sm_count(window_left=-1, batch_size=32, max_split_ctas=640),
            5 * self.KV_HEADS * 32,
        )
        # Room for under 2 splits: no override.
        self.assertEqual(
            self._scope_sm_count(window_left=-1, batch_size=32, max_split_ctas=255),
            self.SMS,
        )

    def test_scope_restores_the_sm_count_after_an_error(self):
        decode = SimpleNamespace(get_device_sm_count=lambda device: self.SMS)
        with self._flashinfer(decode):
            with self.assertRaises(RuntimeError):
                with trtllm_mha_backend._trtllm_decode_sm_count(1024):
                    raise RuntimeError("kernel launch failed")
            self.assertEqual(decode.get_device_sm_count(None), self.SMS)


if __name__ == "__main__":
    unittest.main()
