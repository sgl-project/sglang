"""Padded short paged-prefill extends must match the full prefill bit for bit."""

import unittest
from types import SimpleNamespace

import torch
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferIndicesUpdaterPrefill,
)
from sglang.srt.layers.attention.flashinfer_deterministic import (
    deterministic_prefill_min_q,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_HQ = 32
_HKV = 8
_DIM = 128
_DTYPE = torch.bfloat16
_MIN_Q = deterministic_prefill_min_q(_HQ, _HKV, _DIM)
_FIXED_SPLIT = 16384
_WORKSPACE_BYTES = 1 << 30
_DEVICE = "cuda"


def _indptr(lengths):
    out = [0]
    for length in lengths:
        out.append(out[-1] + length)
    return torch.tensor(out, dtype=torch.int32, device=_DEVICE)


def _kv_cache(kv_lens):
    total = sum(kv_lens)
    cache = torch.empty(total, 2, 1, _HKV, _DIM, dtype=_DTYPE, device=_DEVICE)
    cache[:, 0, 0] = torch.randn(total, _HKV, _DIM, dtype=_DTYPE, device=_DEVICE)
    cache[:, 1, 0] = torch.randn(total, _HKV, _DIM, dtype=_DTYPE, device=_DEVICE)
    return cache


def _run(q_rows, qo_lens, kv_cache, kv_lens):
    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=_DEVICE), "NHD"
    )
    wrapper.plan(
        _indptr(qo_lens),
        _indptr(kv_lens),
        torch.arange(sum(kv_lens), dtype=torch.int32, device=_DEVICE),
        torch.ones(len(kv_lens), dtype=torch.int32, device=_DEVICE),
        _HQ,
        _HKV,
        _DIM,
        1,
        causal=True,
        fixed_split_size=_FIXED_SPLIT,
        sm_scale=_DIM**-0.5,
        q_data_type=_DTYPE,
        kv_data_type=_DTYPE,
    )
    return wrapper.run(q_rows, kv_cache)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDeterministicPrefillPadding(CustomTestCase):
    def _check_short_extend(self, total_len, prefix_len):
        torch.manual_seed(0)
        kv_cache = _kv_cache([total_len])
        q = torch.randn(total_len, _HQ, _DIM, dtype=_DTYPE, device=_DEVICE)
        full = _run(q, [total_len], kv_cache, [total_len])

        extend = total_len - prefix_len
        self.assertLess(extend, _MIN_Q)
        padded_rows = torch.cat(
            [
                torch.zeros(_MIN_Q - extend, _HQ, _DIM, dtype=_DTYPE, device=_DEVICE),
                q[prefix_len:],
            ],
            dim=0,
        )
        padded = _run(padded_rows, [_MIN_Q], kv_cache, [total_len])

        self.assertTrue(
            torch.equal(full[prefix_len:], padded[_MIN_Q - extend :]),
            f"cached rows diverged: total={total_len} prefix={prefix_len}",
        )

    def test_single_token_extend(self):
        self._check_short_extend(101, 100)

    def test_three_token_extend(self):
        self._check_short_extend(98, 95)

    def test_mixed_batch_keeps_short_and_long_requests_exact(self):
        torch.manual_seed(1)
        kv_lens = [100, 20]
        kv_cache = _kv_cache(kv_lens)
        q_a = torch.randn(100, _HQ, _DIM, dtype=_DTYPE, device=_DEVICE)
        q_b = torch.randn(20, _HQ, _DIM, dtype=_DTYPE, device=_DEVICE)
        full = _run(torch.cat([q_a, q_b], dim=0), kv_lens, kv_cache, kv_lens)

        prefix_a, extend_a = 96, 4
        padded_rows = torch.cat(
            [
                torch.zeros(_MIN_Q - extend_a, _HQ, _DIM, dtype=_DTYPE, device=_DEVICE),
                q_a[prefix_a:],
                q_b,
            ],
            dim=0,
        )
        padded = _run(padded_rows, [_MIN_Q, 20], kv_cache, kv_lens)

        self.assertTrue(
            torch.equal(full[prefix_a:100], padded[_MIN_Q - extend_a : _MIN_Q]),
            "short request diverged in a mixed batch",
        )
        self.assertTrue(
            torch.equal(full[100:], padded[_MIN_Q:]),
            "long request diverged when batched with a padded request",
        )


def _updater(bs):
    def pack_read_stream(plan, **kw):
        out = kw["out"]
        offset = 0
        for length in kw["seq_lens"].tolist():
            out[offset : offset + length] = torch.arange(
                length, dtype=torch.int32, device=_DEVICE
            )
            offset += length

    updater = FlashInferIndicesUpdaterPrefill.__new__(FlashInferIndicesUpdaterPrefill)
    updater.attn_backend = SimpleNamespace(
        runner_ids=SimpleNamespace(
            reads_are_translated=False, pack_read_stream=pack_read_stream
        ),
        dq_paged_kernel_lens=None,
    )
    updater.num_qo_heads, updater.num_kv_heads, updater.head_dim = _HQ, _HKV, _DIM
    updater.q_data_type = updater.data_type = _DTYPE
    updater.kv_last_page_len = torch.ones(bs, dtype=torch.int32, device=_DEVICE)
    updater._swa_kv_pool = None
    updater.prefill_q_pad_enabled = True
    updater.prefill_cta_min_q = _MIN_Q
    return updater


def _paged_wrapper(bs, capacity, use_cuda_graph):
    kwargs = {}
    if use_cuda_graph:
        kwargs = dict(
            use_cuda_graph=True,
            qo_indptr_buf=torch.zeros(bs + 1, dtype=torch.int32, device=_DEVICE),
            paged_kv_indptr_buf=torch.zeros(bs + 1, dtype=torch.int32, device=_DEVICE),
            paged_kv_indices_buf=torch.zeros(
                capacity + 256, dtype=torch.int32, device=_DEVICE
            ),
            paged_kv_last_page_len_buf=torch.ones(
                bs, dtype=torch.int32, device=_DEVICE
            ),
        )
    return BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=_DEVICE),
        "NHD",
        backend="fa2",
        **kwargs,
    )


def _begin_forward(updater, wrapper, total_len, prefix_len):
    seq_lens = torch.tensor([total_len], dtype=torch.int32, device=_DEVICE)
    prefix_lens = torch.tensor([prefix_len], dtype=torch.int32, device=_DEVICE)
    kv_indptr = torch.zeros(2, dtype=torch.int32, device=_DEVICE)
    qo_indptr = torch.zeros(2, dtype=torch.int32, device=_DEVICE)
    updater.call_begin_forward(
        None,
        wrapper,
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
        seq_lens,
        total_len,
        seq_lens,
        prefix_lens,
        None,
        kv_indptr,
        qo_indptr,
        False,
        None,
        fixed_split_size=_FIXED_SPLIT,
        seq_lens_cpu=torch.tensor([total_len], dtype=torch.int32),
        extend_prefix_lens_cpu=[prefix_len],
        plan=None,
    )
    return qo_indptr


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDeterministicPrefillPaddingCudaGraph(CustomTestCase):
    def test_predicate_rejects_cuda_graph_wrapper(self):
        updater = _updater(1)
        common = dict(
            use_ragged=False,
            use_sliding_window_kv_pool=False,
            custom_kv_indices=None,
            cross_attention_custom_mask=None,
            multi_item_params=None,
        )
        self.assertTrue(
            updater._can_pad_prefill_q(
                wrapper_paged=SimpleNamespace(is_cuda_graph_enabled=False), **common
            )
        )
        self.assertFalse(
            updater._can_pad_prefill_q(
                wrapper_paged=SimpleNamespace(is_cuda_graph_enabled=True), **common
            )
        )

    def test_eager_wrapper_pads_short_extend(self):
        updater = _updater(1)
        qo_indptr = _begin_forward(updater, _paged_wrapper(1, 101, False), 101, 100)
        self.assertIsNotNone(updater.prefill_q_pad_gather_idx)
        self.assertEqual(qo_indptr[1].item(), _MIN_Q)

    def test_cuda_graph_wrapper_keeps_real_qo_indptr(self):
        updater = _updater(1)
        qo_indptr = _begin_forward(updater, _paged_wrapper(1, 101, True), 101, 100)
        self.assertIsNone(updater.prefill_q_pad_gather_idx)
        self.assertEqual(qo_indptr[1].item(), 1)


if __name__ == "__main__":
    unittest.main()
