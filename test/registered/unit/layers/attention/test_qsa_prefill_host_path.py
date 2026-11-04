"""QSA prefill host path: host lists replace device reads with the same values."""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    _last_rows_of_runs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _device_seed_rows(extend_lens, pool_ids, num_rows):
    """capture_mtp_sparse_indices' device path: is_last.nonzero() over the row layout."""
    row_to_req = torch.repeat_interleave(
        torch.arange(len(extend_lens)), torch.tensor(extend_lens, dtype=torch.long)
    )
    rows = torch.tensor(pool_ids, dtype=torch.long)[row_to_req[:num_rows]]
    is_last = torch.ones_like(rows, dtype=torch.bool)
    if rows.numel() > 1:
        is_last[:-1] = rows[:-1] != rows[1:]
    return is_last.nonzero().flatten().tolist()


def _random_case(rng):
    bs = rng.randint(1, 12)
    extend_lens = [
        rng.choice([0, 1, 1, 2, 3, 5, 9, 64, 700]) if rng.random() < 0.9 else 0
        for _ in range(bs)
    ]
    if rng.random() < 0.7:
        pool_ids = rng.sample(range(1, 4096), bs)
    else:
        # Equal ids side by side: runs merge on the device path.
        pool_ids = [rng.randint(0, 3) for _ in range(bs)]
    total = sum(extend_lens)
    r = rng.random()
    if r < 0.6:
        num_rows = total
    elif r < 0.8:
        num_rows = rng.randint(0, total) if total else 0
    else:
        # Padded query rows past the layout.
        num_rows = total + rng.randint(1, 64)
    return extend_lens, pool_ids, num_rows


class TestSeedRowsFromHost(CustomTestCase):
    def test_matches_device_path_on_random_layouts(self):
        rng = random.Random(0)
        for _ in range(20000):
            extend_lens, pool_ids, num_rows = _random_case(rng)
            self.assertEqual(
                _last_rows_of_runs(
                    extend_lens=extend_lens, pool_ids=pool_ids, num_rows=num_rows
                ),
                _device_seed_rows(extend_lens, pool_ids, num_rows),
                (extend_lens, pool_ids, num_rows),
            )

    def test_served_layout_is_each_requests_last_row(self):
        rng = random.Random(1)
        for _ in range(500):
            bs = rng.randint(1, 40)
            extend_lens = [rng.randint(1, 16384 // bs) for _ in range(bs)]
            pool_ids = rng.sample(range(1, 640), bs)
            want = [sum(extend_lens[: i + 1]) - 1 for i in range(bs)]
            for num_rows in (sum(extend_lens), sum(extend_lens) + 7):
                self.assertEqual(
                    _last_rows_of_runs(
                        extend_lens=extend_lens, pool_ids=pool_ids, num_rows=num_rows
                    ),
                    want,
                )


def _backend(*, host_path=True):
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend._prefill_host_path = host_path
    backend._mtp_shared_sparse_indices = None
    backend._mtp_anchor_host_src = None
    return backend


def _forward_batch(extend_lens, pool_ids):
    return SimpleNamespace(
        extend_seq_lens_cpu=list(extend_lens),
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
        req_pool_indices_cpu=torch.tensor(pool_ids, dtype=torch.int64),
        forward_mode=SimpleNamespace(is_draft_extend_v2=lambda: False),
    )


class TestHostExtendRows(CustomTestCase):
    def test_output_size_equals_the_device_total(self):
        extend_lens = [3, 0, 17, 1]
        fb = _forward_batch(extend_lens, [5, 6, 7, 8])
        rows = _backend()._host_extend_rows(
            forward_batch=fb, extend_seq_lens=fb.extend_seq_lens, batch_size=4
        )
        arange = torch.arange(4, dtype=torch.int32)
        lens = fb.extend_seq_lens.to(torch.long)
        self.assertTrue(
            torch.equal(
                torch.repeat_interleave(arange, lens, output_size=rows),
                torch.repeat_interleave(arange, lens),
            )
        )

    def test_mismatch_or_capture_keeps_the_device_read(self):
        fb = _forward_batch([3, 4], [5, 6])
        # Host list shorter than the device tensor.
        self.assertIsNone(
            _backend()._host_extend_rows(
                forward_batch=fb,
                extend_seq_lens=torch.tensor([3, 4, 5]),
                batch_size=3,
            )
        )
        fb.extend_seq_lens_cpu = None
        self.assertIsNone(
            _backend()._host_extend_rows(
                forward_batch=fb, extend_seq_lens=fb.extend_seq_lens, batch_size=2
            )
        )
        fb = _forward_batch([3, 4], [5, 6])
        with patch.object(qsa_backend, "_stream_capturing", lambda: True):
            self.assertIsNone(
                _backend()._host_extend_rows(
                    forward_batch=fb, extend_seq_lens=fb.extend_seq_lens, batch_size=2
                )
            )


class _CapturingState:
    def __init__(self):
        self.calls = []

    def capture(self, topk, req_rows, lens, layer_id):
        self.calls.append((topk.clone(), req_rows.clone(), lens.clone(), layer_id))


def _backend_and_metadata(extend_lens, pool_ids, *, host_path=True):
    backend = _backend(host_path=host_path)
    backend._mtp_shared_sparse_indices = _CapturingState()
    req_pool_indices = torch.tensor(pool_ids, dtype=torch.int64)
    row_to_req = torch.repeat_interleave(
        torch.arange(len(extend_lens), dtype=torch.int32),
        torch.tensor(extend_lens, dtype=torch.long),
    )
    seqlens_expanded = torch.arange(row_to_req.numel(), dtype=torch.int32) + 100
    metadata = SimpleNamespace(
        is_cuda_graph=False,
        req_pool_indices=req_pool_indices,
        get_token_to_batch_idx=lambda: row_to_req,
        get_seqlens_expanded=lambda: seqlens_expanded,
    )
    return backend, metadata


def _capture_rows(extend_lens, pool_ids, num_rows, *, record_host, host_path=True):
    backend, metadata = _backend_and_metadata(
        extend_lens, pool_ids, host_path=host_path
    )
    fb = _forward_batch(extend_lens, pool_ids)
    if record_host:
        backend._mtp_anchor_host_src = backend._mtp_anchor_host_source(
            forward_batch=fb, indexer_metadata=metadata, batch_size=len(pool_ids)
        )
    topk = torch.arange(num_rows * 4, dtype=torch.int32).reshape(num_rows, 4)
    backend.capture_mtp_sparse_indices(topk, fb, layer_id=3, metadata=metadata)
    return backend._mtp_shared_sparse_indices.calls


class TestCaptureSeedRows(CustomTestCase):
    def test_host_and_device_seed_rows_capture_the_same_rows(self):
        for extend_lens, pool_ids, num_rows in [
            ([5, 3, 7], [11, 4, 9], 15),
            ([5, 0, 7], [11, 4, 9], 12),
            ([5, 3, 7], [4, 4, 9], 15),
            ([5, 3, 7], [11, 4, 9], 6),
            ([1, 1, 1, 1], [2, 3, 4, 5], 4),
        ]:
            host = _capture_rows(extend_lens, pool_ids, num_rows, record_host=True)
            device = _capture_rows(extend_lens, pool_ids, num_rows, record_host=False)
            self.assertEqual(len(host), 1)
            for h, d in zip(host[0][:3], device[0][:3]):
                self.assertTrue(torch.equal(h, d), (extend_lens, pool_ids, num_rows))

    def test_host_rows_apply_only_to_the_metadata_they_were_built_for(self):
        backend, metadata = _backend_and_metadata([2, 2], [1, 2])
        fb = _forward_batch([2, 2], [1, 2])
        backend._mtp_anchor_host_src = backend._mtp_anchor_host_source(
            forward_batch=fb, indexer_metadata=metadata, batch_size=2
        )
        self.assertEqual(backend._mtp_anchor_host_rows(metadata, 4), [1, 3])
        _, other = _backend_and_metadata([2, 2], [1, 2])
        self.assertIsNone(backend._mtp_anchor_host_rows(other, 4))
        with patch.object(qsa_backend, "_stream_capturing", lambda: True):
            self.assertIsNone(backend._mtp_anchor_host_rows(metadata, 4))

    def test_host_source_refuses_lists_that_do_not_line_up(self):
        fb = _forward_batch([2, 2], [1, 2])
        fb.req_pool_indices_cpu = torch.tensor([1], dtype=torch.int64)
        self.assertIsNone(
            QwenSparseAttnBackend._mtp_anchor_host_source(
                forward_batch=fb, indexer_metadata=None, batch_size=2
            )
        )


class TestHostPathSwitch(CustomTestCase):
    """SGLANG_ENABLE_QSA_PREFILL_HOST_PATH: off by default; off restores every
    device read."""

    def test_defaults_off(self):
        self.assertIs(envs.SGLANG_ENABLE_QSA_PREFILL_HOST_PATH.default, False)

    def test_off_keeps_the_device_reads(self):
        fb = _forward_batch([3, 4], [5, 6])
        self.assertIsNone(
            _backend(host_path=False)._host_extend_rows(
                forward_batch=fb, extend_seq_lens=fb.extend_seq_lens, batch_size=2
            )
        )
        extend_lens, pool_ids, num_rows = [5, 3, 7], [11, 4, 9], 15
        device = _capture_rows(extend_lens, pool_ids, num_rows, record_host=False)
        # A host source left from an earlier batch must not be used when off.
        with patch.object(
            qsa_backend, "async_h2d", side_effect=AssertionError("host path used")
        ):
            off = _capture_rows(
                extend_lens, pool_ids, num_rows, record_host=True, host_path=False
            )
        for a, b in zip(off[0][:3], device[0][:3]):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
