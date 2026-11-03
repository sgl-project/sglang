"""Hybrid (GDN) extend preparation stays off the host-device sync path.

Each piece runs under ``torch.cuda.set_sync_debug_mode("error")`` and must
produce exactly what the device-derived version produces: the FLA chunk tables
of an eager Triton extend, the conv-state track indices and the PLE side-state
slot resets.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    _uses_host_chunk_tables,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@contextmanager
def _no_sync():
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _metadata(query_start_loc, trusted=True):
    return SimpleNamespace(
        query_start_loc=query_start_loc,
        host_extend_lens_trusted=trusted,
        logical_num_tokens=int(query_start_loc[-1]),
        extend_chunk_tables=None,
    )


def _gdn_backend():
    backend = object.__new__(GDNAttnBackend)
    backend._host_chunk_tables = True
    backend.conv_states_shape = (1, 8, 4)
    backend.device = torch.device("cuda")
    return backend


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestHybridExtendHostSyncs(CustomTestCase):
    def test_extend_chunk_tables_match_device_derivation(self):
        backend = _gdn_backend()
        lens = [100, 3, 64, 129]
        query_start_loc = torch.tensor(
            [0, 100, 103, 167, 296], dtype=torch.int32, device="cuda"
        )
        metadata = _metadata(query_start_loc)
        batch = SimpleNamespace(extend_seq_lens_cpu=lens)
        with _no_sync():
            indices, offsets = backend._get_extend_chunk_tables(
                forward_batch=batch, forward_metadata=metadata, num_tokens=296
            )
        expected_indices = prepare_chunk_indices(query_start_loc, 64)
        expected_offsets = prepare_chunk_offsets(query_start_loc, 64)
        self.assertEqual(indices.dtype, expected_indices.dtype)
        self.assertEqual(offsets.dtype, expected_offsets.dtype)
        self.assertTrue(torch.equal(indices, expected_indices))
        self.assertTrue(torch.equal(offsets, expected_offsets))
        cached = backend._get_extend_chunk_tables(
            forward_batch=batch, forward_metadata=metadata, num_tokens=296
        )
        self.assertIs(cached[0], indices)

    def test_extend_chunk_tables_skip_sub_chunk_batches(self):
        backend = _gdn_backend()
        batch = SimpleNamespace(extend_seq_lens_cpu=[10, 20])
        metadata = _metadata(
            torch.tensor([0, 10, 30], dtype=torch.int32, device="cuda")
        )
        tables = backend._get_extend_chunk_tables(
            forward_batch=batch, forward_metadata=metadata, num_tokens=30
        )
        self.assertEqual(tables, (None, None))

    def test_extend_chunk_tables_need_trusted_host_lengths(self):
        # e.g. a two-batch-overlap half, whose host list does not describe it
        backend = _gdn_backend()
        batch = SimpleNamespace(extend_seq_lens_cpu=[100, 100])
        qsl = torch.tensor([0, 100, 200], dtype=torch.int32, device="cuda")
        for metadata, num_tokens in (
            (_metadata(qsl, trusted=False), 200),
            (_metadata(qsl), 256),
        ):
            tables = backend._get_extend_chunk_tables(
                forward_batch=batch, forward_metadata=metadata, num_tokens=num_tokens
            )
            self.assertEqual(tables, (None, None))

    def test_host_chunk_tables_only_for_cuda_triton_kernel(self):
        kernel = object.__new__(TritonGDNKernel)
        self.assertTrue(_uses_host_chunk_tables(kernel))
        self.assertFalse(_uses_host_chunk_tables(SimpleNamespace()))
        with patch.object(gdn_backend, "is_cpu", return_value=True):
            self.assertFalse(_uses_host_chunk_tables(kernel))

    def test_track_conv_indices_clamp_from_host_count(self):
        backend = _gdn_backend()
        query_start_loc = torch.tensor(
            [0, 100, 103, 167], dtype=torch.int32, device="cuda"
        )
        aligned = torch.tensor([64, 0, 64], dtype=torch.int32, device="cuda")
        batch = SimpleNamespace(mamba_track_aligned_lens=lambda: aligned)
        rows = torch.tensor([0, 1, 2], device="cuda")
        with _no_sync():
            indices = backend._init_track_conv_indices(
                query_start_loc, batch, num_tokens=167, track_mask_indices=rows
            )
        start = (query_start_loc[:-1] + aligned - 4).unsqueeze(-1)
        offsets = torch.arange(4, dtype=torch.int32, device="cuda")
        expected = (start + offsets).clamp(0, 166)
        self.assertTrue(torch.equal(indices, expected))

    def test_ple_slot_resets(self):
        slots = torch.tensor([3, 1, 3], dtype=torch.int32, device="cuda")
        conv = object.__new__(ShortConvPool)
        conv.conv_state = torch.ones(2, 6, 4, device="cuda")
        ngram = object.__new__(NGramPool)
        ngram.context = torch.zeros(6, 3, dtype=torch.long, device="cuda")
        ngram.eos_token_id = 7
        with _no_sync():
            conv.reset_slots(slots)
            ngram.reset_slots(slots)
        expected_conv = torch.ones(2, 6, 4, device="cuda")
        expected_conv[:, [1, 3]] = 0
        expected_context = torch.zeros(6, 3, dtype=torch.long, device="cuda")
        expected_context[[1, 3]] = 7
        self.assertTrue(torch.equal(conv.conv_state, expected_conv))
        self.assertTrue(torch.equal(ngram.context, expected_context))


if __name__ == "__main__":
    unittest.main()
