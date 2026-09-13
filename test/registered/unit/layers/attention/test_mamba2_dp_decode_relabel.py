"""Mamba metadata for a DP decode batch relabeled as a 1-token extend.

Under MAX_LEN padding ``prepare_mlp_sync_batch`` relabels a decode rank as an
EXTEND batch so every DP rank hands the collectives one shape, padding each
row to a 1-token extend. Read literally, those rows look like a batch of
single-token prefills: ``num_prefills`` becomes the row count, ``num_decodes``
becomes 0, the decode state tracking is skipped, and the prefill path
dereferences ``mamba_track_seqlens``, which a decode batch never builds.

Pure dataclass logic -- CPU only.
"""

import unittest

import torch

from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHUNK_SIZE = 64


def _forward_metadata(num_rows: int) -> ForwardMetadata:
    return ForwardMetadata(
        query_start_loc=torch.arange(num_rows + 1, dtype=torch.int32),
        mamba_cache_indices=torch.arange(num_rows, dtype=torch.int32),
    )


def _relabeled_decode_batch(bs: int) -> ForwardBatch:
    """What prepare_mlp_sync_batch leaves behind for a decoding DP rank whose
    peer is prefilling: EXTEND mode, one token per row, rows untouched."""
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=bs,
        input_ids=torch.arange(bs),
        req_pool_indices=torch.arange(bs),
        seq_lens=torch.full((bs,), 8),
        out_cache_loc=torch.arange(bs),
        seq_lens_sum=8 * bs,
        positions=torch.full((bs,), 7),
        extend_num_tokens=bs,
        extend_seq_lens=torch.ones(bs, dtype=torch.int32),
        extend_prefix_lens=torch.full((bs,), 7),
        extend_seq_lens_cpu=[1] * bs,
        extend_prefix_lens_cpu=[7] * bs,
    )
    fb._original_forward_mode = ForwardMode.DECODE
    fb._original_batch_size = bs
    return fb


class TestMamba2DpDecodeRelabel(CustomTestCase):
    def test_relabeled_decode_builds_decode_metadata(self):
        bs = 3
        metadata = Mamba2Metadata.prepare_mixed(
            forward_metadata=_forward_metadata(bs),
            chunk_size=CHUNK_SIZE,
            forward_batch=_relabeled_decode_batch(bs),
        )

        # Read as prefills these would be num_prefills=3 / num_decodes=0, which
        # skips the decode state update and leaves the rows untracked.
        self.assertEqual(metadata.num_decodes, bs)
        self.assertEqual(metadata.num_prefills, 0)
        self.assertEqual(metadata.num_prefill_tokens, 0)
        self.assertIsNone(metadata.mixed_metadata)

    def test_real_extend_still_builds_prefill_metadata(self):
        # The relabel is the only thing that may be read through: an unrelabeled
        # extend must keep the prefill path, or real prefills lose their chunk
        # metadata and get a decode state update instead.
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            input_ids=torch.arange(7),
            req_pool_indices=torch.arange(2),
            seq_lens=torch.tensor([4, 3]),
            out_cache_loc=torch.arange(7),
            seq_lens_sum=7,
            positions=torch.arange(7),
            extend_num_tokens=7,
            extend_seq_lens=torch.tensor([4, 3], dtype=torch.int32),
            extend_prefix_lens=torch.zeros(2, dtype=torch.int64),
            extend_seq_lens_cpu=[4, 3],
            extend_prefix_lens_cpu=[0, 0],
        )
        self.assertIsNone(fb._original_forward_mode)

        metadata = Mamba2Metadata.prepare_mixed(
            forward_metadata=ForwardMetadata(
                query_start_loc=torch.tensor([0, 4, 7], dtype=torch.int32),
                mamba_cache_indices=torch.arange(2, dtype=torch.int32),
            ),
            chunk_size=CHUNK_SIZE,
            forward_batch=fb,
        )

        self.assertEqual(metadata.num_prefills, 2)
        self.assertEqual(metadata.num_prefill_tokens, 7)
        self.assertEqual(metadata.num_decodes, 0)
        self.assertIsNotNone(metadata.mixed_metadata)


class TestLogicalForwardMode(CustomTestCase):
    """``logical_forward_mode`` reads back only the decode relabel. The idle
    conversions share ``_original_forward_mode``, so a predicate that ignored
    the original mode would route a fabricated prefill row through the decode
    path.
    """

    def _batch(self, mode: ForwardMode, original: ForwardMode) -> ForwardBatch:
        fb = ForwardBatch(
            forward_mode=mode,
            batch_size=1,
            input_ids=torch.arange(1),
            req_pool_indices=torch.arange(1),
            seq_lens=torch.tensor([8]),
            out_cache_loc=torch.arange(1),
            seq_lens_sum=8,
            positions=torch.tensor([7]),
        )
        fb._original_forward_mode = original
        return fb

    def test_decode_relabel_is_read_back(self):
        fb = self._batch(ForwardMode.EXTEND, ForwardMode.DECODE)
        self.assertEqual(fb.logical_forward_mode, ForwardMode.DECODE)

    def test_idle_relabels_and_unrelabeled_batches_keep_their_mode(self):
        cases = {
            # idle rank fabricating a dummy prefill row
            "idle_to_extend": (ForwardMode.EXTEND, ForwardMode.IDLE),
            # idle rank joining a speculative verify
            "idle_to_target_verify": (ForwardMode.TARGET_VERIFY, ForwardMode.IDLE),
            "unrelabeled_extend": (ForwardMode.EXTEND, None),
            "unrelabeled_decode": (ForwardMode.DECODE, None),
        }
        for name, (mode, original) in cases.items():
            with self.subTest(name):
                self.assertEqual(self._batch(mode, original).logical_forward_mode, mode)


if __name__ == "__main__":
    unittest.main()
