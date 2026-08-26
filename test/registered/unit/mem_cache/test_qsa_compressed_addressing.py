"""Compressed write-plan contracts of DSV4-style QSA addressing.

The compressed cache is addressed as ``full_slot // compress_ratio`` over a
page-aligned full-KV allocator, and the per-forward write plan is computed
entirely on device from row lengths (no host loops, no device-to-host sync,
one code path for decode / verify / draft-extend / extend). These tests pin
the derived properties the compression kernels rely on, plus the fail-fast
guards that keep a misconfigured stack from silently corrupting shared
groups.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

RATIO, FULL_PAGE = 4, 64


def _page_aligned_table(num_rows: int, width: int) -> torch.Tensor:
    """token i of row r -> full page (r * 8 + i // 64), offset i % 64."""
    rows = torch.arange(num_rows, dtype=torch.long)[:, None]
    cols = torch.arange(width, dtype=torch.long)[None, :]
    return ((rows * 8 + cols // FULL_PAGE) * FULL_PAGE + cols % FULL_PAGE).to(
        torch.int32
    )


def _plan(*, lengths, table, extend_lens=None):
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = SimpleNamespace(qsa_compress_ratio=RATIO)
    is_decode = extend_lens is None
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: is_decode),
        extend_seq_lens=(
            None
            if extend_lens is None
            else torch.tensor(extend_lens, dtype=torch.int32)
        ),
        input_ids=torch.zeros(
            len(lengths) if extend_lens is None else sum(extend_lens)
        ),
    )
    return backend._qsa_build_write_plan(
        forward_batch=forward_batch,
        speculative_paged=False,
        token_slot_table=table,
        sequence_lengths=torch.tensor(lengths, dtype=torch.int32),
    )


class TestQsaCompressedWritePlan(CustomTestCase):
    def test_group_slot_is_any_raw_slot_over_ratio(self):
        """Derived property: with a page-aligned table every raw slot of a
        group floor-divides to the same compressed slot, so the plan can read
        the group's first slot and divide."""
        table = _page_aligned_table(2, 512)
        write_locs, group_positions, sequence_ids, member_rows = _plan(
            lengths=[256, 260], table=table
        )
        self.assertIsNone(member_rows)  # paged rows source the pending ring
        self.assertEqual(sequence_ids[:2].tolist(), [0, 1])
        self.assertEqual(group_positions[:2].tolist(), [255, 259])
        expected = [int(table[0, 252]) // RATIO, int(table[1, 256]) // RATIO]
        self.assertEqual(write_locs[:2].tolist(), expected)
        for offset in range(RATIO):
            self.assertEqual(int(table[0, 252 + offset]) // RATIO, expected[0])

    def test_non_boundary_rows_only_pad_the_plan(self):
        """A row whose length does not complete a group contributes nothing;
        the shape-derived capacity is padded with writes to the reserved slot
        0, which the compression kernels treat as an inert dump."""
        table = _page_aligned_table(3, 512)
        write_locs, _, _, _ = _plan(lengths=[253, 254, 255], table=table)
        self.assertEqual(write_locs.numel(), 3)  # capacity, not group count
        self.assertEqual(write_locs.tolist(), [0, 0, 0])

    def test_plan_is_compacted_in_row_order(self):
        """Mixed rows compact to the front in row order so the compression
        kernels can consume the plan without a per-row boundary test."""
        table = _page_aligned_table(4, 512)
        write_locs, group_positions, sequence_ids, _ = _plan(
            lengths=[255, 256, 257, 512], table=table
        )
        self.assertEqual(sequence_ids[:2].tolist(), [1, 3])
        self.assertEqual(group_positions[:2].tolist(), [255, 511])
        self.assertEqual(write_locs[2:].tolist(), [0, 0])

    def test_extend_plan_covers_the_whole_chunk(self):
        """Page-granular sharing makes every matched prefix a whole number of
        groups, so an extend row plans exactly its chunk's groups."""
        table = _page_aligned_table(1, 1024)
        write_locs, group_positions, _, member_rows = _plan(
            lengths=[600], table=table, extend_lens=[600 - 128]
        )
        # Extend groups are sourced from this forward's packed tensors: the
        # first group's first member is chunk-local row 0.
        self.assertEqual(int(member_rows[0]), 0)
        groups = 600 // RATIO - 128 // RATIO
        self.assertEqual(int(group_positions[0]), 128 + RATIO - 1)
        self.assertEqual(int(write_locs[0]), int(table[0, 128]) // RATIO)
        self.assertEqual(write_locs[:groups].tolist().count(0), 0)
        self.assertGreaterEqual(write_locs.numel(), groups)

    def test_pool_rejects_unaligned_page_size(self):
        """Bookkeeping-free addressing is only sound on a ratio-aligned paged
        allocator; a token-level pool must fail at boot, not corrupt."""
        from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

        with self.assertRaises(ValueError):
            QSATokenToKVPool(
                size=256,
                dtype=torch.bfloat16,
                page_size=1,
                head_num=2,
                head_dim=64,
                full_attention_layer_ids=[0],
                device="cpu",
                mamba_pool=None,
                qsa_index_kv_heads=1,
                qsa_index_head_dim=128,
                qsa_compress_ratio=4,
                qsa_token_topk=2048,
                num_request_slots=8,
            )


if __name__ == "__main__":
    unittest.main()
