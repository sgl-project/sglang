"""Slot math of the QSA pending index-key ring under speculative forwards.

A target-verify or MTP draft-extend forward of W rows per request stores the
raw index key of every row in the per-request pending ring before it
compresses the groups those rows complete. A group completed at row j reaches
back ``ratio - 1`` positions, past the forward's first row, so its members
survive only if no later row of the same forward lands on their slots. These
tests replay the stores and group reads of the real slot builders on CPU; the
GPU counterpart (real indexer, pool and CUDA-graph metadata kernels) is
``test/registered/kernels/ops/attention/qsa/test_qsa_verify_ring.py``.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.mem_cache.qsa_kv_pool import qsa_pending_ring_size
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RATIO = 4


class _Ring:
    """Logical positions stored in each pending-ring slot (-1 = never written)."""

    def __init__(self, ring_size, num_requests):
        self.ring_size = ring_size
        self.positions = torch.full(((num_requests + 1) * ring_size,), -1)

    def forward(self, requests, bases, width):
        """One decode-like forward (decode, target verify or draft extend): every
        row's store, then every group completed in it read back from the ring.
        Returns ``{(request, group_end): member positions read}``."""
        positions = torch.cat([torch.arange(b, b + width) for b in bases])
        token_to_batch_idx = torch.arange(len(requests)).repeat_interleave(width)
        req_pool_indices = torch.tensor(requests)
        slots = build_pending_ring_slots(
            token_to_batch_idx=token_to_batch_idx,
            req_pool_indices=req_pool_indices,
            sequence_lengths=positions + 1,
            logical_positions=positions,
            compress_ratio=RATIO,
            ring_size=self.ring_size,
            is_extend=False,
        )
        self.positions[slots] = positions
        complete = (positions + 1) % RATIO == 0
        group_ends = positions[complete]
        sequence_ids = token_to_batch_idx[complete]
        members = build_group_ring_slots(
            req_pool_indices=req_pool_indices,
            group_end_positions=group_ends,
            sequence_ids=sequence_ids,
            compress_ratio=RATIO,
            ring_size=self.ring_size,
        )
        return {
            (requests[s], e): self.positions[m].tolist()
            for s, e, m in zip(sequence_ids.tolist(), group_ends.tolist(), members)
        }


def _members(group_end):
    return list(range(group_end - RATIO + 1, group_end + 1))


def _corrupted_groups(ring_size, width, base):
    """Groups a verify window of ``width`` rows starting at ``base`` compresses
    from the wrong members, after the request decoded positions ``0..base-1``."""
    ring = _Ring(ring_size, num_requests=1)
    for position in range(base):
        ring.forward([1], [position], 1)
    read = ring.forward([1], [base], width)
    return [end for (_, end), got in read.items() if got != _members(end)]


class TestQSAPendingRing(CustomTestCase):
    def test_ring_size(self):
        """One group without speculation; otherwise the power of two holding
        ``ratio + rows - 1`` consecutive positions."""
        for rows, ring in {1: 4, 2: 8, 5: 8, 6: 16, 13: 16, 14: 32}.items():
            with self.subTest(rows=rows):
                self.assertEqual(qsa_pending_ring_size(RATIO, rows), ring)

    def test_verify_window_keeps_every_group_member(self):
        """Windows of 1-9 rows on a ring sized for them, two requests per forward
        at drifting alignments, through a long generation."""
        for width in range(1, 10):
            ring = _Ring(qsa_pending_ring_size(RATIO, width), num_requests=2)
            # Different acceptance per request moves their windows out of step.
            committed = {1: 0, 2: 0}
            for step in range(64):
                requests = list(committed)
                read = ring.forward(requests, [committed[r] for r in requests], width)
                for (request, end), got in read.items():
                    with self.subTest(width=width, request=request, end=end):
                        self.assertEqual(got, _members(end))
                # Accept a varying number of rows; rejected rows stay in the ring
                # until the next window overwrites them, as rejected drafts do.
                for request in requests:
                    committed[request] += 1 + (step + request) % width

    def test_shorter_ring_loses_members(self):
        """With fewer than ``ratio + rows - 1`` slots, later rows overwrite the
        members a group completed earlier in the window reaches back for: a
        ``ratio``-slot ring at 4 rows unless the window starts a group, and an
        8-slot ring (the window alone) at 7 rows."""
        cases = [(4, 4, 33, [35]), (4, 4, 35, [35]), (4, 4, 36, []), (8, 7, 35, [35])]
        for ring, width, base, corrupted in cases:
            with self.subTest(ring=ring, width=width, base=base):
                self.assertEqual(_corrupted_groups(ring, width, base), corrupted)

    def test_extend_dump_stays_in_the_unallocated_request_slot(self):
        """Extend tokens before the pending tail dump into slot 0's rows; the tail
        lands in its own request's rows."""
        ring_size = qsa_pending_ring_size(RATIO, 4)
        lengths = torch.tensor([13, 6])
        positions = torch.cat([torch.arange(13), torch.arange(6)])
        slots = build_pending_ring_slots(
            token_to_batch_idx=torch.tensor([0] * 13 + [1] * 6),
            req_pool_indices=torch.tensor([3, 1]),
            sequence_lengths=lengths,
            logical_positions=positions,
            compress_ratio=RATIO,
            ring_size=ring_size,
            is_extend=True,
        )
        self.assertEqual(slots[:12].tolist(), [p % ring_size for p in range(12)])
        self.assertEqual(slots[12].item(), 3 * ring_size + 12 % ring_size)
        self.assertEqual(slots[13:17].tolist(), [p % ring_size for p in range(4)])
        self.assertEqual(slots[17:].tolist(), [ring_size + 4, ring_size + 5])

    def test_speculative_guard(self):
        """Target verify and draft extend are refused past ``ratio + rows - 1``
        ring slots; other modes are not checked."""
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.compress_ratio = RATIO
        verify = ForwardMode.TARGET_VERIFY
        extend = ForwardMode.DRAFT_EXTEND_V2
        # (ring slots, mode, rows per request, admitted)
        cases = [
            (4, verify, 1, True),
            (4, verify, 2, False),
            (8, verify, 5, True),
            (8, verify, 6, False),
            (16, verify, 13, True),
            (16, verify, 14, False),
            (8, extend, 5, True),
            # Multi-layer EAGLE's boundary-KV fix: 3 steps + 1 + 5 front rows.
            (8, extend, 9, False),
            (4, ForwardMode.DECODE, 9, True),
        ]
        for ring, mode, rows, admitted in cases:
            backend.token_to_kv_pool = SimpleNamespace(qsa_ring_size=ring)
            spec_info = SimpleNamespace(
                topk=1, draft_token_num=rows, num_tokens_per_req=rows
            )
            with self.subTest(ring=ring, mode=mode.name, rows=rows):
                if admitted:
                    backend._require_chain_speculation(mode, spec_info)
                else:
                    with self.assertRaisesRegex(
                        NotImplementedError, "pending index-key ring"
                    ):
                        backend._require_chain_speculation(mode, spec_info)


if __name__ == "__main__":
    unittest.main()
