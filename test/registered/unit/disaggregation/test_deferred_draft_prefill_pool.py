import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.deferred_draft_prefill import (
    DeferredDraftPrefillPool,
    build_deferred_draft_hidden_transfer_blocks,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    import torch
except ImportError:
    torch = None

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def request(rid, length=6):
    return SimpleNamespace(rid=rid, origin_input_ids=list(range(length)))


class TestDeferredDraftPrefillPool(unittest.TestCase):
    def pool(self, slots=2, limit=8):
        return DeferredDraftPrefillPool(
            slots, limit, 4, "torch.bfloat16", "cpu", allocate_storage=False
        )

    def test_bounded_admission_and_idempotent_allocation(self):
        pool = self.pool()
        a, b, c = request("a"), request("b"), request("c")
        self.assertEqual(pool.alloc(a), 0)
        self.assertEqual(pool.alloc(a), 0)
        self.assertEqual(pool.available_slots, 1)
        self.assertEqual(pool.alloc(b), 1)
        self.assertFalse(pool.try_alloc(c))
        self.assertTrue(pool.free(a))
        self.assertFalse(pool.free(a))
        self.assertEqual(pool.alloc(c), 0)
        self.assertEqual(pool.available_slots, 0)

    def test_oversized_input_does_not_consume_a_slot(self):
        pool = self.pool()
        for length in (0, 9):
            with self.subTest(length=length), self.assertRaises(ValueError):
                pool.try_alloc(request("oversized", length))
        self.assertEqual(pool.available_slots, 2)

    def test_pending_device_consumer_prevents_early_reuse(self):
        pool = self.pool(slots=1)
        a, b = request("a"), request("b")
        event = SimpleNamespace(done=False)
        event.query = lambda: event.done
        pool.alloc(a)
        pool.free(a, completion_event=event)
        self.assertFalse(pool.try_alloc(b))
        event.done = True
        self.assertTrue(pool.try_alloc(b))
        self.assertEqual(pool.get_slot(b), 0)

    def test_drained_source_reuses_without_an_unrelated_device_fence(self):
        pool = self.pool(slots=1)
        a, b = request("a"), request("b")
        pool.alloc(a)
        self.assertTrue(pool.free_source(a))
        self.assertTrue(pool.try_alloc(b))

    def test_placeholder_pp_rank_has_no_source_storage(self):
        pool = self.pool()
        req = request("a")
        pool.alloc(req)
        self.assertEqual(pool.get_contiguous_buf_infos(), ([], [], []))
        self.assertEqual(pool.transfer_indices(req), [])
        with self.assertRaises(RuntimeError):
            pool.get(req)
        self.assertEqual(
            pool.wire_layout,
            {"version": 1, "hidden_size": 4, "dtype": "torch.bfloat16"},
        )

    def test_received_length_is_checked(self):
        pool = self.pool()
        req = request("a")
        pool.alloc(req)
        with self.assertRaises(ValueError):
            pool.mark_received(req, 5)
        pool.mark_received(req, 6)

    @unittest.skipIf(torch is None, "Tensor coverage requires PyTorch")
    def test_chunk_coverage_and_wire_rows(self):
        pool = DeferredDraftPrefillPool(2, 8, 4, torch.float32, "cpu")
        a, b = request("a"), request("b")
        pool.alloc(a)
        pool.alloc(b)
        hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        with self.assertRaises(RuntimeError):
            pool.get(b)
        with self.assertRaises(ValueError):
            pool.write_chunk(b, hidden[2:4], 2, 4)
        pool.write_chunk(b, hidden[:2], 0, 2)
        with self.assertRaises(RuntimeError):
            pool.get(b)
        pool.write_chunk(b, hidden[2:], 2, 6)
        torch.testing.assert_close(pool.get(b), hidden)
        self.assertEqual(pool.transfer_indices(b), list(range(8, 14)))
        self.assertEqual(pool.transfer_indices(b, 4, start=2), [10, 11])
        with self.assertRaises(ValueError):
            pool.transfer_indices(b, 7)


class TestDeferredDraftHiddenTransfer(unittest.TestCase):
    def blocks(self, source, destination, src_stride=8, dst_stride=8):
        return build_deferred_draft_hidden_transfer_blocks(
            [1000], [src_stride], [2000], [dst_stride], source, destination
        )

    def test_replication_uses_whole_hidden_rows_without_kv_head_slicing(self):
        self.assertEqual(self.blocks([16, 17, 18], [32, 33, 34]), [(1128, 2256, 24)])

    def test_only_live_rows_are_sent_not_uninitialized_slot_padding(self):
        blocks = self.blocks(list(range(8, 14)), list(range(16, 22)))
        self.assertEqual(blocks, [(1064, 2128, 48)])

    def test_nonfinal_pp_component_sends_nothing(self):
        self.assertEqual(
            build_deferred_draft_hidden_transfer_blocks([], [], [2000], [8], [], [0]),
            [],
        )

    def test_invalid_rows_and_layout_are_rejected(self):
        cases = (
            ([1, 2], [3], 8, 8),
            ([1, 3], [3, 4], 8, 8),
            ([-1], [0], 8, 8),
            ([0], [0], 8, 16),
            ([], [], 8, 8),
        )
        for source, destination, src_stride, dst_stride in cases:
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.blocks(source, destination, src_stride, dst_stride)


if __name__ == "__main__":
    unittest.main()
