"""Unit tests for the PD KV-cache checksum (srt/disaggregation/kv_checksum.py).

The property that makes the check usable across a transfer is that the digest
depends on a row's *logical* position in the request and not on the slot it
happens to live in -- prefill and decode hold the same KV at unrelated slots.
"""

import unittest

import torch

from sglang.srt.disaggregation.kv_checksum import (
    KVChecksummer,
    _pool_kv_buffers,
    _unsupported_reason,
    digest_to_u64,
)
from sglang.srt.environ import DisaggKVChecksumLevel, envs
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")

NUM_SLOTS = 512
NUM_BUFFERS = 8
HEAD_NUM = 2
HEAD_DIM = 64


def make_checksummer(level=DisaggKVChecksumLevel.SAMPLED, sampled=2, seed=0):
    torch.manual_seed(seed)
    buffers = [
        torch.randn(NUM_SLOTS, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        for _ in range(NUM_BUFFERS)
    ]
    return KVChecksummer(buffers=buffers, level=level, num_sampled_buffers=sampled)


def copy_rows(src: KVChecksummer, dst: KVChecksummer, src_slots, dst_slots):
    """Move a request's KV between two pools, landing it at different slots."""
    for s_buf, d_buf in zip(src.buffers, dst.buffers):
        d_buf[dst_slots] = s_buf[src_slots]


class TestKVChecksumKernel(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_digest_survives_relocation(self):
        """Same KV at different slots in a different pool digests the same."""
        prefill = make_checksummer(seed=1)
        decode = make_checksummer(seed=2)
        room = 0x1234_5678_9ABC
        n = 130
        src = torch.arange(10, 10 + n, device="cuda", dtype=torch.int64)
        dst = torch.arange(300, 300 + n, device="cuda", dtype=torch.int64).flip(0)
        copy_rows(prefill, decode, src, dst)

        self.assertEqual(
            prefill.compute_one(room, src).item(),
            decode.compute_one(room, dst).item(),
        )

    def test_clobbered_row_is_detected(self):
        prefill = make_checksummer(level=DisaggKVChecksumLevel.FULL)
        room = 7
        n = 96
        src = torch.arange(0, n, device="cuda", dtype=torch.int64)
        before = prefill.compute_one(room, src).item()
        prefill.corrupt_rows_for_test(src)
        self.assertNotEqual(before, prefill.compute_one(room, src).item())

    def test_single_element_change_is_detected(self):
        """A one-element edit inside a digested buffer changes the digest."""
        c = make_checksummer(level=DisaggKVChecksumLevel.FULL)
        room = 3
        n = 64
        slots = torch.arange(0, n, device="cuda", dtype=torch.int64)
        before = c.compute_one(room, slots).item()
        c.buffers[NUM_BUFFERS // 2][slots[n // 2], 1, 5] += 1.0
        self.assertNotEqual(before, c.compute_one(room, slots).item())

    def test_reordered_rows_are_detected(self):
        """Two rows swapped in place is a different request, not the same one."""
        c = make_checksummer(level=DisaggKVChecksumLevel.FULL)
        room = 11
        slots = torch.arange(0, 64, device="cuda", dtype=torch.int64)
        before = c.compute_one(room, slots).item()
        swapped = slots.clone()
        swapped[0], swapped[33] = slots[33].clone(), slots[0].clone()
        self.assertNotEqual(before, c.compute_one(room, swapped).item())

    def test_truncated_range_is_detected(self):
        c = make_checksummer(level=DisaggKVChecksumLevel.FULL)
        room = 5
        slots = torch.arange(0, 96, device="cuda", dtype=torch.int64)
        self.assertNotEqual(
            c.compute_one(room, slots).item(),
            c.compute_one(room, slots[:-1]).item(),
        )

    def test_batched_matches_individual(self):
        """One launch over N requests equals N single-request launches."""
        c = make_checksummer()
        rooms = [1, 1 << 40, 999, 0]
        slot_sets = [
            torch.arange(0, 100, device="cuda", dtype=torch.int64),
            torch.arange(100, 131, device="cuda", dtype=torch.int64),
            torch.arange(200, 200 + 33, device="cuda", dtype=torch.int64),
            torch.arange(0, 0, device="cuda", dtype=torch.int64),
        ]
        batched = c.compute(rooms, slot_sets).tolist()
        for room, slots, got in zip(rooms, slot_sets, batched):
            self.assertEqual(got, c.compute_one(room, slots).item())

    def test_empty_request_digests_to_zero(self):
        c = make_checksummer()
        empty = torch.zeros(0, device="cuda", dtype=torch.int64)
        self.assertEqual(c.compute_one(42, empty).item(), 0)

    def test_tile_boundaries(self):
        """Row counts around the kernel's tile size stay self-consistent."""
        c = make_checksummer(level=DisaggKVChecksumLevel.FULL)
        room = 17
        for n in (1, c.tile - 1, c.tile, c.tile + 1, 2 * c.tile, 3 * c.tile + 7):
            slots = torch.arange(0, n, device="cuda", dtype=torch.int64)
            self.assertEqual(
                c.compute_one(room, slots).item(),
                c.compute(([room] * 2), [slots, slots]).tolist()[0],
                f"n={n}",
            )


class TestSamplingPolicy(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_sample_is_deterministic_and_sized(self):
        c = make_checksummer(sampled=3)
        for room in (0, 1, 1 << 63, 12345678901234567):
            ids = c.sampled_buffer_ids(room)
            self.assertEqual(len(ids), 3)
            self.assertEqual(len(set(ids.tolist())), 3, f"room={room}")
            self.assertTrue((ids >= 0).all() and (ids < NUM_BUFFERS).all())
            self.assertEqual(ids.tolist(), c.sampled_buffer_ids(room).tolist())

    def test_rotation_sweeps_every_buffer(self):
        """Over a stream of requests the sample covers the whole model."""
        c = make_checksummer(sampled=2)
        seen = set()
        for room in range(200):
            seen.update(c.sampled_buffer_ids(room).tolist())
        self.assertEqual(seen, set(range(NUM_BUFFERS)))

    def test_full_level_digests_every_buffer(self):
        c = make_checksummer(level=DisaggKVChecksumLevel.FULL)
        self.assertEqual(c.num_sampled_buffers, NUM_BUFFERS)
        room = 1
        slots = torch.arange(0, 40, device="cuda", dtype=torch.int64)
        for buf in c.buffers:
            before = c.compute_one(room, slots).item()
            buf[slots[0], 0, 0] += 1.0
            self.assertNotEqual(before, c.compute_one(room, slots).item())

    def test_sampled_level_reads_less(self):
        c = make_checksummer(sampled=2)
        self.assertEqual(c.num_sampled_buffers, 2)
        self.assertLess(c.num_sampled_buffers, NUM_BUFFERS)

    def test_sampling_always_catches_a_whole_row_clobber(self):
        """The failure the check exists for: a reused slot rewrites every
        buffer of that row, so any nonempty sample sees it. Not probabilistic."""
        c = make_checksummer(sampled=1)
        slots = torch.arange(0, 200, device="cuda", dtype=torch.int64)
        for room in range(60):
            before = c.compute_one(room, slots).item()
            c.corrupt_rows_for_test(slots)
            self.assertNotEqual(
                before, c.compute_one(room, slots).item(), f"room={room}"
            )

    def test_sampling_misses_are_confined_to_unsampled_buffers(self):
        """A fault in one buffer is caught only when that buffer is sampled --
        the rotation is what makes the whole model reachable over time."""
        c = make_checksummer(sampled=2)
        slots = torch.arange(0, 64, device="cuda", dtype=torch.int64)
        caught = 0
        for room in range(60):
            target = room % NUM_BUFFERS
            before = c.compute_one(room, slots).item()
            c.buffers[target][slots[0], 0, 0] += 1.0
            caught += c.compute_one(room, slots).item() != before
            self.assertEqual(
                target in c.sampled_buffer_ids(room).tolist(),
                c.compute_one(room, slots).item() != before,
            )
        self.assertGreater(caught, 0)


class TestLayoutSignature(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_signature_is_stable_and_nonzero(self):
        a = make_checksummer(seed=1)
        b = make_checksummer(seed=2)
        self.assertEqual(a.signature, b.signature)
        self.assertNotEqual(a.signature, 0)

    def test_signature_separates_incompatible_layouts(self):
        base = make_checksummer(sampled=2)
        self.assertNotEqual(base.signature, make_checksummer(sampled=3).signature)
        self.assertNotEqual(
            base.signature,
            make_checksummer(level=DisaggKVChecksumLevel.FULL).signature,
        )
        # A narrower TP shard: fewer heads per rank, so narrower rows.
        narrow = KVChecksummer(
            buffers=[
                torch.zeros(NUM_SLOTS, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
                for _ in range(NUM_BUFFERS)
            ],
            level=DisaggKVChecksumLevel.SAMPLED,
            num_sampled_buffers=2,
        )
        self.assertNotEqual(base.signature, narrow.signature)
        # A PP prefill stage holds a subset of the layers.
        fewer = KVChecksummer(
            buffers=[
                torch.zeros(
                    NUM_SLOTS, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
                )
                for _ in range(NUM_BUFFERS // 2)
            ],
            level=DisaggKVChecksumLevel.SAMPLED,
            num_sampled_buffers=2,
        )
        self.assertNotEqual(base.signature, fewer.signature)


class FakePool(MLATokenToKVPool):
    """Test double: an MLA pool with only the fields the gate reads.

    Deliberately skips MLATokenToKVPool.__init__ -- allocating a real pool to
    check a layout predicate would test the allocator, not the predicate.
    """

    def __init__(self, buffers, page_size=1, ptrs=None, item_lens=None):
        self.kv_buffer = buffers
        self.page_size = page_size
        self._ptrs = ptrs
        self._item_lens = item_lens

    def get_contiguous_buf_infos(self):
        row = self.kv_buffer[0][0].nbytes
        return (
            (
                self._ptrs
                if self._ptrs is not None
                else [b.data_ptr() for b in self.kv_buffer]
            ),
            [b.nbytes for b in self.kv_buffer],
            (
                self._item_lens
                if self._item_lens is not None
                else [self.page_size * row] * len(self.kv_buffer)
            ),
        )


def cuda_buffers(n=4, elems=64):
    return [
        torch.zeros(32, elems, dtype=torch.bfloat16, device="cuda") for _ in range(n)
    ]


class TestPoolDiscovery(CustomTestCase):
    def test_unknown_pool_disables_the_check(self):
        class NotAPool:
            pass

        self.assertIsNone(_pool_kv_buffers(NotAPool()))

    def test_disabled_by_default(self):
        with envs.SGLANG_DISAGGREGATION_KV_CHECKSUM.override(
            int(DisaggKVChecksumLevel.OFF)
        ):
            self.assertIsNone(KVChecksummer.maybe_create(object()))

    def test_cpu_buffers_are_rejected(self):
        """A host-side pool (HiCache, HiSparse) is not what the transfer writes."""
        pool = FakePool([torch.zeros(32, 64, dtype=torch.bfloat16)])
        self.assertIsNone(_pool_kv_buffers(pool))

    def test_ragged_buffers_are_rejected(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        buffers = cuda_buffers(2)
        buffers[1] = torch.zeros(32, 128, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(_pool_kv_buffers(FakePool(buffers)))

    def test_plain_pool_is_discovered(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        buffers = cuda_buffers()
        self.assertEqual(_pool_kv_buffers(FakePool(buffers)), buffers)


class TestSupportGate(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_matching_pool_is_supported(self):
        buffers = cuda_buffers()
        self.assertIsNone(_unsupported_reason(FakePool(buffers), buffers))

    def test_extra_transferred_regions_are_rejected(self):
        """A pool that transfers buffers beyond the ones digested here: the
        digest would cover only part of what the transfer writes."""
        buffers = cuda_buffers()
        extra = torch.zeros(32, 64, dtype=torch.bfloat16, device="cuda")
        pool = FakePool(
            buffers, ptrs=[b.data_ptr() for b in buffers] + [extra.data_ptr()]
        )
        reason = _unsupported_reason(pool, buffers)
        self.assertIsNotNone(reason)
        self.assertIn("other than", reason)

    def test_packed_rows_are_rejected(self):
        """An item_len that is not page_size x row_bytes means slots are not
        one contiguous row each."""
        buffers = cuda_buffers()
        row = buffers[0][0].nbytes
        pool = FakePool(buffers, page_size=4, item_lens=[2 * row] * len(buffers))
        reason = _unsupported_reason(pool, buffers)
        self.assertIsNotNone(reason)
        self.assertIn("contiguous row per slot", reason)

    def test_pool_without_contiguous_regions_is_rejected(self):
        class NoRegions(FakePool):
            def get_contiguous_buf_infos(self):
                raise NotImplementedError("unified layout")

        buffers = cuda_buffers()
        reason = _unsupported_reason(NoRegions(buffers), buffers)
        self.assertIsNotNone(reason)
        self.assertIn("contiguous KV regions", reason)


class TestDigestEncoding(CustomTestCase):
    def test_int64_round_trips_through_uint64(self):
        self.assertEqual(digest_to_u64(-1), (1 << 64) - 1)
        self.assertEqual(digest_to_u64(0), 0)
        self.assertEqual(digest_to_u64(1 << 40), 1 << 40)


if __name__ == "__main__":
    unittest.main()
