"""Contract tests for the peer KV page reader, against a fake CUDA layer."""

import contextlib
import unittest

import numpy as np

from sglang.srt.mem_cache.kv_peer_read import (
    KvPeerExporter,
    KvPeerReader,
    PeerBuffer,
    PeerReadBusy,
    PeerReadError,
    PeerRegion,
    PeerRegionDescriptor,
    ReadGroup,
    ReadState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

NOT_READY, OK, ILLEGAL_ADDRESS, NON_STICKY = 600, 0, 700, 999
PAGE = 256
OWNER_DEV, READER_DEV = 0, 1


class FakeOps:
    """Byte-addressed fake: launches queue page copies that land when a later event is queried."""

    not_ready = NOT_READY

    def __init__(self, clique=(b"c" * 16, 1)):
        self.allocs = {}  # base -> (bytearray, device)
        self.handles = {}  # fabric handle -> base
        self.clique = clique
        self.pending = []  # copies not yet landed
        self.event_marks = {}  # recorded event -> number of copies queued before it
        self.num_events = 0
        self.landed = 0
        self.log = []
        self.launches = 0
        self.fail_launch_at = None
        self.event_codes = {}  # event -> error code reported instead of completion
        self.sync_code = OK
        self.fail_record = False
        self.fail_event_at = None  # new_event call that raises
        self.hold = False
        self._next = 1 << 20

    def alloc(self, size, device, fill=0):
        base, self._next = self._next, self._next + (size + (1 << 20))
        self.allocs[base] = (bytearray([fill]) * size, device)
        return base

    def _locate(self, addr):
        for base, (buf, _) in self.allocs.items():
            if base <= addr < base + len(buf):
                return buf, addr - base
        raise AssertionError(f"wild address {addr:#x}")

    def read(self, addr, n):
        buf, off = self._locate(addr)
        return bytes(buf[off : off + n])

    def fabric_clique(self, device):
        return self.clique

    def export_handle(self, base):
        handle = base.to_bytes(8, "little") * 8
        self.handles[handle] = base
        return handle

    def map_import(self, fabric_handle, size, device):
        self.log.append("map")
        return self.handles[fabric_handle]  # alias the owner's bytes

    def unmap_import(self, address, size):
        self.log.append("unmap")

    def pointer_device(self, pointer):
        for base, (buf, device) in self.allocs.items():
            if base <= pointer < base + len(buf):
                return device
        return None

    def new_stream(self, device):
        return "copy-stream"

    def wait_stream(self, stream, producer):
        self.log.append(("wait", producer))

    def warmup(self, stream):
        self.log.append("warmup")

    def current(self, stream):
        return contextlib.nullcontext()

    def stage(self, stream, meta, slices):
        self.log.append("stage")
        meta = meta.copy()
        views = [
            [
                (page_bytes, *(meta[o : o + n] for o, n in launch_slices))
                for page_bytes, launch_slices in group
            ]
            for group in slices
        ]
        return meta, views

    def launch(self, view):
        if self.fail_launch_at == self.launches:
            raise RuntimeError("launch failed")
        self.launches += 1
        self.log.append("launch")
        page_bytes, src_bases, dst_bases, src_pages, dst_pages = view
        for s, d in zip(src_bases.tolist(), dst_bases.tolist()):
            for sp, dp in zip(src_pages.tolist(), dst_pages.tolist()):
                self.pending.append(
                    (s + sp * page_bytes, d + dp * page_bytes, page_bytes)
                )

    def new_event(self, stream):
        if self.fail_event_at == self.num_events:
            raise RuntimeError("event creation failed")
        self.num_events += 1
        return self.num_events - 1

    def record(self, event, stream):
        if self.fail_record:
            raise RuntimeError("record failed")
        self.event_marks[event] = len(self.pending)

    def _land(self, upto):
        while self.landed < upto:
            src, dst, n = self.pending[self.landed]
            data = self.read(src, n)
            buf, off = self._locate(dst)
            buf[off : off + n] = data
            self.landed += 1

    def query_event(self, event):
        if event in self.event_codes:
            return self.event_codes[event]
        if self.hold:
            return NOT_READY
        self._land(self.event_marks[event])
        return OK

    def sync_event(self, event):
        self.log.append(("sync", event))
        return self.sync_code

    def destroy_event(self, event):
        self.log.append(("destroy", event))

    def is_sticky(self, code):
        return code == ILLEGAL_ADDRESS


class PeerReadTestBase(unittest.TestCase):
    NUM_PAGES = 8

    def setUp(self):
        self.ops = FakeOps()
        self.layers = 2
        self.region_bytes = self.layers * self.NUM_PAGES * PAGE
        self.src = self.ops.alloc(self.region_bytes, OWNER_DEV)
        buf, _ = self.ops.allocs[self.src]
        buf[:] = bytes(range(256)) * (self.region_bytes // 256)
        for layer in range(self.layers):
            for page in range(self.NUM_PAGES):
                off = (layer * self.NUM_PAGES + page) * PAGE
                buf[off : off + 2] = bytes([layer, page])
        self.dst = [
            self.ops.alloc(self.NUM_PAGES * PAGE, READER_DEV, fill=0xEE)
            for _ in range(self.layers)
        ]
        self.exporter = KvPeerExporter(OWNER_DEV, _ops=self.ops)
        self.reader = KvPeerReader(READER_DEV, _ops=self.ops)
        self.desc = self.exporter.export(self.src, self.region_bytes, region_id="kv")
        self.region = self.reader.import_region(self.desc)

    def group(self, src_pages, dst_pages, layers=None, dst=None, regions=None):
        dst = dst or self.dst
        regions = regions or [self.region] * self.layers
        layers = range(self.layers) if layers is None else layers
        buffers = tuple(
            PeerBuffer(
                src_region=regions[layer],
                src_offset=layer * self.NUM_PAGES * PAGE,
                dst_base=dst[layer],
                page_bytes=PAGE,
                dst_num_pages=self.NUM_PAGES,
            )
            for layer in layers
        )
        return ReadGroup(
            buffers=buffers,
            src_pages=np.array(src_pages),
            dst_pages=np.array(dst_pages),
        )

    def dst_page(self, layer, page):
        return self.ops.read(self.dst[layer] + page * PAGE, PAGE)

    def src_page(self, layer, page):
        return self.ops.read(self.src + (layer * self.NUM_PAGES + page) * PAGE, PAGE)


class TestReadData(PeerReadTestBase):
    def test_pages_land_and_untouched_pages_keep_sentinel(self):
        batch = self.reader.read([self.group([5, 1], [0, 3]), self.group([7], [6])])
        status = self.reader.wait(batch)
        self.assertEqual(status.state, ReadState.SUCCESS)
        self.assertEqual(status.groups_done, 2)
        for layer in range(self.layers):
            for sp, dp in ((5, 0), (1, 3), (7, 6)):
                self.assertEqual(self.dst_page(layer, dp), self.src_page(layer, sp))
            for dp in (1, 2, 4, 5, 7):
                self.assertEqual(self.dst_page(layer, dp), b"\xee" * PAGE)
        self.reader.release(batch)

    def test_kernel_is_warmed_before_the_first_read(self):
        # A first launch inside a (possibly gated) group could load the module there.
        self.assertLess(self.ops.log.index("warmup"), self.ops.log.index("map"))

    def test_producer_stream_is_waited_before_any_launch(self):
        batch = self.reader.read([self.group([0], [0])], producer_stream="prod")
        self.assertLess(
            self.ops.log.index(("wait", "prod")), self.ops.log.index("launch")
        )
        self.reader.wait(batch)
        self.reader.release(batch)


class TestValidationIsAtomic(PeerReadTestBase):
    def assert_rejected(self, groups, error=ValueError):
        before = [bytes(self.ops.allocs[d][0]) for d in self.dst]
        with self.assertRaises(error):
            self.reader.read(groups)
        self.assertEqual(self.ops.launches, 0)
        self.assertEqual([bytes(self.ops.allocs[d][0]) for d in self.dst], before)

    def test_source_page_beyond_region(self):
        self.assert_rejected([self.group([0], [0]), self.group([self.NUM_PAGES], [1])])

    def test_destination_page_beyond_buffer(self):
        self.assert_rejected([self.group([0], [0]), self.group([1], [self.NUM_PAGES])])

    def test_duplicate_destination_page_in_group(self):
        self.assert_rejected([self.group([0, 1], [2, 2])])

    def test_same_destination_page_in_two_groups(self):
        self.assert_rejected([self.group([0], [2]), self.group([1], [2])])

    def test_two_buffers_of_a_group_share_a_destination(self):
        self.assert_rejected([self.group([0], [2], dst=[self.dst[0], self.dst[0]])])

    def test_partially_overlapping_destination_buffers(self):
        shifted = [self.dst[0] + PAGE, self.dst[1]]
        self.assert_rejected([self.group([0], [0]), self.group([1], [5], dst=shifted)])

    def test_destination_on_another_device(self):
        foreign = self.ops.alloc(self.NUM_PAGES * PAGE, OWNER_DEV)
        self.assert_rejected([self.group([0], [0], dst=[foreign, self.dst[1]])])

    def test_destination_inside_an_imported_region(self):
        local = self.ops.alloc(self.region_bytes, READER_DEV)
        peer = self.reader.import_region(
            self.exporter.export(local, self.region_bytes, region_id="local")
        )
        self.assert_rejected([self.group([0], [0], dst=[peer.address, self.dst[1]])])

    def test_destination_running_past_its_allocation(self):
        short = self.ops.alloc(PAGE, READER_DEV)
        self.assert_rejected([self.group([0], [0], dst=[short, self.dst[1]])])

    def test_misaligned_destination(self):
        self.assert_rejected(
            [self.group([0], [0], dst=[self.dst[0] + 16, self.dst[1]])]
        )

    def test_mismatched_page_lists_and_empty_batch(self):
        self.assert_rejected([self.group([0, 1], [0])])
        self.assert_rejected([])


class TestRegions(PeerReadTestBase):
    def test_one_batch_reads_several_regions(self):
        second = self.reader.import_region(
            self.exporter.export(self.src, self.region_bytes, region_id="kv")
        )
        self.ops.hold = True
        batch = self.reader.read([self.group([2], [4], regions=[self.region, second])])
        for region in (self.region, second):
            with self.assertRaises(PeerReadBusy):
                self.reader.unmap(region)
        self.ops.hold = False
        self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)
        for layer in range(self.layers):
            self.assertEqual(self.dst_page(layer, 4), self.src_page(layer, 2))
        self.reader.release(batch)
        self.reader.unmap(second)
        self.reader.unmap(self.region)

    def test_unmap_evicts_cached_buffers_that_read_the_region(self):
        group = self.group([0], [0])
        batch = self.reader.read([group])
        self.reader.wait(batch)
        self.reader.release(batch)
        self.reader.unmap(self.region)
        with self.assertRaises(ValueError):  # same tuple, its region is gone
            self.reader.read([group])


class TestInFlightRanges(PeerReadTestBase):
    def test_conflict_only_while_in_flight(self):
        self.ops.hold = True
        first = self.reader.read([self.group([0, 1], [0, 1])])
        with self.assertRaises(PeerReadBusy):  # retry later, not a caller bug
            self.reader.read([self.group([2], [1])])
        disjoint = self.reader.read([self.group([2], [2])])  # same region, other pages
        self.ops.hold = False
        for batch in (first, disjoint):
            self.reader.wait(batch)
            self.reader.release(batch)
        again = self.reader.read([self.group([3], [1])])
        self.reader.wait(again)
        self.reader.release(again)

    def single(self, dst_base, capacity, page):
        buffer = PeerBuffer(
            src_region=self.region,
            src_offset=0,
            dst_base=dst_base,
            page_bytes=PAGE,
            dst_num_pages=capacity,
        )
        return ReadGroup(
            buffers=(buffer,), src_pages=np.array([0]), dst_pages=np.array([page])
        )

    def test_growing_a_destination_over_an_in_flight_one(self):
        self.ops.hold = True
        base = self.dst[0]
        self.reader.read([self.single(base, 1, 0)])
        self.reader.read([self.single(base + 2 * PAGE, 1, 0)])
        # Same (base, page size) as the first, but now reaching into the second.
        with self.assertRaises(PeerReadBusy):
            self.reader.read([self.single(base, 4, 2)])

    def test_self_overlap_is_a_caller_error_behind_an_in_flight_batch(self):
        """A batch whose own destinations overlap raises ValueError even when it also
        overlaps a batch in flight: an uncertain one would make PeerReadBusy permanent."""
        self.ops.hold = True
        base = self.dst[0]
        self.reader.read([self.single(base, 4, 0)])
        with self.assertRaises(ValueError):
            self.reader.read(
                [self.single(base, 4, 1), self.single(base + 2 * PAGE, 2, 0)]
            )

    def test_finished_but_unpolled_batch_does_not_block(self):
        first = self.reader.read([self.group([0], [0])])  # done, nobody asked
        second = self.reader.read([self.group([1], [0])])
        for batch in (first, second):
            self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)
            self.reader.release(batch)
        self.assertEqual(self.dst_page(0, 0), self.src_page(0, 1))

    def test_timeout_keeps_batch_and_ranges(self):
        self.ops.hold = True
        batch = self.reader.read([self.group([0], [0])])
        with self.assertRaises(TimeoutError):
            self.reader.wait(batch, timeout=0)
        with self.assertRaises(PeerReadBusy):
            self.reader.release(batch)
        with self.assertRaises(PeerReadBusy):
            self.reader.unmap(self.region)
        with self.assertRaises(PeerReadBusy):
            self.reader.read([self.group([1], [0])])
        for bad in (-1, float("inf"), float("nan"), "1"):
            with self.assertRaises(ValueError):
                self.reader.wait(batch, timeout=bad)


class TestFailures(PeerReadTestBase):
    def test_completed_group_survives_later_failure(self):
        self.ops.event_codes[1] = NON_STICKY
        batch = self.reader.read([self.group([4], [0]), self.group([5], [1])])
        status = self.reader.wait(batch)
        self.assertEqual(status.state, ReadState.FAILED_DRAINED)
        self.assertTrue(self.reader.group_done(batch, 0))
        self.assertFalse(self.reader.group_done(batch, 1))
        self.assertEqual(self.dst_page(0, 0), self.src_page(0, 4))
        self.reader.release(batch)
        self.reader.unmap(self.region)

    def test_undrained_failure_is_uncertain_and_pins_everything(self):
        self.ops.event_codes[0] = NON_STICKY
        self.ops.sync_code = NON_STICKY
        batch = self.reader.read([self.group([0], [0])])
        self.assertEqual(self.reader.wait(batch).state, ReadState.FAILED_UNCERTAIN)
        with self.assertRaises(PeerReadBusy):
            self.reader.release(batch)
        with self.assertRaises(PeerReadBusy):
            self.reader.unmap(self.region)
        with self.assertRaises(PeerReadError):
            self.reader.read([self.group([1], [1])])

    def test_sticky_error_fail_stops_every_in_flight_batch(self):
        self.ops.hold = True
        other = self.reader.read([self.group([1], [1])])
        self.ops.hold = False
        self.ops.event_codes[1] = ILLEGAL_ADDRESS
        failing = self.reader.read([self.group([0], [0])])
        self.ops.event_codes[0] = NOT_READY  # the other batch has not finished either
        self.assertEqual(self.reader.wait(failing).state, ReadState.FAILED_UNCERTAIN)
        self.assertEqual(self.reader.status(other).state, ReadState.FAILED_UNCERTAIN)
        with self.assertRaises(PeerReadError):
            self.reader.import_region(
                self.exporter.export(self.src, self.region_bytes, region_id="kv")
            )

    def test_enqueue_failure_with_clean_drain_leaves_nothing_held(self):
        self.ops.fail_launch_at = 1
        with self.assertRaises(RuntimeError):
            self.reader.read([self.group([0], [0]), self.group([1], [1])])
        self.ops.fail_launch_at = None
        batch = self.reader.read([self.group([2], [0]), self.group([3], [1])])
        self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)
        self.reader.release(batch)
        self.reader.unmap(self.region)
        self.reader.close()

    def test_drain_covers_a_partially_enqueued_group(self):
        # Two page sizes make two launches in one group; the second one fails.
        half = [
            PeerBuffer(
                src_region=self.region,
                src_offset=0,
                dst_base=self.dst[0],
                page_bytes=PAGE,
                dst_num_pages=self.NUM_PAGES,
            ),
            PeerBuffer(
                src_region=self.region,
                src_offset=self.NUM_PAGES * PAGE,
                dst_base=self.dst[1],
                page_bytes=PAGE // 2,
                dst_num_pages=self.NUM_PAGES,
            ),
        ]
        group = ReadGroup(
            buffers=tuple(half), src_pages=np.array([0]), dst_pages=np.array([0])
        )
        self.ops.fail_launch_at = 1
        with self.assertRaises(RuntimeError):
            self.reader.read([group])
        self.assertIn(
            ("sync", 0), self.ops.log
        )  # the group's own event, recorded after the failure
        self.ops.fail_launch_at = None
        batch = self.reader.read([self.group([1], [0])])
        self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)

    def test_unrecorded_launch_fail_stops(self):
        self.ops.fail_record = True
        with self.assertRaises(RuntimeError):
            self.reader.read([self.group([0], [0])])
        self.assertNotIn(
            ("sync", 0), self.ops.log
        )  # nothing covers the launch, so no drain is claimed
        with self.assertRaises(PeerReadError):
            self.reader.read([self.group([1], [1])])
        with self.assertRaises(PeerReadBusy):  # a launch may still be reading it
            self.reader.unmap(self.region)

    def test_failed_event_creation_destroys_the_events_made(self):
        self.ops.fail_event_at = 1
        with self.assertRaises(RuntimeError):
            self.reader.read([self.group([0], [0]), self.group([1], [1])])
        self.assertIn(("destroy", 0), self.ops.log)
        self.assertEqual(self.ops.launches, 0)
        self.ops.fail_event_at = None
        batch = self.reader.read([self.group([0], [0])])  # nothing was held
        self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)

    def test_enqueue_failure_without_drain_fail_stops(self):
        self.ops.fail_launch_at = 1  # group 0 is enqueued, group 1 fails, no drain
        self.ops.sync_code = ILLEGAL_ADDRESS
        with self.assertRaises(RuntimeError):
            self.reader.read([self.group([0], [0]), self.group([1], [1])])
        with self.assertRaises(PeerReadError):
            self.reader.read([self.group([2], [2])])
        with self.assertRaises(PeerReadBusy):  # group 0 may still be reading it
            self.reader.unmap(self.region)


class TestRegionsAndLeases(PeerReadTestBase):
    def test_descriptor_round_trip_and_malformed_input(self):
        self.assertEqual(
            PeerRegionDescriptor.from_bytes(self.desc.to_bytes()), self.desc
        )
        for data in (b"", b"\x01garbage", self.desc.to_bytes()[:-3]):
            with self.assertRaises(ValueError):
                PeerRegionDescriptor.from_bytes(data)

    def test_lease_maps_once_even_after_unmap(self):
        with self.assertRaises(ValueError):
            self.reader.import_region(self.desc)
        self.reader.unmap(self.region)
        with self.assertRaises(ValueError):
            self.reader.import_region(self.desc)
        fresh = self.exporter.export(self.src, self.region_bytes, region_id="kv")
        self.reader.import_region(fresh)

    def test_other_fabric_clique_rejected_before_mapping(self):
        other = KvPeerReader(READER_DEV, _ops=FakeOps(clique=(b"x" * 16, 1)))
        maps = self.ops.log.count("map")
        with self.assertRaises(PeerReadError):
            other.import_region(
                self.exporter.export(self.src, self.region_bytes, region_id="kv")
            )
        self.assertEqual(self.ops.log.count("map"), maps)

    def test_forged_region_handle(self):
        forged = PeerRegion(
            self.region.region_id, self.region.size, self.region.address
        )
        with self.assertRaises(ValueError):
            self.reader.read([self.group([0], [0], regions=[forged] * self.layers)])
        with self.assertRaises(ValueError):
            KvPeerReader(READER_DEV, _ops=self.ops).unmap(self.region)

    def test_lease_release_is_exact(self):
        second = self.exporter.export(self.src, self.region_bytes, region_id="kv")
        self.assertEqual(self.exporter.outstanding("kv"), 2)
        self.exporter.release_export(self.desc.lease_id)
        with self.assertRaises(ValueError):
            self.exporter.release_export(self.desc.lease_id)
        self.assertEqual(self.exporter.outstanding("kv"), 1)
        self.exporter.release_export(second.lease_id)
        self.assertEqual(self.exporter.outstanding(), 0)


if __name__ == "__main__":
    unittest.main()
