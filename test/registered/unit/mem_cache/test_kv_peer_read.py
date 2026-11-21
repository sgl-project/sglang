"""Peer KV page reads: contract tests against a fake CUDA layer, and a two-process
GPU test over a CUDA FABRIC export, byte-checked."""

import contextlib
import multiprocessing as mp
import unittest

import numpy as np
import torch
from cuda.bindings import driver as drv

from sglang.srt.mem_cache.kv_peer_read import (
    KvPeerReader,
    PeerBuffer,
    PeerReadBusy,
    PeerReadError,
    PeerRegionDescriptor,
    ReadGroup,
    ReadState,
    export_region,
)
from sglang.srt.utils.cuda_vmm_utils import (
    align_up,
    check_drv,
    get_allocation_granularity,
    get_gpu_fabric_clique,
    make_device_allocation_prop,
    make_rw_access_desc,
    tensor_from_pointer,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cuda_ci(est_time=40, stage="base-c", runner_config="2-gpu-gb300")

PAGE = 256
OWNER_DEV, READER_DEV = 0, 1
CLIQUE = (b"c" * 16, 1)
NOT_STICKY, STICKY = False, True


# ===== Contract tests against a fake CUDA layer =====


def describe(base, size, clique=CLIQUE):
    # The fake's FABRIC handle is the owner's base address.
    return PeerRegionDescriptor(
        version=1,
        size=size,
        fabric_handle=base.to_bytes(8, "little") * 8,
        fabric_clique=clique,
    )


class FakeStream:
    def __init__(self, ops):
        self.ops = ops

    def wait_stream(self, producer):
        self.ops.log.append(("wait", producer))

    def synchronize(self):
        self.ops.log.append("sync")
        if self.ops.sticky:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        self.ops.land(len(self.ops.pending))


class FakeEvent:
    def __init__(self, ops, number, mark):
        self.ops, self.number, self.mark = ops, number, mark

    def query(self):
        sticky = self.ops.event_errors.get(self.number)
        if sticky is not None:
            self.ops.sticky |= sticky
            raise RuntimeError(f"CUDA error reported on event {self.number}")
        if self.ops.hold:
            return False
        self.ops.land(self.mark)
        return True


class FakeOps:
    """Byte-addressed fake: gathers queue page copies that land when a later event
    completes or the stream is synchronized."""

    def __init__(self, clique=CLIQUE):
        self.allocs = {}  # base -> (bytearray, device)
        self.clique = clique
        self.pending = []  # queued copies, in stream order
        self.landed = 0
        self.log = []
        self.events = 0
        self.launches = 0
        self.fail_launch_at = None
        self.event_errors = {}  # event number -> whether its error is sticky
        self.sticky = False
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

    def land(self, upto):
        while self.landed < upto:
            src, dst, n = self.pending[self.landed]
            data = self.read(src, n)
            buf, off = self._locate(dst)
            buf[off : off + n] = data
            self.landed += 1

    def fabric_clique(self):
        return self.clique

    def map_import(self, fabric_handle, size):
        self.log.append("map")
        return int.from_bytes(fabric_handle[:8], "little")  # alias the owner's bytes

    def unmap_import(self, address, size):
        self.log.append("unmap")

    def pointer_device(self, pointer):
        for base, (buf, device) in self.allocs.items():
            if base <= pointer < base + len(buf):
                return device
        return None

    def new_stream(self):
        return FakeStream(self)

    def current(self, stream):
        return contextlib.nullcontext()

    def warmup(self, stream):
        pass

    def upload(self, arrays):
        return [array.copy() for array in arrays]

    def gather(self, *, src_bases, dst_bases, src_pages, dst_pages, page_bytes):
        if self.fail_launch_at == self.launches:
            raise RuntimeError("launch failed")
        self.launches += 1
        self.log.append("launch")
        for s, d in zip(src_bases.tolist(), dst_bases.tolist()):
            for sp, dp in zip(src_pages.tolist(), dst_pages.tolist()):
                self.pending.append(
                    (s + sp * page_bytes, d + dp * page_bytes, page_bytes)
                )

    def copy(self, *, srcs, dsts, sizes, stream):
        if self.fail_launch_at == self.launches:
            raise RuntimeError("copy failed")
        self.launches += 1
        self.log.append(("copy", len(sizes)))
        self.pending += zip(srcs.tolist(), dsts.tolist(), sizes.tolist())

    def record_event(self, stream):
        self.events += 1
        return FakeEvent(self, self.events - 1, len(self.pending))


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
        self.reader = KvPeerReader(READER_DEV, _ops=self.ops)
        self.region = self.reader.import_region(describe(self.src, self.region_bytes))

    def group(self, src_pages, dst_pages, dst=None, regions=None):
        dst = dst or self.dst
        regions = regions or [self.region] * self.layers
        buffers = [
            PeerBuffer(
                src_region=regions[layer],
                src_offset=layer * self.NUM_PAGES * PAGE,
                dst_base=dst[layer],
                page_bytes=PAGE,
                dst_num_pages=self.NUM_PAGES,
            )
            for layer in range(self.layers)
        ]
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
        with self.assertRaises(ValueError):
            self.reader.status(batch)

    def test_copy_engine_merges_pages_adjacent_on_both_sides(self):
        reader = KvPeerReader(READER_DEV, copy_engine=True, _ops=self.ops)
        region = reader.import_region(describe(self.src, self.region_bytes))
        # Runs: 1-3 -> 4-6; 5 -> 7; 6 -> 0 (the source is adjacent, the destination not).
        pairs = ((1, 4), (2, 5), (3, 6), (5, 7), (6, 0))
        group = self.group(
            [s for s, _ in pairs],
            [d for _, d in pairs],
            regions=[region] * self.layers,
        )
        batch = reader.read([group])
        self.assertEqual(reader.wait(batch).state, ReadState.SUCCESS)
        self.assertIn(("copy", 3 * self.layers), self.ops.log)
        self.assertNotIn("launch", self.ops.log)
        for layer in range(self.layers):
            for sp, dp in pairs:
                self.assertEqual(self.dst_page(layer, dp), self.src_page(layer, sp))
            for dp in (1, 2, 3):
                self.assertEqual(self.dst_page(layer, dp), b"\xee" * PAGE)

    def test_producer_stream_is_waited_before_any_launch(self):
        batch = self.reader.read([self.group([0], [0])], producer_stream="prod")
        self.assertLess(
            self.ops.log.index(("wait", "prod")), self.ops.log.index("launch")
        )
        self.reader.wait(batch)
        self.reader.release(batch)


class TestValidation(PeerReadTestBase):
    def test_rejected_batches_touch_nothing(self):
        n = self.NUM_PAGES
        local = self.ops.alloc(self.region_bytes, READER_DEV)
        imported = self.reader.import_region(describe(local, self.region_bytes))
        other_reader = KvPeerReader(READER_DEV, _ops=self.ops)
        foreign = other_reader.import_region(describe(self.src, self.region_bytes))
        on_owner = self.ops.alloc(n * PAGE, OWNER_DEV)
        short = self.ops.alloc(PAGE, READER_DEV)
        cases = {
            "source page beyond region": [self.group([0], [0]), self.group([n], [1])],
            "destination page beyond buffer": [
                self.group([0], [0]),
                self.group([1], [n]),
            ],
            "destination on another device": [
                self.group([0], [0], dst=[on_owner, self.dst[1]])
            ],
            "destination inside an imported region": [
                self.group([0], [0], dst=[imported.address, self.dst[1]])
            ],
            "destination running past its allocation": [
                self.group([0], [0], dst=[short, self.dst[1]])
            ],
            "misaligned destination": [
                self.group([0], [0], dst=[self.dst[0] + 16, self.dst[1]])
            ],
            "page lists of different lengths": [self.group([0, 1], [0])],
            "negative page id": [self.group([-1], [0])],
            "region of another reader": [
                self.group([0], [0], regions=[foreign] * self.layers)
            ],
            "empty batch": [],
        }
        before = [bytes(self.ops.allocs[d][0]) for d in self.dst]
        for name, groups in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.reader.read(groups)
        self.assertEqual(self.ops.launches, 0)
        self.assertEqual([bytes(self.ops.allocs[d][0]) for d in self.dst], before)


class TestRegions(PeerReadTestBase):
    def test_one_batch_reads_several_regions_and_pins_them(self):
        second = self.reader.import_region(describe(self.src, self.region_bytes))
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
        with self.assertRaises(ValueError):  # its handle is stale now
            self.reader.read([self.group([0], [0], regions=[second] * self.layers)])
        self.reader.unmap(self.region)

    def test_timeout_keeps_batch_and_region(self):
        self.ops.hold = True
        batch = self.reader.read([self.group([0], [0])])
        with self.assertRaises(TimeoutError):
            self.reader.wait(batch, timeout=0)
        with self.assertRaises(PeerReadBusy):
            self.reader.release(batch)
        with self.assertRaises(PeerReadBusy):
            self.reader.unmap(self.region)
        for bad in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                self.reader.wait(batch, timeout=bad)
        self.ops.hold = False
        self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)


class TestFailures(PeerReadTestBase):
    def test_completed_group_survives_later_drained_failure(self):
        self.ops.event_errors[1] = NOT_STICKY
        batch = self.reader.read([self.group([4], [0]), self.group([5], [1])])
        status = self.reader.wait(batch)
        self.assertEqual(status.state, ReadState.FAILED_DRAINED)
        self.assertEqual(status.groups_done, 1)
        self.assertEqual(self.dst_page(0, 0), self.src_page(0, 4))
        self.reader.release(batch)
        self.reader.unmap(self.region)

    def test_sticky_error_fail_stops_every_in_flight_batch(self):
        self.ops.hold = True
        other = self.reader.read([self.group([1], [1])])
        self.ops.event_errors[1] = STICKY
        failing = self.reader.read([self.group([0], [0])])
        self.assertEqual(self.reader.wait(failing).state, ReadState.FAILED_UNCERTAIN)
        self.assertEqual(self.reader.status(other).state, ReadState.FAILED_UNCERTAIN)
        with self.assertRaises(PeerReadBusy):
            self.reader.release(other)
        with self.assertRaises(PeerReadBusy):
            self.reader.unmap(self.region)
        with self.assertRaises(PeerReadError):
            self.reader.read([self.group([2], [2])])
        with self.assertRaises(PeerReadError):
            self.reader.import_region(describe(self.src, self.region_bytes))

    def test_enqueue_failure_that_drains_leaves_nothing_held(self):
        self.ops.fail_launch_at = 1  # group 0 is enqueued, group 1 fails
        with self.assertRaises(RuntimeError):
            self.reader.read([self.group([0], [0]), self.group([1], [1])])
        self.assertIn("sync", self.ops.log)
        self.ops.fail_launch_at = None
        batch = self.reader.read([self.group([2], [0]), self.group([3], [1])])
        self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)
        self.reader.release(batch)
        self.reader.unmap(self.region)

    def test_enqueue_failure_without_drain_fail_stops(self):
        self.ops.fail_launch_at = 1
        self.ops.sticky = True
        with self.assertRaises(RuntimeError):
            self.reader.read([self.group([0], [0]), self.group([1], [1])])
        with self.assertRaises(PeerReadError):
            self.reader.read([self.group([2], [2])])
        with self.assertRaises(PeerReadBusy):  # group 0 may still be reading it
            self.reader.unmap(self.region)


class TestDescriptors(PeerReadTestBase):
    def test_round_trip_and_malformed_input(self):
        desc = describe(self.src, self.region_bytes)
        self.assertEqual(PeerRegionDescriptor.from_bytes(desc.to_bytes()), desc)
        for data in (b"", b"\x01garbage", desc.to_bytes()[:-3]):
            with self.assertRaises(ValueError):
                PeerRegionDescriptor.from_bytes(data)
        maps = self.ops.log.count("map")
        for bad in (
            PeerRegionDescriptor(version=2, size=PAGE, fabric_handle=b"h" * 64),
            PeerRegionDescriptor(version=1, size=PAGE, fabric_handle=b"h" * 63),
            PeerRegionDescriptor(version=1, size=0, fabric_handle=b"h" * 64),
        ):
            with self.assertRaises(ValueError):
                self.reader.import_region(bad)
        self.assertEqual(self.ops.log.count("map"), maps)

    def test_other_fabric_clique_rejected_before_mapping(self):
        other = KvPeerReader(READER_DEV, _ops=FakeOps(clique=(b"x" * 16, 1)))
        maps = self.ops.log.count("map")
        with self.assertRaises(PeerReadError):
            other.import_region(describe(self.src, self.region_bytes))
        self.assertEqual(self.ops.log.count("map"), maps)


# ===== Two processes on two GPUs of one NVLink fabric clique =====

LAYERS = 3
PAGE_BYTES = (1024, 384)  # two buffer classes per layer, like MLA + DSA index
CAPACITY = 64
REGION_BYTES = LAYERS * CAPACITY * sum(PAGE_BYTES)
SENTINEL = 0xEE


def _src_offset(layer: int, cls: int) -> int:
    # Class-major, then layer: [cls0 l0 .. l2][cls1 l0 .. l2], CAPACITY pages each.
    return (cls * LAYERS * PAGE_BYTES[0] + layer * PAGE_BYTES[cls]) * CAPACITY


def _pattern(offsets: torch.Tensor) -> torch.Tensor:
    return ((offsets * 7 + offsets // 4096) % 251).to(torch.uint8)


def _owner(conn) -> None:
    torch.cuda.set_device(0)
    fabric = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    prop = make_device_allocation_prop(0, handle_types=int(fabric))
    size = align_up(REGION_BYTES, get_allocation_granularity(prop))
    handle = check_drv(drv.cuMemCreate(size, prop, 0), "cuMemCreate")
    va = int(check_drv(drv.cuMemAddressReserve(size, 0, 0, 0), "reserve"))
    check_drv(drv.cuMemMap(va, size, 0, handle, 0), "map")
    check_drv(drv.cuMemSetAccess(va, size, [make_rw_access_desc(0)], 1), "access")
    view = tensor_from_pointer(va, size, device_id=0)
    view.copy_(_pattern(torch.arange(size, device="cuda", dtype=torch.int64)))
    torch.cuda.synchronize()

    # Two descriptors of one allocation: the reader maps it twice, as two regions.
    conn.send([export_region(handle, size, device=0).to_bytes() for _ in range(2)])
    assert conn.recv() == "unmapped"  # the reader's ACK: released and unmapped
    check_drv(drv.cuMemUnmap(va, size), "unmap")
    check_drv(drv.cuMemAddressFree(va, size), "free")
    check_drv(drv.cuMemRelease(handle), "release")
    conn.send("ok")


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestKvPeerReadCuda(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.device_count() >= 2 and get_gpu_fabric_clique(0) is not None:
            return
        reason = "needs two GPUs in an NVLink fabric clique"
        # The registered CUDA runner has them: a skip there hides a broken setup.
        if is_in_ci():
            raise RuntimeError(reason)
        raise unittest.SkipTest(reason)

    def _destinations(self):
        return [
            [
                torch.full((CAPACITY, pb), SENTINEL, dtype=torch.uint8, device="cuda:1")
                for pb in PAGE_BYTES
            ]
            for _ in range(LAYERS)
        ]

    def _group(self, regions, dst, layers, src_pages, dst_pages):
        buffers = [
            PeerBuffer(
                src_region=regions[layer % len(regions)],
                src_offset=_src_offset(layer, cls),
                dst_base=dst[layer][cls].data_ptr(),
                page_bytes=pb,
                dst_num_pages=CAPACITY,
            )
            for layer in layers
            for cls, pb in enumerate(PAGE_BYTES)
        ]
        return ReadGroup(buffers=buffers, src_pages=src_pages, dst_pages=dst_pages)

    def _check(self, dst, src_pages, dst_pages, layers):
        copied = set(dst_pages.tolist())
        for layer in layers:
            for cls, pb in enumerate(PAGE_BYTES):
                for sp, dp in zip(src_pages.tolist(), dst_pages.tolist()):
                    base = _src_offset(layer, cls) + sp * pb
                    want = _pattern(torch.arange(base, base + pb, device="cuda:1"))
                    self.assertTrue(
                        torch.equal(dst[layer][cls][dp], want), (layer, cls, sp, dp)
                    )
                untouched = [p for p in range(CAPACITY) if p not in copied]
                self.assertTrue(bool((dst[layer][cls][untouched] == SENTINEL).all()))

    def _read_and_check(self, reader, regions):
        rng = np.random.default_rng(0)

        # Two groups (layers 0-1, layer 2), random pages, layers alternating between
        # the two regions; the rest keep the sentinel.
        dst = self._destinations()
        # No producer stream: the caller vouches the targets are ready.
        torch.cuda.synchronize(1)
        src_pages = rng.choice(CAPACITY, 20, replace=False)
        dst_pages = rng.choice(CAPACITY, 20, replace=False)
        groups = [
            self._group(regions, dst, [0, 1], src_pages, dst_pages),
            self._group(regions, dst, [2], src_pages, dst_pages),
        ]
        batch = reader.read(groups)
        status = reader.wait(batch, timeout=30)
        self.assertEqual(status.state, ReadState.SUCCESS)
        self.assertEqual(status.groups_done, 2)
        self._check(dst, src_pages, dst_pages, range(LAYERS))
        reader.release(batch)

        # Every page, reversed, behind a producer stream that still writes the targets.
        dst = self._destinations()
        producer = torch.cuda.Stream(device=1)
        with torch.cuda.stream(producer):
            torch.cuda._sleep(10_000_000)
            for layer in dst:
                for buf in layer:
                    buf.fill_(SENTINEL)
        pages = np.arange(CAPACITY)
        batch = reader.read(
            [self._group(regions[:1], dst, range(LAYERS), pages, pages[::-1].copy())],
            producer_stream=producer,
        )
        self.assertEqual(reader.wait(batch, timeout=30).state, ReadState.SUCCESS)
        self._check(dst, pages, pages[::-1].copy(), range(LAYERS))
        reader.release(batch)

    def test_export_import_read_release(self):
        ctx = mp.get_context("spawn")
        conn, child = ctx.Pipe()
        owner = ctx.Process(target=_owner, args=(child,))
        owner.start()
        child.close()  # so recv() raises EOFError if the owner dies
        try:
            # GPU 0 stays current: the reader must not depend on the caller's device.
            torch.cuda.set_device(0)
            self.assertTrue(conn.poll(300), "the owner sent no descriptors")
            descs = [PeerRegionDescriptor.from_bytes(d) for d in conn.recv()]
            for copy_engine in (False, True):
                with self.subTest(copy_engine=copy_engine):
                    reader = KvPeerReader(1, num_blocks=16, copy_engine=copy_engine)
                    regions = [reader.import_region(desc) for desc in descs]
                    self._read_and_check(reader, regions)
                    for region in regions:
                        reader.unmap(region)
            conn.send("unmapped")
            self.assertEqual(conn.recv(), "ok")
        finally:
            owner.join(timeout=60)
            if owner.is_alive():
                owner.kill()
        self.assertEqual(owner.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
