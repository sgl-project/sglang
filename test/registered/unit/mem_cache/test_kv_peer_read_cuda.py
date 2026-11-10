"""Peer KV page reads between two processes over a CUDA FABRIC export, byte-checked."""

import multiprocessing as mp
import unittest

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci

register_cuda_ci(est_time=40, stage="base-c", runner_config="2-gpu-gb300")

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
    from cuda.bindings import driver as drv

    from sglang.srt.mem_cache.kv_peer_read import KvPeerExporter
    from sglang.srt.utils.cuda_vmm_utils import (
        align_up,
        check_drv,
        get_allocation_granularity,
        make_device_allocation_prop,
        make_rw_access_desc,
        tensor_from_pointer,
    )

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

    exporter = KvPeerExporter(0)
    # Two leases on one allocation: the reader maps it twice, as two regions.
    descs = [exporter.export(handle, size, region_id="kv-pool") for _ in range(2)]
    conn.send([d.to_bytes() for d in descs])
    assert conn.recv() == "unmapped"  # the reader's ACK: drained, released, unmapped
    for desc in descs:
        exporter.release_export(desc.lease_id)
    assert exporter.outstanding() == 0
    check_drv(drv.cuMemUnmap(va, size), "unmap")
    check_drv(drv.cuMemAddressFree(va, size), "free")
    check_drv(drv.cuMemRelease(handle), "release")
    conn.send("ok")


class TestKvPeerReadCuda(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils.cuda_vmm_utils import get_gpu_fabric_clique

        if torch.cuda.device_count() >= 2 and get_gpu_fabric_clique(0) is not None:
            return
        reason = "needs two GPUs in an NVLink fabric clique"
        if (
            is_in_ci()
        ):  # the registered runner has one: a skip there hides a broken setup
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
        from sglang.srt.mem_cache.kv_peer_read import PeerBuffer, ReadGroup

        buffers = tuple(
            PeerBuffer(
                src_region=regions[layer % len(regions)],
                src_offset=_src_offset(layer, cls),
                dst_base=dst[layer][cls].data_ptr(),
                page_bytes=pb,
                dst_num_pages=CAPACITY,
            )
            for layer in layers
            for cls, pb in enumerate(PAGE_BYTES)
        )
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

    def test_export_import_read_release(self):
        from sglang.srt.mem_cache.kv_peer_read import (
            KvPeerReader,
            PeerRegionDescriptor,
            ReadState,
        )

        ctx = mp.get_context("spawn")
        conn, child = ctx.Pipe()
        owner = ctx.Process(target=_owner, args=(child,))
        owner.start()
        try:
            # GPU 0 stays current: the reader must not depend on the caller's device.
            torch.cuda.set_device(0)
            reader = KvPeerReader(1, num_blocks=16)
            regions = [
                reader.import_region(PeerRegionDescriptor.from_bytes(d))
                for d in conn.recv()
            ]
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
            self.assertEqual(reader.wait(batch, timeout=30).state, ReadState.SUCCESS)
            self.assertTrue(reader.group_done(batch, 0))
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
                [
                    self._group(
                        regions[:1], dst, range(LAYERS), pages, pages[::-1].copy()
                    )
                ],
                producer_stream=producer,
            )
            self.assertEqual(reader.wait(batch, timeout=30).state, ReadState.SUCCESS)
            self._check(dst, pages, pages[::-1].copy(), range(LAYERS))
            reader.release(batch)

            for region in regions:
                reader.unmap(region)
            reader.close()
            conn.send("unmapped")
            self.assertEqual(conn.recv(), "ok")
        finally:
            owner.join(timeout=60)
            if owner.is_alive():
                owner.kill()
        self.assertEqual(owner.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
