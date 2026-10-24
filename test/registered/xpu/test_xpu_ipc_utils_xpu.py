"""XPU device memory shared between two processes over Level Zero IPC.

The handle server, the pool classes' failure paths and the probe are covered
without a GPU in test/registered/unit/utils/test_xpu_ipc_utils.py.

Usage:
python3 -m unittest test_xpu_ipc_utils_xpu
"""

import multiprocessing as mp
import queue
import traceback
import unittest

import torch

from sglang.srt.utils.xpu_ipc_utils import (
    LevelZeroIpcContext,
    XpuIpcExportedPool,
    XpuIpcPeerPools,
    ensure_xpu_peer_access,
    is_xpu_ipc_available,
    tensor_from_device_pointer,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=45, suite="stage-a-test-1-gpu-xpu")
# The per-PR runner has one card, so test_across_cards runs only here.
register_xpu_ci(est_time=45, suite="nightly-xpu-kernel-main-2-gpu", nightly=True)

# Not a multiple of the page size, so a driver that rounds the allocation up
# still has to report a segment that covers it.
_SEGMENT_BYTES = (4 << 20) + 4096 + 1
_TIMEOUT_S = 60
_JOIN_TIMEOUT_S = 30


def _pattern(num_bytes: int, seed: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(num_bytes, dtype=torch.int32, device=device)
    return ((values * 31 + seed) % 251).to(torch.uint8)


def _export_two_patterns(gpu, to_importer, from_importer, results):
    """Export a pool, then write two patterns the importer reads through one mapping."""
    try:
        torch.xpu.set_device(gpu)
        device = torch.device("xpu", gpu)
        context = LevelZeroIpcContext()
        pool = XpuIpcExportedPool(context, gpu, _SEGMENT_BYTES)
        try:
            pool.tensor.copy_(_pattern(_SEGMENT_BYTES, 1, device))
            torch.xpu.synchronize(gpu)
            to_importer.put(pool.reference)

            assert from_importer.get(timeout=_TIMEOUT_S) == "read-1"
            pool.tensor.copy_(_pattern(_SEGMENT_BYTES, 2, device))
            torch.xpu.synchronize(gpu)
            to_importer.put("written-2")
            assert from_importer.get(timeout=_TIMEOUT_S) == "read-2"
        finally:
            pool.close()
            context.destroy()
        results.put(("exporter", None))
    except BaseException:
        results.put(("exporter", traceback.format_exc()))


def _import_two_patterns(gpu, from_exporter, to_exporter, results):
    try:
        torch.xpu.set_device(gpu)
        device = torch.device("xpu", gpu)
        context = LevelZeroIpcContext()
        pools = XpuIpcPeerPools(context)
        try:
            reference = from_exporter.get(timeout=_TIMEOUT_S)
            ensure_xpu_peer_access(reference.device_index, gpu)
            # clone() copies into memory this process' allocator owns.
            first = pools.open(reference, gpu).clone()
            torch.xpu.synchronize(gpu)
            to_exporter.put("read-1")
            assert from_exporter.get(timeout=_TIMEOUT_S) == "written-2"
            # The second open reuses the first mapping.
            second = pools.open(reference, gpu).clone()
            torch.xpu.synchronize(gpu)
            to_exporter.put("read-2")
            assert torch.equal(first, _pattern(_SEGMENT_BYTES, 1, device))
            assert torch.equal(second, _pattern(_SEGMENT_BYTES, 2, device))
        finally:
            pools.close()
            context.destroy()
        results.put(("importer", None))
    except BaseException:
        results.put(("importer", traceback.format_exc()))


class TestXpuIpcRoundTrip(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not is_xpu_ipc_available():
            raise unittest.SkipTest("requires an Intel GPU with Level Zero IPC")

    def _round_trip(self, exporter_gpu: int, importer_gpu: int) -> None:
        context = mp.get_context("spawn")
        to_importer = context.Queue()
        to_exporter = context.Queue()
        results = context.Queue()
        processes = [
            context.Process(
                target=_export_two_patterns,
                args=(exporter_gpu, to_importer, to_exporter, results),
            ),
            context.Process(
                target=_import_two_patterns,
                args=(importer_gpu, to_importer, to_exporter, results),
            ),
        ]
        for process in processes:
            process.start()
        failures = {}
        try:
            for _ in processes:
                name, failure = results.get(timeout=2 * _TIMEOUT_S)
                failures[name] = failure
                if failure is not None:
                    break
        except queue.Empty:
            self.fail(f"timed out; reported so far: {failures}")
        finally:
            reported_all = len(failures) == len(processes)
            for process in processes:
                if not reported_all:
                    # A side that has not reported only waits on the one that failed.
                    process.terminate()
                process.join(timeout=_JOIN_TIMEOUT_S)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=_JOIN_TIMEOUT_S)
        for name, failure in failures.items():
            self.assertIsNone(failure, f"the {name} failed:\n{failure}")
        self.assertEqual(sorted(failures), ["exporter", "importer"])
        self.assertEqual([process.exitcode for process in processes], [0, 0])

    def test_same_card(self):
        self._round_trip(0, 0)

    def test_across_cards(self):
        if torch.xpu.device_count() < 2:
            self.skipTest("requires two Intel GPUs")
        self._round_trip(0, 1)


class TestTensorFromDevicePointer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not is_xpu_ipc_available():
            raise unittest.SkipTest("requires an Intel GPU with Level Zero IPC")

    def test_the_tensor_aliases_the_memory(self):
        context = LevelZeroIpcContext()
        self.addCleanup(context.destroy)
        pointer = context.allocate_exportable(0, _SEGMENT_BYTES)
        self.addCleanup(context.free, pointer)

        whole = tensor_from_device_pointer(pointer, _SEGMENT_BYTES, 0)
        self.assertEqual(whole.device, torch.device("xpu", 0))
        self.assertEqual(whole.dtype, torch.uint8)
        self.assertEqual(whole.numel(), _SEGMENT_BYTES)
        # data_ptr() reports the pointer as a signed int64.
        self.assertEqual(whole.data_ptr() % (1 << 64), pointer)

        # Any pointer into the allocation exports the whole allocation.
        segment = context.export_segment(pointer + 4096)
        context.release_segment(segment)
        self.assertEqual(segment.base_pointer, pointer)
        self.assertGreaterEqual(segment.segment_bytes, _SEGMENT_BYTES)

        view = whole[4096:8192]
        del whole
        view.fill_(7)
        again = tensor_from_device_pointer(pointer, _SEGMENT_BYTES, 0)
        torch.xpu.synchronize(0)
        self.assertEqual(int(again[4096]), 7)
        self.assertEqual(int(again[8191]), 7)


if __name__ == "__main__":
    unittest.main()
