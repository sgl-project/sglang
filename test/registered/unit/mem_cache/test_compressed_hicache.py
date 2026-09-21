"""Real tensor/storage/worker tests, plus tree-adapter transaction tests.

SGLANG_COMPRESSION_STANDALONE_TEST=1 bypasses only package __init__ files on a
CPU-only workstation. Compression modules are imported in full; selected
scheduler method bodies are loaded from source for accounting integration tests.
CUDA/nvCOMP/RDMA and the full scheduler require the separate GPU/E2E tests.
"""

import ast
import concurrent.futures
import os
import sys
import threading
import time
import types
import unittest
import weakref
import zlib
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
if os.environ.get("SGLANG_COMPRESSION_STANDALONE_TEST") == "1":
    for name, relative in [
        ("sglang", "python/sglang"),
        ("sglang.srt", "python/sglang/srt"),
        ("sglang.srt.mem_cache", "python/sglang/srt/mem_cache"),
        (
            "sglang.srt.mem_cache.unified_cache",
            "python/sglang/srt/mem_cache/unified_cache",
        ),
    ]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT / relative)]
            sys.modules[name] = module

from sglang.srt.disaggregation.compression.protocol import ChunkDescriptor
from sglang.srt.kv_compression.host_io import CompressedHostIO
from sglang.srt.kv_compression.layout import KVLayoutAdapter
from sglang.srt.kv_compression.provider import HostEncodedKVProvider, RepresentationSpec
from sglang.srt.kv_compression.runtime import KVCompressionRuntime
from sglang.srt.kv_compression.store import (
    BLOCK_BYTES,
    materialize_pages,
)
from sglang.srt.kv_compression.store import (
    CompressedHostKVCache as BlockPool,
)
from sglang.srt.kv_compression.types import (
    BufferDrainError,
    CompressionCapacityError,
    KVVerificationError,
    new_page_refs,
)
from sglang.srt.kv_compression.verification import page_digests
from sglang.srt.mem_cache.hicache_lifecycle import HiCacheLifecycleMixin
from sglang.srt.mem_cache.l2_completion import (
    AsyncL2State,
    RestoreTransferResult,
    TransferCompletion,
    TransferState,
    exceeds_load_quota,
    record_load_back_metrics,
    should_skip_full_load,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


class CompressedHostKVCache(BlockPool):
    """Fixture adapter only: old control tests seed real block I/O synchronously.

    Small fixture budgets include additional staging space; production budget
    tests below instantiate BlockPool directly without any adjustment.
    """

    def __init__(self, raw, budget, **kwargs):
        bound = kwargs.get("reservation_bytes") or raw
        stage = 3 * 64 * ((bound + 4095) // 4096 * 4096)
        super().__init__(raw, budget + stage + 1024**2, **kwargs)
        self.pending_test_payloads = {}

    def destination(self, handle, length):
        value = torch.empty(length, dtype=torch.uint8)
        self.pending_test_payloads[int(handle)] = value
        return value

    def publish(self, handle, ref, encoding, length, raw_sha256=None):
        from sglang.srt.kv_compression.types import EncodedPage

        if self.verify and (raw_sha256 is None or len(raw_sha256) != 32):
            raise ValueError("SHA256 required")
        data = self.pending_test_payloads.pop(
            int(handle), torch.zeros(length, dtype=torch.uint8)
        )
        write = self.prepare_write([handle], [length])
        try:
            with self.stage(read=False) as stage:
                stage[0, :length].copy_(data)
                write.scatter(stage)
                write.publish(
                    [ref],
                    [EncodedPage(data, encoding, self.size_per_token)],
                    [raw_sha256],
                )
        finally:
            write.close()


def payload(page):
    with materialize_pages([page]) as pages:
        return pages[0].data.clone()


def layout(count=8):
    # Arbitrary bit patterns include NaNs; byte equality matters, not allclose.
    buffers = [
        torch.randint(0, 256, (count, 4, 32), dtype=torch.uint8).view(torch.bfloat16)
        for _ in range(4)
    ]
    return KVLayoutAdapter(buffers[:2], buffers[2:])


def result(value=None, error=None):
    future = concurrent.futures.Future()
    if error is None:
        future.set_result(value)
    else:
        future.set_exception(error)
    return future


def source_function(path, name, owner=None, **namespace):
    """Execute actual scheduler method bodies without importing CUDA modules."""
    namespace.setdefault("time", time)
    namespace.setdefault("should_skip_full_load", should_skip_full_load)
    namespace.setdefault("exceeds_load_quota", exceeds_load_quota)
    namespace.setdefault("record_load_back_metrics", record_load_back_metrics)
    path = ROOT / "python/sglang/srt" / path
    nodes = ast.parse(path.read_text()).body
    if owner:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []  # Rank consensus is outside these single-rank tests.
    module = ast.parse("from __future__ import annotations")
    module.body.append(node)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.pool = CompressedHostKVCache(4096, 256 * 1024, pin_memory=False)

    def publish(self, count=1, length=256, encoding="lz4"):
        handles = self.pool.alloc(count)
        refs = new_page_refs(count)
        for handle, ref in zip(handles.tolist(), refs):
            self.pool.destination(handle, length).fill_(ref % 255)
            self.pool.publish(handle, ref, encoding, length)
        return handles, refs

    def test_actual_bytes_release_reservations(self):
        handles, _ = self.publish(4)
        stats = self.pool.snapshot()
        self.assertEqual(stats["payload_bytes"], 4 * BLOCK_BYTES)
        self.assertEqual(stats["reserved_bytes"], 0)
        self.assertLessEqual(
            stats["arena_bytes"] + stats["metadata_bytes"], stats["budget_bytes"]
        )
        self.pool.free(handles)
        self.assertEqual(self.pool.snapshot()["payload_bytes"], 0)
        self.assertEqual(self.pool.free_block_count, self.pool.block_count)

    def test_capacity_greater_than_raw_without_allocating_native_tensor(self):
        handles = []
        while self.pool.can_reserve(1):
            h, _ = self.publish()
            handles.extend(h.tolist())
        self.assertEqual(len(handles), self.pool.arena.numel() // BLOCK_BYTES)
        self.assertFalse(hasattr(self.pool, "kv_buffer"))
        self.pool.free(handles)
        self.assertEqual(
            self.pool.available_size(),
            min(self.pool.size, self.pool.arena.numel() // self.pool.reservation_bytes),
        )

    def test_retired_objects_wait_for_reader(self):
        h, refs = self.publish()
        lease = self.pool.acquire_ref(refs[0], "lz4")
        expected = payload(lease.future.result()).clone()
        self.pool.free(h)
        self.assertIsNone(self.pool.acquire_ref(refs[0], "lz4"))
        self.assertEqual(self.pool.snapshot()["retired_bytes"], BLOCK_BYTES)
        torch.testing.assert_close(payload(lease.future.result()), expected)
        with self.assertRaises(RuntimeError):
            self.pool.clear()
        lease.close()
        lease.close()
        self.assertEqual(self.pool.snapshot()["retired_bytes"], 0)

    def test_stale_handle_after_reuse_and_clear(self):
        h, _ = self.publish()
        old = h[0].item()
        self.pool.clear()
        new, _ = self.publish()
        self.assertNotEqual(old, new[0].item())
        with self.assertRaises(ValueError):
            self.pool.acquire(old)
        with self.assertRaises(ValueError):
            self.pool.free([old])
        self.assertEqual(self.pool.snapshot()["used_pages"], 1)

    def test_no_partial_free_on_invalid_input(self):
        h, _ = self.publish(2)
        with self.assertRaises(ValueError):
            self.pool.free([h[0].item(), -1])
        self.assertEqual(self.pool.snapshot()["used_pages"], 2)
        with self.assertRaises(ValueError):
            self.pool.free([h[0].item()] * 2)

    def test_fragmentation_and_coalescing(self):
        h, _ = self.publish(20)
        self.pool.free(h[::2])
        more, _ = self.publish(10, 1024)
        self.pool.free(h[1::2])
        self.pool.free(more)
        self.assertEqual(self.pool.free_block_count, self.pool.block_count)
        self.assertEqual(
            self.pool.free_block_count * BLOCK_BYTES, self.pool.arena.numel()
        )

    def test_randomized_reclamation_preserves_byte_ledger(self):
        rng = np.random.default_rng(11)
        live = []
        for _ in range(300):
            if live and rng.integers(0, 2):
                self.pool.free([live.pop(int(rng.integers(len(live))))])
            elif self.pool.can_reserve(1):
                h, _ = self.publish(length=int(rng.integers(1, 4097)))
                live.append(h.item())
            s = self.pool.snapshot()
            free = self.pool.free_block_count * BLOCK_BYTES
            self.assertEqual(
                free + s["payload_bytes"] + s["reserved_bytes"] + s["retired_bytes"],
                self.pool.arena.numel(),
            )
        self.pool.free(live)
        self.assertEqual(self.pool.free_block_count, self.pool.block_count)

    def test_raw_mode_and_legacy_io_rejection(self):
        h, refs = self.publish(length=4096, encoding="raw")
        lease = self.pool.acquire_ref(refs[0], "passthrough")
        self.assertEqual(lease.future.result().encoding, "raw")
        lease.close()
        with self.assertRaises(RuntimeError):
            self.pool.get_data_page(h)

    def test_unpublished_reservation_is_not_a_hit(self):
        h = self.pool.alloc(1)
        with self.assertRaises(ValueError):
            self.pool.acquire(h.item())
        self.pool.free(h)
        self.assertEqual(self.pool.reserved_bytes, 0)

    def test_hash_churn_and_collision_preserve_live_readers(self):
        # All these refs collide, including at the wraparound boundary.
        stride = len(self.pool.keys)
        for cycle in range(400):
            handles = self.pool.alloc(3)
            refs = [1 + stride * (3 * cycle + j) for j in range(3)]
            for handle, ref in zip(handles.tolist(), refs):
                self.pool.destination(handle, 4096).fill_(cycle % 256)
                self.pool.publish(handle, ref, "raw", 4096)
            self.pool.free(handles[1:2])
            for ref in (refs[0], refs[2]):
                lease = self.pool.acquire_ref(ref, "lz4")
                self.assertEqual(payload(lease.future.result())[0].item(), cycle % 256)
                lease.close()
            self.pool.free(handles[[0, 2]])
            self.assertIsNone(self.pool.acquire_ref(refs[1], "lz4"))
        self.assertEqual(
            self.pool.available_size(),
            min(self.pool.size, self.pool.arena.numel() // self.pool.reservation_bytes),
        )


class VerifiedStoreTests(unittest.TestCase):
    def test_expansion_reservation_checksum_and_generation(self):
        pool = CompressedHostKVCache(
            4096,
            256 * 1024,
            reservation_bytes=4609,
            verify=True,
            pin_memory=False,
        )
        self.assertEqual(pool.reservation_bytes, 8192)
        self.assertLessEqual(
            pool.metadata_bytes + pool.arena.numel(), pool.budget_bytes
        )
        h = pool.alloc(1).item()
        ref = new_page_refs(1)[0]
        pool.destination(h, 4200).fill_(7)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            pool.publish(h, ref, "lz4", 4200)
        self.assertIsNone(pool.acquire_ref(ref, "lz4"))
        checksum = bytes(range(32))
        pool.publish(h, ref, "lz4", 4200, checksum)
        lease = pool.acquire(h)
        self.assertEqual(lease.future.result().raw_sha256, checksum)
        self.assertEqual(pool.reserved_bytes, 0)
        self.assertEqual(pool.live_payload_bytes, 8192)
        pool.free([h])
        self.assertEqual(lease.future.result().raw_sha256, checksum)
        lease.close()
        self.assertFalse(pool.checksums.any())
        new = pool.alloc(1).item()
        self.assertNotEqual(h, new)
        with self.assertRaises(ValueError):
            pool.acquire(h)
        pool.free([new])
        self.assertEqual(pool.free_block_count, pool.block_count)

    def test_fragmented_admission_uses_output_bound(self):
        pool = CompressedHostKVCache(
            4096, 128 * 1024, reservation_bytes=8192, pin_memory=False
        )
        count = pool.arena.numel() // pool.reservation_bytes
        handles = pool.alloc(count)
        self.assertIsNotNone(handles)
        self.assertFalse(pool.can_reserve(1))
        pool.free(handles[::2])
        self.assertEqual(pool.can_reserve(len(handles[::2])), True)
        self.assertFalse(pool.can_reserve(len(handles[::2]) + 1))
        pool.free(handles[1::2])


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.layout = layout()
        self.runtime = KVCompressionRuntime(self.layout, "passthrough", 1024 * 1024)

    def tearDown(self):
        self.runtime.close()

    def test_pack_restore_preserves_arbitrary_bf16_bits(self):
        expected = self.layout.pack_pages([5, 1, 7]).clone()
        self.layout.unpack_pages(expected, [0, 2, 3])
        self.assertTrue(torch.equal(expected, self.layout.pack_pages([0, 2, 3])))

    def test_existing_scatter_layout_matches_slot_major_bytes(self):
        raw = self.layout.pack_pages([2, 0])
        expected = torch.cat(
            [t[[2, 0]].view(torch.uint8).flatten() for t in self.layout.buffers]
        )
        self.assertTrue(torch.equal(expected, self.layout.to_staging_order(raw)))

    def test_shared_task_encodes_once_and_release_is_independent(self):
        refs = new_page_refs(3)
        first = self.runtime.acquire_pages(refs, [0, 1, 2])
        second = self.runtime.acquire_pages(refs, [0, 1, 2])
        for a, b in zip(first, second):
            self.assertIs(a.future, b.future)
            a.future.result(timeout=5)
            a.close()
            self.assertGreater(b.future.result().nbytes, 0)
            b.close()
        self.assertEqual(self.runtime.stats["encoded_pages"], 3)
        self.assertEqual(self.runtime.stats["raw_fallback_pages"], 0)
        self.assertEqual(self.runtime._live_bytes, 0)

    def test_host_only_restore_and_send_reuse_no_reencode(self):
        pool = CompressedHostKVCache(self.layout.page_bytes, 65536, pin_memory=False)
        self.runtime.provider = HostEncodedKVProvider(
            pool, RepresentationSpec(self.runtime.layout.tag, self.runtime.mode)
        )
        refs = new_page_refs(3)
        expected = self.layout.pack_pages([0, 1, 2]).clone()
        source = self.runtime.acquire_pages(refs, [0, 1, 2])
        handles = pool.alloc(3)
        for h, ref, lease in zip(handles.tolist(), refs, source):
            page = lease.future.result(timeout=5)
            pool.destination(h, page.nbytes).copy_(page.data)
            pool.publish(h, ref, page.encoding, page.nbytes)
            lease.close()
        # Destroy the native copy: this is an actual L2-only restoration.
        for tensor in self.layout.buffers:
            tensor.zero_()
        before = self.runtime.stats["encoded_pages"]
        restored = self.runtime.acquire_pages(refs, [0, 1, 2])
        sending = self.runtime.acquire_pages(refs, [0, 1, 2])
        self.runtime.submit(
            lambda: self.runtime.restore(restored, [0, 1, 2]), 0
        ).result(timeout=5)
        self.assertTrue(torch.equal(expected, self.layout.pack_pages([0, 1, 2])))
        self.assertEqual(self.runtime.stats["encoded_pages"], before)
        for lease in restored + sending:
            lease.close()
        pool.clear()

    def test_recomputed_materialization_does_not_reuse_old_object(self):
        pool = CompressedHostKVCache(self.layout.page_bytes, 65536, pin_memory=False)
        self.runtime.provider = HostEncodedKVProvider(
            pool, RepresentationSpec(self.runtime.layout.tag, self.runtime.mode)
        )
        (old_ref,) = new_page_refs(1)
        handle = pool.alloc(1).item()
        old = self.layout.pack_pages([0])[0]
        pool.destination(handle, old.numel()).copy_(old)
        pool.publish(handle, old_ref, "raw", old.numel())
        for t in self.layout.buffers:
            t[0].zero_()
        (lease,) = self.runtime.acquire_pages(new_page_refs(1), [0])
        self.assertTrue(
            torch.equal(lease.future.result(timeout=5).data, torch.zeros_like(old))
        )
        self.assertEqual(self.runtime.stats["encoded_pages"], 1)
        lease.close()

    def test_budget_failure_is_terminal_and_releases_references(self):
        runtime = KVCompressionRuntime(self.layout, "passthrough", 1)
        try:
            (lease,) = runtime.acquire_pages(new_page_refs(1), [0])
            with self.assertRaises(CompressionCapacityError):
                lease.future.result(timeout=5)
            lease.close()
            self.assertEqual(runtime._live_bytes, 0)
            self.assertFalse(runtime._entries)
        finally:
            runtime.close()

    def test_cancel_one_consumer_does_not_cancel_other(self):
        gate = threading.Event()
        self.runtime.submit(lambda: gate.wait(5), 0)
        refs = new_page_refs(1)
        (first,) = self.runtime.acquire_pages(refs, [0])
        (second,) = self.runtime.acquire_pages(refs, [0])
        first.close()
        gate.set()
        second.future.result(timeout=5)
        second.close()
        self.assertEqual(self.runtime.stats["encoded_pages"], 1)

    def test_completed_job_does_not_retain_result(self):
        (lease,) = self.runtime.acquire_pages(new_page_refs(1), [0])
        page = lease.future.result(timeout=5)
        reference = weakref.ref(page)
        del page
        lease.close()
        self.runtime.submit(lambda: None).result(timeout=5)
        self.assertIsNone(reference())

    def test_foreground_reuse_promotes_queued_background_task(self):
        gate, running = threading.Event(), threading.Event()

        def block():
            running.set()
            gate.wait(5)

        self.runtime.submit(block, -1)
        self.assertTrue(running.wait(5))
        refs = new_page_refs(1)
        (background,) = self.runtime.acquire_pages(refs, [0], priority=2)
        order = []
        background.future.add_done_callback(lambda _: order.append("restored"))
        middle = self.runtime.submit(lambda: order.append("background"), 1)
        (foreground,) = self.runtime.acquire_pages(refs, [0], priority=0)
        gate.set()
        middle.result(timeout=5)
        self.assertEqual(order, ["restored", "background"])
        background.close()
        foreground.close()

    def test_cancelled_task_does_not_kill_shared_worker(self):
        gate, running = threading.Event(), threading.Event()
        first = self.runtime.submit(lambda: (running.set(), gate.wait(5)), -1)
        self.assertTrue(running.wait(5))
        fn = Mock()
        cancelled = self.runtime.submit(fn)
        self.assertTrue(cancelled.cancel())
        gate.set()
        first.result(timeout=5)
        self.assertEqual(self.runtime.submit(lambda: 42).result(timeout=5), 42)
        fn.assert_not_called()

    def test_queue_rejection_creates_no_orphan_entries(self):
        gate, running = threading.Event(), threading.Event()

        def block():
            running.set()
            gate.wait(5)

        self.runtime.submit(block, -1)
        self.assertTrue(running.wait(5))
        try:
            for _ in range(256):
                self.runtime.submit(lambda: None)
            with self.assertRaises(CompressionCapacityError):
                self.runtime.acquire_pages(new_page_refs(1), [0])
            self.assertFalse(self.runtime._entries)
        finally:
            gate.set()

    def test_acquire_error_drains_submitted_source_reads_before_returning(self):
        started, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        original_pack = self.layout.pack_pages
        errors = []

        def slow_pack(indices):
            started.set()
            if not release.wait(5):
                raise RuntimeError("test did not release source read")
            return original_pack(indices)

        def acquire():
            try:
                self.runtime.acquire_pages(new_page_refs(1), [0])
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()

        self.runtime.trace_reuse = True
        with (
            patch.object(self.layout, "pack_pages", side_effect=slow_pack),
            patch(
                "sglang.srt.kv_compression.runtime.logger.info",
                side_effect=ValueError("log sink failed"),
            ),
        ):
            caller = threading.Thread(target=acquire)
            caller.start()
            try:
                self.assertTrue(started.wait(5))
                self.assertFalse(
                    finished.wait(0.05),
                    "source owner could release while pack still reads",
                )
            finally:
                release.set()
                caller.join(5)
        self.assertFalse(caller.is_alive())
        self.assertIsInstance(errors[0], ValueError)
        self.runtime.submit(lambda: None).result(timeout=5)
        self.assertEqual(self.runtime._live_bytes, 0)
        self.assertFalse(self.runtime._entries)

    def test_quarantine_stops_already_queued_encode_and_restore(self):
        started, release = threading.Event(), threading.Event()

        def fail_drain():
            started.set()
            release.wait(5)
            raise BufferDrainError("injected uncertain stream")

        first = self.runtime.submit(fail_drain, -1)
        self.assertTrue(started.wait(5))
        with patch.object(
            self.layout, "pack_pages", wraps=self.layout.pack_pages
        ) as pack:
            leases = self.runtime.acquire_pages(new_page_refs(1), [0])
            restore_body = Mock()
            restore = self.runtime.submit(restore_body, 0)
            release.set()
            with self.assertRaises(BufferDrainError):
                first.result(timeout=5)
            with self.assertRaises(BufferDrainError):
                restore.result(timeout=5)
            with self.assertRaises(BufferDrainError):
                leases[0].future.result(timeout=5)
            pack.assert_not_called()
            restore_body.assert_not_called()
        self.assertGreater(self.runtime.snapshot()["quarantined"], 0)
        leases[0].close()

    def test_restore_rejects_pending_encode_without_blocking_sole_worker(self):
        gate, running = threading.Event(), threading.Event()
        self.runtime.submit(lambda: (running.set(), gate.wait(5)), -1)
        self.assertTrue(running.wait(5))
        leases = self.runtime.acquire_pages(new_page_refs(1), [0], priority=2)
        restore = self.runtime.submit(lambda: self.runtime.restore(leases, [1]), 0)
        gate.set()
        try:
            with self.assertRaisesRegex(RuntimeError, "completed object leases"):
                restore.result(timeout=5)
            leases[0].future.result(timeout=5)
            self.assertEqual(self.runtime.submit(lambda: 42).result(timeout=5), 42)
        finally:
            leases[0].close()

    def test_unpack_failure_drains_and_releases_working_budget(self):
        leases = self.runtime.acquire_pages(new_page_refs(1), [0])
        leases[0].future.result(timeout=5)
        before = self.runtime._live_bytes
        with (
            patch.object(
                self.layout, "unpack_pages", side_effect=ValueError("bad target")
            ),
            patch.object(self.runtime, "drain", wraps=self.runtime.drain) as drain,
        ):
            with self.assertRaises(ValueError):
                self.runtime.submit(lambda: self.runtime.restore(leases, [0])).result(
                    timeout=5
                )
            self.assertGreaterEqual(drain.call_count, 2)
            self.assertEqual(self.runtime._live_bytes, before)
        leases[0].close()


class _TestBatchBackend:
    """CPU double for batch failure/fallback logic, NOT a supported backend."""

    def __init__(self, *_):
        pass

    def max_output_bytes(self, source):
        return source.numel() + 512

    def compress_batch(self, sources, outputs):
        lengths = []
        for source, output in zip(sources, outputs):
            data = zlib.compress(source.numpy().tobytes())
            output[: len(data)].copy_(
                torch.frombuffer(bytearray(data), dtype=torch.uint8)
            )
            lengths.append(len(data))
        return lengths

    def decompress_batch(self, sources, outputs):
        for source, output in zip(sources, outputs):
            data = zlib.decompress(source.numpy().tobytes())
            output.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))


class BatchedTransformTests(unittest.TestCase):
    def test_verified_forced_l2_source_eviction_and_reuse(self):
        model = layout()
        runtime = KVCompressionRuntime(
            model,
            "lz4",
            1024 * 1024,
            _TestBatchBackend,
            force=True,
            verify=True,
        )
        pool = CompressedHostKVCache(
            model.page_bytes,
            65536,
            pin_memory=False,
            verify=True,
            reservation_bytes=runtime.output_bound,
        )
        adapter = CompressedHostIO(runtime, pool)
        runtime.provider = HostEncodedKVProvider(
            pool, RepresentationSpec(runtime.layout.tag, runtime.mode)
        )
        source, target = [6, 2, 4], [1, 3, 5]
        expected = model.pack_pages(source).clone()
        handles, refs = pool.alloc(3), new_page_refs(3)
        leases = []
        try:
            adapter._write(handles.tolist(), refs, source, None)
            for handle, checksum in zip(handles.tolist(), page_digests(expected)):
                with pool.acquire(handle) as lease:
                    self.assertEqual(lease.future.result().raw_sha256, checksum)
            self.assertEqual(adapter.stats["published_lz4_pages"], 3)
            self.assertEqual(adapter.stats["verified_backup_pages"], 3)
            self.assertEqual(pool.reserved_bytes, 0)
            for tensor in model.buffers:
                tensor.zero_()
            leases = runtime.acquire_pages(refs, target)
            runtime.submit(lambda: runtime.restore(leases, target), 0).result(timeout=5)
            self.assertTrue(torch.equal(expected, model.pack_pages(target)))
            self.assertEqual(runtime.stats["verified_restore_pages"], 3)
            self.assertEqual(runtime.stats["encoded_pages"], 3)
            for lease in leases:
                lease.close()
            leases = runtime.acquire_pages(refs, target)
            self.assertEqual(runtime.stats["encoded_pages"], 3)
            self.assertEqual(runtime.stats["reused_host_pages"], 6)
            self.assertTrue(all(l.future.result().encoding == "lz4" for l in leases))
        finally:
            for lease in leases:
                lease.close()
            runtime.close()
            pool.clear()

    def test_verified_restore_detects_payload_and_target_corruption(self):
        model = layout()
        runtime = KVCompressionRuntime(model, "passthrough", 1024 * 1024, verify=True)
        pool = CompressedHostKVCache(
            model.page_bytes, 65536, pin_memory=False, verify=True
        )
        adapter = CompressedHostIO(runtime, pool)
        runtime.provider = HostEncodedKVProvider(
            pool, RepresentationSpec(runtime.layout.tag, runtime.mode)
        )
        refs, handles = new_page_refs(1), pool.alloc(1)
        leases = []
        try:
            adapter._write(handles.tolist(), refs, [0], None)
            leases = runtime.acquire_pages(refs, [1])
            first = int(pool.records[int(handles[0]) & 0xFFFFFFFF]["first"])
            data = pool.arena[first * BLOCK_BYTES : (first + 1) * BLOCK_BYTES]
            data[0] ^= 1
            with self.assertRaises(KVVerificationError):
                runtime.submit(lambda: runtime.restore(leases, [1]), 0).result(
                    timeout=5
                )
            data[0] ^= 1
            original_unpack = model.unpack_pages

            def corrupt_target(raw, indices):
                original_unpack(raw, indices)
                model.buffers[0][indices[0]].view(torch.uint8).flatten()[0] ^= 1

            with patch.object(model, "unpack_pages", side_effect=corrupt_target):
                with self.assertRaises(KVVerificationError):
                    runtime.submit(lambda: runtime.restore(leases, [1]), 0).result(
                        timeout=5
                    )
            self.assertEqual(runtime.stats["restore_verification_failures"], 2)
            self.assertEqual(runtime.stats["verified_restore_pages"], 0)
            runtime.submit(lambda: runtime.restore(leases, [1]), 0).result(timeout=5)
            self.assertEqual(runtime.stats["verified_restore_pages"], 1)
            self.assertEqual(runtime.snapshot()["resident_bytes"], 0)
        finally:
            for lease in leases:
                lease.close()
            runtime.close()
            pool.clear()

    def test_missing_checksum_never_writes_destination(self):
        model = layout()
        runtime = KVCompressionRuntime(model, "passthrough", 1024 * 1024, verify=True)
        leases = runtime.acquire_pages(new_page_refs(1), [0])
        leases[0].future.result(timeout=5)
        try:
            with patch.object(
                model, "unpack_pages", wraps=model.unpack_pages
            ) as unpack:
                with self.assertRaisesRegex(KVVerificationError, "Missing"):
                    runtime.submit(lambda: runtime.restore(leases, [1]), 0).result(
                        timeout=5
                    )
                unpack.assert_not_called()
        finally:
            for lease in leases:
                lease.close()
            runtime.close()

    def test_mixed_fallback_restore_and_cached_reuse(self):
        model = layout()
        for tensor in model.buffers:
            tensor[0].zero_()
        runtime = KVCompressionRuntime(model, "lz4", 1024 * 1024, _TestBatchBackend)
        pool = CompressedHostKVCache(model.page_bytes, 65536, pin_memory=False)
        runtime.provider = HostEncodedKVProvider(
            pool, RepresentationSpec(runtime.layout.tag, runtime.mode)
        )
        expected = model.pack_pages([0, 1]).clone()
        refs = new_page_refs(2)
        leases = runtime.acquire_pages(refs, [0, 1])
        try:
            pages = [lease.future.result(timeout=5) for lease in leases]
            self.assertEqual([p.encoding for p in pages], ["lz4", "raw"])
            handles = pool.alloc(2)
            for h, ref, p in zip(handles.tolist(), refs, pages):
                pool.destination(h, p.nbytes).copy_(p.data)
                pool.publish(h, ref, p.encoding, p.nbytes)
            for lease in leases:
                lease.close()
            for tensor in model.buffers:
                tensor.zero_()
            leases = runtime.acquire_pages(refs, [0, 1])
            runtime.submit(lambda: runtime.restore(leases, [0, 1]), 0).result(timeout=5)
            self.assertTrue(torch.equal(expected, model.pack_pages([0, 1])))
            self.assertEqual(runtime.stats["encoded_pages"], 2)
        finally:
            for lease in leases:
                lease.close()
            runtime.close()
            pool.clear()

    def test_forced_expansion_restores_actual_bytes(self):
        model = layout()
        runtime = KVCompressionRuntime(
            model, "lz4", 1024 * 1024, _TestBatchBackend, force=True
        )
        expected = model.pack_pages([0]).clone()
        leases = runtime.acquire_pages(new_page_refs(1), [0])
        try:
            page = leases[0].future.result(timeout=5)
            self.assertEqual(page.encoding, "lz4")
            self.assertGreater(page.nbytes, model.page_bytes)
            runtime.submit(lambda: runtime.restore(leases, [1]), 0).result(timeout=5)
            self.assertTrue(torch.equal(expected, model.pack_pages([1])))
            stats = runtime.snapshot()
            self.assertEqual(stats["attempted_encoded_bytes"], page.nbytes)
            self.assertGreater(stats["attempted_capacity_bytes"], page.nbytes)
        finally:
            for lease in leases:
                lease.close()
            runtime.close()

    def test_long_backup_keeps_one_payload_window(self):
        model = layout(8192)
        budget = 64 * model.page_bytes * 3
        runtime = KVCompressionRuntime(model, "passthrough", budget)
        pool = CompressedHostKVCache(
            model.page_bytes, 128 * 1024 * 1024, pin_memory=False
        )
        adapter = CompressedHostIO(runtime, pool)
        runtime.provider = HostEncodedKVProvider(
            pool, RepresentationSpec(runtime.layout.tag, runtime.mode)
        )
        refs = new_page_refs(8192)
        handles = pool.alloc(8192)
        try:
            adapter._write(handles.tolist(), refs, list(range(8192)), None)
            self.assertLessEqual(runtime.snapshot()["peak_bytes"], budget)
            self.assertEqual(runtime.snapshot()["inflight_objects"], 0)
            self.assertEqual(adapter.stats["backup_window_pages"], 64)
            self.assertEqual(pool.snapshot()["used_pages"], 8192)
            # Source eviction followed by real byte restoration at distant pages.
            expected = model.pack_pages([0, 4096, 8191]).clone()
            for tensor in model.buffers:
                tensor.zero_()
            leases = [pool.acquire(handles[i].item()) for i in [0, 4096, 8191]]
            runtime.submit(lambda: runtime.restore(leases, [0, 4096, 8191]), 0).result(
                timeout=5
            )
            self.assertTrue(torch.equal(expected, model.pack_pages([0, 4096, 8191])))
            for lease in leases:
                lease.close()
        finally:
            runtime.close()
            pool.clear()


class ManifestTests(unittest.TestCase):
    def test_mixed_manifest(self):
        desc = ChunkDescriptor(
            "nonce", "pages", 8192, 4352, pages=((0, 111, "lz4"), (256, 4096, "raw"))
        )
        desc = ChunkDescriptor.from_bytes(desc.to_bytes())
        desc.validate(
            nonce="nonce", raw_bytes=8192, capacity=8192, mode="lz4", page_bytes=4096
        )
        with self.assertRaises(ValueError):
            desc.validate(
                nonce="nonce",
                raw_bytes=8192,
                capacity=8192,
                mode="passthrough",
                page_bytes=4096,
            )

    def test_bad_offsets_overlap_count_and_tail(self):
        for pages, size in [
            (((0, 100, "lz4"), (0, 100, "lz4")), 100),
            (((1, 100, "lz4"),), 101),
            (((0, 100, "lz4"),), 101),
        ]:
            with self.subTest(pages=pages), self.assertRaises(ValueError):
                ChunkDescriptor.from_bytes(
                    ChunkDescriptor("x", "pages", 4096, size, pages=pages).to_bytes()
                )
        desc = ChunkDescriptor("x", "pages", 8192, 100, pages=((0, 100, "lz4"),))
        with self.assertRaises(ValueError):
            desc.validate(
                nonce="x", raw_bytes=8192, capacity=200, mode="lz4", page_bytes=4096
            )


class CompletionTests(unittest.TestCase):
    def test_future_must_complete_even_if_cuda_event_already_ready(self):
        future = concurrent.futures.Future()
        event = NS(query=lambda: True, synchronize=Mock())
        done = TransferCompletion(finish_event=event, future=future)
        self.assertFalse(done.query())
        self.assertEqual(done.state, TransferState.PENDING)
        with self.assertRaises(RuntimeError):
            done.result()
        future.set_result(123)
        self.assertEqual(done.state, TransferState.SUCCESS)
        self.assertEqual(done.actual_bytes, 123)
        event.synchronize.assert_not_called()

    def test_failed_drained_and_uncertain_are_not_success(self):
        for error, state in [
            (ValueError("bad"), TransferState.FAILED),
            (BufferDrainError("busy"), TransferState.UNCERTAIN),
        ]:
            with self.subTest(state=state):
                completion = TransferCompletion(future=result(error=error))
                self.assertTrue(completion.query())
                self.assertEqual(completion.state, state)
                self.assertIsNone(completion.actual_bytes)
                with self.assertRaises(type(error)):
                    completion.result()

    def test_native_event_path_preserved(self):
        event = NS(query=Mock(return_value=False), synchronize=Mock())
        completion = TransferCompletion(finish_event=event)
        self.assertFalse(completion.query())
        event.query.return_value = True
        self.assertEqual(completion.state, TransferState.SUCCESS)
        completion.result()
        event.synchronize.assert_called_once()


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.pool = CompressedHostKVCache(4096, 65536, pin_memory=False)
        self.ref = new_page_refs(1)[0]
        self.handle = self.pool.alloc(1)
        self.pool.destination(self.handle.item(), 4096).zero_()
        self.pool.publish(self.handle.item(), self.ref, "raw", 4096)
        self.spec = RepresentationSpec("layout-a", "lz4")
        self.provider = HostEncodedKVProvider(self.pool, self.spec)

    def test_incompatible_spec_rejected_without_leaking_reader(self):
        for spec in [
            RepresentationSpec("other", "lz4"),
            RepresentationSpec("layout-a", "lz4", force=True),
            RepresentationSpec("layout-a", "lz4", verify=True),
            RepresentationSpec("layout-a", "lz4", version=99),
        ]:
            with self.subTest(spec=spec):
                self.assertIsNone(self.provider.acquire(self.ref, spec))
        self.pool.free(self.handle)
        self.assertEqual(self.pool.snapshot()["retired_bytes"], 0)

    def test_borrow_survives_eviction_until_copy_completes(self):
        lease = self.provider.acquire(self.ref, self.spec)
        self.pool.free(self.handle)
        self.assertIsNone(self.provider.acquire(self.ref, self.spec))
        self.assertEqual(lease.future.result().nbytes, 4096)
        self.assertGreater(self.pool.snapshot()["retired_bytes"], 0)
        lease.close()
        self.assertEqual(self.pool.snapshot()["retired_bytes"], 0)


def ack_type():
    # Actual controller ACK class, isolated only from server/CUDA imports.
    from typing import Any, List, NamedTuple, Optional

    path = ROOT / "python/sglang/srt/managers/cache_controller.py"
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HiCacheAck"
    )
    namespace = {
        "NamedTuple": NamedTuple,
        "Any": Any,
        "List": List,
        "Optional": Optional,
        "device_module": NS(Event=object),
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["HiCacheAck"]


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.pool = CompressedHostKVCache(4096, 256 * 1024, pin_memory=False)
        self.handles, self.refs = self.pool.alloc(3), new_page_refs(3)
        for h, ref in zip(self.handles.tolist(), self.refs):
            self.pool.destination(h, 4096).zero_()
            self.pool.publish(h, ref, "raw", 4096)
        self.cd = NS(
            value=None,
            host_value=self.handles.clone(),
            metadata={"compression_page_refs": self.refs},
        )
        self.node = NS(
            id=1,
            key=[10, 20, 30],
            component_data={ComponentType.FULL: self.cd},
            write_through_pending_id=None,
        )
        self.nodes = {1: self.node}
        self.kv = NS(host_indices=self.handles.clone(), nodes_to_load=[1])
        self.future = concurrent.futures.Future()
        self.runtime = NS(
            verify=False,
            close=Mock(),
            idle=Mock(return_value=True),
            has_pending_work=Mock(return_value=False),
        )
        self.state = AsyncL2State(self.runtime, self.pool, None)
        self.cc = NS(
            async_l2=self.state,
            ack_load_queue=[],
            ack_write_queue=[],
            write_queue=[],
            load_fence_stream=None,
            l2_transfer_engine=NS(
                submit_async_restore=Mock(
                    side_effect=lambda *a: TransferCompletion(future=self.future)
                )
            ),
        )
        self.cache = HiCacheLifecycleMixin()
        self.cache.__dict__.update(
            cache_controller=self.cc,
            tree_core=NS(
                node_by_id=lambda n: self.nodes[n],
                build_load_back_spec=Mock(side_effect=lambda *a, **kw: (self.kv, {})),
                commit_load_back=Mock(return_value=[]),
                finish_load_back=Mock(),
                _update_duplicate_tracking=Mock(),
                _update_evictable_leaf_sets=Mock(),
                empty_match_result=NS(device_indices=torch.empty(0, dtype=torch.int64)),
            ),
            inc_host_lock_ref=Mock(return_value=NS(to_dec_params=lambda: "host-pin")),
            inc_lock_ref=Mock(return_value=NS(to_dec_params=lambda: "device-pin")),
            dec_host_lock_ref=Mock(),
            dec_lock_ref=Mock(),
            _apply_cache_actions=Mock(),
            token_to_kv_pool_allocator=NS(
                available_size=lambda: 20,
                alloc=Mock(return_value=torch.tensor([9, 11, 5])),
                free=Mock(),
            ),
            ongoing_write_through={},
            _finish_write_through_ack=Mock(),
        )
        # Small lifecycle fixtures intentionally permit three-page restores.
        self.cache.load_back_threshold = 1
        self.cache.inc_lock_ref.return_value.delta = 0
        self.cache.metrics_collector = Mock()
        self.req = NS(
            rid="test",
            last_node=0,
            finished_reason=None,
            to_finish=None,
            set_finish_with_abort=Mock(),
            retracted_stain=False,
        )
        self.params = NS(best_match_node=1, req=self.req, mem_quota=None)
        self.Ack = ack_type()

    def tearDown(self):
        self.state.close()

    def begin(self):
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.managers.cache_controller": NS(HiCacheAck=self.Ack),
                "sglang.srt.mem_cache.base_prefix_cache": NS(EvictParams=NS),
            },
        ):
            return self.cache.init_async_load_back(self.params)

    def poll(self):
        check = source_function(
            "mem_cache/unified_radix_cache.py", "loading_check", "UnifiedRadixCache"
        )
        while self.cc.ack_load_queue and self.cc.ack_load_queue[0].query():
            check(self.cache, finish_count=1)

    def complete(self):
        pages = [lease.future.result() for lease in self.state.restore_ticket.leases]
        self.future.set_result(
            RestoreTransferResult(
                len(pages),
                sum(p.nbytes for p in pages),
                sum(p.raw_bytes for p in pages),
                0.01,
                0.02,
            )
        )
        self.poll()

    def test_completion_does_not_publish_until_admission(self):
        self.assertIsNone(self.begin())
        self.cache.tree_core.commit_load_back.assert_not_called()
        self.complete()
        self.cache.tree_core.commit_load_back.assert_not_called()
        self.assertIsNone(self.cd.value)
        indices, node = self.begin()
        self.assertEqual(indices.tolist(), [9, 11, 5])
        self.assertEqual(node, 1)
        self.assertEqual(self.state.stats["restored_pages"], 3)
        self.assertIsNone(self.state.restore_ticket)
        self.cache.tree_core.commit_load_back.assert_called_once()
        self.cache.token_to_kv_pool_allocator.free.assert_not_called()
        self.cache.dec_host_lock_ref.assert_called_once()

    def test_valid_but_wrong_handle_cannot_restore_other_kv(self):
        other = self.pool.alloc(1)
        ref = new_page_refs(1)[0]
        self.pool.destination(other.item(), 4096).zero_()
        self.pool.publish(other.item(), ref, "raw", 4096)
        self.kv.host_indices[1] = other.item()
        self.cd.host_value = self.kv.host_indices.clone()
        self.assertIsNone(self.begin())
        self.cc.l2_transfer_engine.submit_async_restore.assert_not_called()
        self.req.set_finish_with_abort.assert_called_once()
        lease = self.pool.acquire(other.item(), expected_ref=ref)
        lease.close()
        self.assertIsNone(self.cd.host_value)

    def test_ticket_creation_failure_retains_submitted_targets(self):
        with patch(
            "sglang.srt.mem_cache.hicache_lifecycle.RestoreTicket",
            side_effect=ValueError("ticket"),
        ):
            self.assertIsNone(self.begin())
        self.cache.token_to_kv_pool_allocator.free.assert_not_called()
        self.cache.dec_lock_ref.assert_not_called()
        self.assertEqual(len(self.state.quarantined), 1)
        self.assertFalse(self.future.done())

    def test_source_accounting_uses_standard_admission_fields(self):
        self.begin()
        self.complete()
        indices, _ = self.begin()
        # Same assignments performed by PrefillAdder after init_load_back.
        self.req.host_hit_length = 3
        self.req.host_loaded_length = len(indices)
        self.req.prefix_indices = torch.cat([torch.tensor([7, 8]), indices])
        materialized = source_function(
            "managers/schedule_batch.py", "materialized_host_hit_len", "Req"
        )
        split = source_function(
            "managers/schedule_batch.py", "split_cached_prefix_by_tier"
        )
        self.assertEqual(materialized(self.req), 3)
        self.assertEqual(
            split(prefix_len=5, host_hit_len=materialized(self.req), storage_hit_len=0),
            (2, 3, 0),
        )
        self.assertFalse(hasattr(self.req, "_async_host_restore_pages"))

    def test_pending_does_not_block_or_resubmit(self):
        self.begin()
        self.poll()
        self.assertIsNone(self.begin())
        self.cc.l2_transfer_engine.submit_async_restore.assert_called_once()
        self.cache.tree_core.commit_load_back.assert_not_called()

    def test_other_request_cannot_consume_ticket(self):
        self.begin()
        self.complete()
        self.params.req = NS()
        self.assertIsNone(self.begin())
        self.assertFalse(self.state.restore_ticket.consumed)

    def test_ticket_replay_cannot_publish_twice(self):
        self.begin()
        self.complete()
        ticket = self.state.restore_ticket
        self.begin()
        self.assertTrue(ticket.consumed)
        self.assertIsNone(self.cache._consume_restore_ticket(ticket, self.params))
        self.cache.tree_core.commit_load_back.assert_called_once()
        # Replay must not free successfully published pages.
        self.cache.token_to_kv_pool_allocator.free.assert_not_called()

    def test_workspace_failure_retries_without_invalidating_host(self):
        self.begin()
        self.future.set_exception(CompressionCapacityError("busy"))
        self.poll()
        self.assertEqual(self.state.stats["restore_admission_retries"], 1)
        self.assertEqual(self.pool.snapshot()["used_pages"], 3)
        self.req.set_finish_with_abort.assert_not_called()
        self.cache.tree_core.commit_load_back.assert_not_called()
        self.cache.token_to_kv_pool_allocator.free.assert_called_once()

    def test_submission_capacity_failure_releases_unowned_targets(self):
        self.cc.l2_transfer_engine.submit_async_restore.side_effect = (
            CompressionCapacityError("full")
        )
        self.assertIsNone(self.begin())
        self.assertIsNone(self.state.restore_ticket)
        self.assertEqual(len(self.cc.ack_load_queue), 0)
        self.cache.token_to_kv_pool_allocator.free.assert_called_once()
        self.assertEqual(self.pool.snapshot()["used_pages"], 3)

    def test_partial_lease_acquire_failure_releases_earlier_readers(self):
        original = self.pool.acquire

        def acquire(handle, expected_ref=None):
            if handle == self.handles[1].item():
                raise ValueError("stale")
            return original(handle, expected_ref=expected_ref)

        with patch.object(self.pool, "acquire", side_effect=acquire):
            self.assertIsNone(self.begin())
        self.req.set_finish_with_abort.assert_called_once()
        self.assertEqual(self.pool.snapshot()["used_pages"], 0)
        self.assertEqual(self.pool.snapshot()["retired_bytes"], 0)

    def test_verification_failure_never_publishes(self):
        self.begin()
        self.future.set_exception(KVVerificationError("mismatch"))
        self.poll()
        self.cache.tree_core.commit_load_back.assert_not_called()
        self.req.set_finish_with_abort.assert_called_once()
        self.assertIsNone(self.cd.host_value)
        self.assertEqual(self.pool.snapshot()["used_pages"], 0)
        self.assertTrue(self.cache.background_work_is_idle())

    def test_cancel_waits_for_completion_before_free(self):
        self.begin()
        self.req.to_finish = "cancel"
        self.poll()
        self.cache.token_to_kv_pool_allocator.free.assert_not_called()
        self.complete()
        self.cache.token_to_kv_pool_allocator.free.assert_called_once()
        self.cache.tree_core.commit_load_back.assert_not_called()
        self.assertEqual(self.state.stats["cancelled_restores"], 1)

    def test_cancel_after_ready_is_reaped_without_admission(self):
        self.begin()
        self.complete()
        self.req.to_finish = "cancel"
        self.cache._reap_cancelled_restore()
        self.assertIsNone(self.state.restore_ticket)
        self.cache.tree_core.commit_load_back.assert_not_called()

    def test_uncertain_restore_retains_all_resources(self):
        self.begin()
        self.future.set_exception(BufferDrainError("cuda"))
        self.poll()
        self.cache.token_to_kv_pool_allocator.free.assert_not_called()
        self.cache.dec_lock_ref.assert_not_called()
        self.cache.dec_host_lock_ref.assert_not_called()
        self.assertEqual(len(self.state.quarantined), 1)
        self.assertFalse(self.cache.background_work_is_idle())

    def test_partial_publish_failure_quarantines_targets(self):
        self.begin()
        self.complete()
        self.cache.tree_core.commit_load_back.side_effect = RuntimeError("partial")
        self.assertIsNone(self.begin())
        self.cache.token_to_kv_pool_allocator.free.assert_not_called()
        self.cache.dec_lock_ref.assert_not_called()
        self.assertEqual(len(self.state.quarantined), 1)

    def test_changed_anchor_discards_drained_ticket(self):
        self.begin()
        self.complete()
        self.params.best_match_node = 2
        self.assertIsNone(self.begin())
        self.cache.token_to_kv_pool_allocator.free.assert_called_once()
        self.cache.tree_core.commit_load_back.assert_not_called()

    def test_rematch_abandons_ticket_but_waits_for_drain(self):
        self.begin()
        params = NS(req=self.req)
        self.cache.reconcile_restore_ticket(
            params, NS(host_hit_length=0, best_match_node=1)
        )
        self.cache.token_to_kv_pool_allocator.free.assert_not_called()
        self.complete()
        self.assertIsNone(self.state.restore_ticket)
        self.cache.tree_core.commit_load_back.assert_not_called()

    def test_split_preserves_identities_and_partial_adoption(self):
        self.begin()
        self.complete()
        self.cd.metadata["compression_page_refs"] = self.refs[1:]
        self.node.key = [20, 30]
        self.kv.host_indices = self.handles[1:]
        adopted, _ = self.begin()
        self.assertEqual(adopted.tolist(), [11, 5])
        self.assertEqual(
            self.cache.token_to_kv_pool_allocator.free.call_args.args[0].tolist(), [9]
        )
        self.assertEqual(self.state.stats["restored_pages"], 2)

    def test_recomputed_identity_cannot_adopt_old_restore(self):
        self.begin()
        self.complete()
        self.cd.metadata["compression_page_refs"] = new_page_refs(3)
        self.assertIsNone(self.begin())
        self.cache.tree_core.commit_load_back.assert_not_called()
        self.cache.token_to_kv_pool_allocator.free.assert_called_once()

    def test_background_and_reset_include_ready_ticket(self):
        self.begin()
        self.complete()
        self.assertTrue(self.cache.has_pending_background_work())
        self.assertFalse(self.cache.background_work_is_idle())
        with self.assertRaises(RuntimeError):
            self.cache.reset_async_l2()
        self.begin()
        self.assertTrue(self.cache.background_work_is_idle())
        self.cache.reset_async_l2()
        self.assertEqual(self.pool.snapshot()["used_pages"], 0)

    def test_recompute_invalidates_host_binding_and_identity(self):
        self.cache.on_kv_recomputed(self.node)
        self.assertIsNone(self.cd.host_value)
        self.assertNotEqual(self.cache.kv_page_refs(self.node), self.refs)
        self.assertEqual(self.pool.snapshot()["used_pages"], 0)

    def test_recompute_refuses_pending_backup(self):
        self.node.write_through_pending_id = 1
        with self.assertRaises(RuntimeError):
            self.cache.on_kv_recomputed(self.node)
        self.assertEqual(self.pool.snapshot()["used_pages"], 3)

    def test_duplicate_backup_preserves_binding(self):
        self.assertIsNone(self.cache._prepare_async_backup(1))
        self.assertEqual(self.state.stats["backup_duplicate_skips"], 1)
        self.assertFalse(self.state.backup_pins)

    def test_threshold_skip_acquires_no_resources(self):
        self.cache.load_back_threshold = 10
        indices, node = self.begin()
        self.assertEqual(indices.numel(), 0)
        self.assertEqual(node, self.req.last_node)
        self.cache.inc_host_lock_ref.assert_not_called()
        self.cache.inc_lock_ref.assert_not_called()
        self.cache.token_to_kv_pool_allocator.alloc.assert_not_called()
        self.cc.l2_transfer_engine.submit_async_restore.assert_not_called()
        self.assertEqual(self.pool.active_readers, 0)

    def test_quota_counts_newly_protected_pages(self):
        self.params.mem_quota = 4
        self.cache.inc_lock_ref.return_value.delta = 5
        indices, node = self.begin()
        self.assertEqual(indices.numel(), 0)
        self.assertEqual(node, self.req.last_node)
        self.cache.token_to_kv_pool_allocator.alloc.assert_not_called()
        self.cache.dec_host_lock_ref.assert_called_once()
        self.cache.dec_lock_ref.assert_called_once()
        self.assertIsNone(self.state.restore_ticket)

    def test_quota_rejection_is_not_waiting(self):
        self.params.mem_quota = 2
        indices, _ = self.begin()
        self.assertEqual(indices.numel(), 0)
        self.assertIsNone(self.state.restore_ticket)
        self.cc.l2_transfer_engine.submit_async_restore.assert_not_called()

    def test_allocation_failure_returns_native_fallback(self):
        self.cache.token_to_kv_pool_allocator.alloc.return_value = None
        indices, node = self.begin()
        self.assertEqual(indices.numel(), 0)
        self.assertEqual(node, self.req.last_node)
        self.cache.dec_host_lock_ref.assert_called_once()
        self.cache.dec_lock_ref.assert_called_once()

    def test_existing_ticket_is_consumed_before_new_threshold_check(self):
        self.begin()
        self.cache.load_back_threshold = 100
        self.assertIsNone(self.begin())
        self.complete()
        indices, _ = self.begin()
        self.assertEqual(len(indices), 3)
        self.cc.l2_transfer_engine.submit_async_restore.assert_called_once()

    def test_successful_io_metrics_precede_adoption_and_are_once_only(self):
        self.begin()
        self.complete()
        self.cache.metrics_collector.increment_load_back_num_tokens.assert_called_once_with(
            num_tokens=3, pool="kv"
        )
        self.cache.metrics_collector.increment_load_back_num_bytes.assert_called_once_with(
            3 * 4096
        )
        self.assertEqual(self.state.stats["restored_pages"], 0)
        self.assertEqual(self.state.stats["h2d_bytes"], 3 * 4096)
        self.poll()
        self.begin()
        self.cache.metrics_collector.increment_load_back_num_tokens.assert_called_once()
        self.assertEqual(self.state.stats["restored_pages"], 3)
        self.assertEqual(self.state.stats["h2d_bytes"], 3 * 4096)

    def test_backup_identity_error_precedes_pins(self):
        self.node.parent = self.cache.tree_core.root_node = NS()
        self.cd.host_value = None
        self.cd.metadata["compression_page_refs"] = (123,)
        with self.assertRaisesRegex(RuntimeError, "identities"):
            self.cache._prepare_async_backup(1)
        self.cache.inc_lock_ref.assert_not_called()
        self.assertFalse(self.state.backup_pins)

    def test_backup_source_pin_failure_releases_parent(self):
        self.cache.tree_core.root_node = NS()
        self.node.parent = NS(id=2, backuped=True)
        self.cd.host_value = None
        self.cache.inc_lock_ref.side_effect = RuntimeError("pin failed")
        with self.assertRaisesRegex(RuntimeError, "pin failed"):
            self.cache._prepare_async_backup(1)
        self.cache.dec_host_lock_ref.assert_called_once_with(2, "host-pin")
        self.assertFalse(self.state.backup_pins)

    def test_failed_backup_rolls_back_host_but_preserves_source(self):
        self.state.backup_pins[1] = ("device-pin", None)
        self.cache.ongoing_write_through[1] = NS(
            node_id=1, publish_node_ids=[1], lock_params="device-pin"
        )
        ack = self.Ack(
            None,
            None,
            [1],
            3,
            completion=TransferCompletion(future=result(error=ValueError("copy"))),
        )
        self.assertFalse(self.cache._complete_async_write_ack(ack))
        self.assertIsNone(self.cd.host_value)
        self.cache.token_to_kv_pool_allocator.free.assert_not_called()
        self.cache.dec_lock_ref.assert_called_once_with(1, "device-pin")
        self.assertFalse(self.state.backup_pins)

    def test_uncertain_backup_keeps_pins_and_host_memory(self):
        self.state.backup_pins[1] = ("device-pin", None)
        ack = self.Ack(
            None,
            None,
            [1],
            3,
            completion=TransferCompletion(
                future=result(error=BufferDrainError("copy"))
            ),
        )
        self.assertFalse(self.cache._complete_async_write_ack(ack))
        self.assertEqual(self.pool.snapshot()["used_pages"], 3)
        self.cache.dec_lock_ref.assert_not_called()
        self.assertFalse(self.cache.background_work_is_idle())

    def test_successful_backup_uses_existing_ack_and_actual_bytes(self):
        self.state.backup_pins[1] = ("device-pin", (0, "parent-pin"))
        ack = self.Ack(
            None, None, [1], 3, completion=TransferCompletion(future=result(12288))
        )
        self.assertTrue(self.cache._complete_async_write_ack(ack))
        self.cache._finish_write_through_ack.assert_called_once_with(1)
        self.cache.dec_host_lock_ref.assert_called_once_with(0, "parent-pin")
        self.assertEqual(self.state.stats["d2h_bytes"], 12288)
        self.assertEqual(self.state.stats["backed_up_pages"], 3)


class ControllerBoundaryTests(unittest.TestCase):
    def test_write_submission_enters_standard_ack_queue(self):
        Ack = ack_type()
        future = concurrent.futures.Future()
        completion = TransferCompletion(future=future)
        op = NS(
            host_indices=torch.tensor([1]),
            device_indices=torch.tensor([3]),
            node_ids=[7],
            page_refs=(42,),
        )
        state = NS(backup_completions={})
        cc = NS(
            write_queue=[op],
            ack_write_queue=[],
            async_l2=state,
            _move_write_operation=lambda op: (op.host_indices, op.device_indices, None),
            _l2_transfers=lambda *args: ["transfer"],
            _num_tokens_by_pool=lambda op: {"kv": 1},
            _transfer_num_bytes=lambda op: 4096,
            l2_transfer_engine=NS(submit_device_to_host=Mock(return_value=completion)),
        )
        start = source_function(
            "managers/cache_controller.py",
            "start_writing",
            "HiCacheController",
            CacheOperation=NS(merge_ops=lambda ops: ops[0]),
            HiCacheAck=Ack,
        )
        start(cc)
        self.assertEqual(cc.write_queue, [])
        self.assertEqual(len(cc.ack_write_queue), 1)
        self.assertIs(state.backup_completions[7], completion)
        self.assertFalse(cc.ack_write_queue[0].query())
        future.set_result(2048)
        self.assertEqual(cc.ack_write_queue[0].result(), 2048)
        self.assertEqual(
            cc.l2_transfer_engine.submit_device_to_host.call_args.kwargs["page_refs"],
            (42,),
        )

    def test_native_loading_retains_layer_callbacks(self):
        Ack = ack_type()
        finish = NS(query=lambda: True, synchronize=Mock())
        completion = TransferCompletion(NS(), finish, True)
        event = NS(start_event=NS(record=Mock()), complete=Mock())
        op = NS(device_indices=torch.tensor([3]), node_ids=[7])
        cc = NS(
            load_queue=[op],
            layer_done_counter=NS(update_producer=lambda: 0, events=[event]),
            load_fence_stream=None,
            _move_op_indices=lambda op: (torch.tensor([1]), op.device_indices, None),
            _l2_load_transfers=lambda *args: ["native"],
            l2_transfer_engine=NS(submit_host_to_device=Mock(return_value=completion)),
            layer_num=2,
            _num_tokens_by_pool=lambda op: {"kv": 1},
            _transfer_num_bytes=lambda op: 4096,
            ack_load_queue=[],
        )
        start = source_function(
            "managers/cache_controller.py",
            "start_loading",
            "HiCacheController",
            CacheOperation=NS(merge_ops=lambda ops: ops[0]),
            HiCacheAck=Ack,
        )
        self.assertEqual(start(cc), 0)
        self.assertIs(
            cc.l2_transfer_engine.submit_host_to_device.call_args.kwargs[
                "on_layer_done"
            ],
            event.complete,
        )
        self.assertTrue(cc.ack_load_queue[0].query())
        cc.ack_load_queue[0].result()
        finish.synchronize.assert_called_once()

    def test_backup_parent_failure_prevents_child_io(self):
        runtime = NS(close=Mock(), idle=lambda: True)
        io = NS(_write=Mock(return_value=1))
        state = AsyncL2State(runtime, NS(io=io), None)
        state.backup_dependencies[2] = TransferCompletion(
            future=result(error=ValueError("parent"))
        )
        try:
            completion = state.submit_backup(
                torch.tensor([1]), (42,), torch.tensor([3]), None, (2,)
            )
            with self.assertRaises(ValueError):
                completion.future.result(timeout=5)
            io._write.assert_not_called()
            self.assertEqual(completion.state, TransferState.FAILED)
        finally:
            state.close()

    def test_uncertain_backup_stops_already_queued_tasks(self):
        entered, release = threading.Event(), threading.Event()

        def fail(*args):
            entered.set()
            release.wait(5)
            raise BufferDrainError("uncertain")

        io = NS(_write=Mock(side_effect=fail))
        state = AsyncL2State(NS(close=Mock(), idle=lambda: True), NS(io=io), None)
        try:
            first = state.submit_backup(
                torch.tensor([1]), (42,), torch.tensor([3]), None
            )
            self.assertTrue(entered.wait(5))
            second = state.submit_backup(
                torch.tensor([2]), (43,), torch.tensor([4]), None
            )
            release.set()
            for completion in [first, second]:
                with self.assertRaises(BufferDrainError):
                    completion.future.result(timeout=5)
            io._write.assert_called_once()
            self.assertFalse(state.idle())
        finally:
            release.set()
            state.close()

    def test_split_preserves_host_handles_and_materialization(self):
        ct = ComponentType.FULL
        child = NS(
            component_data={
                ct: NS(
                    value=torch.tensor([4, 7, 8]),
                    host_value=torch.tensor([100, 101, 102]),
                    metadata={"compression_page_refs": (10, 11, 12)},
                    lock_ref=2,
                    session_ref=1,
                )
            }
        )
        parent = NS(key=[1], component_data={ct: NS(metadata={}, session_ids=None)})
        fn = source_function(
            "mem_cache/unified_cache/components/full.py",
            "redistribute_on_node_split",
            "FullComponent",
        )
        fn(NS(component_type=ct), parent, child)
        self.assertEqual(
            parent.component_data[ct].metadata["compression_page_refs"], (10,)
        )
        self.assertEqual(
            child.component_data[ct].metadata["compression_page_refs"], (11, 12)
        )
        self.assertEqual(child.component_data[ct].host_value.tolist(), [101, 102])

    def test_transfer_identity_checks_physical_mapping(self):
        cache = HiCacheLifecycleMixin()
        cache.cache_controller = NS(async_l2=object())
        refs = new_page_refs(3)
        node = NS(
            key=[1, 2, 3],
            component_data={
                ComponentType.FULL: NS(
                    value=torch.tensor([9, 11, 5]),
                    metadata={"compression_page_refs": refs},
                )
            },
        )
        cache.tree_core = NS(_walk_span=lambda *a: [(node, 0, 3)])
        req = NS(get_fill_ids=lambda: [1, 2, 3], extra_key=None, cache_salt=None)
        with patch.dict(
            sys.modules,
            {"sglang.srt.mem_cache.radix_cache": NS(RadixKey=lambda *a, **kw: None)},
        ):
            self.assertEqual(cache.get_kv_transfer_refs(req, 0, 3, [9, 11, 5]), refs)
            self.assertNotEqual(cache.get_kv_transfer_refs(req, 0, 3, [9, 11, 6]), refs)

    def test_pd_and_executor_have_no_hicache_private_dependency(self):
        for path in [
            "disaggregation/prefill.py",
            "disaggregation/mooncake/compression.py",
            "kv_compression/runtime.py",
        ]:
            text = (ROOT / "python/sglang/srt" / path).read_text()
            self.assertNotIn("compressed_hicache", text)
            self.assertNotIn("._walk_span(", text)
        attach = (
            ROOT / "python/sglang/srt/mem_cache/hicache_compression.py"
        ).read_text()
        self.assertNotIn("class HiCacheCompressionAdapter", attach)
        self.assertNotIn("def poll(", attach)


class AdditionalLifecycleTests(unittest.TestCase):
    def test_host_reader_prevents_idle_until_release(self):
        pool = CompressedHostKVCache(4096, 65536, pin_memory=False)
        h = pool.alloc(1)
        ref = new_page_refs(1)[0]
        pool.destination(h.item(), 4096).zero_()
        pool.publish(h.item(), ref, "raw", 4096)
        state = AsyncL2State(NS(idle=lambda: True, close=lambda: None), pool, None)
        try:
            lease = pool.acquire(h.item())
            self.assertFalse(state.idle())
            pool.free(h)
            self.assertFalse(state.idle())
            lease.close()
            self.assertTrue(state.idle())
        finally:
            state.close()

    def test_without_l2_no_persistent_provider_or_refs(self):
        cache = HiCacheLifecycleMixin()
        self.assertEqual(cache.get_kv_compression_context(), (None, None))
        self.assertIsNone(cache.get_kv_transfer_refs(None, 0, 10, None))
        self.assertFalse(cache.has_pending_background_work())
        self.assertTrue(cache.background_work_is_idle())

    def test_cancelled_completion_is_failed_not_success(self):
        future = concurrent.futures.Future()
        future.cancel()
        completion = TransferCompletion(future=future)
        self.assertEqual(completion.state, TransferState.FAILED)
        self.assertIsNone(completion.actual_bytes)


class AuditR31Tests(unittest.TestCase):
    def test_native_writeback_waits_for_pending_event(self):
        event = NS(query=Mock(return_value=False), synchronize=Mock())
        ack = ack_type()(
            None, event, [1], completion=TransferCompletion(finish_event=event)
        )
        cache = NS(
            cache_controller=NS(ack_write_queue=[ack]),
            ongoing_write_through={1: object()},
            _log_write_ack_metrics=Mock(),
        )
        cache._finish_write_through_ack = lambda n: cache.ongoing_write_through.pop(n)
        write_check = source_function(
            "mem_cache/unified_radix_cache.py", "writing_check", "UnifiedRadixCache"
        )
        write_check(cache, write_back=True)
        event.synchronize.assert_called_once()
        self.assertFalse(cache.ongoing_write_through)
        self.assertFalse(cache.cache_controller.ack_write_queue)

    def test_host_pin_is_released_after_failed_backup_unbinds_parent(self):
        acquire = source_function(
            "mem_cache/unified_cache/components/full.py",
            "acquire_component_lock",
            "FullComponent",
        )
        release = source_function(
            "mem_cache/unified_cache/components/full.py",
            "release_component_lock",
            "FullComponent",
        )
        cd = NS(host_value=torch.tensor([1]), host_lock_ref=0)
        node = NS(component_data={ComponentType.FULL: cd})
        core = NS(is_write_back=False, _update_evictable_leaf_sets=Mock())
        component = NS(component_type=ComponentType.FULL, tree_core=core)
        receipt = NS(skipped_lock_components=())
        acquire(component, node, receipt, lock_host=True)
        self.assertEqual(cd.host_lock_ref, 1)
        cd.host_value = None  # Parent backup rolled back before child's ACK.
        release(component, node, receipt, lock_host=True)
        self.assertEqual(cd.host_lock_ref, 0)

    def test_unacquired_host_pin_cannot_release_another_holder(self):
        acquire = source_function(
            "mem_cache/unified_cache/components/full.py",
            "acquire_component_lock",
            "FullComponent",
        )
        release = source_function(
            "mem_cache/unified_cache/components/full.py",
            "release_component_lock",
            "FullComponent",
        )
        cd = NS(host_value=None, host_lock_ref=0)
        node = NS(component_data={ComponentType.FULL: cd})
        core = NS(is_write_back=False, _update_evictable_leaf_sets=Mock())
        component = NS(component_type=ComponentType.FULL, tree_core=core)
        receipt = NS(skipped_lock_components=())
        acquire(component, node, receipt, lock_host=True)
        cd.host_value, cd.host_lock_ref = torch.tensor([2]), 1
        release(component, node, receipt, lock_host=True)
        self.assertEqual(cd.host_lock_ref, 1)

    def test_sender_source_verification_drains_before_failure(self):
        import contextlib
        import time

        fake_torch = NS(
            cuda=NS(set_device=Mock(), stream=lambda s: contextlib.nullcontext())
        )
        namespace = {
            "torch": fake_torch,
            "time": time,
            "BufferDrainError": BufferDrainError,
            "check_peer": lambda *a: None,
            "capability": lambda m: m,
        }
        run = source_function(
            "disaggregation/mooncake/conn.py",
            "_transfer_compressed_chunk",
            "MooncakeKVManager",
            **namespace,
        )
        for uncertain in (False, True):
            with self.subTest(uncertain=uncertain):
                stream = NS(
                    synchronize=Mock(
                        side_effect=RuntimeError("CUDA fault") if uncertain else None
                    )
                )
                runtime = NS(
                    transport_failed=False,
                    layout_tag="test",
                    device="cuda",
                    verify=True,
                    encode_pages=Mock(return_value=(None, (), 0)),
                    bytes_per_token=16,
                    layout=NS(
                        pack_pages=Mock(side_effect=ValueError("pack failed")),
                        to_staging_order=Mock(),
                    ),
                    shared=NS(quarantine=Mock()),
                )
                helper = source_function(
                    "disaggregation/mooncake/compression.py",
                    "source_digest",
                    "CompressionRuntime",
                    torch=fake_torch,
                    BufferDrainError=BufferDrainError,
                    digest=Mock(),
                )
                runtime.source_digest = types.MethodType(helper, runtime)
                manager = NS(compression_runtime=runtime, compression_mode="lz4")
                ready = object()
                stream.wait_event = Mock()
                chunk = NS(
                    prefill_kv_indices=[0], compression_refs=[1], wait_event=ready
                )
                with self.assertRaises(BufferDrainError if uncertain else ValueError):
                    run(
                        manager,
                        NS(get_gather_stream=lambda: stream),
                        chunk,
                        NS(room=7, compression_nonce="nonce"),
                        NS(compression_capability="lz4", compression_layout="test"),
                        0,
                        0,
                        0,
                    )
                stream.wait_event.assert_called_once_with(ready)
                stream.synchronize.assert_called_once()
                if uncertain:
                    runtime.shared.quarantine.assert_called_once()
                    self.assertTrue(runtime.transport_failed)
                else:
                    runtime.shared.quarantine.assert_not_called()

    def test_pd_capacity_pressure_defers_without_failing_request(self):
        run = source_function(
            "disaggregation/mooncake/conn.py",
            "_do_staging_transfer",
            "MooncakeKVManager",
            CompressionCapacityError=CompressionCapacityError,
            BufferDrainError=BufferDrainError,
            time=__import__("time"),
        )
        manager = NS(
            compression_mode="lz4",
            _transfer_compressed_chunk=Mock(
                side_effect=[CompressionCapacityError("busy"), 0]
            ),
            conclude_failure=Mock(),
        )
        strategy = NS(check_ready=lambda *a: (True, 0, 0, 0, 0), staging_buffer=None)
        chunk = NS(index_slice=slice(0, 1), prefill_kv_indices=[1])
        queue = NS(put=Mock())
        with patch("time.sleep"):
            self.assertEqual(
                run(manager, strategy, chunk, NS(room=7), None, None, None, queue, 0),
                (-1, True),
            )
            self.assertEqual(
                run(manager, strategy, chunk, NS(room=7), None, None, None, queue, 0),
                (0, False),
            )
        manager.conclude_failure.assert_not_called()
        queue.put.assert_called_once_with(chunk)


class BackupAdmissionTests(unittest.TestCase):
    """Actual admission/eviction methods and pool; controlled tree and I/O.

    Both branches use the same object pool to isolate admission semantics from
    metadata overhead. Synthetic lz4 payloads test allocation, not decoding.
    """

    def setUp(self):
        self.backup = source_function(
            "mem_cache/unified_radix_cache.py",
            "_execute_kv_backup",
            "UnifiedRadixCache",
        )
        self.drive = source_function(
            "mem_cache/unified_cache/components/full.py",
            "drive_host_eviction",
            "FullComponent",
            heapq=__import__("heapq"),
        )

    def cache(self, pool, evict, compressed=True):
        return NS(
            cache_controller=NS(
                mem_pool_host=pool,
                write=Mock(side_effect=lambda indices, **kw: pool.alloc(len(indices))),
            ),
            async_l2=NS(pool=pool, stats={"admission_skips": 0})
            if compressed
            else None,
            evict_host=evict,
        )

    def submit(self, cache, count):
        return self.backup(cache, 77, list(range(count)), {}, [], page_refs=None)

    def filled_pool(self, free=0, length=4096):
        pool = CompressedHostKVCache(4096, 8 * 1024 * 1024, pin_memory=False)
        handles = pool.alloc(pool.available_size() - free)
        for ref, h in enumerate(handles.tolist(), 1):
            pool.destination(h, length).zero_()
            pool.publish(h, ref, "raw" if length == 4096 else "lz4", length)
        return pool, handles

    def matrix_case(self, free, requested, compressed):
        pool, handles = self.filled_pool(free)

        class Node:
            def __init__(self, priority, indices):
                self.priority, self.indices = priority, indices
                self.parent = None

        nodes = [
            Node(0, handles[:100]),
            Node(1, handles[100:300]),
            Node(2, handles[300:]),
        ]
        tree = NS(evictable_host_leaves=set(nodes))
        events = []

        def remove(node, tracker, device_frees, host_frees):
            tree.evictable_host_leaves.remove(node)
            tracker[ComponentType.FULL] += len(node.indices)
            host_frees.setdefault(ComponentType.FULL, []).append(node.indices)
            events.append(("victim", node.priority, len(node.indices)))

        tree._evict_host_leaf = remove
        component = NS(
            tree_core=tree,
            component_type=ComponentType.FULL,
            _ensure_eviction_strategy=lambda: None,
            session_ref_eviction_strategy=lambda node: node.priority,
        )

        def evict(count):
            events.append(("request", count))
            tracker, frees = {ComponentType.FULL: 0}, {}
            self.drive(component, count, tracker, {}, frees)
            for indices in frees.get(ComponentType.FULL, []):
                pool.free(indices)
            return tracker[ComponentType.FULL]

        cache = self.cache(pool, evict, compressed)
        self.assertEqual(pool.available_size(), free)
        self.assertIsNotNone(self.submit(cache, requested))
        retained = 0
        for ref in range(101, 301):
            lease = pool.acquire_ref(ref, "passthrough")
            if lease is not None:
                retained += 1
                lease.close()
        self.assertEqual(pool.active_readers, 0)
        return events, retained

    def test_same_capacity_matches_ordinary_admission(self):
        for free, requested, retained in [
            (200, 300, 200),
            (400, 300, 200),
            (0, 100, 200),
            (200, 250, 200),
            (299, 300, 200),
            (300, 300, 200),
            (200, 350, 0),
        ]:
            with self.subTest(free=free, requested=requested):
                ordinary = self.matrix_case(free, requested, False)
                compressed = self.matrix_case(free, requested, True)
                self.assertEqual(ordinary, compressed)
                self.assertEqual(compressed[1], retained)

    def test_background_release_between_queries_avoids_eviction(self):
        pool, handles = self.filled_pool()
        original = pool.can_reserve

        def query(count):
            ready = original(count)
            if not ready:
                pool.free(handles[:1])
            return ready

        pool.can_reserve = query
        cache = self.cache(pool, Mock(return_value=0))
        self.assertIsNotNone(self.submit(cache, 1))
        cache.evict_host.assert_not_called()

    def test_small_payloads_recompute_shortfall_after_each_round(self):
        pool, handles = self.filled_pool()
        # Replace two adjacent full reservations with 512-byte stored objects.
        pool.free(handles[:2])
        small = pool.alloc(2)
        for i, h in enumerate(small.tolist()):
            pool.destination(h, 512).zero_()
            pool.publish(h, 100000 + i, "lz4", 512)
        # Two separated 3584-byte holes cannot admit a 4096-byte reservation.
        self.assertEqual(pool.available_size(), 0)
        self.assertGreater(pool.free_count, 0)
        groups = iter([small[:1], small[1:], handles[2:3]])

        def evict(count):
            group = next(groups)
            pool.free(group)
            return len(group)

        cache = self.cache(pool, Mock(side_effect=evict))
        self.assertIsNotNone(self.submit(cache, 3))
        self.assertEqual(
            [c.args[0] for c in cache.evict_host.call_args_list], [3, 2, 1]
        )

    def test_leased_eviction_is_not_available_until_release(self):
        pool, handles = self.filled_pool()
        lease = pool.acquire(handles[0])
        try:
            calls = 0

            def evict(count):
                nonlocal calls
                calls += 1
                if calls == 1:
                    pool.free(handles[:1])
                    return 1
                return 0  # All other nodes are protected in this scenario.

            cache = self.cache(pool, Mock(side_effect=evict))
            self.assertIsNone(self.submit(cache, 1))
            cache.cache_controller.write.assert_not_called()
            self.assertEqual(pool.available_size(), 0)
            self.assertEqual(pool.snapshot()["retired_bytes"], 4096)
            self.assertEqual(cache.async_l2.stats["admission_skips"], 1)
            self.assertIsNone(pool.acquire_ref(1, "lz4"))
        finally:
            lease.close()
        self.assertEqual(pool.snapshot()["retired_bytes"], 0)
        cache.evict_host.reset_mock()
        self.assertIsNotNone(self.submit(cache, 1))
        cache.evict_host.assert_not_called()

    def test_no_victim_or_no_progress_stops_without_submission(self):
        for reported_eviction in (0, 1):
            with self.subTest(reported_eviction=reported_eviction):
                pool, _ = self.filled_pool()
                cache = self.cache(pool, Mock(return_value=reported_eviction))
                self.assertIsNone(self.submit(cache, 1))
                cache.evict_host.assert_called_once_with(1)
                cache.cache_controller.write.assert_not_called()
                self.assertEqual(cache.async_l2.stats["admission_skips"], 1)

    def test_final_allocation_failure_does_not_publish_or_evict_again(self):
        pool, _ = self.filled_pool(free=1)
        cache = self.cache(pool, Mock(return_value=0))
        # A failed final allocation remains a normal admission failure.
        cache.cache_controller.write = Mock(return_value=None)
        self.assertIsNone(self.submit(cache, 1))
        cache.evict_host.assert_not_called()
        cache.cache_controller.write.assert_called_once()


class HiCacheContractTests(unittest.TestCase):
    def fixture(self, count=3):
        f = LifecycleTests()
        f.setUp()
        self.addCleanup(f.tearDown)
        f.pool.clear()
        f.handles, f.refs = f.pool.alloc(count), new_page_refs(count)
        for handle, ref in zip(f.handles.tolist(), f.refs):
            f.pool.destination(handle, 4096).zero_()
            f.pool.publish(handle, ref, "raw", 4096)
        f.cd.host_value = f.handles.clone()
        f.cd.metadata["compression_page_refs"] = f.refs
        f.node.key = list(range(count))
        f.kv.host_indices = f.handles.clone()
        f.cache.token_to_kv_pool_allocator.alloc.return_value = torch.arange(count)
        f.cache._build_sidecar_transfers = lambda *a: []
        f.cache._component_available_size = lambda *a: 100
        f.cache.ongoing_load_back = {}
        f.cc.load = Mock(return_value=torch.arange(count))
        return f

    def native(self, f, delta=0):
        fn = source_function(
            "mem_cache/unified_radix_cache.py",
            "_load_back_transfers",
            "UnifiedRadixCache",
            CacheTransferPhase=NS(LOAD_BACK="load"),
            ComponentType=ComponentType,
            _OngoingLoadBack=lambda *args: args,
        )
        return fn(
            f.cache,
            node_id=1,
            mem_quota=f.params.mem_quota,
            req=f.req,
            result=NS(delta=delta),
            ancestor_lock_params="device-pin",
            host_anchor_params="host-pin",
        )

    def test_threshold_matrix_matches_native(self):
        for threshold, counts in [
            (10, (0, 1, 4, 5, 9, 10, 11, 16)),
            (0, (0, 1, 4)),
            (1, (0, 1, 4)),
        ]:
            for count in counts:
                with self.subTest(threshold=threshold, count=count):
                    f = self.fixture(count)
                    f.cache.load_back_threshold = threshold
                    native = self.native(f)
                    f.cache.inc_lock_ref.reset_mock()
                    f.cache.inc_host_lock_ref.reset_mock()
                    answer = f.begin()
                    if f.state.restore_ticket is not None:
                        f.complete()
                        answer = f.begin()
                    self.assertEqual(len(answer[0]), count if native else 0)
                    self.assertEqual(
                        f.cc.load.call_count,
                        f.cc.l2_transfer_engine.submit_async_restore.call_count,
                    )
                    if not native:
                        f.cache.inc_lock_ref.assert_not_called()
                        f.cache.inc_host_lock_ref.assert_not_called()
                    self.assertEqual(f.pool.active_readers, 0)

    def test_quota_matrix_matches_native(self):
        for delta, quota in [(0, None), (0, 2), (0, 3), (5, 4), (5, 8), (5, 9)]:
            with self.subTest(delta=delta, quota=quota):
                f = self.fixture()
                f.params.mem_quota = quota
                f.cache.inc_lock_ref.return_value.delta = delta
                native = self.native(f, delta)
                answer = f.begin()
                if f.state.restore_ticket is not None:
                    f.complete()
                    answer = f.begin()
                self.assertIsNotNone(answer)
                self.assertEqual(len(answer[0]), 3 if native else 0)
                self.assertEqual(f.pool.active_readers, 0)

    def test_aux_threshold_exception_and_quota_are_independent(self):
        self.assertTrue(should_skip_full_load(0, 0))
        self.assertFalse(should_skip_full_load(0, 10, has_aux=True))
        self.assertTrue(exceeds_load_quota(0, 3, 2))

    def test_host_usage_tracks_slots_not_allocation_capacity(self):
        f = self.fixture(0)
        observe = source_function(
            "managers/scheduler_components/metrics_reporter.py",
            "_log_hicache_stats",
            "SchedulerMetricsReporter",
        )
        reporter = NS(
            scheduler=NS(
                enable_hierarchical_cache=True, tree_cache=NS(full_kv_pool_host=f.pool)
            ),
            stats=NS(),
        )

        def check(expected):
            observe(reporter)
            self.assertEqual(reporter.stats.hicache_host_used_tokens, expected)
            self.assertEqual(
                reporter.stats.hicache_host_total_tokens, f.pool.logical_size
            )

        check(0)
        handles = f.pool.alloc(2)
        check(2)
        for i, h in enumerate(handles.tolist()):
            f.pool.destination(h, 512).zero_()
            f.pool.publish(h, 10000 + i, "lz4", 512)
        check(2)
        lease = f.pool.acquire(handles[0])
        f.pool.free(handles)
        check(1)
        self.assertEqual(f.pool.snapshot()["retired_bytes"], BLOCK_BYTES)
        lease.close()
        check(0)
        reporter.scheduler.tree_cache.full_kv_pool_host = NS(
            logical_size=100, available_size=lambda: 20
        )
        observe(reporter)
        self.assertEqual(reporter.stats.hicache_host_used_tokens, 80)

    def test_cancelled_completed_io_counted_but_not_adopted(self):
        f = self.fixture()
        f.begin()
        f.req.to_finish = "cancel"
        f.complete()
        self.assertEqual(f.state.stats["completed_restore_pages"], 3)
        self.assertEqual(f.state.stats["restored_pages"], 0)
        f.cache.metrics_collector.increment_load_back_num_tokens.assert_called_once()
        f.poll()
        f.cache.metrics_collector.increment_load_back_num_tokens.assert_called_once()
        self.assertEqual(f.pool.active_readers, 0)

    def test_duplicate_ready_ack_does_not_count_twice(self):
        f = self.fixture()
        f.begin()
        ack = f.cc.ack_load_queue[0]
        f.complete()
        f.cache._complete_async_load_ack(ack)
        f.cache.metrics_collector.increment_load_back_num_tokens.assert_called_once()
        f.begin()

    def test_standard_duration_uses_only_explicit_gpu_time(self):
        collector = Mock()
        record_load_back_metrics(collector, {"kv": 3}, 512, 0.005)
        collector.increment_load_back_num_tokens.assert_called_once_with(
            num_tokens=3, pool="kv"
        )
        collector.increment_load_back_num_bytes.assert_called_once_with(512)
        collector.observe_load_back_duration.assert_called_once_with(0.005)
        collector.reset_mock()
        record_load_back_metrics(collector, {"kv": 3}, 512)
        collector.observe_load_back_duration.assert_not_called()

    def test_failed_io_does_not_count_success(self):
        f = self.fixture()
        f.begin()
        f.future.set_exception(ValueError("restore failed"))
        f.poll()
        self.assertEqual(f.state.stats["completed_restore_pages"], 0)
        self.assertEqual(f.cache.metrics_collector.mock_calls, [])
        self.assertEqual(f.pool.active_readers, 0)

    def test_device_pin_failure_releases_host_without_invalidating_cache(self):
        f = self.fixture()
        f.cache.inc_lock_ref.side_effect = ValueError("lock failed")
        self.assertIsNone(f.begin())
        f.cache.dec_host_lock_ref.assert_called_once()
        f.cache.dec_lock_ref.assert_not_called()
        self.assertIsNotNone(f.cd.host_value)
        f.req.set_finish_with_abort.assert_called_once()

    def test_ack_append_failure_after_submit_retains_resources(self):
        for error in [RuntimeError("ack"), CompressionCapacityError("ack")]:
            with self.subTest(error=type(error).__name__):
                f = self.fixture()
                f.cc.ack_load_queue = NS(append=Mock(side_effect=error))
                self.assertIsNone(f.begin())
                self.assertTrue(f.state.quarantined)
                f.cache.token_to_kv_pool_allocator.free.assert_not_called()
                f.cache.dec_host_lock_ref.assert_not_called()
                self.assertEqual(f.pool.active_readers, 3)
                # Simulate an external drain solely to clean up this fixture.
                f.future.set_result(None)
                for lease in f.state.quarantined[0][1]:
                    lease.close()

    def test_backup_spec_failure_rolls_back_prepared_pins(self):
        f = self.fixture()
        f.cd.host_value = None
        f.node.parent = f.cache.tree_core.root_node = NS()
        f.cache.buffer_pipeline = None
        f.cache.tree_core.build_backup_spec = Mock(
            side_effect=ValueError("spec failed")
        )
        run = source_function(
            "mem_cache/unified_radix_cache.py",
            "_execute_and_commit_kv_backup",
            "UnifiedRadixCache",
        )
        with self.assertRaisesRegex(ValueError, "spec failed"):
            run(f.cache, NS(node_ids=[1]))
        f.cache.dec_lock_ref.assert_called_once()
        self.assertFalse(f.state.backup_pins)
        self.assertFalse(f.state.quarantined)

    def test_backup_sidecar_failure_rolls_back_and_submission_failure_isolated(self):
        for stage in ("sidecar", "submission"):
            with self.subTest(stage=stage):
                f = self.fixture()
                f.cd.host_value = None
                f.node.parent = f.cache.tree_core.root_node = NS()
                f.cache.buffer_pipeline = None
                f.cache.tree_core.build_backup_spec = Mock(
                    return_value=(torch.arange(3), {})
                )
                f.cache._build_backup_sidecar = Mock(return_value=[])
                f.cache._execute_kv_backup = Mock(side_effect=RuntimeError("handoff"))
                if stage == "sidecar":
                    f.cache._build_backup_sidecar.side_effect = RuntimeError("sidecar")
                run = source_function(
                    "mem_cache/unified_radix_cache.py",
                    "_execute_and_commit_kv_backup",
                    "UnifiedRadixCache",
                )
                with self.assertRaises(RuntimeError):
                    run(f.cache, NS(node_ids=[1]))
                if stage == "sidecar":
                    self.assertFalse(f.state.backup_pins)
                    f.cache.dec_lock_ref.assert_called_once()
                    f.cache._execute_kv_backup.assert_not_called()
                else:
                    self.assertTrue(f.state.quarantined)
                    f.cache.dec_lock_ref.assert_not_called()

    def test_restore_submission_uncertain_retains_references(self):
        f = self.fixture()
        f.cc.l2_transfer_engine.submit_async_restore.side_effect = BufferDrainError(
            "uncertain"
        )
        self.assertIsNone(f.begin())
        self.assertTrue(f.state.quarantined)
        f.cache.token_to_kv_pool_allocator.free.assert_not_called()
        f.cache.dec_host_lock_ref.assert_not_called()
        self.assertEqual(f.pool.active_readers, 3)
        for lease in f.state.quarantined[0][1]:
            lease.close()  # Fixture-only simulated external drain.

    def test_restore_preparation_cleanup_failure_quarantines(self):
        f = self.fixture()
        f.params.mem_quota = 2
        f.cache.dec_lock_ref.side_effect = RuntimeError("release failed")
        with self.assertRaises(BufferDrainError):
            f.begin()
        self.assertTrue(f.state.quarantined)
        f.cc.l2_transfer_engine.submit_async_restore.assert_not_called()

    def test_restore_task_reports_physical_logical_and_gpu_time_separately(self):
        from collections import defaultdict

        f = self.fixture()
        leases = [f.pool.acquire(h) for h in f.handles.tolist()]
        runtime = NS(stream=Mock(), stats=defaultdict(float))

        def restore(*a):
            runtime.stats["restore_copy_gpu_ms"] += 2
            runtime.stats["decompress_gpu_ms"] += 3
            runtime.stats["writeback_gpu_ms"] += 5

        runtime.restore = restore
        runtime.submit = lambda fn, priority: result(fn())
        fn = source_function(
            "mem_cache/l2_transfer.py",
            "submit_async_restore",
            "L2TransferEngine",
            RestoreTransferResult=RestoreTransferResult,
            TransferCompletion=TransferCompletion,
        )
        try:
            completion = fn(
                NS(_start_event=lambda e: object()),
                NS(runtime=runtime),
                leases,
                [0, 1, 2],
            )
            stats = completion.result()
            self.assertEqual(stats.pages, 3)
            self.assertEqual(stats.actual_bytes, 12288)
            self.assertEqual(stats.logical_bytes, 12288)
            self.assertEqual(stats.gpu_seconds, 0.01)
            self.assertGreaterEqual(stats.queue_seconds, 0)
            self.assertEqual(completion.actual_bytes, stats.actual_bytes)
        finally:
            for lease in leases:
                lease.close()

    def test_restore_fence_failure_requires_a_drain(self):
        from collections import defaultdict

        for uncertain in (False, True):
            with self.subTest(uncertain=uncertain):
                runtime = NS(
                    stream=NS(wait_event=Mock(side_effect=ValueError("fence"))),
                    stats=defaultdict(float),
                    restore=Mock(),
                    drain=Mock(),
                    quarantine=Mock(),
                )
                if uncertain:
                    runtime.drain.side_effect = BufferDrainError("drain failed")

                def submit(fn, priority):
                    try:
                        return result(fn())
                    except (ValueError, BufferDrainError) as exc:
                        return result(error=exc)

                runtime.submit = submit
                fn = source_function(
                    "mem_cache/l2_transfer.py",
                    "submit_async_restore",
                    "L2TransferEngine",
                    RestoreTransferResult=RestoreTransferResult,
                    TransferCompletion=TransferCompletion,
                    BufferDrainError=BufferDrainError,
                )
                completion = fn(
                    NS(_start_event=lambda e: object()), NS(runtime=runtime), [], []
                )
                self.assertEqual(
                    completion.state,
                    TransferState.UNCERTAIN if uncertain else TransferState.FAILED,
                )
                runtime.drain.assert_called_once()
                self.assertEqual(runtime.quarantine.call_count, int(uncertain))
                runtime.restore.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class FragmentedBatchAllocationTests(unittest.TestCase):
    def pool(self):
        return BlockPool(8192, 32 * 1024**2, reservation_bytes=12288, pin_memory=False)

    def assert_capacity(self, pool):
        self.assertEqual(
            pool.available_size(),
            min(pool.free_count, pool.free_block_count // pool.blocks_per_page),
        )
        self.assertEqual(
            pool.free_block_count * BLOCK_BYTES
            + pool.live_payload_bytes
            + pool.reserved_bytes
            + pool.retired_bytes,
            pool.arena.numel(),
        )
        free = pool.free_blocks[: pool.free_block_count]
        self.assertEqual(len(np.unique(free)), len(free))
        self.assertTrue(np.all(pool.block_owner[free] == -1))

    def test_batch_never_restarts_search_per_page(self):
        pool = self.pool()
        old = pool.alloc(1024)
        pool.free(old[::2])
        before = pool.allocator_snapshot()
        new = pool.alloc(512)
        self.assertEqual(
            pool.allocator_snapshot()["allocation_blocks"]
            - before["allocation_blocks"],
            1536,
        )
        self.assertFalse(hasattr(pool, "extents"))
        self.assert_capacity(pool)
        pool.free(new)
        pool.free(old[1::2])
        self.assert_capacity(pool)

    def test_capacity_queries_do_not_scan_blocks(self):
        pool = self.pool()
        with patch("numpy.sum", side_effect=AssertionError("capacity scan")):
            self.assertTrue(pool.can_reserve(1024))

    def test_generation_failure_does_not_change_reservations(self):
        pool = self.pool()
        slot = pool.free_slots[pool.free_count - 1]
        pool.records[slot]["generation"] = 0x7FFFFFFF
        before = pool.snapshot()
        with self.assertRaises(RuntimeError):
            pool.alloc(3)
        self.assertEqual(pool.snapshot(), before)

    def test_tensor_creation_failure_leaves_pool_unchanged(self):
        pool = self.pool()
        before = pool.snapshot()
        with (
            patch("torch.from_numpy", side_effect=MemoryError("injected")),
            self.assertRaises(MemoryError),
        ):
            pool.alloc(3)
        self.assertEqual(pool.snapshot(), before)

    def test_failed_capacity_records_cost_without_mutating_state(self):
        pool = self.pool()
        before = pool.snapshot()
        self.assertIsNone(pool.alloc(pool.size + 1))
        self.assertEqual(pool.snapshot(), before)
        self.assertEqual(pool.allocator_snapshot()["allocation_failures"], 1)

    def test_batch_free_preparation_failure_is_atomic(self):
        pool = self.pool()
        h = pool.alloc(64)
        before = pool.snapshot()
        with (
            patch.object(pool, "_chains", side_effect=MemoryError("injected")),
            self.assertRaises(MemoryError),
        ):
            pool.free(h)
        self.assertEqual(pool.snapshot(), before)
        pool.free(h)
        self.assert_capacity(pool)

    def test_no_mutation_after_failed_allocation_preparation(self):
        pool = self.pool()
        h = pool.alloc(pool.available_size())
        pool.free(h[::2])
        before = pool.snapshot()
        with (
            patch("numpy.repeat", side_effect=MemoryError("injected")),
            self.assertRaises(MemoryError),
        ):
            pool.alloc(8)
        self.assertEqual(pool.snapshot(), before)
        new = pool.alloc(len(h[::2]))
        pool.free(new)
        pool.free(h[1::2])
        self.assert_capacity(pool)

    def test_large_node_reservation_with_reverse_free(self):
        pool = BlockPool(
            147456, 256 * 1024**2, reservation_bytes=197632, pin_memory=False
        )
        h = pool.alloc(pool.available_size())
        pool.free(h.flip(0))
        self.assert_capacity(pool)

    def test_random_churn(self):
        rng = np.random.default_rng(34)
        pool = self.pool()
        live = []
        for _ in range(200):
            if live and rng.random() < 0.55:
                rng.shuffle(live)
                removed, live = live[:16], live[16:]
                pool.free(removed)
            else:
                live.extend(pool.alloc(min(pool.available_size(), 32)).tolist())
            self.assert_capacity(pool)
        pool.free(live)
        self.assert_capacity(pool)

    def test_parallel_reservation_release(self):
        pool = self.pool()

        def work():
            for _ in range(50):
                h = pool.alloc(32)
                pool.free(h)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for f in [executor.submit(work) for _ in range(4)]:
                f.result(timeout=30)
        self.assert_capacity(pool)
        self.assertEqual(pool.active_readers, 0)


class BackupAdmissionTimingTests(unittest.TestCase):
    def test_success_skip_and_error_all_record_admission_time(self):
        method = source_function(
            "mem_cache/unified_radix_cache.py",
            "_execute_kv_backup",
            "UnifiedRadixCache",
        )
        for outcome in ("success", "skip", "error"):
            with self.subTest(outcome=outcome):
                pool = CompressedHostKVCache(1024, 1024**2, pin_memory=False)
                if outcome == "skip":
                    pool.alloc(pool.available_size())
                controller = NS(
                    mem_pool_host=pool, write=Mock(return_value="submitted")
                )
                if outcome == "error":
                    controller.write.side_effect = RuntimeError("injected")
                cache = NS(
                    cache_controller=controller,
                    async_l2=NS(pool=pool, stats={"admission_skips": 0}),
                    evict_host=Mock(return_value=0),
                )
                with patch.object(time, "perf_counter", side_effect=[10.0, 12.0]):
                    if outcome == "error":
                        with self.assertRaises(RuntimeError):
                            method(cache, 1, [1], {}, [])
                    else:
                        self.assertEqual(
                            method(cache, 1, [1], {}, []),
                            None if outcome == "skip" else "submitted",
                        )
                self.assertEqual(cache.async_l2.stats["backup_admission_seconds"], 2.0)

    def test_native_path_does_not_start_compression_timer(self):
        method = source_function(
            "mem_cache/unified_radix_cache.py",
            "_execute_kv_backup",
            "UnifiedRadixCache",
        )
        pool = CompressedHostKVCache(1024, 1024**2, pin_memory=False)
        cache = NS(
            cache_controller=NS(mem_pool_host=pool, write=Mock(return_value="native")),
            async_l2=None,
        )
        with patch.object(
            time, "perf_counter", side_effect=AssertionError("compression timer")
        ):
            self.assertEqual(method(cache, 1, [1], {}, []), "native")

    def test_allocator_and_admission_metrics_are_exported(self):
        from collections import defaultdict

        update = source_function(
            "kv_compression/metrics.py", "update", "CompressionMetrics"
        )
        exporter = NS(
            memory=Mock(),
            pages=Mock(),
            events=Mock(),
            copies=Mock(),
            seconds=Mock(),
            allocator_work=Mock(),
            allocator_lock_max=Mock(),
            _increment=Mock(),
        )
        store = defaultdict(
            float,
            allocation_seconds=2.0,
            allocation_max_lock_seconds=1.0,
            allocation_blocks=40000,
        )
        adapter = defaultdict(float, backup_admission_seconds=3.0)
        update(exporter, defaultdict(float), adapter, store)
        exporter._increment.assert_any_call(exporter.seconds, "backup_admission", 3.0)
        exporter._increment.assert_any_call(exporter.seconds, "allocation_seconds", 2.0)
        exporter._increment.assert_any_call(
            exporter.allocator_work, "allocation_blocks", 40000
        )
        exporter.allocator_lock_max.set.assert_called_once_with(1.0)
