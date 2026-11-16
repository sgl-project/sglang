"""A band tracks the pages a host copy touches by the ids the copy reads.

A host copy reads its controller ids through the transfer engine's rule
(collapsed by the host pool's DCP width, then paged as the storage's
``host_ids`` says), while the band tracks transfers in its local virtual ids
over physical-row pages: both must name the same physical pages.
"""

import unittest
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.memory.virtual_slot import translate_token_ids
from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.mem_cache import unified_radix_cache
from sglang.srt.mem_cache.common import RetractionBackup
from sglang.srt.mem_cache.kv.alloc.band import Band
from sglang.srt.mem_cache.kv.alloc.chain import ChainRoles
from sglang.srt.mem_cache.kv.alloc.id_space import UnitIds
from sglang.srt.mem_cache.kv.alloc.page_usage import PageUsage
from sglang.srt.mem_cache.kv.storage import KVStorage
from sglang.srt.mem_cache.l2_transfer import L2Transfer
from sglang.srt.runtime_context import get_memory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _band_stub(*, page_size, pool_page_size, device="cpu"):
    ids = UnitIds(
        num_virtual_ids=16,
        num_physical_units=16,
        unit_tokens=pool_page_size,
        shards_under_dcp=False,
        device=device,
    )
    return SimpleNamespace(
        ids=ids,
        page_size=page_size,
        pool_page_size=pool_page_size,
        shards_under_dcp=False,
    )


def _storage_stub(band, *, owner="full"):
    return SimpleNamespace(
        _allocator=SimpleNamespace(roles=ChainRoles(owner=owner)),
        _member=lambda _name: band,
    )


def _transfer(ids, *, host_dcp_size=1):
    return L2Transfer(
        host_pool=SimpleNamespace(dcp_size=host_dcp_size),
        device_pool=None,
        host_indices=None,
        device_indices=ids,
    )


def _tracked_pages(storage, band, sub_pool, transfer):
    ids = KVStorage._host_copy_ids(storage, sub_pool, transfer)
    return Band._physical_pages_of(band, ids, widened=False)


class TestHostCopyIds(CustomTestCase):
    def test_widened_envelope_pages_match_the_copy(self):
        """DCP 2 over an envelope: 4 physical rows a page, 8 logical ids a
        page; ids 24..31 name virtual page 3, bound to physical page 9."""
        band = _band_stub(page_size=8, pool_page_size=4)
        band.ids.virtual_to_physical[3] = 9
        ids = torch.arange(24, 32, dtype=torch.int64)
        copied = translate_token_ids(
            loc=ids,
            v2p=band.ids.virtual_to_physical,
            page_size=band.page_size,
            dcp_size=1,
            dcp_rank=0,
        )
        self.assertEqual(set((copied // band.page_size).tolist()), {9})
        tracked = _tracked_pages(_storage_stub(band), band, "full", _transfer(ids))
        self.assertEqual(tracked, {9})

    def test_host_pool_dcp_collapses_before_paging(self):
        """A host pool that shards under DCP hands widened ids: the copy
        translates them collapsed, and so does the tracking."""
        band = _band_stub(page_size=4, pool_page_size=4)
        band.ids.virtual_to_physical[2] = 11
        ids = torch.arange(16, 24, dtype=torch.int64)  # widened, DCP 2
        tracked = _tracked_pages(
            _storage_stub(band), band, "full", _transfer(ids, host_dcp_size=2)
        )
        self.assertEqual(tracked, {11})

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_host_resident_ids_translate_on_the_band_device(self):
        """A direct I/O backend hands the ids on the host; the band's tables
        live on the device."""
        band = _band_stub(page_size=4, pool_page_size=4, device="cuda")
        band.ids.virtual_to_physical[5] = 7
        ids = torch.arange(20, 24, dtype=torch.int64)  # on the host
        tracked = _tracked_pages(_storage_stub(band), band, "full", _transfer(ids))
        self.assertEqual(tracked, {7})


class _Streams:
    """Stand-ins for the scheduler's streams. Logs, in order, which stream is
    current at each step of a restore and every cross-stream wait."""

    def __init__(self):
        self.log = []
        self.schedule = _Stream("schedule", self)
        self.forward = _Stream("forward", self)
        self.current = self.schedule

    @contextmanager
    def on(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    def step(self, what):
        self.log.append((what, self.current.name))


class _Stream:
    def __init__(self, name, streams):
        self.name = name
        self.streams = streams

    def wait_stream(self, other):
        self.streams.log.append(("wait", f"{self.name}<-{other.name}"))


def _restoring_cache(streams):
    """A cache whose restore runs the real `start_restore_kv_cache` over
    stand-in ids, copy and storage, logging the stream of each step."""
    engine = SimpleNamespace(
        host_to_device_stream=_Stream("copy", streams),
        on_current_stream=lambda: _logged(streams, "copy-on-current"),
        submit_host_to_device=lambda transfers, **_: (
            streams.step("copy"),
            SimpleNamespace(finish_event=object()),
        )[1],
    )

    def device_transfers(req):
        streams.step("ids")
        return torch.arange(4, dtype=torch.int64), []

    cache = SimpleNamespace(
        _retraction_device_transfers=device_transfers,
        cache_controller=SimpleNamespace(
            l2_transfer_engine=engine,
            transfer_layer_id_max=1,
            _resolve_device_transfers=lambda *_, **__: None,
            _move_op_indices=lambda op: (
                op.host_indices,
                op.device_indices,
                op.pool_transfers,
            ),
            _l2_load_transfers=lambda host, device, pools: [
                SimpleNamespace(device_indices=device)
            ],
        ),
        kv_storage=lambda: SimpleNamespace(
            record_host_transfer=lambda *_, **__: streams.step("record")
        ),
        _pending_restore_backups=[],
    )
    cache.start_restore_kv_cache = MethodType(
        unified_radix_cache.UnifiedRadixCache.start_restore_kv_cache, cache
    )
    return cache


@contextmanager
def _logged(streams, what):
    streams.step(what)
    yield


class TestRestoreRecordedWhereMovesReadIt(CustomTestCase):
    """A decode host restore builds and records its copy's ids on the schedule
    stream, where page moves read them; built on the forward stream, behind
    the running forward, a move could read them before they exist and copy
    over a page the restore writes."""

    def _start(self, io_backend):
        setup_allocator_context()
        streams = _Streams()
        req = SimpleNamespace(
            seqlen=5,
            kv=SimpleNamespace(
                retraction_backup=RetractionBackup(
                    host_indices=torch.arange(4), pool_transfers=None
                ),
                restore_done_event=None,
            ),
        )
        scheduler = SimpleNamespace(
            waiting_queue=[req],
            tree_cache=_restoring_cache(streams),
            forward_stream=streams.forward,
            schedule_stream=streams.schedule,
            enable_overlap=True,
        )
        device = SimpleNamespace(
            current_stream=lambda: streams.current, stream=streams.on
        )
        with (
            get_memory().override(hicache_io_backend=io_backend),
            mock.patch.object(unified_radix_cache, "device_module", device),
            mock.patch.object(torch.cuda, "stream", streams.on),
        ):
            SchedulerDisaggregationDecodeMixin._start_host_restores(scheduler)
        self.assertIsNotNone(req.kv.restore_done_event)
        return streams.log

    def test_ids_and_record_on_the_schedule_stream_copy_on_the_forward(self):
        log = self._start("kernel")
        self.assertIn(("ids", "schedule"), log)
        self.assertIn(("record", "schedule"), log)
        self.assertIn(("copy", "forward"), log)
        # The copy reads the ids: its stream waits for them first.
        ids, copy = log.index(("ids", "schedule")), log.index(("copy", "forward"))
        waits = [
            i for i, step in enumerate(log) if step == ("wait", "forward<-schedule")
        ]
        self.assertTrue(any(ids < i < copy for i in waits), log)

    def test_direct_backend_records_on_the_schedule_stream(self):
        log = self._start("direct")
        self.assertIn(("ids", "schedule"), log)
        self.assertIn(("record", "schedule"), log)
        self.assertNotIn(("copy", "forward"), log)


class TestRetirementBookkeeping(CustomTestCase):
    def test_urgent_retirements_keep_the_event_map_bounded(self):
        usage = PageUsage()
        for _ in range(10_000):
            event = object()
            usage.register_forward(event)
            usage.retire_through(event)  # an urgent wait retires it first
            usage.retire_forward(event)  # then the scheduler processes it
        self.assertEqual(len(usage.event_seq), 0)
        for _ in range(10_000):
            usage.register_forward(object())
            usage.retire_all()  # a settle waits the newest forward
        self.assertEqual(len(usage.event_seq), 0)


if __name__ == "__main__":
    unittest.main()
