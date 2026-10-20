"""Direct GPU delivery ordering and ownership, with a controlled KVCR boundary."""

import json
import threading
import time
import unittest
from collections import Counter, deque
from concurrent.futures import Future
from copy import copy
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class Event:
    def __init__(self):
        self.ready = True

    def record(self):
        pass

    def query(self):
        return self.ready


class Client:
    def __init__(self):
        self.calls = []
        self.results = deque()
        self.hints = {}
        self.status = {}
        self.next_handle = 0
        self.polls = 0
        self.fail_poll = False
        self.fetch_handles = set()
        self.claims = set()
        self.fetch_data = {}

    def submit_hint(self, hint, request_id=None):
        self.hints[request_id] = hint

    def discard_hint(self, request_id):
        self.hints.pop(request_id, None)

    def query(self, keys, request_id=None):
        return [self.status.get(k, (SimpleNamespace(value="MISS"), None)) for k in keys]

    def deliver(self, blocks, request_id=None):
        return self._submit("deliver", blocks, request_id)

    def deposit(self, blocks):
        return self._submit("deposit", blocks, None)

    def fetch(self, keys, request_id=None, expected_layout=None):
        handle = self._submit("fetch", dict.fromkeys(keys, expected_layout), request_id)
        self.fetch_handles.add(handle)
        self.complete(
            handle,
            success={key: self.fetch_data.get(key) == expected_layout for key in keys},
        )
        return handle

    def release(self, handles):
        results = [(handle, handle in self.claims) for handle in handles]
        self.claims.difference_update(handles)
        return results

    def _submit(self, kind, blocks, request_id):
        self.next_handle += 1
        self.calls.append(
            (self.next_handle, kind, blocks, request_id, threading.get_ident())
        )
        return self.next_handle

    def complete(self, handle, success=True):
        blocks = next(call[2] for call in self.calls if call[0] == handle)
        result = {}
        for key in blocks:
            ok = success if isinstance(success, bool) else success[key]
            claim = (handle, key) if ok and handle in self.fetch_handles else None
            if claim is not None:
                self.claims.add(claim)
            result[key] = SimpleNamespace(success=ok, release_handle=claim)
        self.results.append((handle, result))

    def poll_completed(self):
        self.polls += 1
        if self.fail_poll:
            raise RuntimeError("owner poll failed")
        while self.results:
            yield self.results.popleft()


class Runtime:
    def __init__(self, *args, **kwargs):
        self.client = Client()
        self.events = deque()
        self.removals = {}
        self.closed = False
        self.close_error = False

    def log_stats(self):
        pass

    def close(self):
        if self.close_error:
            raise RuntimeError("DMA still active")
        self.closed = True


def eventually(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("owner did not make progress")
        time.sleep(0.001)


class TestKVCRDirectLinker(CustomTestCase):
    def setUp(self):
        from sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker import (
            KVCRDirectLinker,
        )

        self.scope = get_context().override_server_args(model_path="dummy")
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)
        self.group = DevicePoolGroup(
            [
                DevicePoolEntry(
                    name=PoolName.KV,
                    indices_from_pool=PoolName.KV,
                    device_pool=None,
                    components=[[torch.zeros((4096, 4)) for _ in range(3)]],
                    layer_mapping={0: 0, 1: 1, 2: 2},
                    page_size=1,
                    rows_are_pages=True,
                )
            ],
            num_layers=3,
            page_size=1,
        )
        self.runtime = Runtime()
        module = "sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker"
        self.patches = [
            patch.dict(
                "sys.modules",
                {
                    "kvcr.types": SimpleNamespace(
                        RegionDescriptor=SimpleNamespace, MemoryRef=SimpleNamespace
                    )
                },
            ),
            patch(
                module + ".resolve_hybrid_device_pool_group", return_value=self.group
            ),
            patch(module + ".torch.cuda.Event", side_effect=Event),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.params = SimpleNamespace(
            page_size=1,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            tp_cache_group=None,
            attn_tp_cache_group=None,
        )
        self.linker = KVCRDirectLinker(
            SimpleNamespace(enable_linker_mla_dedup=False),
            self.params,
            components=set(),
            runtime_factory=lambda *args, **kwargs: self.runtime,
        )
        self.addCleanup(self.linker.close)
        self.client = self.runtime.client

    def transfer(self, keys, start=0):
        return [
            PoolTransfer(
                name=PoolName.KV,
                keys=keys,
                device_indices=torch.arange(start, start + len(keys)),
            )
        ]

    def test_layer_first_window_and_no_implicit_deposit(self):
        self.linker.load("a", self.transfer(["same"], 0))
        self.linker.load("b", self.transfer(["same"], 1))
        index = self.linker.start_layer_wise_loading()
        eventually(lambda: len(self.client.calls) == 4)
        first = next(iter(self.client.calls[0][2].values()))[0]
        second = next(iter(self.client.calls[1][2].values()))[0]
        self.assertEqual((first.element_index, second.element_index), (0, 1))
        # The second request's first layer may finish before the first request's.
        self.client.complete(2)
        eventually(lambda: len(self.client.calls) == 5)
        self.client.complete(1)
        self.linker.layer_done_counter.set_consumer(index)
        self.linker.layer_done_counter.wait_until(0)
        self.assertEqual(self.linker.num_completed_loads(), 0)
        eventually(lambda: len(self.client.calls) == 6)
        for handle in [6, 4, 5, 3]:
            self.client.complete(handle)
        eventually(lambda: self.linker.num_completed_loads() == 1)
        self.assertEqual(self.linker.pop_completed_load(), ["a", "b"])
        self.assertEqual(
            {call[4] for call in self.client.calls}, {self.linker._thread.ident}
        )
        self.assertTrue(all(call[1] == "deliver" for call in self.client.calls))

    def test_large_delivery_is_not_page_chunked_and_snapshots_indices(self):
        transfers = self.transfer([f"{i:064x}" for i in range(2051)])
        self.linker.load("large", transfers)
        transfers[0].device_indices.fill_(0)
        self.linker.start_layer_wise_loading()
        eventually(lambda: len(self.client.calls) == 3)
        self.assertEqual([len(call[2]) for call in self.client.calls], [2051] * 3)
        refs = [refs[0].element_index for refs in self.client.calls[0][2].values()]
        self.assertEqual(refs, list(range(2051)))
        for handle in [1, 2, 3]:
            self.client.complete(handle)

    def test_queued_cancel_never_starts_device_writes(self):
        """Only an unstarted, request-private transfer may return its slots."""
        self.linker.load("cancel", self.transfer(["page"]))
        self.assertTrue(self.linker.cancel_queued_load("cancel"))
        self.assertFalse(self.linker.cancel_queued_load("cancel"))
        self.assertEqual(self.linker.start_layer_wise_loading(), -1)
        self.assertEqual(self.client.calls, [])

    def test_forward_completion_releases_batch_and_layer_bookkeeping(self):
        """Readiness for the last layer is not a substitute for batch drain."""
        self.linker.load("finish", self.transfer(["page"]))
        index = self.linker.start_layer_wise_loading()
        eventually(lambda: len(self.client.calls) == 3)
        for handle in [1, 2, 3]:
            self.client.complete(handle)
        self.assertTrue(self.linker.finish_layer_wise_loading(index))
        self.assertNotIn(index, self.linker.layer_done_counter.futures)
        self.assertEqual(self.linker._batch_results, {})
        self.assertEqual(self.linker._loads, {})

    def test_active_progress_skips_idle_wait_then_resumes_peer_polling(self):
        """An empty command queue must not pace completion and window refills."""
        active_waits, idle_waits = [], []
        get_command, submit = self.linker._commands.get, self.client._submit

        def get(block=True, timeout=None):
            if self.client.calls:
                waits = (
                    idle_waits if self.linker.num_completed_offloads() else active_waits
                )
                waits.append(timeout if block else 0)
            return get_command(block=block, timeout=timeout)

        def complete_immediately(kind, blocks, request_id):
            handle = submit(kind, blocks, request_id)
            self.client.complete(handle)
            return handle

        with (
            patch.object(self.linker._commands, "get", side_effect=get),
            patch.object(self.client, "_submit", side_effect=complete_immediately),
        ):
            for rid in ("a", "b"):
                self.linker.load(rid, self.transfer([rid]))
            self.linker.offload(self.transfer(["a"]))
            self.linker.start_layer_wise_loading()
            eventually(lambda: len(idle_waits) >= 2)
        self.assertEqual(self.linker.pop_completed_load(), ["a", "b"])
        self.assertTrue(self.linker.pop_completed_offload())
        self.assertEqual(len(self.client.calls), 7)
        self.assertTrue(active_waits)
        self.assertEqual(set(active_waits), {0})
        self.assertTrue(all(wait is not None and wait > 0 for wait in idle_waits))

    def test_offload_snapshots_before_preparation_and_holds_physical_keys(self):
        """Pending producers must not force row reads or permit reused indices/keys."""
        from sglang.srt.mem_cache.storage.kvcr.device_layout import KVCRLayout

        base = self.group.entries[0]
        indexer = copy(base)
        indexer.name = PoolName.INDEXER
        self.linker.layout = KVCRLayout(
            DevicePoolGroup([base, indexer], 3, 1),
            model_name="dummy",
            endpoint_name="test",
        )
        ready = Event()
        ready.ready = False
        transfers = self.transfer(["before"], 1)
        keys = {
            self.linker.layout.encode_key("before", name)
            for name in (PoolName.KV, PoolName.INDEXER)
        }
        resolve = self.linker.layout.prepare

        def prepare_off_thread(*args, **kwargs):
            self.assertEqual(threading.get_ident(), self.linker._thread.ident)
            return resolve(*args, **kwargs)

        with (
            patch("torch.cuda.Event", return_value=ready),
            patch.object(
                self.linker.layout, "prepare", side_effect=prepare_off_thread
            ) as prepare,
        ):
            self.assertTrue(self.linker.offload(transfers))
            transfers[0].device_indices.fill_(7)
            transfers[0].keys[0] = "after"
            self.runtime.removals.update({key: "before" for key in keys})
            self.assertEqual(self.linker.pop_storage_removals(), [])
            self.linker._call(lambda: None)
            prepare.assert_not_called()
            ready.ready = True
            eventually(lambda: len(self.client.calls) == 1)
        blocks = self.client.calls[0][2]
        self.assertEqual(set(blocks), keys)
        for refs in blocks.values():
            self.assertEqual(len(refs), 3)
            self.assertEqual({ref.element_index for ref in refs}, {1})
        self.client.complete(1)
        eventually(lambda: self.linker.num_completed_offloads() == 1)
        self.assertTrue(self.linker.pop_completed_offload())
        self.assertCountEqual(self.linker.pop_storage_removals(), ["before", "before"])
        with patch.object(
            self.linker.layout, "prepare", side_effect=ValueError("bad indices")
        ):
            self.assertTrue(self.linker.offload(self.transfer(["bad"])))
            eventually(lambda: self.linker.num_completed_offloads() == 1)
            self.assertFalse(self.linker.pop_completed_offload())
            self.assertIsNone(self.linker._fatal)

    def test_offload_yields_to_commands_and_serializes_duplicate_destinations(self):
        """Queued commands precede offload work; repeated keys need separate deposits."""
        polling, resume = threading.Event(), threading.Event()
        poll = self.client.poll_completed

        def gated_poll():
            if self.linker._offloads:
                polling.set()
                resume.wait(5)
            return poll()

        with patch.object(self.client, "poll_completed", side_effect=gated_poll):
            try:
                self.linker.offload(self.transfer(["a", "a"]))
                self.assertTrue(polling.wait(5))
                with patch.object(self.linker, "_call", side_effect=AssertionError):
                    self.assertEqual(self.linker.start_layer_wise_loading(), -1)
                command = Future()
                self.linker._commands.put((lambda: len(self.client.calls), (), command))
                resume.set()
                self.assertEqual(command.result(timeout=5), 0)
                eventually(lambda: len(self.client.calls) == 1)
                polls = self.client.polls
                eventually(lambda: self.client.polls > polls + 2)
                self.assertEqual(len(self.client.calls), 1)
                self.client.complete(1)
                eventually(lambda: len(self.client.calls) == 2)
                refs = [next(iter(call[2].values())) for call in self.client.calls]
                self.assertEqual([len(group) for group in refs], [3, 3])
                self.assertEqual([group[0].element_index for group in refs], [0, 1])
                self.client.complete(2)
            finally:
                resume.set()
        eventually(lambda: self.linker.num_completed_offloads() == 1)
        self.assertTrue(self.linker.pop_completed_offload())

    def test_offload_waits_for_producer_and_earlier_queued_load(self):
        self.linker.load("a", self.transfer(["a"]))
        first_ready, second_ready, load_ready = Event(), Event(), Event()
        second_ready.ready = load_ready.ready = False
        with patch(
            "torch.cuda.Event", side_effect=[first_ready, second_ready, load_ready]
        ):
            self.linker.offload(self.transfer(["a"]))
            self.linker.offload(self.transfer(["b"], 1))
            self.linker.start_layer_wise_loading()
        before = self.client.polls
        eventually(lambda: self.client.polls > before + 2)
        self.assertEqual(self.client.calls, [])
        load_ready.ready = True
        eventually(lambda: len(self.client.calls) == 3)
        self.assertTrue(all(call[1] == "deliver" for call in self.client.calls))
        for handle in [1, 3, 2]:
            self.client.complete(handle)
        eventually(lambda: len(self.client.calls) == 4)
        self.assertEqual(self.client.calls[-1][1], "deposit")
        self.client.complete(4)
        eventually(lambda: self.linker.num_completed_offloads() == 1)
        self.assertTrue(self.linker.pop_completed_offload())
        before = self.client.polls
        eventually(lambda: self.client.polls > before + 2)
        self.assertEqual(len(self.client.calls), 4)
        second_ready.ready = True
        eventually(lambda: len(self.client.calls) == 5)
        self.assertEqual(self.client.calls[-1][1], "deposit")
        self.client.complete(5)
        eventually(lambda: self.linker.num_completed_offloads() == 1)
        self.assertTrue(self.linker.pop_completed_offload())

    def test_hint_scope_cleanup_waits_for_accepted_load(self):
        a, b = CacheRequestHandle("a", 1), CacheRequestHandle("b", 2)
        hints = {"actions": [{"source": ["a"]}]}
        self.linker.set_request_context(a, hints)
        scope = next(iter(self.client.hints))
        hints["actions"][0]["source"].append("changed")
        self.assertEqual(self.client.hints[scope], {"actions": [{"source": ["a"]}]})
        self.linker.set_request_context(b, {"actions": [{"source": "b"}]})
        self.assertEqual(len(self.client.hints), 2)
        self.linker.load("a", self.transfer(["a"]))
        self.linker.release_request_context(a)
        next_attempt = CacheRequestHandle("a", 2)
        self.linker.set_request_context(next_attempt, {"actions": [{"source": "new"}]})
        self.linker.release_request_context(b)
        self.assertEqual(len(self.client.hints), 2)
        self.linker.release_request_context(next_attempt)
        self.assertEqual(self.client.hints, {scope: {"actions": [{"source": ["a"]}]}})
        self.linker.start_layer_wise_loading()
        eventually(lambda: len(self.client.calls) == 3)
        self.assertEqual({call[3] for call in self.client.calls}, {scope})
        for handle in [1, 2, 3]:
            self.client.complete(handle)
        eventually(lambda: not self.client.hints)

    def test_failure_holds_dma_ownership_until_quiesced(self):
        self.linker.load("a", self.transfer(["a"]))
        index = self.linker.start_layer_wise_loading()
        eventually(lambda: len(self.client.calls) == 3)
        self.runtime.events.append(SimpleNamespace(state="uncertain", op_handle=-9))
        for handle in [1, 2, 3]:
            self.client.complete(handle, success=False)
        self.linker.layer_done_counter.set_consumer(index)
        self.linker.layer_done_counter.wait_until(0)
        self.assertEqual(self.linker.num_completed_loads(), 0)
        self.runtime.events.append(SimpleNamespace(state="quiesced", op_handle=-9))
        eventually(lambda: self.linker.num_completed_loads() == 1)
        self.assertEqual(self.linker.pop_completed_load(), ["a"])
        self.assertFalse(self.linker.finish_layer_wise_loading(index))

    def test_owner_failure_rejects_commands_and_close_retains_resources(self):
        self.client.fail_poll = True
        eventually(lambda: self.linker._fatal is not None)
        with self.assertRaisesRegex(RuntimeError, "owner poll failed"):
            self.linker.lookup("a", self.transfer(["a"]))
        self.runtime.close_error = True
        with self.assertRaisesRegex(RuntimeError, "DMA still active"):
            self.linker.close()
        self.assertFalse(self.runtime.closed)
        self.assertTrue(self.linker._thread.is_alive())
        self.runtime.close_error = False
        self.linker.close()
        self.assertTrue(self.runtime.closed)

    def test_owner_failure_unblocks_finish_without_releasing_native_memory(self):
        """A poll failure after finish subscribes must not strand its waiter."""
        self.linker.load("finish-race", self.transfer(["page"]))
        index = self.linker.start_layer_wise_loading()
        eventually(lambda: len(self.client.calls) == 3)
        result = Future()

        def finish():
            try:
                result.set_result(self.linker.finish_layer_wise_loading(index))
            except BaseException as error:
                result.set_exception(error)

        waiter = threading.Thread(target=finish, daemon=True)
        waiter.start()
        try:
            eventually(lambda: index not in self.linker._batch_results)
            self.client.fail_poll = True
            with self.assertRaisesRegex(RuntimeError, "owner poll failed"):
                result.result(timeout=2)
            self.assertFalse(self.runtime.closed)
            self.assertIn("finish-race", self.linker._loads)
        finally:
            self.linker.close()
            waiter.join(timeout=2)

    def test_lookup_checks_all_and_trailing_boundaries_for_physical_pools(self):
        from sglang.srt.mem_cache.storage.kvcr.device_layout import KVCRLayout

        base = self.group.entries[0]
        # DeepSeek-V4's logical KV component has no physical pool named KV.
        base.name = PoolName.DEEPSEEK_V4_C4
        swa = DevicePoolEntry(
            name=PoolName.SWA,
            indices_from_pool=PoolName.SWA,
            device_pool=None,
            components=base.components,
            layer_mapping=base.layer_mapping,
            page_size=1,
            rows_are_pages=True,
        )
        self.linker.layout = KVCRLayout(
            DevicePoolGroup([base, swa], 3, 1), model_name="dummy", endpoint_name="test"
        )
        layout = self.linker.layout
        keys = ["a", "b", "c", "d", "e"]
        for key in keys:
            self.client.status[layout.encode_key(key, base.name)] = (
                SimpleNamespace(value="HIT"),
                SimpleNamespace(value="DRAM"),
            )
        for key in ["b", "c", "e"]:
            self.client.status[layout.encode_key(key, PoolName.SWA)] = (
                SimpleNamespace(value="FETCHABLE"),
                SimpleNamespace(value="REMOTE_G2"),
            )
        transfers = [
            PoolTransfer(PoolName.KV, keys=keys),
            PoolTransfer(
                PoolName.SWA, keys=keys[-2:], hit_policy=PoolHitPolicy.TRAILING_PAGES
            ),
        ]
        self.assertEqual(self.linker.lookup("a", transfers), [3])
        self.client.status[layout.encode_key("b", base.name)] = (
            SimpleNamespace(value="FETCHING"),
            SimpleNamespace(value="DRAM"),
        )
        self.assertEqual(self.linker.lookup("a", transfers), [])

    def trailing_transfers(self):
        from sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker import (
            KVCRDirectLinker,
        )

        self.assertFalse(self.linker.requires_lookup_confirmation)
        self.linker.close()
        base = self.group.entries[0]
        swa = copy(base)
        swa.name = swa.indices_from_pool = PoolName.SWA
        self.runtime = Runtime()
        with patch(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker.resolve_hybrid_device_pool_group",
            return_value=DevicePoolGroup([base, swa], 3, 1),
        ):
            self.linker = KVCRDirectLinker(
                None,
                self.params,
                components=set(),
                runtime_factory=lambda *a, **kw: self.runtime,
            )
        self.addCleanup(self.linker.close)
        self.client = self.runtime.client
        self.assertTrue(self.linker.requires_lookup_confirmation)
        self.linker.set_request_context(CacheRequestHandle("a", 0), {"actions": []})
        return [
            PoolTransfer(PoolName.KV, keys=["a", "b", "c"]),
            PoolTransfer(
                PoolName.SWA, keys=["c"], hit_policy=PoolHitPolicy.TRAILING_PAGES
            ),
        ]

    def test_trailing_confirmation_retains_only_complete_state_through_load(self):
        transfers = self.trailing_transfers()
        encode = self.linker.layout.encode_key
        labels = list(self.linker.layout._labels[PoolName.SWA][0])
        for pool in (PoolName.KV, PoolName.SWA):
            for key in "abc":
                self.client.status[encode(key, pool)] = (
                    SimpleNamespace(value="HIT"),
                    SimpleNamespace(value="DRAM"),
                )
        self.client.fetch_data[encode("b", PoolName.SWA)] = labels
        self.client.fetch_data[encode("c", PoolName.SWA)] = labels[:1]
        self.assertEqual(self.linker.lookup("a", transfers), [1, 2, 3])
        self.assertFalse(self.linker.confirm_lookup("a", transfers, 3))
        self.assertTrue(self.linker.confirm_lookup("a", transfers, 2))
        transfers[1].keys, transfers[1].device_indices = ["b"], torch.tensor([1])
        self.assertTrue(self.linker.load("a", [transfers[1]]))
        self.linker.release_request_context(CacheRequestHandle("a", 0))
        self.linker.start_layer_wise_loading()
        eventually(lambda: len(self.client.calls) == 5)
        self.client.complete(3, success=False)
        self.client.complete(4)
        eventually(lambda: len(self.linker._operations) == 1)
        self.assertTrue(self.client.claims)
        self.assertEqual(self.linker.num_completed_loads(), 0)
        self.client.complete(5)
        eventually(lambda: self.linker.num_completed_loads() == 1)
        self.assertFalse(self.client.claims)

    def test_trailing_confirmation_reuses_overlap_and_discards_unused_claims(self):
        transfers = self.trailing_transfers()
        transfers[1].keys = ["b", "c"]
        labels = list(self.linker.layout._labels[PoolName.SWA][0])
        a, b, c = [self.linker.layout.encode_key(k, PoolName.SWA) for k in "abc"]
        self.client.fetch_data.update({b: labels, c: labels})
        self.assertTrue(self.linker.confirm_lookup("a", transfers, 3))
        self.assertFalse(self.linker.confirm_lookup("a", transfers, 2))
        self.assertEqual({claim[1] for claim in self.client.claims}, {b})
        self.assertEqual(set(self.client.calls[-1][2]), {a})
        self.assertFalse(self.linker.confirm_lookup("a", transfers, 1))
        self.assertEqual(len(self.client.calls), 2)
        self.assertFalse(self.client.claims)
        self.client.fetch_data[a] = labels
        self.linker.lookup("a", transfers)
        self.assertTrue(self.linker.confirm_lookup("a", transfers, 1))
        self.linker.confirm_lookup("a", [], 0)
        self.assertFalse(self.client.claims)
        self.assertTrue(self.linker.confirm_lookup("a", transfers, 1))
        self.linker.release_request_context(CacheRequestHandle("a", 0))
        self.linker._call(lambda: None)
        self.assertFalse(self.client.claims)

    def test_trailing_confirmation_rechecks_after_reset(self):
        transfers = self.trailing_transfers()
        key = self.linker.layout.encode_key("c", PoolName.SWA)
        self.client.fetch_data[key] = list(self.linker.layout._labels[PoolName.SWA][0])
        self.assertTrue(self.linker.confirm_lookup("a", transfers, 3))
        with patch.object(self.linker, "_make_runtime", return_value=Runtime()):
            self.linker.reset()
        self.assertTrue(self.runtime.closed)
        self.linker.set_request_context(CacheRequestHandle("a", 0), {"actions": []})
        self.assertFalse(self.linker.confirm_lookup("a", transfers, 3))

    def test_failure_stops_unsubmitted_work_and_fails_dependent_offload(self):
        for i in range(3):
            self.linker.load(str(i), self.transfer([str(i)], i))
        index = self.linker.start_layer_wise_loading()
        self.linker.offload(self.transfer(["0"]))
        eventually(lambda: len(self.client.calls) == 4)
        self.client.complete(1, success=False)
        self.linker.layer_done_counter.set_consumer(index)
        self.linker.layer_done_counter.wait_until(0)
        self.assertEqual(self.linker.num_completed_loads(), 0)
        for handle in [2, 3, 4]:
            self.client.complete(handle)
        eventually(lambda: self.linker.num_completed_offloads() == 1)
        self.assertFalse(self.linker.pop_completed_offload())
        self.assertEqual(len(self.client.calls), 4)
        self.assertFalse(self.linker.finish_layer_wise_loading(index))

    def test_reset_quiesces_and_preserves_registered_counter(self):
        counter = self.linker.layer_done_counter
        self.linker.load("a", self.transfer(["a"]))
        self.linker.start_layer_wise_loading()
        self.runtime.removals[self.linker.layout.encode_key("a", PoolName.KV)] = "a"
        replacement = Runtime()
        replacement.removals[self.linker.layout.encode_key("b", PoolName.KV)] = "b"
        with patch.object(self.linker, "_make_runtime", return_value=replacement):
            self.linker.reset()
        self.assertTrue(self.runtime.closed)
        self.assertEqual(set(self.linker.pop_storage_removals()), {"a", "b"})
        self.assertIs(self.linker.layer_done_counter, counter)
        self.assertEqual(self.linker.num_completed_loads(), 0)
        self.assertEqual(self.linker.start_layer_wise_loading(), -1)
        self.linker.load("b", self.transfer(["b"]))
        self.linker.start_layer_wise_loading()
        eventually(lambda: len(replacement.client.calls) == 3)
        for handle in [1, 2, 3]:
            replacement.client.complete(handle)
        eventually(lambda: self.linker.num_completed_loads() == 1)
        self.assertEqual(self.linker.pop_completed_load(), ["b"])

    def test_request_cleanup_after_close_and_shutdown_rejects_waiting_commands(self):
        handle = CacheRequestHandle("a", 0)
        self.linker.set_request_context(handle, {"actions": []})
        entered, release = threading.Event(), threading.Event()
        original_close = self.runtime.close

        def gated_close():
            entered.set()
            release.wait(5)
            original_close()

        self.runtime.close = gated_close
        errors = []
        close = threading.Thread(target=self.linker.close)
        close.start()
        self.assertTrue(entered.wait(5))

        def query():
            try:
                self.linker.lookup("a", self.transfer(["a"]))
            except RuntimeError as error:
                errors.append(str(error))

        waiter = threading.Thread(target=query)
        waiter.start()
        try:
            eventually(lambda: not self.linker._commands.empty())
        finally:
            release.set()
        close.join(5)
        waiter.join(5)
        self.assertFalse(close.is_alive())
        self.assertFalse(waiter.is_alive())
        self.assertEqual(errors, ["KVCR linker is closed"])
        self.linker.release_request_context(handle)

        with self.assertRaisesRegex(RuntimeError, "KVCR linker is closed"):
            self.linker.start_layer_wise_loading()

    def test_batches_complete_in_admission_order_and_idle_source_keeps_polling(self):
        self.linker.load("a", self.transfer(["a"]))
        first = self.linker.start_layer_wise_loading()
        self.linker.load("b", self.transfer(["b"], 1))
        second = self.linker.start_layer_wise_loading()
        eventually(lambda: len(self.client.calls) == 4)
        for handle in [4, 5, 6]:
            eventually(lambda: len(self.client.calls) >= handle)
            self.client.complete(handle)
        self.linker.layer_done_counter.set_consumer(second)
        self.linker.layer_done_counter.wait_until(2)
        self.assertEqual(self.linker.num_completed_loads(), 0)
        for handle in [3, 1, 2]:
            self.client.complete(handle)
        eventually(lambda: self.linker.num_completed_loads() == 2)
        self.assertEqual(self.linker.pop_completed_load(), ["a"])
        self.assertEqual(self.linker.pop_completed_load(), ["b"])
        self.linker.layer_done_counter.set_consumer(first)
        self.linker.layer_done_counter.wait_until(2)
        polls = self.client.polls
        eventually(lambda: self.client.polls > polls + 2)

    def test_removal_waits_for_its_offload_ack_but_not_unrelated_offloads(self):
        self.linker.offload(self.transfer(["a"]))
        eventually(lambda: len(self.client.calls) == 1)
        self.client.complete(1)
        eventually(lambda: self.linker.num_completed_offloads() == 1)
        key = self.linker.layout.encode_key("a", PoolName.KV)
        other_key = self.linker.layout.encode_key("b", PoolName.KV)
        self.runtime.removals.update({key: "a", other_key: "b"})
        # Another rank has not acknowledged this offload yet.
        self.assertEqual(self.linker.pop_storage_removals(), ["b"])
        self.assertTrue(self.linker.pop_completed_offload())
        self.assertEqual(self.linker.pop_storage_removals(), ["a"])

    def assert_local_hit(self, linker, client):
        key = linker.layout.encode_key("a", PoolName.KV)
        client.status[key] = (
            SimpleNamespace(value="HIT"),
            SimpleNamespace(value="DRAM"),
        )
        self.assertEqual(linker.lookup("a", self.transfer(["a"])), [1])

    def ranked_linker(self, options, *, replicated=True, cp=1, pp=1, nnodes=1):
        from sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker import (
            KVCRDirectLinker,
        )

        params = SimpleNamespace(**vars(self.params))
        params.attn_tp_cache_group = object()
        params.attn_cp_size, params.pp_size = cp, pp
        self.group.rank_replicated = replicated
        runtime = Runtime()
        with (
            get_context().override_server_args(
                tp_size=8,
                nnodes=nnodes,
                hicache_storage_backend_extra_config=json.dumps(options),
            ),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=4),
            patch.object(
                torch.distributed,
                "get_rank",
                side_effect=lambda group=None: (
                    2 if group is params.attn_tp_cache_group else 6
                ),
            ),
        ):
            linker = KVCRDirectLinker(
                None, params, components=set(), runtime_factory=lambda *a, **kw: runtime
            )
        self.addCleanup(linker.close)
        return linker, runtime.client

    def test_replicated_source_rank_routing_is_opt_in(self):
        for enabled, endpoint, expected in (
            (False, "tcp://source:25000", "tcp://source:25000"),
            (True, "tcp://[::1]:25000", "tcp://[::1]:25002"),
            (True, "tcp://source:65535", "tcp://source:65535"),
        ):
            options = {"replicated_source_rank_routing": True} if enabled else {}
            with self.subTest(options=options, endpoint=endpoint):
                linker, client = self.ranked_linker(options)
                hint = {
                    "actions": [
                        {
                            "action_type": action,
                            "payload": {"source_control_endpoint": endpoint},
                        }
                        for action in ("kv.other", "kv.fetch")
                    ]
                }
                linker.set_request_context(CacheRequestHandle("a", 0), hint)
                sent = next(iter(client.hints.values()))
                self.assertEqual(
                    [
                        action["payload"]["source_control_endpoint"]
                        for action in sent["actions"]
                    ],
                    [endpoint, expected],
                )
                self.assertEqual(
                    hint["actions"][1]["payload"]["source_control_endpoint"], endpoint
                )
                linker.close()

    def test_replicated_source_rank_routing_rejects_unsupported_configuration(self):
        for option, topology in (
            ("true", {}),
            (True, {"replicated": False}),
            (True, {"cp": 2}),
            (True, {"pp": 2}),
            (True, {"nnodes": 2}),
        ):
            with (
                self.subTest(option=option, topology=topology),
                self.assertRaises(ValueError),
            ):
                self.ranked_linker(
                    {"replicated_source_rank_routing": option}, **topology
                )

    def test_unaddressed_remote_shard_hints_do_not_admit_failed_deliveries(self):
        from sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker import (
            KVCRDirectLinker,
        )

        self.linker.close()
        self.params.attn_cp_size = 2
        runtime = Runtime()
        linker = KVCRDirectLinker(
            None,
            self.params,
            components=set(),
            runtime_factory=lambda *args, **kwargs: runtime,
        )
        self.addCleanup(linker.close)
        linker.set_request_context(
            CacheRequestHandle("a", 0), {"actions": [{"action_type": "kv.fetch"}]}
        )
        self.assertEqual(runtime.client.hints, {})
        self.assert_local_hit(linker, runtime.client)

    def test_unrelated_hint_actions_preserve_local_lookup(self):
        from sglang.srt.managers.kv_hints import decode_kv_hints_envelope

        hint = decode_kv_hints_envelope(
            {"protocol_version": "0.1", "message_id": "other", "actions": []}
        )
        with patch.object(
            self.client, "submit_hint", side_effect=ValueError("no kv.fetch action")
        ):
            self.linker.set_request_context(CacheRequestHandle("a", 0), hint)
        self.assert_local_hit(self.linker, self.client)


class TestKVCRRuntime(CustomTestCase):
    def setUp(self):
        import sys
        from types import ModuleType

        from sglang.srt.mem_cache.storage.kvcr.runtime import KVCRRuntime

        self.runtime_type = KVCRRuntime
        self.startup_error = type("KVCRStartupError", (RuntimeError,), {})
        self.service_error = type("KVCRServiceError", (RuntimeError,), {})
        self.socket_error = type("KVCRSocketError", (self.service_error,), {})
        self.guard_error = self.socket_error("Guard unavailable")
        self.guard_error.__cause__ = FileNotFoundError()
        self.attempts = []
        self.local_tier = object()
        self.constructor_error = None
        self.close_error = False

        def close():
            if self.close_error:
                raise RuntimeError("DMA is still active")

        def client(config, bindings, backends, guard_config=None):
            self.attempts.append(guard_config)
            if guard_config is not None and self.guard_error is not None:
                raise self.guard_error
            if self.constructor_error is not None:
                raise self.constructor_error
            instance = SimpleNamespace(
                config=config,
                bindings=bindings,
                backends=backends,
                guard_config=guard_config,
                close=close,
                stats=bindings.stats_factory() if config.enable_telemetry else None,
            )

            def get_stats():
                stats = instance.stats
                instance.stats = bindings.stats_factory() if stats is not None else None
                return stats

            instance.get_stats = get_stats
            return instance

        modules = {
            name: ModuleType(name)
            for name in ("kvcr", "kvcr.config", "kvcr.control_channels", "kvcr.types")
        }
        modules["kvcr"].KVCR = client
        modules["kvcr"].KVCRBindings = SimpleNamespace
        modules["kvcr"].KVCRSocketError = self.socket_error
        config_module = modules["kvcr.config"]
        config_module.KVCRConfig = SimpleNamespace
        config_module.KVCRBackendConfigs = SimpleNamespace
        config_module.KVCRGuardConfig = SimpleNamespace
        config_module.LocalDramOptions = lambda pools: SimpleNamespace(pools=pools)
        config_module.RemoteFWDramOptions = SimpleNamespace
        modules["kvcr.control_channels"].ZmqPeerControlChannel = (
            lambda host, port, advertise: SimpleNamespace(
                host=host, port=port, advertise=advertise
            )
        )
        modules["kvcr.types"].KVCRStartupError = self.startup_error
        modules["kvcr.types"].CacheTier = SimpleNamespace(LOCAL_G2=self.local_tier)
        scoped_modules = patch.dict(sys.modules, modules)
        scoped_modules.start()
        self.addCleanup(scoped_modules.stop)
        self.layout = SimpleNamespace(
            endpoint_name="test-rank",
            namespace="model-layout-shard",
            pool_layouts=[("small", 8), ("large", 24)],
            pool_counts={"small": 2, "large": 2},
            trailing_counts=Counter(),
            registrations=[],
            decode_key=lambda key: tuple(key.decode().split(":", 1)),
        )

    def make_runtime(self, **options):
        runtime = self.runtime_type(
            self.layout,
            {"dram_size_gb": 128 / (1 << 30), "advertise_host": "127.0.0.1", **options},
            rank=2,
        )
        self.addCleanup(runtime.close)
        return runtime

    def test_guard_is_default_and_allocates_no_process_dram(self):
        self.guard_error = None
        with patch("mmap.mmap", side_effect=AssertionError("unexpected allocation")):
            runtime = self.make_runtime()
        guard = runtime.client.guard_config
        self.assertEqual(guard.kvcr_service_socket_path, "/run/kvcr/memory.sock")
        self.assertEqual(guard.guard_index, 2)
        self.assertEqual(guard.compatibility_digest, self.layout.namespace)
        self.assertIsNone(runtime.client.backends.local_dram)
        self.assertFalse(runtime._buffers)
        self.assertFalse(runtime.client.config.enable_telemetry)
        self.assertIsNone(runtime.client.bindings.stats_factory)
        with patch.object(runtime.client, "get_stats", side_effect=AssertionError):
            runtime.log_stats(force=True)
        configured = self.make_runtime(
            kvcr_service_socket_path="/tmp/kvcr.sock",
            guard_index=4,
            compatibility_digest="deployment-layout",
        ).client.guard_config
        self.assertEqual(configured.kvcr_service_socket_path, "/tmp/kvcr.sock")
        self.assertEqual(configured.guard_index, 6)
        self.assertEqual(configured.compatibility_digest, "deployment-layout")

    def test_process_dram_fallback_preserves_capacity_and_rank_ports(self):
        for cause in (FileNotFoundError(), ConnectionRefusedError(), None):
            with self.subTest(cause=type(cause).__name__):
                self.guard_error.__cause__ = cause
                self.attempts.clear()
                options = {} if cause else {"kvcr_service_socket_path": None}
                runtime = self.make_runtime(**options)
                self.assertEqual(len(runtime._buffers), 2)
                self.assertEqual(
                    [size for _, _, size in runtime.client.backends.local_dram.pools],
                    [32, 96],
                )
                self.assertEqual(runtime.client.config.nixl_listen_port, 20502)
                self.assertEqual(runtime.client.bindings.framework_control.port, 19502)
                self.assertFalse(
                    runtime.client.backends.remote_fw_dram.opportunistic_query
                )
                self.assertEqual(len(self.attempts), 2 if cause else 1)
                self.assertIsNone(self.attempts[-1])

    def test_trailing_capacity_preserves_minimum_and_total_budget(self):
        self.layout.trailing_counts["large"] = 2
        for divisor, budget, expected in (
            (1, 128, [32, 96]),
            (4, 128, [64, 48]),
            (1024, 64, [16, 48]),
        ):
            with self.subTest(divisor=divisor, budget=budget):
                runtime = self.make_runtime(
                    kvcr_service_socket_path=None,
                    dram_size_gb=budget / (1 << 30),
                    trailing_capacity_divisor=divisor,
                )
                sizes = [n for _, _, n in runtime.client.backends.local_dram.pools]
                self.assertEqual(sizes, expected)
        with self.assertRaisesRegex(ValueError, "complete page"):
            self.make_runtime(dram_size_gb=63 / (1 << 30), trailing_capacity_divisor=4)

    def test_telemetry_logs_labelled_interval_aggregates_and_flushes_on_close(self):
        module = "sglang.srt.mem_cache.storage.kvcr.runtime"
        for guard_error in (None, self.guard_error):
            with self.subTest(guard=guard_error is None):
                self.guard_error = guard_error
                with patch(module + ".time.monotonic", return_value=100) as clock:
                    runtime = self.make_runtime(enable_telemetry=True)
                    stats = runtime.client.stats
                    for duration in (0.1, 0.3):
                        stats.observe_histogram(
                            "kvcr_duration_seconds",
                            duration,
                            ("deposit_submit", "submitted"),
                        )
                    stats.observe_histogram(
                        "kvcr_duration_seconds", 0.2, ("deposit_submit", "failed")
                    )
                    stats.increase_counter(
                        "kvcr_deposit_duplicate_pieces", 2, ("pending",)
                    )
                    stats.increase_counter(
                        "kvcr_deposit_duplicate_pieces", 3, ("pending",)
                    )
                    stats.set_gauge("kvcr_state", 3, ("in_flight_ops",))
                    stats.set_gauge("kvcr_state", 1, ("in_flight_ops",))
                    clock.return_value = 104
                    with self.assertNoLogs(module, level="INFO"):
                        runtime.log_stats()
                    clock.return_value = 105
                    with self.assertLogs(module, level="INFO") as logs:
                        runtime.log_stats()
                    record = logs.records[0].getMessage()
                    self.assertIn("rank=2 interval_s=5.000000", record)
                    values = json.loads(record.split("stats=", 1)[1])
                    prefix = "kvcr_duration_seconds[deposit_submit,submitted]"
                    self.assertEqual(values[prefix + ".count"], 2)
                    self.assertAlmostEqual(values[prefix + ".sum"], 0.4)
                    self.assertEqual(values[prefix + ".max"], 0.3)
                    self.assertEqual(
                        values["kvcr_duration_seconds[deposit_submit,failed].count"], 1
                    )
                    self.assertEqual(
                        values["kvcr_deposit_duplicate_pieces[pending].counter"], 5
                    )
                    self.assertEqual(values["kvcr_state[in_flight_ops].gauge"], 1)
                    runtime.client.stats.increase_counter(
                        "kvcr_source_fast_path", 1, ("hit",)
                    )
                    clock.return_value = 106
                    with self.assertLogs(module, level="INFO") as logs:
                        runtime.close()
                    values = json.loads(
                        logs.records[0].getMessage().split("stats=", 1)[1]
                    )
                    self.assertEqual(values, {"kvcr_source_fast_path[hit].counter": 1})

    def test_refused_or_uncertain_guard_claim_never_falls_back(self):
        for error in (
            self.service_error("incompatible or busy Guard"),
            self.startup_error("native resources retained"),
            PermissionError(),
            TimeoutError(),
            EOFError(),
        ):
            with self.subTest(error=type(error).__name__):
                self.guard_error = error
                if isinstance(error, OSError) or isinstance(error, EOFError):
                    self.guard_error = self.socket_error("claim failed")
                    self.guard_error.__cause__ = error
                with patch("mmap.mmap", side_effect=AssertionError("unsafe fallback")):
                    with self.assertRaises(type(self.guard_error)) as caught:
                        self.make_runtime()
                if isinstance(error, self.startup_error):
                    runtime = caught.exception.kvcr_runtime
                    self.assertFalse(runtime._buffers)
                    with self.assertRaisesRegex(RuntimeError, "startup"):
                        runtime.close()

    def test_declines_pins_and_cancels_removals_per_physical_key(self):
        runtime = self.make_runtime()
        bindings = runtime.client.bindings
        request = bindings.request_pin([b"small:1234"])
        cancelled = bindings.request_pin([b"large:1234"])
        bindings.cancel_pin_request(cancelled)
        self.assertEqual(list(bindings.poll_pin_results()), [(request, None)])
        self.assertEqual(list(bindings.poll_pin_results()), [])
        for keys, removed in (
            ((b"small:1234", b"large:1234"), True),
            ((b"small:1234",), False),
        ):
            bindings.inventory_sink(
                SimpleNamespace(keys=keys, removed=removed, tier=self.local_tier)
            )
        self.assertEqual(list(runtime.removals.values()), ["1234"])
        runtime.removals.clear()
        bindings.inventory_sink(
            SimpleNamespace(keys=(b"large:1234",), removed=False, tier=self.local_tier)
        )
        self.assertFalse(runtime.removals)

    def test_failed_close_retains_mappings_until_success(self):
        runtime = self.make_runtime(enable_telemetry=True)
        buffers = tuple(runtime._buffers)
        self.close_error = True
        try:
            with (
                patch.object(
                    runtime.client,
                    "get_stats",
                    side_effect=RuntimeError("stats failed"),
                ),
                self.assertLogs(
                    "sglang.srt.mem_cache.storage.kvcr.runtime", level="WARNING"
                ),
                self.assertRaisesRegex(RuntimeError, "DMA is still active"),
            ):
                runtime.close()
            self.assertTrue(all(not buffer.closed for buffer in buffers))
        finally:
            self.close_error = False
        runtime.close()
        self.assertTrue(all(buffer.closed for buffer in buffers))

    def test_nonquiescent_startup_attaches_retained_runtime(self):
        self.constructor_error = self.startup_error("native resources retained")
        with self.assertRaises(self.startup_error) as caught:
            self.make_runtime()
        runtime = caught.exception.kvcr_runtime
        self.addCleanup(runtime._close_buffers)
        self.assertTrue(runtime._buffers)
        self.assertTrue(all(not buffer.closed for buffer in runtime._buffers))
        with self.assertRaisesRegex(RuntimeError, "startup"):
            runtime.close()

    def test_invalid_options_fail_before_allocation(self):
        for options in (
            {"unknown_option": 1},
            {"dram_size_gb": 0},
            {"dram_size_gb": float("nan")},
            {"dram_size_gb": 1e-20},
            {"control_port": 65535},
            {"operation_timeout_ms": 0},
            {"abandon_timeout_ms": 1000},
            {"kvcr_service_socket_path": ""},
            {"guard_index": -1},
            {"guard_index": True},
            {"compatibility_digest": ""},
            {"enable_telemetry": "true"},
            {"trailing_capacity_divisor": 0},
            {"trailing_capacity_divisor": True},
            {"trailing_capacity_divisor": 1.5},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.runtime_type(self.layout, options, rank=2)


if __name__ == "__main__":
    unittest.main()
