import asyncio
import dataclasses
import unittest
from concurrent.futures import Future
from types import MethodType, SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import msgspec
import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVSender,
)
from sglang.srt.disaggregation.p2p_kv_transfer import (
    P2PTransferState,
    PendingP2PTransfer,
    PrefillP2PMooncakeTransferEngine,
    _p2p_req_to_builtins,
    _P2PCacheIntegrityError,
    _TargetAllocation,
)
from sglang.srt.managers.io_struct import P2PKVTransferReqInput, P2PKVTransferReqOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_control_mixin import (
    P2P_PAIR_GATE_ACQUIRE_REASON,
    P2P_PAIR_GATE_RELEASE_REASON,
    TokenizerControlMixin,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeAllocator:
    def __init__(self, start=100, available=1_000_000):
        self.start = start
        self._available = available
        self.allocations = []
        self.freed = []
        self.live = set()

    def available_size(self):
        return self._available

    def alloc(self, n):
        self.allocations.append(n)
        if self._available < n:
            return None
        self._available -= n
        value = torch.arange(self.start, self.start + n, dtype=torch.int32)
        indices = {int(index) for index in value.tolist()}
        if self.live.intersection(indices):
            raise AssertionError("fake allocator reused live indices")
        self.live.update(indices)
        return value

    def free(self, value):
        indices = [int(index) for index in value.detach().cpu().tolist()]
        duplicate = [index for index in indices if index not in self.live]
        if duplicate:
            raise AssertionError(f"double free: {duplicate}")
        self.freed.append(indices)
        self.live.difference_update(indices)
        self._available += len(indices)

    def reclaim(self, n):
        self._available += n


class _FakeMambaAllocator(_FakeAllocator):
    def __init__(self, start=10_000, available=128):
        super().__init__(start=start, available=available)


class _FakeTreeCache:
    page_size = 4
    mamba_cache_chunk_size = 1

    def __init__(
        self,
        matched_indices=None,
        last_device_node=None,
        insert_prefix_len=0,
        mamba_exist=False,
        duplicate_kv_handled_by_cache=False,
        on_insert=None,
    ):
        self.inserted = []
        self.matched_indices = matched_indices
        self.last_device_node = last_device_node or SimpleNamespace()
        self.locked = []
        self.unlocked = []
        self.evicted = []
        self.token_to_kv_pool_allocator = None
        self.insert_prefix_len = insert_prefix_len
        self.mamba_exist = mamba_exist
        self.duplicate_kv_handled_by_cache = duplicate_kv_handled_by_cache
        self.on_insert = on_insert

    def insert(self, params):
        self.inserted.append(params)
        if self.on_insert is not None:
            self.on_insert(params)
        return SimpleNamespace(
            prefix_len=self.insert_prefix_len,
            mamba_exist=self.mamba_exist,
            duplicate_kv_handled_by_cache=self.duplicate_kv_handled_by_cache,
        )

    def match_prefix(self, params):
        if self.matched_indices is not None:
            device_indices = self.matched_indices
        elif self.inserted:
            device_indices = self.inserted[-1].value
        else:
            device_indices = torch.empty((0,), dtype=torch.int32)
        return SimpleNamespace(
            device_indices=device_indices,
            last_device_node=self.last_device_node,
        )

    def inc_lock_ref(self, node):
        self.locked.append(node)
        return SimpleNamespace(to_dec_params=lambda: SimpleNamespace())

    def dec_lock_ref(self, node, params):
        self.unlocked.append((node, params))

    def available_and_evictable_str(self):
        return "fake capacity"

    def is_chunk_cache(self):
        return False

    def evict(self, params):
        self.evicted.append(params)
        self.token_to_kv_pool_allocator.reclaim(params.num_tokens)


class _FakeKVManager:
    def __init__(self, state_types=None):
        self.kv_args = SimpleNamespace(
            engine_rank=0,
            page_size=4,
            state_types=state_types or [],
            kv_data_ptrs=[],
            kv_item_lens=[],
            state_data_ptrs=[],
            state_item_lens=[],
        )
        self.attn_tp_size = 1
        self.attn_tp_rank = 0
        self.attn_cp_size = 1
        self.attn_cp_rank = 0
        self.pp_size = 1
        self.pp_rank = 0

    def get_session_id(self):
        return "target-session"


class _FakeSender:
    instances: ClassVar[list["_FakeSender"]] = []

    def __init__(
        self,
        mgr,
        bootstrap_addr,
        bootstrap_room,
        dest_tp_ranks,
        pp_rank,
        force_cp_rank_transfer=False,
    ):
        self.polls = [KVPoll.WaitingForInput, KVPoll.Success]
        self.sent = []
        self.force_cp_rank_transfer = force_cp_rank_transfer
        _FakeSender.instances.append(self)

    def init(self, num_pages, aux_index=None):
        self.num_pages = num_pages
        self.aux_index = aux_index

    def poll(self):
        return self.polls.pop(0)

    def send(self, kv_indices, state_indices=None):
        self.sent.append((kv_indices, state_indices))


class TestP2PKVTransferEngine(unittest.TestCase):
    def _req(self):
        return P2PKVTransferReqInput(
            source_url="http://127.0.0.1:30000",
            target_url="http://127.0.0.1:30001",
            token_ids=list(range(8)),
            matched_tokens=8,
            request_id="rid-1",
            reason="load_imbalance",
        )

    def _reverse_req(self):
        return P2PKVTransferReqInput(
            source_url="http://127.0.0.1:30001",
            target_url="http://127.0.0.1:30000",
            token_ids=list(range(8)),
            matched_tokens=8,
            request_id="rid-reverse",
            reason="load_imbalance",
        )

    def test_capacity_detail_sampling_logs_first_and_every_sixty_fourth(self):
        engine = PrefillP2PMooncakeTransferEngine(
            self._scheduler(_FakeKVManager(state_types=[]))
        )

        decisions = [engine._sample_capacity_detail() for _ in range(129)]

        self.assertEqual(
            [index + 1 for index, sampled in enumerate(decisions) if sampled],
            [1, 64, 128],
        )

    def test_source_transfer_progresses_without_blocking_scheduler(self):
        _FakeSender.instances = []
        kv_manager = _FakeKVManager(state_types=[])
        tree_cache = _FakeTreeCache(
            matched_indices=torch.arange(10, 18, dtype=torch.int32),
            last_device_node=SimpleNamespace(),
        )
        engine = PrefillP2PMooncakeTransferEngine(
            self._scheduler(kv_manager, tree_cache=tree_cache)
        )
        req = self._req()
        req.p2p_bootstrap_room = 123
        req.p2p_source_send = True

        with (
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer.get_kv_class",
                return_value=_FakeSender,
            ),
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer.time.sleep",
                side_effect=AssertionError("scheduler progress must never sleep"),
            ),
            patch.object(
                engine,
                "_progress_world_min",
                wraps=engine._progress_world_min,
            ) as progress_world_min,
        ):
            self.assertIsNone(engine.start_transfer(req))
            self.assertEqual(engine.progress_transfers(), [])
            self.assertEqual(
                progress_world_min.call_args.args[1], "source-hicache-load"
            )
            completions = []
            for _ in range(8):
                completions = engine.progress_transfers()
                if completions:
                    break

        self.assertEqual(len(completions), 1)
        completed_req, output = completions[0]
        self.assertIs(completed_req, req)
        self.assertTrue(output.success)
        self.assertEqual(output.transferred_tokens, 8)
        sent_indices = _FakeSender.instances[0].sent[0][0]
        self.assertIsInstance(sent_indices, np.ndarray)
        self.assertEqual(sent_indices.dtype, np.int32)
        self.assertTrue(sent_indices.flags.c_contiguous)

    def test_source_hicache_load_back_progresses_without_blocking_scheduler(self):
        class FinishEvent:
            def __init__(self):
                self.ready = False

            def query(self):
                return self.ready

            def synchronize(self):
                raise AssertionError("scheduler progress must never synchronize")

        class HiCacheTree(_FakeTreeCache):
            def __init__(self):
                self.cache_node = SimpleNamespace()
                super().__init__(last_device_node=self.cache_node)
                self.finish_event = FinishEvent()
                self.load_back_calls = []
                self.loading_checks = 0
                self.cache_controller = SimpleNamespace(
                    layer_done_counter=SimpleNamespace(
                        events=[
                            SimpleNamespace(finish_event=self.finish_event),
                        ]
                    )
                )

            def match_prefix(self, params):
                restored = self.finish_event.ready
                return SimpleNamespace(
                    device_indices=(
                        torch.arange(10, 18, dtype=torch.int32)
                        if restored
                        else torch.empty((0,), dtype=torch.int32)
                    ),
                    last_device_node=self.cache_node,
                    last_host_node=self.cache_node,
                    best_match_node=self.cache_node,
                    host_hit_length=0 if restored else 8,
                    mamba_host_hit_length=0,
                )

            def load_back(self, node, mem_quota=None):
                self.load_back_calls.append((node, mem_quota))
                return True

            def ready_to_load_host_cache(self):
                return 0

            def loading_check(self):
                self.loading_checks += 1

        _FakeSender.instances = []
        kv_manager = _FakeKVManager(state_types=[])
        tree_cache = HiCacheTree()
        engine = PrefillP2PMooncakeTransferEngine(
            self._scheduler(kv_manager, tree_cache=tree_cache)
        )
        req = self._req()
        req.p2p_bootstrap_room = 123
        req.p2p_source_send = True

        with patch(
            "sglang.srt.disaggregation.p2p_kv_transfer.get_kv_class",
            return_value=_FakeSender,
        ):
            self.assertIsNone(engine.start_transfer(req))
            self.assertEqual(engine.progress_transfers(), [])
            self.assertEqual(_FakeSender.instances, [])

            tree_cache.finish_event.ready = True
            for _ in range(4):
                self.assertEqual(engine.progress_transfers(), [])
                if _FakeSender.instances:
                    break
            self.assertEqual(len(_FakeSender.instances), 1)
            completions = []
            for _ in range(6):
                completions = engine.progress_transfers()
                if completions:
                    break

        self.assertEqual(len(completions), 1)
        _, output = completions[0]
        self.assertTrue(output.success)
        self.assertEqual(output.transferred_tokens, 8)
        self.assertEqual(tree_cache.load_back_calls, [(tree_cache.cache_node, None)])
        self.assertEqual(tree_cache.loading_checks, 0)

    def test_source_hicache_load_back_timeout_falls_back_and_clears_pending(self):
        class FinishEvent:
            def query(self):
                return False

        class HiCacheTree(_FakeTreeCache):
            def __init__(self):
                self.cache_node = SimpleNamespace()
                super().__init__(last_device_node=self.cache_node)
                self.finish_event = FinishEvent()
                self.cache_controller = SimpleNamespace(
                    layer_done_counter=SimpleNamespace(
                        events=[
                            SimpleNamespace(finish_event=self.finish_event),
                        ]
                    )
                )

            def match_prefix(self, params):
                return SimpleNamespace(
                    device_indices=torch.empty((0,), dtype=torch.int32),
                    last_device_node=self.cache_node,
                    last_host_node=self.cache_node,
                    best_match_node=self.cache_node,
                    host_hit_length=8,
                    mamba_host_hit_length=0,
                )

            def load_back(self, node, mem_quota=None):
                return True

            def ready_to_load_host_cache(self):
                return 0

            def loading_check(self):
                raise AssertionError("an unfinished load must not be finalized")

        _FakeSender.instances = []
        engine = PrefillP2PMooncakeTransferEngine(
            self._scheduler(_FakeKVManager(state_types=[]), tree_cache=HiCacheTree())
        )
        engine._TRANSFER_TIMEOUT_S = -1
        req = self._req()
        req.p2p_bootstrap_room = 123
        req.p2p_source_send = True

        with (
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer.get_kv_class",
                return_value=_FakeSender,
            ),
            patch.object(
                engine,
                "_progress_world_min",
                wraps=engine._progress_world_min,
            ) as progress_world_min,
        ):
            self.assertIsNone(engine.start_transfer(req))
            completions = engine.progress_transfers()

        self.assertEqual(len(completions), 1)
        _, output = completions[0]
        self.assertFalse(output.success)
        self.assertTrue(output.fallback_recompute)
        self.assertIn("load-back timed out", output.message)
        self.assertFalse(engine.has_pending_transfers())
        self.assertEqual(_FakeSender.instances, [])
        progress_world_min.assert_called_once()
        self.assertEqual(
            progress_world_min.call_args.args[1:], ("source-hicache-load", 0)
        )

    def test_source_sender_init_failure_reaches_consensus_and_cleans_up(self):
        class BrokenSender:
            instances = []

            def __init__(self, **kwargs):
                self.aborted = False
                self.instances.append(self)

            def init(self, num_pages, aux_index=None):
                raise RuntimeError("sender init failed")

            def abort(self):
                self.aborted = True

        tree_cache = _FakeTreeCache(
            matched_indices=torch.arange(10, 18, dtype=torch.int32),
            last_device_node=SimpleNamespace(),
        )
        engine = PrefillP2PMooncakeTransferEngine(
            self._scheduler(_FakeKVManager(state_types=[]), tree_cache=tree_cache)
        )
        req = self._req()
        req.p2p_bootstrap_room = 123
        req.p2p_source_send = True

        with patch(
            "sglang.srt.disaggregation.p2p_kv_transfer.get_kv_class",
            return_value=BrokenSender,
        ):
            self.assertIsNone(engine.start_transfer(req))
            completions = []
            for _ in range(8):
                completions = engine.progress_transfers()
                if completions:
                    break

        self.assertEqual(len(completions), 1)
        _, output = completions[0]
        self.assertFalse(output.success)
        self.assertTrue(output.fallback_recompute)
        self.assertIn("sender init failed", output.message)
        self.assertFalse(engine.has_pending_transfers())
        self.assertEqual(len(BrokenSender.instances), 1)
        self.assertTrue(BrokenSender.instances[0].aborted)

    def test_target_transfer_progresses_without_blocking_scheduler(self):
        class ImmediateExecutor:
            def submit(self, fn, *args, **kwargs):
                future = Future()
                try:
                    future.set_result(fn(*args, **kwargs))
                except Exception as exc:
                    future.set_exception(exc)
                return future

        class Receiver:
            def __init__(self, *args, **kwargs):
                self.polls = [
                    KVPoll.WaitingForInput,
                    KVPoll.WaitingForInput,
                    KVPoll.Transferring,
                    KVPoll.Success,
                ]

            def init(self, prefill_dp_rank):
                pass

            def poll(self):
                return self.polls.pop(0)

            def send_metadata(self, *args, **kwargs):
                pass

        kv_manager = _FakeKVManager(state_types=[])
        kv_manager.try_ensure_parallel_info = lambda addr, **kwargs: True
        kv_manager.prefill_info_table = {
            "127.0.0.1:32400": SimpleNamespace(
                attn_tp_size=1,
                attn_cp_size=1,
                pp_size=1,
            )
        }
        scheduler = self._scheduler(kv_manager)
        engine = PrefillP2PMooncakeTransferEngine(
            scheduler, http_executor=ImmediateExecutor()
        )
        req = self._req()
        req.p2p_bootstrap_room = 123
        req.source_bootstrap_addr = "127.0.0.1:32400"

        response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"success": True, "transferred_tokens": 8},
        )
        with (
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer.get_kv_class",
                return_value=Receiver,
            ),
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer.requests.post",
                return_value=response,
            ),
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer.time.sleep",
                side_effect=AssertionError("scheduler progress must never sleep"),
            ),
        ):
            self.assertIsNone(engine.start_transfer(req))
            self.assertEqual(engine.progress_transfers(), [])
            completions = engine.progress_transfers()

        self.assertEqual(len(completions), 1)
        completed_req, output = completions[0]
        self.assertIs(completed_req, req)
        self.assertTrue(output.success)
        self.assertEqual(output.transferred_tokens, 8)
        self.assertEqual(len(scheduler.tree_cache.inserted), 1)

    def test_scheduler_defers_p2p_reply_until_transfer_progress_completes(self):
        req = self._req()
        output = P2PKVTransferReqOutput(
            success=True,
            message="done",
            source_url=req.source_url,
            target_url=req.target_url,
            matched_tokens=req.matched_tokens,
            transferred_tokens=req.matched_tokens,
            fallback_recompute=False,
            experimental_limitations=[],
        )
        transfer_engine = SimpleNamespace(
            start_transfer=MagicMock(return_value=None),
            progress_transfers=MagicMock(return_value=[(req, output)]),
        )
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(enable_prefill_p2p_kv_transfer=True),
            p2p_kv_transfer_engine=transfer_engine,
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
            ),
        )

        immediate = Scheduler.handle_p2p_kv_transfer(scheduler, req)
        Scheduler.progress_p2p_kv_transfers(scheduler)

        self.assertIsNone(immediate)
        transfer_engine.start_transfer.assert_called_once_with(req)
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once_with(
            output, req
        )

    def test_async_progress_preserves_cache_integrity_fail_stop(self):
        scheduler = self._scheduler(_FakeKVManager(state_types=[]))
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        req = self._req()
        engine._pending_transfers[req.request_id] = PendingP2PTransfer(
            req=req,
            role="target",
            state=P2PTransferState.COMMIT,
            deadline=10.0,
            kv_manager=SimpleNamespace(),
        )

        with (
            patch.object(
                engine,
                "_progress_target_transfer",
                side_effect=_P2PCacheIntegrityError("insert exploded"),
            ),
            self.assertRaisesRegex(_P2PCacheIntegrityError, "insert exploded"),
        ):
            engine.progress_transfers()

    def test_receiver_poll_uses_tp_cp_consensus(self):
        scheduler = self._scheduler(_FakeKVManager(state_types=[]))
        scheduler.attn_tp_cpu_group = "tp-group"
        scheduler.attn_cp_cpu_group = "cp-group"
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        receiver = SimpleNamespace(poll=lambda: KVPoll.Success)

        with patch(
            "sglang.srt.disaggregation.p2p_kv_transfer."
            "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.Success],
        ) as consensus:
            poll = engine._poll_receiver_consensus(receiver)

        self.assertEqual(poll, KVPoll.Success)
        consensus.assert_called_once_with(
            [receiver], scheduler.attn_cp_cpu_group, scheduler.attn_tp_cpu_group
        )

    def test_transferred_tokens_use_minimum_across_tp_cp_ranks(self):
        scheduler = self._scheduler(_FakeKVManager(state_types=[]))
        scheduler.attn_tp_cpu_group = "tp-group"
        scheduler.attn_cp_cpu_group = "cp-group"
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        groups = []

        def all_reduce(value, op, group):
            groups.append(group)
            if group == scheduler.attn_tp_cpu_group:
                value.fill_(4)

        with patch(
            "sglang.srt.disaggregation.p2p_kv_transfer.torch.distributed.all_reduce",
            side_effect=all_reduce,
        ):
            transferred_tokens = engine._min_target_consensus(8)

        self.assertEqual(transferred_tokens, 4)
        self.assertEqual(
            groups, [scheduler.attn_tp_cpu_group, scheduler.attn_cp_cpu_group]
        )

    def test_p2p_identical_layout_maps_same_tp_pp_cp_tuple(self):
        manager = object.__new__(CommonKVManager)
        manager.attn_tp_size = 2
        manager.attn_tp_rank = 1
        manager.attn_cp_size = 2
        manager.attn_cp_rank = 1
        manager.pp_size = 2
        manager.pp_rank = 1
        manager.kv_args = SimpleNamespace(engine_rank=7)
        manager.is_mla_backend = False
        manager.enable_all_cp_ranks_for_transfer = False
        manager.p2p_layout_fingerprint = "same-layout"
        info = SimpleNamespace(
            attn_tp_size=2,
            attn_cp_size=2,
            pp_size=2,
            p2p_layout_fingerprint="same-layout",
        )

        manager._resolve_rank_mapping(info, p2p_identical_layout=True)

        self.assertEqual(info.target_tp_rank, 1)
        self.assertEqual(info.target_tp_ranks, [1])
        self.assertEqual(info.target_cp_ranks, [1])
        self.assertEqual(info.target_pp_ranks, [1])
        self.assertEqual(info.required_dst_info_num, 1)
        self.assertEqual(info.required_prefill_response_num, 1)

    def test_p2p_identical_layout_rejects_mismatched_fingerprint(self):
        manager = object.__new__(CommonKVManager)
        manager.attn_tp_size = 1
        manager.attn_tp_rank = 0
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.p2p_layout_fingerprint = "target-layout"
        info = SimpleNamespace(
            attn_tp_size=1,
            attn_cp_size=1,
            pp_size=1,
            p2p_layout_fingerprint="source-layout",
        )

        with self.assertRaisesRegex(AssertionError, "identical model and KV layouts"):
            manager._resolve_rank_mapping(info, p2p_identical_layout=True)

    def test_bootstrap_parallel_info_hides_p2p_metadata_from_legacy_decode(self):
        server = object.__new__(CommonKVBootstrapServer)
        server.attn_tp_size = 2
        server.attn_cp_size = 1
        server.dp_size = 1
        server.pp_size = 2
        server.page_size = 64
        server.kv_cache_dtype = "fp8_e4m3"
        server.p2p_layout_fingerprint = "layout-v1"
        server.follow_bootstrap_room = True
        server.enable_dsa_cache_layer_split = False
        server.prefill_http_port = 30000

        legacy_payload = server._parallel_info_payload(include_p2p_metadata=False)
        p2p_payload = server._parallel_info_payload(include_p2p_metadata=True)

        self.assertNotIn("p2p_layout_fingerprint", legacy_payload)
        self.assertEqual(p2p_payload["p2p_layout_fingerprint"], "layout-v1")

    def test_p2p_parallel_info_query_explicitly_requests_layout_metadata(self):
        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=64)
        manager._resolve_rank_mapping = MagicMock()
        response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "attn_tp_size": 1,
                "attn_cp_size": 1,
                "dp_size": 1,
                "pp_size": 1,
                "page_size": 64,
                "kv_cache_dtype": "fp8_e4m3",
                "follow_bootstrap_room": True,
                "p2p_layout_fingerprint": "layout-v1",
            },
        )

        with (
            patch(
                "sglang.srt.disaggregation.common.conn.requests.get",
                return_value=response,
            ) as get,
            patch(
                "sglang.srt.disaggregation.common.conn.get_model",
                return_value=SimpleNamespace(kv_cache_dtype="fp8_e4m3"),
            ),
        ):
            self.assertTrue(
                manager.try_ensure_parallel_info(
                    "127.0.0.1:8998", p2p_identical_layout=True
                )
            )

        self.assertIn("include_p2p_metadata=1", get.call_args.args[0])

    def test_p2p_parallel_info_refetches_legacy_cached_entry(self):
        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {
            "127.0.0.1:8998": SimpleNamespace(p2p_layout_fingerprint=None)
        }
        manager.kv_args = SimpleNamespace(page_size=64)
        manager._resolve_rank_mapping = MagicMock()
        response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "attn_tp_size": 1,
                "attn_cp_size": 1,
                "dp_size": 1,
                "pp_size": 1,
                "page_size": 64,
                "kv_cache_dtype": "fp8_e4m3",
                "follow_bootstrap_room": True,
                "p2p_layout_fingerprint": "layout-v1",
            },
        )

        with (
            patch(
                "sglang.srt.disaggregation.common.conn.requests.get",
                return_value=response,
            ) as get,
            patch(
                "sglang.srt.disaggregation.common.conn.get_model",
                return_value=SimpleNamespace(kv_cache_dtype="fp8_e4m3"),
            ),
        ):
            self.assertTrue(
                manager.try_ensure_parallel_info(
                    "127.0.0.1:8998", p2p_identical_layout=True
                )
            )

        get.assert_called_once()
        self.assertEqual(
            manager.prefill_info_table["127.0.0.1:8998"].p2p_layout_fingerprint,
            "layout-v1",
        )

    def test_p2p_sender_forces_nonzero_cp_rank_without_changing_pd_default(self):
        class Manager:
            is_dummy_cp_rank = True
            enable_all_cp_ranks_for_transfer = True
            server_args = SimpleNamespace(dp_size=1)

            def __init__(self):
                self.statuses = []

            def update_status(self, room, status):
                self.statuses.append((room, status))

        ordinary_manager = Manager()
        CommonKVSender(ordinary_manager, "", 7, [0], 0)
        self.assertEqual(ordinary_manager.statuses[-1], (7, KVPoll.WaitingForInput))

        p2p_manager = Manager()
        sender = CommonKVSender(
            p2p_manager,
            "",
            8,
            [0],
            0,
            force_cp_rank_transfer=True,
        )
        sender.init(2)
        indices, index_slice, is_last, should_skip = sender._prepare_send_indices(
            torch.tensor([11, 12], dtype=torch.int32).numpy()
        )

        self.assertEqual(p2p_manager.statuses[-1], (8, KVPoll.Bootstrapping))
        self.assertEqual(indices.tolist(), [11, 12])
        self.assertEqual(index_slice, slice(0, 2))
        self.assertTrue(is_last)
        self.assertFalse(should_skip)

    def test_target_metadata_reaches_consensus_before_source_trigger(self):
        events = []

        class Receiver:
            def __init__(self, *args, **kwargs):
                pass

            def init(self, prefill_dp_rank):
                pass

            def poll(self):
                raise AssertionError("receiver polling must use consensus")

            def send_metadata(self, *args, **kwargs):
                events.append("metadata")

        kv_manager = _FakeKVManager(state_types=[])
        kv_manager.try_ensure_parallel_info = lambda addr, **kwargs: True
        kv_manager.prefill_info_table = {"127.0.0.1:32400": SimpleNamespace()}
        scheduler = self._scheduler(kv_manager)
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        engine._source_bootstrap_addr = MethodType(
            lambda self, source_url: "127.0.0.1:32400", engine
        )
        engine._validate_identical_layout = MethodType(
            lambda self, manager, info: None, engine
        )
        polls = iter([KVPoll.WaitingForInput, KVPoll.WaitingForInput, KVPoll.Success])

        def consensus(self, receiver, force_failed=False):
            events.append("consensus")
            return next(polls)

        engine._poll_receiver_consensus = MethodType(consensus, engine)
        engine._min_target_consensus = MethodType(lambda self, value: value, engine)
        engine._world_phase_consensus = MethodType(
            lambda self, local_success, phase: (
                events.append(f"world:{phase}") or local_success
            ),
            engine,
        )

        def post(*args, **kwargs):
            events.append("http")
            return SimpleNamespace(
                status_code=200,
                json=lambda: {"success": True, "transferred_tokens": 8},
            )

        with (
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer.get_kv_class",
                return_value=Receiver,
            ),
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer.requests.post",
                side_effect=post,
            ),
            patch("sglang.srt.disaggregation.p2p_kv_transfer.time.sleep"),
        ):
            result = engine._target_pull_via_receiver(
                self._req(), kv_manager, torch.arange(8), None, 8
            )

        self.assertTrue(result.success)
        consensus_events = [i for i, event in enumerate(events) if event == "consensus"]
        self.assertLess(consensus_events[1], events.index("http"))
        self.assertLess(events.index("world:target metadata"), events.index("http"))

    def test_p2p_transfer_io_struct_serialization_supports_base_runtime_type(self):
        req = self._req()
        output = P2PKVTransferReqOutput(success=True)

        self.assertTrue(
            dataclasses.is_dataclass(req) or isinstance(req, msgspec.Struct)
        )
        self.assertTrue(
            dataclasses.is_dataclass(output) or isinstance(output, msgspec.Struct)
        )
        self.assertEqual(_p2p_req_to_builtins(req)["source_url"], req.source_url)
        if dataclasses.is_dataclass(output):
            serialized_output = dataclasses.asdict(output)
        else:
            serialized_output = msgspec_to_builtins(output)
        self.assertEqual(serialized_output["experimental_limitations"], [])

    def _scheduler(self, kv_manager, tree_cache=None, allocator=None):
        tree_cache = tree_cache or _FakeTreeCache()
        allocator = allocator or _FakeAllocator()
        tree_cache.token_to_kv_pool_allocator = allocator
        return SimpleNamespace(
            disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=kv_manager),
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=SimpleNamespace(),
            tree_cache=tree_cache,
        )

    def _successful_receiver(self, transferred_tokens=None):
        def receiver(engine, req, kv_manager, dst_kv, dst_mamba, prefix_len):
            return P2PKVTransferReqOutput(
                success=True,
                message="ok",
                source_url=req.source_url,
                target_url=req.target_url,
                matched_tokens=req.matched_tokens,
                transferred_tokens=(
                    prefix_len if transferred_tokens is None else transferred_tokens
                ),
                fallback_recompute=False,
                experimental_limitations=engine._limitations(),
            )

        return receiver

    def test_mamba_cache_owned_duplicate_kv_is_not_freed_twice(self):
        kv_allocator = _FakeAllocator()
        mamba_allocator = _FakeMambaAllocator()
        tree_cache = _FakeTreeCache(
            insert_prefix_len=8,
            mamba_exist=True,
            duplicate_kv_handled_by_cache=True,
            on_insert=lambda params: kv_allocator.free(params.value),
        )
        scheduler = self._scheduler(
            _FakeKVManager(state_types=[StateType.MAMBA]),
            tree_cache=tree_cache,
            allocator=kv_allocator,
        )
        scheduler.req_to_token_pool.mamba_allocator = mamba_allocator
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        engine._target_pull_via_receiver = MethodType(
            self._successful_receiver(), engine
        )

        kv_before = kv_allocator.available_size()
        mamba_before = mamba_allocator.available_size()
        ret = engine.transfer(self._req())

        self.assertTrue(ret.success)
        self.assertEqual(kv_allocator.available_size(), kv_before)
        self.assertEqual(mamba_allocator.available_size(), mamba_before)
        self.assertEqual(kv_allocator.freed, [list(range(100, 108))])
        self.assertEqual(mamba_allocator.freed, [[10_000]])

    def test_control_plane_resolves_source_bootstrap_once_before_fanout(self):
        class Manager(TokenizerControlMixin):
            def auto_create_handle_loop(self):
                pass

        manager = Manager()
        manager.p2p_kv_transfer_communicator = AsyncMock(
            return_value=[
                P2PKVTransferReqOutput(
                    success=True,
                    transferred_tokens=8,
                    fallback_recompute=False,
                )
            ]
        )
        req = self._req()
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"disaggregation_bootstrap_port": 32400},
        )
        gate_ok = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"success": True, "message": "ok"},
        )

        with (
            patch(
                "sglang.srt.managers.tokenizer_control_mixin.requests.get",
                return_value=response,
            ) as get,
            patch(
                "sglang.srt.managers.tokenizer_control_mixin.requests.post",
                return_value=gate_ok,
            ),
        ):
            ret = asyncio.run(manager.p2p_kv_transfer(req))

        self.assertTrue(ret.success)
        get.assert_called_once_with("http://127.0.0.1:30000/server_info", timeout=5)
        dispatched = manager.p2p_kv_transfer_communicator.await_args.args[0]
        self.assertEqual(dispatched.source_bootstrap_addr, "127.0.0.1:32400")

    def test_control_plane_pair_gate_serializes_opposite_directions_by_owner(self):
        class Manager(TokenizerControlMixin):
            def auto_create_handle_loop(self):
                pass

        manager = Manager()
        manager.p2p_kv_transfer_communicator = AsyncMock(
            side_effect=AssertionError("pair-gate controls must not reach schedulers")
        )
        first = self._req()
        first.dry_run = True
        first.reason = P2P_PAIR_GATE_ACQUIRE_REASON
        reverse = self._reverse_req()
        reverse.dry_run = True
        reverse.reason = P2P_PAIR_GATE_ACQUIRE_REASON

        first_acquire = asyncio.run(manager.p2p_kv_transfer(first))
        reverse_while_busy = asyncio.run(manager.p2p_kv_transfer(reverse))

        self.assertTrue(first_acquire.success)
        self.assertFalse(reverse_while_busy.success)
        self.assertIn("busy", reverse_while_busy.message)

        first.reason = P2P_PAIR_GATE_RELEASE_REASON
        release = asyncio.run(manager.p2p_kv_transfer(first))
        reverse_after_release = asyncio.run(manager.p2p_kv_transfer(reverse))

        self.assertTrue(release.success)
        self.assertTrue(reverse_after_release.success)
        manager.p2p_kv_transfer_communicator.assert_not_awaited()

    def test_control_plane_remote_pair_gate_wraps_target_scheduler_dispatch(self):
        class Manager(TokenizerControlMixin):
            def auto_create_handle_loop(self):
                pass

        manager = Manager()
        manager.p2p_kv_transfer_communicator = AsyncMock(
            return_value=[
                P2PKVTransferReqOutput(
                    success=True,
                    transferred_tokens=8,
                    fallback_recompute=False,
                )
            ]
        )
        req = self._req()
        bootstrap = SimpleNamespace(
            status_code=200,
            json=lambda: {"disaggregation_bootstrap_port": 32400},
        )
        gate_ok = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"success": True, "message": "ok"},
        )

        with (
            patch(
                "sglang.srt.managers.tokenizer_control_mixin.requests.get",
                return_value=bootstrap,
            ),
            patch(
                "sglang.srt.managers.tokenizer_control_mixin.requests.post",
                return_value=gate_ok,
            ) as post,
        ):
            ret = asyncio.run(manager.p2p_kv_transfer(req))

        self.assertTrue(ret.success)
        self.assertEqual(post.call_count, 2)
        acquire = post.call_args_list[0]
        release = post.call_args_list[1]
        self.assertEqual(
            acquire.args[0],
            "http://127.0.0.1:30000/experimental/p2p_kv_transfer",
        )
        self.assertEqual(acquire.kwargs["json"]["reason"], P2P_PAIR_GATE_ACQUIRE_REASON)
        self.assertEqual(release.kwargs["json"]["reason"], P2P_PAIR_GATE_RELEASE_REASON)

    def test_control_plane_remote_pair_gate_busy_falls_back_without_dispatch(self):
        class Manager(TokenizerControlMixin):
            def auto_create_handle_loop(self):
                pass

        manager = Manager()
        manager.p2p_kv_transfer_communicator = AsyncMock()
        req = self._req()
        gate_busy = SimpleNamespace(
            status_code=409,
            text="busy",
            json=lambda: {"success": False, "message": "pair gate busy"},
        )

        with patch(
            "sglang.srt.managers.tokenizer_control_mixin.requests.post",
            return_value=gate_busy,
        ):
            ret = asyncio.run(manager.p2p_kv_transfer(req))

        self.assertFalse(ret.success)
        self.assertTrue(ret.fallback_recompute)
        self.assertIn("busy", ret.message)
        manager.p2p_kv_transfer_communicator.assert_not_awaited()

    def test_control_plane_uses_valid_prefetched_source_bootstrap_without_http(self):
        class Manager(TokenizerControlMixin):
            def auto_create_handle_loop(self):
                pass

        manager = Manager()
        manager.p2p_kv_transfer_communicator = AsyncMock(
            return_value=[P2PKVTransferReqOutput(success=True)]
        )
        req = self._req()
        req.source_bootstrap_addr = "127.0.0.1:32400"
        gate_ok = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"success": True, "message": "ok"},
        )

        with (
            patch(
                "sglang.srt.managers.tokenizer_control_mixin.requests.get",
                side_effect=AssertionError("prefetched address must skip discovery"),
            ),
            patch(
                "sglang.srt.managers.tokenizer_control_mixin.requests.post",
                return_value=gate_ok,
            ),
        ):
            ret = asyncio.run(manager.p2p_kv_transfer(req))

        self.assertTrue(ret.success)
        dispatched = manager.p2p_kv_transfer_communicator.await_args.args[0]
        self.assertEqual(dispatched.source_bootstrap_addr, "127.0.0.1:32400")

    def test_control_plane_discards_prefetched_bootstrap_for_wrong_source_host(self):
        class Manager(TokenizerControlMixin):
            def auto_create_handle_loop(self):
                pass

        manager = Manager()
        manager.p2p_kv_transfer_communicator = AsyncMock(
            return_value=[P2PKVTransferReqOutput(success=True)]
        )
        req = self._req()
        req.source_bootstrap_addr = "127.0.0.2:32400"
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"disaggregation_bootstrap_port": 32400},
        )
        gate_ok = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"success": True, "message": "ok"},
        )

        with (
            patch(
                "sglang.srt.managers.tokenizer_control_mixin.requests.get",
                return_value=response,
            ) as get,
            patch(
                "sglang.srt.managers.tokenizer_control_mixin.requests.post",
                return_value=gate_ok,
            ),
        ):
            ret = asyncio.run(manager.p2p_kv_transfer(req))

        self.assertTrue(ret.success)
        get.assert_called_once_with("http://127.0.0.1:30000/server_info", timeout=5)
        dispatched = manager.p2p_kv_transfer_communicator.await_args.args[0]
        self.assertEqual(dispatched.source_bootstrap_addr, "127.0.0.1:32400")

    def test_short_receiver_result_frees_untransferred_tail_once(self):
        kv_allocator = _FakeAllocator()
        tree_cache = _FakeTreeCache()
        scheduler = self._scheduler(
            _FakeKVManager(state_types=[]),
            tree_cache=tree_cache,
            allocator=kv_allocator,
        )
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        engine._target_pull_via_receiver = MethodType(
            self._successful_receiver(transferred_tokens=4), engine
        )

        before = kv_allocator.available_size()
        ret = engine.transfer(self._req())

        self.assertTrue(ret.success)
        self.assertEqual(ret.transferred_tokens, 4)
        self.assertEqual(tree_cache.inserted[0].value.tolist(), list(range(100, 104)))
        self.assertEqual(kv_allocator.freed, [list(range(104, 108))])
        self.assertEqual(kv_allocator.available_size(), before - 4)
        self.assertEqual(kv_allocator.live, set(range(100, 104)))

    def test_invalid_receiver_counts_fail_before_insert_and_cleanup_allocations(self):
        for transferred_tokens in (0, 9):
            with self.subTest(transferred_tokens=transferred_tokens):
                kv_allocator = _FakeAllocator()
                mamba_allocator = _FakeMambaAllocator()
                tree_cache = _FakeTreeCache()
                scheduler = self._scheduler(
                    _FakeKVManager(state_types=[StateType.MAMBA]),
                    tree_cache=tree_cache,
                    allocator=kv_allocator,
                )
                scheduler.req_to_token_pool.mamba_allocator = mamba_allocator
                engine = PrefillP2PMooncakeTransferEngine(scheduler)
                engine._target_pull_via_receiver = MethodType(
                    self._successful_receiver(transferred_tokens=transferred_tokens),
                    engine,
                )

                kv_before = kv_allocator.available_size()
                mamba_before = mamba_allocator.available_size()
                ret = engine.transfer(self._req())

                self.assertFalse(ret.success)
                self.assertTrue(ret.fallback_recompute)
                self.assertIn("invalid transferred token count", ret.message)
                self.assertEqual(tree_cache.inserted, [])
                self.assertEqual(kv_allocator.available_size(), kv_before)
                self.assertEqual(mamba_allocator.available_size(), mamba_before)
                self.assertEqual(kv_allocator.live, set())
                self.assertEqual(mamba_allocator.live, set())

    def test_target_does_not_commit_when_another_pp_stage_fails(self):
        kv_allocator = _FakeAllocator()
        tree_cache = _FakeTreeCache()
        scheduler = self._scheduler(
            _FakeKVManager(state_types=[]),
            tree_cache=tree_cache,
            allocator=kv_allocator,
        )
        scheduler.world_group = SimpleNamespace(cpu_group="world-group")
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        engine._target_pull_via_receiver = MethodType(
            self._successful_receiver(), engine
        )
        before = kv_allocator.available_size()

        def all_reduce(value, op, group):
            self.assertEqual(group, "world-group")
            value[0] = 0
            value[1] = 0

        with (
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer."
                "torch.distributed.is_available",
                return_value=True,
            ),
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer."
                "torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer."
                "torch.distributed.all_reduce",
                side_effect=all_reduce,
            ) as reduce,
        ):
            ret = engine.transfer(self._req())

        self.assertFalse(ret.success)
        self.assertTrue(ret.fallback_recompute)
        self.assertIn("model-parallel rank", ret.message)
        self.assertEqual(tree_cache.inserted, [])
        self.assertEqual(kv_allocator.available_size(), before)
        self.assertEqual(kv_allocator.live, set())
        reduce.assert_called_once()

    def test_source_does_not_report_success_when_another_pp_stage_fails(self):
        scheduler = self._scheduler(_FakeKVManager(state_types=[]))
        scheduler.world_group = SimpleNamespace(cpu_group="world-group")
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        engine._source_send_via_sender = MethodType(
            lambda self, req: P2PKVTransferReqOutput(
                success=True,
                message="local source rank sent",
                source_url=req.source_url,
                target_url=req.target_url,
                matched_tokens=req.matched_tokens,
                transferred_tokens=8,
                fallback_recompute=False,
            ),
            engine,
        )
        req = self._req()
        req.p2p_source_send = True
        req.p2p_bootstrap_room = 9

        def all_reduce(value, op, group):
            value[0] = 0
            value[1] = 0

        with (
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer."
                "torch.distributed.is_available",
                return_value=True,
            ),
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer."
                "torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer."
                "torch.distributed.all_reduce",
                side_effect=all_reduce,
            ) as reduce,
        ):
            ret = engine.transfer(req)

        self.assertFalse(ret.success)
        self.assertTrue(ret.fallback_recompute)
        self.assertIn("source transfer failed", ret.message)
        reduce.assert_called_once()

    def test_insert_failure_after_tail_trim_is_fail_stop_without_double_free(self):
        kv_allocator = _FakeAllocator()
        mamba_allocator = _FakeMambaAllocator()

        def fail_insert(params):
            raise RuntimeError("insert exploded")

        tree_cache = _FakeTreeCache(on_insert=fail_insert)
        scheduler = self._scheduler(
            _FakeKVManager(state_types=[StateType.MAMBA]),
            tree_cache=tree_cache,
            allocator=kv_allocator,
        )
        scheduler.req_to_token_pool.mamba_allocator = mamba_allocator
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        engine._target_pull_via_receiver = MethodType(
            self._successful_receiver(transferred_tokens=4), engine
        )

        with self.assertRaisesRegex(_P2PCacheIntegrityError, "insert exploded"):
            engine.transfer(self._req())

        self.assertEqual(kv_allocator.freed, [list(range(104, 108))])
        self.assertEqual(kv_allocator.live, set(range(100, 104)))
        self.assertEqual(mamba_allocator.freed, [])
        self.assertEqual(mamba_allocator.live, {10_000})

    def test_invalid_insert_prefix_len_is_fail_stop_without_clamping(self):
        for insert_prefix_len in (-1, 9):
            with self.subTest(insert_prefix_len=insert_prefix_len):
                kv_allocator = _FakeAllocator()
                tree_cache = _FakeTreeCache(
                    insert_prefix_len=insert_prefix_len,
                    duplicate_kv_handled_by_cache=False,
                )
                scheduler = self._scheduler(
                    _FakeKVManager(state_types=[]),
                    tree_cache=tree_cache,
                    allocator=kv_allocator,
                )
                engine = PrefillP2PMooncakeTransferEngine(scheduler)
                engine._target_pull_via_receiver = MethodType(
                    self._successful_receiver(), engine
                )

                with self.assertRaisesRegex(
                    _P2PCacheIntegrityError, "invalid insert prefix_len"
                ):
                    engine.transfer(self._req())

                self.assertEqual(len(tree_cache.inserted), 1)
                self.assertEqual(kv_allocator.freed, [])
                self.assertEqual(kv_allocator.live, set(range(100, 108)))

    def test_post_commit_verify_failure_keeps_transport_successful(self):
        kv_allocator = _FakeAllocator()
        tree_cache = _FakeTreeCache()
        scheduler = self._scheduler(
            _FakeKVManager(state_types=[]),
            tree_cache=tree_cache,
            allocator=kv_allocator,
        )
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        engine._target_pull_via_receiver = MethodType(
            self._successful_receiver(), engine
        )
        original_insert = tree_cache.insert

        def insert_then_break_verify(params):
            result = original_insert(params)

            def broken_match(match_params):
                raise RuntimeError("verify exploded")

            tree_cache.match_prefix = broken_match
            return result

        tree_cache.insert = insert_then_break_verify

        with self.assertLogs(
            "sglang.srt.disaggregation.p2p_kv_transfer", level="WARNING"
        ) as logs:
            ret = engine.transfer(self._req())

        self.assertTrue(ret.success)
        self.assertFalse(ret.fallback_recompute)
        self.assertEqual(kv_allocator.freed, [])
        self.assertEqual(kv_allocator.live, set(range(100, 108)))
        self.assertIn("p2p_target_post_commit_verify_failed", "\n".join(logs.output))

    def test_registration_result_reports_ordinary_duplicate_settlement(self):
        kv_allocator = _FakeAllocator()
        tree_cache = _FakeTreeCache(
            insert_prefix_len=4,
            duplicate_kv_handled_by_cache=False,
        )
        scheduler = self._scheduler(
            _FakeKVManager(state_types=[]),
            tree_cache=tree_cache,
            allocator=kv_allocator,
        )
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        allocation = _TargetAllocation(kv=kv_allocator.alloc(8))

        result = engine._register_target_prefix(self._req(), allocation, 8)

        self.assertEqual(result.committed_tokens, 8)
        self.assertEqual(result.duplicate_kv_owner, "p2p")
        self.assertTrue(result.duplicate_kv_freed_by_p2p)
        self.assertFalse(result.cache_accepted_mamba)
        self.assertEqual(kv_allocator.freed, [list(range(100, 104))])
        self.assertEqual(kv_allocator.live, set(range(104, 108)))

    def test_target_pull_evicts_cache_before_allocating_kv_slots(self):
        kv_manager = _FakeKVManager(state_types=[])
        allocator = _FakeAllocator(available=4)
        scheduler = self._scheduler(kv_manager, allocator=allocator)
        engine = PrefillP2PMooncakeTransferEngine(scheduler)

        def fake_receiver(self, req, kv_manager, dst_kv, dst_mamba, prefix_len):
            return P2PKVTransferReqOutput(
                success=True,
                message="ok",
                source_url=req.source_url,
                target_url=req.target_url,
                matched_tokens=req.matched_tokens,
                transferred_tokens=prefix_len,
                fallback_recompute=False,
                experimental_limitations=self._limitations(),
            )

        engine._target_pull_via_receiver = MethodType(fake_receiver, engine)

        ret = engine.transfer(self._req())

        self.assertTrue(ret.success)
        self.assertEqual(allocator.allocations, [8])
        self.assertEqual(
            [params.num_tokens for params in scheduler.tree_cache.evicted], [4]
        )

    def test_target_pull_kv_only_does_not_require_mamba_allocator(self):
        kv_manager = _FakeKVManager(state_types=[])
        scheduler = self._scheduler(kv_manager)
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        captured = {}

        def fake_receiver(self, req, kv_manager, dst_kv, dst_mamba, prefix_len):
            captured["dst_mamba"] = dst_mamba
            return P2PKVTransferReqOutput(
                success=True,
                message="ok",
                source_url=req.source_url,
                target_url=req.target_url,
                matched_tokens=req.matched_tokens,
                transferred_tokens=prefix_len,
                fallback_recompute=False,
                experimental_limitations=self._limitations(),
            )

        engine._target_pull_via_receiver = MethodType(fake_receiver, engine)

        with self.assertLogs(
            "sglang.srt.disaggregation.p2p_kv_transfer", level="INFO"
        ) as logs:
            ret = engine.transfer(self._req())

        self.assertTrue(ret.success)
        self.assertFalse(ret.fallback_recompute)
        self.assertEqual(ret.transferred_tokens, 8)
        self.assertIsNone(captured["dst_mamba"])
        self.assertEqual(len(scheduler.tree_cache.inserted), 1)
        self.assertIsNone(scheduler.tree_cache.inserted[0].mamba_value)
        log_text = "\n".join(logs.output)
        self.assertIn("p2p_target_plan", log_text)
        self.assertIn("state_types=[]", log_text)
        self.assertIn("requires_mamba=False", log_text)

    def test_target_pull_allows_reverse_pair_direction(self):
        kv_manager = _FakeKVManager(state_types=[])
        scheduler = self._scheduler(kv_manager)
        engine = PrefillP2PMooncakeTransferEngine(scheduler)

        def fake_receiver(self, req, kv_manager, dst_kv, dst_mamba, prefix_len):
            return P2PKVTransferReqOutput(
                success=True,
                message="ok",
                source_url=req.source_url,
                target_url=req.target_url,
                matched_tokens=req.matched_tokens,
                transferred_tokens=prefix_len,
                fallback_recompute=False,
                experimental_limitations=self._limitations(),
            )

        engine._target_pull_via_receiver = MethodType(fake_receiver, engine)

        ret = engine.transfer(self._reverse_req())

        self.assertTrue(ret.success)
        self.assertFalse(ret.fallback_recompute)
        self.assertEqual(ret.transferred_tokens, 8)

    def test_target_pull_disables_direct_pointer_payload_when_receiver_unavailable(
        self,
    ):
        kv_manager = _FakeKVManager(state_types=[])
        scheduler = self._scheduler(kv_manager)
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        engine._target_pull_via_receiver = MethodType(
            lambda self, req, kv_manager, dst_kv, dst_mamba, prefix_len: None,
            engine,
        )

        with patch("sglang.srt.disaggregation.p2p_kv_transfer.requests.post") as post:
            ret = engine.transfer(self._req())

        self.assertFalse(ret.success)
        self.assertTrue(ret.fallback_recompute)
        self.assertIn("direct P2P transfer is disabled", ret.message)
        post.assert_not_called()

    def test_fallback_log_contains_request_context(self):
        kv_manager = _FakeKVManager(state_types=[StateType.MINIMAX_INDEX_K])
        scheduler = self._scheduler(kv_manager)
        engine = PrefillP2PMooncakeTransferEngine(scheduler)

        with self.assertLogs(
            "sglang.srt.disaggregation.p2p_kv_transfer", level="WARNING"
        ) as logs:
            ret = engine._fail(self._req(), "boom")

        self.assertFalse(ret.success)
        log_text = "\n".join(logs.output)
        self.assertIn("p2p_transfer_fallback", log_text)
        self.assertIn("request_id=rid-1", log_text)
        self.assertIn("reason=load_imbalance", log_text)
        self.assertIn("state_types=['minimax_index_k']", log_text)

    def test_source_sender_kv_only_does_not_require_cached_mamba_value(self):
        _FakeSender.instances = []
        kv_manager = _FakeKVManager(state_types=[])
        tree_cache = _FakeTreeCache(
            matched_indices=torch.arange(10, 18, dtype=torch.int32),
            last_device_node=SimpleNamespace(),
        )
        scheduler = self._scheduler(kv_manager, tree_cache=tree_cache)
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        req = self._req()
        req.p2p_bootstrap_room = 123
        req.p2p_source_send = True

        with patch(
            "sglang.srt.disaggregation.p2p_kv_transfer.get_kv_class",
            return_value=_FakeSender,
        ):
            ret = engine.transfer(req)

        self.assertTrue(ret.success)
        self.assertFalse(ret.fallback_recompute)
        self.assertEqual(ret.transferred_tokens, 8)
        self.assertEqual(len(_FakeSender.instances), 1)
        sent_kv_indices, sent_state_indices = _FakeSender.instances[0].sent[0]
        self.assertEqual(sent_kv_indices.tolist(), [2, 3])
        self.assertEqual(sent_state_indices, [])

    def test_source_sender_loads_hicache_l2_prefix_before_mooncake(self):
        class _FinishEvent:
            def synchronize(self):
                pass

        class _HiCacheTree(_FakeTreeCache):
            def __init__(self):
                self.cache_node = SimpleNamespace()
                super().__init__(last_device_node=self.cache_node)
                self.restored = False
                self.load_back_calls = []
                self.loading_checks = 0
                self.cache_controller = SimpleNamespace(
                    layer_done_counter=SimpleNamespace(
                        events=[SimpleNamespace(finish_event=_FinishEvent())]
                    )
                )

            def match_prefix(self, params):
                if self.restored:
                    device_indices = torch.arange(10, 18, dtype=torch.int32)
                    host_hit_length = 0
                else:
                    device_indices = torch.empty((0,), dtype=torch.int32)
                    host_hit_length = 8
                return SimpleNamespace(
                    device_indices=device_indices,
                    last_device_node=self.cache_node,
                    last_host_node=self.cache_node,
                    best_match_node=self.cache_node,
                    host_hit_length=host_hit_length,
                    mamba_host_hit_length=0,
                )

            def load_back(self, node, mem_quota=None):
                self.load_back_calls.append((node, mem_quota))
                self.restored = True
                return True

            def ready_to_load_host_cache(self):
                return 0

            def loading_check(self):
                self.loading_checks += 1

        _FakeSender.instances = []
        kv_manager = _FakeKVManager(state_types=[])
        tree_cache = _HiCacheTree()
        scheduler = self._scheduler(kv_manager, tree_cache=tree_cache)
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        req = self._req()
        req.p2p_bootstrap_room = 123
        req.p2p_source_send = True

        with patch(
            "sglang.srt.disaggregation.p2p_kv_transfer.get_kv_class",
            return_value=_FakeSender,
        ):
            ret = engine.transfer(req)

        self.assertTrue(ret.success)
        self.assertEqual(ret.transferred_tokens, 8)
        self.assertEqual(tree_cache.load_back_calls, [(tree_cache.cache_node, None)])
        self.assertEqual(tree_cache.loading_checks, 1)
        sent_kv_indices, sent_state_indices = _FakeSender.instances[0].sent[0]
        self.assertEqual(sent_kv_indices.tolist(), [2, 3])
        self.assertEqual(sent_state_indices, [])

    def test_source_bootstrap_addr_comes_from_source_server_info(self):
        kv_manager = _FakeKVManager(state_types=[])
        kv_manager.server_args = SimpleNamespace(disaggregation_bootstrap_port=32401)
        scheduler = self._scheduler(kv_manager)
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"disaggregation_bootstrap_port": 32400},
        )

        with patch(
            "sglang.srt.disaggregation.p2p_kv_transfer.requests.get",
            return_value=response,
        ) as get:
            addr = engine._source_bootstrap_addr("http://127.0.0.1:30000")

        self.assertEqual(addr, "127.0.0.1:32400")
        get.assert_called_once_with("http://127.0.0.1:30000/server_info", timeout=5)

    def test_source_bootstrap_addr_reuses_successful_discovery_when_source_is_busy(
        self,
    ):
        kv_manager = _FakeKVManager(state_types=[])
        scheduler = self._scheduler(kv_manager)
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"disaggregation_bootstrap_port": 32400},
        )

        with patch(
            "sglang.srt.disaggregation.p2p_kv_transfer.requests.get",
            side_effect=[response, TimeoutError("source is busy")],
        ) as get:
            first = engine._source_bootstrap_addr("http://127.0.0.1:30000")
            second = engine._source_bootstrap_addr("http://127.0.0.1:30000")

        self.assertEqual(first, "127.0.0.1:32400")
        self.assertEqual(second, "127.0.0.1:32400")
        get.assert_called_once_with("http://127.0.0.1:30000/server_info", timeout=5)

    def test_target_receiver_uses_prefetched_source_bootstrap(self):
        kv_manager = _FakeKVManager(state_types=[])
        kv_manager.try_ensure_parallel_info = lambda addr, **kwargs: True
        kv_manager.prefill_info_table = {
            "127.0.0.1:32400": SimpleNamespace(),
        }
        scheduler = self._scheduler(kv_manager)
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        req = self._req()
        req.source_bootstrap_addr = "127.0.0.1:32400"

        class Receiver:
            def __init__(self, *args, **kwargs):
                pass

            def init(self, prefill_dp_rank):
                pass

            def send_metadata(self, *args, **kwargs):
                pass

        engine._validate_identical_layout = MethodType(
            lambda self, manager, info: None, engine
        )
        polls = iter([KVPoll.WaitingForInput, KVPoll.WaitingForInput, KVPoll.Success])
        engine._poll_receiver_consensus = MethodType(
            lambda self, receiver, force_failed=False: next(polls), engine
        )
        engine._min_target_consensus = MethodType(lambda self, value: value, engine)

        with (
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer.get_kv_class",
                return_value=Receiver,
            ),
            patch.object(
                engine,
                "_source_bootstrap_addr",
                side_effect=AssertionError("must not rediscover source bootstrap"),
            ),
            patch(
                "sglang.srt.disaggregation.p2p_kv_transfer.requests.post",
                return_value=SimpleNamespace(
                    status_code=200,
                    json=lambda: {"success": True, "transferred_tokens": 8},
                ),
            ),
            patch("sglang.srt.disaggregation.p2p_kv_transfer.time.sleep"),
        ):
            ret = engine._target_pull_via_receiver(
                req, kv_manager, torch.arange(8), None, 8
            )

        self.assertTrue(ret.success)

    def test_cached_mamba_index_reads_unified_component_data(self):
        kv_manager = _FakeKVManager(state_types=[StateType.MAMBA])
        scheduler = self._scheduler(kv_manager)
        engine = PrefillP2PMooncakeTransferEngine(scheduler)
        node = SimpleNamespace(
            tree_components=(ComponentType.MAMBA,),
            component_data=[
                SimpleNamespace(value=None),
                SimpleNamespace(value=None),
                SimpleNamespace(value=torch.tensor([7])),
            ],
        )
        match = SimpleNamespace(last_device_node=node)

        self.assertEqual(
            engine._cached_mamba_index_if_needed(self._req(), kv_manager, match),
            7,
        )

    def test_minimax_index_k_state_reuses_page_indices(self):
        kv_manager = _FakeKVManager(state_types=[StateType.MINIMAX_INDEX_K])
        scheduler = self._scheduler(kv_manager)
        engine = PrefillP2PMooncakeTransferEngine(scheduler)

        state_indices = engine._state_indices_for_pages(
            kv_manager,
            torch.tensor([8, 9], dtype=torch.int32),
            mamba_index=None,
        )

        self.assertEqual(state_indices, [[8, 9]])

    def test_backend_page_indices_are_contiguous_numpy_int32(self):
        engine = PrefillP2PMooncakeTransferEngine(
            self._scheduler(_FakeKVManager(state_types=[]))
        )
        source = torch.tensor([8, 12, 16, 20], dtype=torch.int64)[::2]

        indices = engine._backend_page_indices(source)

        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.dtype, np.int32)
        self.assertTrue(indices.flags.c_contiguous)
        self.assertEqual(indices.tolist(), [8, 16])


if __name__ == "__main__":
    unittest.main()
