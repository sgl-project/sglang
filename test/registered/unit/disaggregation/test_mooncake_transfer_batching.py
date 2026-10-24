import concurrent.futures
import ctypes
import unittest
from collections import defaultdict
from queue import SimpleQueue
from threading import Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeTransferBatching(unittest.TestCase):
    @staticmethod
    def _make_manager(
        side_effect=None, enable_custom_mem_pool=False, max_batch_indices=0
    ):
        engine = MagicMock()
        if side_effect is None:
            engine.batch_transfer_sync.return_value = 0
        else:
            engine.batch_transfer_sync.side_effect = side_effect
        manager = SimpleNamespace(
            engine=engine,
            is_mla_backend=True,
            is_hybrid_mla_backend=False,
            pp_size=1,
            enable_custom_mem_pool=enable_custom_mem_pool,
            custom_mem_pool_type="NVLINK" if enable_custom_mem_pool else None,
            enable_deferred_decode_kv_release=False,
            max_transfer_batch_indices=max_batch_indices,
            get_mla_kv_ptrs_with_pp=MagicMock(
                return_value=([1000, 2000], [5000, 6000], 2)
            ),
        )
        manager._transfer_data = lambda session, blocks: (
            MooncakeKVManager._transfer_data(manager, session, blocks)
        )
        manager._await_transfer_futures = lambda futures: (
            MooncakeKVManager._await_transfer_futures(manager, futures)
        )
        manager._get_kvcache_layer_params = (
            MooncakeKVManager._get_kvcache_layer_params.__get__(manager)
        )
        return manager

    @staticmethod
    def _send(
        manager,
        dst_device_data_indices=None,
        dst_device_data_ptrs=None,
    ):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return MooncakeKVManager._send_kvcache_generic(
                manager,
                mooncake_session_id="session",
                src_data_ptrs=[1000, 2000],
                dst_data_ptrs=[5000, 6000],
                item_lens=[10, 20],
                prefill_data_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
                dst_data_indices=np.array([10, 11, 12, 13, 14], dtype=np.int32),
                executor=executor,
                dst_device_data_indices=dst_device_data_indices,
                dst_device_data_ptrs=dst_device_data_ptrs,
            )

    def test_slices_index_arrays_before_forming_transfer_ranges(self):
        manager = self._make_manager(max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, 0)
        self.assertEqual(
            manager.engine.batch_transfer_sync.call_args_list,
            [
                call("session", [1000, 2000], [5100, 6200], [20, 40]),
                call("session", [1020, 2040], [5120, 6240], [20, 40]),
                call("session", [1040, 2080], [5140, 6280], [10, 20]),
            ],
        )

    def test_preserves_legacy_single_batch_path_for_short_transfers(self):
        for max_batch_indices in (0, 5, 6):
            with self.subTest(max_batch_indices=max_batch_indices):
                manager = self._make_manager(max_batch_indices=max_batch_indices)
                ret = self._send(manager)

                self.assertEqual(ret, 0)
                manager.engine.batch_transfer_sync.assert_called_once_with(
                    "session",
                    [1000, 2000],
                    [5100, 6200],
                    [50, 100],
                )

    def test_stops_after_first_failed_index_batch(self):
        manager = self._make_manager(side_effect=[0, -1], max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, -1)
        self.assertEqual(manager.engine.batch_transfer_sync.call_count, 2)

    def test_uses_device_page_indices_in_batched_path(self):
        manager = self._make_manager(max_batch_indices=2)
        ret = self._send(
            manager,
            dst_device_data_indices=np.array([20, 21, 22, 23, 24], dtype=np.int32),
            dst_device_data_ptrs={6000},
        )

        self.assertEqual(ret, 0)
        self.assertEqual(
            manager.engine.batch_transfer_sync.call_args_list,
            [
                call("session", [1000, 2000], [5100, 6400], [20, 40]),
                call("session", [1020, 2040], [5120, 6440], [20, 40]),
                call("session", [1040, 2080], [5140, 6480], [10, 20]),
            ],
        )

    def test_preserves_one_transfer_per_layer_for_custom_mem_pool(self):
        manager = self._make_manager(enable_custom_mem_pool=True, max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, 0)
        self.assertEqual(manager.engine.batch_transfer_sync.call_count, 2)
        manager.engine.batch_transfer_sync.assert_has_calls(
            [
                call("session", [1000], [5100], [50]),
                call("session", [2000], [6200], [100]),
            ],
            any_order=True,
        )


class TestMiniMaxStateTransfer(CustomTestCase):
    def test_index_truncates_but_dense_rejects_mismatched_page_lists(self):
        """Legacy index transfers copy the common prefix; incomplete dense KV must fail."""

        def copy_bytes(session, sources, destinations, lengths):
            for src, dst, length in zip(sources, destinations, lengths, strict=True):
                ctypes.memmove(dst, src, length)
            return 0

        for state in (StateType.MINIMAX_INDEX_K, StateType.MINIMAX_DENSE_KV):
            for src_pages, dst_pages in (([1], [0]), ([1, 2], [0]), ([1], [0, 2])):
                with self.subTest(state=state, src=src_pages, dst=dst_pages):
                    src = np.arange(3, dtype=np.int32)
                    dst = np.full(3, -1, dtype=np.int32)
                    manager = MooncakeKVManager.__new__(MooncakeKVManager)
                    manager.kv_args = SimpleNamespace(
                        state_types=[state],
                        state_data_ptrs=[[src.ctypes.data]],
                        state_item_lens=[[src.itemsize]],
                        state_dim_per_tensor=[[]],
                        state_layer_ids=[[]],
                    )
                    manager.engine = SimpleNamespace(batch_transfer_sync=copy_bytes)
                    manager.pp_size = manager.attn_tp_size = 1
                    manager.is_mla_backend = manager.is_hybrid_mla_backend = False
                    manager.enable_custom_mem_pool = False
                    manager.max_transfer_batch_indices = 0
                    peer = SimpleNamespace(
                        dst_state_data_ptrs=[[dst.ctypes.data]],
                        dst_state_item_lens=[[dst.itemsize]],
                        dst_state_dim_per_tensor=[[]],
                        dst_state_layer_ids=[[]],
                        dst_attn_tp_size=1,
                    )
                    kwargs = dict(
                        req=SimpleNamespace(
                            mooncake_session_id="cpu", dst_state_indices=[dst_pages]
                        ),
                        prefill_state_indices=[src_pages],
                        executor=None,
                        target_rank_registration_info=peer,
                    )
                    if state == StateType.MINIMAX_DENSE_KV and len(src_pages) != len(
                        dst_pages
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "state index length mismatch"
                        ):
                            manager.maybe_send_extra(**kwargs)
                        np.testing.assert_array_equal(dst, [-1, -1, -1])
                    else:
                        self.assertEqual(manager.maybe_send_extra(**kwargs), 0)
                        np.testing.assert_array_equal(dst, [1, -1, -1])


class TestDcpDraftHeadTransfer(unittest.TestCase):
    def test_transfers_draft_heads_to_logical_destination_rows(self):
        for src_tp, dst_tp in ((4, 8), (8, 4), (8, 8), (4, 32), (32, 4)):
            for custom_pool in (False, True):
                for batch_size in (0, 37):
                    with self.subTest(
                        src_tp=src_tp,
                        dst_tp=dst_tp,
                        custom_pool=custom_pool,
                        batch_size=batch_size,
                    ):
                        self._check_transfer(src_tp, dst_tp, custom_pool, batch_size)

    def test_rejects_pure_mla_with_unequal_draft_head_widths(self):
        for src_tp, dst_tp in ((4, 8), (8, 4)):
            with self.subTest(src_tp=src_tp, dst_tp=dst_tp):
                with self.assertRaisesRegex(ValueError, "dummy prefill senders"):
                    self._check_transfer(src_tp, dst_tp, False, 37, pure_mla=True)

    def test_sliced_draft_stops_after_failed_batch(self):
        self._check_transfer(4, 8, False, 37, fail_draft=True)

    def _check_transfer(
        self, src_tp, dst_tp, custom_pool, batch_size, fail_draft=False, pure_mla=False
    ):
        page_size, tokens, heads, head_bytes = 64, 249, 16, 4
        src_width, dst_width = (
            max(1, heads // src_tp) * head_bytes,
            max(1, heads // dst_tp) * head_bytes,
        )
        src_pages = np.array([1, 3, 4, 7], dtype=np.int32)
        logical = np.arange(tokens)
        src_rows = src_pages[logical // page_size] * page_size + logical % page_size
        expected = (
            np.arange(tokens * heads * head_bytes, dtype=np.int64)
            .reshape(tokens, heads, head_bytes)
            .astype(np.uint8)
        )
        for dst_rank in range(dst_tp):
            dst_buffers = {
                base: np.zeros(16384 * max(8, dst_width), dtype=np.uint8)
                for base in (1000000, 2000000, 3000000, 4000000)
            }
            source_ranks = (
                range(dst_rank * src_tp // dst_tp, (dst_rank + 1) * src_tp // dst_tp)
                if src_tp >= dst_tp
                else [dst_rank * src_tp // dst_tp]
            )
            for src_rank in source_ranks:
                src_head_start = (src_rank // max(1, src_tp // heads)) * max(
                    1, heads // src_tp
                )
                source = np.zeros(1024 * src_width, dtype=np.uint8)
                source.reshape(-1, src_width)[src_rows] = expected[
                    :, src_head_start : src_head_start + max(1, heads // src_tp)
                ].reshape(tokens, src_width)
                target = np.zeros(1024 * 8, dtype=np.uint8)
                target.reshape(-1, 8)[src_rows] = (
                    np.arange(tokens * 8).reshape(tokens, 8).astype(np.uint8)
                )
                src_buffers = {10000: target, 100000: source, 200000: source}

                failed_batches = []

                def transfer(
                    session, blocks, src_buffers=src_buffers, dst_buffers=dst_buffers
                ):
                    draft_blocks = [block for block in blocks if block[1] >= 3000000]
                    if fail_draft and draft_blocks:
                        failed_batches.append(draft_blocks)
                        return 17
                    if batch_size and src_width != dst_width:
                        self.assertLessEqual(
                            len(draft_blocks), batch_size * (1 if custom_pool else 2)
                        )
                    for src, dst, size in blocks:
                        src_base = max(base for base in src_buffers if base <= src)
                        dst_base = max(base for base in dst_buffers if base <= dst)
                        dst_buffers[dst_base][
                            dst - dst_base : dst - dst_base + size
                        ] = src_buffers[src_base][
                            src - src_base : src - src_base + size
                        ]
                    return 0

                manager = SimpleNamespace(
                    is_mla_backend=pure_mla,
                    kv_args=SimpleNamespace(
                        page_size=page_size,
                        kv_layer_ids=[47, 93, 93],
                        kv_data_ptrs=[10000, 100000, 200000],
                        num_draft_entries=2,
                        engine_rank=src_rank + 2 * src_tp,
                    ),
                    attn_tp_size=src_tp,
                    max_transfer_batch_indices=batch_size,
                    enable_custom_mem_pool=custom_pool,
                    _transfer_data=transfer,
                    _await_transfer_futures=lambda futures: max(
                        f.result() for f in futures
                    ),
                )
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = MooncakeKVManager.send_kvcache_dcp(
                        manager,
                        "session",
                        src_pages,
                        [1000000, 2000000, 3000000, 4000000],
                        np.array([2], dtype=np.int32),
                        dcp_token_item_lens=[8, src_width, src_width],
                        dst_dcp_size=dst_tp,
                        dst_dcp_rank=dst_rank,
                        src_page_offset=0,
                        decode_prefix_len=0,
                        num_kv_tokens=tokens,
                        executor=executor,
                        dst_layer_ids=[3, 47, 93, 93],
                        dst_kv_item_lens=[
                            page_size * 8,
                            page_size * 8,
                            page_size * dst_tp * dst_width,
                            page_size * dst_tp * dst_width,
                        ],
                        dst_tp_rank=dst_rank,
                        dst_attn_tp_size=dst_tp,
                    )
                if fail_draft:
                    self.assertEqual(result, 17)
                    self.assertEqual(len(failed_batches), 1)
                    return
                self.assertEqual(result, 0)
            dst_head_start = (dst_rank // max(1, dst_tp // heads)) * max(
                1, heads // dst_tp
            )
            for base in (3000000, 4000000):
                actual = dst_buffers[base].reshape(-1, dst_width)[
                    2 * page_size * dst_tp + logical
                ]
                np.testing.assert_array_equal(
                    actual,
                    expected[
                        :,
                        dst_head_start : dst_head_start + max(1, heads // dst_tp),
                    ].reshape(tokens, dst_width),
                )
            owned = np.arange(dst_rank, tokens, dst_tp)
            actual_target = dst_buffers[2000000].reshape(-1, 8)[
                2 * page_size + owned // dst_tp
            ]
            np.testing.assert_array_equal(
                actual_target,
                np.arange(tokens * 8).reshape(tokens, 8).astype(np.uint8)[owned],
            )
            self.assertFalse(dst_buffers[1000000].any())


class TestDcpPackLifetime(CustomTestCase):
    def test_failed_transfer_drains_before_pack_buffer_reuse(self):
        """A failed layer must not release the pack buffer while another transfer reads it."""
        manager = TestMooncakeTransferBatching._make_manager(
            enable_custom_mem_pool=True
        )
        manager.kv_args = SimpleNamespace(
            page_size=1, kv_layer_ids=[], kv_data_ptrs=[1000, 2000], num_draft_entries=0
        )
        source = np.array([11], dtype=np.uint8)
        observed = []
        running, release = Event(), Event()

        def transfer(session, blocks):
            if blocks[0][0] == 1000:
                self.assertTrue(running.wait(10))
                return 17
            running.set()
            self.assertTrue(release.wait(10))
            observed.append(int(source[0]))
            return 0

        def send(executor):
            result = MooncakeKVManager.send_kvcache_dcp(
                manager,
                "session",
                np.array([0, 1], dtype=np.int32),
                [5000, 6000],
                np.array([0], dtype=np.int32),
                dcp_token_item_lens=[1, 1],
                dst_dcp_size=2,
                dst_dcp_rank=0,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=2,
                executor=executor,
                dst_layer_ids=[],
                pack_buffer=object(),
            )
            source[0] = 22
            return result

        manager._transfer_data = transfer
        with (
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.try_pack_dcp_src",
                return_value=([1000, 2000], np.array([0], dtype=np.int64)),
            ),
            concurrent.futures.ThreadPoolExecutor(max_workers=2) as transfers,
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker,
        ):
            future = worker.submit(send, transfers)
            try:
                self.assertTrue(running.wait(10))
                with self.assertRaises(concurrent.futures.TimeoutError):
                    future.result(timeout=1)
            finally:
                release.set()
            self.assertEqual(future.result(timeout=10), 17)
        self.assertEqual(observed, [11])


class TestMooncakeEarlySend(CustomTestCase):
    def test_sender_queues_event_once_without_waiting_on_scheduler(self):
        for last in (False, True):
            with self.subTest(last=last):
                ready = MagicMock()
                manager = MagicMock()
                sender = object.__new__(MooncakeKVSender)
                vars(sender).update(
                    kv_mgr=manager,
                    bootstrap_room=1,
                    aux_index=0,
                    trace_ctx=MagicMock(),
                    conclude_state=None,
                    _transfer_start_time=None,
                    _early_send_wait_event=ready,
                    _prepare_send_indices=lambda ids, state: (
                        ids,
                        slice(0, 1),
                        last,
                        False,
                    ),
                    _record_transfer_indices=MagicMock(),
                )
                indices_ready = MagicMock()
                sender.send([0], indices_event=indices_ready)
                kwargs = manager.add_transfer_request.call_args.kwargs
                self.assertIs(kwargs["wait_event"], ready)
                self.assertIs(kwargs["indices_event"], indices_ready)
                ready.synchronize.assert_not_called()
                indices_ready.synchronize.assert_not_called()
                sender.send([1])
                kwargs = manager.add_transfer_request.call_args.kwargs
                self.assertIsNone(kwargs["wait_event"])
                self.assertIsNone(kwargs["indices_event"])

    def test_manager_preserves_event_in_queued_chunk(self):
        queue = SimpleQueue()
        ready = MagicMock()
        manager = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.PREFILL,
            request_status={1: KVPoll.WaitingForInput},
            check_status=lambda room: KVPoll.WaitingForInput,
            transfer_infos={1: {"127.0.0.1:80": object()}},
            transfer_queues=[queue],
        )
        indices_ready = MagicMock()
        MooncakeKVManager.add_transfer_request(
            manager,
            1,
            [0],
            slice(0, 1),
            False,
            wait_event=ready,
            indices_event=indices_ready,
        )
        chunk = queue.get_nowait()
        self.assertIs(chunk.wait_event, ready)
        self.assertIs(chunk.indices_event, indices_ready)
        ready.synchronize.assert_not_called()
        indices_ready.synchronize.assert_not_called()

    def _worker(self, event):
        chunk = TransferKVChunk(
            1, [0], slice(0, 1), False, None, None, wait_event=event
        )
        queue = SimpleQueue()
        queue.put(chunk)
        queue.put(None)
        peer = SimpleNamespace(
            is_dummy=False,
            mooncake_session_id="peer",
            dst_device_kv_indices=None,
            dst_kv_indices=[10],
        )
        registration = SimpleNamespace(
            requires_dcp_relayout=False,
            dst_kv_ptrs=[1000],
            dst_kv_layer_ids=[0],
            dst_kv_item_len=16,
            dst_attn_tp_size=1,
        )
        manager = SimpleNamespace(
            enable_trace=False,
            enable_staging=False,
            enable_deferred_decode_kv_release=True,
            _staging_outstanding=defaultdict(int),
            request_status={1: KVPoll.WaitingForInput},
            transfer_infos={1: {"peer": peer}},
            _prefill_unique_rank=lambda: 0,
            session_lock=Lock(),
            failed_sessions=set(),
            # Read under session_lock alongside failed_sessions on every
            # non-dummy req, and invoked by the worker's except path.
            state_layout_rejections={},
            poison_deferred_ack_room=MagicMock(),
            decode_kv_args_table={"peer": registration},
            _get_dsa_cache_transfer_skip_flags=lambda reg: (False, False),
            kv_args=SimpleNamespace(kv_data_ptrs=[1]),
            is_mla_backend=True,
            send_kvcache=MagicMock(return_value=0),
            _maybe_ack_drained_abort=MagicMock(),
            bootstrap_port=8998,
        )
        manager.check_status = lambda room: manager.request_status[room]
        manager._request_batching_enabled = (
            MooncakeKVManager._request_batching_enabled.__get__(manager)
        )
        return manager, queue

    def test_worker_waits_before_reading_kv(self):
        ready = MagicMock()
        manager, queue = self._worker(ready)

        def transfer(*args, **kwargs):
            ready.synchronize.assert_called_once_with()
            return 0

        manager.send_kvcache.side_effect = transfer
        MooncakeKVManager.transfer_worker(manager, queue, MagicMock())
        manager.send_kvcache.assert_called_once()

    def test_worker_sends_indices_after_copy_completes(self):
        manager, queue = self._worker(None)
        chunk = queue.get_nowait()
        host_indices = torch.full((1,), -1, dtype=torch.int32)
        chunk.prefill_kv_indices = host_indices
        chunk.indices_event = MagicMock()
        chunk.indices_event.synchronize.side_effect = lambda: host_indices.fill_(7)
        queue.get_nowait()  # Replace the stop sentinel after the updated chunk.
        queue.put(chunk)
        queue.put(None)
        sent_indices = []

        def transfer(session, indices, *args, **kwargs):
            self.assertIsInstance(indices, np.ndarray)
            sent_indices.extend(indices.tolist())
            return 0

        manager.send_kvcache.side_effect = transfer
        MooncakeKVManager.transfer_worker(manager, queue, MagicMock())

        self.assertEqual(sent_indices, [7])

    def test_abort_during_wait_prevents_transfer_and_releases_ack(self):
        ready = MagicMock()
        manager, queue = self._worker(ready)
        ready.synchronize.side_effect = lambda: manager.request_status.update(
            {1: KVPoll.Failed}
        )
        MooncakeKVManager.transfer_worker(manager, queue, MagicMock())
        ready.synchronize.assert_called_once_with()
        manager.send_kvcache.assert_not_called()
        manager._maybe_ack_drained_abort.assert_called_once_with(1)
        self.assertNotIn(1, manager._staging_outstanding)

    def test_failed_room_skips_wait(self):
        ready = MagicMock()
        manager, queue = self._worker(ready)
        manager.request_status[1] = KVPoll.Failed
        MooncakeKVManager.transfer_worker(manager, queue, MagicMock())
        ready.synchronize.assert_not_called()
        manager.send_kvcache.assert_not_called()
        manager._maybe_ack_drained_abort.assert_called_once_with(1)
        self.assertNotIn(1, manager._staging_outstanding)

    def test_worker_preserves_no_event_path(self):
        manager, queue = self._worker(None)
        MooncakeKVManager.transfer_worker(manager, queue, MagicMock())
        manager.send_kvcache.assert_called_once()


class TestMooncakeStagingDeviceType(unittest.TestCase):
    """Staging init must receive the runtime's device, not a hardcoded "cuda".

    device_type is keyword-only with no default, so dropping it raises. The
    silent failure mode is a literal: it works on a CUDA host and allocates
    staging on the wrong device for every other accelerator.
    """

    @staticmethod
    def _manager():
        return SimpleNamespace(
            kv_args=SimpleNamespace(gpu_id=3),
            _staging_ctx=SimpleNamespace(buffers=None, allocator=None),
            kv_buffer_tensors=MagicMock(),
            _register_staging_memory=MagicMock(),
        )

    def test_prefill_staging_buffers_get_the_runtime_device(self):
        manager = self._manager()
        with (
            patch(
                "sglang.srt.disaggregation.mooncake.conn.get_device",
                return_value=SimpleNamespace(device="xpu"),
            ),
            patch(
                "sglang.srt.disaggregation.mooncake.conn.get_schedule",
                return_value=SimpleNamespace(chunked_prefill_size=2048),
            ),
            patch(
                "sglang.srt.disaggregation.common.staging_handler.init_staging_buffers"
            ) as init_buffers,
        ):
            MooncakeKVManager._init_staging_buffers(manager, 2)

        self.assertEqual(init_buffers.call_args.kwargs["device_type"], "xpu")

    def test_decode_staging_allocator_gets_the_runtime_device(self):
        manager = self._manager()
        with (
            patch(
                "sglang.srt.disaggregation.mooncake.conn.get_device",
                return_value=SimpleNamespace(device="xpu"),
            ),
            patch(
                "sglang.srt.disaggregation.common.staging_handler.init_staging_allocator"
            ) as init_allocator,
        ):
            MooncakeKVManager._init_staging_allocator(manager)

        self.assertEqual(init_allocator.call_args.kwargs["device_type"], "xpu")


class TestMooncakeRequestBatching(CustomTestCase):
    """Protect request page mappings and completion/abort ordering across a batch."""

    def setUp(self):
        memory = patch(
            "sglang.srt.disaggregation.mooncake.conn.get_memory",
            return_value=SimpleNamespace(enable_unified_memory=False),
        )
        memory.start()
        self.addCleanup(memory.stop)
        self.sources = [np.arange(48, dtype=np.int32) + i * 100 for i in range(2)]
        self.state_source = np.arange(48, dtype=np.int32) + 1000
        self.aux_source = np.arange(48, dtype=np.int32) + 2000
        self.destinations = {}
        self.state_destinations = {}
        self.aux_destinations = {}
        self.calls = []
        self.notifications = []
        self.before_transfer = None
        self.failed_session = None
        self.failed_submission = None

    def _manager(self, batch_size=8, sessions=("peer",)):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.attn_tp_size = manager.attn_cp_size = manager.pp_size = 1
        manager.attn_tp_rank = manager.attn_cp_rank = manager.pp_rank = 0
        manager.enable_trace = manager.enable_staging = False
        manager.enable_custom_mem_pool = False
        manager.custom_mem_pool_type = None
        manager.enable_deferred_decode_kv_release = True
        manager.defer_decode_allocation = False
        manager.deferred_bootstrap = None
        manager.max_transfer_batch_indices = 0
        manager.max_request_batch_size = batch_size
        manager.max_request_batch_bytes = 1024
        manager.max_request_batch_descriptors = 32
        manager._request_batch_lock = Lock()
        manager.session_lock = Lock()
        manager.failure_lock = Lock()
        manager._staging_outstanding = defaultdict(int)
        manager._deferred_ack_targets = {}
        manager._deferred_ack_poisoned_rooms = set()
        manager.failed_sessions = set()
        manager.failed_session_probes = {}
        manager.session_failures = defaultdict(int)
        manager.state_layout_rejections = {}
        manager.state_strides_validated = set(sessions)
        manager.request_status = {}
        manager.failure_records = {}
        manager.transfer_infos = {}
        manager.req_to_decode_prefix_len = {}
        manager.decode_kv_args_table = {}
        manager.bootstrap_port = 8998
        manager.kv_args = SimpleNamespace(
            prefill_start_layer=0,
            kv_data_ptrs=[source.ctypes.data for source in self.sources],
            kv_item_lens=[4, 4],
            kv_layer_ids=[],
            aux_data_ptrs=[self.aux_source.ctypes.data],
            aux_item_lens=[4],
            state_types=[StateType.SWA],
            state_data_ptrs=[[self.state_source.ctypes.data]],
            state_item_lens=[[4]],
            state_dim_per_tensor=[[]],
            state_layer_ids=[[]],
        )

        def transfer(session, sources, destinations, lengths):
            self.calls.append((session, sources, destinations, lengths))
            if self.before_transfer is not None:
                self.before_transfer(manager)
            if (
                session == self.failed_session
                or len(self.calls) == self.failed_submission
            ):
                return -1
            for src, dst, length in zip(sources, destinations, lengths, strict=True):
                ctypes.memmove(dst, src, length)
            return 0

        manager.engine = SimpleNamespace(batch_transfer_sync=transfer)
        manager._send_multipart_locked = lambda endpoint, parts, **kwargs: (
            self.notifications.append((endpoint, parts))
        )
        for session in sessions:
            buffers = [np.full(64, -1, dtype=np.int32) for _ in self.sources]
            state = np.full(64, -1, dtype=np.int32)
            aux = np.full(64, -1, dtype=np.int32)
            self.destinations[session] = buffers
            self.state_destinations[session] = state
            self.aux_destinations[session] = aux
            manager.decode_kv_args_table[session] = SimpleNamespace(
                requires_dcp_relayout=False,
                dst_attn_tp_size=1,
                dst_kv_ptrs=[buf.ctypes.data for buf in buffers],
                dst_kv_item_len=4,
                dst_kv_layer_ids=[],
                dst_aux_ptrs=[aux.ctypes.data],
                dst_state_data_ptrs=[[state.ctypes.data]],
                dst_state_item_lens=[[4]],
                dst_state_dim_per_tensor=[[]],
                dst_state_layer_ids=[[]],
            )
        return manager

    def _chunk(self, manager, room, src, dst, *, last=False, sessions=("peer",)):
        manager.request_status[room] = KVPoll.WaitingForInput
        manager.transfer_infos[room] = {
            session: SimpleNamespace(
                room=room,
                is_dummy=False,
                mooncake_session_id=session,
                dst_kv_indices=np.array(dst, dtype=np.int32),
                dst_device_kv_indices=None,
                dst_aux_index=room + 10,
                dst_state_indices=[[room + 5]],
                endpoint="127.0.0.1",
                dst_port=6000 + i,
                required_dst_info_num=len(sessions),
            )
            for i, session in enumerate(sessions)
        }
        return TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array(src, dtype=np.int32),
            index_slice=slice(0, len(src)),
            is_last_chunk=last,
            prefill_aux_index=room if last else None,
            state_indices=[[room]] if last else None,
        )

    def _run(self, manager, chunks):
        queue = FastQueue()
        for chunk in chunks:
            queue.put(chunk)
        queue.put(None)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            manager.transfer_worker(queue, executor)

    def _assert_pages(self, session, src, dst):
        for source, destination in zip(self.sources, self.destinations[session]):
            np.testing.assert_array_equal(destination[dst], source[src])
            untouched = np.ones(destination.size, dtype=bool)
            untouched[dst] = False
            np.testing.assert_array_equal(destination[untouched], -1)

    def test_merges_requests_without_changing_fragmented_page_mapping(self):
        for mla, tensor_indices in (
            (False, False), (True, False), (False, True), (True, True)
        ):
            with self.subTest(mla=mla, tensor_indices=tensor_indices):
                self.calls.clear()
                manager = self._manager()
                manager.is_mla_backend = mla
                chunks = [
                    self._chunk(manager, 1, [1, 2, 5], [11, 12, 15]),
                    self._chunk(manager, 2, [7, 9], [21, 24]),
                ]

                if tensor_indices:
                    for chunk in chunks:
                        chunk.prefill_kv_indices = torch.from_numpy(
                            chunk.prefill_kv_indices
                        )
                        chunk.indices_event = MagicMock()
                        chunk.indices_event.query.return_value = True

                def assert_inflight(manager):
                    self.assertEqual(dict(manager._staging_outstanding), {1: 1, 2: 1})
                    self.assertEqual(self.notifications, [])

                self.before_transfer = assert_inflight
                self._run(manager, chunks)
                self._assert_pages("peer", [1, 2, 5, 7, 9], [11, 12, 15, 21, 24])
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(
                    set(manager.request_status.values()), {KVPoll.WaitingForInput}
                )

    def test_groups_each_destination_session_and_keeps_room_notifications(self):
        manager = self._manager(sessions=("peer", "other"))
        chunks = [
            self._chunk(manager, 1, [1], [11], last=True, sessions=("peer", "other")),
            self._chunk(manager, 2, [2], [12], last=True, sessions=("peer", "other")),
        ]
        self._run(manager, chunks)
        for session in ("peer", "other"):
            self._assert_pages(session, [1, 2], [11, 12])
            np.testing.assert_array_equal(
                self.state_destinations[session][[6, 7]], self.state_source[[1, 2]]
            )
            np.testing.assert_array_equal(
                self.aux_destinations[session][[11, 12]], self.aux_source[[1, 2]]
            )
        # One main-KV call per session, followed by each room's state and aux.
        self.assertEqual([call[0] for call in self.calls[:2]], ["peer", "other"])
        self.assertEqual(len(self.calls), 10)
        self.assertEqual(set(manager.request_status.values()), {KVPoll.Success})
        self.assertCountEqual(
            [(int(parts[0]), int(parts[1])) for _, parts in self.notifications],
            [(1, KVPoll.Success)] * 2 + [(2, KVPoll.Success)] * 2,
        )
        self.assertEqual(dict(manager._staging_outstanding), {})

    def test_limits_and_default_path_preserve_data(self):
        for limit in ("disabled", "bytes", "descriptors", "oversized", "singleton"):
            with self.subTest(limit=limit):
                self.calls.clear()
                manager = self._manager(batch_size=1 if limit == "disabled" else 8)
                if limit == "bytes":
                    manager.max_request_batch_bytes = 8
                elif limit == "descriptors":
                    manager.max_request_batch_descriptors = 2
                elif limit == "oversized":
                    manager.max_request_batch_bytes = 4
                chunks = [
                    self._chunk(manager, 1, [1], [11]),
                    self._chunk(manager, 2, [2], [12]),
                ]
                if limit == "singleton":
                    chunks = chunks[:1]
                self._run(manager, chunks)
                if limit == "singleton":
                    self._assert_pages("peer", [1], [11])
                    self.assertEqual(len(self.calls), 1)
                else:
                    self._assert_pages("peer", [1, 2], [11, 12])
                    self.assertEqual(len(self.calls), 2)

    def test_bounded_batches_split_without_losing_member_results(self):
        manager = self._manager()
        manager.max_request_batch_bytes = 16
        chunks = [self._chunk(manager, i, [i], [i + 10]) for i in range(1, 5)]
        self._run(manager, chunks)
        self._assert_pages("peer", [1, 2, 3, 4], [11, 12, 13, 14])
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(sum(call[3]) == 16 for call in self.calls))
        self.assertEqual(set(manager.request_status.values()), {KVPoll.WaitingForInput})

    def test_later_chunk_of_same_room_is_a_fifo_barrier(self):
        manager = self._manager()
        first = self._chunk(manager, 1, [1], [11, 12], last=False)
        first.index_slice = slice(0, 1)
        last = TransferKVChunk(1, np.array([2], np.int32), slice(1, 2), True, 1, [[1]])
        other = self._chunk(manager, 2, [3], [13])
        self._run(manager, [first, last, other])
        self._assert_pages("peer", [1, 2, 3], [11, 12, 13])
        # Two room-1 KV chunks, then its state and aux, then room 2.
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(manager.request_status[1], KVPoll.Success)
        self.assertEqual(manager.request_status[2], KVPoll.WaitingForInput)
        self.assertEqual(len(self.notifications), 1)

    def test_failed_session_does_not_fail_other_batch_members(self):
        manager = self._manager(sessions=("peer", "other"))
        self.failed_session = "peer"
        chunks = [
            self._chunk(manager, 1, [1], [11], sessions=("peer",)),
            self._chunk(manager, 2, [2], [12], sessions=("peer",)),
            self._chunk(manager, 3, [3], [13], last=True, sessions=("other",)),
        ]
        self._run(manager, chunks)
        self.assertEqual(manager.request_status[1], KVPoll.Failed)
        self.assertEqual(manager.request_status[2], KVPoll.Failed)
        self.assertEqual(manager.request_status[3], KVPoll.Success)
        self._assert_pages("other", [3], [13])
        self.assertTrue(all(np.all(buf == -1) for buf in self.destinations["peer"]))
        self.assertEqual(manager._deferred_ack_poisoned_rooms, {1, 2})

    def test_next_session_cohort_waits_for_earlier_room_completion(self):
        manager = self._manager(sessions=("peer", "other"))
        chunks = [
            self._chunk(manager, 1, [1], [11], last=True),
            self._chunk(manager, 2, [2], [12], last=True),
            self._chunk(manager, 3, [3], [13], last=True, sessions=("other",)),
            self._chunk(manager, 4, [4], [14], last=True, sessions=("other",)),
        ]

        def check_previous_cohort(manager):
            if self.calls[-1][0] == "other":
                self.assertEqual(manager.request_status[1], KVPoll.Success)
                self.assertEqual(manager.request_status[2], KVPoll.Success)

        self.before_transfer = check_previous_cohort
        self._run(manager, chunks)
        self._assert_pages("peer", [1, 2], [11, 12])
        self._assert_pages("other", [3, 4], [13, 14])
        self.assertEqual(set(manager.request_status.values()), {KVPoll.Success})

    def test_aux_failure_after_merged_kv_preserves_other_room_completion(self):
        manager = self._manager()
        chunks = [
            self._chunk(manager, 1, [1], [11], last=True),
            self._chunk(manager, 2, [2], [12], last=True),
        ]
        # Main KV succeeds, then room 1's state succeeds but its aux fails.
        self.failed_submission = 3
        self._run(manager, chunks)
        self._assert_pages("peer", [1, 2], [11, 12])
        self.assertEqual(manager.request_status[1], KVPoll.Failed)
        self.assertEqual(manager.request_status[2], KVPoll.Success)
        self.assertEqual(self.aux_destinations["peer"][11], -1)
        self.assertEqual(self.aux_destinations["peer"][12], self.aux_source[2])
        self.assertCountEqual(
            [(int(parts[0]), int(parts[1])) for _, parts in self.notifications],
            [(1, KVPoll.Failed), (2, KVPoll.Success)],
        )
        self.assertEqual(dict(manager._staging_outstanding), {})

    def test_abort_during_batch_holds_all_chunks_until_write_returns(self):
        manager = self._manager()
        chunks = [
            self._chunk(manager, 1, [1], [11]),
            self._chunk(manager, 2, [2], [12]),
        ]
        acks = []
        manager._deferred_ack_targets[2] = {("127.0.0.1", 6000): object()}
        manager._send_abort_ack = lambda room, target: acks.append(room)

        started, finish = Event(), Event()
        errors = []

        def hold_transfer(manager):
            started.set()
            if not finish.wait(5):
                raise TimeoutError("test did not release the in-flight transfer")

        def worker():
            try:
                self._run(manager, chunks)
            except Exception as exc:
                errors.append(exc)

        self.before_transfer = hold_transfer
        thread = Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(started.wait(5))
            manager.update_status(2, KVPoll.Failed)
            manager._maybe_ack_drained_abort(2)
            self.assertEqual(acks, [])
            self.assertEqual(dict(manager._staging_outstanding), {1: 1, 2: 1})
        finally:
            finish.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self._assert_pages("peer", [1, 2], [11, 12])
        self.assertEqual(acks, [2])
        self.assertEqual(manager.request_status[2], KVPoll.Failed)

    def test_skipping_one_aborted_chunk_does_not_ack_its_remaining_chunk(self):
        manager = self._manager()
        first = self._chunk(manager, 1, [1], [11, 12])
        first.index_slice = slice(0, 1)
        other = self._chunk(manager, 2, [3], [13])
        last = TransferKVChunk(
            1, np.array([2], np.int32), slice(1, 2), False, None, None
        )
        acks = []
        manager._deferred_ack_targets[1] = {("127.0.0.1", 6000): object()}
        manager._send_abort_ack = lambda room, target: acks.append(room)

        def abort(manager):
            self.assertEqual(manager._staging_outstanding[1], 2)
            manager.update_status(1, KVPoll.Failed)
            manager._maybe_ack_drained_abort(1)
            self.assertEqual(acks, [])

        self.before_transfer = abort
        self._run(manager, [first, other, last])
        self._assert_pages("peer", [1, 3], [11, 13])
        self.assertEqual(acks, [1])
        self.assertNotIn(1, manager._staging_outstanding)
        self.assertEqual(len(self.calls), 1)

    def test_fallback_event_and_dcp_do_not_overtake_earlier_chunks(self):
        for special in ("event", "dcp", "pending_indices"):
            with self.subTest(special=special):
                manager = self._manager()
                first = self._chunk(manager, 1, [1], [11])
                second = self._chunk(manager, 2, [2], [12])
                if special == "event":
                    first.wait_event = object()
                elif special == "dcp":
                    manager.decode_kv_args_table["peer"].requires_dcp_relayout = True
                else:
                    first.indices_event = MagicMock()
                    first.indices_event.query.return_value = False
                results = manager._send_request_batch([first, second])
                self.assertEqual(results, {})
                self.assertEqual(self.calls, [])
                self.assertTrue(
                    all(np.all(buf == -1) for buf in self.destinations["peer"])
                )

    def test_queue_drain_preserves_shutdown_sentinel_and_fifo(self):
        queue = FastQueue()
        for item in (1, 2, None, 3):
            queue.put(item)
        self.assertEqual(queue.drain(8), [1, 2])
        self.assertIsNone(queue.get())
        self.assertEqual(queue.get(), 3)
        self.assertEqual(queue.drain(8), [])

    def test_special_submission_paths_keep_legacy_worker(self):
        for settings in (
            {"enable_staging": True},
            {"enable_trace": True},
            {"max_transfer_batch_indices": 1},
            {"enable_custom_mem_pool": True, "custom_mem_pool_type": "NVLINK"},
            {"enable_custom_mem_pool": True, "custom_mem_pool_type": "BAREX"},
        ):
            with self.subTest(settings=settings):
                manager = self._manager()
                manager.__dict__.update(settings)
                self.assertFalse(manager._request_batching_enabled())
        manager = self._manager()
        manager.enable_custom_mem_pool = True
        manager.custom_mem_pool_type = "INTRA_NODE_NVLINK"
        self.assertTrue(manager._request_batching_enabled())


if __name__ == "__main__":
    unittest.main()
