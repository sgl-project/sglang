import threading
import unittest
from collections import defaultdict
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    MooncakeKVSender,
    TransferKVChunk,
    _LayerPipelineRequestDispatch,
    split_layer_groups,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_DEFAULT_LAYER_RANGE = object()


class _QueueDone(Exception):
    pass


class _OneShotQueue:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def get(self):
        if not self.chunks:
            raise _QueueDone("stop test worker")
        return self.chunks.pop(0)


class _FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _run_worker(testcase, mgr, chunks):
    with testcase.assertRaises(RuntimeError):
        MooncakeKVManager.transfer_worker(
            mgr, _OneShotQueue(chunks), executor=None, staging_buffer=None
        )


def _chunk(
    room, group_id=0, *, last=True, total=1, layer_range=_DEFAULT_LAYER_RANGE, **kwargs
):
    if layer_range is _DEFAULT_LAYER_RANGE:
        layer_range = (group_id * 4, (group_id + 1) * 4)
    return TransferKVChunk(
        room=room,
        prefill_kv_indices=np.array([], dtype=np.int32),
        index_slice=slice(0, 0),
        is_last_chunk=last,
        prefill_aux_index=42 if last else None,
        state_indices=None,
        layer_group_id=group_id,
        layer_range=layer_range,
        total_layer_groups=total,
        kv_dtype="auto",
        total_chunks_in_request=total if last else None,
        **kwargs,
    )


class _WorkerMgr:
    def __init__(self, room=7, num_dst=1):
        self.request_status = {room: KVPoll.WaitingForInput}
        self.session_lock = threading.RLock()
        self.session_failures = defaultdict(int)
        self.failed_sessions = set()
        self.enable_staging = False
        self.layer_pipeline_enabled = True
        self.bootstrap_port = 0
        self.attn_tp_rank = self.pp_rank = self.attn_cp_rank = 0
        self.pp_size = self.attn_cp_size = self.attn_tp_size = 1
        self.is_mla_backend = True
        self.layer_pipeline_progress = {}
        self.layer_pipeline_progress_lock = threading.Lock()
        self._lp_metrics_lock = threading.Lock()
        self._lp_chunks_total = 0
        self._lp_chunks_periodic = 0
        self._lp_chunk_ms_samples = []
        self._LP_SAMPLE_BUFFER_CAP = 4096
        self.sync_status_calls = []
        self.send_kvcache_calls = 0
        self.send_aux_calls = 0

        sessions = {}
        self.decode_kv_args_table = {}
        for i in range(num_dst):
            sid = f"sess-{i}"
            sessions[sid] = SimpleNamespace(
                is_dummy=False,
                mooncake_session_id=sid,
                endpoint=f"10.0.0.{i + 1}",
                dst_port=20000 + i,
                room=room,
                dst_kv_indices=np.array([], dtype=np.int32),
                dst_aux_index=0,
                dst_state_indices=[],
                required_dst_info_num=num_dst,
            )
            self.decode_kv_args_table[sid] = SimpleNamespace(
                dst_kv_ptrs=[],
                dst_aux_ptrs=[],
                dst_state_data_ptrs=[],
                dst_attn_tp_size=1,
                dst_num_main_kv_layers=None,
                staging=None,
            )
        self.transfer_infos = {room: sessions}

        for name in (
            "_record_chunk_done",
            "_record_aux_sent",
            "_maybe_sync_success_locked",
            "_dispatch_sync_outside_lock",
            "_clear_layer_pipeline_progress",
            "_record_layer_group_metric",
            "pop_layer_pipeline_metrics",
        ):
            setattr(self, name, MethodType(getattr(MooncakeKVManager, name), self))

    def check_status(self, room):
        return self.request_status.get(room, KVPoll.Bootstrapping)

    def update_status(self, room, status):
        self.request_status[room] = status

    def record_failure(self, room, msg):
        pass

    def sync_status_to_decode_endpoint(self, *args, **kwargs):
        self.sync_status_calls.append((args, kwargs))

    def _try_create_staging_strategy(self, *_args, **_kwargs):
        return None

    def send_kvcache(self, *_args, **_kwargs):
        self.send_kvcache_calls += 1
        return 0

    def send_aux(self, *_args, **_kwargs):
        self.send_aux_calls += 1
        return 0

    def maybe_send_extra(self, *_args, **_kwargs):
        return 0


class _FakeSendMgr:
    def __init__(self, *, num_layers=8, group_size=4):
        self.layer_pipeline_enabled = True
        self.layer_group_size = group_size
        self.layer_pipeline_min_prefill_len = 0
        self.enable_staging = False
        self.is_mla_backend = False
        self.is_dummy_cp_rank = False
        self.enable_all_cp_ranks_for_transfer = False
        self.attn_cp_size = 1
        self.attn_cp_rank = 0
        self.attn_dp_rank = 0
        self.server_args = SimpleNamespace(
            kv_cache_dtype="auto",
            dp_size=1,
            load_balance_method="follow_bootstrap_room",
        )
        self.request_status = {}
        self.transfer_queues = [_FakeQueue()]
        self.session_id = "127.0.0.1:1234"
        self.transfer_infos = {
            7: {
                self.session_id: SimpleNamespace(
                    is_dummy=False, mooncake_session_id=self.session_id
                )
            }
        }
        self.decode_kv_args_table = {
            self.session_id: SimpleNamespace(
                layer_pipeline_enabled=True,
                layer_group_size=group_size,
                kv_dtype="auto",
            )
        }
        self.kv_args = SimpleNamespace(
            page_size=1,
            kv_data_ptrs=[0] * num_layers,
            prefill_num_main_kv_layers=None,
            engine_rank=0,
        )
        self._num_layers = num_layers

    def update_status(self, room, status):
        self.request_status[room] = status

    def check_status(self, room):
        return self.request_status.get(room, KVPoll.Bootstrapping)

    def local_num_kv_layers(self):
        return self._num_layers

    def use_layer_cp_shard_for_transfer(self):
        return (
            self.layer_pipeline_enabled
            and self.enable_all_cp_ranks_for_transfer
            and self.attn_cp_size > 1
        )

    def _room_supports_layer_pipeline(self, room):
        info = self.decode_kv_args_table[self.session_id]
        return (
            room in self.transfer_infos
            and info.layer_pipeline_enabled
            and info.layer_group_size == self.layer_group_size
            and info.kv_dtype == self.server_args.kv_cache_dtype
        )

    def add_transfer_request(
        self,
        bootstrap_room,
        kv_indices,
        index_slice,
        is_last_chunk,
        aux_index=None,
        state_indices=None,
        layer_group_id=0,
        layer_range=None,
        total_layer_groups=1,
        total_chunks_in_request=None,
        transfer_event=None,
        skip_aux_rdma=False,
        trace_ctx=None,
    ):
        self.transfer_queues[0].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
                layer_group_id=layer_group_id,
                layer_range=layer_range,
                total_layer_groups=total_layer_groups,
                kv_dtype=self.server_args.kv_cache_dtype,
                total_chunks_in_request=total_chunks_in_request,
                transfer_event=transfer_event,
                skip_aux_rdma=skip_aux_rdma,
            )
        )


class _FakeEvent:
    def __init__(self):
        self.synchronize_calls = 0

    def record(self):
        pass

    def synchronize(self):
        self.synchronize_calls += 1


def _make_hook_mgr(
    *,
    num_layers=8,
    group_size=4,
    prefill_start_layer=0,
    num_main=None,
    cp_size=1,
    cp_rank=0,
):
    captured = []
    mgr = SimpleNamespace(
        layer_group_size=group_size,
        local_num_kv_layers=lambda: num_layers,
        add_transfer_request=lambda **kw: captured.append(kw),
        kv_args=SimpleNamespace(
            prefill_start_layer=prefill_start_layer,
            prefill_num_main_kv_layers=num_main,
        ),
        is_dummy_cp_rank=False,
        attn_cp_size=cp_size,
        attn_cp_rank=cp_rank,
        use_layer_cp_shard_for_transfer=lambda: cp_size > 1,
    )
    mgr.make_layer_pipeline_hook = MethodType(
        MooncakeKVManager.make_layer_pipeline_hook, mgr
    )
    sender = SimpleNamespace(_hook_enqueued_chunks=0, bootstrap_room=900)
    dispatch = [
        _LayerPipelineRequestDispatch(
            room=900,
            sender=sender,
            page_indices=np.array([0], dtype=np.int32),
            index_slice=slice(0, 1),
        )
    ]
    return mgr, dispatch, captured, sender


def _fire_hook(mgr, dispatch, layer_ids):
    import torch

    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True))
    with patch.object(torch.cuda, "Event", _FakeEvent):
        hook = mgr.make_layer_pipeline_hook(dispatch)
        for layer_id in layer_ids:
            hook(layer_id, fb)


class TestLayerPipelineBasics(CustomTestCase):
    def test_split_layer_groups(self):
        self.assertEqual(split_layer_groups(0, 4), [])
        self.assertEqual(split_layer_groups(8, 0), [(0, 8)])
        self.assertEqual(split_layer_groups(8, 99), [(0, 8)])
        self.assertEqual(split_layer_groups(10, 4), [(0, 4), (4, 8), (8, 10)])

    def test_kv_args_register_info_parses_num_main_slot(self):
        msg = [
            b"None", b"127.0.0.1", b"2345", b"127.0.0.1:1234",
            b"", b"", b"", b"0", b"1", b"64", b"", b"",
            b"1", b"4", b"fp8_e4m3", b"", b"", b"64",
        ]
        info = KVArgsRegisterInfo.from_zmq(msg)
        self.assertTrue(info.layer_pipeline_enabled)
        self.assertEqual(info.layer_group_size, 4)
        self.assertEqual(info.kv_dtype, "fp8_e4m3")
        self.assertEqual(info.dst_num_main_kv_layers, 64)
        self.assertIsNone(info.staging)

class TestLayerPipelineWorker(CustomTestCase):
    def test_worker_syncs_after_all_chunks_and_aux(self):
        room = 100
        mgr = _WorkerMgr(room=room)
        chunks = [
            _chunk(room, 0, last=False, total=2),
            _chunk(room, 1, last=True, total=2),
        ]
        _run_worker(self, mgr, chunks)

        self.assertEqual(mgr.send_kvcache_calls, 2)
        self.assertEqual(mgr.send_aux_calls, 1)
        self.assertEqual(mgr.request_status[room], KVPoll.Success)
        self.assertEqual(len(mgr.sync_status_calls), 1)
        self.assertEqual(mgr.sync_status_calls[0][0][3], KVPoll.Success)

    def test_worker_does_not_sync_when_a_chunk_is_missing(self):
        room = 101
        mgr = _WorkerMgr(room=room)
        chunks = [
            _chunk(room, 0, last=False, total=3),
            _chunk(room, 2, last=True, total=3),
        ]
        _run_worker(self, mgr, chunks)

        self.assertEqual(mgr.send_kvcache_calls, 2)
        self.assertEqual(mgr.send_aux_calls, 1)
        self.assertNotEqual(mgr.request_status[room], KVPoll.Success)
        self.assertEqual(mgr.sync_status_calls, [])

    def test_event_is_synchronized_before_send(self):
        room = 102
        mgr = _WorkerMgr(room=room)
        event = _FakeEvent()
        _run_worker(self, mgr, [_chunk(room, transfer_event=event)])
        self.assertGreaterEqual(event.synchronize_calls, 1)
        self.assertEqual(mgr.send_kvcache_calls, 1)

    def test_skip_aux_rdma_still_closes_watermark(self):
        room = 103
        mgr = _WorkerMgr(room=room)
        aux_only = _chunk(
            room,
            layer_range=None,
            skip_aux_rdma=True,
            total=1,
        )
        _run_worker(self, mgr, [aux_only])

        self.assertEqual(mgr.send_kvcache_calls, 0)
        self.assertEqual(mgr.send_aux_calls, 0)
        self.assertEqual(mgr.request_status[room], KVPoll.Success)
        self.assertEqual(len(mgr.sync_status_calls), 1)


class TestLayerPipelineHook(CustomTestCase):
    def test_hook_uses_ceil_groups_and_local_ranges(self):
        mgr, dispatch, captured, sender = _make_hook_mgr(
            num_layers=10,
            group_size=4,
            prefill_start_layer=32,
        )
        _fire_hook(mgr, dispatch, range(32, 42))

        self.assertEqual([c["layer_range"] for c in captured], [(0, 4), (4, 8), (8, 10)])
        self.assertEqual([c["total_layer_groups"] for c in captured], [3, 3, 3])
        self.assertEqual(sender._hook_enqueued_chunks, 3)

    def test_hook_main_layers_do_not_include_appended_draft_slot(self):
        mgr, dispatch, captured, sender = _make_hook_mgr(
            num_layers=79,
            num_main=78,
            group_size=4,
        )
        _fire_hook(mgr, dispatch, range(78))

        self.assertEqual(len(captured), 20)
        self.assertEqual(captured[-1]["layer_range"], (76, 78))
        self.assertEqual(sender._hook_enqueued_chunks, 20)

    def test_cp_layer_shard_partitions_layer_groups(self):
        owned_by_rank = []
        for rank in range(4):
            mgr, dispatch, captured, _ = _make_hook_mgr(
                num_layers=12,
                group_size=4,
                cp_size=4,
                cp_rank=rank,
            )
            _fire_hook(mgr, dispatch, range(12))
            owned_by_rank.append({c["layer_group_id"] for c in captured})

        self.assertEqual(owned_by_rank, [{0}, {1}, {2}, set()])
        self.assertEqual(set().union(*owned_by_rank), {0, 1, 2})


class TestLayerPipelineSender(CustomTestCase):
    def _make_sender(self, *, num_layers=8, num_main=None):
        mgr = _FakeSendMgr(num_layers=num_layers)
        mgr.kv_args.prefill_num_main_kv_layers = num_main
        sender = MooncakeKVSender(
            mgr,
            bootstrap_addr="bootstrap-host:30002",
            bootstrap_room=7,
            dest_tp_ranks=[0],
            pp_rank=0,
        )
        sender.init(num_kv_indices=3, aux_index=7)
        return mgr, sender

    def test_hook_mode_without_hook_chunks_fails_loudly(self):
        _, sender = self._make_sender()
        sender._hook_handled_in_current_send = True

        with self.assertRaisesRegex(RuntimeError, "Layer-pipeline contract violated"):
            sender.send(np.array([0, 1, 2], dtype=np.int32))
        self.assertFalse(sender._hook_handled_in_current_send)

    def test_draft_kv_counts_toward_aux_finalizer_watermark(self):
        mgr, sender = self._make_sender(num_layers=79, num_main=78)
        sender._hook_handled_in_current_send = True
        sender._hook_enqueued_chunks = 20

        sender.send_draft_kv(np.array([0, 1, 2], dtype=np.int32))
        sender.send(np.array([0, 1, 2], dtype=np.int32))

        chunks = mgr.transfer_queues[0].items
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].layer_range, (78, 79))
        self.assertFalse(chunks[0].is_last_chunk)
        self.assertTrue(chunks[1].is_last_chunk)
        self.assertIsNone(chunks[1].layer_range)
        self.assertEqual(chunks[1].total_chunks_in_request, 22)


if __name__ == "__main__":
    unittest.main()
