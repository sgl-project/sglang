import threading
import unittest
from collections import Counter, defaultdict, deque
from types import MethodType
from types import SimpleNamespace as NS

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.staging_handler import DecodeStagingHandler
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class RetryProtocol(CustomTestCase):
    def test_duplicate_source_cannot_complete_pp_fanin(self):
        for pp in (2, 8):
            with self.subTest(pp=pp):
                receiver = NS(
                    prefill_info=NS(attn_tp_size=1, pp_size=pp),
                    chunk_staging_infos=[(0, 0, 0, 8192, 4)],
                )
                handler = object.__new__(DecodeStagingHandler)
                handler.kv_manager = NS(pp_size=1)
                handler.decode_tp = 8
                handler._staging_lifecycle = threading.Condition()
                handler._room_to_decode_req = {7: NS(_staging_failed=False)}
                handler._room_to_receiver = {7: receiver}
                handler._room_tokens = {7: object()}
                handler._writer_counts = {}
                handler._pending_scatters = deque()
                handler._defer_scatter = True
                manager = object.__new__(CommonKVManager)
                manager.request_status = {7: KVPoll.Transferring}
                manager._staging_handler = handler
                messages = []
                sender = NS(
                    _send_multipart_locked=lambda endpoint, parts, **kw: (
                        messages.append(parts)
                    )
                )
                for rank in range(pp):
                    sender._prefill_unique_rank = lambda rank=rank: rank
                    CommonKVManager.send_chunk_ready(
                        sender,
                        room=7,
                        chunk_idx=0,
                        page_start=0,
                        num_pages=4,
                        writer_id="same-destination-session",
                        targets=[("127.0.0.1", 9000)],
                    )
                manager.handle_chunk_ready(messages[0])
                manager.handle_chunk_ready(messages[0])
                self.assertEqual(
                    len(handler._pending_scatters),
                    0,
                    "one writer retransmit falsely completed PP fan-in",
                )
                for message in messages[1:]:
                    manager.handle_chunk_ready(message)
                self.assertEqual(
                    len(handler._pending_scatters),
                    1,
                    "all distinct PP writers must complete exactly one chunk",
                )
                for message in messages:
                    manager.handle_chunk_ready(message)
                self.assertEqual(
                    len(handler._pending_scatters),
                    1,
                    "late duplicate notifications queued the same chunk again",
                )

    def test_legacy_frames_preserve_multiwriter_arrival_counting(self):
        handler = object.__new__(DecodeStagingHandler)
        handler.kv_manager = NS(pp_size=1)
        handler.decode_tp = 1
        receiver = NS(prefill_info=NS(attn_tp_size=2, pp_size=1))
        handler._staging_lifecycle = threading.Condition()
        handler._room_to_decode_req = {7: NS(_staging_failed=False)}
        handler._room_to_receiver = {7: receiver}
        handler._room_tokens = {7: object()}
        handler._writer_counts = {}
        handler._pending_scatters = deque()
        handler._defer_scatter = True
        manager = object.__new__(CommonKVManager)
        manager.request_status = {7: KVPoll.Transferring}
        manager._staging_handler = handler
        frame = [b"CHUNK_READY", b"7", b"0", b"0", b"4", b"destination"]
        manager.handle_chunk_ready(frame)
        self.assertFalse(handler._pending_scatters)
        manager.handle_chunk_ready(frame)
        self.assertEqual(len(handler._pending_scatters), 1)
        manager.handle_chunk_ready(frame)
        self.assertEqual(len(handler._pending_scatters), 1)

    def test_destination_backpressure_preserves_earlier_progress(self):
        for last, fail_aux in ((False, False), (True, False), (True, True)):
            sessions, blocked, delays = "AB", "B", 1
            with self.subTest(last=last, fail_aux=fail_aux):
                chunk = TransferKVChunk(
                    7, np.array([0], dtype=np.int32), slice(0, 1), last, 0, None
                )
                items = deque([chunk])
                queue = NS(
                    get=lambda: items.popleft() if items else None, put=items.append
                )
                sent = []
                aux = []
                conclusions = []
                finished = []
                deferred = [0]

                def check(req, *args, **kwargs):
                    if req.mooncake_session_id == blocked and deferred[0] < delays:
                        deferred[0] += 1
                        return False, 0, 0, 0, 16
                    return True, 0, 0, 0, 16

                strategy = NS(
                    check_ready=check,
                    transfer=lambda session, *args: sent.append(session) or 0,
                )

                def send_aux(req, *args):
                    session = req.mooncake_session_id
                    aux.append(session)
                    return (
                        -1 if fail_aux and session == "A" and aux.count("A") == 1 else 0
                    )

                peers = {
                    s: NS(
                        is_dummy=False,
                        mooncake_session_id=s,
                        dst_device_kv_indices=None,
                        dst_kv_indices=np.array([10], dtype=np.int32),
                        room=7,
                        endpoint="127.0.0.1",
                        dst_port=9000 + i,
                        required_dst_info_num=len(sessions),
                    )
                    for i, s in enumerate(sessions)
                }
                registrations = {
                    s: NS(
                        requires_dcp_relayout=False,
                        dst_kv_ptrs=[1000],
                        dst_kv_layer_ids=[0],
                        dst_kv_item_len=16,
                        dst_attn_tp_size=8,
                        dst_aux_ptrs=[2000],
                        dst_aux_item_lens=[16],
                        staging_base_ptr=4096,
                        staging_total_size=8192,
                    )
                    for s in peers
                }
                manager = NS(
                    enable_trace=False,
                    enable_staging=True,
                    enable_deferred_decode_kv_release=True,
                    _staging_outstanding=defaultdict(int),
                    request_status={7: KVPoll.WaitingForInput},
                    transfer_infos={7: peers},
                    session_lock=threading.Lock(),
                    failed_sessions=set(),
                    state_layout_rejections={},
                    decode_kv_args_table=registrations,
                    kv_args=NS(kv_data_ptrs=[1]),
                    is_mla_backend=False,
                    is_hybrid_mla_backend=False,
                    attn_tp_size=4,
                    _try_create_staging_strategy=lambda buffer: strategy,
                    _staging_ctx=NS(
                        watermark_cv=threading.Condition(),
                        prefetch_requested=set(),
                        prefetched_rooms=set(),
                    ),
                    _get_dsa_cache_transfer_skip_flags=lambda registration: (
                        False,
                        False,
                    ),
                    validate_aux_buffers=lambda *args: None,
                    send_chunk_ready=lambda **kwargs: None,
                    send_aux=send_aux,
                    _poison_unsettled_send=lambda room: None,
                    _maybe_ack_drained_abort=lambda room: None,
                    note_send_finished=lambda *args: finished.append(args),
                    req_to_decode_prefix_len={},
                    bootstrap_port=8998,
                )
                manager.check_status = lambda room: manager.request_status[room]

                def conclude(**kwargs):
                    conclusions.append(kwargs)
                    manager.request_status[7] = kwargs["status"]

                manager.conclude_transfer = conclude
                manager._do_staging_transfer = MethodType(
                    MooncakeKVManager._do_staging_transfer, manager
                )
                MooncakeKVManager.transfer_worker(
                    manager, queue, None, staging_buffer=object()
                )
                if last:
                    self.assertEqual(
                        conclusions[0]["status"],
                        KVPoll.Failed if fail_aux else KVPoll.Success,
                        "retry lost the earlier destination failure",
                    )
                self.assertEqual(
                    Counter(sent),
                    Counter(sessions),
                    "watermark retry replayed an already delivered destination",
                )
                self.assertEqual(Counter(aux), Counter(sessions) if last else Counter())
                self.assertEqual(
                    len(finished),
                    1,
                    "the retried chunk must release its send record once",
                )
                self.assertEqual(len(conclusions), int(last))
                self.assertEqual(deferred[0], delays)
                self.assertEqual(manager._staging_outstanding.get(7, 0), 0)
                if last:
                    self.assertFalse(manager._staging_outstanding)


if __name__ == "__main__":
    unittest.main(verbosity=2)
