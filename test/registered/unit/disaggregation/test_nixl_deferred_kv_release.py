"""Deferred decode-side KV release on the NIXL backend.

NIXL posts transfers asynchronously and the worker polls their handles. The
bootstrap thread records the ACK target while a transfer is outstanding; only
the worker can send that ACK after every handle reaches DONE.
"""

import threading
import time
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from test_deferred_decode_kv_release import (
    ABORT_GENERATION,
    DeferredAbortNotificationScenarios,
    TaggedAbortNotificationScenarios,
    WorkerFailureAbortScenarios,
)

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import AckTarget
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.nixl import conn as nixl_conn
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

ROOM = 11
DECODE_IP = "10.0.0.3"
DECODE_PORT = 6000


def _manager(
    *, enabled: bool = True, status: KVPoll | None = KVPoll.WaitingForInput
) -> NixlKVManager:
    manager = NixlKVManager.__new__(NixlKVManager)
    manager.enable_deferred_decode_kv_release = enabled
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager._staging_outstanding = {}
    manager.request_status = {} if status is None else {ROOM: status}
    manager._sent = []
    manager._send_abort_ack = lambda *args: manager._sent.append(args)
    manager.record_failure = MagicMock()
    manager.update_status = MagicMock(
        side_effect=lambda room, value: manager.request_status.__setitem__(room, value)
    )
    manager.check_status = lambda room: manager.request_status[room]
    return manager


def _chunk() -> TransferKVChunk:
    return TransferKVChunk(
        room=ROOM,
        prefill_kv_indices=np.array([1], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last_chunk=False,
        prefill_aux_index=None,
        state_indices=None,
    )


_HANDLE = object()


class _ProcForeverAgent:
    """Fake NIXL agent whose submitted handle stays healthy but non-terminal."""

    def __init__(self):
        self.state = "PROC"
        self.reads = 0

    def check_xfer_state(self, handle):
        self.reads += 1
        return self.state


def _wait_until_polling(test_case, agent: _ProcForeverAgent) -> None:
    deadline = time.monotonic() + 10
    while agent.reads == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    test_case.assertGreater(agent.reads, 0, "worker never started polling handles")


def _abort_worker_manager(agent: _ProcForeverAgent) -> NixlKVManager:
    """Manager able to run transfer_worker() up to the handle wait with one
    fake handle produced by a mocked send_kvcache."""
    manager = _manager()
    manager.agent = agent
    manager.enable_staging = False
    manager._staging_ctx = None
    manager._staging_outstanding = defaultdict(int)
    manager.exceptions = {}
    manager.is_mla_backend = False
    manager.is_hybrid_mla_backend = False
    manager.attn_tp_size = 1
    manager.transfer_source_rank = 0
    manager.kv_args = SimpleNamespace(kv_data_ptrs=[0x1000])
    manager.decode_kv_args_table = {
        "agent": SimpleNamespace(
            decode_tp_size=1,
            requires_dcp_relayout=False,
            kv_xfer_segments=None,
            dst_homogeneous_mem_kind="VRAM",
            dst_kv_ptrs=[0x2000],
            gpu_id=0,
        )
    }
    manager.transfer_infos = {
        ROOM: {
            "req": SimpleNamespace(
                room=ROOM,
                is_dummy=False,
                agent_name="agent",
                dst_kv_indices=np.array([0], dtype=np.int64),
                dst_aux_index=np.array([0], dtype=np.int64),
                decode_prefix_len=0,
            )
        }
    }
    manager.send_kvcache = MagicMock(return_value=_HANDLE)
    return manager


class TestNixlAbortNotification(
    DeferredAbortNotificationScenarios,
    TaggedAbortNotificationScenarios,
    WorkerFailureAbortScenarios,
    CustomTestCase,
):
    room = ROOM
    decode_ip = DECODE_IP
    decode_port = DECODE_PORT

    def _make_abort_manager(self, status: KVPoll | None):
        return _manager(status=status)

    def _provoke_worker_failure(self, manager) -> None:
        manager.enable_staging = False
        manager._staging_ctx = None
        manager._staging_outstanding = defaultdict(int)
        manager.transfer_infos = {ROOM: {}}
        manager.exceptions = {}
        # A handle that never settles may still be writing, so the chunk stays
        # counted and the room is poisoned instead of notified.
        manager.conclude_failure = MagicMock()
        manager._await_handles = MagicMock(return_value=(False, True, False))
        manager.check_status = MagicMock(side_effect=RuntimeError("worker error"))
        queue = SimpleNamespace(
            get=MagicMock(side_effect=(_chunk(), KeyboardInterrupt()))
        )

        with patch("sglang.srt.disaggregation.nixl.conn.logger.exception"):
            with self.assertRaises(KeyboardInterrupt):
                manager.transfer_worker(queue)
        manager.check_status = lambda room: manager.request_status[room]

    def test_worker_skip_between_failure_and_target_registration_still_acks(self):
        manager = _manager()
        manager._staging_outstanding[ROOM] = 1
        update_status = manager.update_status.side_effect

        def fail_then_skip(room, status):
            update_status(room, status)
            # Window 1: the worker skips after Failed but before target
            # registration, so the bootstrap thread must observe zero and ACK.
            manager._staging_outstanding.pop(room, None)
            manager._maybe_ack_drained_abort(room)

        manager.update_status = MagicMock(side_effect=fail_then_skip)

        claimed = manager._handle_abort_notification(self._abort_message())

        self.assertTrue(claimed)
        self.assertEqual(
            manager._sent,
            [(ROOM, AckTarget(DECODE_IP, DECODE_PORT, ABORT_GENERATION))],
        )
        self.assertEqual(manager._deferred_ack_targets, {})

    def test_settled_worker_failure_uncounts_chunk_and_acks(self):
        # Every handle settled, so the writes are over: decode is told via
        # conclude_failure, and its ABORT, which lands while the chunk is
        # still counted, must be ACKed once the worker uncounts the chunk.
        manager = _manager()
        manager.enable_staging = False
        manager._staging_ctx = None
        manager._staging_outstanding = defaultdict(int)
        manager.transfer_infos = {ROOM: {}}
        manager.exceptions = {}
        manager.check_status = MagicMock(side_effect=RuntimeError("worker error"))

        def notify_decode(**kwargs):
            manager.check_status = lambda room: manager.request_status[room]
            self._dispatch_abort(manager)

        manager.conclude_failure = MagicMock(side_effect=notify_decode)
        queue = SimpleNamespace(
            get=MagicMock(side_effect=(_chunk(), KeyboardInterrupt()))
        )

        with patch("sglang.srt.disaggregation.nixl.conn.logger.exception"):
            with self.assertRaises(KeyboardInterrupt):
                manager.transfer_worker(queue)

        manager.conclude_failure.assert_called_once()
        self.assertEqual(manager._staging_outstanding[ROOM], 0)
        self.assertNotIn(ROOM, manager._deferred_ack_poisoned_rooms)
        self.assertEqual(
            manager._sent,
            [(ROOM, AckTarget(DECODE_IP, DECODE_PORT, ABORT_GENERATION))],
        )
        self.assertEqual(manager._deferred_ack_targets, {})

    def _prepare_worker(self, manager, num_peers: int) -> None:
        """Wire transfer_worker to MLA decode peers over the prepped path."""
        manager.enable_staging = False
        manager._staging_ctx = None
        manager._staging_outstanding = defaultdict(int)
        manager.exceptions = {}
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.attn_tp_size = 1
        manager.transfer_source_rank = 0
        manager.src_mem_kind = "VRAM"
        manager._num_slots_src = 4
        manager.kv_args = SimpleNamespace(kv_data_ptrs=[0x1000], kv_item_lens=[16])
        manager.prep_handles = {"": "src-prep"}
        manager.decode_kv_args_table = {}
        manager.transfer_infos = {ROOM: {}}
        for i in range(num_peers):
            agent_name = f"decode-{i}"
            manager.prep_handles[agent_name] = f"dst-prep-{i}"
            manager.decode_kv_args_table[agent_name] = SimpleNamespace(
                decode_tp_size=1,
                requires_dcp_relayout=False,
                kv_xfer_segments=None,
                dst_homogeneous_mem_kind="VRAM",
                dst_kv_ptrs=[0x2000],
                dst_num_slots=None,
                gpu_id=0,
            )
            manager.transfer_infos[ROOM][agent_name] = SimpleNamespace(
                room=ROOM,
                is_dummy=False,
                agent_name=agent_name,
                dst_kv_indices=np.array([1], dtype=np.int32),
            )
        manager.conclude_failure = MagicMock(
            side_effect=lambda **kw: manager.update_status(
                kw["bootstrap_room"], KVPoll.Failed
            )
        )
        manager._reload_invalidated_peers = MagicMock()
        manager.agent = MagicMock()
        manager.agent.make_prepped_xfer.side_effect = lambda *args: object()

    def test_rejected_post_never_acks(self):
        """NIXL raises when it rejects a post and does not wait for writes it
        already put on the wire, so after the first decode peer's write is DONE
        and the second peer's post raises, neither an ABORT that arrived
        mid-transfer nor a later one may be ACKed."""
        manager = _manager(status=KVPoll.Transferring)
        self._prepare_worker(manager, num_peers=2)
        posts = iter(("PROC", nixl_conn._NIXL_TRANSPORT_ERRORS[0]("NIXL_ERR_BACKEND")))

        def post(handle):
            result = next(posts)
            if isinstance(result, Exception):
                self._dispatch_abort(manager)
                raise result
            return result

        manager.agent.transfer.side_effect = post
        manager.agent.check_xfer_state.return_value = "DONE"
        queue = SimpleNamespace(
            get=MagicMock(side_effect=(_chunk(), KeyboardInterrupt()))
        )

        with self.assertRaises(KeyboardInterrupt):
            manager.transfer_worker(queue)
        self._dispatch_abort(manager)

        self.assertEqual(manager.agent.transfer.call_count, 2)
        manager.conclude_failure.assert_called_once()
        self.assertEqual(manager._staging_outstanding[ROOM], 0)
        self.assertEqual(manager._sent, [])

    def test_await_handles_wakes_when_room_aborted_mid_wait(self):
        # https://github.com/sgl-project/sglang/issues/44369 - a healthy but
        # non-terminal handle must not keep _await_handles() spinning after
        # the room is aborted.
        agent = _ProcForeverAgent()
        manager = _manager()
        manager.agent = agent
        returned = threading.Event()
        result = {}

        def wait_for_handle():
            result["value"] = manager._await_handles(
                [_HANDLE], failure_seen=False, room=ROOM
            )
            returned.set()

        thread = threading.Thread(target=wait_for_handle, daemon=True)
        thread.start()
        _wait_until_polling(self, agent)
        # Sanity: without the abort the wait really does block.
        self.assertFalse(returned.wait(0.2))

        self._dispatch_abort(manager)

        self.assertTrue(returned.wait(5))
        self.assertEqual(result["value"], (False, False, True))
        thread.join(5)
        self.assertFalse(thread.is_alive())

    def test_abort_during_handle_wait_releases_worker_and_acks(self):
        # https://github.com/sgl-project/sglang/issues/44369 - aborting a
        # room while its transfer worker waits on a healthy non-terminal
        # handle must release the worker, drain the staging count, and fire
        # the deferred abort ACK.
        agent = _ProcForeverAgent()
        manager = _abort_worker_manager(agent)
        queue = SimpleNamespace(
            get=MagicMock(side_effect=(_chunk(), KeyboardInterrupt()))
        )
        worker_done = threading.Event()
        worker_errors = []

        def run_worker():
            try:
                manager.transfer_worker(queue)
            except KeyboardInterrupt:
                pass
            except BaseException as e:  # noqa: BLE001 - surfaced below
                worker_errors.append(e)
            finally:
                worker_done.set()

        thread = threading.Thread(target=run_worker, daemon=True)
        thread.start()
        _wait_until_polling(self, agent)

        self._dispatch_abort(manager)

        self.assertTrue(worker_done.wait(10))
        thread.join(5)
        self.assertEqual(worker_errors, [])
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        self.assertEqual(manager._staging_outstanding[ROOM], 0)
        self.assertNotIn(ROOM, manager._deferred_ack_targets)
        self.assertNotIn(ROOM, manager._deferred_ack_poisoned_rooms)
        self.assertEqual(
            manager._sent,
            [(ROOM, AckTarget(DECODE_IP, DECODE_PORT, ABORT_GENERATION))],
        )


if __name__ == "__main__":
    unittest.main()
