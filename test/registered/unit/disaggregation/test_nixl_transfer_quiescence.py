"""CPU regressions for NIXL transfer-group ERR ownership.

NIXL ERR is terminal at the public handle but does not prove transport
quiescence. These tests exercise the real transfer-worker loop with fake
handles and a fake agent; no GPU, RDMA device, or NIXL installation is needed.
"""

import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.nixl.conn import NixlKVManager, TransferInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ScriptedAgent:
    def __init__(self, scripts, on_check=None):
        self.scripts = {handle: list(states) for handle, states in scripts.items()}
        self.checks = defaultdict(int)
        self.on_check = on_check
        self.release_proc = threading.Event()
        self.release_xfer_handle = MagicMock()

    def check_xfer_state(self, handle):
        self.checks[handle] += 1
        if self.on_check is not None:
            self.on_check(handle, self.checks[handle])
        states = self.scripts[handle]
        state = states.pop(0)
        if not states:
            self.scripts[handle] = [state]
        if state == "PROC" and self.release_proc.is_set():
            return "DONE"
        return state


class TestNixlTransferGroupErrOwnership(unittest.TestCase):
    room = 73
    later_room = 74

    @staticmethod
    def _transfer_info(room):
        return TransferInfo(
            room=room,
            endpoint="127.0.0.1",
            dst_port=5555,
            agent_name="agent",
            dst_kv_indices=np.array([], dtype=np.int32),
            dst_aux_index=0,
            required_dst_info_num=1,
            dst_state_indices=[[9]],
            decode_prefix_len=1,
        )

    def _make_manager(self, scripts, *, deferred=False, on_check=None):
        mgr = object.__new__(NixlKVManager)
        rooms = (self.room, self.later_room)
        mgr.request_status = {room: KVPoll.WaitingForInput for room in rooms}
        mgr.transfer_infos = {
            room: {"agent": self._transfer_info(room)} for room in rooms
        }
        mgr.decode_kv_args_table = {
            "agent": SimpleNamespace(
                decode_tp_size=1,
                decode_tp_rank=0,
                dst_kv_ptrs=[],
                dst_aux_ptrs=[0],
                dst_state_data_ptrs=[[0]],
                dst_state_item_lens=[[1]],
                dst_state_dim_per_tensor=[[]],
                dst_state_layer_ids=[[]],
                gpu_id=0,
                staging_base_ptr=0,
                staging_total_size=0,
                kv_xfer_segments=None,
                dst_homogeneous_mem_kind="VRAM",
                requires_dcp_relayout=False,
            )
        }
        mgr.req_to_decode_prefix_len = {room: 0 for room in rooms}
        mgr.enable_staging = False
        mgr.enable_deferred_decode_kv_release = deferred
        mgr._staging_ctx = None
        mgr._staging_outstanding = defaultdict(int)
        mgr._failed_transfer_handles = defaultdict(list)
        mgr._deferred_ack_targets = {}
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = False
        mgr.attn_tp_size = 1
        mgr.transfer_source_rank = 0
        mgr.kv_args = SimpleNamespace(engine_rank=0, kv_data_ptrs=[])
        mgr.exceptions = {}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr.agent = _ScriptedAgent(scripts, on_check=on_check)
        mgr._sent_acks = []
        mgr._send_abort_ack = lambda ip, port, room: mgr._sent_acks.append(
            (ip, port, room)
        )
        return mgr

    def _chunk(self, *, room=None, with_state=False):
        return TransferKVChunk(
            room=self.room if room is None else room,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=True,
            chunk_id=0,
            prefill_aux_index=0,
            state_indices=[[1]] if with_state else None,
        )

    def _run_once(self, mgr, chunk):
        queue = SimpleNamespace(get=MagicMock(side_effect=[chunk, SystemExit()]))
        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue)

    def _assert_failed_group(self, mgr, expected_checks, expected_handles):
        self.assertEqual(dict(mgr.agent.checks), expected_checks)
        self.assertEqual(mgr._staging_outstanding.get(self.room, 0), 1)
        self.assertEqual(mgr._failed_transfer_handles[self.room], expected_handles)
        self.assertEqual(mgr.request_status[self.room], KVPoll.Failed)
        self.assertIn(self.room, mgr.exceptions)
        self.assertEqual(mgr._sent_acks, [])
        mgr.agent.release_xfer_handle.assert_not_called()

    def test_failed_room_skip_preserves_unresolved_group_and_does_not_ack(self):
        mgr = self._make_manager({}, deferred=True)
        mgr.request_status[self.room] = KVPoll.Failed
        mgr._staging_outstanding[self.room] = 1
        mgr.register_deferred_ack_target(self.room, "10.0.0.8", 6000)
        mgr.maybe_send_extra = MagicMock()
        mgr.send_aux = MagicMock()

        self._run_once(mgr, self._chunk(with_state=True))

        self.assertEqual(mgr._staging_outstanding[self.room], 1)
        self.assertEqual(mgr._sent_acks, [])
        self.assertIn(self.room, mgr._deferred_ack_targets)
        mgr.maybe_send_extra.assert_not_called()
        mgr.send_aux.assert_not_called()

    def test_err_proc_forever_and_done_returns_to_later_queue_work(self):
        mgr = self._make_manager(
            {
                "h1-err": ["ERR"],
                "h2-proc": ["PROC"],
                "h3-done": ["DONE"],
                "later-done": ["DONE"],
            },
            deferred=True,
        )
        mgr.maybe_send_extra = MagicMock(return_value=["h1-err", "h2-proc"])
        mgr.send_aux = MagicMock(side_effect=["h3-done", "later-done"])
        mgr.register_deferred_ack_target(self.room, "10.0.0.8", 6000)

        # Include a queued chunk for the failed room before independent work.
        # Skipping it must not erase the failed group's outstanding count.
        queue = SimpleNamespace(
            get=MagicMock(
                side_effect=[
                    self._chunk(with_state=True),
                    self._chunk(),
                    self._chunk(room=self.later_room),
                    SystemExit(),
                ]
            )
        )

        def run_worker():
            try:
                mgr.transfer_worker(queue)
            except SystemExit:
                pass

        worker = threading.Thread(target=run_worker, daemon=True)
        worker.start()
        worker.join(timeout=1)
        returned_promptly = not worker.is_alive()
        if not returned_promptly:
            # Let a regressed worker exit so it cannot spin after the assertion.
            mgr.agent.release_proc.set()
            worker.join(timeout=1)

        self.assertTrue(returned_promptly, "worker blocked on the PROC sibling")
        self._assert_failed_group(
            mgr,
            {
                "h1-err": 1,
                "h2-proc": 1,
                "h3-done": 1,
                "later-done": 1,
            },
            ["h1-err", "h2-proc"],
        )
        self.assertNotIn("h3-done", mgr._failed_transfer_handles[self.room])
        self.assertEqual(mgr.request_status[self.later_room], KVPoll.Success)
        self.assertNotIn(self.later_room, mgr.transfer_infos)
        self.assertIn(self.room, mgr._deferred_ack_targets)

    def test_err_and_done_retains_only_err(self):
        mgr = self._make_manager({"err": ["ERR"], "done": ["DONE"]})
        mgr.maybe_send_extra = MagicMock(return_value=["err"])
        mgr.send_aux = MagicMock(return_value="done")

        self._run_once(mgr, self._chunk(with_state=True))

        self._assert_failed_group(mgr, {"err": 1, "done": 1}, ["err"])

    def test_multiple_err_handles_are_retained(self):
        mgr = self._make_manager({"err-1": ["ERR"], "err-2": ["ERR"]})
        mgr.maybe_send_extra = MagicMock(return_value=["err-1"])
        mgr.send_aux = MagicMock(return_value="err-2")

        self._run_once(mgr, self._chunk(with_state=True))

        self._assert_failed_group(mgr, {"err-1": 1, "err-2": 1}, ["err-1", "err-2"])

    def test_state_handle_err_is_retained(self):
        mgr = self._make_manager({"state": ["ERR"], "aux": ["DONE"]})
        mgr.maybe_send_extra = MagicMock(return_value=["state"])
        mgr.send_aux = MagicMock(return_value="aux")

        self._run_once(mgr, self._chunk(with_state=True))

        self._assert_failed_group(mgr, {"state": 1, "aux": 1}, ["state"])
        mgr.maybe_send_extra.assert_called_once()

    def test_aux_handle_err_is_retained(self):
        mgr = self._make_manager({"state": ["DONE"], "aux": ["ERR"]})
        mgr.maybe_send_extra = MagicMock(return_value=["state"])
        mgr.send_aux = MagicMock(return_value="aux")

        self._run_once(mgr, self._chunk(with_state=True))

        self._assert_failed_group(mgr, {"state": 1, "aux": 1}, ["aux"])

    def test_abort_racing_with_err_never_acks(self):
        mgr = None
        abort_sent = False

        def abort_on_first_check(_handle, _check_number):
            nonlocal abort_sent
            if abort_sent:
                return
            abort_sent = True
            mgr._handle_abort_notification(
                [b"ABORT", str(self.room).encode("ascii"), b"10.0.0.8", b"6000"]
            )

        mgr = self._make_manager(
            {"err": ["ERR"], "done": ["DONE"]},
            deferred=True,
            on_check=abort_on_first_check,
        )
        mgr.maybe_send_extra = MagicMock(return_value=["err"])
        mgr.send_aux = MagicMock(return_value="done")

        self._run_once(mgr, self._chunk(with_state=True))

        self._assert_failed_group(mgr, {"err": 1, "done": 1}, ["err"])
        self.assertIn(self.room, mgr._deferred_ack_targets)


if __name__ == "__main__":
    unittest.main()
