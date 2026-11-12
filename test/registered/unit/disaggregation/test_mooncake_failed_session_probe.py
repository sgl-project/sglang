"""Backoff and idle pruning in the Mooncake failed-session probe loop."""

import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, TransferInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CLOCK = "sglang.srt.disaggregation.mooncake.conn.time.monotonic"
SESSION = "10.0.0.2:15000"
ROOM = 7


def _manager(probe_rc: int = -1, idle_timeout: float = 600.0) -> MooncakeKVManager:
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.session_lock = threading.Lock()
    manager.failed_sessions = {SESSION}
    manager.session_failures = defaultdict(int, {SESSION: 1})
    manager.failed_session_probes = {}
    manager.failed_session_probe_interval = 30.0
    manager.failed_session_probe_max_interval = 300.0
    manager.failed_session_idle_timeout = idle_timeout
    manager.engine = SimpleNamespace(send_probe=Mock(return_value=probe_rc))
    return manager


def _run_passes(manager, until: float, step: float = 30.0) -> list:
    """Run the probe loop every `step` seconds; return when probes were sent."""
    sent = []
    manager.engine.send_probe.side_effect = lambda session_id: (
        sent.append(now) or manager.engine.send_probe.return_value
    )
    now = 0.0
    while now <= until:
        with patch(CLOCK, return_value=now):
            manager._run_one_probe_pass()
        now += step
    return sent


def _request_blacklisted_session(manager, now: float) -> None:
    """Send one chunk for the blacklisted session through transfer_worker."""
    manager.kv_args = SimpleNamespace(kv_data_ptrs=[], state_types=[])
    manager.is_mla_backend = True
    manager.is_hybrid_mla_backend = False
    manager.attn_tp_size = manager.attn_cp_size = manager.pp_size = 1
    manager.attn_tp_rank = manager.attn_cp_rank = manager.pp_rank = 0
    manager.enable_custom_mem_pool = False
    manager.max_transfer_batch_indices = 0
    manager.enable_trace = manager.enable_staging = False
    manager.bootstrap_port = 1234
    manager.failure_lock = threading.Lock()
    manager.failure_records = {}
    manager._staging_outstanding = defaultdict(int)
    manager.state_layout_rejections = {}
    manager.state_strides_validated = set()
    manager.decode_kv_args_table = {}
    manager.req_to_decode_prefix_len = {ROOM: 0}
    manager.request_status = {ROOM: KVPoll.Transferring}
    manager.transfer_infos = {
        ROOM: {
            SESSION: TransferInfo(
                room=ROOM,
                endpoint="10.0.0.2",
                dst_port=1234,
                mooncake_session_id=SESSION,
                dst_kv_indices=np.array([], dtype=np.int32),
                dst_aux_index=0,
                dst_state_indices=None,
                required_dst_info_num=1,
                is_dummy=False,
            )
        }
    }
    manager._send_multipart_locked = Mock()
    manager._transfer_data = Mock(return_value=0)
    queue = FastQueue()
    queue.put(
        TransferKVChunk(
            room=ROOM,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
        )
    )
    queue.put(None)
    with patch(CLOCK, return_value=now):
        manager.transfer_worker(queue, executor=None)
    assert manager.request_status[ROOM] == KVPoll.Failed
    manager._transfer_data.assert_not_called()


class TestMooncakeFailedSessionProbe(CustomTestCase):
    def test_failing_probe_backs_off_to_the_cap(self):
        manager = _manager(idle_timeout=1e9)

        sent = _run_passes(manager, until=1200)

        self.assertEqual(sent, [0, 60, 180, 420, 720, 1020])
        self.assertIn(SESSION, manager.failed_sessions)

    def test_recovery_unblacklists_and_resets_the_schedule(self):
        manager = _manager()
        _run_passes(manager, until=60)
        self.assertEqual(manager.failed_session_probes[SESSION].interval, 120)

        manager.engine.send_probe.return_value = 0
        with patch(CLOCK, return_value=180):
            manager._run_one_probe_pass()
        self.assertFalse(manager.failed_sessions)
        self.assertFalse(manager.session_failures)
        self.assertFalse(manager.failed_session_probes)

        # Blacklisted again later: probed on the next pass, from the base interval.
        manager.failed_sessions.add(SESSION)
        manager.engine.send_probe.return_value = -1
        manager.engine.send_probe.reset_mock()
        with patch(CLOCK, return_value=500):
            manager._run_one_probe_pass()
        manager.engine.send_probe.assert_called_once_with(SESSION)
        self.assertEqual(manager.failed_session_probes[SESSION].interval, 60)

    def test_unrequested_session_is_dropped(self):
        # E.g. the old session id of a restarted decode.
        manager = _manager()

        sent = _run_passes(manager, until=1200)

        self.assertEqual(sent, [0, 60, 180, 420])
        self.assertFalse(manager.failed_sessions)
        self.assertFalse(manager.session_failures)
        self.assertFalse(manager.failed_session_probes)

    def test_requested_session_stays_blacklisted_and_probed(self):
        manager = _manager()
        with patch(CLOCK, return_value=0):
            manager._run_one_probe_pass()

        _request_blacklisted_session(manager, now=590)
        with patch(CLOCK, return_value=600):
            manager._run_one_probe_pass()
        self.assertIn(SESSION, manager.failed_sessions)
        self.assertEqual(manager.failed_session_probes[SESSION].last_requested, 590)

        with patch(CLOCK, return_value=1190):
            manager._run_one_probe_pass()
        self.assertFalse(manager.failed_sessions)


if __name__ == "__main__":
    unittest.main()
