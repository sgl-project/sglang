"""CPU coverage for bounded session repair and real-transfer readmission."""

import threading
import unittest
from collections import defaultdict
from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeSessionRepair(CustomTestCase):
    def setUp(self):
        self.manager = MooncakeKVManager.__new__(MooncakeKVManager)
        self.manager.engine = Mock()
        self.manager.engine.refresh_session.return_value = 0
        self.manager.engine.send_probe.return_value = 0
        self.manager.enable_failed_session_repair = True
        self.manager.failed_sessions = {"peer"}
        self.manager.session_failures = defaultdict(int, {"peer": 1})
        self.manager.session_lock = threading.Lock()
        self.manager.session_repair_attempted = set()
        self.manager.session_repair_rooms = {}
        self.manager.failed_session_probes = {}
        self.manager.failed_session_probe_interval = 30.0
        self.manager.failed_session_probe_max_interval = 300.0
        self.manager.failed_session_idle_timeout = 600.0
        self.manager._staging_outstanding = defaultdict(int)
        self.manager.request_status = {7: KVPoll.Transferring, 8: KVPoll.Transferring}
        self.clock = patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=0
        )
        self.clock_mock = self.clock.start()
        self.addCleanup(self.clock.stop)

    def test_control_success_only_allows_one_trial_room(self):
        mgr = self.manager
        mgr._run_one_probe_pass()
        self.assertIn("peer", mgr.failed_sessions)
        self.assertTrue(mgr._session_transfer_allowed("peer", 7))
        self.assertTrue(mgr._session_transfer_allowed("peer", 7))
        self.assertFalse(mgr._session_transfer_allowed("peer", 8))
        mgr._run_one_probe_pass()
        mgr.engine.refresh_session.assert_called_once_with("peer")
        mgr._finish_session_repair("peer", 8)
        self.assertIn("peer", mgr.failed_sessions)
        mgr._finish_session_repair("peer", 7)
        self.assertNotIn("peer", mgr.failed_sessions)
        self.assertNotIn("peer", mgr.session_repair_attempted)
        self.assertTrue(mgr._session_transfer_allowed("peer", 8))

    def test_failed_trial_cannot_repair_twice(self):
        mgr = self.manager
        mgr._run_one_probe_pass()
        self.assertTrue(mgr._session_transfer_allowed("peer", 7))
        mgr._record_session_failure("peer")
        mgr._finish_session_repair("peer", 7)
        mgr._run_one_probe_pass()
        self.assertIn("peer", mgr.failed_sessions)
        self.assertFalse(mgr._session_transfer_allowed("peer", 7))
        mgr.engine.refresh_session.assert_called_once_with("peer")

    def test_refresh_failure_retains_isolation(self):
        mgr = self.manager
        mgr.engine.refresh_session.return_value = -1
        mgr._run_one_probe_pass()
        mgr.engine.send_probe.assert_not_called()
        self.assertFalse(mgr._session_transfer_allowed("peer", 7))
        mgr.engine.refresh_session.return_value = 0
        mgr._run_one_probe_pass()
        self.assertEqual(mgr.engine.refresh_session.call_count, 1)
        self.clock_mock.return_value = 60
        mgr._run_one_probe_pass()
        self.assertTrue(mgr._session_transfer_allowed("peer", 7))
        self.assertEqual(mgr.engine.refresh_session.call_count, 2)

    def test_probe_failure_retains_isolation(self):
        mgr = self.manager
        mgr.engine.send_probe.return_value = -1
        mgr._run_one_probe_pass()
        self.assertFalse(mgr._session_transfer_allowed("peer", 7))
        self.assertNotIn("peer", mgr.session_repair_attempted)

    def test_new_failure_during_control_probe_rejects_stale_success(self):
        mgr = self.manager

        def fail_during_probe(_):
            mgr._record_session_failure("peer")
            return 0

        mgr.engine.send_probe.side_effect = fail_during_probe
        mgr._run_one_probe_pass()
        self.assertFalse(mgr._session_transfer_allowed("peer", 7))

    def test_legacy_probe_behavior_is_unchanged(self):
        mgr = self.manager
        mgr.enable_failed_session_repair = False
        mgr._run_one_probe_pass()
        mgr.engine.refresh_session.assert_not_called()
        self.assertNotIn("peer", mgr.failed_sessions)

    def test_healthy_session_does_not_require_a_repair(self):
        self.assertTrue(self.manager._session_transfer_allowed("healthy", 9))

    def test_successful_real_trial_allows_a_new_incident(self):
        mgr = self.manager
        mgr._run_one_probe_pass()
        self.assertTrue(mgr._session_transfer_allowed("peer", 7))
        mgr._finish_session_repair("peer", 7)
        mgr._record_session_failure("peer")
        mgr._run_one_probe_pass()
        self.assertEqual(mgr.engine.refresh_session.call_count, 2)
        self.assertIn("peer", mgr.failed_sessions)

    def test_cancelled_room_releases_binding_without_another_refresh(self):
        mgr = self.manager
        mgr._room_lifecycle_lock = threading.Lock()
        mgr._concluded_rooms = {}
        mgr._run_one_probe_pass()
        self.assertTrue(mgr._session_transfer_allowed("peer", 7))
        mgr._release_session_repair_room(7)
        mgr._finish_session_repair("peer", 7)
        self.assertIn("peer", mgr.failed_sessions)
        self.assertTrue(mgr._session_transfer_allowed("peer", 8))
        mgr._run_one_probe_pass()
        mgr.engine.refresh_session.assert_called_once_with("peer")
        mgr._finish_session_repair("peer", 8)
        self.assertNotIn("peer", mgr.failed_sessions)

    def test_refresh_exception_keeps_peer_isolated_until_later_check(self):
        mgr = self.manager
        mgr.engine.refresh_session.side_effect = [RuntimeError("unreachable"), 0]
        mgr._run_one_probe_pass()
        self.assertFalse(mgr._session_transfer_allowed("peer", 7))
        self.clock_mock.return_value = 60
        mgr._run_one_probe_pass()
        self.assertTrue(mgr._session_transfer_allowed("peer", 7))

    def test_idle_timeout_does_not_bypass_trial(self):
        mgr = self.manager
        mgr._run_one_probe_pass()
        self.clock_mock.return_value = 1200
        mgr._run_one_probe_pass()
        self.assertIn("peer", mgr.failed_sessions)
        self.assertIn("peer", mgr.session_repair_rooms)
        mgr.engine.refresh_session.assert_called_once()

    def test_cancelled_inflight_trial_releases_only_after_worker_finishes(self):
        mgr = self.manager
        mgr._run_one_probe_pass()
        self.assertTrue(mgr._session_transfer_allowed("peer", 7))
        mgr._staging_outstanding[7] = 1
        mgr.request_status[7] = KVPoll.Failed
        mgr._release_session_repair_room(7)
        mgr._finish_session_repair("peer", 7)
        self.assertFalse(mgr._session_transfer_allowed("peer", 8))
        self.assertIn("peer", mgr.failed_sessions)
        mgr._staging_outstanding[7] = 0
        mgr._release_session_repair_room(7)
        self.assertTrue(mgr._session_transfer_allowed("peer", 8))

    def test_reregistration_does_not_bypass_repair(self):
        mgr = self.manager
        mgr._validate_peer_state_layout = Mock(return_value=(True, None))
        mgr.check_peer_aux_buffer_count = Mock()
        mgr.state_layout_rejections = {}
        mgr.state_strides_validated = set()
        mgr.decode_kv_args_table = {}
        mgr._publish_peer_registration("peer", SimpleNamespace(dst_aux_ptrs=[]))
        self.assertIn("peer", mgr.failed_sessions)
        self.assertFalse(mgr._session_transfer_allowed("peer", 7))
        mgr.enable_failed_session_repair = False
        mgr._publish_peer_registration("peer", SimpleNamespace(dst_aux_ptrs=[]))
        self.assertNotIn("peer", mgr.failed_sessions)

    def test_prefill_startup_requires_native_refresh_only_when_enabled(self):
        self.manager.disaggregation_mode = DisaggregationMode.PREFILL
        module = "sglang.srt.disaggregation.mooncake.conn"
        old_engine = SimpleNamespace(engine=object())
        with patch(f"{module}.get_mooncake_transfer_engine", return_value=old_engine):
            with patch(
                f"{module}.envs.SGLANG_ENABLE_FAILED_SESSION_REPAIR.get",
                return_value=False,
            ):
                self.manager.init_engine()
            with patch(
                f"{module}.envs.SGLANG_ENABLE_FAILED_SESSION_REPAIR.get",
                return_value=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "refresh_session"):
                    self.manager.init_engine()
        new_engine = SimpleNamespace(engine=SimpleNamespace(refresh_session=Mock()))
        with (
            patch(f"{module}.get_mooncake_transfer_engine", return_value=new_engine),
            patch(
                f"{module}.envs.SGLANG_ENABLE_FAILED_SESSION_REPAIR.get",
                return_value=True,
            ),
        ):
            self.manager.init_engine()

    def _run_transfer(
        self,
        *,
        kv_rc=0,
        state_rc=0,
        aux_rc=0,
        empty=False,
        abort=False,
        last=True,
        abort_state=False,
    ):
        mgr = self.manager
        mgr._run_one_probe_pass()
        mgr.enable_trace = mgr.enable_staging = False
        mgr.enable_deferred_decode_kv_release = False
        mgr.is_mla_backend = True
        mgr.bootstrap_port = 1234
        mgr.kv_args = SimpleNamespace(kv_data_ptrs=[1])
        mgr.req_to_decode_prefix_len = {}
        mgr.state_layout_rejections = {}
        mgr._get_dsa_cache_transfer_skip_flags = lambda info: (False, False)
        mgr.validate_aux_buffers = Mock()
        mgr.check_status = lambda room: mgr.request_status[room]
        mgr.conclude_failure = Mock(
            side_effect=lambda **kw: mgr.request_status.__setitem__(
                kw["bootstrap_room"], KVPoll.Failed
            )
        )
        mgr.conclude_transfer = Mock(
            side_effect=lambda **kw: mgr.request_status.__setitem__(
                kw["bootstrap_room"], kw["status"]
            )
        )
        mgr.decode_kv_args_table = {
            "peer": SimpleNamespace(
                requires_dcp_relayout=False,
                dst_aux_ptrs=[2],
                dst_aux_item_lens=[8],
                dst_kv_ptrs=[3],
                dst_tp_rank=0,
                dst_attn_tp_size=1,
                dst_kv_item_len=8,
                dst_kv_layer_ids=None,
            )
        }
        mgr.transfer_infos = {
            7: {
                "peer": SimpleNamespace(
                    room=7,
                    mooncake_session_id="peer",
                    is_dummy=False,
                    endpoint="127.0.0.1",
                    dst_port=1234,
                    dst_kv_indices=np.array([0]),
                    dst_device_kv_indices=None,
                    required_dst_info_num=1,
                )
            }
        }
        mgr.send_kvcache = Mock(return_value=kv_rc)
        mgr.maybe_send_extra = Mock(return_value=state_rc)
        mgr.send_aux = Mock(return_value=aux_rc)
        if abort_state:

            def state(*args, **kwargs):
                mgr.request_status[7] = KVPoll.Failed
                return state_rc

            mgr.maybe_send_extra.side_effect = state
        if abort:

            def aux(*args, **kwargs):
                mgr.request_status[7] = KVPoll.Failed
                mgr._release_session_repair_room(7)
                return aux_rc

            mgr.send_aux.side_effect = aux
        if abort and not last:

            def kv(*args, **kwargs):
                mgr.request_status[7] = KVPoll.Failed
                mgr._release_session_repair_room(7)
                return kv_rc

            mgr.send_kvcache.side_effect = kv
        queue = SimpleQueue()
        queue.put(
            SimpleNamespace(
                room=7,
                prefill_kv_indices=np.array([] if empty else [0]),
                index_slice=slice(0, 1),
                is_last_chunk=last,
                prefill_aux_index=0,
                state_indices=[[0]],
                staging_counted=False,
                wait_event=None,
                indices_event=None,
            )
        )
        queue.put(None)
        mgr.transfer_worker(queue, executor=None)
        return mgr

    def test_worker_requires_kv_state_and_aux_success(self):
        mgr = self._run_transfer()
        mgr.send_kvcache.assert_called_once()
        mgr.maybe_send_extra.assert_called_once()
        mgr.send_aux.assert_called_once()
        self.assertNotIn("peer", mgr.failed_sessions)

    def test_worker_failures_consume_the_trial(self):
        for stage in ("kv_rc", "state_rc", "aux_rc"):
            with self.subTest(stage=stage):
                # A fresh incident for each failure stage.
                self.manager.session_repair_attempted.clear()
                self.manager.session_repair_rooms.clear()
                self.manager.request_status[7] = KVPoll.Transferring
                mgr = self._run_transfer(**{stage: -1})
                self.assertIn("peer", mgr.failed_sessions)
                self.assertNotIn("peer", mgr.session_repair_rooms)
                self.assertFalse(mgr._session_transfer_allowed("peer", 8))
                calls = mgr.engine.refresh_session.call_count
                mgr._run_one_probe_pass()
                self.assertEqual(mgr.engine.refresh_session.call_count, calls)

    def test_failed_state_after_abort_still_consumes_trial(self):
        mgr = self._run_transfer(state_rc=-1, abort_state=True)
        self.assertIn("peer", mgr.failed_sessions)
        self.assertNotIn("peer", mgr.session_repair_rooms)
        self.assertFalse(mgr._session_transfer_allowed("peer", 8))

    def test_empty_kv_does_not_prove_recovery(self):
        mgr = self._run_transfer(empty=True)
        mgr.send_kvcache.assert_not_called()
        self.assertIn("peer", mgr.failed_sessions)

    def test_abort_during_aux_does_not_prove_recovery(self):
        mgr = self._run_transfer(abort=True)
        self.assertIn("peer", mgr.failed_sessions)
        self.assertTrue(mgr._session_transfer_allowed("peer", 8))

    def test_abort_during_nonfinal_chunk_releases_after_write(self):
        mgr = self._run_transfer(abort=True, last=False)
        self.assertIn("peer", mgr.failed_sessions)
        self.assertTrue(mgr._session_transfer_allowed("peer", 8))

    def test_aux_failure_keeps_legacy_blacklist_behavior(self):
        mgr = self.manager
        mgr.enable_failed_session_repair = False
        mgr.failed_sessions.clear()
        mgr = self._run_transfer(aux_rc=-1)
        self.assertNotIn("peer", mgr.failed_sessions)


if __name__ == "__main__":
    unittest.main()
