import threading
import unittest
from unittest import mock

from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVSender
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefillDPRankRegistration(unittest.TestCase):
    @staticmethod
    def _manager(*, leader: bool) -> CommonKVManager:
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.attn_tp_rank = 0 if leader else 1
        mgr.attn_cp_rank = 0
        mgr.pp_rank = 0
        mgr.attn_dp_rank = 3
        mgr._dp_rank_registration_executor = None
        mgr._dp_rank_registration_executor_lock = threading.Lock()
        mgr._dp_rank_registration_sessions = threading.local()
        return mgr

    def test_nonleader_does_not_submit_duplicate_registration(self):
        mgr = self._manager(leader=False)
        mgr._ensure_dp_rank_registration_executor = mock.Mock()

        mgr.submit_prefill_dp_rank_registration("decode:8999", 17)

        mgr._ensure_dp_rank_registration_executor.assert_not_called()

    def test_leader_submits_without_posting_on_caller_thread(self):
        mgr = self._manager(leader=True)
        executor = mock.Mock()
        mgr._ensure_dp_rank_registration_executor = mock.Mock(return_value=executor)

        mgr.submit_prefill_dp_rank_registration("decode:8999", 17)

        executor.submit.assert_called_once_with(
            mgr._run_prefill_dp_rank_registration,
            "http://decode:8999/register_dp_rank",
            {"bootstrap_room": 17, "dp_rank": 3},
        )

    def test_worker_reuses_session_and_preserves_request_payload(self):
        mgr = self._manager(leader=True)
        session = mock.Mock()
        session.post.return_value = mock.Mock(status_code=200)
        mgr._get_dp_rank_registration_session = mock.Mock(return_value=session)
        payload = {"bootstrap_room": 17, "dp_rank": 3}

        mgr._run_prefill_dp_rank_registration(
            "http://decode:8999/register_dp_rank", payload
        )

        session.post.assert_called_once_with(
            "http://decode:8999/register_dp_rank", json=payload, timeout=5
        )

    def test_sender_delegates_registration_to_manager(self):
        sender = CommonKVSender.__new__(CommonKVSender)
        sender.kv_mgr = mock.Mock()
        sender.bootstrap_server_url = "decode:8999"
        sender.bootstrap_room = 17

        sender._register_prefill_dp_rank()

        sender.kv_mgr.submit_prefill_dp_rank_registration.assert_called_once_with(
            "decode:8999", 17
        )


if __name__ == "__main__":
    unittest.main()
