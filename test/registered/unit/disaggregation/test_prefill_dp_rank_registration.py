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
        return mgr

    @mock.patch("sglang.srt.disaggregation.common.conn.requests.post")
    def test_nonleader_does_not_submit_duplicate_registration(self, post):
        mgr = self._manager(leader=False)

        mgr.register_prefill_dp_rank("decode:8999", 17)

        post.assert_not_called()

    @mock.patch("sglang.srt.disaggregation.common.conn.requests.post")
    def test_leader_registers_before_returning(self, post):
        mgr = self._manager(leader=True)
        post.return_value = mock.Mock(status_code=200)

        mgr.register_prefill_dp_rank("decode:8999", 17)

        post.assert_called_once_with(
            "http://decode:8999/register_dp_rank",
            json={"bootstrap_room": 17, "dp_rank": 3},
            timeout=5,
        )

    def test_sender_delegates_registration_to_manager(self):
        sender = CommonKVSender.__new__(CommonKVSender)
        sender.kv_mgr = mock.Mock()
        sender.bootstrap_server_url = "decode:8999"
        sender.bootstrap_room = 17

        sender._register_prefill_dp_rank()

        sender.kv_mgr.register_prefill_dp_rank.assert_called_once_with(
            "decode:8999", 17
        )


if __name__ == "__main__":
    unittest.main()
