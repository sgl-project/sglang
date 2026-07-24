"""Unit tests for srt/disaggregation/common/conn — CommonKVSender DP rank routing.

These tests cover the follow_bootstrap_room conflict check and the
_register_prefill_dp_rank payload, which under plain DP must use the
per-process system_dp_rank rather than attn_dp_rank (which is the rank
*inside* the attention DP group and is constant 0 when DP attention is off).

See also: PR #22901 (which introduced the conflict check) and the bug where
every DP subprocess saw attn_dp_rank=0 and either rejected ~half of all
requests (default LB) or registered every request as dp_rank=0 (with the
SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=1 workaround).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVSender
from sglang.test.test_utils import CustomTestCase


def _make_manager_mock(
    *,
    effective_dp_rank: int,
    dp_size: int,
    load_balance_method: str = "follow_bootstrap_room",
):
    """Build a CommonKVManager-spec mock with the fields CommonKVSender reads.

    `effective_dp_rank` is provided directly so the test fixture does not
    re-derive what the source under test computes.
    """
    mgr = MagicMock(spec=CommonKVManager)
    mgr.effective_dp_rank = effective_dp_rank
    mgr.is_dummy_cp_rank = False
    mgr.server_args = MagicMock()
    mgr.server_args.dp_size = dp_size
    mgr.server_args.load_balance_method = load_balance_method
    return mgr


class TestFollowBootstrapRoomConflictCheck(CustomTestCase):
    """The conflict check in CommonKVSender.__init__ must compare the *effective*
    DP rank (= system_dp_rank under plain DP) against bootstrap_room % dp_size,
    not attn_dp_rank (which is constant 0 under plain DP).

    We construct a real CommonKVSender with a mocked manager so the actual
    __init__ source is exercised.
    """

    def _build_sender(self, mgr, bootstrap_room):
        sender = CommonKVSender(
            mgr=mgr,
            bootstrap_addr="127.0.0.1:8765",
            bootstrap_room=bootstrap_room,
            dest_tp_ranks=[0],
            pp_rank=0,
        )
        return sender

    def test_dp1_with_room_implying_dp1_does_not_fail(self):
        """A DP1 subprocess receives a request whose room%dp_size==1.  Under
        the fix, effective_dp_rank=1 matches and the request passes through.
        Before the fix, attn_dp_rank=0 != 1 and the request was rejected."""
        mgr = _make_manager_mock(effective_dp_rank=1, dp_size=2)
        self._build_sender(mgr, bootstrap_room=43)  # 43 % 2 == 1
        mgr.record_failure.assert_not_called()
        statuses = [c.args[1] for c in mgr.update_status.call_args_list]
        self.assertNotIn(KVPoll.Failed, statuses)

    def test_dp0_with_room_implying_dp1_fails_with_correct_rank_in_message(self):
        """A genuine routing conflict (room%2==1 dispatched to DP0) is still
        flagged after the fix.  The error message must report the actual
        dispatched DP rank (0 here), not always 0."""
        mgr = _make_manager_mock(effective_dp_rank=0, dp_size=2)
        self._build_sender(mgr, bootstrap_room=43)
        mgr.record_failure.assert_called_once()
        msg = mgr.record_failure.call_args[0][1]
        self.assertIn("dispatched to dp_rank 0", msg)
        self.assertIn("implies dp_rank 1", msg)
        statuses = [c.args[1] for c in mgr.update_status.call_args_list]
        self.assertIn(KVPoll.Failed, statuses)

    def test_dp_attention_rank_match(self):
        """With DP attention on, effective_dp_rank == attn_dp_rank.  A DP-rank-1
        worker getting a room%2==1 request must pass through without rejection."""
        mgr = _make_manager_mock(effective_dp_rank=1, dp_size=2)
        self._build_sender(mgr, bootstrap_room=99)  # 99 % 2 == 1
        mgr.record_failure.assert_not_called()


class TestRegisterPrefillDpRankPayload(CustomTestCase):
    """The /register_dp_rank payload must carry the per-process effective DP
    rank, not attn_dp_rank.  Otherwise every DP subprocess registers as
    dp_rank=0 and the decode side always queries DP0's port."""

    @patch("sglang.srt.disaggregation.common.conn.requests.post")
    def test_payload_uses_effective_dp_rank(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        mgr = _make_manager_mock(
            effective_dp_rank=1,
            dp_size=2,
            # Non-default LB drives the sender into _register_prefill_dp_rank
            load_balance_method="round_robin",
        )
        CommonKVSender(
            mgr=mgr,
            bootstrap_addr="127.0.0.1:8765",
            bootstrap_room=12345,
            dest_tp_ranks=[0],
            pp_rank=0,
        )

        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["bootstrap_room"], 12345)
        # Must be the effective rank (1), not attn_dp_rank (0).
        self.assertEqual(payload["dp_rank"], 1)


if __name__ == "__main__":
    unittest.main()
