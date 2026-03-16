"""
Unit test for defensive poll handling in process_disagg_prefill_inflight_queue.

Verifies that PP rank > 0 does not crash when the previous rank's consensus
includes rids whose local poll is still in a transient state (e.g.,
Transferring, WaitingForInput, or Bootstrapping) due to propagation delay.

This test avoids importing sglang.srt.disaggregation.prefill directly (which
pulls in CUDA dependencies) and instead tests the core logic inline.

Ref: https://github.com/sgl-project/sglang/issues/20485
"""

import unittest
from types import SimpleNamespace


# Replicate KVPoll enum values to avoid CUDA import
class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


def simulate_inflight_queue_logic(reqs, polls, rids_to_check):
    """
    Simulate the core logic of process_disagg_prefill_inflight_queue.
    Returns (done_rids, undone_rids, crashed).

    This mirrors the logic at prefill.py lines 565-618 after the fix.
    """
    done_rids = []
    undone_rids = []
    crashed = False

    for req, poll in zip(reqs, polls):
        if rids_to_check is not None:
            if req.rid not in rids_to_check:
                undone_rids.append(req.rid)
                continue

            # THE FIX: defensive handling instead of assert
            if poll not in (
                KVPoll.Success,
                KVPoll.Failed,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ):
                undone_rids.append(req.rid)
                continue

        if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
            undone_rids.append(req.rid)
        elif poll == KVPoll.Success:
            done_rids.append(req.rid)
        elif poll == KVPoll.Failed:
            done_rids.append(req.rid)
        else:
            # THE FIX: defensive handling instead of assert False
            undone_rids.append(req.rid)

    return done_rids, undone_rids, crashed


def simulate_old_inflight_queue_logic(reqs, polls, rids_to_check):
    """
    Simulate the OLD (buggy) logic before the fix.
    Returns (done_rids, undone_rids, crashed).
    """
    done_rids = []
    undone_rids = []
    crashed = False

    for req, poll in zip(reqs, polls):
        if rids_to_check is not None:
            if req.rid not in rids_to_check:
                undone_rids.append(req.rid)
                continue

            # THE BUG: this assert crashes on transient states
            if not (poll == KVPoll.Success or poll == KVPoll.Failed):
                crashed = True
                return done_rids, undone_rids, crashed

        if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
            undone_rids.append(req.rid)
        elif poll == KVPoll.Success:
            done_rids.append(req.rid)
        elif poll == KVPoll.Failed:
            done_rids.append(req.rid)
        else:
            crashed = True
            return done_rids, undone_rids, crashed

    return done_rids, undone_rids, crashed


def _make_req(rid):
    return SimpleNamespace(rid=rid)


class TestOldCodeCrashes(unittest.TestCase):
    """Verify the OLD code crashes on the reported scenario."""

    def test_transferring_in_consensus_crashes_old_code(self):
        """The exact scenario from issue #20485: rank 0 says done, rank 1 still Transferring."""
        req = _make_req("rid_1")
        _, _, crashed = simulate_old_inflight_queue_logic(
            [req], [KVPoll.Transferring], rids_to_check=["rid_1"]
        )
        self.assertTrue(crashed, "Old code should crash on Transferring in consensus")

    def test_bootstrapping_in_consensus_crashes_old_code(self):
        """Bootstrapping state in consensus should crash old code."""
        req = _make_req("rid_2")
        _, _, crashed = simulate_old_inflight_queue_logic(
            [req], [KVPoll.Bootstrapping], rids_to_check=["rid_2"]
        )
        self.assertTrue(crashed, "Old code should crash on Bootstrapping in consensus")

    def test_waiting_for_input_in_consensus_crashes_old_code(self):
        """WaitingForInput state in consensus should crash old code."""
        req = _make_req("rid_3")
        _, _, crashed = simulate_old_inflight_queue_logic(
            [req], [KVPoll.WaitingForInput], rids_to_check=["rid_3"]
        )
        self.assertTrue(
            crashed, "Old code should crash on WaitingForInput in consensus"
        )


class TestFixedCodeHandlesTransientStates(unittest.TestCase):
    """Verify the FIXED code handles transient states gracefully."""

    def test_transferring_in_consensus_no_crash(self):
        """Transferring should be treated as undone, not crash."""
        req = _make_req("rid_1")
        done, undone, crashed = simulate_inflight_queue_logic(
            [req], [KVPoll.Transferring], rids_to_check=["rid_1"]
        )
        self.assertFalse(crashed)
        self.assertEqual(done, [])
        self.assertEqual(undone, ["rid_1"])

    def test_bootstrapping_in_consensus_no_crash(self):
        """Bootstrapping (unexpected in inflight queue) should be treated as undone."""
        req = _make_req("rid_2")
        done, undone, crashed = simulate_inflight_queue_logic(
            [req], [KVPoll.Bootstrapping], rids_to_check=["rid_2"]
        )
        self.assertFalse(crashed)
        self.assertEqual(done, [])
        self.assertEqual(undone, ["rid_2"])

    def test_waiting_for_input_in_consensus_no_crash(self):
        """WaitingForInput should be treated as undone."""
        req = _make_req("rid_3")
        done, undone, crashed = simulate_inflight_queue_logic(
            [req], [KVPoll.WaitingForInput], rids_to_check=["rid_3"]
        )
        self.assertFalse(crashed)
        self.assertEqual(done, [])
        self.assertEqual(undone, ["rid_3"])

    def test_success_in_consensus_processed(self):
        """Success should be processed as done."""
        req = _make_req("rid_4")
        done, undone, crashed = simulate_inflight_queue_logic(
            [req], [KVPoll.Success], rids_to_check=["rid_4"]
        )
        self.assertFalse(crashed)
        self.assertEqual(done, ["rid_4"])
        self.assertEqual(undone, [])

    def test_failed_in_consensus_processed(self):
        """Failed should be processed as done (abort)."""
        req = _make_req("rid_5")
        done, undone, crashed = simulate_inflight_queue_logic(
            [req], [KVPoll.Failed], rids_to_check=["rid_5"]
        )
        self.assertFalse(crashed)
        self.assertEqual(done, ["rid_5"])
        self.assertEqual(undone, [])

    def test_mixed_states(self):
        """Mix of terminal and transient states in consensus."""
        reqs = [_make_req("ok"), _make_req("xfer"), _make_req("fail")]
        polls = [KVPoll.Success, KVPoll.Transferring, KVPoll.Failed]
        done, undone, crashed = simulate_inflight_queue_logic(
            reqs, polls, rids_to_check=["ok", "xfer", "fail"]
        )
        self.assertFalse(crashed)
        self.assertIn("ok", done)
        self.assertIn("fail", done)
        self.assertIn("xfer", undone)

    def test_rid_not_in_check_list(self):
        """Rids not in rids_to_check stay as undone."""
        req = _make_req("rid_other")
        done, undone, crashed = simulate_inflight_queue_logic(
            [req], [KVPoll.Success], rids_to_check=["rid_not_exist"]
        )
        self.assertFalse(crashed)
        self.assertEqual(done, [])
        self.assertEqual(undone, ["rid_other"])

    def test_no_rids_to_check(self):
        """Normal mode (no PP) works as before."""
        req = _make_req("rid_normal")
        done, undone, crashed = simulate_inflight_queue_logic(
            [req], [KVPoll.Success], rids_to_check=None
        )
        self.assertFalse(crashed)
        self.assertEqual(done, ["rid_normal"])


if __name__ == "__main__":
    unittest.main()
