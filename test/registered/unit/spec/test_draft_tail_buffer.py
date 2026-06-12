"""CPU unit tests for the decoupled-spec verifier-side DraftTailBuffer.

DraftTailBuffer is the verifier-side reconciliation state machine: it mirrors the
draft tokens the drafter streams asynchronously and reconciles them against the
verifier's committed prefix. These tests drive the state machine directly (no GPU,
no transport) and assert every reconciliation outcome.
"""

import threading
import unittest
from collections import deque

from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    VerifyCommit,
)
from sglang.srt.speculative.draft_tail_buffer import (
    DraftTailBuffer,
    DraftTailSnapshot,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeReq:
    """Minimal stand-in for a scheduler request; get_draft_snapshots only reads .rid."""

    def __init__(self, rid: str) -> None:
        self.rid = rid


def _sync(rid, *, committed_len=0, drafter_rank=0, src_verifier_rank=0) -> DraftSync:
    return DraftSync(
        request_id=rid,
        src_verifier_rank=src_verifier_rank,
        dst_drafter_rank=drafter_rank,
        prompt_token_ids=[],
        committed_output_ids=[0] * committed_len,
    )


def _commit(rid, *, pre, tokens, drafter_rank=0, src_verifier_rank=0) -> VerifyCommit:
    return VerifyCommit(
        request_id=rid,
        src_verifier_rank=src_verifier_rank,
        dst_drafter_rank=drafter_rank,
        pre_verify_committed_len=pre,
        committed_token_ids=list(tokens),
    )


def _close(rid, *, drafter_rank=0, reason="finished") -> DraftClose:
    return DraftClose(
        request_id=rid,
        src_verifier_rank=0,
        dst_drafter_rank=drafter_rank,
        reason=reason,
    )


def _out(
    rid, *, base, pos, tok, drafter_rank=0, verifier_rank=0
) -> DraftTailStreamOutput:
    return DraftTailStreamOutput(
        src_drafter_rank=drafter_rank,
        dst_verifier_rank=verifier_rank,
        request_id=rid,
        base_committed_len=base,
        new_token_pos=pos,
        new_token_id=tok,
    )


def _stream(*outputs) -> DraftTailStreamOutputBatch:
    return DraftTailStreamOutputBatch(outputs=list(outputs))


class TestDraftTailBufferBasics(unittest.TestCase):
    def _buf(self, required_tail_len=2):
        return DraftTailBuffer(verifier_rank=0, required_tail_len=required_tail_len)

    def test_open_tracks_committed_len_and_can_accept(self):
        buf = self._buf()
        buf.open_requests([_sync("r0", committed_len=3, drafter_rank=0)])
        self.assertTrue(buf.has_request("r0"))
        self.assertEqual(buf.get_committed_len("r0"), 3)
        # Unknown request reads back as None / False.
        self.assertFalse(buf.has_request("nope"))
        self.assertIsNone(buf.get_committed_len("nope"))

    def test_append_and_snapshot(self):
        buf = self._buf()
        buf.open_requests([_sync("r0", committed_len=3)])
        stats = buf.append_draft_stream_batch(
            _stream(
                _out("r0", base=3, pos=3, tok=100), _out("r0", base=3, pos=4, tok=101)
            ),
            collect_stats=True,
        )
        self.assertEqual(stats["num_appended_outputs"], 2)
        snap = buf.get_draft_snapshots([_FakeReq("r0")])[0]
        self.assertIsInstance(snap, DraftTailSnapshot)
        self.assertEqual(snap.committed_len, 3)
        self.assertEqual(snap.tail_tokens, [100, 101])
        self.assertEqual(snap.raw_tail_len, 2)

    def test_duplicate_is_idempotent(self):
        buf = self._buf()
        buf.open_requests([_sync("r0", committed_len=3)])
        buf.append_draft_stream_batch(_stream(_out("r0", base=3, pos=3, tok=100)))
        stats = buf.append_draft_stream_batch(
            _stream(_out("r0", base=3, pos=3, tok=100)), collect_stats=True
        )
        self.assertEqual(stats["num_duplicate_outputs"], 1)
        snap = buf.get_draft_snapshots([_FakeReq("r0")])[0]
        self.assertEqual(snap.tail_tokens, [100])

    def test_conflicting_token_raises_with_batch_context(self):
        buf = self._buf()
        buf.open_requests([_sync("r0", committed_len=3)])
        buf.append_draft_stream_batch(_stream(_out("r0", base=3, pos=3, tok=100)))
        with self.assertRaises(RuntimeError) as ctx:
            buf.append_draft_stream_batch(_stream(_out("r0", base=3, pos=3, tok=999)))
        msg = str(ctx.exception)
        self.assertIn("conflicts with buffered tail", msg)
        # The O(N^2) fix must still surface the whole-batch diagnostic on error.
        self.assertIn("batch_request_ids=", msg)
        self.assertIn("batch_new_token_id=", msg)

    def test_already_committed_output_ignored(self):
        buf = self._buf()
        buf.open_requests([_sync("r0", committed_len=3)])
        stats = buf.append_draft_stream_batch(
            _stream(_out("r0", base=3, pos=2, tok=55)), collect_stats=True
        )
        self.assertEqual(stats["num_already_committed_outputs"], 1)
        self.assertEqual(buf.get_committed_len("r0"), 3)

    def test_unknown_request_output_dropped(self):
        buf = self._buf()
        stats = buf.append_draft_stream_batch(
            _stream(_out("rx", base=0, pos=0, tok=1)), collect_stats=True
        )
        self.assertEqual(stats["num_unknown_request_outputs"], 1)

    def test_wrong_verifier_rank_raises(self):
        buf = self._buf()
        buf.open_requests([_sync("r0", committed_len=0)])
        with self.assertRaises(RuntimeError) as ctx:
            buf.append_draft_stream_batch(
                _stream(_out("r0", base=0, pos=0, tok=1, verifier_rank=7))
            )
        self.assertIn("targets the wrong verifier", str(ctx.exception))

    def test_wrong_drafter_rank_raises(self):
        buf = self._buf()
        buf.open_requests([_sync("r0", committed_len=0, drafter_rank=0)])
        with self.assertRaises(RuntimeError) as ctx:
            buf.append_draft_stream_batch(
                _stream(_out("r0", base=0, pos=0, tok=1, drafter_rank=5))
            )
        self.assertIn("Unexpected draft stream drafter rank", str(ctx.exception))

    def test_base_ahead_of_state_raises(self):
        buf = self._buf()
        buf.open_requests([_sync("r0", committed_len=0)])
        with self.assertRaises(RuntimeError) as ctx:
            buf.append_draft_stream_batch(_stream(_out("r0", base=5, pos=0, tok=1)))
        self.assertIn("base is ahead of verifier state", str(ctx.exception))

    def test_skips_buffered_tail_raises(self):
        buf = self._buf()
        buf.open_requests([_sync("r0", committed_len=0)])
        buf.append_draft_stream_batch(_stream(_out("r0", base=0, pos=0, tok=10)))
        with self.assertRaises(RuntimeError) as ctx:
            buf.append_draft_stream_batch(_stream(_out("r0", base=0, pos=5, tok=11)))
        self.assertIn("skips buffered tail", str(ctx.exception))

    def test_close_removes_state(self):
        buf = self._buf()
        buf.open_requests([_sync("r0", committed_len=3)])
        buf.append_draft_stream_batch(_stream(_out("r0", base=3, pos=3, tok=100)))
        buf.close_requests([_close("r0")])
        self.assertFalse(buf.has_request("r0"))
        self.assertIsNone(buf.get_committed_len("r0"))
        # Late stream outputs for a closed request are silently dropped.
        stats = buf.append_draft_stream_batch(
            _stream(_out("r0", base=3, pos=4, tok=101)), collect_stats=True
        )
        self.assertEqual(stats["num_unknown_request_outputs"], 1)


class TestDraftTailBufferCommitReconciliation(unittest.TestCase):
    def _buf(self):
        return DraftTailBuffer(verifier_rank=0, required_tail_len=2)

    def test_full_match_commit_advances_prefix(self):
        buf = self._buf()
        buf.open_requests([_sync("r", committed_len=0)])
        buf.append_draft_stream_batch(
            _stream(
                _out("r", base=0, pos=0, tok=10),
                _out("r", base=0, pos=1, tok=11),
                _out("r", base=0, pos=2, tok=12),
            )
        )
        buf.apply_verify_commits([_commit("r", pre=0, tokens=[10, 11])])
        self.assertEqual(buf.get_committed_len("r"), 2)
        snap = buf.get_draft_snapshots([_FakeReq("r")])[0]
        self.assertEqual(snap.committed_len, 2)
        self.assertEqual(snap.tail_tokens, [12])

    def test_partial_commit_then_pending_expected_match(self):
        buf = self._buf()
        buf.open_requests([_sync("r", committed_len=0)])
        # Drafter has only yielded one token but verifier commits three.
        buf.append_draft_stream_batch(_stream(_out("r", base=0, pos=0, tok=10)))
        buf.apply_verify_commits([_commit("r", pre=0, tokens=[10, 11, 12])])
        # One token matched -> committed_len 1; the other two become pending.
        self.assertEqual(buf.get_committed_len("r"), 1)
        snap = buf.get_draft_snapshots([_FakeReq("r")])[0]
        # While pending tokens exist, nothing is consumable.
        self.assertEqual(snap.tail_tokens, [])
        self.assertEqual(snap.raw_tail_len, 0)
        # Drafter now confirms the pending tokens in order.
        stats = buf.append_draft_stream_batch(
            _stream(
                _out("r", base=1, pos=1, tok=11),
                _out("r", base=2, pos=2, tok=12),
            ),
            collect_stats=True,
        )
        self.assertEqual(stats["num_pending_expected_match_outputs"], 2)
        self.assertEqual(buf.get_committed_len("r"), 3)

    def test_commit_mismatch_advances_can_accept_and_rejects_stale(self):
        buf = self._buf()
        buf.open_requests([_sync("r", committed_len=0)])
        # Drafter yielded [10, 99]; verifier commits [10, 11] -> mismatch at pos 1.
        buf.append_draft_stream_batch(
            _stream(
                _out("r", base=0, pos=0, tok=10),
                _out("r", base=0, pos=1, tok=99),
            )
        )
        buf.apply_verify_commits([_commit("r", pre=0, tokens=[10, 11])])
        # Matched prefix [10] committed; remaining draft tail dropped; expect 11.
        self.assertEqual(buf.get_committed_len("r"), 1)
        snap = buf.get_draft_snapshots([_FakeReq("r")])[0]
        self.assertEqual(snap.tail_tokens, [])
        # A stream output whose base predates the mismatch boundary is stale.
        # Reject (wrong token) keeps the boundary, then a correct confirm advances.
        stats = buf.append_draft_stream_batch(
            _stream(
                _out("r", base=0, pos=1, tok=11),  # base < can_accept(1) -> stale_base
                _out("r", base=1, pos=1, tok=77),  # wrong token -> reject
                _out("r", base=1, pos=1, tok=11),  # correct -> match
            ),
            collect_stats=True,
        )
        self.assertEqual(stats["num_stale_base_outputs"], 1)
        self.assertEqual(stats["num_pending_expected_reject_outputs"], 1)
        self.assertEqual(stats["num_pending_expected_match_outputs"], 1)
        self.assertEqual(buf.get_committed_len("r"), 2)

    def test_stale_gap_after_can_accept_lags_committed(self):
        buf = self._buf()
        buf.open_requests([_sync("r", committed_len=3)])
        buf.append_draft_stream_batch(
            _stream(
                _out("r", base=3, pos=3, tok=100), _out("r", base=3, pos=4, tok=101)
            )
        )
        # Commit one matched token: committed_len -> 4, can_accept stays 3.
        buf.apply_verify_commits([_commit("r", pre=3, tokens=[100])])
        self.assertEqual(buf.get_committed_len("r"), 4)
        # base in [can_accept=3, committed=4) and pos beyond buffer end -> stale_gap.
        stats = buf.append_draft_stream_batch(
            _stream(_out("r", base=3, pos=10, tok=123)), collect_stats=True
        )
        self.assertEqual(stats["num_stale_gap_outputs"], 1)

    def test_noncontiguous_commit_raises(self):
        buf = self._buf()
        buf.open_requests([_sync("r", committed_len=0)])
        with self.assertRaises(RuntimeError) as ctx:
            buf.apply_verify_commits([_commit("r", pre=5, tokens=[1])])
        self.assertIn(
            "does not match draft-tail confirmed plus pending prefix",
            str(ctx.exception),
        )

    def test_commit_for_unknown_request_is_noop(self):
        buf = self._buf()
        # No open; commit should not raise and should not create state.
        buf.apply_verify_commits([_commit("ghost", pre=0, tokens=[1])])
        self.assertFalse(buf.has_request("ghost"))


class TestDraftTailBufferControlBatchAndSnapshots(unittest.TestCase):
    def _buf(self, required_tail_len=2):
        return DraftTailBuffer(verifier_rank=0, required_tail_len=required_tail_len)

    def test_apply_control_batch_open_commit_close_in_order(self):
        buf = self._buf()
        # Open + commit (full match needs a tail; here commit with no tail -> pending).
        buf.open_requests([_sync("r", committed_len=0)])
        buf.append_draft_stream_batch(_stream(_out("r", base=0, pos=0, tok=10)))
        stats = buf.apply_control_batch(
            DraftControlBatch(
                dst_drafter_rank=0,
                verify_commit_messages=[_commit("r", pre=0, tokens=[10])],
            ),
            collect_stats=True,
        )
        self.assertEqual(stats["commit_rids"], ["r"])
        self.assertEqual(stats["committed_segment_lens_by_req"], [1])
        self.assertEqual(stats["matched_tail_lens_by_req"], [1])
        self.assertEqual(buf.get_committed_len("r"), 1)
        # Close via control batch removes the state.
        buf.apply_control_batch(
            DraftControlBatch(dst_drafter_rank=0, close_messages=[_close("r")])
        )
        self.assertFalse(buf.has_request("r"))

    def test_snapshot_allow_partial_returns_immediately(self):
        buf = self._buf(required_tail_len=4)
        buf.open_requests([_sync("r", committed_len=0)])
        # No tail tokens yet; allow_partial must not block.
        snaps = buf.get_draft_snapshots([_FakeReq("r")], allow_partial=True)
        self.assertEqual(snaps[0].tail_tokens, [])

    def test_snapshot_no_partial_returns_when_tokens_present(self):
        buf = self._buf(required_tail_len=2)
        buf.open_requests([_sync("r", committed_len=0)])
        buf.append_draft_stream_batch(
            _stream(_out("r", base=0, pos=0, tok=10), _out("r", base=0, pos=1, tok=11))
        )
        # Enough tokens already buffered -> no blocking even with allow_partial=False.
        snaps = buf.get_draft_snapshots([_FakeReq("r")], allow_partial=False)
        self.assertEqual(snaps[0].tail_tokens, [10, 11])

    def test_include_raw_tail_tokens(self):
        buf = self._buf()
        buf.open_requests([_sync("r", committed_len=0)])
        buf.append_draft_stream_batch(_stream(_out("r", base=0, pos=0, tok=10)))
        snap = buf.get_draft_snapshots([_FakeReq("r")], include_raw_tail_tokens=True)[0]
        self.assertEqual(snap.raw_tail_tokens, [10])


class TestDraftTailBufferWaiting(unittest.TestCase):
    """Threaded tests for the blocking wait path (allow_partial=False)."""

    def test_waiter_wakes_when_tokens_arrive(self):
        buf = DraftTailBuffer(verifier_rank=0, required_tail_len=2)
        buf.open_requests([_sync("r", committed_len=0)])
        result = {}

        def waiter():
            try:
                snaps = buf.get_draft_snapshots([_FakeReq("r")], allow_partial=False)
                result["tail"] = snaps[0].tail_tokens
            except Exception as exc:  # pragma: no cover - failure path
                result["error"] = exc

        t = threading.Thread(target=waiter)
        t.start()
        # Produce the required tokens; the append notifies the condition.
        buf.append_draft_stream_batch(
            _stream(_out("r", base=0, pos=0, tok=10), _out("r", base=0, pos=1, tok=11))
        )
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive(), "waiter did not wake up")
        self.assertNotIn("error", result)
        self.assertEqual(result["tail"], [10, 11])

    def test_close_unblocks_waiter_with_error(self):
        buf = DraftTailBuffer(verifier_rank=0, required_tail_len=2)
        buf.open_requests([_sync("r", committed_len=0)])
        result = {}

        def waiter():
            try:
                buf.wait_for_draft_tokens(["r"], 2)
            except RuntimeError as exc:
                result["error"] = exc

        t = threading.Thread(target=waiter)
        t.start()
        buf.close()
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive(), "waiter did not wake up on close")
        self.assertIn("error", result)
        self.assertIn("closed while waiting", str(result["error"]))


class TestDraftTailBufferPendingEdgeCases(unittest.TestCase):
    """Coverage for the pending_expected_tokens branches (drafter lagging the verifier)."""

    def _buf(self):
        return DraftTailBuffer(verifier_rank=0, required_tail_len=2)

    def _into_mismatch_pending_state(self, buf):
        """Drive r into committed_len=1, can_accept=1, pending=[11] via a mismatch commit."""
        buf.open_requests([_sync("r", committed_len=0)])
        buf.append_draft_stream_batch(
            _stream(_out("r", base=0, pos=0, tok=10), _out("r", base=0, pos=1, tok=99))
        )
        buf.apply_verify_commits([_commit("r", pre=0, tokens=[10, 11])])

    def test_second_commit_while_pending_extends_queue(self):
        # The HIGH-risk path: verifier commits again while the drafter is still
        # behind, so a second commit appends to an already-non-empty pending queue.
        buf = self._buf()
        buf.open_requests([_sync("r", committed_len=0)])
        buf.append_draft_stream_batch(_stream(_out("r", base=0, pos=0, tok=10)))
        # First commit: 1 of 2 tokens match the tail -> committed_len=1, pending=[11].
        buf.apply_verify_commits([_commit("r", pre=0, tokens=[10, 11])])
        self.assertEqual(buf.get_committed_len("r"), 1)
        # Second commit arrives while pending is already non-empty. Its pre-verify
        # length must equal committed_len + pending_len (= 1 + 1 = 2); it extends
        # the pending queue rather than matching a (currently empty) tail.
        stats = buf.apply_control_batch(
            DraftControlBatch(
                dst_drafter_rank=0,
                verify_commit_messages=[_commit("r", pre=2, tokens=[12])],
            ),
            collect_stats=True,
        )
        self.assertEqual(stats["pending_expected_lens_before_by_req"], [1])
        self.assertEqual(stats["pending_expected_added_by_req"], [1])
        self.assertEqual(stats["pending_expected_lens_after_by_req"], [2])
        self.assertEqual(stats["matched_tail_lens_by_req"], [0])
        self.assertEqual(buf.get_committed_len("r"), 1)
        # The drafter then catches up and confirms both queued tokens in order.
        buf.append_draft_stream_batch(
            _stream(
                _out("r", base=1, pos=1, tok=11),
                _out("r", base=2, pos=2, tok=12),
            )
        )
        self.assertEqual(buf.get_committed_len("r"), 3)

    def test_already_committed_in_pending_path(self):
        buf = self._buf()
        self._into_mismatch_pending_state(
            buf
        )  # committed_len=1, can_accept=1, pending=[11]
        # base==can_accept (not stale), token_pos < committed_len -> already_committed.
        stats = buf.append_draft_stream_batch(
            _stream(_out("r", base=1, pos=0, tok=11)), collect_stats=True
        )
        self.assertEqual(stats["num_already_committed_outputs"], 1)
        self.assertEqual(buf.get_committed_len("r"), 1)

    def test_pending_expected_gap_for_base_or_pos_ahead(self):
        buf = self._buf()
        self._into_mismatch_pending_state(
            buf
        )  # committed_len=1, can_accept=1, pending=[11]
        # Output A: base ahead of committed_len -> pending_expected_gap.
        # Output B: base == committed_len but token_pos ahead -> pending_expected_gap.
        stats = buf.append_draft_stream_batch(
            _stream(
                _out("r", base=2, pos=1, tok=11),
                _out("r", base=1, pos=2, tok=11),
            ),
            collect_stats=True,
        )
        self.assertEqual(stats["num_pending_expected_gap_outputs"], 2)
        # Neither output advances the queue; the request stays unresolved.
        self.assertEqual(buf.get_committed_len("r"), 1)

    def test_tail_nonempty_while_pending_commit_raises(self):
        # White-box defensive guard: the public API can never produce tail and
        # pending both non-empty, so force the invalid state directly to confirm
        # the last-resort RuntimeError still fires.
        buf = self._buf()
        buf.open_requests([_sync("r", committed_len=0)])
        state = buf._states["r"]
        state.committed_len = 1
        state.pending_expected_tokens = deque([11])
        state.tail_tokens = [99]
        with self.assertRaises(RuntimeError) as ctx:
            buf.apply_verify_commits([_commit("r", pre=2, tokens=[12])])
        self.assertIn("must be empty while expected prefix", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
