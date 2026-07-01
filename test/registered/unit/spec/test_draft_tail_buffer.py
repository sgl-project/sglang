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
from sglang.test.test_utils import CustomTestCase

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
        committed_outputs=[0] * committed_len,
    )


def _commit(rid, *, pre, tokens, drafter_rank=0, src_verifier_rank=0) -> VerifyCommit:
    return VerifyCommit(
        request_id=rid,
        src_verifier_rank=src_verifier_rank,
        dst_drafter_rank=drafter_rank,
        pre_verify_committed_len=pre,
        committed_tokens=list(tokens),
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
        new_token=tok,
    )


def _stream(*outputs) -> DraftTailStreamOutputBatch:
    return DraftTailStreamOutputBatch(outputs=list(outputs))


class TestDraftTailBufferBasics(CustomTestCase):
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
        self.assertIn("batch_new_token=", msg)

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


class TestDraftTailBufferCommitReconciliation(CustomTestCase):
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


class TestDraftTailBufferControlBatchAndSnapshots(CustomTestCase):
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
        self.assertEqual(stats.commit_rids, ["r"])
        self.assertEqual(stats.committed_segment_lens, [1])
        self.assertEqual(stats.matched_tail_lens, [1])
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


class TestDraftTailBufferWaiting(CustomTestCase):
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

    def test_waiter_blocks_on_pending_then_wakes(self):
        # A request stuck in the pending-expected state (committed_len advanced,
        # tail empty) must keep a blocking waiter parked even though tokens were
        # "committed": the consumable tail is 0 until the drafter confirms the
        # pending token. This exercises the pending-queue branch of
        # _has_min_draft_tokens_locked, distinct from the tail-too-short branch.
        buf = DraftTailBuffer(verifier_rank=0, required_tail_len=2)
        buf.open_requests([_sync("r", committed_len=0)])
        buf.append_draft_stream_batch(_stream(_out("r", base=0, pos=0, tok=10)))
        # Commit [10, 11]: 10 matches the tail, 11 outruns the drafter -> pending.
        buf.apply_verify_commits([_commit("r", pre=0, tokens=[10, 11])])
        result = {}

        def waiter():
            try:
                snaps = buf.get_draft_snapshots([_FakeReq("r")], allow_partial=False)
                result["tail"] = snaps[0].tail_tokens
            except Exception as exc:  # pragma: no cover - failure path
                result["error"] = exc

        t = threading.Thread(target=waiter)
        t.start()
        # The waiter must NOT return while the request is blocked on pending.
        t.join(timeout=0.2)
        self.assertTrue(t.is_alive(), "waiter should stay blocked while pending")
        # Drafter confirms the pending token, then streams enough tail to satisfy
        # required_tail_len; the append notifies the waiting condition.
        buf.append_draft_stream_batch(
            _stream(
                _out("r", base=1, pos=1, tok=11),  # confirms pending -> committed=2
                _out("r", base=2, pos=2, tok=12),
                _out("r", base=2, pos=3, tok=13),
            )
        )
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive(), "waiter did not wake after pending resolved")
        self.assertNotIn("error", result)
        self.assertEqual(result["tail"], [12, 13])


class TestDraftTailBufferPendingEdgeCases(CustomTestCase):
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
        self.assertEqual(stats.pending_expected_lens_before, [1])
        self.assertEqual(stats.pending_expected_added, [1])
        self.assertEqual(stats.pending_expected_lens_after, [2])
        self.assertEqual(stats.matched_tail_lens, [0])
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


class TestDraftTailBufferMultiRequest(CustomTestCase):
    """Batch-level coverage: one buffer driving several distinct requests at once.

    The per-request stats arrays (``*_by_req``) and the ``index_by_request_id``
    bookkeeping are only exercised when a single batch carries more than one
    distinct rid, so these tests interleave outputs across requests (plus a
    same-batch duplicate) and assert the parallel arrays line up with the rids.
    """

    def _buf(self):
        return DraftTailBuffer(verifier_rank=0, required_tail_len=2)

    def test_interleaved_stream_batch_tracks_each_request(self):
        buf = self._buf()
        buf.open_requests(
            [
                _sync("r0", committed_len=0),
                _sync("r1", committed_len=0),
                _sync("r2", committed_len=0),
            ]
        )
        # One batch interleaves three requests; r0 repeats pos=1 within the batch
        # (a same-batch duplicate), so its raw-output count and appended count
        # diverge while r1/r2 stay in lockstep.
        stats = buf.append_draft_stream_batch(
            _stream(
                _out("r0", base=0, pos=0, tok=10),
                _out("r1", base=0, pos=0, tok=20),
                _out("r0", base=0, pos=1, tok=11),
                _out("r2", base=0, pos=0, tok=30),
                _out("r0", base=0, pos=1, tok=11),  # duplicate within the batch
                _out("r1", base=0, pos=1, tok=21),
            ),
            collect_stats=True,
        )
        # rids follow first-appearance order; the duplicate r0 is deduped here.
        self.assertEqual(stats["rids"], ["r0", "r1", "r2"])
        # draft_token_lens counts every output (incl. the duplicate); appended
        # excludes it -> r0 is [3] vs [2], r1/r2 match.
        self.assertEqual(stats["draft_token_lens_by_req"], [3, 2, 1])
        self.assertEqual(stats["appended_token_lens_by_req"], [2, 2, 1])
        self.assertEqual(stats["num_appended_outputs"], 5)
        self.assertEqual(stats["num_duplicate_outputs"], 1)
        # The append-side after-len arrays are per-request and ordered like rids.
        self.assertEqual(stats["tail_lens_after_by_req"], [2, 2, 1])
        self.assertEqual(stats["consumable_tail_lens_after_by_req"], [2, 2, 1])
        self.assertEqual(stats["committed_lens_after_by_req"], [0, 0, 0])
        self.assertEqual(stats["pending_expected_lens_after_by_req"], [0, 0, 0])
        # Each request keeps an independent tail.
        snaps = buf.get_draft_snapshots(
            [_FakeReq("r0"), _FakeReq("r1"), _FakeReq("r2")]
        )
        self.assertEqual(
            [snap.tail_tokens for snap in snaps], [[10, 11], [20, 21], [30]]
        )

    def test_control_batch_commits_multiple_requests(self):
        buf = self._buf()
        buf.open_requests([_sync("a", committed_len=0), _sync("b", committed_len=0)])
        buf.append_draft_stream_batch(
            _stream(
                _out("a", base=0, pos=0, tok=10),
                _out("a", base=0, pos=1, tok=11),
                _out("b", base=0, pos=0, tok=20),
            )
        )
        # a: its single committed token fully matches the tail. b: the drafter
        # lags (verifier commits two, only one is buffered) -> one matches and one
        # becomes pending. The commit-side *_by_req arrays must track both.
        stats = buf.apply_control_batch(
            DraftControlBatch(
                dst_drafter_rank=0,
                verify_commit_messages=[
                    _commit("a", pre=0, tokens=[10]),
                    _commit("b", pre=0, tokens=[20, 21]),
                ],
            ),
            collect_stats=True,
        )
        self.assertEqual(stats.commit_rids, ["a", "b"])
        self.assertEqual(stats.committed_segment_lens, [1, 2])
        self.assertEqual(stats.matched_tail_lens, [1, 1])
        self.assertEqual(stats.pending_expected_lens_after, [0, 1])
        self.assertEqual(buf.get_committed_len("a"), 1)
        self.assertEqual(buf.get_committed_len("b"), 1)


class TestDraftTailBufferLifecycle(CustomTestCase):
    """End-to-end trajectory through every reconciliation phase in sequence.

    Isolated single-transition tests can miss state that accumulates across
    steps. This drives one request through open -> stream -> full-match commit ->
    partial commit (pending) -> drafter confirm -> re-stream -> close, asserting
    committed_len / tail / pending after each step.
    """

    def _snapshot(self, buf, rid):
        return buf.get_draft_snapshots([_FakeReq(rid)])[0]

    def test_full_request_trajectory(self):
        buf = DraftTailBuffer(verifier_rank=0, required_tail_len=2)
        # Open through a control batch (the scheduler's real combined entry point)
        # rather than open_requests(), exercising the sync_messages leg.
        buf.apply_control_batch(
            DraftControlBatch(
                dst_drafter_rank=0, sync_messages=[_sync("r", committed_len=0)]
            )
        )
        self.assertTrue(buf.has_request("r"))
        self.assertEqual(buf.get_committed_len("r"), 0)

        # 1) Drafter streams three guesses ahead of the committed prefix.
        buf.append_draft_stream_batch(
            _stream(
                _out("r", base=0, pos=0, tok=10),
                _out("r", base=0, pos=1, tok=11),
                _out("r", base=0, pos=2, tok=12),
            )
        )
        snap = self._snapshot(buf, "r")
        self.assertEqual(snap.committed_len, 0)
        self.assertEqual(snap.tail_tokens, [10, 11, 12])

        # 2) Verifier commits a fully-matched prefix; the tail shrinks.
        buf.apply_verify_commits([_commit("r", pre=0, tokens=[10, 11])])
        snap = self._snapshot(buf, "r")
        self.assertEqual(snap.committed_len, 2)
        self.assertEqual(snap.tail_tokens, [12])

        # 3) Verifier commits [12, 13]: 12 matches the buffered tail, 13 outruns
        #    the drafter and becomes pending. While pending, nothing is consumable.
        buf.apply_verify_commits([_commit("r", pre=2, tokens=[12, 13])])
        snap = self._snapshot(buf, "r")
        self.assertEqual(snap.committed_len, 3)
        self.assertEqual(snap.tail_tokens, [])
        self.assertEqual(snap.raw_tail_len, 0)

        # 4) Drafter catches up and confirms the pending token.
        stats = buf.append_draft_stream_batch(
            _stream(_out("r", base=3, pos=3, tok=13)), collect_stats=True
        )
        self.assertEqual(stats["num_pending_expected_match_outputs"], 1)
        self.assertEqual(buf.get_committed_len("r"), 4)

        # 5) Drafter resumes buffering fresh guesses ahead of the prefix.
        buf.append_draft_stream_batch(
            _stream(
                _out("r", base=4, pos=4, tok=14),
                _out("r", base=4, pos=5, tok=15),
            )
        )
        snap = self._snapshot(buf, "r")
        self.assertEqual(snap.committed_len, 4)
        self.assertEqual(snap.tail_tokens, [14, 15])

        # 6) Close tears the request state down.
        buf.close_requests([_close("r")])
        self.assertFalse(buf.has_request("r"))
        self.assertIsNone(buf.get_committed_len("r"))


if __name__ == "__main__":
    unittest.main()
