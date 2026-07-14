"""Unit tests for srt/speculative/decoupled_spec_io.

decoupled_spec_io is the schema-only IPC layer for decoupled speculative
decoding: protocol message dataclasses, the cross-process request id codec, and
the drafter-side reconciliation helpers. These tests drive the real logic (id
round-trip + parse errors, commit validation, segment coalescing / contiguity /
prefix extraction, and inbox routing) on CPU; there is no GPU or transport here.
"""

import unittest

from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftEnumerationBuffer,
    DraftEnumerationBufferBatch,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftReqKey,
    DraftSync,
    VerifierCommitSegment,
    VerifyCommit,
    build_draft_scheduler_rid,
    parse_draft_scheduler_rid,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _commit(rid, *, pre, tokens, src_verifier_rank=0, drafter_rank=0) -> VerifyCommit:
    return VerifyCommit(
        request_id=rid,
        src_verifier_rank=src_verifier_rank,
        dst_drafter_rank=drafter_rank,
        pre_verify_committed_len=pre,
        committed_tokens=list(tokens),
    )


def _segment(
    rid, *, pre=0, drafter_rank=0, src_verifier_rank=0
) -> VerifierCommitSegment:
    return VerifierCommitSegment(
        draft_key=DraftReqKey(src_verifier_rank=src_verifier_rank, request_id=rid),
        dst_drafter_rank=drafter_rank,
        pre_verify_committed_len=pre,
    )


class TestDraftSchedulerRid(CustomTestCase):
    def test_round_trip(self):
        key = DraftReqKey(src_verifier_rank=3, request_id="req-1")
        rid = build_draft_scheduler_rid(key)
        self.assertEqual(rid, "draft:3:req-1")
        self.assertEqual(parse_draft_scheduler_rid(rid), key)

    def test_request_id_containing_colon_round_trips(self):
        # request_id may contain ':' — parse splits on the first ':' only.
        key = DraftReqKey(src_verifier_rank=1, request_id="a:b:c")
        self.assertEqual(parse_draft_scheduler_rid(build_draft_scheduler_rid(key)), key)

    def test_parse_invalid_rid_raises(self):
        for bad in ["no-prefix", "draft:", "draft:0:", "draft:notint:r"]:
            with self.subTest(rid=bad):
                with self.assertRaises(ValueError):
                    parse_draft_scheduler_rid(bad)


class TestVerifyCommitValidation(CustomTestCase):
    def test_empty_tokens_raises(self):
        with self.assertRaises(ValueError):
            _commit("r", pre=0, tokens=[]).validate_committed_tokens()

    def test_negative_pre_len_raises(self):
        with self.assertRaises(ValueError):
            _commit("r", pre=-1, tokens=[1]).validate_committed_tokens()

    def test_valid_commit_passes(self):
        # Should not raise.
        _commit("r", pre=0, tokens=[1, 2]).validate_committed_tokens()


class TestVerifierCommitSegment(CustomTestCase):
    def test_append_coalesces_contiguous_commits(self):
        seg = _segment("r", pre=0)
        seg.append_message(_commit("r", pre=0, tokens=[10, 11]))
        self.assertEqual(seg.committed_tokens, [10, 11])
        self.assertEqual(seg.end_committed_len, 2)
        seg.append_message(_commit("r", pre=2, tokens=[12]))
        self.assertEqual(seg.committed_tokens, [10, 11, 12])
        self.assertEqual(seg.end_committed_len, 3)

    def test_append_wrong_request_raises(self):
        seg = _segment("r", pre=0)
        with self.assertRaises(RuntimeError):
            seg.append_message(_commit("other", pre=0, tokens=[1]))

    def test_append_wrong_drafter_rank_raises(self):
        seg = _segment("r", pre=0, drafter_rank=0)
        with self.assertRaises(RuntimeError):
            seg.append_message(_commit("r", pre=0, tokens=[1], drafter_rank=7))

    def test_append_non_contiguous_raises(self):
        seg = _segment("r", pre=0)
        seg.append_message(_commit("r", pre=0, tokens=[10]))  # end -> 1
        with self.assertRaises(RuntimeError):
            seg.append_message(_commit("r", pre=5, tokens=[11]))  # gap

    def test_append_runs_message_validation(self):
        seg = _segment("r", pre=0)
        with self.assertRaises(ValueError):
            seg.append_message(_commit("r", pre=0, tokens=[]))  # empty -> validate

    def test_extract_prefix_splits_segment(self):
        seg = _segment("r", pre=0)
        seg.append_message(_commit("r", pre=0, tokens=[10, 11, 12, 13]))
        prefix = seg.extract_prefix(2)
        self.assertEqual(prefix.committed_tokens, [10, 11])
        self.assertEqual(prefix.pre_verify_committed_len, 0)
        # Remainder stays in the original segment, with pre advanced by 2.
        self.assertEqual(seg.committed_tokens, [12, 13])
        self.assertEqual(seg.pre_verify_committed_len, 2)
        self.assertEqual(seg.end_committed_len, 4)

    def test_extract_prefix_bounds(self):
        seg = _segment("r", pre=0)
        seg.append_message(_commit("r", pre=0, tokens=[10, 11]))
        with self.assertRaises(ValueError):
            seg.extract_prefix(0)
        with self.assertRaises(ValueError):
            seg.extract_prefix(3)  # exceeds segment length


class TestDraftControlInbox(CustomTestCase):
    def _inbox(self):
        from sglang.srt.speculative.decoupled_spec_io import DraftControlInbox

        return DraftControlInbox()

    def _sync(self, rid, drafter_rank=0):
        return DraftSync(
            request_id=rid, src_verifier_rank=0, dst_drafter_rank=drafter_rank
        )

    def _close(self, rid, drafter_rank=0):
        return DraftClose(
            request_id=rid,
            src_verifier_rank=0,
            dst_drafter_rank=drafter_rank,
            reason="x",
        )

    def test_add_control_batch_routes_each_message_type(self):
        inbox = self._inbox()
        inbox.add_control_batch_locked(
            DraftControlBatch(
                dst_drafter_rank=0,
                sync_messages=[self._sync("s")],
                verify_commit_messages=[_commit("c", pre=0, tokens=[1])],
                close_messages=[self._close("x")],
            )
        )
        self.assertEqual([m.request_id for m in inbox.sync_messages], ["s"])
        self.assertIn(DraftReqKey(0, "c"), inbox.verifier_commit_segments)
        self.assertIn(DraftReqKey(0, "x"), inbox.close_keys)

    def test_close_drops_pending_segment_and_sync(self):
        inbox = self._inbox()
        inbox.add_control_batch_locked(
            DraftControlBatch(
                dst_drafter_rank=0,
                sync_messages=[self._sync("r")],
                verify_commit_messages=[_commit("r", pre=0, tokens=[1])],
            )
        )
        inbox.add_close_key_locked(DraftReqKey(0, "r"))
        self.assertEqual(inbox.sync_messages, [])
        self.assertNotIn(DraftReqKey(0, "r"), inbox.verifier_commit_segments)
        self.assertIn(DraftReqKey(0, "r"), inbox.close_keys)

    def test_verify_commit_for_closed_key_is_ignored(self):
        inbox = self._inbox()
        inbox.add_close_key_locked(DraftReqKey(0, "r"))
        inbox.add_verify_commit_locked(_commit("r", pre=0, tokens=[1]))
        self.assertNotIn(DraftReqKey(0, "r"), inbox.verifier_commit_segments)

    def test_extract_ready_controls_full_consume(self):
        inbox = self._inbox()
        inbox.add_control_batch_locked(
            DraftControlBatch(
                dst_drafter_rank=0,
                sync_messages=[self._sync("s")],
                verify_commit_messages=[_commit("c", pre=0, tokens=[1, 2])],
                close_messages=[self._close("x")],
            )
        )
        ready = inbox.extract_ready_controls_locked(
            lambda seg: len(seg.committed_tokens)
        )
        self.assertEqual([m.request_id for m in ready.sync_messages], ["s"])
        self.assertEqual({k.request_id for k in ready.close_keys}, {"x"})
        self.assertEqual(len(ready.ready_commit_segments), 1)
        self.assertEqual(ready.ready_commit_segments[0].committed_tokens, [1, 2])
        # Fully consumed -> the segment is gone; inbox drained.
        self.assertTrue(inbox.is_empty())

    def test_extract_ready_controls_zero_consumable_keeps_segment(self):
        inbox = self._inbox()
        inbox.add_verify_commit_locked(_commit("c", pre=0, tokens=[1, 2]))
        ready = inbox.extract_ready_controls_locked(lambda seg: 0)
        self.assertEqual(ready.ready_commit_segments, [])
        # Segment is left buffered for a later step.
        self.assertIn(DraftReqKey(0, "c"), inbox.verifier_commit_segments)

    def test_extract_ready_controls_partial_consume_buffers_remainder(self):
        inbox = self._inbox()
        inbox.add_verify_commit_locked(_commit("c", pre=0, tokens=[1, 2, 3]))
        ready = inbox.extract_ready_controls_locked(lambda seg: 1)
        self.assertEqual(ready.ready_commit_segments[0].committed_tokens, [1])
        # Remainder [2, 3] stays buffered with pre advanced to 1.
        seg = inbox.verifier_commit_segments[DraftReqKey(0, "c")]
        self.assertEqual(seg.committed_tokens, [2, 3])
        self.assertEqual(seg.pre_verify_committed_len, 1)

    def test_close_in_same_batch_drops_sync_and_commit(self):
        # add_control_batch applies close first, so a same-key sync/commit in the
        # same batch is dropped/ignored: close wins within one batch.
        inbox = self._inbox()
        inbox.add_control_batch_locked(
            DraftControlBatch(
                dst_drafter_rank=0,
                sync_messages=[self._sync("r")],
                verify_commit_messages=[_commit("r", pre=0, tokens=[1])],
                close_messages=[self._close("r")],
            )
        )
        self.assertEqual(inbox.sync_messages, [])
        self.assertNotIn(DraftReqKey(0, "r"), inbox.verifier_commit_segments)
        self.assertEqual({k.request_id for k in inbox.close_keys}, {"r"})

    def test_two_commits_same_key_coalesce_in_inbox(self):
        # A second commit for an existing key appends to the buffered segment.
        inbox = self._inbox()
        inbox.add_verify_commit_locked(_commit("r", pre=0, tokens=[1, 2]))
        inbox.add_verify_commit_locked(_commit("r", pre=2, tokens=[3]))
        seg = inbox.verifier_commit_segments[DraftReqKey(0, "r")]
        self.assertEqual(seg.committed_tokens, [1, 2, 3])
        self.assertEqual(seg.end_committed_len, 3)


def _enum_buffer(
    rid="r",
    *,
    num_steps=2,
    fanout=3,
    base_committed_len=0,
    src_drafter_rank=0,
    dst_verifier_rank=0,
    tokens=None,
) -> DraftEnumerationBuffer:
    if tokens is None:
        # A well-formed (K + 1) * F * K flat block.
        tokens = tuple(range((num_steps + 1) * fanout * num_steps))
    return DraftEnumerationBuffer(
        src_drafter_rank=src_drafter_rank,
        dst_verifier_rank=dst_verifier_rank,
        request_id=rid,
        base_committed_len=base_committed_len,
        num_steps=num_steps,
        fanout=fanout,
        tokens=tokens,
    )


class TestDraftEnumerationBuffer(CustomTestCase):
    def test_num_tokens_and_valid_block_passes(self):
        # num_tokens is the (K + 1) * F * K layout size; a matching block
        # validates without raising.
        buf = _enum_buffer(num_steps=2, fanout=3)
        self.assertEqual(buf.num_tokens, (2 + 1) * 3 * 2)
        self.assertEqual(len(buf.tokens), buf.num_tokens)
        buf.validate()  # should not raise

    def test_num_steps_below_one_raises(self):
        with self.assertRaises(ValueError):
            _enum_buffer(num_steps=0, fanout=3, tokens=()).validate()

    def test_fanout_below_one_raises(self):
        with self.assertRaises(ValueError):
            _enum_buffer(num_steps=2, fanout=0, tokens=()).validate()

    def test_negative_base_committed_len_raises(self):
        with self.assertRaises(ValueError):
            _enum_buffer(base_committed_len=-1).validate()

    def test_wrong_tokens_length_raises(self):
        # One token short of the (K + 1) * F * K block.
        buf = _enum_buffer(num_steps=2, fanout=3)
        short = DraftEnumerationBuffer(
            src_drafter_rank=buf.src_drafter_rank,
            dst_verifier_rank=buf.dst_verifier_rank,
            request_id=buf.request_id,
            base_committed_len=buf.base_committed_len,
            num_steps=buf.num_steps,
            fanout=buf.fanout,
            tokens=buf.tokens[:-1],
        )
        with self.assertRaises(ValueError):
            short.validate()

    def test_draft_key_uses_dst_verifier_rank(self):
        # The owning verifier is the destination rank; draft_key must key on it
        # so the drafter-side table is unambiguous across verifier ranks.
        buf = _enum_buffer("req-9", dst_verifier_rank=4)
        self.assertEqual(
            buf.draft_key, DraftReqKey(src_verifier_rank=4, request_id="req-9")
        )


class TestDraftMeshMessageEnvelope(CustomTestCase):
    def test_from_control_batch_sets_discriminant_and_slot(self):
        batch = DraftControlBatch(dst_drafter_rank=0)
        msg = DraftMeshMessage.from_control_batch(batch)
        self.assertEqual(msg.message_type, DraftMeshMessageType.CONTROL_BATCH)
        self.assertIs(msg.control_batch, batch)
        self.assertIsNone(msg.enumeration_buffer_batch)

    def test_from_enumeration_buffer_batch_sets_discriminant_and_slot(self):
        batch = DraftEnumerationBufferBatch(buffers=[_enum_buffer()])
        msg = DraftMeshMessage.from_enumeration_buffer_batch(batch)
        self.assertEqual(
            msg.message_type, DraftMeshMessageType.ENUMERATION_BUFFER_BATCH
        )
        self.assertIs(msg.enumeration_buffer_batch, batch)
        self.assertIsNone(msg.control_batch)


if __name__ == "__main__":
    unittest.main(verbosity=3)
