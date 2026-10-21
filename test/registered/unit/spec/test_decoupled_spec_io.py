"""Unit tests for srt/speculative/decoupled_spec_io.

decoupled_spec_io is the schema-only IPC layer for decoupled speculative
decoding. These tests drive its logic (commit validation, segment coalescing /
contiguity / prefix extraction, and the message envelope) on CPU; there is no
GPU or transport here.
"""

import unittest

from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftReqKey,
    VerifierCommitSegment,
    VerifyCommit,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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


class TestDraftMeshMessageEnvelope(CustomTestCase):
    def test_from_control_batch_sets_discriminant_and_slot(self):
        batch = DraftControlBatch(dst_drafter_rank=0)
        msg = DraftMeshMessage.from_control_batch(batch)
        self.assertEqual(msg.message_type, DraftMeshMessageType.CONTROL_BATCH)
        self.assertIs(msg.control_batch, batch)
        self.assertIsNone(msg.tail_stream_output_batch)

    def test_from_tail_stream_output_batch_sets_discriminant_and_slot(self):
        from sglang.srt.speculative.decoupled_spec_io import DraftTailStreamOutputBatch

        batch = DraftTailStreamOutputBatch()
        msg = DraftMeshMessage.from_tail_stream_output_batch(batch)
        self.assertEqual(
            msg.message_type, DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH
        )
        self.assertIs(msg.tail_stream_output_batch, batch)
        self.assertIsNone(msg.control_batch)


if __name__ == "__main__":
    unittest.main(verbosity=3)
