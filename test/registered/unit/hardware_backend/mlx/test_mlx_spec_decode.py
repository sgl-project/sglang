from __future__ import annotations

import unittest

from sglang.srt.hardware_backend.mlx.spec_decode import (
    MlxVerifySegment,
    build_linear_verify_queries,
    verify_greedy_segment,
    verify_greedy_segments,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


class TestMlxGreedyVerifier(unittest.TestCase):
    def test_accept_reject_and_plain_segments(self):
        cases = (
            # draft row, valid count, target rows, emitted, accepted
            ((-1,), 0, (7,), (7,), 0),
            ((5,), 1, (5, 9), (5, 9), 1),
            ((5,), 1, (6, 9), (6,), 0),
            ((2, 3, 4), 3, (2, 3, 4, 8), (2, 3, 4, 8), 3),
            ((2, 3, 4), 3, (2, 7, 9, 8), (2, 7), 1),
            ((2, 3, 4), 3, (9, 7, 8, 6), (9,), 0),
        )
        for draft, valid, targets, emitted, accepted in cases:
            with self.subTest(draft=draft, targets=targets):
                segment = MlxVerifySegment(
                    request_id="r",
                    draft_tokens=draft,
                    valid_draft_count=valid,
                    invalid_draft_count=len(draft) - valid,
                    target_token_ids=targets,
                    verification_query_count=valid + 1,
                )
                decision = verify_greedy_segment(segment)
                self.assertEqual(decision.emitted_token_ids, emitted)
                self.assertEqual(decision.accepted_draft_count, accepted)
                self.assertEqual(decision.committed_query_count, accepted + 1)
                self.assertEqual(decision.seed_hidden_row_index, len(emitted) - 1)

    def test_padding_is_removed_before_target_query_construction(self):
        segment = MlxVerifySegment("r", (11, -1, -1), 1, (11, 12), 2)
        self.assertEqual(segment.valid_draft_tokens, (11,))
        self.assertEqual(build_linear_verify_queries(10, segment), (10, 11))

    def test_mixed_plain_and_drafted_order_is_explicit(self):
        segments = (
            MlxVerifySegment("plain", (-1,), 0, (3,), 1),
            MlxVerifySegment("drafted", (4,), 1, (4, 5), 0),
        )
        decisions = verify_greedy_segments(
            segments, expected_request_ids=("plain", "drafted")
        )
        self.assertEqual([item.accept_len for item in decisions], [1, 2])
        with self.assertRaisesRegex(ValueError, "ordering"):
            verify_greedy_segments(segments, expected_request_ids=("drafted", "plain"))

    def test_invalid_count_and_sentinel_metadata_is_rejected(self):
        invalid = (
            ((1,), -1, (2,)),
            ((1,), 2, (2, 3, 4)),
            ((-1,), 1, (2, 3)),
            ((1, 2), 1, (2, 3)),
        )
        for draft, valid, targets in invalid:
            with self.subTest(draft=draft, valid=valid):
                with self.assertRaises(ValueError):
                    MlxVerifySegment("r", draft, valid, targets)

        with self.assertRaisesRegex(ValueError, "invalid_draft_count"):
            MlxVerifySegment("r", (1, -1), 1, (1, 2), 0)
        with self.assertRaisesRegex(ValueError, "verification_query_count"):
            MlxVerifySegment("r", (1,), 1, (1, 2), 0, 1)
        with self.assertRaisesRegex(ValueError, "verification rows"):
            MlxVerifySegment("r", (1,), 1, (1,), 0)


if __name__ == "__main__":
    unittest.main()
