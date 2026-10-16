# SPDX-License-Identifier: Apache-2.0
import unittest

import msgspec

from sglang.srt.layers.valen_layout import (
    ValenLayout,
    ValenQuestionLayout,
    pack_valen_layout,
    parse_valen_layout,
    rebase_valen_context,
    validate_valen_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def layout():
    return ValenLayout(
        24,
        (0, 3),
        (
            ValenQuestionLayout("choice", (4, 6), ((7, 8), (9, 11)), 20),
            ValenQuestionLayout("score", (12, 13), ((14, 15), (16, 17)), 22),
        ),
    )


class TestValenLayout(CustomTestCase):
    def test_roundtrip_and_serialization(self):
        expected = layout()
        packed = pack_valen_layout(expected)
        assert parse_valen_layout(packed, 24) == expected
        assert (
            msgspec.json.decode(msgspec.json.encode(expected), type=ValenLayout)
            == expected
        )

    def test_reject_malformed_wire_layout(self):
        for change in [
            lambda v: v[:-1],
            lambda v: v + [0],
            lambda v: [2, *v[1:]],
            lambda v: [True, *v[1:]],
            lambda v: [v[0], 23, *v[2:]],
            lambda v: [*v[:5], 99, *v[6:]],
            lambda v: [*v[:9], 256, *v[10:]],
        ]:
            with self.assertRaises(ValueError):
                parse_valen_layout(change(pack_valen_layout(layout())), 24)

    def test_no_implicit_image_expansion_or_truncation(self):
        for length in (23, 25, 240):
            with self.assertRaisesRegex(ValueError, "prompt length"):
                parse_valen_layout(pack_valen_layout(layout()), length)

    def test_expansion_keeps_context_start_and_moves_all_schema_roles(self):
        old = layout()
        expanded = rebase_valen_context(old, 15)
        assert expanded.context_span == (0, 15)
        assert expanded.prompt_length == 36
        assert expanded.questions[0].instruction_span == (16, 18)
        assert expanded.questions[0].candidate_spans == ((19, 20), (21, 23))
        assert expanded.questions[0].decision_position == 32
        assert expanded.questions[1].decision_position == 34
        assert old == layout()
        with self.assertRaisesRegex(ValueError, "remove tokens"):
            rebase_valen_context(old, 2)

    def test_readout_must_follow_every_question_schema(self):
        old = layout()
        first = old.questions[0]
        bad = ValenLayout(
            24,
            old.context_span,
            (
                ValenQuestionLayout(
                    first.kind, first.instruction_span, first.candidate_spans, 12
                ),
                old.questions[1],
            ),
        )
        with self.assertRaisesRegex(ValueError, "readouts"):
            validate_valen_layout(bad, 24)

    def test_task_candidate_limits(self):
        for kind, count in [("choice", 0), ("noul", 1), ("noul", 3), ("score", 1)]:
            spans = tuple(((4 + i, 5 + i) for i in range(count)))
            bad = ValenLayout(
                24, (0, 2), (ValenQuestionLayout(kind, (2, 3), spans, 20),)
            )
            with self.assertRaisesRegex(ValueError, "candidate count"):
                validate_valen_layout(bad, 24)


if __name__ == "__main__":
    unittest.main()
