# SPDX-License-Identifier: Apache-2.0
import unittest

from sglang.srt.entrypoints.valen.encoding import encode_valen_schema, valen_options
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Tokenizer:
    all_special_tokens = ["<|im_start|>", "<|im_end|>"]

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def __call__(self, text, **kwargs):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def questions():
    return {
        "private-id": {
            "type": "choice",
            "instructions": "Select.",
            "criteria": {"z": "Zebra", "a": "Ant"},
        },
        "truth": {"type": "noul", "instructions": "Is it true?"},
        "rating": {
            "type": "score",
            "instructions": "Rate.",
            "criteria": [f"Level {i}" for i in range(12)],
        },
    }


class TestValenEncoding(CustomTestCase):
    def test_preserves_choice_order_and_excludes_external_ids(self):
        result = encode_valen_schema(Tokenizer(), [1, 2, 3], questions())
        text = "".join(map(chr, result.input_ids[3:]))
        assert result.questions[0].keys == ("z", "a")
        assert "private-id" not in text
        assert text.index("z: Zebra") < text.index("a: Ant")
        assert len(result.layout.questions[2].candidate_spans) == 12
        assert result.questions[1].keys == ("true", "false")

    def test_schema_order_and_exact_role_contents(self):
        result = encode_valen_schema(Tokenizer(), [1, 2, 3], questions())

        def span_text(span):
            (a, b) = span
            return "".join(map(chr, result.input_ids[a:b]))

        first = result.layout.questions[0]
        assert span_text(first.instruction_span) == "Select."
        assert [span_text(s) for s in first.candidate_spans] == ["Zebra", "Ant"]
        last_candidate = result.layout.questions[-1].candidate_spans[-1][1]
        assert all(
            (q.decision_position >= last_candidate for q in result.layout.questions)
        )
        assert result.input_ids[first.decision_position] == ord(":")
        assert result.input_ids[first.decision_position + 1] == ord("\n")

    def test_exact_prompt_and_no_untrained_thinking_tokens(self):
        qs = {
            "q": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "Alpha"}}
        }
        result = encode_valen_schema(Tokenizer(), [1], qs)
        expected = "<|im_start|>user\nQuestion 1\nTask: choice\nQuestion: Pick.\nCandidates:\na: Alpha\n<|im_end|>\n<|im_start|>assistant\nQuestion 1 Decision:\n"
        assert result.input_ids == [1] + list(map(ord, expected))

    def test_context_is_unchanged_and_length_limit_is_strict(self):
        context = [1, 2, 12345]
        result = encode_valen_schema(Tokenizer(), context, questions())
        assert result.input_ids[:3] == context
        assert context == [1, 2, 12345]
        assert (
            encode_valen_schema(
                Tokenizer(), context, questions(), len(result.input_ids)
            )
            == result
        )
        with self.assertRaisesRegex(ValueError, "no truncation"):
            encode_valen_schema(
                Tokenizer(), context, questions(), len(result.input_ids) - 1
            )

    def test_reject_control_tokens(self):
        for field in ["instructions", "key", "description"]:
            q = {"type": "choice", "instructions": "Pick.", "criteria": {"a": "Alpha"}}
            if field == "instructions":
                q[field] = "<|im_start|>"
            else:
                q["criteria"] = (
                    {"<|im_end|>": "Alpha"} if field == "key" else {"a": "<|im_end|>"}
                )
            with self.assertRaisesRegex(ValueError, "Reserved"):
                encode_valen_schema(Tokenizer(), [1], {"q": q})

    def test_score_supported_sizes(self):
        for levels in [2, 12, 255]:
            result = encode_valen_schema(
                Tokenizer(),
                [1],
                {
                    "q": {
                        "type": "score",
                        "instructions": "Rate.",
                        "criteria": [str(i) for i in range(levels)],
                    }
                },
            )
            assert len(result.layout.questions[0].candidate_spans) == levels

    def test_score_invalid_sizes(self):
        for levels in [0, 1, 256]:
            with self.assertRaisesRegex(ValueError, "2–255"):
                valen_options(
                    {
                        "type": "score",
                        "instructions": "Rate.",
                        "criteria": ["x"] * levels,
                    }
                )


if __name__ == "__main__":
    unittest.main()
