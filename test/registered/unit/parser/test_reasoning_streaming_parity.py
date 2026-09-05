"""A reasoning detector must split reasoning from content the same way however
the output is chunked, since chunk boundaries come from detokenization.

Boundaries fall on token boundaries, and these markers are single vocabulary
tokens, so the sample is chunked with markers kept whole -- splitting inside one
is stricter than anything a real stream produces.
"""

import re
from typing import List, Tuple

from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

REASONING = "checking the answer"
CONTENT = "Done."

Parsed = Tuple[str, str]


def _detector(kind: str):
    return ReasoningParser(model_type=kind).detector


def _sample(kind: str) -> Tuple[str, List[str]]:
    """Build a valid generation for ``kind`` from the detector's own markers."""
    d = _detector(kind)
    start = d.think_start_token + (d.think_start_self_label or "")
    end = d.think_end_token
    # Cohere wraps its answer in START_TEXT/END_TEXT rather than trailing text.
    text_start = getattr(type(d), "TEXT_START_TOKEN", None)
    text_end = getattr(type(d), "TEXT_END_TOKEN", None)
    tail = f"{text_start}{CONTENT}{text_end}" if text_start else CONTENT
    text = f"{start}{REASONING}{end}{tail}"
    return text, [m for m in (start, end, text_start, text_end) if m]


def _oneshot(kind: str, text: str) -> Parsed:
    r = _detector(kind).detect_and_parse(text)
    return (r.reasoning_text or "", r.normal_text or "")


def _streamed(kind: str, chunks: List[str]) -> Parsed:
    d = _detector(kind)
    reasoning = normal = ""
    for chunk in list(chunks) + ["", ""]:
        r = d.parse_streaming_increment(chunk)
        reasoning += r.reasoning_text or ""
        normal += r.normal_text or ""
    # The serving layer flushes the detector when generation ends; without this a
    # held-back marker suffix is never emitted.
    r = d.finish()
    reasoning += r.reasoning_text or ""
    normal += r.normal_text or ""
    return (reasoning, normal)


def _by_token(text: str, markers: List[str]) -> List[str]:
    """Markers whole, everything else in small pieces -- as detokenization emits."""
    pattern = "(" + "|".join(re.escape(m) for m in markers) + ")"
    out: List[str] = []
    for part in [p for p in re.split(pattern, text) if p]:
        if part in markers:
            out.append(part)
        else:
            out.extend(part[i : i + 4] for i in range(0, len(part), 4))
    return out


class TestReasoningStreamingParity(CustomTestCase):
    def test_token_chunking_matches_one_shot(self):
        for kind in sorted(ReasoningParser.DetectorMap):
            with self.subTest(parser=kind):
                text, markers = _sample(kind)
                self.assertEqual(
                    _streamed(kind, _by_token(text, markers)), _oneshot(kind, text)
                )

    def test_single_chunk_matches_one_shot(self):
        for kind in sorted(ReasoningParser.DetectorMap):
            with self.subTest(parser=kind):
                text, _ = _sample(kind)
                self.assertEqual(_streamed(kind, [text]), _oneshot(kind, text))


class TestCohereEchoedThinkStart(CustomTestCase):
    """Cohere's chat template emits ``<|START_THINKING|>`` in the prefix, but some
    checkpoints echo it back. It must not reach the client either way.
    """

    TEXT = (
        "<|START_THINKING|>checking the answer<|END_THINKING|>"
        "<|START_TEXT|>Done.<|END_TEXT|>"
    )

    MARKERS = [
        "<|START_THINKING|>",
        "<|END_THINKING|>",
        "<|START_TEXT|>",
        "<|END_TEXT|>",
    ]

    def test_echoed_marker_is_stripped_when_streaming(self):
        self.assertEqual(
            _streamed("cohere_command4", _by_token(self.TEXT, self.MARKERS)),
            ("checking the answer", "Done."),
        )

    def test_marker_split_across_chunks_is_not_leaked(self):
        head, tail = "<|START", "_THINKING|>checking the answer<|END_THINKING|>"
        self.assertEqual(
            _streamed("cohere_command4", [head, tail]), ("checking the answer", "")
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
