"""Unit tests for srt/parser/reasoning_parser.py — Apertus2509 / CohereCommand4
detectors and ReasoningParser construction/dispatch.

Complements test_reasoning_parser.py, targeting previously uncovered paths:
- Apertus2509Detector.detect_and_parse / detect_and_parse_block_sequence
- Apertus2509Detector.parse_streaming_increment (incl. tool blocks, partial
  tokens across chunk boundaries, stream_reasoning=False)
- CohereCommand4Detector.detect_and_parse / parse_streaming_increment
  (incl. action passthrough, reasoning=False path, truncated streams)
- ReasoningParser __init__ branches (force_reasoning overrides, minimax-m3
  thinking_mode, continue_final_message, force_nonempty_content, tokenizer
  passthrough) and parse_non_stream_blocks / parse_stream_chunk /
  parse_stream_end
"""

import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.parser.reasoning_parser import (
    Apertus2509Detector,
    CohereCommand4Detector,
    ReasoningParser,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

APERTUS_START = "<|inner_prefix|>"
APERTUS_END = "<|inner_suffix|>"
APERTUS_TOOL_START = "<|tools_prefix|>["
APERTUS_TOOL_END = "<|tools_suffix|>"

COHERE_THINK_START = "<|START_THINKING|>"
COHERE_THINK_END = "<|END_THINKING|>"
COHERE_TEXT_START = "<|START_TEXT|>"
COHERE_TEXT_END = "<|END_TEXT|>"
COHERE_ACTION_START = "<|START_ACTION|>"
COHERE_ACTION_END = "<|END_ACTION|>"


def stream_all(detector, chunks):
    """Feed chunks through parse_streaming_increment and return concatenated
    (reasoning, normal) outputs plus the finish() flush."""
    reasoning_parts = []
    normal_parts = []
    for chunk in chunks:
        ret = detector.parse_streaming_increment(chunk)
        reasoning_parts.append(ret.reasoning_text)
        normal_parts.append(ret.normal_text)
    fin = detector.finish()
    reasoning_parts.append(fin.reasoning_text)
    normal_parts.append(fin.normal_text)
    return "".join(reasoning_parts), "".join(normal_parts)


def chunked(text, size=1):
    return [text[i : i + size] for i in range(0, len(text), size)]


class TestApertus2509DetectAndParse(CustomTestCase):
    def test_plain_text_without_markers(self):
        detector = Apertus2509Detector()
        ret = detector.detect_and_parse("hello world")
        self.assertEqual(ret.reasoning_text, "")
        self.assertEqual(ret.normal_text, "hello world")

    def test_single_reasoning_block(self):
        detector = Apertus2509Detector()
        text = APERTUS_START + "thoughts" + APERTUS_END + "answer"
        ret = detector.detect_and_parse(text)
        self.assertEqual(ret.reasoning_text, "thoughts")
        self.assertEqual(ret.normal_text, "answer")

    def test_text_before_block(self):
        detector = Apertus2509Detector()
        text = "hello " + APERTUS_START + "think" + APERTUS_END + "world"
        ret = detector.detect_and_parse(text)
        self.assertEqual(ret.reasoning_text, "think")
        self.assertEqual(ret.normal_text, "hello world")

    def test_multiple_reasoning_blocks(self):
        detector = Apertus2509Detector()
        text = (
            APERTUS_START
            + "a"
            + APERTUS_END
            + "x"
            + APERTUS_START
            + "b"
            + APERTUS_END
            + "y"
        )
        ret = detector.detect_and_parse(text)
        self.assertEqual(ret.reasoning_text, "ab")
        self.assertEqual(ret.normal_text, "xy")

    def test_tool_call_inside_reasoning_kept_in_normal_text(self):
        detector = Apertus2509Detector()
        tool_block = APERTUS_TOOL_START + '{"name": "get_weather"}' + APERTUS_TOOL_END
        text = APERTUS_START + "plan" + tool_block + "done" + APERTUS_END + "final"
        ret = detector.detect_and_parse(text)
        self.assertEqual(ret.reasoning_text, "plandone")
        self.assertEqual(ret.normal_text, tool_block + "final")

    def test_truncated_reasoning_block(self):
        detector = Apertus2509Detector()
        ret = detector.detect_and_parse(APERTUS_START + "still thinking")
        self.assertEqual(ret.reasoning_text, "still thinking")
        self.assertEqual(ret.normal_text, "")

    def test_unclosed_tool_block_inside_reasoning(self):
        detector = Apertus2509Detector()
        tail = APERTUS_TOOL_START + '{"name": "x"}'
        ret = detector.detect_and_parse(APERTUS_START + "plan" + tail)
        self.assertEqual(ret.reasoning_text, "plan")
        self.assertEqual(ret.normal_text, tail)

    def test_block_sequence_order_and_trailing_empty_text(self):
        detector = Apertus2509Detector()
        tool_block = APERTUS_TOOL_START + "[]" + APERTUS_TOOL_END
        text = APERTUS_START + "r1" + tool_block + APERTUS_END
        blocks = detector.detect_and_parse_block_sequence(text)
        self.assertEqual(
            blocks,
            [("reasoning", "r1"), ("text", tool_block), ("text", "")],
        )

    def test_block_sequence_filters_interior_empty_text_blocks(self):
        detector = Apertus2509Detector()
        text = APERTUS_START + "a" + APERTUS_END + APERTUS_START + "b" + APERTUS_END
        blocks = detector.detect_and_parse_block_sequence(text)
        self.assertEqual(blocks, [("reasoning", "a"), ("reasoning", "b"), ("text", "")])

    def test_continue_final_message_resumes_inside_reasoning(self):
        detector = Apertus2509Detector(
            continue_final_message=True,
            previous_content=APERTUS_START + "earlier",
        )
        self.assertTrue(detector._in_reasoning)
        ret = detector.detect_and_parse("more" + APERTUS_END + "answer")
        self.assertEqual(ret.reasoning_text, "more")
        self.assertEqual(ret.normal_text, "answer")

    def test_continue_final_message_stays_normal_after_closed_block(self):
        detector = Apertus2509Detector(
            continue_final_message=True,
            previous_content=APERTUS_START + "r" + APERTUS_END + "done",
        )
        self.assertFalse(detector._in_reasoning)
        ret = detector.detect_and_parse("more text")
        self.assertEqual(ret.reasoning_text, "")
        self.assertEqual(ret.normal_text, "more text")


class TestApertus2509Streaming(CustomTestCase):
    def test_streaming_char_by_char(self):
        text = APERTUS_START + "think" + APERTUS_END + "answer"
        detector = Apertus2509Detector()
        reasoning, normal = stream_all(detector, chunked(text))
        self.assertEqual(reasoning, "think")
        self.assertEqual(normal, "answer")

    def test_streaming_with_tool_block_char_by_char(self):
        tool_block = APERTUS_TOOL_START + '{"a": 1}' + APERTUS_TOOL_END
        text = APERTUS_START + "think" + tool_block + "post" + APERTUS_END + "final"
        detector = Apertus2509Detector()
        reasoning, normal = stream_all(detector, chunked(text))
        self.assertEqual(reasoning, "thinkpost")
        self.assertEqual(normal, tool_block + "final")

    def test_streaming_plain_text_passthrough(self):
        detector = Apertus2509Detector()
        reasoning, normal = stream_all(detector, chunked("just text", size=3))
        self.assertEqual(reasoning, "")
        self.assertEqual(normal, "just text")

    def test_partial_marker_held_across_chunks(self):
        detector = Apertus2509Detector()
        # Split inside the start marker, then inside the end marker.
        head = APERTUS_START[:5]
        rest_start = APERTUS_START[5:] + "abc" + APERTUS_END[:7]
        tail = APERTUS_END[7:] + "ans"
        outs = [detector.parse_streaming_increment(c) for c in (head, rest_start, tail)]
        fin = detector.finish()
        reasoning = "".join(o.reasoning_text for o in outs) + fin.reasoning_text
        normal = "".join(o.normal_text for o in outs) + fin.normal_text
        self.assertEqual(reasoning, "abc")
        self.assertEqual(normal, "ans")

    def test_partial_start_marker_emitted_when_not_completed(self):
        detector = Apertus2509Detector()
        # "<|inner" is a prefix of the start token and must be held back...
        r1 = detector.parse_streaming_increment("x<|inner")
        self.assertEqual(r1.normal_text, "x")
        # ...but once the next chunk proves it is not the marker, it flushes.
        r2 = detector.parse_streaming_increment("X")
        r3 = detector.parse_streaming_increment("y")
        fin = detector.finish()
        normal = r2.normal_text + r3.normal_text + fin.normal_text
        self.assertEqual(normal, "<|innerXy")

    def test_stream_reasoning_false_emits_on_close(self):
        text = APERTUS_START + "think" + APERTUS_END + "answer"
        detector = Apertus2509Detector(stream_reasoning=False)
        seen = []
        for chunk in chunked(text, size=4):
            ret = detector.parse_streaming_increment(chunk)
            if ret.reasoning_text:
                seen.append(ret.reasoning_text)
        # Reasoning must appear in a single emission, only after the end token.
        self.assertEqual(seen, ["think"])
        fin = detector.finish()
        self.assertEqual(fin.reasoning_text, "")

    def test_stream_reasoning_false_truncated_flushed_by_finish(self):
        detector = Apertus2509Detector(stream_reasoning=False)
        for chunk in chunked(APERTUS_START + "partial thinking", size=5):
            ret = detector.parse_streaming_increment(chunk)
            self.assertEqual(ret.reasoning_text, "")
        fin = detector.finish()
        self.assertEqual(fin.reasoning_text, "partial thinking")

    def test_truncated_streaming_reasoning_flushed_by_finish(self):
        detector = Apertus2509Detector()
        reasoning, normal = stream_all(
            detector, chunked(APERTUS_START + "cut short", size=3)
        )
        self.assertEqual(reasoning, "cut short")
        self.assertEqual(normal, "")


class TestCohereCommand4DetectAndParse(CustomTestCase):
    def test_standard_thinking_then_text(self):
        detector = CohereCommand4Detector()
        text = (
            "thinking"
            + COHERE_THINK_END
            + COHERE_TEXT_START
            + "answer"
            + COHERE_TEXT_END
        )
        ret = detector.detect_and_parse(text)
        self.assertEqual(ret.reasoning_text, "thinking")
        self.assertEqual(ret.normal_text, "answer")

    def test_echoed_start_thinking_is_stripped(self):
        detector = CohereCommand4Detector()
        text = (
            COHERE_THINK_START
            + "thinking"
            + COHERE_THINK_END
            + COHERE_TEXT_START
            + "answer"
            + COHERE_TEXT_END
        )
        ret = detector.detect_and_parse(text)
        self.assertEqual(ret.reasoning_text, "thinking")
        self.assertEqual(ret.normal_text, "answer")

    def test_reasoning_false_text_only(self):
        detector = CohereCommand4Detector()
        text = COHERE_TEXT_START + "answer" + COHERE_TEXT_END
        ret = detector.detect_and_parse(text)
        self.assertEqual(ret.reasoning_text, "")
        self.assertEqual(ret.normal_text, "answer")

    def test_action_block_passed_through_intact(self):
        detector = CohereCommand4Detector()
        action = COHERE_ACTION_START + '{"tool": "search"}' + COHERE_ACTION_END
        text = "thinking" + COHERE_THINK_END + action
        ret = detector.detect_and_parse(text)
        self.assertEqual(ret.reasoning_text, "thinking")
        self.assertEqual(ret.normal_text, action)

    def test_reasoning_false_action_block(self):
        detector = CohereCommand4Detector()
        action = COHERE_ACTION_START + '{"tool": "search"}' + COHERE_ACTION_END
        ret = detector.detect_and_parse(action)
        self.assertEqual(ret.reasoning_text, "")
        self.assertEqual(ret.normal_text, action)

    def test_truncated_inside_thinking(self):
        detector = CohereCommand4Detector()
        ret = detector.detect_and_parse("thinking without any marker")
        self.assertEqual(ret.reasoning_text, "thinking without any marker")
        self.assertEqual(ret.normal_text, "")

    def test_missing_text_end_marker(self):
        detector = CohereCommand4Detector()
        text = "th" + COHERE_THINK_END + COHERE_TEXT_START + "cut off"
        ret = detector.detect_and_parse(text)
        self.assertEqual(ret.reasoning_text, "th")
        self.assertEqual(ret.normal_text, "cut off")

    def test_force_nonempty_content_swaps_reasoning_only_output(self):
        detector = CohereCommand4Detector(force_nonempty_content=True)
        text = "only reasoning" + COHERE_THINK_END
        ret = detector.detect_and_parse(text)
        self.assertEqual(ret.normal_text, "only reasoning")
        self.assertEqual(ret.reasoning_text, "")


class TestCohereCommand4Streaming(CustomTestCase):
    def test_streaming_char_by_char(self):
        text = (
            "thinking"
            + COHERE_THINK_END
            + COHERE_TEXT_START
            + "answer"
            + COHERE_TEXT_END
        )
        detector = CohereCommand4Detector()
        reasoning, normal = stream_all(detector, chunked(text))
        self.assertEqual(reasoning, "thinking")
        self.assertEqual(normal, "answer")

    def test_streaming_markers_split_across_chunks(self):
        text = (
            "thinking"
            + COHERE_THINK_END
            + COHERE_TEXT_START
            + "answer"
            + COHERE_TEXT_END
        )
        detector = CohereCommand4Detector()
        # Chunk size 3 forces every marker to be split across boundaries.
        reasoning, normal = stream_all(detector, chunked(text, size=3))
        self.assertEqual(reasoning, "thinking")
        self.assertEqual(normal, "answer")

    def test_stream_reasoning_false_emits_once_at_end_thinking(self):
        text = (
            "thinking"
            + COHERE_THINK_END
            + COHERE_TEXT_START
            + "answer"
            + COHERE_TEXT_END
        )
        detector = CohereCommand4Detector(stream_reasoning=False)
        seen = []
        for chunk in chunked(text, size=4):
            ret = detector.parse_streaming_increment(chunk)
            if ret.reasoning_text:
                seen.append(ret.reasoning_text)
        self.assertEqual(seen, ["thinking"])

    def test_action_mode_passthrough(self):
        text = "th" + COHERE_THINK_END + COHERE_ACTION_START + "payload"
        detector = CohereCommand4Detector()
        reasoning, normal = stream_all(detector, chunked(text, size=3))
        self.assertEqual(reasoning, "th")
        self.assertEqual(normal, COHERE_ACTION_START + "payload")

    def test_reasoning_false_streaming_text(self):
        text = COHERE_TEXT_START + "hi there" + COHERE_TEXT_END
        detector = CohereCommand4Detector()
        reasoning, normal = stream_all(detector, chunked(text, size=2))
        self.assertEqual(reasoning, "")
        self.assertEqual(normal, "hi there")

    def test_truncated_text_flushed_by_finish(self):
        detector = CohereCommand4Detector()
        for chunk in chunked(
            "th" + COHERE_THINK_END + COHERE_TEXT_START + "partial", size=4
        ):
            detector.parse_streaming_increment(chunk)
        fin = detector.finish()
        self.assertEqual(fin.normal_text, "partial")
        self.assertEqual(fin.reasoning_text, "")

    def test_truncated_reasoning_flushed_by_finish(self):
        detector = CohereCommand4Detector(stream_reasoning=True)
        reasoning, normal = stream_all(detector, chunked("thinking only", size=3))
        self.assertEqual(reasoning, "thinking only")
        self.assertEqual(normal, "")

    def test_truncated_reasoning_stream_false_flushed_by_finish(self):
        detector = CohereCommand4Detector(stream_reasoning=False)
        for chunk in chunked("thinking only", size=3):
            ret = detector.parse_streaming_increment(chunk)
            self.assertEqual(ret.reasoning_text, "")
        fin = detector.finish()
        self.assertEqual(fin.reasoning_text, "thinking only")

    def test_finish_after_complete_stream_is_empty(self):
        text = "th" + COHERE_THINK_END + COHERE_TEXT_START + "ans" + COHERE_TEXT_END
        detector = CohereCommand4Detector()
        stream_all(detector, chunked(text, size=3))
        # stream_all already consumed finish(); a second call must be empty.
        fin = detector.finish()
        self.assertEqual(fin.reasoning_text, "")
        self.assertEqual(fin.normal_text, "")


class TestReasoningParserConstruction(CustomTestCase):
    def test_force_reasoning_override_model_types(self):
        for model_type in ("qwen3-thinking", "gpt-oss", "minimax"):
            parser = ReasoningParser(model_type)
            self.assertTrue(
                parser.detector.force_reasoning, msg=f"model_type={model_type}"
            )

    def test_minimax_m3_thinking_mode_enabled(self):
        request = ChatCompletionRequest(
            model="m", messages=[], chat_template_kwargs={"thinking_mode": "enabled"}
        )
        parser = ReasoningParser("minimax-m3", request=request)
        self.assertTrue(parser.detector.force_reasoning)

    def test_minimax_m3_thinking_mode_disabled_or_absent(self):
        for kwargs in ({"thinking_mode": "disabled"}, {}):
            request = ChatCompletionRequest(
                model="m", messages=[], chat_template_kwargs=kwargs or None
            )
            parser = ReasoningParser("minimax-m3", request=request)
            self.assertFalse(parser.detector.force_reasoning)

    def test_continue_final_message_propagates_previous_content(self):
        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "assistant", "content": "<think>prev"}],
            continue_final_message=True,
        )
        parser = ReasoningParser("qwen3", request=request)
        self.assertTrue(parser.detector.continue_final_message)
        self.assertEqual(parser.detector.previous_content, "<think>prev")
        self.assertTrue(parser.detector._in_reasoning)

    def test_continue_final_message_requires_assistant_last_message(self):
        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            continue_final_message=True,
        )
        parser = ReasoningParser("qwen3", request=request)
        self.assertFalse(parser.detector.continue_final_message)
        self.assertEqual(parser.detector.previous_content, "")

    def test_force_nonempty_content_from_chat_template_kwargs(self):
        request = ChatCompletionRequest(
            model="m",
            messages=[],
            chat_template_kwargs={"force_nonempty_content": True},
        )
        parser = ReasoningParser("qwen3", request=request)
        self.assertTrue(parser.detector._force_nonempty_content)

    def test_tokenizer_passed_to_supporting_detector(self):
        tokenizer = SimpleNamespace(
            get_vocab=lambda: {"<think:hy3>": 1, "<tool_calls:hy3>": 2}
        )
        parser = ReasoningParser("hunyuan", tokenizer=tokenizer)
        self.assertEqual(parser.detector.think_start_token, "<think:hy3>")
        self.assertEqual(parser.detector.think_end_token, "</think:hy3>")
        self.assertEqual(parser.detector.tool_start_token, "<tool_calls:hy3>")

    def test_tokenizer_ignored_by_unsupporting_detector(self):
        tokenizer = SimpleNamespace(get_vocab=lambda: {})
        parser = ReasoningParser("qwen3", tokenizer=tokenizer)
        self.assertEqual(parser.detector.think_start_token, "<think>")


class TestReasoningParserParsingAPIs(CustomTestCase):
    def test_parse_non_stream(self):
        parser = ReasoningParser("qwen3")
        reasoning, normal = parser.parse_non_stream("<think>r</think>ans")
        self.assertEqual(reasoning, "r")
        self.assertEqual(normal, "ans")

    def test_parse_non_stream_blocks_with_block_sequence_detector(self):
        parser = ReasoningParser("apertus2509")
        tool_block = APERTUS_TOOL_START + "[]" + APERTUS_TOOL_END
        text = APERTUS_START + "r1" + tool_block + APERTUS_END + "final"
        blocks = parser.parse_non_stream_blocks(text)
        self.assertEqual(
            blocks,
            [
                {"type": "reasoning", "text": "r1"},
                {"type": "text", "text": tool_block},
                {"type": "text", "text": "final"},
            ],
        )

    def test_parse_non_stream_blocks_fallback_for_plain_detector(self):
        parser = ReasoningParser("qwen3")
        blocks = parser.parse_non_stream_blocks("<think>r</think>ans")
        self.assertEqual(
            blocks,
            [
                {"type": "reasoning", "text": "r"},
                {"type": "text", "text": "ans"},
            ],
        )

    def test_parse_non_stream_blocks_without_reasoning(self):
        parser = ReasoningParser("qwen3")
        blocks = parser.parse_non_stream_blocks("plain answer")
        self.assertEqual(blocks, [{"type": "text", "text": "plain answer"}])

    def test_parse_stream_chunk_and_end(self):
        parser = ReasoningParser("qwen3")
        reasoning_parts = []
        normal_parts = []
        for chunk in ("<think>r1", "r2</think>", "ans"):
            reasoning, normal = parser.parse_stream_chunk(chunk)
            reasoning_parts.append(reasoning)
            normal_parts.append(normal)
        self.assertEqual("".join(reasoning_parts), "r1r2")
        self.assertEqual("".join(normal_parts), "ans")
        reasoning, normal = parser.parse_stream_end()
        self.assertEqual(reasoning, "")
        self.assertEqual(normal, "")

    def test_parse_stream_end_flushes_truncated_reasoning(self):
        parser = ReasoningParser("qwen3", stream_reasoning=False)
        parser.parse_stream_chunk("<think>partial")
        reasoning, normal = parser.parse_stream_end()
        self.assertEqual(reasoning, "partial")
        self.assertEqual(normal, "")


if __name__ == "__main__":
    unittest.main()
