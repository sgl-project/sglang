"""GigaChat 4 reasoning, truncation, and terminal-marker parsing."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGigaChat4ReasoningParser(CustomTestCase):
    def _assert_parses(self, text, expected, *, reasoning):
        for stream_reasoning in (True, False):
            parser_kwargs = dict(
                model_type="gigachat4",
                force_reasoning=reasoning,
                stream_reasoning=stream_reasoning,
            )
            self.assertEqual(
                ReasoningParser(**parser_kwargs).parse_non_stream(text), expected
            )
            # Every two-part split, plus a stream with one character per delta.
            chunkings = [[text[:i], text[i:]] for i in range(len(text) + 1)]
            chunkings.append(list(text))
            for chunks in chunkings:
                with self.subTest(chunks=chunks, stream_reasoning=stream_reasoning):
                    parser = ReasoningParser(**parser_kwargs)
                    results = [parser.parse_stream_chunk(chunk) for chunk in chunks]
                    results.append(parser.parse_stream_end())
                    self.assertEqual(
                        tuple(
                            "".join(result[i] or "" for result in results)
                            for i in (0, 1)
                        ),
                        expected,
                    )
                    self.assertEqual(parser.parse_stream_end(), ("", ""))

    def test_terminal_markers_with_and_without_reasoning(self):
        for reasoning in (True, False):
            prefix = "Thought</think>" if reasoning else ""
            expected = ("Thought" if reasoning else "", "\n\nAnswer.")
            for suffix in (
                "</s>",
                "<|message_sep|>",
                "<|message_sep|></s>",
                "</s></s>",
            ):
                for whitespace in ("", " \n\t"):
                    self._assert_parses(
                        prefix + "\n\nAnswer." + suffix + whitespace,
                        expected,
                        reasoning=reasoning,
                    )

    def test_literal_markers_and_incomplete_suffixes_are_preserved(self):
        for text in (
            "Use </s> as a delimiter.",
            "Use <|message_sep|> between messages.",
            "Answer</s>\nMore text.",
            "Answer <",
            "Answer </s",
            "Answer <|message_sep|",
            "Answer </s><|mess",
            "Answer.\n\n",
        ):
            self._assert_parses(text, ("", text), reasoning=False)

    def test_truncated_reasoning_strips_only_complete_terminal_markers(self):
        self._assert_parses(
            "Thought<|message_sep|></s>\n", ("Thought", ""), reasoning=True
        )
        self._assert_parses("Thought <|mess", ("Thought <|mess", ""), reasoning=True)

    def test_only_first_think_start_is_consumed(self):
        for opening, thought in (
            ("<think>", "Thought"),
            ("<think><think>", "<think>Thought"),
            ("<think><think><think>", "<think><think>Thought"),
        ):
            self._assert_parses(
                opening + "Thought</think>\n\nAnswer.",
                (thought, "\n\nAnswer."),
                reasoning=True,
            )

    def test_think_block_never_closes(self):
        for text, thought in (
            ("Unfinished thought", "Unfinished thought"),
            ("<think>Unfinished thought", "Unfinished thought"),
            ("<think><think>Unfinished thought", "<think>Unfinished thought"),
        ):
            self._assert_parses(
                text,
                (thought, ""),
                reasoning=True,
            )

    def test_generation_ends_inside_or_after_think_start_tag(self):
        for text, thought in (
            ("<thi", "<thi"),
            ("<think>", ""),
            ("<think><thi", "<thi"),
            ("<think><think>", "<think>"),
        ):
            self._assert_parses(text, (thought, ""), reasoning=True)

    def test_ordinary_text_is_emitted_without_waiting_for_finish(self):
        parser = ReasoningParser("gigachat4", force_reasoning=False)
        self.assertEqual(parser.parse_stream_chunk("Answer."), ("", "Answer."))
        self.assertEqual(parser.parse_stream_chunk("<|message_"), ("", ""))
        self.assertEqual(parser.parse_stream_chunk("sep|>"), ("", ""))
        self.assertEqual(parser.parse_stream_end(), ("", ""))

    def test_gcml_arguments_keep_literal_markers(self):
        tools = [Tool(type="function", function=Function(name="lookup", parameters={}))]
        value = "</s><|message_sep|>"
        text = (
            "Thought</think>\n"
            '<｜GCML｜tool_calls><｜GCML｜invoke name="lookup">'
            f'<｜GCML｜parameter name="key" string="true">{value}'
            "</｜GCML｜parameter></｜GCML｜invoke></｜GCML｜tool_calls></s>"
        )
        for chunks in ([text], list(text)):
            reasoning = ReasoningParser("gigachat4", force_reasoning=True)
            functions = FunctionCallParser(tools, "gigachat4")
            content, calls = "", []
            for chunk in chunks:
                _, normal = reasoning.parse_stream_chunk(chunk)
                normal, parsed = functions.parse_stream_chunk(normal)
                content += normal
                calls.extend(parsed)
            _, normal = reasoning.parse_stream_end()
            normal, parsed = functions.parse_stream_chunk(normal)
            content += normal
            calls.extend(parsed)
            normal, parsed = functions.parse_stream_end()
            content += normal
            calls.extend(parsed)
            self.assertEqual(content, "\n")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].name, "lookup")
            self.assertEqual(json.loads(calls[0].parameters), {"key": value})

    def test_gigachat35_is_unchanged(self):
        parser = ReasoningParser("gigachat35")
        self.assertEqual(
            parser.parse_non_stream("Thought</think>Answer</s>"),
            ("Thought", "Answer</s>"),
        )


if __name__ == "__main__":
    unittest.main()
