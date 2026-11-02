"""Complete invalid Qwen JSON tool calls must not look like a normal stop."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestQwen25InvalidJSON(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="edit",
                    parameters={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                ),
            )
        ]
        # Reduced from a synthetic Nanbeige4.3 deployment reproduction:
        # the JSON chat template was selected, but the model emitted XML.
        self.invalid_payloads = [
            "<function=edit>\n<parameter=text>synthetic payload</parameter>\n</function>",
            '{"name":"edit","arguments":{"text":invalid}}',
            '{"name":"edit","arguments":{"text":"ok"}} trailing garbage',
        ]

    def parser(self, name="qwen"):
        return FunctionCallParser(self.tools, name)

    def test_complete_invalid_block_non_streaming(self):
        for name in ("qwen", "qwen25"):
            for payload in self.invalid_payloads:
                with self.subTest(parser=name, payload=payload):
                    wire = (
                        "I'll edit the file.\n<tool_call>\n"
                        + payload
                        + "\n</tool_call>"
                    )
                    with self.assertRaisesRegex(ValueError, "Invalid JSON") as cm:
                        self.parser(name).parse_non_stream(wire)
                    self.assertNotIn(payload, str(cm.exception))

    def test_complete_invalid_block_streaming_at_every_split(self):
        for end_tag in ("\n</tool_call>", "</tool_call>"):
            for payload in self.invalid_payloads:
                wire = "I'll edit the file.\n<tool_call>\n" + payload + end_tag
                for split in range(len(wire) + 1):
                    with self.subTest(end_tag=end_tag, payload=payload, split=split):
                        parser = self.parser()
                        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                            parser.parse_stream_chunk(wire[:split])
                            parser.parse_stream_chunk(wire[split:])
                            parser.parse_stream_end()

    def test_streaming_calls_with_mixed_end_tags(self):
        values = (
            "first",
            "literal </tool_call> inside JSON",
            'escaped quote "</tool_call> and backslash \\',
            'escaped newline\n</tool_call> and "quotes"',
            r"backslash and quote \"</tool_call>",
            r"two backslashes \\</tool_call>",
            "many </tool_call> tags " * 8,
        )
        for first_end in ("</tool_call>", "\n</tool_call>"):
            for second_end in ("</tool_call>", "\n</tool_call>"):
                for value in values:
                    for width in (1, 2, 7, 16, 1024):
                        with self.subTest(
                            first_end=first_end,
                            second_end=second_end,
                            value=value,
                            width=width,
                        ):
                            parser = self.parser()
                            calls = []
                            for text, end_tag in (
                                (value, first_end),
                                ("second", second_end),
                            ):
                                for part in (
                                    "<tool_call>\n",
                                    '{"name":"edit",',
                                    '"arguments":' + json.dumps({"text": text}) + "}",
                                    end_tag,
                                ):
                                    for start in range(0, len(part), width):
                                        calls.extend(
                                            parser.parse_stream_chunk(
                                                part[start : start + width]
                                            )[1]
                                        )
                            calls.extend(parser.parse_stream_end()[1])
                            self.assertEqual(
                                [call.name for call in calls if call.name],
                                ["edit", "edit"],
                            )
                            for index, text in enumerate((value, "second")):
                                arguments = "".join(
                                    call.parameters
                                    for call in calls
                                    if call.tool_index == index
                                )
                                self.assertEqual(json.loads(arguments), {"text": text})

    def test_unclosed_json_string_before_standard_end_tag(self):
        wire = (
            '<tool_call>\n{"name":"edit","arguments":{"text":"unfinished\n</tool_call>'
        )
        for split in range(len(wire) + 1):
            with self.subTest(split=split):
                parser = self.parser()
                with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                    parser.parse_stream_chunk(wire[:split])
                    parser.parse_stream_chunk(wire[split:])

    def test_invalid_block_after_valid_call(self):
        valid = '<tool_call>\n{"name":"edit","arguments":{"text":"ok"}}\n</tool_call>'
        invalid = "<tool_call>\n" + self.invalid_payloads[0] + "\n</tool_call>"
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            self.parser().parse_non_stream(valid + invalid)
        parser = self.parser()
        for chunk in (
            "<tool_call>\n",
            '{"name":"edit",',
            '"arguments":{"text":"ok"}}',
            "\n</tool_call>",
        ):
            parser.parse_stream_chunk(chunk)
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            parser.parse_stream_chunk(invalid)

    def test_streaming_validation_survives_consumed_arguments(self):
        wire = "<tool_call>\n" + self.invalid_payloads[-1] + "\n</tool_call>"
        for width in (1, 2, 7, 16, len(wire)):
            with self.subTest(width=width):
                parser = self.parser()
                with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                    for start in range(0, len(wire), width):
                        parser.parse_stream_chunk(wire[start : start + width])
                    parser.parse_stream_end()

    def test_unterminated_block_keeps_existing_truncation_behavior(self):
        for payload in self.invalid_payloads:
            wire = "<tool_call>\n" + payload
            self.parser().parse_non_stream(wire)
            parser = self.parser()
            for character in wire:
                parser.parse_stream_chunk(character)
            parser.parse_stream_end()

    def test_valid_json_preserves_literal_markup_and_unicode(self):
        value = 'before </parameter> after\n"quotes" & unicode é'
        payload = json.dumps({"name": "edit", "arguments": {"text": value}})
        _, calls = self.parser().parse_non_stream(
            "<tool_call>\n" + payload + "\n</tool_call>"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].parameters), {"text": value})
        parser = self.parser()
        calls = []
        for chunk in (
            "<tool_call>\n",
            '{"name":"edit",',
            '"arguments":' + json.dumps({"text": value}) + "}",
            "\n</tool_call>",
        ):
            calls.extend(parser.parse_stream_chunk(chunk)[1])
        calls.extend(parser.parse_stream_end()[1])
        self.assertEqual([call.name for call in calls if call.name], ["edit"])
        self.assertEqual(
            json.loads("".join(call.parameters for call in calls)), {"text": value}
        )


if __name__ == "__main__":
    unittest.main()
