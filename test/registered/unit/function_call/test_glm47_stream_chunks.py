import json
import random
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGlm47StreamChunks(unittest.TestCase):
    def make_parser(self):
        return FunctionCallParser(
            tools=[
                Tool(
                    type="function",
                    function=Function(
                        name="read_file",
                        parameters={
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    ),
                ),
                Tool(
                    type="function",
                    function=Function(name="status", parameters={"type": "object"}),
                ),
            ],
            tool_call_parser="glm47",
        )

    @staticmethod
    def tool_call(path):
        return (
            "<tool_call>read_file<arg_key>path</arg_key>"
            f"<arg_value>{path}</arg_value></tool_call>"
        )

    def assert_calls(self, deltas, expected):
        def unique_object(pairs):
            self.assertEqual(len(pairs), len(dict(pairs)), "Duplicate argument keys")
            return dict(pairs)

        names = []
        arguments = {}
        for delta in deltas:
            if delta.name is not None:
                names.append((delta.tool_index, delta.name))
            arguments.setdefault(delta.tool_index, "")
            arguments[delta.tool_index] += delta.parameters
        self.assertEqual(names, [(i, name) for i, (name, _) in enumerate(expected)])
        self.assertEqual(
            {
                index: json.loads(value, object_pairs_hook=unique_object)
                for index, value in arguments.items()
            },
            {i: args for i, (_, args) in enumerate(expected)},
        )

    def test_complete_calls_are_drained_before_finish(self):
        text = self.tool_call("first.py") + self.tool_call("目录/café.py")
        parser = self.make_parser()
        normal_text, calls = parser.parse_stream_chunk(text)
        self.assertEqual(normal_text, "")
        self.assert_calls(
            calls,
            [
                ("read_file", {"path": "first.py"}),
                ("read_file", {"path": "目录/café.py"}),
            ],
        )
        self.assertEqual(parser.parse_stream_chunk(""), ("", []))
        self.assertEqual(parser.parse_stream_end(), ("", []))

    def test_partition_invariance_with_normal_text_and_no_args(self):
        text = (
            "Before: "
            + self.tool_call("目录/café.py")
            + " Between: "
            + "<tool_call>status</tool_call>"
            + self.tool_call('quote"and\\slash.py')
            + " After."
        )
        expected = [
            ("read_file", {"path": "目录/café.py"}),
            ("status", {}),
            ("read_file", {"path": 'quote"and\\slash.py'}),
        ]
        partitions = [
            [text[i : i + size] for i in range(0, len(text), size)]
            for size in (1, 7, 17, len(text))
        ]
        # Every two-chunk partition exercises splits within opening/closing tags.
        partitions.extend([text[:i], text[i:]] for i in range(len(text) + 1))
        for chunks in partitions:
            with self.subTest(chunk_lengths=list(map(len, chunks))):
                parser = self.make_parser()
                normal_text = ""
                calls = []
                for chunk in chunks + [""]:
                    normal, deltas = parser.parse_stream_chunk(chunk)
                    normal_text += normal
                    calls.extend(deltas)
                normal, deltas = parser.parse_stream_end()
                normal_text += normal
                calls.extend(deltas)
                self.assertEqual(normal_text, "Before:  Between:  After.")
                self.assert_calls(calls, expected)
                non_stream_text, non_stream_calls = self.make_parser().parse_non_stream(
                    text
                )
                self.assertEqual(normal_text, non_stream_text)
                self.assertEqual(
                    [json.loads(call.parameters) for call in non_stream_calls],
                    [args for _, args in expected],
                )

    def test_single_call_arguments_are_not_repeated(self):
        expected = {
            "path_prefix": "Doc/library/glob.rst",
            "max_matches": 5,
            "query": "filenames starting with",
        }
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path_prefix": {"type": "string"},
                            "max_matches": {"type": "integer"},
                            "query": {"type": "string"},
                        },
                    },
                ),
            )
        ]
        text = (
            "<tool_call>search"
            + "".join(
                f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
                for key, value in expected.items()
            )
            + "</tool_call>"
        )
        partitions = [[text[:i], text[i:]] for i in range(len(text) + 1)]
        rng = random.Random(47)
        for _ in range(50):
            cuts = [0] + sorted(rng.sample(range(1, len(text)), 12)) + [len(text)]
            partitions.append([text[a:b] for a, b in zip(cuts, cuts[1:])])
        for chunks in partitions:
            with self.subTest(chunk_lengths=list(map(len, chunks))):
                parser = FunctionCallParser(tools=tools, tool_call_parser="glm47")
                calls = []
                for chunk in chunks + [""]:
                    normal, deltas = parser.parse_stream_chunk(chunk)
                    self.assertEqual(normal, "")
                    calls.extend(deltas)
                self.assertEqual(parser.parse_stream_end(), ("", []))
                self.assert_calls(calls, [("search", expected)])

    def test_incomplete_final_call_preserves_state(self):
        first = self.tool_call("first.py")
        second = self.tool_call("目录/second.py")
        for split in (5, 20, second.index("second") + 3, len(second) - 1):
            with self.subTest(split=split):
                parser = self.make_parser()
                normal, calls = parser.parse_stream_chunk(first + second[:split])
                self.assertEqual(normal, "")
                self.assert_calls(
                    [call for call in calls if call.tool_index == 0],
                    [("read_file", {"path": "first.py"})],
                )
                self.assertEqual(parser.parse_stream_chunk(""), ("", []))
                # A truncated stream must not invent completion or repeat deltas.
                truncated = self.make_parser()
                truncated.parse_stream_chunk(first + second[:split])
                self.assertEqual(truncated.parse_stream_end(), ("", []))
                normal, remainder = parser.parse_stream_chunk(second[split:])
                self.assertEqual(normal, "")
                self.assert_calls(
                    calls + remainder,
                    [
                        ("read_file", {"path": "first.py"}),
                        ("read_file", {"path": "目录/second.py"}),
                    ],
                )
                self.assertEqual(parser.parse_stream_end(), ("", []))


if __name__ == "__main__":
    unittest.main()
