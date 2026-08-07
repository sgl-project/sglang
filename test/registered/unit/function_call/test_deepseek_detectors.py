import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepSeekStreamingContent(CustomTestCase):
    detector_cases = (
        (DeepSeekV32Detector, "function_calls"),
        (DeepSeekV4Detector, "tool_calls"),
    )
    content = "I'll navigate there now."

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="navigate",
                    description="Navigate to a destination.",
                    parameters={
                        "type": "object",
                        "properties": {"destination": {"type": "string"}},
                        "required": ["destination"],
                    },
                ),
            )
        ]

    @staticmethod
    def _wire_text(section_name, destinations=("library",)):
        invokes = "".join(
            f'<｜DSML｜invoke name="navigate">{{"destination":"{destination}"}}'
            "</｜DSML｜invoke>"
            for destination in destinations
        )
        return (
            f"{TestDeepSeekStreamingContent.content}<｜DSML｜{section_name}>"
            f"{invokes}</｜DSML｜{section_name}>"
        )

    def _parse_chunks(self, detector, chunks):
        normal_text = []
        calls = {}
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_text.append(result.normal_text)
            for call in result.calls:
                parsed = calls.setdefault(
                    call.tool_index, {"name": None, "parameters": ""}
                )
                if call.name is not None:
                    parsed["name"] = call.name
                parsed["parameters"] += call.parameters
        return "".join(normal_text), calls

    def _assert_parsed(self, detector, chunks, destinations=("library",)):
        normal_text, calls = self._parse_chunks(detector, chunks)
        self.assertEqual(normal_text, self.content)
        self.assertEqual(len(calls), len(destinations))
        for index, destination in enumerate(destinations):
            self.assertEqual(calls[index]["name"], "navigate")
            self.assertEqual(
                json.loads(calls[index]["parameters"]),
                {"destination": destination},
            )

    def test_content_and_complete_tool_call_in_single_increment(self):
        """A delta containing content and a complete DSML call preserves both."""
        for detector_class, section_name in self.detector_cases:
            with self.subTest(detector=detector_class.__name__):
                wire_text = self._wire_text(section_name)
                self._assert_parsed(detector_class(), [wire_text])

    def test_content_is_invariant_across_opener_split_boundaries(self):
        """Every split around the DSML opener preserves content exactly once."""
        for detector_class, section_name in self.detector_cases:
            opener = f"<｜DSML｜{section_name}>"
            wire_text = self._wire_text(section_name)
            opener_start = wire_text.index(opener)
            split_start = opener_start - 1
            split_end = opener_start + len(opener) + 1
            for split_at in range(split_start, split_end + 1):
                with self.subTest(detector=detector_class.__name__, split_at=split_at):
                    self._assert_parsed(
                        detector_class(),
                        [wire_text[:split_at], wire_text[split_at:]],
                    )

    def test_content_prefix_with_consecutive_tool_calls(self):
        """Preserving the prefix does not disrupt consecutive DSML calls."""
        destinations = ("library", "museum")
        for detector_class, section_name in self.detector_cases:
            with self.subTest(detector=detector_class.__name__):
                wire_text = self._wire_text(section_name, destinations)
                self._assert_parsed(
                    detector_class(), [wire_text], destinations=destinations
                )


if __name__ == "__main__":
    import unittest

    unittest.main()
