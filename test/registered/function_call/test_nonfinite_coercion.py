"""Tool-call argument coercion must reject non-finite numbers.

A model can emit a numeric argument like "1e999"/"inf"/"nan". These must not
crash coercion (int(float("inf")) raises OverflowError) nor leak into the
tool-call output as invalid JSON (Infinity/NaN); they degrade to the raw
string. Ports vllm-project/vllm#43984 across SGLang's typed-coercion detectors.
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import _convert_param_value as _mimo_convert
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.srt.function_call.poolside_v1_detector import PoolsideV1Detector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(3, "base-a-test-cpu")

NON_FINITE_NUMBERS = ["inf", "-inf", "Infinity", "1e999", "-1e999", "nan", "NaN"]
NON_FINITE_CONTAINERS = [
    "[1e999]",
    "[1, 2, 1e999]",
    "[Infinity]",
    "[NaN]",
    '{"x": 1e999}',
    # Non-JSON Python literals that json.loads rejects but ast.literal_eval
    # accepts (exercises the ast fallback path).
    "(1e999,)",
    "{'x': 1e999}",
]


def _mimo(raw, ptype):
    tool = Tool(
        type="function",
        function=Function(
            name="f",
            parameters={"type": "object", "properties": {"p": {"type": ptype}}},
        ),
    )
    return _mimo_convert(raw, "p", "f", [tool])


def _minimax(raw, ptype):
    return MinimaxM2Detector()._convert_param_value(raw, ptype)


def _qwen3(raw, ptype):
    return Qwen3CoderDetector()._convert_param_value(
        raw, "p", {"p": {"type": ptype}}, "f"
    )


def _poolside(raw, ptype):
    return PoolsideV1Detector._convert_param_value(raw, {"p": {"type": ptype}}, "p")


COERCERS = {
    "mimo": _mimo,
    "minimax_m2": _minimax,
    "qwen3_coder": _qwen3,
    "poolside_v1": _poolside,
}


class TestNonFiniteCoercion(unittest.TestCase):
    def test_non_finite_degrades_to_string(self):
        for name, coerce in COERCERS.items():
            for raw in NON_FINITE_NUMBERS:
                with self.subTest(detector=name, value=raw):
                    self.assertEqual(coerce(raw, "number"), raw)
            for raw in NON_FINITE_CONTAINERS:
                with self.subTest(detector=name, value=raw):
                    ptype = "object" if raw.startswith("{") else "array"
                    self.assertEqual(coerce(raw, ptype), raw)

    def test_finite_values_still_coerced(self):
        for name, coerce in COERCERS.items():
            with self.subTest(detector=name):
                self.assertEqual(coerce("42", "number"), 42)
                self.assertEqual(coerce("3.14", "number"), 3.14)
                self.assertEqual(coerce("[1, 2, 3]", "array"), [1, 2, 3])
                self.assertEqual(coerce('{"a": 1}', "object"), {"a": 1})


if __name__ == "__main__":
    unittest.main()
