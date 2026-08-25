import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import _convert_param_value as mimo_convert
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.srt.function_call.poolside_v1_detector import PoolsideV1Detector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _mimo(raw, param_type):
    tool = Tool(
        type="function",
        function=Function(
            name="f",
            parameters={
                "type": "object",
                "properties": {"p": {"type": param_type}},
            },
        ),
    )
    return mimo_convert(raw, "p", "f", [tool])


def _minimax(raw, param_type):
    return MinimaxM2Detector()._convert_param_value(raw, param_type)


def _qwen3(raw, param_type):
    return Qwen3CoderDetector()._convert_param_value(
        raw, "p", {"p": {"type": param_type}}, "f"
    )


def _poolside(raw, param_type):
    return PoolsideV1Detector._convert_param_value(
        raw, {"p": {"type": param_type}}, "p"
    )


COERCERS = (_mimo, _minimax, _qwen3, _poolside)


class TestNonFiniteCoercion(unittest.TestCase):
    def test_non_finite_numbers_remain_strings(self):
        for coerce in COERCERS:
            for param_type in ("integer", "number"):
                for raw in ("inf", "-inf", "Infinity", "1e999", "nan"):
                    with self.subTest(
                        detector=coerce.__name__, param_type=param_type, value=raw
                    ):
                        self.assertEqual(coerce(raw, param_type), raw)

    def test_non_finite_containers_remain_strings(self):
        cases = (
            ("array", "[1e999]"),
            ("array", "[NaN]"),
            ("array", "(1e999,)"),
            ("object", '{"x": 1e999}'),
            ("object", "{'x': 1e999}"),
        )
        for coerce in COERCERS:
            for param_type, raw in cases:
                with self.subTest(
                    detector=coerce.__name__, param_type=param_type, value=raw
                ):
                    self.assertEqual(coerce(raw, param_type), raw)

    def test_finite_values_are_still_coerced(self):
        for coerce in COERCERS:
            with self.subTest(detector=coerce.__name__):
                self.assertEqual(coerce("42", "number"), 42)
                self.assertEqual(coerce("3.14", "number"), 3.14)
                self.assertEqual(coerce("[1, 2, 3]", "array"), [1, 2, 3])
                self.assertEqual(coerce('{"a": 1}', "object"), {"a": 1})


if __name__ == "__main__":
    unittest.main()
