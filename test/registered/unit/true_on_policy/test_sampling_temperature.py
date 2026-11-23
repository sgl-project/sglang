import math
import struct
import unittest

from sglang.srt.sampling.sampling_params import _SAMPLING_EPS
from sglang.srt.true_on_policy.sampling import validate_sampling_temperature
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _fp32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def _next_fp32_up(x: float) -> float:
    (bits,) = struct.unpack("I", struct.pack("f", x))
    return struct.unpack("f", struct.pack("I", bits + 1))[0]


class TestTemperatureFloor(unittest.TestCase):
    def test_refuses_what_serving_turns_greedy(self):
        """SamplingParams.normalize turns 0 <= t < _SAMPLING_EPS greedy; FP32(1e-6) is below it."""
        self.assertLess(_fp32(1e-6), _SAMPLING_EPS)
        for t in (0.0, -1.0, 1e-7, 1e-6, _fp32(1e-6)):
            with self.subTest(t=t), self.assertRaisesRegex(ValueError, "greedy"):
                validate_sampling_temperature(t)

    def test_accepts_the_first_fp32_value_serving_samples(self):
        t = _next_fp32_up(1e-6)
        self.assertGreaterEqual(t, _SAMPLING_EPS)
        self.assertEqual(validate_sampling_temperature(t), t)

    def test_returns_the_fp32_value(self):
        for t in (1e-5, 0.7, 1.0, 1.1):
            with self.subTest(t=t):
                self.assertEqual(validate_sampling_temperature(t), _fp32(t))

    def test_refuses_non_finite_and_fp32_overflow(self):
        for t in (math.inf, -math.inf, math.nan, 1e39):
            with self.subTest(t=t), self.assertRaisesRegex(ValueError, "finite"):
                validate_sampling_temperature(t)

    def test_error_names_the_temperature(self):
        with self.assertRaisesRegex(ValueError, r"1e-06"):
            validate_sampling_temperature(1e-6)


if __name__ == "__main__":
    unittest.main()
