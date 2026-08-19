import unittest

from sglang.benchmark.serving import (
    RequestFuncOutput,
    _extract_spec_metrics_from_sglext,
    _resolve_accept_length,
)
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _output(
    completion_tokens: int = 0,
    verify_ct: int = 0,
    *,
    success: bool = True,
) -> RequestFuncOutput:
    return RequestFuncOutput(
        success=success,
        spec_completion_tokens=completion_tokens,
        spec_verify_ct=verify_ct,
    )


class TestBenchServingAcceptLength(unittest.TestCase):
    def test_dp_one_keeps_server_info_behavior(self):
        server_info = {
            "dp_size": 1,
            "internal_states": [{"avg_spec_accept_length": 3.25}],
        }

        value, source = _resolve_accept_length(
            server_info, [_output(completion_tokens=400, verify_ct=100)]
        )

        self.assertEqual(value, 3.25)
        self.assertEqual(source, "server_info")

    def test_unknown_dp_size_keeps_server_info_behavior(self):
        server_info = {
            "internal_states": [{"avg_spec_accept_length": 3.5}],
        }

        value, source = _resolve_accept_length(server_info, [])

        self.assertEqual(value, 3.5)
        self.assertEqual(source, "server_info")

    def test_dp_uses_verify_count_weighted_request_metrics(self):
        server_info = {
            "dp_size": 2,
            "internal_states": [
                {"avg_spec_accept_length": 99.0},
                {"avg_spec_accept_length": 99.0},
            ],
        }
        outputs = [
            _output(completion_tokens=400, verify_ct=100),
            _output(completion_tokens=20, verify_ct=10),
        ]

        value, source = _resolve_accept_length(server_info, outputs)

        self.assertAlmostEqual(value, 420 / 110)
        self.assertEqual(source, "per_request")

    def test_pd_wrapper_uses_decode_dp_size(self):
        server_info = {
            "prefill": [{"dp_size": 1}],
            "decode": [
                {
                    "dp_size": 4,
                    "internal_states": [{"avg_spec_accept_length": 99.0}],
                }
            ],
        }

        value, source = _resolve_accept_length(
            server_info, [_output(completion_tokens=37, verify_ct=10)]
        )

        self.assertEqual(value, 3.7)
        self.assertEqual(source, "per_request")

    def test_dp_ignores_failed_and_missing_request_metrics(self):
        server_info = {"dp_size": 2}
        outputs = [
            _output(completion_tokens=400, verify_ct=100, success=False),
            _output(),
        ]

        value, source = _resolve_accept_length(server_info, outputs)

        self.assertIsNone(value)
        self.assertEqual(source, "unavailable")

    def test_extracts_streaming_sglext_counters(self):
        output = RequestFuncOutput()

        _extract_spec_metrics_from_sglext(
            {
                "sglext": {
                    "spec_verify_ct": 12,
                    "spec_completion_tokens": 45,
                }
            },
            output,
        )

        self.assertEqual(output.spec_verify_ct, 12)
        self.assertEqual(output.spec_completion_tokens, 45)


if __name__ == "__main__":
    unittest.main()
