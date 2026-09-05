import unittest

from sglang.benchmark.serving import (
    RequestFuncOutput,
    _extract_spec_metrics_from_sglext,
    _resolve_accept_length,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _output(
    accept_length: float = 0.0,
    verify_ct: int = 0,
    *,
    success: bool = True,
) -> RequestFuncOutput:
    return RequestFuncOutput(
        success=success,
        spec_accept_length=accept_length,
        spec_verify_ct=verify_ct,
    )


class TestBenchServingAcceptLength(unittest.TestCase):
    def test_dp_one_keeps_server_info_behavior(self):
        server_info = {
            "dp_size": 1,
            "internal_states": [{"avg_spec_accept_length": 3.25}],
        }

        value, source = _resolve_accept_length(
            server_info, [_output(accept_length=4.0, verify_ct=100)]
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
        server_info = {"dp_size": 2}
        outputs = [
            _output(accept_length=4.0, verify_ct=100),
            _output(accept_length=2.0, verify_ct=10),
        ]

        value, source = _resolve_accept_length(server_info, outputs)

        self.assertAlmostEqual(value, 420 / 110)
        self.assertEqual(source, "per_request")

    def test_pd_wrapper_uses_decode_dp_size(self):
        server_info = {
            "prefill": [{"dp_size": 1}],
            "decode": [{"dp_size": 4}],
        }

        value, source = _resolve_accept_length(
            server_info, [_output(accept_length=3.7, verify_ct=10)]
        )

        self.assertEqual(value, 3.7)
        self.assertEqual(source, "per_request")

    def test_dp_ignores_failed_and_missing_request_metrics(self):
        server_info = {"dp_size": 2}
        outputs = [
            _output(accept_length=4.0, verify_ct=100, success=False),
            _output(),
        ]

        value, source = _resolve_accept_length(server_info, outputs)

        self.assertIsNone(value)
        self.assertEqual(source, "unavailable")

    def test_extracts_streaming_sglext_details(self):
        output = RequestFuncOutput()

        _extract_spec_metrics_from_sglext(
            {
                "sglext": {
                    "spec_tokens_details": {
                        "spec_accept_length": 3.75,
                        "spec_verify_ct": 12,
                    }
                }
            },
            output,
        )

        self.assertEqual(output.spec_accept_length, 3.75)
        self.assertEqual(output.spec_verify_ct, 12)

    def test_extracts_and_aggregates_multiple_choices(self):
        output = RequestFuncOutput()

        _extract_spec_metrics_from_sglext(
            {
                "sglext": {
                    "spec_tokens_details": [
                        {"spec_accept_length": 4.0, "spec_verify_ct": 10},
                        {"spec_accept_length": 2.0, "spec_verify_ct": 2},
                    ]
                }
            },
            output,
        )

        self.assertAlmostEqual(output.spec_accept_length, 44 / 12)
        self.assertEqual(output.spec_verify_ct, 12)


if __name__ == "__main__":
    unittest.main()
