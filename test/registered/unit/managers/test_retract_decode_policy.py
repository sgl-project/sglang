import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=2, suite="base-c-test-cpu")


class TestRetractDecodePolicy(unittest.TestCase):
    def _req(
        self,
        rid: str,
        *,
        priority: int,
        input_len: int,
        output_len: int,
    ):
        return SimpleNamespace(
            rid=rid,
            priority=priority,
            origin_input_ids=[0] * input_len,
            output_ids=[0] * output_len,
        )

    def _server_args(
        self, *, retraction_policy: str, low_priority_values_first: bool = False
    ):
        return SimpleNamespace(
            retraction_policy=retraction_policy,
            schedule_low_priority_values_first=low_priority_values_first,
        )

    def _retraction_sequence(self, reqs, server_args, *, allow_policy_sort=True):
        keep_order = ScheduleBatch._get_decode_retraction_order(
            reqs, server_args, allow_policy_sort=allow_policy_sort
        )
        return [reqs[i].rid for i in reversed(keep_order)]

    def test_length_policy_preserves_existing_order(self):
        reqs = [
            self._req("long-output", priority=100, input_len=10, output_len=10),
            self._req(
                "short-output-short-input", priority=1, input_len=10, output_len=1
            ),
            self._req(
                "short-output-long-input", priority=1, input_len=50, output_len=1
            ),
        ]

        self.assertEqual(
            self._retraction_sequence(
                reqs, self._server_args(retraction_policy="length")
            ),
            [
                "short-output-long-input",
                "short-output-short-input",
                "long-output",
            ],
        )

    def test_priority_policy_retracts_lower_priority_values_first_by_default(self):
        reqs = [
            self._req("low", priority=1, input_len=1, output_len=100),
            self._req("high", priority=10, input_len=100, output_len=0),
            self._req("mid", priority=5, input_len=10, output_len=50),
        ]

        self.assertEqual(
            self._retraction_sequence(
                reqs, self._server_args(retraction_policy="priority")
            ),
            ["low", "mid", "high"],
        )

    def test_priority_policy_can_retract_higher_priority_values_first(self):
        reqs = [
            self._req("high", priority=1, input_len=100, output_len=0),
            self._req("low", priority=10, input_len=1, output_len=100),
            self._req("mid", priority=5, input_len=10, output_len=50),
        ]

        self.assertEqual(
            self._retraction_sequence(
                reqs,
                self._server_args(
                    retraction_policy="priority", low_priority_values_first=True
                ),
            ),
            ["low", "mid", "high"],
        )

    def test_priority_policy_uses_length_as_tiebreaker(self):
        reqs = [
            self._req("long-output", priority=5, input_len=10, output_len=10),
            self._req(
                "short-output-short-input", priority=5, input_len=10, output_len=1
            ),
            self._req(
                "short-output-long-input", priority=5, input_len=50, output_len=1
            ),
        ]

        self.assertEqual(
            self._retraction_sequence(
                reqs, self._server_args(retraction_policy="priority")
            ),
            [
                "short-output-long-input",
                "short-output-short-input",
                "long-output",
            ],
        )

    def test_spec_decode_keeps_back_of_batch_retraction(self):
        reqs = [
            self._req("low", priority=1, input_len=1, output_len=100),
            self._req("high", priority=10, input_len=100, output_len=0),
            self._req("mid", priority=5, input_len=10, output_len=50),
        ]

        self.assertEqual(
            self._retraction_sequence(
                reqs,
                self._server_args(retraction_policy="priority"),
                allow_policy_sort=False,
            ),
            ["mid", "high", "low"],
        )


if __name__ == "__main__":
    unittest.main()
