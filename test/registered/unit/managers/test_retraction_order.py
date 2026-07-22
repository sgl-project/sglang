import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=2, suite="base-c-test-cpu")


def _req(output_len: int, input_len: int = 8, priority=None):
    return SimpleNamespace(
        output_ids=[0] * output_len,
        origin_input_ids=[0] * input_len,
        priority=priority,
    )


def _args(policy: str = "length", low_first: bool = False):
    return SimpleNamespace(
        retraction_policy=policy,
        schedule_low_priority_values_first=low_first,
    )


def _order(reqs, args):
    return ScheduleBatch._get_decode_retraction_order(reqs, args)


class TestRetractionOrder(CustomTestCase):
    """The retraction loop pops from the END of the returned list, so the
    last index is the first request retracted."""

    def test_length_policy_retracts_shortest_output_first(self):
        reqs = [_req(5), _req(1), _req(3)]
        self.assertEqual(_order(reqs, _args()), [0, 2, 1])

    def test_length_policy_tie_breaks_on_longer_input(self):
        # Equal outputs: the longer-input request is retracted first
        # (frees more tokens for the same rework).
        reqs = [_req(4, input_len=10), _req(4, input_len=20)]
        self.assertEqual(_order(reqs, _args()), [0, 1])

    def test_priority_policy_low_values_first(self):
        # Low value = more important; None sorts as least important.
        reqs = [_req(4, priority=2), _req(4, priority=0), _req(4, priority=None)]
        self.assertEqual(_order(reqs, _args("priority", low_first=True)), [1, 0, 2])

    def test_priority_policy_high_values_first(self):
        reqs = [_req(4, priority=2), _req(4, priority=0), _req(4, priority=None)]
        self.assertEqual(_order(reqs, _args("priority", low_first=False)), [0, 1, 2])

    def test_priority_ties_fall_back_to_length(self):
        reqs = [_req(1, priority=1), _req(5, priority=1)]
        self.assertEqual(_order(reqs, _args("priority", low_first=True)), [1, 0])


if __name__ == "__main__":
    unittest.main()
