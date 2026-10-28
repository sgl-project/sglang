import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import (
    ReqKvInfo,
    get_decode_retraction_order,
    new_tokens_required_next_decode,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _req(output_len: int, input_len: int = 8, priority=None):
    return SimpleNamespace(
        output_ids=[0] * output_len,
        origin_input_ids=[0] * input_len,
        priority=priority,
    )


def _args(policy: str = "length", low_first: bool = False):
    """The retraction order reads the schedule bag, so the policy has to be
    published rather than handed in."""
    return ServerArgs(
        model_path="dummy",
        retraction_policy=policy,
        schedule_low_priority_values_first=low_first,
    )


def _order(reqs, args):
    publish(args, role="test")
    try:
        return get_decode_retraction_order(reqs)
    finally:
        reset_context()


class TestRetractionOrder(CustomTestCase):
    """The retraction loop pops from the END of the returned list, so the
    last index is the first request retracted."""

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


class TestNextDecodePages(CustomTestCase):
    """The next decode appends at the allocated end, so a row needs a new page
    when its allocated length sits on a page boundary, whatever its committed
    length (launched steps allocate ahead of their commit)."""

    def _num_tokens(self, committed: int, allocated: int) -> int:
        req = SimpleNamespace(
            kv=ReqKvInfo(kv_committed_len=committed, kv_allocated_len=allocated),
            beam_group=None,
        )
        return new_tokens_required_next_decode(
            [req], page_size=4, spec_algorithm=SpeculativeAlgorithm.NONE
        )

    def test_allocated_on_a_boundary_needs_a_page(self):
        self.assertEqual(self._num_tokens(committed=6, allocated=8), 4)

    def test_committed_on_a_boundary_needs_none(self):
        self.assertEqual(self._num_tokens(committed=4, allocated=7), 0)


if __name__ == "__main__":
    unittest.main()
