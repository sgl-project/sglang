import unittest

from sglang.srt.managers.io_struct import (
    AbortReq,
    ContinueGenerationReqInput,
    FlushCacheReqInput,
    PauseGenerationReqInput,
    SequencedControlReq,
)
from sglang.srt.managers.scheduler_components.control_seq_gate import ControlSeqGate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _numbered(seq, req):
    return SequencedControlReq.wrap(req, seq)


class TestControlSeqGate(CustomTestCase):
    """Every attention-DP group must apply a numbered control request between
    the same two MLP-sync gathers, and a paused engine runs none."""

    def test_held_until_every_group_received_it(self):
        gate = ControlSeqGate()
        abort = AbortReq(rid="a")
        passed = gate.take_inputs(
            [_numbered(1, FlushCacheReqInput()), abort], align=True
        )
        # Work passes the waiting control request.
        self.assertEqual(passed, [abort])
        self.assertEqual(gate.seq_received, 1)
        gate.release(1)
        self.assertIsInstance(gate.take_inputs([], align=True)[0], FlushCacheReqInput)

    def test_paused_engine_applies_on_arrival_until_continue(self):
        gate = ControlSeqGate()
        gate.release(1)
        passed = gate.take_inputs(
            [
                _numbered(1, PauseGenerationReqInput(mode="in_place")),
                _numbered(2, FlushCacheReqInput()),
                _numbered(3, ContinueGenerationReqInput()),
                _numbered(4, FlushCacheReqInput()),
            ],
            align=True,
        )
        # 2 and 3 follow the pause into the paused window; 4 follows the
        # continue, after which the groups gather again, so it waits.
        self.assertEqual(
            [type(req) for req in passed],
            [
                PauseGenerationReqInput,
                FlushCacheReqInput,
                ContinueGenerationReqInput,
            ],
        )
        self.assertEqual(gate.take_inputs([], align=True), [])

    def test_paused_in_an_earlier_call_applies_on_arrival(self):
        gate = ControlSeqGate()
        gate.release(1)
        gate.take_inputs(
            [_numbered(1, PauseGenerationReqInput(mode="in_place"))], align=True
        )
        passed = gate.take_inputs([_numbered(2, FlushCacheReqInput())], align=True)
        self.assertIsInstance(passed[0], FlushCacheReqInput)

    def test_a_group_goes_on_from_the_numbers_it_is_sent(self):
        # The DP controller sends nothing to a group while it is inactive, and
        # nothing before an elastic EP joiner joins: 2 never comes, 3 must not
        # wait for it.
        gate = ControlSeqGate()
        received = [_numbered(seq, FlushCacheReqInput()) for seq in (1, 3)]
        self.assertEqual(gate.take_inputs(received, align=True), [])
        self.assertEqual(gate.seq_received, 3)
        gate.release(3)
        self.assertEqual(len(gate.take_inputs([], align=True)), 2)


if __name__ == "__main__":
    unittest.main()
