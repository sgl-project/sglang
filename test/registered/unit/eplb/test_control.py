"""PP control terminals, replay and failure recovery without model loading."""

import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.eplb.control import (
    ExpertDistributionCommunicator,
    ExpertDistributionControl,
    recorder_control_fan_out,
)
from sglang.srt.eplb.expert_distribution import _ExpertDistributionRecorderReal
from sglang.srt.managers.io_struct import (
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
)
from sglang.srt.runtime_context import get_context, get_resources
from sglang.srt.utils import Withable
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _request(seq, action=ExpertDistributionReqType.START_RECORD):
    return ExpertDistributionReq(action=action, operation_id=f"test:{seq}")


def _reply(req, pp, *, success=True):
    return ExpertDistributionReqOutput(
        operation_id=req.operation_id,
        pp_rank=pp,
        root_global_rank=pp * 2,
        success=success,
        message="PP1 dump failed" if not success else "",
    )


class TestExpertDistributionControl(CustomTestCase):
    def test_unconfigured_recorder_returns_terminal_without_killing_owner(self):
        """#42071: the default Noop must not throw out of scheduler dispatch."""
        with get_context().override_server_args(device="cpu"):
            control = ExpertDistributionControl(tokenizer_ipc_name="unused")
            domain = SimpleNamespace(ep_size=1, pp_rank=0, root_global_rank=0)
            for seq, action in enumerate(ExpertDistributionReqType, 1):
                result = control.execute(_request(seq, action), domain=domain)
                self.assertFalse(result.success)
                self.assertIn("not configured", result.message)

    def test_terminal_count_uses_ep_owners_across_stages_and_replicas(self):
        """Attention DP heads and EP owners can have different group counts."""
        with get_context().override_server_args(
            device="cpu",
            tp_size=8,
            pp_size=2,
            ep_size=4,
            moe_dp_size=2,
            attn_dp_size=2,
        ):
            self.assertEqual(recorder_control_fan_out(), 4)
        with get_context().override_server_args(
            device="cpu",
            tp_size=8,
            pp_size=2,
            ep_size=8,
            dp_size=2,
        ):
            self.assertEqual(recorder_control_fan_out(), 4)

    def test_waits_for_distinct_stage_terminals_and_recovers_after_error(self):
        """PP0 and its duplicate cannot hide PP1 failure or satisfy the next op."""

        async def scenario():
            sent = []
            comm = ExpertDistributionCommunicator(sent.append, fan_out=2)
            req = _request(1)
            task = asyncio.create_task(comm(req))
            await asyncio.sleep(0)
            comm.handle_recv(_reply(req, 0))
            comm.handle_recv(_reply(req, 0))
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            comm.handle_recv(_reply(req, 1, success=False))
            with self.assertRaisesRegex(RuntimeError, "PP1 dump failed"):
                await task

            next_req = _request(2)
            task = asyncio.create_task(comm(next_req))
            await asyncio.sleep(0)
            comm.handle_recv(_reply(req, 1))
            comm.handle_recv(_reply(next_req, 0))
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            comm.handle_recv(_reply(next_req, 1))
            self.assertEqual(len(await task), 2)

        asyncio.run(scenario())

    def test_timeout_drops_late_reply_and_unblocks_next_request(self):
        """A cancelled waiter must not inherit an earlier operation's terminal."""

        async def scenario():
            comm = ExpertDistributionCommunicator(lambda req: None, 1, timeout=0)
            req = _request(1)
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                await comm(req)
            comm._timeout = 2
            next_req = _request(2)
            task = asyncio.create_task(comm(next_req))
            await asyncio.sleep(0)
            comm.handle_recv(_reply(req, 0))
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            comm.handle_recv(_reply(next_req, 0))
            self.assertEqual(len(await task), 1)

        asyncio.run(scenario())

    def test_recorder_replay_and_io_failure_do_not_poison_next_operation(self):
        """An evicted START cannot clear counters; failed DUMP can be retried anew."""

        class Accumulator:
            resets = 0
            dumps = 0
            fail = False

            def reset(self):
                self.resets += 1

            def dump(self, **kwargs):
                if self.fail:
                    raise OSError("disk full")
                self.dumps += 1

        with get_context().override_server_args(
            expert_distribution_recorder_mode="stat", enable_eplb=False, device="cpu"
        ):
            recorder = _ExpertDistributionRecorderReal.__new__(
                _ExpertDistributionRecorderReal
            )
            recorder._recording = False
            recorder._current_layer_idx = Withable()
            recorder._single_pass_gatherers = {}
            recorder._accumulator = Accumulator()
            get_resources().expert_distribution_recorder = recorder
            control = ExpertDistributionControl(tokenizer_ipc_name="unused")
            domain = SimpleNamespace(ep_size=1, pp_rank=0, root_global_rank=0)
            req = _request(1)
            self.assertTrue(control.execute(req, domain=domain).success)
            self.assertTrue(control.execute(req, domain=domain).success)
            self.assertEqual(recorder._accumulator.resets, 1)
            # Fill the terminal cache, then replay an evicted id.
            for seq in range(2, 67):
                self.assertTrue(
                    control.execute(
                        _request(seq, ExpertDistributionReqType.STOP_RECORD),
                        domain=domain,
                    ).success
                )
            self.assertFalse(control.execute(req, domain=domain).success)
            self.assertEqual(recorder._accumulator.resets, 1)
            recorder._accumulator.fail = True
            resets_before_failure = recorder._accumulator.resets
            failed = control.execute(
                _request(67, ExpertDistributionReqType.DUMP_RECORD), domain=domain
            )
            self.assertFalse(failed.success)
            self.assertIn("disk full", failed.message)
            self.assertEqual(recorder._accumulator.resets, resets_before_failure + 1)
            recorder._accumulator.fail = False
            self.assertTrue(
                control.execute(
                    _request(68, ExpertDistributionReqType.STOP_RECORD), domain=domain
                ).success
            )
            self.assertFalse(recorder.recording)
            self.assertTrue(
                control.execute(
                    _request(69, ExpertDistributionReqType.DUMP_RECORD), domain=domain
                ).success
            )
            self.assertEqual(recorder._accumulator.dumps, 1)


if __name__ == "__main__":
    unittest.main()
