"""Control calls stop waiting on retirees as soon as a shrink starts: a retiree can
exit long before the survivors commit the narrower width."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import sglang.srt.managers.tokenizer_control_mixin as mixin
from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.test_utils import CustomTestCase

LAUNCH_WORKERS = 4
SURVIVORS = 2


class _Tokenizer:
    _close_elastic_shrink_gate = TokenizerManager._close_elastic_shrink_gate
    _open_elastic_shrink_gate = TokenizerManager._open_elastic_shrink_gate
    update_control_communicator_fan_out = (
        TokenizerManager.update_control_communicator_fan_out
    )

    def __init__(self):
        self.elastic_worker_count = LAUNCH_WORKERS
        self._elastic_shrink_retiring = set()
        self._elastic_shrink_pause_event = asyncio.Event()
        self._elastic_shrink_pause_event.set()
        for spec in mixin._COMMUNICATOR_SPECS:
            comm = FanOutCommunicator(send=lambda obj: None, fan_out=LAUNCH_WORKERS)
            setattr(self, f"{spec[0]}_communicator", comm)

    def fan_outs(self):
        return {
            spec[0]: getattr(self, f"{spec[0]}_communicator")._fan_out
            for spec in mixin._COMMUNICATOR_SPECS
        }


class TestShrinkControlFanOut(CustomTestCase):
    def setUp(self):
        super().setUp()
        parallel = SimpleNamespace(
            attn_dp_enabled=True,
            enable_dp_attention_local_control_broadcast=True,
            tp_size=1,
        )
        for patch in (
            mock.patch.object(mixin, "get_parallel", lambda: parallel),
            mock.patch.object(mixin, "is_control_broadcast_local", lambda: True),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_shrink_start_narrows_all_but_the_scale_call_and_reopen_restores(self):
        tm = _Tokenizer()
        tm._close_elastic_shrink_gate(range(SURVIVORS, LAUNCH_WORKERS))
        narrowed = tm.fan_outs()
        self.assertEqual(narrowed.pop("scale_elastic_ep"), LAUNCH_WORKERS)
        self.assertEqual(set(narrowed.values()), {SURVIVORS})

        tm._open_elastic_shrink_gate()
        self.assertEqual(set(tm.fan_outs().values()), {LAUNCH_WORKERS})

    def test_an_in_flight_call_returns_without_the_retirees(self):
        async def body():
            tm = _Tokenizer()
            comm = tm.get_internal_state_communicator
            call = asyncio.ensure_future(comm(None))
            await asyncio.sleep(0)
            for sender in range(SURVIVORS):
                comm.handle_recv(f"r{sender}", sender=sender)
            tm._close_elastic_shrink_gate(range(SURVIVORS, LAUNCH_WORKERS))
            # A retiree that is still alive replies too; it must not be counted.
            comm.handle_recv("retiree", sender=SURVIVORS)
            return await asyncio.wait_for(call, timeout=3.0)

        self.assertEqual(asyncio.run(body()), ["r0", "r1"])

    def test_an_empty_gate_leaves_the_fan_out_alone(self):
        tm = _Tokenizer()
        tm._close_elastic_shrink_gate(())
        tm._open_elastic_shrink_gate()
        self.assertEqual(set(tm.fan_outs().values()), {LAUNCH_WORKERS})


if __name__ == "__main__":
    unittest.main()
