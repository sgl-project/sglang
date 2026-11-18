"""Expert-distribution record controls on a server without a recorder must fail
the request, not raise inside the scheduler and take the server down."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.io_struct import (
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ACTIONS = {
    ExpertDistributionReqType.START_RECORD: "start_record",
    ExpertDistributionReqType.STOP_RECORD: "stop_record",
    ExpertDistributionReqType.DUMP_RECORD: "dump_record",
}
_VERBS = ("start", "stop", "dump")
_FAILED = ExpertDistributionReqOutput(success=False, message="recorder disabled")


def _exec_with_mode(mode):
    return lambda: SimpleNamespace(
        moe=SimpleNamespace(expert_distribution_recorder_mode=mode)
    )


class TestSchedulerExpertDistributionHandle(CustomTestCase):
    def _handle(self, mode, action):
        recorder = MagicMock()
        with (
            patch.object(scheduler_module, "get_exec", _exec_with_mode(mode)),
            patch.object(
                scheduler_module,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            out = Scheduler.expert_distribution_handle(
                MagicMock(), ExpertDistributionReq(action=action)
            )
        return out, recorder

    def test_disabled_recorder_returns_failure_without_calling_recorder(self):
        for action, method in _ACTIONS.items():
            with self.subTest(action=action):
                out, recorder = self._handle(None, action)
                self.assertFalse(out.success)
                self.assertIn("--expert-distribution-recorder-mode", out.message)
                getattr(recorder, method).assert_not_called()

    def test_enabled_recorder_runs_action(self):
        for action, method in _ACTIONS.items():
            with self.subTest(action=action):
                out, recorder = self._handle("stat", action)
                self.assertTrue(out.success)
                getattr(recorder, method).assert_called_once_with()


class TestTokenizerExpertDistributionMerge(CustomTestCase):
    def _run(self, verb, results):
        manager = MagicMock()
        manager.expert_distribution_communicator = AsyncMock(return_value=results)
        manager._execute_expert_distribution = lambda action: (
            TokenizerControlMixin._execute_expert_distribution(manager, action)
        )
        method = getattr(TokenizerControlMixin, f"{verb}_expert_distribution_record")
        return asyncio.run(method(manager))

    def test_any_failed_rank_fails_the_request(self):
        ok = ExpertDistributionReqOutput()
        for verb in _VERBS:
            with self.subTest(verb=verb):
                self.assertIs(self._run(verb, [ok, _FAILED]), _FAILED)
                self.assertTrue(self._run(verb, [ok, ok]).success)


class TestExpertDistributionEntrypoints(CustomTestCase):
    def _call_route(self, verb, result):
        tm = MagicMock()
        setattr(
            tm, f"{verb}_expert_distribution_record", AsyncMock(return_value=result)
        )
        route = getattr(http_server, f"{verb}_expert_distribution_record_async")
        with patch.object(
            http_server, "_global_state", SimpleNamespace(tokenizer_manager=tm)
        ):
            return asyncio.run(route())

    def test_http_routes_return_400_only_on_failure(self):
        for verb in _VERBS:
            with self.subTest(verb=verb):
                resp = self._call_route(verb, _FAILED)
                self.assertEqual(resp.status_code, 400)
                self.assertEqual(resp.body, b"recorder disabled")
                ok = self._call_route(verb, ExpertDistributionReqOutput())
                self.assertEqual(ok.status_code, 200)

    def test_engine_raises_on_failure(self):
        for verb in _VERBS:
            with self.subTest(verb=verb):
                loop = asyncio.new_event_loop()
                try:
                    engine = SimpleNamespace(loop=loop, tokenizer_manager=MagicMock())
                    setattr(
                        engine.tokenizer_manager,
                        f"{verb}_expert_distribution_record",
                        AsyncMock(return_value=_FAILED),
                    )
                    method = getattr(Engine, f"{verb}_expert_distribution_record")
                    with self.assertRaisesRegex(RuntimeError, "recorder disabled"):
                        method(engine)
                finally:
                    loop.close()


if __name__ == "__main__":
    unittest.main()
