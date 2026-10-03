"""Reject disabled expert recording before sending scheduler control requests."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import FastAPI

from sglang.srt.entrypoints import http_server
from sglang.srt.managers import tokenizer_control_mixin
from sglang.srt.managers.io_struct import ExpertDistributionReqType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ACTIONS = (
    ("start", ExpertDistributionReqType.START_RECORD),
    ("stop", ExpertDistributionReqType.STOP_RECORD),
    ("dump", ExpertDistributionReqType.DUMP_RECORD),
)


class TestExpertDistributionControl(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = tokenizer_control_mixin.TokenizerControlMixin()
        self.manager.auto_create_handle_loop = Mock()
        self.manager.expert_distribution_communicator = AsyncMock(return_value=[])
        self.exec_config = SimpleNamespace(
            moe=SimpleNamespace(expert_distribution_recorder_mode=None)
        )
        config = patch.object(
            tokenizer_control_mixin, "get_exec", return_value=self.exec_config
        )
        config.start()
        self.addCleanup(config.stop)
        state = patch.object(
            http_server,
            "_global_state",
            SimpleNamespace(tokenizer_manager=self.manager),
        )
        state.start()
        self.addCleanup(state.stop)
        self.app = FastAPI()
        for action, _ in ACTIONS:
            self.app.api_route(
                f"/{action}_expert_distribution_record", methods=["GET", "POST"]
            )(getattr(http_server, f"{action}_expert_distribution_record_async"))

    async def test_disabled_controls_do_not_send_scheduler_requests(self):
        for action, _ in ACTIONS:
            with self.subTest(action=action):
                with self.assertRaisesRegex(
                    ValueError, "--expert-distribution-recorder-mode"
                ):
                    await getattr(
                        self.manager, f"{action}_expert_distribution_record"
                    )()
        self.manager.auto_create_handle_loop.assert_not_called()
        self.manager.expert_distribution_communicator.assert_not_awaited()

    async def test_enabled_modes_dispatch_the_requested_action(self):
        for mode in ("stat", "stat_approx", "per_pass", "per_token"):
            self.exec_config.moe.expert_distribution_recorder_mode = mode
            for action, expected in ACTIONS:
                with self.subTest(mode=mode, action=action):
                    await getattr(
                        self.manager, f"{action}_expert_distribution_record"
                    )()
                    request = (
                        self.manager.expert_distribution_communicator.await_args.args[0]
                    )
                    self.assertEqual(request.action, expected)
        self.assertEqual(self.manager.auto_create_handle_loop.call_count, 12)
        self.assertEqual(self.manager.expert_distribution_communicator.await_count, 12)

    async def test_disabled_http_controls_return_bad_request(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            for method in ("GET", "POST"):
                for action, _ in ACTIONS:
                    with self.subTest(method=method, action=action):
                        response = await client.request(
                            method, f"/{action}_expert_distribution_record"
                        )
                        self.assertEqual(response.status_code, 400)
                        self.assertIn(
                            "--expert-distribution-recorder-mode",
                            response.json()["error"]["message"],
                        )
        self.manager.expert_distribution_communicator.assert_not_awaited()

    async def test_enabled_http_controls_keep_success_response(self):
        self.exec_config.moe.expert_distribution_recorder_mode = "stat"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            for action, _ in ACTIONS:
                with self.subTest(action=action):
                    response = await client.post(
                        f"/{action}_expert_distribution_record"
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(response.text.startswith(action.capitalize()))
        self.assertEqual(self.manager.expert_distribution_communicator.await_count, 3)

    async def test_unrelated_transport_failure_is_not_converted_to_bad_request(self):
        self.exec_config.moe.expert_distribution_recorder_mode = "stat"
        failure = RuntimeError("controlled transport failure")
        self.manager.expert_distribution_communicator.side_effect = failure
        for action, _ in ACTIONS:
            with self.subTest(action=action):
                with self.assertRaises(RuntimeError) as caught:
                    await getattr(
                        http_server, f"{action}_expert_distribution_record_async"
                    )()
                self.assertIs(caught.exception, failure)


if __name__ == "__main__":
    unittest.main()
