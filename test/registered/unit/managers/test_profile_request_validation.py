"""Reject invalid stage limits before a profiling request reaches workers."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import msgspec

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestProfileRequestValidation(CustomTestCase):
    def test_stage_requires_positive_steps(self):
        for limit in (None, 0, -1):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "positive num_steps"):
                    ProfileReq(profile_by_stage=True, num_steps=limit)

    def test_http_rejects_before_calling_tokenizer_manager(self):
        async def run():
            manager = SimpleNamespace(start_profile=AsyncMock())
            state = SimpleNamespace(tokenizer_manager=manager)
            transport = httpx.ASGITransport(app=http_server.app)
            with patch.object(http_server, "_global_state", state):
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    for verb in ("GET", "POST"):
                        for req_type in (1, 2):
                            for limit in (None, 0, -1):
                                with self.subTest(
                                    verb=verb, req_type=req_type, limit=limit
                                ):
                                    response = await client.request(
                                        verb,
                                        "/start_profile",
                                        json={
                                            "req_type": req_type,
                                            "profile_by_stage": True,
                                            "num_steps": limit,
                                        },
                                    )
                                    self.assertEqual(response.status_code, 400)
                                    self.assertIn("positive num_steps", response.text)
            manager.start_profile.assert_not_called()

        asyncio.run(run())

    def test_engine_rejects_before_calling_tokenizer_manager(self):
        manager = Mock()
        loop = Mock()
        engine = SimpleNamespace(tokenizer_manager=manager, loop=loop)
        for limit in (None, 0, -1):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "positive num_steps"):
                    Engine.start_profile(engine, profile_by_stage=True, num_steps=limit)
        manager.start_profile.assert_not_called()
        loop.run_until_complete.assert_not_called()

    def test_invalid_stage_limits_are_rejected_on_ipc_decode(self):
        for limit in (None, 0, -1):
            with self.subTest(limit=limit):
                request = ProfileReq(profile_by_stage=True, num_steps=1)
                request.num_steps = limit
                payload = msgspec.msgpack.encode(request)
                with self.assertRaisesRegex(
                    msgspec.ValidationError, "positive num_steps"
                ):
                    msgspec.msgpack.decode(payload, type=ProfileReq)

    def test_valid_requests_preserve_serialization(self):
        requests = [
            ProfileReq(),
            ProfileReq(num_steps=0),
            ProfileReq(num_steps=-1),
            ProfileReq(start_step=3, num_steps=2),
            ProfileReq(profile_by_stage=True, num_steps=1),
            ProfileReq(req_type=ProfileReqType.STOP_PROFILE),
        ]
        for request in requests:
            with self.subTest(request=request):
                self.assertEqual(
                    msgspec.msgpack.decode(
                        msgspec.msgpack.encode(request), type=ProfileReq
                    ),
                    request,
                )

    def test_http_keeps_positive_stage_and_manual_requests(self):
        async def run():
            manager = SimpleNamespace(start_profile=AsyncMock())
            transport = httpx.ASGITransport(app=http_server.app)
            with patch.object(
                http_server, "_global_state", SimpleNamespace(tokenizer_manager=manager)
            ):
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    for payload in (
                        {},
                        {"num_steps": 0},
                        {"profile_by_stage": True, "num_steps": 2},
                        {"profile_by_stage": True, "num_steps": "2"},
                    ):
                        with self.subTest(payload=payload):
                            response = await client.post("/start_profile", json=payload)
                            self.assertEqual(response.status_code, 200)
                    self.assertEqual(manager.start_profile.await_count, 4)
                    self.assertEqual(
                        manager.start_profile.call_args.args[0].num_steps, 2
                    )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
