import asyncio
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestProfileConflicts(CustomTestCase):
    def test_start_profile_success(self):
        tokenizer_manager = SimpleNamespace(
            start_profile=AsyncMock(return_value=None),
        )
        prior_state = http_server.get_global_state()
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=tokenizer_manager))
        try:
            response = asyncio.run(http_server.start_profile_async())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.body, b"Start profiling.\n")
        finally:
            http_server._global_state = prior_state

    def test_start_profile_conflict_returns_409(self):
        conflict_msg = "Profiling is already in progress. Call /stop_profile first."
        tokenizer_manager = SimpleNamespace(
            start_profile=AsyncMock(side_effect=RuntimeError(conflict_msg)),
        )
        prior_state = http_server.get_global_state()
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=tokenizer_manager))
        try:
            response = asyncio.run(http_server.start_profile_async())
            self.assertEqual(response.status_code, HTTPStatus.CONFLICT)
            self.assertEqual(response.body.decode().strip(), conflict_msg)
        finally:
            http_server._global_state = prior_state

    def test_stop_profile_success(self):
        tokenizer_manager = SimpleNamespace(
            stop_profile=AsyncMock(return_value=None),
        )
        prior_state = http_server.get_global_state()
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=tokenizer_manager))
        try:
            response = asyncio.run(http_server.stop_profile_async())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.body, b"Stop profiling. This will take some time.\n")
        finally:
            http_server._global_state = prior_state

    def test_stop_profile_conflict_returns_409(self):
        conflict_msg = "Profiling is not in progress. Call /start_profile first."
        tokenizer_manager = SimpleNamespace(
            stop_profile=AsyncMock(side_effect=RuntimeError(conflict_msg)),
        )
        prior_state = http_server.get_global_state()
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=tokenizer_manager))
        try:
            response = asyncio.run(http_server.stop_profile_async())
            self.assertEqual(response.status_code, HTTPStatus.CONFLICT)
            self.assertEqual(response.body.decode().strip(), conflict_msg)
        finally:
            http_server._global_state = prior_state


if __name__ == "__main__":
    unittest.main(verbosity=2)
