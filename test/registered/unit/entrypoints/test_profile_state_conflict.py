"""`/start_profile` and `/stop_profile` must report profiler state conflicts as 409.

Refused profiler requests (`ProfileReqOutput(success=False, ...)`) used to be
turned into a generic `RuntimeError`, which the HTTP endpoints did not catch, so
a client calling `/stop_profile` after a completed (or never started) profile
got a bare 500 and an ASGI traceback. They now surface as 409 Conflict with the
scheduler's message, while unexpected failures still propagate as 500.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.tokenizer_control_mixin import ProfileStateError
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RefusingTokenizerManager:
    """Tokenizer manager whose profiler refuses the request."""

    def __init__(self, message: str):
        self.message = message

    async def start_profile(self, _obj=None):
        raise ProfileStateError(self.message)

    async def stop_profile(self):
        raise ProfileStateError(self.message)


class _AcceptingTokenizerManager:
    """Tokenizer manager whose profiler accepts the request."""

    async def start_profile(self, _obj=None):
        return None

    async def stop_profile(self):
        return None


class _BrokenTokenizerManager:
    """Tokenizer manager that fails for an unrelated reason."""

    async def start_profile(self, _obj=None):
        raise RuntimeError("communicator exploded")

    async def stop_profile(self):
        raise RuntimeError("communicator exploded")


class TestProfileStateConflict(unittest.TestCase):
    def _call(self, endpoint, manager):
        with patch.object(
            http_server, "_global_state", SimpleNamespace(tokenizer_manager=manager)
        ):
            return asyncio.run(endpoint())

    def test_stop_profile_conflict_is_409(self):
        message = "Profiling is not in progress. Call /start_profile first."
        with self.assertRaises(HTTPException) as ctx:
            self._call(
                http_server.stop_profile_async, _RefusingTokenizerManager(message)
            )
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.detail, message)

    def test_start_profile_conflict_is_409(self):
        message = "Profiling is already in progress."
        with self.assertRaises(HTTPException) as ctx:
            self._call(
                http_server.start_profile_async, _RefusingTokenizerManager(message)
            )
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.detail, message)

    def test_accepted_requests_still_return_200(self):
        stop_response = self._call(
            http_server.stop_profile_async, _AcceptingTokenizerManager()
        )
        self.assertEqual(stop_response.status_code, 200)
        start_response = self._call(
            http_server.start_profile_async, _AcceptingTokenizerManager()
        )
        self.assertEqual(start_response.status_code, 200)

    def test_unexpected_failures_are_not_translated(self):
        """Only profiler state conflicts become 409; real failures keep failing loudly."""
        with self.assertRaises(RuntimeError):
            self._call(http_server.stop_profile_async, _BrokenTokenizerManager())

    def test_profile_state_error_keeps_runtime_error_compatibility(self):
        """The engine-level API still raises something `except RuntimeError` catches."""
        error = ProfileStateError("Profiling is not in progress.")
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), "Profiling is not in progress.")


if __name__ == "__main__":
    unittest.main()
