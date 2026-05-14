"""
Unit test for _PureASGIDispatch: verify that the ASGI ``receive`` callable
is passed through untouched so that request.is_disconnected() works.

Background: @app.middleware("http") wraps handlers with BaseHTTPMiddleware
whose call_next() replaces the ASGI ``receive``, breaking
request.is_disconnected() and preventing non-streaming abort on client
disconnect.  _PureASGIDispatch fixes this.  The existing test_abort.py
already covers the full e2e abort flow.
"""

import asyncio
import unittest

from starlette.requests import Request

from sglang.srt.utils.http_middleware_patch import _PureASGIDispatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="stage-a-test-cpu")

_HTTP_SCOPE = {
    "type": "http",
    "asgi": {"version": "3.0"},
    "http_version": "1.1",
    "method": "POST",
    "path": "/test",
    "query_string": b"",
    "root_path": "",
    "headers": [],
}


class TestPureASGIDispatchReceivePassthrough(CustomTestCase):
    """Verify _PureASGIDispatch passes ``receive`` through untouched."""

    @staticmethod
    async def _run_with_receive(receive_msg):
        """Invoke _PureASGIDispatch and return request.is_disconnected()."""
        result = {}

        async def dispatch(request: Request, call_next):
            result["disconnected"] = await request.is_disconnected()
            await call_next(request)

        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = _PureASGIDispatch(inner_app, dispatch=dispatch)

        async def receive():
            return receive_msg

        sent = []

        async def send(msg):
            sent.append(msg)

        await middleware(_HTTP_SCOPE, receive, send)
        return result["disconnected"]

    def test_is_disconnected_on_client_disconnect(self):
        """receive() -> http.disconnect: is_disconnected() must return True."""
        self.assertTrue(
            asyncio.run(self._run_with_receive({"type": "http.disconnect"}))
        )

    def test_not_disconnected_when_connected(self):
        """receive() -> http.request: is_disconnected() must return False."""
        self.assertFalse(
            asyncio.run(self._run_with_receive({"type": "http.request", "body": b""}))
        )


if __name__ == "__main__":
    unittest.main()
