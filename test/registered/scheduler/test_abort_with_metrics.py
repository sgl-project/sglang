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
from starlette.responses import JSONResponse

from sglang.srt.utils.http_middleware_patch import _PureASGIDispatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="extra-test-cpu-intel")

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

    def test_dispatch_can_return_a_response_without_calling_next(self):
        async def run():
            async def dispatch(request, call_next):
                return JSONResponse(
                    {"detail": "retry later"},
                    status_code=429,
                    headers={"retry-after": "10"},
                )

            async def inner_app(scope, receive, send):
                self.fail("A short-circuited request must not reach the application")

            async def receive():
                return {"type": "http.request", "body": b""}

            sent = []

            async def send(message):
                sent.append(message)

            await _PureASGIDispatch(inner_app, dispatch)(_HTTP_SCOPE, receive, send)
            return sent

        sent = asyncio.run(run())
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0]["status"], 429)
        self.assertIn((b"retry-after", b"10"), sent[0]["headers"])
        self.assertEqual(sent[1]["body"], b'{"detail":"retry later"}')

    def test_returning_call_next_response_sends_it_once(self):
        async def run():
            async def dispatch(request, call_next):
                return await call_next(request)

            async def inner_app(scope, receive, send):
                await JSONResponse({"ok": True})(scope, receive, send)

            async def receive():
                return {"type": "http.request", "body": b""}

            sent = []

            async def send(message):
                sent.append(message)

            await _PureASGIDispatch(inner_app, dispatch)(_HTTP_SCOPE, receive, send)
            return sent

        sent = asyncio.run(run())
        self.assertEqual(
            [m["type"] for m in sent], ["http.response.start", "http.response.body"]
        )
        self.assertEqual(sent[1]["body"], b'{"ok":true}')

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
