"""Unit tests for the pure-ASGI HTTP middleware patch."""

import asyncio
import unittest

from sglang.srt.utils.http_middleware_patch import (
    _PureASGIDispatch,
    patch_app_http_middleware,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


_HTTP_SCOPE = {
    "type": "http",
    "asgi": {"version": "3.0"},
    "http_version": "1.1",
    "method": "GET",
    "scheme": "http",
    "path": "/test",
    "raw_path": b"/test",
    "query_string": b"",
    "root_path": "",
    "headers": [],
    "server": ("testserver", 80),
    "client": ("testclient", 50000),
}


class TestPureASGIDispatch(CustomTestCase):
    def test_non_http_scope_passes_through_without_dispatch(self):
        calls = []

        async def receive():
            return {"type": "websocket.disconnect"}

        async def send(message):
            calls.append(("send", message))

        async def inner_app(scope, inner_receive, inner_send):
            calls.append(("app", scope, inner_receive, inner_send))

        async def dispatch(request, call_next):
            calls.append(("dispatch", request, call_next))

        scope = {"type": "websocket"}
        middleware = _PureASGIDispatch(inner_app, dispatch)
        asyncio.run(middleware(scope, receive, send))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "app")
        self.assertIs(calls[0][1], scope)
        self.assertIs(calls[0][2], receive)
        self.assertIs(calls[0][3], send)

    def test_call_next_forwards_messages_and_returns_status(self):
        response_messages = [
            {"type": "http.response.start", "status": 201, "headers": []},
            {"type": "http.response.body", "body": b"created"},
        ]
        sent_messages = []
        observed = {}

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent_messages.append(message)

        async def inner_app(scope, inner_receive, inner_send):
            self.assertIs(scope, _HTTP_SCOPE)
            self.assertIs(inner_receive, receive)
            for message in response_messages:
                await inner_send(message)

        async def dispatch(request, call_next):
            response = await call_next(request)
            observed["request"] = request
            observed["status_code"] = response.status_code

        middleware = _PureASGIDispatch(inner_app, dispatch)
        asyncio.run(middleware(_HTTP_SCOPE, receive, send))

        self.assertIs(observed["request"]._receive, receive)
        self.assertEqual(observed["status_code"], 201)
        self.assertEqual(sent_messages, response_messages)

    def test_dispatch_exception_propagates(self):
        async def inner_app(scope, receive, send):
            self.fail("inner app should not run")

        async def dispatch(request, call_next):
            raise RuntimeError("middleware failed")

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            self.fail("send should not run")

        middleware = _PureASGIDispatch(inner_app, dispatch)
        with self.assertRaisesRegex(RuntimeError, "middleware failed"):
            asyncio.run(middleware(_HTTP_SCOPE, receive, send))


class TestPatchAppHttpMiddleware(CustomTestCase):
    def test_only_http_decorator_is_replaced(self):
        class FakeApp:
            def __init__(self):
                self.added_middleware = []
                self.original_calls = []

            def middleware(self, middleware_type):
                self.original_calls.append(middleware_type)
                return f"original:{middleware_type}"

            def add_middleware(self, middleware_class, **kwargs):
                self.added_middleware.append((middleware_class, kwargs))

        app = FakeApp()
        patch_app_http_middleware(app)

        def dispatch(request, call_next):
            pass

        returned_dispatch = app.middleware("http")(dispatch)

        self.assertIs(returned_dispatch, dispatch)
        self.assertEqual(
            app.added_middleware,
            [(_PureASGIDispatch, {"dispatch": dispatch})],
        )
        self.assertEqual(app.middleware("websocket"), "original:websocket")
        self.assertEqual(app.original_calls, ["websocket"])


if __name__ == "__main__":
    unittest.main()
