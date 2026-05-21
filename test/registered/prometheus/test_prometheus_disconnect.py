"""
Regression test for issue #15686: Request.is_disconnected() broken by the
metrics middleware.

The metrics middleware added by `add_prometheus_track_response_middleware`
is a pure ASGI middleware. It must pass the ASGI `receive` callable through
to the downstream app untouched, so that a route handler calling
`await request.is_disconnected()` can still detect when the client has
dropped the connection.

This is a sibling of test/registered/scheduler/test_abort_with_metrics.py,
which covers _PureASGIDispatch. This test exercises the higher-level
MetricsMiddleware that wraps the FastAPI app once metrics are enabled.
"""

import asyncio
import unittest
from typing import List

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


_HTTP_SCOPE = {
    "type": "http",
    "asgi": {"version": "3.0"},
    "http_version": "1.1",
    "method": "GET",
    "path": "/probe",
    "raw_path": b"/probe",
    "query_string": b"",
    "root_path": "",
    "scheme": "http",
    "server": ("127.0.0.1", 0),
    "client": ("127.0.0.1", 0),
    "headers": [],
}


def _build_app_with_metrics_middleware():
    """Build a minimal FastAPI app and wire up the metrics middleware.

    The handler stashes the result of `await request.is_disconnected()` in a
    mutable cell so the test can inspect it after driving the ASGI app.
    """
    from fastapi import FastAPI, Request

    from sglang.srt.utils.common import add_prometheus_track_response_middleware

    app = FastAPI()
    cell: dict = {}

    @app.get("/probe")
    async def probe(request: Request):
        cell["disconnected"] = await request.is_disconnected()
        return {"ok": True}

    add_prometheus_track_response_middleware(app)
    return app, cell


async def _drive_once(app, scope, receive_msgs: List[dict]):
    """Drive an ASGI app exactly once and return the list of sent messages."""
    idx = 0
    sent: List[dict] = []

    async def receive():
        nonlocal idx
        if idx < len(receive_msgs):
            msg = receive_msgs[idx]
            idx += 1
            return msg
        return {"type": "http.disconnect"}

    async def send(msg):
        sent.append(msg)

    await app(scope, receive, send)
    return sent


class TestMetricsMiddlewareReceivePassthrough(CustomTestCase):
    """The metrics middleware must not break request.is_disconnected()."""

    @classmethod
    def setUpClass(cls):
        cls.app, cls.cell = _build_app_with_metrics_middleware()

    def setUp(self):
        self.cell.clear()

    def test_is_disconnected_when_client_drops(self):
        """receive() returning http.disconnect must propagate to the handler."""
        asyncio.run(_drive_once(self.app, _HTTP_SCOPE, [{"type": "http.disconnect"}]))
        self.assertEqual(self.cell.get("disconnected"), True)

    def test_not_disconnected_on_normal_request(self):
        """receive() returning http.request must not flip is_disconnected()."""
        asyncio.run(
            _drive_once(
                self.app,
                _HTTP_SCOPE,
                [{"type": "http.request", "body": b"", "more_body": False}],
            )
        )
        self.assertEqual(self.cell.get("disconnected"), False)


if __name__ == "__main__":
    unittest.main()
