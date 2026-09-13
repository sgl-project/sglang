"""Endpoint-level tests for rid propagation on the native `/generate` route.

`/generate` builds its `GenerateReqInput` straight from the body, so the
`x-request-id` header has to be applied by the handler itself rather than by the
OpenAI serving layer. These tests exercise the handler against a stub global
state: it is an `async def` reading module-level `_global_state`, so awaiting the
coroutine directly is enough, with no model server booted.
"""

import asyncio
import unittest
from types import SimpleNamespace

from starlette.datastructures import Headers

from sglang.srt.entrypoints import http_server
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _StubTokenizerManager:
    def __init__(self):
        self.received = []

    def generate_request(self, obj, request=None):
        self.received.append(obj)
        return self._one_response()

    async def _one_response(self):
        yield {"text": "ok", "meta_info": {"id": "stub"}}


def _dispatch(headers: dict, body_rid=None) -> GenerateReqInput:
    """Run the `/generate` handler and return the object it sent downstream."""
    obj = GenerateReqInput(text="hi", rid=body_rid)
    request = SimpleNamespace(headers=Headers(headers))
    tokenizer_manager = _StubTokenizerManager()

    prior_state = http_server.get_global_state()
    http_server.set_global_state(
        SimpleNamespace(tokenizer_manager=tokenizer_manager, scheduler_info={})
    )
    try:
        asyncio.run(http_server.generate_request(obj, request))
    finally:
        http_server._global_state = prior_state

    return tokenizer_manager.received[0]


class TestGenerateRequestIdHeader(CustomTestCase):
    def test_header_rid_reaches_the_tokenizer_manager(self):
        """`/generate` honors x-request-id so PD workers derive the same rid.

        The router dispatches one request to a prefill and a decode worker; each
        normalizes independently, so without a shared header rid the two sides
        generate unrelated ids and cannot be correlated.
        """
        sent = _dispatch({"x-request-id": "header-rid"}, body_rid="body-rid")
        self.assertEqual(sent.rid, "header-rid")

    def test_body_rid_survives_without_the_header(self):
        """A caller that sets no header keeps the rid it put in the body."""
        sent = _dispatch({}, body_rid="body-rid")
        self.assertEqual(sent.rid, "body-rid")

    def test_override_header_wins_over_the_request_id(self):
        """x-override-rid is the explicit, gated mechanism, so it is applied last.

        A trusted upstream that names an exact rid must not be overruled by the
        correlation id a client happened to attach.
        """
        with envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.override(True):
            sent = _dispatch(
                {"x-request-id": "header-rid", "x-override-rid": "override-rid"}
            )
        self.assertEqual(sent.rid, "override-rid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
