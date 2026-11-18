"""Error and watchdog logs report sizes, not request content, unless the server
runs with --log-request-content-on-error. Pure CPU."""

import logging
import unittest
from array import array
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_override,
    maybe_stub_sgl_kernel,
)

maybe_stub_sgl_kernel()

from fastapi import FastAPI
from fastapi.testclient import TestClient
from xgrammar import TokenizerInfo

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.srt.entrypoints import http_server
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.runtime_context import get_context, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

CANARY = "canary7f3a9c"
# `uniqueItems` is a keyword the xgrammar backend refuses.
BAD_SCHEMA = '{"properties": {"%s": {"type": "array", "uniqueItems": true}}}' % CANARY
GOOD_SCHEMA = '{"properties": {"%s": {"type": "string"}}, "required": ["%s"]}' % (
    CANARY,
    CANARY,
)


def _backend() -> XGrammarGrammarBackend:
    vocab = [chr(i) for i in range(128)]
    info = TokenizerInfo(vocab, stop_token_ids=[0])
    return XGrammarGrammarBackend(
        SimpleNamespace(init_xgrammar=lambda: (info, None)), len(vocab)
    )


def _req_repr(test: CustomTestCase) -> str:
    req = Req(
        rid="rid-abc",
        origin_input_text="",
        origin_input_ids=array("q", [90001, 90002, 90003]),
        sampling_params=SamplingParams(max_new_tokens=8, stop=CANARY),
        vocab_size=100_000,
    )
    # The watchdog formats the batch's request list with an f-string `=`.
    return f"{[req]=}".replace("90001", CANARY)


def _compile_error_log(test: CustomTestCase) -> str:
    logger = "sglang.srt.constrained.xgrammar_backend"
    with test.assertLogs(logger, logging.ERROR) as logs:
        grammar = _backend().dispatch_json(BAD_SCHEMA)
    test.assertIn(CANARY, grammar.error_message)
    return "\n".join(logs.output)


def _rejected_token_error(test: CustomTestCase) -> str:
    grammar = _backend().dispatch_json(GOOD_SCHEMA)
    grammar.accept_token(ord("{"))
    with test.assertRaises(ValueError) as ctx:
        grammar.accept_token(ord("z"))
    return str(ctx.exception)


def _generate_route_log(test: CustomTestCase) -> str:
    class FailingTokenizerManager:
        async def generate_request(self, obj, request):
            raise ValueError(f"Failed to compile json grammar: {CANARY}")
            yield

    app = FastAPI()
    app.add_api_route("/generate", http_server.generate_request, methods=["POST"])
    prior_state = http_server.get_global_state()
    http_server.set_global_state(
        SimpleNamespace(tokenizer_manager=FailingTokenizerManager())
    )
    test.addCleanup(http_server.set_global_state, prior_state)
    with test.assertLogs("sglang.srt.entrypoints.http_server", logging.ERROR) as logs:
        response = TestClient(app).post("/generate", json={"text": "hi"})
    test.assertEqual(response.status_code, 400)
    test.assertIn(CANARY, response.json()["error"]["message"])
    return "\n".join(logs.output)


def _tool_argument_warning(test: CustomTestCase) -> str:
    logger = "sglang.srt.function_call.qwen3_coder_detector"
    with test.assertLogs(logger, logging.WARNING) as logs:
        Qwen3CoderDetector()._convert_param_value(
            param_value=CANARY,
            param_name="count",
            param_config={"count": {"type": "integer"}},
            func_name="set_count",
        )
    return "\n".join(logs.output)


SITES = (_req_repr, _compile_error_log, _generate_route_log, _tool_argument_warning)


class TestErrorLogRedaction(CustomTestCase):
    def _publish(self, **fields):
        enter_override(self, get_context().override_server_args(**fields))

    def test_content_is_redacted_by_default(self):
        self._publish()
        for site in SITES:
            with self.subTest(site=site.__name__):
                self.assertNotIn(CANARY, site(self))

    def test_flag_restores_content(self):
        self._publish(log_request_content_on_error=True)
        for site in SITES:
            with self.subTest(site=site.__name__):
                self.assertIn(CANARY, site(self))

    def test_redacted_req_repr_reports_sizes(self):
        self._publish()
        self.assertEqual(
            _req_repr(self),
            "[req]=[Req(rid=rid-abc, input_len=3, output_len=0, grammar=None)]",
        )

    def test_rejected_token_error_keeps_detail_for_the_client(self):
        # The scheduler copies this exception into the client's abort response.
        self._publish()
        self.assertIn(CANARY, _rejected_token_error(self))

    def test_process_without_published_config_redacts(self):
        reset_context()
        self.addCleanup(reset_context)
        self.assertNotIn(CANARY, _req_repr(self))


if __name__ == "__main__":
    unittest.main()
