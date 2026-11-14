"""Priority metadata must survive the real HTTP routes and serving conversion.

Only generation is mocked: request parsing, chat rendering, conversion, response
serialization, and TokenizerManager's default/range validation are real.
"""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.background import BackgroundTask
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang.srt.configs.unlimited_ocr_v2 import UocrDocumentConfig
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.environ import envs
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.models.unlimited_ocr_v2_support.chat import ChatDocSplitter
from sglang.srt.runtime_context import (
    get_context,
    get_schedule,
    get_serving,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_PATHS = ("/v1/chat/completions", "/v1/completions")


class _RecordingEngine:
    def __init__(self, server_args):
        self.server_args = server_args
        self.requests = []
        self.received_priorities = []
        self.max_req_input_len = 65536
        self.rid_to_state = {}
        self.request_logger = SimpleNamespace(log_requests=False, log_requests_level=0)
        backend = Tokenizer(
            models.WordLevel({"[UNK]": 0, "hello": 1, "ok": 2}, "[UNK]")
        )
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            chat_template="{% for m in messages %}{{ m['content'] }}{% endfor %}",
        )
        self.model_config = SimpleNamespace(
            is_multimodal=False,
            hf_config=SimpleNamespace(
                model_type="llama",
                architectures=["LlamaForCausalLM"],
                sparse_attention_config={"page_token_id": 128827},
            ),
            get_default_sampling_params=lambda: {},
        )
        self.served_model_name = "test-model"
        self.model_path = "dummy"
        self.lora_registry = None

    def config_value(self, name):
        return get_context().config_leaf(name)

    @property
    def enable_priority_scheduling(self):
        return get_schedule().enable_priority_scheduling

    @property
    def default_priority_value(self):
        return get_schedule().default_priority_value

    async def generate_request(self, obj, request):
        self.received_priorities.append(obj.priority)
        TokenizerManager._set_default_priority(self, obj)
        self.requests.append(obj)
        yield {
            "text": "ok",
            "index": 0,
            "meta_info": {
                "id": "priority-test",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cached_tokens": 0,
                "weight_version": "default",
                "finish_reason": {"type": "stop", "matched": None},
            },
        }

    def create_abort_task(self, obj):
        return BackgroundTask(lambda: None)


class TestPriorityHeaders(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        override = get_context().override_server_args(
            enable_priority_scheduling=True, default_priority_value=37
        )
        server_args = override.install()
        self.addCleanup(override.restore)
        self.engine = _RecordingEngine(server_args)
        template_manager = SimpleNamespace(
            chat_template_name=None,
            jinja_template_content_format=None,
            completion_template_name=None,
            reasoning_config=None,
            force_reasoning=False,
            jinja_template_may_reorder_tool_results=False,
            python_chat_template=None,
        )
        app = FastAPI()
        # Use the production APIRoutes, including their JSON validation dependency,
        # without the model-launching lifespan of the production app.
        app.router.routes.extend(
            route
            for route in http_server.app.routes
            if getattr(route, "path", None) in _PATHS
        )
        app.state.openai_serving_chat = OpenAIServingChat(self.engine, template_manager)
        app.state.openai_serving_completion = OpenAIServingCompletion(
            self.engine, template_manager
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def _post(self, path, headers=None, **fields):
        body = {"model": "test-model", "max_tokens": 1, **fields}
        if path == _PATHS[0]:
            body["messages"] = [{"role": "user", "content": "hello"}]
        else:
            body["prompt"] = "hello"
        return self.client.post(path, json=body, headers=headers)

    def _assert_priority(self, path, expected, headers=None, **fields):
        before = len(self.engine.requests)
        response = self._post(path, headers, **fields)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.engine.requests), before + 1)
        self.assertEqual(self.engine.requests[-1].priority, expected)
        if fields.get("stream"):
            self.assertIn("[DONE]", response.text)
        else:
            choice = response.json()["choices"][0]
            text = choice.get("text", choice.get("message", {}).get("content"))
            self.assertEqual(text, "ok")

    def test_split_document_stream_resolves_priority_before_generation(self):
        serving = self.client.app.state.openai_serving_chat
        splitter = ChatDocSplitter(
            serving,
            handle_window=serving.handle_request,
            config=UocrDocumentConfig(),
        )
        with (
            patch.object(serving, "_doc_splitter", splitter),
            patch.object(splitter, "_plan_doc_split", return_value=[(0, 1, 0)]),
        ):
            self._assert_priority(_PATHS[0], -5, {"X-Priority": "-5"}, stream=True)
            self._assert_priority(_PATHS[0], 20, {"Priority": "u=2"}, stream=True)
            self._assert_priority(
                _PATHS[0], 0, {"X-Priority": "bad"}, stream=True, priority=0
            )
            response = self._post(_PATHS[0], {"X-Priority": "bad"}, stream=True)
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("X-Priority must be a single integer", response.text)
            self.assertEqual(self.engine.received_priorities, [-5, 20, 0])

    def test_body_and_header_precedence(self):
        """Zero and explicit null cannot be mistaken for an omitted body field."""
        cases = [
            ({"priority": 0}, {"X-Priority": "9", "Priority": "u=7"}, 0),
            ({"priority": -7}, {"X-Priority": "invalid"}, -7),
            ({"priority": None}, {"X-Priority": "invalid"}, 37),
            ({}, {"X-Priority": "0", "Priority": "u=7"}, 0),
            ({}, {"x-PrIoRiTy": " -12 \t", "Priority": "invalid"}, -12),
            ({}, {"X-Priority": "+12"}, 12),
            ({}, {"pRiOrItY": "u=0, i"}, 0),
            ({}, {"Priority": "u=7"}, 70),
            ({}, {}, 37),
        ]
        for path in _PATHS:
            for fields, headers, expected in cases:
                with self.subTest(path=path, fields=fields, headers=headers):
                    self._assert_priority(path, expected, headers, **fields)

    def test_structured_dictionary_extensibility_and_duplicates(self):
        """RFC8941 dictionaries combine field lines and keep the last member."""
        cases = [
            ({"Priority": 'x="a,b", i, u=2;ext=?1;label="a,b"'}, 20),
            ({"Priority": 'u=4, x=(1;v=foo "text" ?0);p=2, bin=:aGVsbG8=:'}, 40),
            ({"Priority": "u=1, unknown=3.5, token=abc:def/*, u=6"}, 60),
            ([("Priority", "u=1;foo=bar"), ("Priority", "u=5, i")], 50),
        ]
        for path in _PATHS:
            for headers, expected in cases:
                with self.subTest(path=path, headers=headers):
                    self._assert_priority(path, expected, headers)

    def test_invalid_rfc_urgency_is_ignored(self):
        """Invalid RFC dictionaries/types/ranges fall back without failing requests."""
        invalid = [
            "u=?1",
            "u=?0",
            'u="0"',
            "u=1.0",
            "u=low",
            "u",
            "u=(0)",
            "u=-1",
            "u=8",
            "u=9999999999999999",
            "u=0, u=?1",
            "i",
            "",
            "u =1",
            "u=1;",
            "u=1,",
            "u=1 trailing",
            'u=1, x="unterminated',
            "u=1, =bad",
            "u=1, x=(1",
            "u=1, x=@",
            "u=1;bad=",
            "u=1, x=:!:",
        ]
        for path in _PATHS:
            for value in invalid:
                with self.subTest(path=path, value=value):
                    self._assert_priority(path, 37, {"Priority": value})
                    self.assertIsNone(self.engine.received_priorities[-1])
            self._assert_priority(path, 37, [(b"priority", b"u=1, x=\xff")])

    def test_invalid_or_duplicate_x_priority_is_400(self):
        invalid = ["high", "", "1.0", "1_000", "1,2", "1 2", "?1", '"1"', "++1"]
        headers = [{"X-Priority": value, "Priority": "u=0"} for value in invalid]
        headers.append([("X-Priority", "1"), ("x-priority", "1")])
        headers.append({"X-Priority": "1" * 5000})
        for path in _PATHS:
            for header in headers:
                with self.subTest(path=path, headers=header):
                    before = len(self.engine.received_priorities)
                    response = self._post(path, header)
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertIn("X-Priority", response.json()["message"])
                    self.assertEqual(len(self.engine.received_priorities), before)

    def test_header_flag_and_priority_mode_matrix(self):
        """Explicit enable never enables priority scheduling itself; FCFS is unchanged."""
        for mode in (False, True):
            for allow in (None, False, True):
                with get_context().override_server_args(
                    enable_priority_scheduling=mode,
                    allow_header_priority=allow,
                    default_priority_value=37,
                    abort_on_priority_when_disabled=True,
                ):
                    expected = (
                        0 if mode and allow is not False else (37 if mode else None)
                    )
                    for path in _PATHS:
                        for headers in ({"X-Priority": "0"}, {"Priority": "u=0"}):
                            with self.subTest(mode=mode, allow=allow, path=path):
                                self._assert_priority(path, expected, headers)
                        if not mode or allow is False:
                            self._assert_priority(path, expected, {"X-Priority": "bad"})
                        self._assert_priority(
                            path, -3, {"X-Priority": "bad"}, priority=-3
                        )

    def test_unconfigured_default_remains_none(self):
        with get_schedule().override(default_priority_value=None):
            for path in _PATHS:
                self._assert_priority(path, None)
                self._assert_priority(path, None, {"Priority": "i"})

    def test_native_priorities_keep_signed_64_bit_validation(self):
        for path in _PATHS:
            for priority in (-(1 << 63), (1 << 63) - 1):
                self._assert_priority(path, priority, {"X-Priority": str(priority)})
            for priority in (-(1 << 63) - 1, 1 << 63):
                with self.subTest(path=path, priority=priority):
                    response = self._post(path, {"X-Priority": str(priority)})
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertIn("signed 64-bit", response.json()["message"])

    def test_scheduler_direction_does_not_remap_rfc_values(self):
        for low_first in (False, True):
            with get_schedule().override(schedule_low_priority_values_first=low_first):
                for path in _PATHS:
                    self._assert_priority(path, 20, {"Priority": "u=2"})

    def test_streaming_uses_the_same_priority_resolution(self):
        for path in _PATHS:
            self._assert_priority(path, 0, {"X-Priority": "0"}, stream=True)
            self._assert_priority(path, 70, {"Priority": "u=7"}, stream=True)

    def test_legacy_override_retains_chat_only_opt_in_and_body_override(self):
        for enabled in (False, True):
            with envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.override(enabled):
                for path in _PATHS:
                    expected = 8 if enabled and path == _PATHS[0] else 0
                    self._assert_priority(
                        path,
                        expected,
                        {
                            "x-override-priority": "8",
                            "X-Priority": "2",
                            "Priority": "u=7",
                        },
                        priority=0,
                    )
                if enabled:
                    # A legacy override also wins over new invalid headers and
                    # is not controlled by the new header flag.
                    for allow in (None, False):
                        with get_serving().override(allow_header_priority=allow):
                            self._assert_priority(
                                _PATHS[0],
                                -3,
                                {"x-override-priority": "-3", "X-Priority": "bad"},
                            )
                    response = self._post(
                        _PATHS[0], {"x-override-priority": "bad", "X-Priority": "0"}
                    )
                    self.assertEqual(response.status_code, 400, response.text)
                else:
                    for path in _PATHS:
                        self._assert_priority(
                            path, 2, {"x-override-priority": "8", "X-Priority": "2"}
                        )

    def test_cli_tristate_projects_into_runtime_bag(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        for flags, expected in (
            ([], None),
            (["--allow-header-priority"], True),
            (["--no-allow-header-priority"], False),
        ):
            parsed = parser.parse_args(["--model-path", "dummy", *flags])
            self.assertIs(parsed.allow_header_priority, expected)
            self.assertIs(
                ServerArgs.from_cli_args(parsed).allow_header_priority, expected
            )
            with get_context().override_server_args(
                allow_header_priority=parsed.allow_header_priority
            ):
                self.assertIs(get_serving().allow_header_priority, expected)


if __name__ == "__main__":
    unittest.main()
