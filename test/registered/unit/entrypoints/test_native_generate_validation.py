"""CPU admission tests for malformed native generation requests.

Exercise the real FastAPI routes, request normalization, tokenization and IPC
dispatch boundary. Only model-independent runtime services and the scheduler
response are replaced; these tests do not start a model or a GPU scheduler.
"""

import asyncio
import copy
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
from fastapi.testclient import TestClient

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import http_server  # noqa: E402
from sglang.srt.managers.embed_types import PositionalEmbeds  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    BatchTokenizedGenerateReqInput,
    GenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ServerStatus,
    TokenizerManager,
)
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.sampling.sampling_params import (  # noqa: E402
    TOP_K_ALL,
    SamplingParams,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _invalid_requests():
    # All ten crash-producing fields reported in #41466. Include empty grammar
    # containers, which must not disappear during default normalization.
    for value in (0, 1.5, True, "invalid", [], {}, [[1.0]]):
        yield "positional_embed_overrides", {"positional_embed_overrides": value}
    yield "lora_id", {"lora_id": ["adapter"]}
    for field, value in (
        ("max_new_tokens", 1.5),
        ("beam_width", 1.5),
        ("regex", ["a"]),
        ("regex", []),
        ("json_schema", {"type": "object"}),
        ("json_schema", {}),
        ("ebnf", ["root ::= 'a'"]),
        ("structural_tag", {"format": "json"}),
        ("skip_special_tokens", 1.5),
        ("top_k", 2**31),
    ):
        yield field, {"sampling_params": {field: value}}


class TestNativeGenerateValidation(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            enable_strict_thinking=False,
            enable_tokenizer_batch_encode=False,
            enable_dp_attention=False,
            language_only=False,
            language_model_only=False,
            speculative_algorithm=None,
        )
        server_args = override.install()
        self.addCleanup(override.restore)

        # Skip __init__, which starts processes and opens model-runtime sockets.
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.server_args = server_args
        manager.auto_create_handle_loop = Mock()
        manager.create_abort_task = Mock(return_value=None)
        manager.request_logger = Mock()
        manager.tokenizer = SimpleNamespace(
            is_fast=False, encode=lambda text: [1, 2, 3], eos_token_id=2
        )
        manager.async_dynamic_batch_tokenizer = None
        manager.mm_processor = None
        manager.model_config = SimpleNamespace(
            vocab_size=32,
            hidden_size=4,
            is_embedding_gemma=False,
            hf_config=SimpleNamespace(architectures=[]),
        )
        manager.preferred_sampling_params = None
        manager.sampling_params_class = SamplingParams
        manager.is_generation = True
        manager.context_len = 128
        manager.num_reserved_tokens = 0
        manager.allow_auto_truncate = False
        manager.validate_total_tokens = False
        manager.enable_lora = False
        manager.enable_trace = False
        manager.enable_priority_scheduling = False
        manager.is_pause = False
        manager.is_pause_cond = asyncio.Condition()
        manager.model_update_lock = SimpleNamespace(reader_lock=nullcontext())
        manager.rid_to_state = {}
        manager.encoder_dispatch_ready = {}
        manager.disaggregation_mode = "null"
        manager.tokenizer_ipc_name = None
        manager.send_to_scheduler = object()
        manager.cuda_vmm_feature_transport = SimpleNamespace(
            prepare_for_dispatch_async=AsyncMock(return_value=[]),
            cancel_for_dispatch=Mock(),
        )
        manager.gracefully_exit = False
        manager.server_status = ServerStatus.Up

        async def scheduler_response(obj, request=None):
            # Dispatch and all admission code remain real. Supply one result at
            # the response-wait boundary instead of running model inference.
            state = manager.rid_to_state.pop(obj.rid)
            self.assertTrue(state.dispatched)
            self.send.assert_called()
            yield {"text": "ok", "meta_info": {"id": obj.rid}}

        manager._wait_one_response = scheduler_response
        self.manager = manager
        transport = patch("sglang.srt.managers.tokenizer_manager.sock_send")
        self.send = transport.start()
        self.addCleanup(transport.stop)
        prior_state = http_server.get_global_state()
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
        self.addCleanup(http_server.set_global_state, prior_state)
        # Do not enter TestClient's context: the production lifespan launches
        # model processes. Requests still use the complete production ASGI app.
        self.client = TestClient(http_server.app)
        self.addCleanup(self.client.close)

    def _assert_http_rejected(self, payload, field, *, method="POST", stream=False):
        self.send.reset_mock()
        response = self.client.request(
            method, "/generate", json={"text": "Hello", "stream": stream, **payload}
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("application/json", response.headers["content-type"])
        self.assertNotIn("data: ", response.text)
        error = response.json()
        message = error.get("error", error)["message"]
        self.assertIn(field, message)
        self.send.assert_not_called()
        self.assertEqual(self.manager.rid_to_state, {})

    def _assert_http_healthy(self, *, method="POST", stream=False):
        self.assertEqual(self.client.get("/ready").status_code, 200)
        response = self.client.request(
            method,
            "/generate",
            json={
                "text": "Hello again",
                "stream": stream,
                "sampling_params": {"max_new_tokens": 4},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        if stream:
            self.assertIn("text/event-stream", response.headers["content-type"])
            self.assertIn('"text":"ok"', response.text)
            self.assertTrue(response.text.endswith("data: [DONE]\n\n"))
        else:
            self.assertEqual(response.json()["text"], "ok")
        self.send.assert_called_once()
        self.assertEqual(self.manager.rid_to_state, {})

    def test_reported_crashes_reject_before_sse_headers_and_scheduler_dispatch(self):
        for field, payload in _invalid_requests():
            for method in ("POST", "PUT"):
                for stream in (False, True):
                    with self.subTest(
                        field=field, payload=payload, method=method, stream=stream
                    ):
                        self._assert_http_rejected(
                            payload, field, method=method, stream=stream
                        )
                        self._assert_http_healthy(method=method, stream=stream)

    def test_invalid_later_batch_item_does_not_dispatch_the_valid_first_item(self):
        for field, value in (("regex", []), ("top_k", 2**31)):
            for stream in (False, True):
                with self.subTest(field=field, stream=stream):
                    self._assert_http_rejected(
                        {
                            "text": ["valid first", "invalid second"],
                            "sampling_params": [{"max_new_tokens": 4}, {field: value}],
                        },
                        field,
                        stream=stream,
                    )
                    self._assert_http_healthy(stream=stream)

    def test_session_types_reject_before_streaming(self):
        for field, value in (
            ("id", []),
            ("rid", {}),
            ("offset", 1.5),
            ("replace", []),
            ("drop_previous_output", {}),
        ):
            with self.subTest(field=field):
                self._assert_http_rejected(
                    {"session_params": {field: value}}, field, stream=True
                )
                self._assert_http_healthy(stream=True)

    def test_sampling_transport_fields_cannot_bypass_null_normalization(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self._assert_http_rejected(
                    {
                        "sampling_params": {
                            "is_normalized": True,
                            "skip_special_tokens": None,
                        }
                    },
                    "is_normalized",
                    stream=stream,
                )
                self._assert_http_healthy(stream=stream)

    def test_python_entrypoint_rejects_without_http_validation(self):
        async def generate(payload):
            obj = GenerateReqInput(text="Hello", **copy.deepcopy(payload))
            return await self.manager.generate_request(obj).__anext__()

        for field, payload in _invalid_requests():
            with self.subTest(field=field, payload=payload):
                self.send.reset_mock()
                with self.assertRaisesRegex(ValueError, field):
                    asyncio.run(generate(payload))
                self.send.assert_not_called()
                self.assertEqual(self.manager.rid_to_state, {})
                self.assertEqual(asyncio.run(generate({}))["text"], "ok")
                self.send.assert_called_once()

    def test_valid_batch_preserves_null_defaults_and_disable_sentinels(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.send.reset_mock()
                response = self.client.post(
                    "/generate",
                    json={
                        "input_ids": [[1, 2], [3, 4]],
                        "sampling_params": [
                            {
                                "max_new_tokens": None,
                                "top_k": -1,
                                "skip_special_tokens": None,
                                "regex": "",
                            },
                            {
                                "max_new_tokens": 4,
                                "top_k": None,
                                "n": None,
                                "skip_special_tokens": False,
                            },
                        ],
                        "lora_id": [None, None],
                        "positional_embed_overrides": [None, None],
                        "stream": stream,
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.send.assert_called_once()
                tokenized = self.send.call_args.args[1]
                self.assertIsInstance(tokenized, BatchTokenizedGenerateReqInput)
                first, second = tokenized.batch
                self.assertIsNone(first.sampling_params.max_new_tokens)
                self.assertEqual(first.sampling_params.top_k, TOP_K_ALL)
                self.assertTrue(first.sampling_params.skip_special_tokens)
                self.assertIsNone(first.sampling_params.regex)
                self.assertEqual(second.sampling_params.top_k, TOP_K_ALL)
                self.assertFalse(second.sampling_params.skip_special_tokens)
                self.assertEqual(self.manager.rid_to_state, {})

    def test_python_positional_embeddings_validate_model_dimension(self):
        async def generate(hidden_dim):
            obj = GenerateReqInput(
                input_ids=[1, 2],
                positional_embed_overrides=PositionalEmbeds(
                    embeds=torch.zeros(1, hidden_dim), positions=[0]
                ),
                sampling_params={"max_new_tokens": 4},
            )
            return await self.manager.generate_request(obj).__anext__()

        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            asyncio.run(generate(3))
        self.send.assert_not_called()
        self.assertEqual(self.manager.rid_to_state, {})
        self.assertEqual(asyncio.run(generate(4))["text"], "ok")
        self.send.assert_called_once()

    def test_singleton_lora_path_list_resolves_to_scalar_scheduler_id(self):
        self.manager.enable_lora = True
        self.manager.lora_registry = SimpleNamespace(
            get_unregistered_loras=AsyncMock(return_value=[]),
            acquire=AsyncMock(return_value="resolved-adapter-id"),
        )
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.send.reset_mock()
                self.manager.lora_registry.acquire.reset_mock()
                response = self.client.post(
                    "/generate",
                    json={
                        "text": "Hello",
                        "lora_path": ["adapter"],
                        "sampling_params": {"max_new_tokens": 4},
                        "stream": stream,
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.manager.lora_registry.acquire.assert_awaited_once_with("adapter")
                self.send.assert_called_once()
                self.assertEqual(
                    self.send.call_args.args[1].lora_id, "resolved-adapter-id"
                )
                self.assertEqual(self.manager.rid_to_state, {})


if __name__ == "__main__":
    unittest.main()
