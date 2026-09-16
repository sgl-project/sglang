"""
Unit-tests for the refactored completions-serving handler (no pytest).
Run with:
    python -m unittest discover -s test/registered/unit/entrypoints/openai -p test_serving_completions.py -v
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import json
import unittest
from http import HTTPStatus
from itertools import product
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, Mock

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _spec_result(index):
    return {
        "text": f"choice-{index}",
        "meta_info": {
            "id": "cmpl-spec-test",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cached_tokens": 0,
            "finish_reason": {"type": "stop"},
            "weight_version": "default",
            "spec_accept_rate": 0.5,
            "spec_accept_length": 2.0,
            "spec_cap_length": index + 1.0,
            "spec_block_accept_length": index + 0.5,
            "spec_num_correct_drafts": 1,
            "spec_num_proposed_drafts": 2,
            "spec_verify_ct": 1,
            "spec_correct_drafts_histogram": [0, 1],
            "spec_cap_lens_histogram": [index, 1],
        },
        "index": index,
    }


class _MockTemplateManager:
    """Minimal mock for TemplateManager."""

    def __init__(self):
        self.chat_template_name: Optional[str] = None
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = (
            None  # Set to None to avoid template processing
        )
        self.jinja_template_may_reorder_tool_results = False


class ServingCompletionTestCase(unittest.TestCase):
    """Bundle all prompt/echo tests in one TestCase."""

    # ---------- shared test fixtures ----------
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        # Serving only needs this interface; importing the real tokenizer manager
        # also imports GPU schedulers and model executors unrelated to these tests.
        tm = SimpleNamespace()

        tm.tokenizer = Mock()
        tm.tokenizer.encode.return_value = [1, 2, 3, 4]
        tm.tokenizer.decode.return_value = "decoded text"
        tm.tokenizer.bos_token_id = 1

        tm.model_config = Mock(is_multimodal=False)
        tm.server_args = Mock(enable_cache_report=False)

        tm.generate_request = AsyncMock()
        tm.create_abort_task = Mock()

        self.template_manager = _MockTemplateManager()
        self.sc = OpenAIServingCompletion(tm, self.template_manager)
        self.fastapi_request = Mock(spec=Request)

    # ---------- prompt-handling ----------
    def test_single_token_ids_prompt(self):
        req = CompletionRequest(model="x", prompt=[1, 2, 3, 4], max_tokens=100)
        internal, _ = self.sc._convert_to_internal_request(req)
        self.assertEqual(internal.input_ids, [1, 2, 3, 4])

    def test_cache_salt_and_extra_key_remain_distinct(self):
        req = CompletionRequest(
            model="x",
            prompt=[1, 2, 3, 4],
            max_tokens=1,
            cache_salt="tenant-a",
            extra_key="classification",
        )
        internal, _ = self.sc._convert_to_internal_request(req)
        self.assertEqual(internal.cache_salt, "tenant-a")
        self.assertEqual(internal.extra_key, "classification")

    def test_single_request_rejects_batched_cache_salt(self):
        req = CompletionRequest(
            model="x",
            prompt=[1, 2, 3, 4],
            max_tokens=1,
            cache_salt=["tenant-a"],
        )
        internal, _ = self.sc._convert_to_internal_request(req)
        with self.assertRaisesRegex(ValueError, "single request"):
            internal.normalize_batch_and_arguments()

    # ---------- echo-handling ----------
    def test_echo_with_list_of_strings_streaming(self):
        req = CompletionRequest(
            model="x", prompt=["A", "B"], max_tokens=1, echo=True, n=1
        )
        self.assertEqual(self.sc._get_echo_text(req, 0), "A")
        self.assertEqual(self.sc._get_echo_text(req, 1), "B")

    def test_echo_with_token_ids_streaming(self):
        req = CompletionRequest(model="x", prompt=[1, 2, 3], max_tokens=1, echo=True)
        self.sc.tokenizer_manager.tokenizer.decode.return_value = "decoded_prompt"
        self.assertEqual(self.sc._get_echo_text(req, 0), "decoded_prompt")

    def test_echo_with_multiple_token_ids_streaming(self):
        req = CompletionRequest(
            model="x", prompt=[[1, 2], [3, 4]], max_tokens=1, echo=True, n=1
        )
        self.sc.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        self.assertEqual(self.sc._get_echo_text(req, 0), "decoded")

    def test_prepare_echo_prompts_non_streaming(self):
        # single string
        req = CompletionRequest(model="x", prompt="Hi", echo=True)
        self.assertEqual(self.sc._prepare_echo_prompts(req), ["Hi"])

        # list of strings
        req = CompletionRequest(model="x", prompt=["Hi", "Yo"], echo=True)
        self.assertEqual(self.sc._prepare_echo_prompts(req), ["Hi", "Yo"])

        # token IDs
        req = CompletionRequest(model="x", prompt=[1, 2, 3], echo=True)
        self.sc.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        self.assertEqual(self.sc._prepare_echo_prompts(req), ["decoded"])

    # ---------- response_format handling ----------
    def test_response_format_json_object(self):
        """Test that response_format json_object is correctly processed in sampling params."""
        req = CompletionRequest(
            model="x",
            prompt="Generate a JSON object:",
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        sampling_params = self.sc._build_sampling_params(req)
        self.assertEqual(sampling_params["json_schema"], '{"type": "object"}')

    def test_response_format_json_schema(self):
        """Test that response_format json_schema is correctly processed in sampling params."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        req = CompletionRequest(
            model="x",
            prompt="Generate a JSON object:",
            max_tokens=100,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": schema},
            },
        )
        sampling_params = self.sc._build_sampling_params(req)
        # The schema should be converted to string by convert_json_schema_to_str
        self.assertIn("json_schema", sampling_params)
        self.assertIsInstance(sampling_params["json_schema"], str)

    def test_response_format_json_schema_missing_schema(self):
        """Test that json_schema response_format without a schema raises a ValueError."""
        req = CompletionRequest(
            model="x",
            prompt="Generate a JSON object:",
            max_tokens=100,
            response_format={"type": "json_schema"},
        )
        with self.assertRaises(ValueError):
            self.sc._build_sampling_params(req)

    def test_response_format_structural_tag(self):
        """Test that response_format structural_tag is correctly processed in sampling params."""
        req = CompletionRequest(
            model="x",
            prompt="Generate structured output:",
            max_tokens=100,
            response_format={
                "type": "structural_tag",
                "structures": [{"begin": "<data>", "end": "</data>"}],
                "triggers": ["<data>"],
            },
        )
        sampling_params = self.sc._build_sampling_params(req)
        # The structural_tag should be processed
        self.assertIn("structural_tag", sampling_params)
        self.assertIsInstance(sampling_params["structural_tag"], str)

    def test_response_format_none(self):
        """Test that no response_format doesn't add extra constraints."""
        req = CompletionRequest(model="x", prompt="Generate text:", max_tokens=100)
        sampling_params = self.sc._build_sampling_params(req)
        # Should not have json_schema or structural_tag from response_format
        # (but might have json_schema from the legacy json_schema field)
        self.assertIsNone(sampling_params.get("structural_tag"))

    def test_non_streaming_token_id_logprobs(self):
        for option in (None, False, True):
            with self.subTest(option=option):
                req = CompletionRequest(
                    model="x",
                    prompt=[1],
                    echo=True,
                    logprobs=2,
                    return_token_ids=True,
                    return_tokens_as_token_ids=option,
                )
                ret = [
                    {
                        "text": " world",
                        "output_ids": [2],
                        "prompt_token_ids": [1],
                        "meta_info": {
                            "id": "test-id",
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "finish_reason": {"type": "stop"},
                            "weight_version": "v1",
                            "input_token_logprobs": [(None, 1, "hello")],
                            "input_top_logprobs": [None],
                            "output_token_logprobs": [(-0.1, 2, " world")],
                            "output_top_logprobs": [
                                [(-0.1, 2, " world"), (-0.2, 3, " world")]
                            ],
                        },
                    }
                ]
                response = self.sc._build_completion_response(req, ret, 1234567890)
                choice = response.choices[0]
                self.assertEqual(
                    choice.logprobs.tokens,
                    ["token_id:1", "token_id:2"] if option else ["hello", " world"],
                )
                if option:
                    self.assertEqual(
                        choice.logprobs.top_logprobs[-1],
                        {"token_id:2": -0.1, "token_id:3": -0.2},
                    )
                self.assertEqual(choice.token_ids, [2])
                self.assertEqual(choice.prompt_token_ids, [1])

    def test_streaming_token_id_logprobs_with_echo(self):
        async def mock_generate(*args, **kwargs):
            for index, text in enumerate([" world", " world!"]):
                yield {
                    "text": text,
                    "output_ids": [2, 4][: index + 1],
                    "prompt_token_ids": [1],
                    "meta_info": {
                        "id": "test-stream",
                        "prompt_tokens": 1,
                        "completion_tokens": index + 1,
                        "output_token_logprobs_length": index + 1,
                        "finish_reason": {"type": "stop"} if index else None,
                        "input_token_logprobs": [(None, 1, "hello")],
                        "input_top_logprobs": [None],
                        "output_token_logprobs": [(-0.1, 2, " world"), (-0.3, 4, "!")][
                            : index + 1
                        ],
                        "output_top_logprobs": [
                            [(-0.1, 2, " world"), (-0.2, 3, " world")],
                            [(-0.3, 4, "!")],
                        ][: index + 1],
                    },
                }

        self.sc.tokenizer_manager.generate_request = mock_generate
        req = CompletionRequest(
            model="x",
            prompt=[1],
            echo=True,
            stream=True,
            logprobs=2,
            return_token_ids=True,
            return_tokens_as_token_ids=True,
        )
        adapted, _ = self.sc._convert_to_internal_request(req)

        async def collect():
            return [
                chunk
                async for chunk in self.sc._generate_completion_stream(
                    adapted, req, self.fastapi_request
                )
            ]

        chunks = get_or_create_event_loop().run_until_complete(collect())
        choices = [
            json.loads(chunk[len("data: ") :])["choices"][0]
            for chunk in chunks
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]"
        ]
        self.assertEqual(
            [choice["logprobs"]["tokens"] for choice in choices],
            [["token_id:1", "token_id:2"], ["token_id:4"]],
        )
        self.assertEqual(
            choices[0]["logprobs"]["top_logprobs"][-1],
            {"token_id:2": -0.1, "token_id:3": -0.2},
        )
        self.assertEqual([choice["token_ids"] for choice in choices], [[2], [4]])
        self.assertEqual(choices[0]["prompt_token_ids"], [1])

    @staticmethod
    def _token_logprob_result(index, step=1, incremental=False, finished=False):
        prompt_id = 11 if index < 2 else 22
        token_ids = [index * 100, index * 100 + 1]
        rows = [(-0.1, token_ids[0], "a"), (-0.2, token_ids[1], "b")]
        top_rows = [
            [rows[0], (-0.3, token_ids[0] + 10, "a")],
            [rows[1]],
        ]
        selection = slice(step, step + 1) if incremental else slice(0, step + 1)
        if finished and incremental:
            selection = slice(0, 0)
        return {
            "index": index,
            "text": "".join(row[2] for row in rows[selection]),
            "output_ids": token_ids[selection],
            "prompt_token_ids": [prompt_id],
            "meta_info": {
                "id": "cmpl-token-logprobs",
                "prompt_tokens": 1,
                "completion_tokens": step + 1,
                "cached_tokens": 0,
                "weight_version": "v1",
                "finish_reason": {"type": "stop"} if finished else None,
                "input_token_logprobs": [(None, prompt_id, "prompt")],
                "input_top_logprobs": [None],
                "output_token_logprobs": rows[selection],
                "output_top_logprobs": top_rows[selection],
                "output_token_logprobs_length": step + 1,
            },
        }

    def test_non_streaming_logprob_options_for_batched_parallel_sampling(self):
        for flag, echo, logprobs, return_ids in product(
            (None, False, True), (False, True), (None, 0, 2), (False, True)
        ):
            with self.subTest(
                flag=flag, echo=echo, logprobs=logprobs, return_ids=return_ids
            ):
                request = CompletionRequest(
                    model="x",
                    prompt=[[11], [22]],
                    n=2,
                    echo=echo,
                    logprobs=logprobs,
                    return_tokens_as_token_ids=flag,
                    return_token_ids=return_ids,
                )
                adapted, _ = self.sc._convert_to_internal_request(request)
                self.assertEqual(adapted.input_ids, [[11], [22]])
                self.assertEqual(adapted.sampling_params["n"], 2)
                self.assertEqual(adapted.return_logprob, logprobs is not None)
                self.assertEqual(adapted.return_prompt_token_ids, return_ids)
                results = [
                    self._token_logprob_result(i, finished=True) for i in range(4)
                ]
                if logprobs is None:
                    for result in results:
                        for key in list(result["meta_info"]):
                            if "logprob" in key:
                                del result["meta_info"][key]
                elif logprobs == 0:
                    for result in results:
                        result["meta_info"]["output_top_logprobs"] = []
                response = self.sc._build_completion_response(
                    request, results, 1234567890
                )
                self.assertEqual(len(response.choices), 4)
                for index, choice in enumerate(response.choices):
                    prompt_id = 11 if index < 2 else 22
                    ids = [index * 100, index * 100 + 1]
                    self.assertEqual(choice.index, index)
                    self.assertEqual(
                        choice.text, ("decoded text" if echo else "") + "ab"
                    )
                    if logprobs is None:
                        self.assertIsNone(choice.logprobs)
                    else:
                        expected = (
                            [f"token_id:{token}" for token in ids]
                            if flag
                            else ["a", "b"]
                        )
                        if echo:
                            expected.insert(
                                0, f"token_id:{prompt_id}" if flag else "prompt"
                            )
                        self.assertEqual(choice.logprobs.tokens, expected)
                        self.assertEqual(
                            choice.logprobs.token_logprobs,
                            ([None] if echo else []) + [-0.1, -0.2],
                        )
                        if logprobs == 2 and flag:
                            self.assertEqual(
                                choice.logprobs.top_logprobs[-2],
                                {
                                    f"token_id:{ids[0]}": -0.1,
                                    f"token_id:{ids[0] + 10}": -0.3,
                                },
                            )
                    encoded = choice.model_dump()
                    if return_ids:
                        self.assertEqual(encoded["token_ids"], ids)
                        self.assertEqual(encoded["prompt_token_ids"], [prompt_id])
                    else:
                        self.assertNotIn("token_ids", encoded)
                        self.assertNotIn("prompt_token_ids", encoded)

    def test_streaming_logprob_options_keep_interleaved_choices_independent(self):
        for incremental, flag, echo, return_ids, logprobs in product(
            (False, True), (None, False, True), (False, True), (False, True), (0, 2)
        ):
            with (
                self.subTest(
                    incremental=incremental,
                    flag=flag,
                    echo=echo,
                    return_ids=return_ids,
                    logprobs=logprobs,
                ),
                get_context().override_server_args(
                    incremental_streaming_output=incremental,
                    stream_response_default_include_usage=True,
                ),
            ):
                request = CompletionRequest(
                    model="x",
                    prompt=[[11], [22]],
                    n=2,
                    stream=True,
                    echo=echo,
                    logprobs=logprobs,
                    return_tokens_as_token_ids=flag,
                    return_token_ids=return_ids,
                )
                adapted, _ = self.sc._convert_to_internal_request(request)

                async def generate(*args, **kwargs):
                    for step in (0, 1):
                        for index in (2, 0, 3, 1):
                            result = self._token_logprob_result(
                                index, step, incremental
                            )
                            if logprobs == 0:
                                result["meta_info"]["output_top_logprobs"] = []
                            yield result
                    for index in (3, 1, 2, 0):
                        yield self._token_logprob_result(
                            index, 1, incremental, finished=True
                        )

                self.sc.tokenizer_manager.generate_request = generate

                async def collect():
                    return [
                        chunk
                        async for chunk in self.sc._generate_completion_stream(
                            adapted, request, self.fastapi_request
                        )
                    ]

                chunks = get_or_create_event_loop().run_until_complete(collect())
                self.assertEqual(chunks[-1], "data: [DONE]\n\n")
                packets = [json.loads(chunk[len("data: ") :]) for chunk in chunks[:-1]]
                for packet in packets:
                    self.assertNotIn("error", packet)
                self.assertEqual(packets[-1]["choices"], [])
                self.assertEqual(packets[-1]["usage"]["completion_tokens"], 8)
                for index in range(4):
                    choices = [
                        choice
                        for packet in packets
                        for choice in packet["choices"]
                        if choice["index"] == index
                    ]
                    self.assertEqual(len(choices), 3)
                    prompt_id = 11 if index < 2 else 22
                    ids = [index * 100, index * 100 + 1]
                    expected_first = [f"token_id:{ids[0]}" if flag else "a"]
                    if echo:
                        expected_first.insert(
                            0, f"token_id:{prompt_id}" if flag else "prompt"
                        )
                    self.assertEqual(choices[0]["logprobs"]["tokens"], expected_first)
                    self.assertEqual(
                        choices[1]["logprobs"]["tokens"],
                        [f"token_id:{ids[1]}" if flag else "b"],
                    )
                    self.assertIsNone(choices[2]["logprobs"])
                    self.assertEqual(
                        [choice["text"] for choice in choices],
                        [("decoded text" if echo else "") + "a", "b", ""],
                    )
                    self.assertEqual(choices[2]["finish_reason"], "stop")
                    if flag and logprobs == 2:
                        self.assertEqual(
                            choices[0]["logprobs"]["top_logprobs"][-1],
                            {
                                f"token_id:{ids[0]}": -0.1,
                                f"token_id:{ids[0] + 10}": -0.3,
                            },
                        )
                    if return_ids:
                        self.assertEqual(
                            [choice["token_ids"] for choice in choices],
                            [[ids[0]], [ids[1]], []],
                        )
                        self.assertEqual(choices[0]["prompt_token_ids"], [prompt_id])
                        self.assertNotIn("prompt_token_ids", choices[1])
                    else:
                        for choice in choices:
                            self.assertNotIn("token_ids", choice)
                            self.assertNotIn("prompt_token_ids", choice)

    def test_token_id_logprob_flag_without_logprobs_does_not_require_metadata(self):
        request = CompletionRequest(
            prompt=[11], stream=True, return_tokens_as_token_ids=True
        )
        adapted, _ = self.sc._convert_to_internal_request(request)
        self.assertFalse(adapted.return_logprob)

        async def generate(*args, **kwargs):
            yield {
                "text": "a",
                "meta_info": {
                    "id": "test-no-logprobs",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "finish_reason": {"type": "stop"},
                },
            }

        self.sc.tokenizer_manager.generate_request = generate

        async def collect():
            return [
                chunk
                async for chunk in self.sc._generate_completion_stream(
                    adapted, request, self.fastapi_request
                )
            ]

        chunks = get_or_create_event_loop().run_until_complete(collect())
        packet = json.loads(chunks[0][len("data: ") :])
        self.assertEqual(packet["choices"][0]["text"], "a")
        self.assertIsNone(packet["choices"][0]["logprobs"])
        self.assertNotIn("token_ids", packet["choices"][0])

    def test_non_streaming_response(self):
        req = CompletionRequest(
            model="x",
            prompt="Hello",
            max_tokens=10,
            logprobs=False,
            return_token_ids=True,
        )

        mock_ret = [
            {
                "text": " world",
                "output_ids": [3, 4],
                "prompt_token_ids": [1, 2],
                "meta_info": {
                    "id": "test-id",
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "finish_reason": {"type": "stop"},
                    "weight_version": "v1",
                },
            }
        ]

        response = self.sc._build_completion_response(req, mock_ret, 1234567890)

        self.assertEqual(len(response.choices), 1)
        self.assertEqual(response.choices[0].text, " world")
        self.assertEqual(len(response.choices[0].logprobs.top_logprobs), 0)
        self.assertEqual(response.choices[0].token_ids, [3, 4])
        self.assertEqual(response.choices[0].prompt_token_ids, [1, 2])

    def test_streaming_abort_yields_error(self):
        """Test that an abort finish reason during streaming correctly yields an error and stops."""
        err_msg = "Aborted by scheduler"
        err_code = HTTPStatus.INTERNAL_SERVER_ERROR

        async def _mock_generate_abort(*args, **kwargs):
            yield {
                "text": "Partial ",
                "meta_info": {
                    "id": "cmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {
                        "type": "abort",
                        "status_code": err_code,
                        "message": err_msg,
                    },
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.sc.tokenizer_manager.generate_request = _mock_generate_abort

        req = CompletionRequest(
            model="x",
            prompt="Hello world",
            max_tokens=100,
            stream=True,
        )

        adapted_request, _ = self.sc._convert_to_internal_request(req)

        async def run_stream():
            chunks = []
            try:
                async for chunk in self.sc._generate_completion_stream(
                    adapted_request, req, self.fastapi_request
                ):
                    chunks.append(chunk)
            except Exception as e:
                print(f"Error during stream iteration: {e}")
            return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        error_chunk_data = None
        for c in chunks:
            if "error" in c:
                error_chunk_data = json.loads(c[len("data: ") :])
                break
        self.assertIsNotNone(error_chunk_data, "Error chunk not found in stream")
        self.assertEqual(error_chunk_data["error"]["message"], err_msg)
        self.assertEqual(error_chunk_data["error"]["code"], err_code.value)

        # Ensure the stream stops after the abort error
        # The last chunk should be "data: [DONE]\n\n"
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

        # Check that there is an error chunk and a DONE chunk, and possibly a role chunk
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("error", chunks[0])

    def test_streaming_token_ids_deltas_cover_output_exactly(self):
        req = CompletionRequest(
            model="x",
            prompt="Hi",
            max_tokens=10,
            stream=True,
            return_token_ids=True,
        )
        adapted_request, _ = self.sc._convert_to_internal_request(req)

        for incremental in (False, True):
            # Both of these are read through `get_serving()` now, so assigning
            # them on the mock manager's record has no effect on what the code
            # under test sees. State them where the code reads them.
            with (
                self.subTest(incremental_streaming_output=incremental),
                get_context().override_server_args(
                    stream_response_default_include_usage=False,
                    incremental_streaming_output=incremental,
                ),
            ):
                texts = ("a", "b", "c") if incremental else ("a", "ab", "abc")
                output_ids = (
                    ([5], [6], [7]) if incremental else ([5], [5, 6], [5, 6, 7])
                )
                chunks = [
                    {
                        "text": text,
                        "output_ids": ids,
                        "prompt_token_ids": [1, 2],
                        "meta_info": {
                            "id": "cmpl-test",
                            "prompt_tokens": 2,
                            "completion_tokens": i + 1,
                            "finish_reason": {"type": "stop"} if i == 2 else None,
                        },
                        "index": 0,
                    }
                    for i, (text, ids) in enumerate(zip(texts, output_ids))
                ]

                async def _mock_generate(*args, _chunks=chunks, **kwargs):
                    for chunk in _chunks:
                        yield chunk

                self.sc.tokenizer_manager.generate_request = _mock_generate

                async def run_stream():
                    return [
                        chunk
                        async for chunk in self.sc._generate_completion_stream(
                            adapted_request, req, self.fastapi_request
                        )
                    ]

                loop = get_or_create_event_loop()
                raw_chunks = loop.run_until_complete(run_stream())

                choices = []
                for raw in raw_chunks:
                    if not raw.startswith("data: ") or raw.strip() == "data: [DONE]":
                        continue
                    data = json.loads(raw[len("data: ") :])
                    choices.extend(data.get("choices", []))

                token_ids = [tid for c in choices for tid in c.get("token_ids", [])]
                text = "".join(c["text"] for c in choices)
                self.assertEqual(text, "abc")
                self.assertEqual(token_ids, [5, 6, 7])
                self.assertEqual(choices[0]["prompt_token_ids"], [1, 2])
                for choice in choices[1:]:
                    self.assertNotIn("prompt_token_ids", choice)

    def test_non_streaming_cached_tokens_details_emits_sglext(self):
        """Test that non-streaming completion responses emit cached token details in sglext."""

        req = CompletionRequest(
            model="x",
            prompt="Hello world",
            max_tokens=100,
            return_cached_tokens_details=True,
        )
        ret = [
            {
                "text": "Cached response",
                "meta_info": {
                    "id": "cmpl-cache-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 6,
                    "cached_tokens_details": {
                        "device": 4,
                        "host": 1,
                        "storage": 1,
                        "storage_backend": "file",
                    },
                    "finish_reason": {"type": "stop", "matched": None},
                    "weight_version": "default",
                },
            }
        ]

        response = self.sc._build_completion_response(req, ret, 1234567890)

        self.assertIsNotNone(response.sglext)
        self.assertEqual(
            response.sglext.cached_tokens_details.model_dump(exclude_none=True),
            {
                "device": 4,
                "host": 1,
                "storage": 1,
                "storage_backend": "file",
            },
        )

    def test_parallel_sampling_returns_spec_details_per_choice(self):
        req = CompletionRequest(
            model="x",
            prompt="Hello world",
            max_tokens=100,
            n=2,
            return_spec_tokens_details=True,
        )
        ret = [_spec_result(index) for index in range(2)]

        response = self.sc._build_completion_response(req, ret, 1234567890)

        details = response.sglext.spec_tokens_details
        self.assertEqual(len(details), 2)
        self.assertEqual(details[0].spec_cap_length, 1.0)
        self.assertEqual(details[0].spec_block_accept_length, 0.5)
        self.assertEqual(details[0].spec_cap_lens_histogram, [0, 1])
        self.assertEqual(details[1].spec_cap_length, 2.0)
        self.assertEqual(details[1].spec_block_accept_length, 1.5)
        self.assertEqual(details[1].spec_cap_lens_histogram, [1, 1])

        single_req = req.model_copy(update={"n": 1})
        single_response = self.sc._build_completion_response(
            single_req, ret[:1], 1234567890
        )
        self.assertEqual(
            single_response.sglext.spec_tokens_details.spec_cap_length,
            1.0,
        )

        disabled_req = single_req.model_copy(
            update={"return_spec_tokens_details": False}
        )
        disabled_response = self.sc._build_completion_response(
            disabled_req, ret[:1], 1234567890
        )
        self.assertIsNone(disabled_response.sglext)

    def test_streaming_parallel_sampling_orders_spec_details_by_choice(self):
        async def mock_generate(*args, **kwargs):
            for index in (1, 0):
                yield _spec_result(index)

        self.sc.tokenizer_manager.generate_request = mock_generate
        req = CompletionRequest(
            model="x",
            prompt="Hello world",
            max_tokens=100,
            n=2,
            stream=True,
            return_spec_tokens_details=True,
        )
        adapted_request, _ = self.sc._convert_to_internal_request(req)

        async def run_stream(request):
            return [
                chunk
                async for chunk in self.sc._generate_completion_stream(
                    adapted_request, request, self.fastapi_request
                )
            ]

        chunks = get_or_create_event_loop().run_until_complete(run_stream(req))
        parsed = [
            json.loads(chunk[len("data: ") :])
            for chunk in chunks
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]"
        ]
        details = next(chunk["sglext"] for chunk in parsed if "sglext" in chunk)[
            "spec_tokens_details"
        ]
        self.assertEqual([item["spec_cap_length"] for item in details], [1.0, 2.0])
        self.assertEqual(
            [item["spec_cap_lens_histogram"] for item in details],
            [[0, 1], [1, 1]],
        )

        async def mock_single_generate(*args, **kwargs):
            async for content in mock_generate():
                if content["index"] == 0:
                    yield content

        self.sc.tokenizer_manager.generate_request = mock_single_generate
        single_req = req.model_copy(update={"n": 1})
        single_chunks = get_or_create_event_loop().run_until_complete(
            run_stream(single_req)
        )
        single_parsed = [
            json.loads(chunk[len("data: ") :])
            for chunk in single_chunks
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]"
        ]
        single_details = next(
            chunk["sglext"] for chunk in single_parsed if "sglext" in chunk
        )["spec_tokens_details"]
        self.assertIsInstance(single_details, dict)

    def test_streaming_cached_tokens_details_emits_sglext(self):
        """Test that streaming completion responses emit cached token details in sglext."""

        async def _mock_generate_with_cached_tokens_details(*args, **kwargs):
            yield {
                "text": "Cached response",
                "meta_info": {
                    "id": "cmpl-cache-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 6,
                    "cached_tokens_details": {
                        "device": 4,
                        "host": 1,
                        "storage": 1,
                        "storage_backend": "file",
                    },
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.sc.tokenizer_manager.generate_request = (
            _mock_generate_with_cached_tokens_details
        )

        req = CompletionRequest(
            model="x",
            prompt="Hello world",
            max_tokens=100,
            stream=True,
            return_cached_tokens_details=True,
        )

        adapted_request, _ = self.sc._convert_to_internal_request(req)

        async def run_stream():
            chunks = []
            async for chunk in self.sc._generate_completion_stream(
                adapted_request, req, self.fastapi_request
            ):
                chunks.append(chunk)
            return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        sglext_chunks = []
        for chunk in chunks:
            if not chunk.startswith("data: ") or chunk.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk[len("data: ") :])
            if "sglext" in data:
                sglext_chunks.append(data)

        self.assertEqual(len(sglext_chunks), 1)
        self.assertEqual(sglext_chunks[0]["choices"], [])
        self.assertEqual(
            sglext_chunks[0]["sglext"]["cached_tokens_details"],
            {
                "device": 4,
                "host": 1,
                "storage": 1,
                "storage_backend": "file",
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
