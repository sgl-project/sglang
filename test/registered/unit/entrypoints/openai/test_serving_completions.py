"""
Unit-tests for the refactored completions-serving handler (no pytest).
Run with:
    python -m unittest tests.test_serving_completions_unit -v
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import json
import unittest
from http import HTTPStatus
from typing import Optional
from unittest.mock import AsyncMock, Mock

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=14, suite="base-a-test-cpu")


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
        # build the mock TokenizerManager once for every test
        tm = Mock(spec=TokenizerManager)

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


class ServingCompletionOutputPaddingTestCase(unittest.TestCase):
    """``--padded-output-tokens``: nothing an observer can count may vary with
    the completion's real length -- not the token count, the frame count, the
    reported ``completion_tokens``, or the terminal frame."""

    TARGET = 4
    PAD_ID = 128001
    PAD_TEXT = "<|eot|>"

    # Verdicts of every admissible length, standing in for the short structured
    # answers whose length is the thing being hidden.
    VERDICTS = {
        1: [("y", 11)],
        2: [("sa", 11), ("fe", 12)],
        3: [("un", 11), ("sa", 12), ("fe", 13)],
        4: [("t", 11), ("r", 12), ("u", 13), ("e", 14)],
    }

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", padded_output_tokens=self.TARGET),
            role="tokenizer",
        )
        self.sc = self._make_serving()
        self.fastapi_request = Mock(spec=Request)

    def _make_serving(self):
        tm = Mock(spec=TokenizerManager)
        tm.tokenizer = Mock()
        tm.tokenizer.decode.return_value = self.PAD_TEXT
        tm.model_config = Mock(is_multimodal=False)
        tm.model_config.hf_eos_token_id = {self.PAD_ID, self.PAD_ID + 8}
        tm.server_args = Mock(enable_cache_report=False)
        tm.generate_request = AsyncMock()
        tm.create_abort_task = Mock()
        return OpenAIServingCompletion(tm, _MockTemplateManager())

    def _request(self, *, logprobs=None, max_tokens=None, **kwargs):
        # Bind the budget to the padding target. Hardcoding a larger max_tokens
        # here would send every case down the rejection path instead, so the
        # padding itself would never be exercised.
        return CompletionRequest(
            model="x",
            prompt="Hello",
            max_tokens=self.TARGET if max_tokens is None else max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            logprobs=logprobs,
            **kwargs,
        )

    def _chunks(
        self,
        pieces,
        *,
        want_logprobs=False,
        supply_logprobs=True,
        finish="stop",
        report_tokens=None,
        coalesce=False,
    ):
        """Cumulative manager chunks, one per token unless ``coalesce``."""
        groups = [pieces] if coalesce else [pieces[: i + 1] for i in range(len(pieces))]
        chunks = []
        for i, prefix in enumerate(groups):
            is_last = i == len(groups) - 1
            ids = [tid for _, tid in prefix]
            meta = {
                "id": "cmpl-pad",
                "prompt_tokens": 3,
                "completion_tokens": (
                    report_tokens if is_last and report_tokens is not None else len(ids)
                ),
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "finish_reason": ({"type": finish} if is_last else None),
            }
            if is_last and finish == "stop":
                meta["finish_reason"]["matched"] = 2
            if want_logprobs:
                # Distinct per position, so a mis-split pairs a visibly wrong
                # value with its token instead of an identical one.
                supplied = (
                    [
                        (self._logprob_at(n), tid, text)
                        for n, (text, tid) in enumerate(prefix)
                    ]
                    if supply_logprobs
                    else []
                )
                meta["output_token_logprobs"] = supplied
                meta["output_token_logprobs_length"] = len(supplied)
                meta["output_top_logprobs"] = (
                    [
                        [(self._logprob_at(n), tid, text)]
                        for n, (text, tid) in enumerate(prefix)
                    ]
                    if supply_logprobs
                    else []
                )
            chunks.append(
                {
                    "text": "".join(text for text, _ in prefix),
                    "output_ids": ids,
                    "meta_info": meta,
                    "index": 0,
                }
            )
        return chunks

    def _run(self, request, chunks, serving=None):
        serving = serving or self.sc

        async def fake_generate(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        serving.tokenizer_manager.generate_request = fake_generate
        adapted_request, _ = serving._convert_to_internal_request(request)

        async def collect():
            return [
                frame
                async for frame in serving._generate_completion_stream(
                    adapted_request, request, self.fastapi_request
                )
            ]

        return get_or_create_event_loop().run_until_complete(collect())

    def _shape(self, frames):
        """Everything an observer can count, with the verdict text projected out."""
        shape = []
        for frame in frames:
            payload = frame[len("data: ") :].strip()
            if payload == "[DONE]":
                shape.append(("done",))
                continue
            data = json.loads(payload)
            if "error" in data:
                shape.append(("error",))
                continue
            if data.get("usage"):
                shape.append(("usage", data["usage"]["completion_tokens"]))
                continue
            choice = data["choices"][0]
            logprobs = choice.get("logprobs")
            shape.append(
                (
                    "choice",
                    choice["finish_reason"],
                    choice.get("matched_stop"),
                    None if logprobs is None else len(logprobs["token_logprobs"]),
                )
            )
        return shape

    def _text(self, frames):
        out = []
        for frame in frames:
            payload = frame[len("data: ") :].strip()
            if payload == "[DONE]":
                continue
            for choice in json.loads(payload).get("choices", []):
                out.append(choice.get("text") or "")
        return "".join(out)

    def _expected_shape(self, *, logprob_positions=None):
        return (
            [("choice", None, None, logprob_positions)] * self.TARGET
            + [("choice", "length", None, None)]
            + [("usage", self.TARGET), ("done",)]
        )

    def test_every_admissible_length_pads_to_one_shape(self):
        for length, pieces in self.VERDICTS.items():
            frames = self._run(self._request(), self._chunks(pieces))
            self.assertEqual(
                self._shape(frames),
                self._expected_shape(),
                f"a {length}-token completion is distinguishable",
            )
            self.assertEqual(self._text(frames), "".join(t for t, _ in pieces))

    def test_logprob_request_modes_do_not_leak_length(self):
        for mode in (None, 0, 1, 3):
            expected = self._expected_shape(
                logprob_positions=None if mode is None else 1
            )
            for length, pieces in self.VERDICTS.items():
                frames = self._run(
                    self._request(logprobs=mode),
                    self._chunks(pieces, want_logprobs=mode is not None),
                )
                self.assertEqual(
                    self._shape(frames),
                    expected,
                    f"logprobs={mode} leaks a {length}-token completion",
                )

    def test_missing_manager_logprobs_fall_back_to_filler(self):
        """A backend that under-supplies logprobs must not put real tokens and
        pads on differently shaped frames."""
        for mode in (0, 1, 3):
            expected = self._expected_shape(logprob_positions=1)
            for length, pieces in self.VERDICTS.items():
                frames = self._run(
                    self._request(logprobs=mode),
                    self._chunks(pieces, want_logprobs=True, supply_logprobs=False),
                )
                self.assertEqual(
                    self._shape(frames),
                    expected,
                    f"logprobs={mode} with no manager metadata leaks {length}",
                )

    @staticmethod
    def _logprob_at(position):
        return -0.5 - position

    def _logprobs_of(self, frames):
        """(tokens, token_logprobs) per content frame, in order."""
        out = []
        for frame in frames:
            payload = frame[len("data: ") :].strip()
            if payload == "[DONE]":
                continue
            data = json.loads(payload)
            if "error" in data or data.get("usage") or not data["choices"]:
                continue
            logprobs = data["choices"][0].get("logprobs")
            out.append(
                None
                if logprobs is None
                else (logprobs["tokens"], logprobs["token_logprobs"])
            )
        return out

    def test_each_frame_carries_its_own_positions_logprob(self):
        """The re-split must pair position i with the i-th new token. Handing
        every frame the filler, or shifting the pairing by one, both leave the
        frame shape intact and are only visible in the values."""
        pieces = self.VERDICTS[3]
        frames = self._run(
            self._request(logprobs=0), self._chunks(pieces, want_logprobs=True)
        )
        expected = [
            ([text], [self._logprob_at(n)]) for n, (text, _) in enumerate(pieces)
        ]
        expected += [([self.PAD_TEXT], [0.0])] * (self.TARGET - len(pieces))
        self.assertEqual(self._logprobs_of(frames)[: self.TARGET], expected)

    def test_coalesced_chunk_splits_logprobs_positionally(self):
        """count>1 with a partial supply: the manager merged three tokens into one
        delta but only priced two of them."""
        pieces = self.VERDICTS[3]
        chunk = self._chunks(pieces, want_logprobs=True, coalesce=True)[0]
        chunk["meta_info"]["output_token_logprobs"] = chunk["meta_info"][
            "output_token_logprobs"
        ][:2]
        chunk["meta_info"]["output_token_logprobs_length"] = 2
        chunk["meta_info"]["output_top_logprobs"] = chunk["meta_info"][
            "output_top_logprobs"
        ][:1]
        frames = self._run(self._request(logprobs=1), [chunk])
        self.assertEqual(
            self._logprobs_of(frames)[:3],
            [
                ([pieces[0][0]], [self._logprob_at(0)]),
                ([pieces[1][0]], [self._logprob_at(1)]),
                ([self.PAD_TEXT], [0.0]),
            ],
        )
        self.assertEqual(self._shape(frames), self._expected_shape(logprob_positions=1))

    def test_filler_logprob_is_a_finite_number_not_null(self):
        """pydantic renders a non-finite float as ``null``, a shape no real
        position ever has, so the pad frame would be identifiable by that alone."""
        frames = self._run(
            self._request(logprobs=2),
            self._chunks(self.VERDICTS[1], want_logprobs=True, supply_logprobs=False),
        )
        first = json.loads(frames[0][len("data: ") :])["choices"][0]["logprobs"]
        self.assertEqual(first["token_logprobs"], [0.0])
        self.assertIsInstance(first["token_logprobs"][0], float)
        self.assertEqual(first["top_logprobs"], [{self.PAD_TEXT: 0.0}])

    def test_pad_run_follows_the_wire_count_not_the_manager_count(self):
        """The pad run is sized from tokens this loop actually framed. Sizing it
        from meta_info's completion_tokens, which mirrors the manager and can lag
        or lead, puts the padded total off the target."""
        pieces = self.VERDICTS[2]
        chunks = self._chunks(pieces)
        chunks[-1]["meta_info"]["completion_tokens"] = 1  # manager lags the wire
        frames = self._run(self._request(), chunks)
        self.assertEqual(self._shape(frames), self._expected_shape())

    def test_graceful_abort_is_not_padded(self):
        """A cancelled request keeps its own finish_reason and honest count. An
        abort carries no status_code unless it was a system error, so padding it
        would relabel the cancellation as a normal length-capped completion."""
        aborted = [
            {
                "text": "un",
                "output_ids": [11],
                "index": 0,
                "meta_info": {
                    "id": "cmpl-pad",
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "finish_reason": {"type": "abort", "message": "user abort"},
                },
            }
        ]
        frames = self._run(self._request(), aborted)
        self.assertEqual(
            self._shape(frames),
            [("choice", "abort", None, None), ("usage", 1), ("done",)],
        )
        self.assertEqual(self._text(frames), "un")

    def test_coalesced_delta_is_resplit_one_frame_per_token(self):
        """The manager merges several decode steps into one delta when the
        consumer lags; without re-splitting, the frame count would still track
        the real length even though the token count is padded."""
        pieces = self.VERDICTS[3]
        frames = self._run(self._request(), self._chunks(pieces, coalesce=True))
        self.assertEqual(self._shape(frames), self._expected_shape())
        self.assertEqual(self._text(frames), "unsafe")

    def test_stop_and_length_finish_are_indistinguishable(self):
        pieces = self.VERDICTS[2]
        on_stop = self._run(self._request(), self._chunks(pieces, finish="stop"))
        on_length = self._run(self._request(), self._chunks(pieces, finish="length"))
        self.assertEqual(self._shape(on_stop), self._shape(on_length))
        self.assertEqual(self._shape(on_stop), self._expected_shape())

    def test_more_ids_than_target_refuses_to_report_a_padded_count(self):
        overrun = [("x", 20 + i) for i in range(self.TARGET + 2)]
        frames = self._run(self._request(), self._chunks(overrun))
        error = json.loads(frames[-2][len("data: ") :])
        self.assertIn("outran --padded-output-tokens", error["error"]["message"])
        self.assertNotIn(("usage", self.TARGET), self._shape(frames))

    def test_manager_count_above_target_refuses_to_report_a_padded_count(self):
        frames = self._run(
            self._request(),
            self._chunks(self.VERDICTS[2], report_tokens=self.TARGET + 3),
        )
        error = json.loads(frames[-2][len("data: ") :])
        self.assertIn("outran --padded-output-tokens", error["error"]["message"])
        self.assertIn(str(self.TARGET + 3), error["error"]["message"])

    def test_rejects_a_budget_over_the_target_before_the_engine(self):
        message = self.sc._validate_request(self._request(max_tokens=self.TARGET + 1))
        self.assertIsNotNone(message)
        self.assertIn("padded-output-tokens", message)
        self.sc.tokenizer_manager.generate_request.assert_not_called()

    def test_admits_a_budget_exactly_equal_to_the_target(self):
        self.assertIsNone(
            self.sc._validate_request(self._request(max_tokens=self.TARGET))
        )

    def test_rejects_when_the_model_exposes_no_terminal_token(self):
        self.sc.tokenizer_manager.model_config.hf_eos_token_id = set()
        message = self.sc._validate_request(self._request())
        self.assertIsNotNone(message)
        self.assertIn("eos_token_id", message)

    def test_zero_token_completion_is_padded_like_any_other(self):
        """A completion that emits no tokens at all must still produce the full
        padded stream. Keying the pad run off frames-already-emitted instead of
        choices-seen dropped it to a bare usage frame reporting 0, which is the
        most distinguishable stream of all."""
        empty = [
            {
                "text": "",
                "output_ids": [],
                "index": 0,
                "meta_info": {
                    "id": "cmpl-pad",
                    "prompt_tokens": 3,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": 2},
                },
            }
        ]
        frames = self._run(self._request(), empty)
        self.assertEqual(self._shape(frames), self._expected_shape())
        self.assertEqual(self._text(frames), "")

    def test_every_choice_is_padded_when_n_is_above_one(self):
        """With n>1 each choice pads independently. A choice that generated
        nothing was previously absent from the stream, so the frame count
        revealed how many choices produced tokens."""
        chunks = [
            {
                "text": "a",
                "output_ids": [11],
                "index": 0,
                "meta_info": {
                    "id": "cmpl-pad",
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": 2},
                },
            },
            {
                "text": "",
                "output_ids": [],
                "index": 1,
                "meta_info": {
                    "id": "cmpl-pad",
                    "prompt_tokens": 3,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": 2},
                },
            },
        ]
        frames = self._run(self._request(n=2), chunks)
        per_choice = [("choice", None, None, None)] * self.TARGET + [
            ("choice", "length", None, None)
        ]
        self.assertEqual(
            self._shape(frames),
            per_choice * 2 + [("usage", self.TARGET * 2), ("done",)],
        )

    def test_rejects_echo(self):
        """Echo replays the prompt through the completion stream, which padding
        does not cover, and with logprobs it prepends input positions into the
        payload the per-token split consumes."""
        message = self.sc._validate_request(self._request(echo=True))
        self.assertIsNotNone(message)
        self.assertIn("echo", message)

    def test_flag_unset_leaves_the_stream_untouched(self):
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        serving = self._make_serving()
        pieces = self.VERDICTS[2]
        frames = self._run(self._request(), self._chunks(pieces), serving=serving)
        self.assertEqual(
            self._shape(frames),
            [
                ("choice", None, None, None),
                ("choice", "stop", 2, None),
                ("usage", 2),
                ("done",),
            ],
        )
        self.assertEqual(self._text(frames), "safe")


if __name__ == "__main__":
    unittest.main(verbosity=2)
