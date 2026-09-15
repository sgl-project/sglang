"""Python OpenAI adapter contracts shared with the native HTTP tests."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, Request
from msgspec.structs import asdict

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    MessageProcessingResult,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FIXTURE = (
    Path(__file__).resolve().parents[5]
    / "rust/sglang-server/testdata/openai_requests_python.json"
)
REQUEST_FIELDS = (
    "text",
    "input_ids",
    "cache_salt",
    "extra_key",
    "priority",
    "routing_key",
    "routed_dp_rank",
    "disagg_prefill_dp_rank",
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
    "return_hidden_states",
    "return_routed_experts",
    "routed_experts_start_len",
    "return_prompt_token_ids",
    "return_sampling_mask",
    "custom_logit_processor",
    "return_logprob",
    "logprob_start_len",
    "top_logprobs_num",
)
SAMPLING_FIELDS = (
    "max_new_tokens",
    "min_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "stop_token_ids",
    "stop_strs",
    "stop_regex_strs",
    "no_stop_trim",
    "ignore_eos",
    "skip_special_tokens",
    "json_schema",
    "regex",
    "ebnf",
    "structural_tag",
    "custom_params",
    "sampling_seed",
)


class ContractChat(OpenAIServingChat):
    """Keep template rendering outside the adapter contract under test."""

    def __init__(self, manager, defaults):
        OpenAIServingBase.__init__(self, manager)
        self.is_gpt_oss = False
        self.chat_encoding_spec = None
        self.default_sampling_params = defaults
        self.reasoning_parser = None
        self.tool_call_parser = None
        self.template_manager = SimpleNamespace(reasoning_config=None)

    def _process_messages(self, request, is_multimodal):
        assert not is_multimodal and request.input_ids is not None
        return MessageProcessingResult(
            prompt="",
            prompt_ids=request.input_ids,
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=request.stop or [],
        )


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


def request_contract_fixture():
    defaults = {"top_k": 23, "min_p": 0.125, "repetition_penalty": 1.125}
    manager = SimpleNamespace(
        server_args=SimpleNamespace(return_input_ids=False),
        model_config=SimpleNamespace(is_multimodal=False),
    )
    completion = OpenAIServingCompletion(
        manager, SimpleNamespace(completion_template_name=None)
    )
    chat = ContractChat(manager, defaults)
    common = {
        "model": "org/model",
        "rid": "contract",
        "max_tokens": 8,
        "top_k": 7,
        "min_p": 0.25,
        "min_tokens": 1,
        "repetition_penalty": 1.25,
        "temperature": 0.75,
        "top_p": 0.875,
        "frequency_penalty": 0.25,
        "presence_penalty": -0.25,
        "seed": 17,
        "stop": ["END"],
        "stop_token_ids": [12, 13],
        "stop_regex": "X{2,3}",
        "no_stop_trim": True,
        "ignore_eos": True,
        "skip_special_tokens": False,
        "custom_params": {"scale": 2},
        "cache_salt": "tenant",
        "extra_key": "classification",
        "priority": 3,
        "routed_dp_rank": 1,
        "disagg_prefill_dp_rank": 2,
        "bootstrap_host": "::1",
        "bootstrap_port": 8998,
        "bootstrap_room": 10,
        "return_hidden_states": "last",
        "return_routed_experts": True,
        "routed_experts_start_len": 2,
        "return_token_ids": True,
        "custom_logit_processor": "processor",
    }
    messages = [{"role": "user", "content": "unused pretokenized prompt"}]
    cases = []
    for endpoint, bodies in (
        (
            "completions",
            [
                {**common, "prompt": [1, 2, 3]},
                {
                    **common,
                    "prompt": [[1, 2], [3, 4, 5]],
                    "n": 2,
                    "rid": None,
                    "cache_salt": ["first", "second"],
                    "extra_key": ["a", "b"],
                    "bootstrap_room": [10, 11],
                    "return_hidden_states": True,
                    "stream": True,
                },
                {
                    "model": "org/model",
                    "prompt": "json",
                    "max_tokens": 4,
                    "response_format": {"type": "json_object"},
                },
                {
                    "model": "org/model",
                    "prompt": "regex",
                    "max_tokens": 4,
                    "regex": "[a-z]+",
                },
                {
                    "model": "org/model",
                    "prompt": "ebnf",
                    "max_tokens": 4,
                    "ebnf": 'root ::= "yes"',
                },
            ],
        ),
        (
            "chat",
            [
                {
                    **common,
                    "messages": messages,
                    "input_ids": [1, 2, 3],
                    "return_meta_info": True,
                    "return_sampling_mask": True,
                },
                {
                    "model": "org/model",
                    "messages": messages,
                    "input_ids": [2, 3],
                    "n": 2,
                    "max_tokens": 5,
                    "cache_salt": "chat",
                    "return_input_ids_in_sglext": True,
                    "return_output_ids_in_sglext": True,
                },
                {
                    "model": "org/model",
                    "messages": messages,
                    "input_ids": [4, 5],
                    "max_tokens": 6,
                    "stream": True,
                    "json_schema": "ignored by the Python chat protocol",
                },
            ],
        ),
    ):
        for body in bodies:
            headers = {"x-smg-routing-key": "session", "x-data-parallel-rank": "3"}
            if endpoint == "chat":
                headers.update(
                    {"x-override-priority": "-5", "x-override-routed-dp-rank": "4"}
                )
            raw = Request(
                {
                    "type": "http",
                    "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
                }
            )
            request = (
                ChatCompletionRequest if endpoint == "chat" else CompletionRequest
            ).model_validate(body)
            with (
                envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.override(True),
                get_context().override_server_args(return_input_ids=False),
            ):
                internal, _ = (
                    chat if endpoint == "chat" else completion
                )._convert_to_internal_request(request, raw)
            internal.normalize_batch_and_arguments()
            items = (
                [internal]
                if internal.is_single
                else [internal[index] for index in range(internal.batch_size)]
            )
            expected = []
            for item in items:
                params = SamplingParams(**item.sampling_params)
                params.normalize(CharacterTokenizer())
                params.verify(100)
                fields = vars(item)
                sampling = asdict(params)
                observed = {name: fields[name] for name in REQUEST_FIELDS}
                observed["sampling"] = {
                    name: (
                        sorted(sampling[name])
                        if isinstance(sampling[name], set)
                        else sampling[name]
                    )
                    for name in SAMPLING_FIELDS
                }
                expected.extend([observed] * internal.parallel_sample_num)
            cases.append(
                {
                    "endpoint": endpoint,
                    "body": body,
                    "headers": headers,
                    "model_defaults": defaults if endpoint == "chat" else {},
                    "expected": expected,
                }
            )
    invalid = []
    for endpoint, body, headers in [
        ("completions", {"prompt": [1, 2], "return_hidden_states": "full"}, {}),
        ("completions", {"prompt": [1, 2], "priority": []}, {}),
        ("completions", {"prompt": [1, 2], "min_p": 1.5}, {}),
        ("completions", {"prompt": [1, 2], "cache_salt": ["tenant"]}, {}),
        ("completions", {"prompt": [1, 2], "regex": "x", "json_schema": "{}"}, {}),
        ("completions", {"prompt": [1, 2]}, {"x-data-parallel-rank": "bad"}),
        *[
            (
                "chat",
                {"messages": messages, "input_ids": [1, 2], "stream": True, name: True},
                {},
            )
            for name in (
                "return_prompt_token_ids",
                "return_token_ids",
                "return_meta_info",
            )
        ],
    ]:
        body = {"model": "org/model", **body}
        raw = Request(
            {
                "type": "http",
                "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
            }
        )
        try:
            request = (
                ChatCompletionRequest if endpoint == "chat" else CompletionRequest
            ).model_validate(body)
            internal, _ = (
                chat if endpoint == "chat" else completion
            )._convert_to_internal_request(request, raw)
            internal.normalize_batch_and_arguments()
            params = SamplingParams(**internal.sampling_params)
            params.normalize(CharacterTokenizer())
            params.verify(100)
        except (ValueError, HTTPException):
            invalid.append({"endpoint": endpoint, "body": body, "headers": headers})
        else:
            raise AssertionError(f"Python accepted invalid fixture {body}")
    return {
        "request_fields": REQUEST_FIELDS,
        "sampling_fields": SAMPLING_FIELDS,
        "cases": cases,
        "invalid": invalid,
    }


def test_rust_request_fixture_matches_python_openai_adapters():
    assert json.loads(FIXTURE.read_text()) == json.loads(
        json.dumps(request_contract_fixture())
    )


RESPONSE_FIXTURE = FIXTURE.with_name("openai_responses_python.json")


def response_contract_fixture():
    cases = []
    for endpoint, n, count, hidden, meta, global_ids, enabled in [
        ("completions", 1, 1, "last", False, False, True),
        ("completions", 2, 4, True, False, False, True),
        ("completions", 1, 1, False, False, False, False),
        ("chat", 1, 1, True, True, False, True),
        ("chat", 2, 2, "last", False, False, True),
        ("chat", 1, 1, False, False, True, False),
    ]:
        args = SimpleNamespace(
            enable_cache_report=enabled,
            return_input_ids=global_ids,
            return_output_ids=global_ids,
        )
        manager = SimpleNamespace(server_args=args)
        adapter = (
            ContractChat(manager, {})
            if endpoint == "chat"
            else OpenAIServingCompletion(
                manager, SimpleNamespace(completion_template_name=None)
            )
        )
        body = {
            "model": "org/model",
            "n": n,
            "return_hidden_states": hidden,
            "return_routed_experts": enabled,
            "return_cached_tokens_details": enabled,
            "return_spec_tokens_details": enabled,
            "return_token_ids": enabled,
        }
        if endpoint == "chat":
            body.update(
                messages=[{"role": "user", "content": "hi"}],
                return_meta_info=meta,
                return_input_ids_in_sglext=enabled,
                return_output_ids_in_sglext=enabled,
            )
            request = ChatCompletionRequest.model_validate(body)
        else:
            body["prompt"] = [[1, 2], [3, 4, 5]] if count > n else [1, 2]
            request = CompletionRequest.model_validate(body)
        items = []
        for index in range(count):
            prompt = [1, 2] if index // n == 0 else [3, 4, 5]
            tokens = [10 + index, 20 + index]
            metadata = {
                "id": f"contract-{index}",
                "prompt_tokens": len(prompt),
                "completion_tokens": 2,
                "reasoning_tokens": index + 1,
                "cached_tokens": 1,
                "num_retractions": 0,
                "cached_tokens_details": {
                    "device": 1,
                    "host": 0,
                    **(
                        {"storage": 0, "storage_backend": "file"}
                        if index % 2 == 0
                        else {}
                    ),
                },
                "image_tokens": 3,
                "audio_tokens": 4,
                "video_tokens": 5,
                "weight_version": "v2",
                "weight_versions": [
                    {"version": "v1", "start": 0, "end": 1},
                    {"version": "v2", "start": 1, "end": 2},
                ],
                "finish_reason": {"type": "stop", "matched": "END"},
                "routed_experts": "AAECAw==",
                "spec_accept_rate": 0.5,
                "spec_accept_length": 2.0,
                "spec_num_correct_drafts": 1,
                "spec_num_proposed_drafts": 2,
                "spec_verify_ct": 1,
            }
            if hidden:
                metadata["hidden_states"] = (
                    [0.25, 0.5] if hidden == "last" else [[[1.0, 2.0]], [0.25, 0.5]]
                )
            if meta:
                metadata["output_token_sampling_mask"] = [True, False]
                metadata["model_info"] = {"checkpoint_path": "model/checkpoint"}
            items.append(
                {
                    "text": "" if index == 0 else "ok",
                    "output_ids": tokens,
                    "prompt_token_ids": prompt,
                    "meta_info": metadata,
                }
            )
        with get_context().override_server_args(**vars(args)):
            result = (
                adapter._build_chat_response(request, items, 123)
                if endpoint == "chat"
                else adapter._build_completion_response(request, items, 123)
            )
        cases.append(
            {
                "endpoint": endpoint,
                "body": body,
                "args": vars(args),
                "items": items,
                "expected": result.model_dump(),
            }
        )
    return cases


def test_rust_response_fixture_matches_python_openai_formatters():
    assert json.loads(RESPONSE_FIXTURE.read_text()) == response_contract_fixture()


STREAM_FIXTURE = FIXTURE.with_name("openai_streams_python.json")


class StreamManager:
    def __init__(self, server_args, items):
        self.server_args = server_args
        self.items = items

    async def generate_request(self, request, raw_request):
        for item in self.items:
            yield item


async def stream_contract_fixture():
    cases = []
    for case in response_contract_fixture():
        body = copy.deepcopy(case["body"])
        chat = case["endpoint"] == "chat"
        if chat:
            body["return_token_ids"] = False
            body["return_meta_info"] = False
        body["stream"] = True
        body["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
        args = {
            **case["args"],
            "incremental_streaming_output": False,
            "stream_response_default_include_usage": False,
        }
        chunks = []
        for step in range(2):
            for index, item in enumerate(case["items"]):
                chunk = copy.deepcopy(item)
                chunk["index"] = index
                chunk["output_ids"] = item["output_ids"][: step + 1]
                chunk["text"] = item["text"][: step + 1]
                chunk["meta_info"]["completion_tokens"] = step + 1
                if step == 0:
                    chunk["meta_info"]["finish_reason"] = None
                    chunk["meta_info"].pop("hidden_states", None)
                chunks.append(chunk)
        manager = StreamManager(SimpleNamespace(**args), chunks)
        adapter = (
            ContractChat(manager, {})
            if chat
            else OpenAIServingCompletion(
                manager, SimpleNamespace(completion_template_name=None)
            )
        )
        request = (ChatCompletionRequest if chat else CompletionRequest).model_validate(
            body
        )
        headers = {"x-sglext-ids-framed": "1"} if chat and body["n"] == 2 else {}
        raw = Request(
            {
                "type": "http",
                "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
            }
        )
        generator = (
            adapter._generate_chat_stream
            if chat
            else adapter._generate_completion_stream
        )(None, request, raw)
        frames = []
        with (
            patch("time.time", return_value=123),
            get_context().override_server_args(**args),
        ):
            async for chunk in generator:
                event = None
                for line in chunk.splitlines():
                    if line.startswith("event: "):
                        event = line[7:]
                    elif line.startswith("data: ") and line != "data: [DONE]":
                        frames.append({"event": event, "data": json.loads(line[6:])})
        tail = [
            frame
            for frame in frames
            if not frame["data"]["choices"]
            or any(
                "hidden_states" in choice or "hidden_states" in choice.get("delta", {})
                for choice in frame["data"]["choices"]
            )
        ]
        cases.append(
            {
                "endpoint": case["endpoint"],
                "body": body,
                "args": args,
                "headers": headers,
                "chunks": chunks,
                "tail": tail,
            }
        )
    return cases


def test_rust_stream_fixture_matches_python_openai_generators():
    assert json.loads(STREAM_FIXTURE.read_text()) == asyncio.run(
        stream_contract_fixture()
    )


ERROR_FIXTURE = FIXTURE.with_name("openai_errors_python.json")


class ErrorManager(StreamManager):
    async def generate_request(self, request, raw_request):
        async for item in super().generate_request(request, raw_request):
            yield item
        raise ValueError("validation failed: bad input")

    def create_abort_task(self, request):
        return None


async def error_contract_fixture():
    cases = []
    for chat in (False, True):
        for late in (False, True):
            args = SimpleNamespace(
                enable_cache_report=False,
                return_input_ids=True,
                return_output_ids=True,
                incremental_streaming_output=False,
                stream_response_default_include_usage=False,
            )
            items = (
                [
                    {
                        "text": "x",
                        "output_ids": [10],
                        "prompt_token_ids": [1, 2],
                        "meta_info": {
                            "id": "contract",
                            "prompt_tokens": 2,
                            "completion_tokens": 1,
                            "finish_reason": None,
                        },
                    }
                ]
                if late
                else []
            )
            manager = ErrorManager(args, items)
            adapter = (
                ContractChat(manager, {})
                if chat
                else OpenAIServingCompletion(
                    manager, SimpleNamespace(completion_template_name=None)
                )
            )
            body = {"model": "org/model", "stream": True}
            if chat:
                body.update(
                    messages=[{"role": "user", "content": "hi"}], input_ids=[1, 2]
                )
            else:
                body["prompt"] = [1, 2]
            request = (
                ChatCompletionRequest if chat else CompletionRequest
            ).model_validate(body)
            with get_context().override_server_args(**vars(args)):
                response = await adapter._handle_streaming_request(None, request, None)
                if late:
                    frames = []
                    async for chunk in response.body_iterator:
                        for line in chunk.splitlines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                frames.append(json.loads(line[6:]))
                    errors = [frame for frame in frames if "error" in frame]
                    assert len(errors) == 1
                    assert not any("sglext" in frame for frame in frames)
                    error = errors[0]
                else:
                    error = json.loads(response.body)
            cases.append(
                {
                    "endpoint": "chat" if chat else "completions",
                    "late": late,
                    "body": body,
                    "status": response.status_code,
                    "error": error,
                }
            )
    return cases


def test_rust_error_fixture_matches_python_stream_validation():
    assert json.loads(ERROR_FIXTURE.read_text()) == asyncio.run(
        error_contract_fixture()
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
