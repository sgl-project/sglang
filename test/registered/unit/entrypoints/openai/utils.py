"""Stub CUDA-only deps before importing sglang.srt serving modules. Must
be imported first by every /v1/responses test that runs on CPU."""

try:
    import torch

    _ORIGINAL_TORCH_COMPILE = torch.compile

    def _identity_compile(fn=None, **kwargs):
        if fn is None:
            return lambda inner_fn: inner_fn
        return fn

    torch.compile = _identity_compile
except ImportError:
    torch = None
    _ORIGINAL_TORCH_COMPILE = None

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import asyncio
import json
from typing import AsyncIterator
from unittest.mock import Mock

from sglang.srt.entrypoints.openai.protocol import RequestResponseMetadata
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.runtime_context import get_context, publish
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=0,
    suite="base-a-test-cpu",
    disabled="helper module — exported fixtures, not a test",
)

if torch is not None:
    torch.compile = _ORIGINAL_TORCH_COMPILE


class MockTokenizerManager:
    def __init__(self, *, is_multimodal: bool = False):
        self.model_path = "dummy"
        self.model_config = Mock(is_multimodal=is_multimodal, context_len=4096)
        self.model_config.get_default_sampling_params.return_value = {}
        self.model_config.hf_config = Mock(
            model_type="llama", architectures=["LlamaForCausalLM"]
        )
        self.server_args = Mock(
            enable_cache_report=False,
            reasoning_parser=None,
            stream_response_default_include_usage=False,
            tokenizer_metrics_allowed_custom_labels=None,
            tool_call_parser=None,
            incremental_streaming_output=False,
        )
        # Stands in for the context's resolved leaves: an override replaces the
        # field's one live value, the seed stays on server_args.
        self._config_overrides = {}
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1
        self.num_reserved_tokens = 0
        self.generate_request = Mock()
        self.create_abort_task = Mock()

    def config_value(self, name: str):
        """The value in effect for one config field."""
        if name in self._config_overrides:
            return self._config_overrides[name]
        return getattr(self.server_args, name)


class MockTemplateManager:
    def __init__(self):
        self.chat_template_name = "llama-3"
        self.jinja_template_content_format = None
        self.completion_template_name = None
        self.reasoning_config = None
        self.force_reasoning = False
        self.jinja_template_may_reorder_tool_results = False


def make_serving(*, is_multimodal: bool = False) -> OpenAIServingResponses:
    """The serving layer reads its config from the bags, so the fixture
    publishes one. Idempotent: a caller that already published keeps its own,
    which is how a test states a value the default record does not carry."""
    if not get_context().is_config_namespace_published("serving"):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
    return OpenAIServingResponses(
        MockTokenizerManager(is_multimodal=is_multimodal), MockTemplateManager()
    )


async def collect_stream_events(stream: AsyncIterator[str]) -> list[str]:
    events = []
    async for chunk in stream:
        events.append(chunk)
    return events


def event_types(events: list[str]) -> list[str]:
    return [
        line[len("event: ") :].strip()
        for chunk in events
        for line in chunk.splitlines()
        if line.startswith("event: ")
    ]


def event_payloads(events: list[str]) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for chunk in events
        for line in chunk.splitlines()
        if line.startswith("data: ")
    ]


def find_completed_event(events: list[str]) -> dict:
    for chunk in events:
        lines = chunk.splitlines()
        if lines and lines[0] == "event: response.completed":
            return json.loads(lines[1][len("data: ") :])
    raise AssertionError("response.completed event missing from stream")


def engine_chunk(text, completion_tokens=1, *, finish=False):
    return {
        "text": text,
        "meta_info": {
            "id": "rid",
            "prompt_tokens": 5,
            "completion_tokens": completion_tokens,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "finish_reason": {"type": "stop"} if finish else None,
        },
    }


class StreamFixture:
    """Drives ``responses_stream_generator_non_harmony`` over a chunk list."""

    def __init__(self, serving, request, *, require_reasoning=False):
        self.serving = serving
        self.request = request
        self.require_reasoning = require_reasoning
        self.request_metadata = RequestResponseMetadata(request_id=request.request_id)

    def run(self, chunks) -> list[str]:
        async def gen():
            for ch in chunks:
                yield ch

        async def collect():
            return await collect_stream_events(
                self.serving.responses_stream_generator_non_harmony(
                    self.request,
                    sampling_params={},
                    result_generator=gen(),
                    model_name="x",
                    tokenizer=Mock(),
                    request_metadata=self.request_metadata,
                    require_reasoning=self.require_reasoning,
                )
            )

        return asyncio.run(collect())

    def run_seq(self, chunks) -> list[tuple]:
        """``run`` plus (event type, payload) pairing, the common assertion shape."""
        events = self.run(chunks)
        return list(zip(event_types(events), event_payloads(events)))


def input_processor(serving, **overrides):
    from dataclasses import replace

    from sglang.srt.entrypoints.chat_input.processor import ChatInputProcessor
    from sglang.srt.entrypoints.chat_input.types import ChatModelConfig

    processor = getattr(serving, "input_processor", None)
    if processor is None:
        config = ChatModelConfig(
            tokenizer=serving.tokenizer_manager.tokenizer,
            template_manager=getattr(
                serving, "template_manager", Mock(reasoning_config=None)
            ),
            is_multimodal=False,
        )
        processor = ChatInputProcessor(config)
        serving.input_processor = processor
    aliases = {
        "chat_encoding_spec": "chat_encoding_spec",
        "tool_call_parser": "tool_call_parser",
        "reasoning_parser": "reasoning_parser",
        "reasoning_detector": "_reasoning_detector",
        "default_chat_template_kwargs": "default_chat_template_kwargs",
        "dsv4_reasoning_effort_profile": "_dsv4_reasoning_effort_profile",
        "inkling_default_reasoning_effort": "_inkling_default_reasoning_effort",
        "tokenizer_auto_adds_specials": "_tokenizer_auto_adds_specials",
        "is_gpt_oss": "is_gpt_oss",
        "is_gemma4": "is_gemma4",
    }
    values = {
        name: vars(serving)[alias]
        for name, alias in aliases.items()
        if alias in vars(serving)
    }
    values["tokenizer"] = serving.tokenizer_manager.tokenizer
    model_config = getattr(serving.tokenizer_manager, "model_config", None)
    if model_config is not None and isinstance(model_config.is_multimodal, bool):
        values["is_multimodal"] = model_config.is_multimodal
    values.update(overrides)
    processor.config = replace(processor.config, **values)
    processor.renderer.config = processor.config
    return processor


def sync_serving(serving):
    input_processor(serving)
    return serving


def prepared_chat(
    prompt,
    *,
    image_data=None,
    audio_data=None,
    video_data=None,
    modalities=None,
    stop=None,
    tool_call_constraint=None,
    skip_special_tokens=True,
    require_reasoning=False,
    reasoning_end_token_ids=None,
    chat_template_kwargs=None,
    reasoning_effort=None,
):
    from sglang.srt.entrypoints.chat_input.types import (
        PreparedChat,
        TextPrompt,
        TokenPrompt,
    )

    return PreparedChat(
        prompt=TextPrompt(prompt) if isinstance(prompt, str) else TokenPrompt(prompt),
        image_data=image_data,
        audio_data=audio_data,
        video_data=video_data,
        modalities=modalities or [],
        stop=stop,
        tool_call_constraint=tool_call_constraint,
        skip_special_tokens=skip_special_tokens,
        require_reasoning=require_reasoning,
        reasoning_end_token_ids=reasoning_end_token_ids,
        chat_template_kwargs=chat_template_kwargs,
        reasoning_effort=reasoning_effort,
    )


def prompt_value(result):
    from sglang.srt.entrypoints.chat_input.types import TextPrompt

    prompt = result.prompt
    return prompt.text if isinstance(prompt, TextPrompt) else prompt.token_ids
