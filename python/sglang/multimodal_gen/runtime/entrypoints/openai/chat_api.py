# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible chat completions for image understanding pipelines."""

import math
import os
from typing import Literal, cast

from fastapi import APIRouter, HTTPException, Request

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    build_sampling_params,
    process_generation_batch,
    save_image_to_path,
    temp_dir_if_disabled,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageUserParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from sglang.srt.observability.trace import extract_trace_headers

router = APIRouter(prefix="/v1/chat", tags=["chat"])


def _bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _validate_request_features(request: ChatCompletionRequest) -> None:
    """Reject OpenAI features that the first Understanding endpoint omits."""
    if request.n != 1:
        raise _bad_request("BAGEL Understanding supports n=1 only")
    if request.stream:
        raise _bad_request("BAGEL Understanding does not support streaming")
    if request.tools:
        raise _bad_request("BAGEL Understanding does not support tools")
    if request.response_format is not None:
        raise _bad_request("BAGEL Understanding does not support response_format")
    if request.stop is not None:
        raise _bad_request("BAGEL Understanding does not support stop sequences")
    if request.logprobs or request.top_logprobs is not None:
        raise _bad_request("BAGEL Understanding does not support logprobs")
    if request.frequency_penalty != 0 or request.presence_penalty != 0:
        raise _bad_request("BAGEL Understanding does not support token penalties")
    if request.logit_bias:
        raise _bad_request("BAGEL Understanding does not support logit_bias")
    if request.top_p not in (None, 1.0):
        raise _bad_request("BAGEL Understanding does not support top_p")
    if request.top_k not in (None, -1):
        raise _bad_request("BAGEL Understanding does not support top_k")
    if request.min_p not in (None, 0.0) or request.min_tokens != 0:
        raise _bad_request("BAGEL Understanding does not support min_p or min_tokens")
    if request.repetition_penalty not in (None, 1.0):
        raise _bad_request("BAGEL Understanding does not support repetition_penalty")


def _extract_user_input(request: ChatCompletionRequest) -> tuple[str, str]:
    """Extract one image URL and the aggregate non-empty user text."""
    if len(request.messages) != 1:
        raise _bad_request("BAGEL Understanding requires exactly one user turn")

    message = request.messages[0]
    if not isinstance(message, ChatCompletionMessageUserParam):
        raise _bad_request("BAGEL Understanding requires exactly one user turn")
    if not isinstance(message.content, list):
        raise _bad_request(
            "BAGEL Understanding user content must contain text and image_url parts"
        )

    text_parts: list[str] = []
    image_urls: list[str] = []
    for part in message.content:
        if isinstance(part, ChatCompletionMessageContentTextPart):
            text_parts.append(part.text)
        elif isinstance(part, ChatCompletionMessageContentImagePart):
            if part.modalities not in (None, "image"):
                raise _bad_request(
                    "BAGEL Understanding image_url must describe one image"
                )
            image_urls.append(part.image_url.url)
        else:
            raise _bad_request(
                "BAGEL Understanding supports text and image_url content only"
            )

    prompt = "\n".join(text_parts).strip()
    if not prompt:
        raise _bad_request("BAGEL Understanding requires non-empty user text")
    if len(image_urls) != 1:
        raise _bad_request("BAGEL Understanding requires exactly one image_url")

    image_url = image_urls[0].strip()
    if not image_url:
        raise _bad_request("BAGEL Understanding image_url must not be empty")
    return prompt, image_url


def _resolve_max_new_tokens(request: ChatCompletionRequest) -> int | None:
    """Map the current and deprecated OpenAI token limits to BAGEL."""
    max_new_tokens = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens
    )
    if max_new_tokens is not None and max_new_tokens <= 0:
        raise _bad_request("max_completion_tokens must be a positive integer")
    # BAGEL's official max_length counts the internal <|im_start|> iteration;
    # OpenAI limits count only completion content tokens.
    return None if max_new_tokens is None else max_new_tokens + 1


def _resolve_sampling(request: ChatCompletionRequest) -> tuple[bool, float | None]:
    """Map OpenAI temperature semantics to BAGEL's explicit sampling switch."""
    temperature = request.temperature
    if temperature is None:
        return False, None
    if not math.isfinite(temperature) or temperature < 0:
        raise _bad_request("temperature must be finite and non-negative")
    return temperature > 0, temperature


def _resolve_enable_thinking(request: ChatCompletionRequest) -> bool:
    """Resolve BAGEL's reasoning switch from standard and SGLang controls."""
    enable_thinking = request.reasoning_effort not in (None, "none")
    template_kwargs = request.chat_template_kwargs or {}
    for name in ("enable_thinking", "thinking"):
        if name not in template_kwargs:
            continue
        value = template_kwargs[name]
        if not isinstance(value, bool):
            raise _bad_request(f"chat_template_kwargs.{name} must be a boolean")
        enable_thinking = value
        break
    return enable_thinking


@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
) -> ChatCompletionResponse:
    """Run one non-streaming BAGEL image-understanding chat completion.

    Args:
        request: OpenAI chat request containing one user turn, one image URL,
            and non-empty text.
        raw_request: FastAPI request used to propagate distributed trace headers.

    Returns:
        A typed OpenAI chat completion with assistant text and token usage.

    Raises:
        HTTPException: If the request uses an unsupported feature, has invalid
            multimodal content, or the image source cannot be loaded.
        RuntimeError: If the backend does not return exactly one valid text output.

    Example:
        Send a non-streaming ``POST /v1/chat/completions`` request whose user
        content contains one ``text`` part and one ``image_url`` part.
    """
    _validate_request_features(request)
    prompt, image_url = _extract_user_input(request)
    max_new_tokens = _resolve_max_new_tokens(request)
    do_sample, temperature = _resolve_sampling(request)
    enable_thinking = _resolve_enable_thinking(request)

    server_args = get_global_server_args()
    task_type = server_args.pipeline_config.task_type
    if not task_type.is_text_gen() or not task_type.accepts_image_input():
        raise _bad_request("/v1/chat/completions requires an image-to-text pipeline")

    request_id = f"chatcmpl-{generate_request_id()}"
    with temp_dir_if_disabled(server_args.input_save_path) as input_dir:
        # The shared URL loader currently normalizes transport failures to the
        # base Exception type, so this boundary must translate that public contract.
        try:
            image_path = await save_image_to_path(
                image_url,
                os.path.join(input_dir, f"{request_id}_input"),
                # Chat input is untrusted: always prefetch through the bounded,
                # redirect-aware URL policy before it reaches the model worker.
                prefer_remote_source=False,
            )
        except Exception as exc:
            raise _bad_request(f"Failed to process image_url: {exc}") from exc

        try:
            sampling = build_sampling_params(
                request_id,
                prompt=prompt,
                image_path=image_path,
                num_outputs_per_prompt=1,
                num_frames=1,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                enable_thinking=enable_thinking,
                seed=request.seed,
                save_output=False,
                return_file_paths_only=False,
            )
        except (TypeError, ValueError) as exc:
            raise _bad_request(str(exc)) from exc

        batch = prepare_request(
            server_args=server_args,
            sampling_params=sampling,
            external_trace_header=extract_trace_headers(raw_request.headers),
        )
        _, result = await process_generation_batch(async_scheduler_client, batch)

    if result.text_outputs is None or len(result.text_outputs) != 1:
        raise RuntimeError(
            "BAGEL Understanding must return exactly one text generation output"
        )
    text_output = result.text_outputs[0]
    if text_output.finish_reason not in ("stop", "length"):
        raise RuntimeError(
            f"Unsupported BAGEL finish reason: {text_output.finish_reason!r}"
        )

    finish_reason = cast(Literal["stop", "length"], text_output.finish_reason)
    usage = UsageInfo(
        prompt_tokens=text_output.prompt_tokens,
        completion_tokens=text_output.completion_tokens,
        total_tokens=text_output.prompt_tokens + text_output.completion_tokens,
    )
    return ChatCompletionResponse(
        id=request_id,
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text_output.text),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )
