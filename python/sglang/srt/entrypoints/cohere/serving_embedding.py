# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Translate Cohere embedding requests to the shared SGLang embedding engine."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import jinja2
from fastapi import HTTPException, Request
from transformers.utils import cached_file

from sglang.srt.entrypoints.cohere.encoding import build_embed_response
from sglang.srt.entrypoints.cohere.protocol import (
    CohereEmbedRequest,
    create_error_response,
)
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.parser.template_manager import TemplateManager


def _string_instructions(value) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(item, str)}


@lru_cache(maxsize=32)
def _load_sentence_transformer_prompts(
    model_path: str, revision: Optional[str]
) -> dict[str, str]:
    # Optional metadata, resolved at the same revision as the model. Only load
    # it when a client asks for input_type and the HF config has no instructions.
    try:
        path = cached_file(
            model_path,
            "config_sentence_transformers.json",
            revision=revision,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if path is None:
            return {}
        with open(path) as file:
            config = json.load(file)
        return (
            _string_instructions(config.get("prompts"))
            if isinstance(config, dict)
            else {}
        )
    except (OSError, ValueError):
        return {}


class CohereServingEmbedding(OpenAIServingBase):
    """Use the common request lifecycle with a separate Cohere wire protocol."""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager
        self._task_instructions = _string_instructions(
            tokenizer_manager.model_config.hf_config.to_dict().get("task_instructions")
        )

    def _request_id_prefix(self) -> str:
        return "embd-"

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
        param: Optional[str] = None,
    ):
        # The shared lifecycle uses legacy OpenAI error names in some branches.
        if status_code == 400:
            err_type = "BadRequestError"
        elif status_code == 404:
            err_type = "NotFoundError"
        return create_error_response(message, status_code, err_type, param)

    def _validate_request(self, request: CohereEmbedRequest) -> Optional[str]:
        if self.tokenizer_manager.is_generation:
            return "This model does not support embeddings; launch with --is-embedding"
        if request.has_images and not self.tokenizer_manager.model_config.is_multimodal:
            return "This model does not support image inputs"
        if (
            request.output_dimension is not None
            and request.output_dimension % 8
            and {"binary", "ubinary"}.intersection(request.embedding_types or [])
        ):
            return "output_dimension must be a multiple of 8 for binary embedding types"
        return None

    def _task_instruction(self, input_type: Optional[str]) -> Optional[str]:
        if input_type is None:
            return None
        instructions = self._task_instructions
        if not instructions:
            config = self.tokenizer_manager.model_config
            instructions = _load_sentence_transformer_prompts(
                config.model_path, config.revision
            )
        if input_type not in instructions:
            supported = ", ".join(sorted(instructions)) or "none"
            raise ValueError(
                f"Unsupported input_type {input_type!r}; supported input types: {supported}"
            )
        return instructions[input_type]

    def _has_chat_template(self) -> bool:
        return bool(
            self.template_manager.chat_template_name
            or self.template_manager.python_chat_template
            or self.tokenizer_manager.tokenizer.chat_template
        )

    def _render_content(self, content: list[dict], instruction: Optional[str]) -> str:
        messages = []
        if instruction:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": instruction}]}
            )
        messages.append({"role": "user", "content": content})

        # Preserve SGLang's precedence for explicitly selected conversation
        # templates. Its chat helper appends an empty assistant turn for
        # generation; embedding encodes only the supplied system/user turns.
        if self.template_manager.chat_template_name is not None:
            conv = generate_chat_conv(
                ChatCompletionRequest(
                    model=self.tokenizer_manager.served_model_name, messages=messages
                ),
                self.template_manager.chat_template_name,
            )
            conv.messages.pop()
            return conv.get_prompt()

        if not self._has_chat_template():
            raise ValueError(
                "images and inputs require an embedding-compatible chat template"
            )
        content_format = self.template_manager.jinja_template_content_format
        if content_format == "string" and any(
            part["type"] == "image_url" for part in content
        ):
            raise ValueError(
                "The selected string chat template cannot represent image inputs"
            )
        messages = [
            process_content_for_template_format(
                message,
                content_format,
                image_data=[],
                video_data=[],
                audio_data=[],
                modalities=[],
            )
            for message in messages
        ]
        try:
            return self.tokenizer_manager.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except (jinja2.TemplateError, TypeError, KeyError, AttributeError) as error:
            raise ValueError(
                f"Failed to render embedding chat template: {error}"
            ) from error

    def _convert_to_internal_request(
        self, request: CohereEmbedRequest, raw_request: Request = None
    ) -> tuple[EmbeddingReqInput, CohereEmbedRequest]:
        manager = self.tokenizer_manager
        model = request.model or manager.served_model_name
        base_model = (
            model
            if model == manager.model_path
            else self._parse_model_parameter(model)[0]
        )
        if base_model not in (manager.served_model_name, manager.model_path):
            raise HTTPException(
                status_code=404, detail=f"The model {model!r} does not exist"
            )
        if manager.tokenizer is None:
            raise ValueError("/v2/embed requires an initialized tokenizer")
        instruction = self._task_instruction(request.input_type)
        image_data = None
        add_special_tokens = None

        if request.texts is not None and not (
            instruction and self._has_chat_template()
        ):
            # vLLM's texts path leaves ordinary text alone. A template is used
            # only for a nonempty task instruction; otherwise prefix exactly.
            prompts = [(instruction or "") + text for text in request.texts]
        else:
            if request.texts is not None:
                contents = [[{"type": "text", "text": text}] for text in request.texts]
            elif request.images is not None:
                contents = [
                    [{"type": "image_url", "image_url": {"url": url}}]
                    for url in request.images
                ]
            else:
                contents = [
                    [part.model_dump(exclude_none=True) for part in item.content]
                    for item in request.inputs
                ]
            prompts = [
                self._render_content(content, instruction) for content in contents
            ]
            # A nested list preserves both the sample boundary and image order.
            # None marks a text-only sample in a mixed batch.
            if request.has_images:
                image_data = [
                    [
                        part["image_url"]["url"]
                        for part in content
                        if part["type"] == "image_url"
                    ]
                    or None
                    for content in contents
                ]
            add_special_tokens = False

        adapted = EmbeddingReqInput(
            text=prompts,
            image_data=image_data,
            dimensions=request.output_dimension,
            priority=request.priority,
            lora_path=(
                None
                if model == manager.model_path
                else self._resolve_lora_path(model, None)
            ),
            embedding_truncate=request.truncate,
            embedding_max_tokens=request.max_tokens,
            embedding_add_special_tokens=add_special_tokens,
        )
        return adapted, request

    # /v2/embed has no stream field, so the base class always takes this path.
    async def _handle_non_streaming_request(
        self,
        adapted_request: EmbeddingReqInput,
        request: CohereEmbedRequest,
        raw_request: Request,
    ):
        result = await self.tokenizer_manager.generate_request(
            adapted_request, raw_request
        ).__anext__()
        if not isinstance(result, list):
            result = [result]
        return build_embed_response(request, result)
