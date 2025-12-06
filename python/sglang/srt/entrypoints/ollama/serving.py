# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0

"""
Ollama-compatible API serving handlers.

This module provides handlers that convert Ollama API requests to SGLang's
internal format and return Ollama-compatible responses.
"""

import time
from datetime import datetime, timezone
from typing import AsyncIterator, Union

import orjson
from fastapi import Request
from fastapi.responses import StreamingResponse

from sglang.srt.entrypoints.ollama.protocol import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaChatStreamResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaGenerateStreamResponse,
    OllamaMessage,
    OllamaModelInfo,
    OllamaShowResponse,
    OllamaTagsResponse,
)
from sglang.srt.managers.io_struct import GenerateReqInput


class OllamaServing:
    """Handler for Ollama-compatible API endpoints."""

    def __init__(self, tokenizer_manager):
        self.tokenizer_manager = tokenizer_manager

    def _get_timestamp(self) -> str:
        """Get current timestamp in Ollama format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _convert_options_to_sampling_params(
        self, options: dict = None, max_tokens: int = None
    ) -> dict:
        """Convert Ollama options to SGLang sampling params."""
        sampling_params = {}

        if options:
            # Map Ollama options to SGLang params
            param_mapping = {
                "temperature": "temperature",
                "top_p": "top_p",
                "top_k": "top_k",
                "num_predict": "max_new_tokens",
                "stop": "stop",
                "presence_penalty": "presence_penalty",
                "frequency_penalty": "frequency_penalty",
                "seed": "seed",
            }
            for ollama_param, sglang_param in param_mapping.items():
                if ollama_param in options:
                    sampling_params[sglang_param] = options[ollama_param]

        # Default max tokens if not specified
        if "max_new_tokens" not in sampling_params:
            sampling_params["max_new_tokens"] = max_tokens or 2048

        return sampling_params

    async def handle_chat(
        self, request: OllamaChatRequest, raw_request: Request
    ) -> Union[OllamaChatResponse, StreamingResponse]:
        """Handle /api/chat endpoint."""
        model_name = self.tokenizer_manager.served_model_name

        # Convert messages to SGLang format
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Apply chat template using tokenizer
        prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        # Convert options to sampling params
        sampling_params = self._convert_options_to_sampling_params(request.options)

        # Create SGLang request with input_ids
        gen_request = GenerateReqInput(
            input_ids=prompt_ids,
            sampling_params=sampling_params,
            stream=request.stream,
        )

        if request.stream:
            return await self._stream_chat_response(
                gen_request, raw_request, model_name
            )
        else:
            return await self._generate_chat_response(
                gen_request, raw_request, model_name
            )

    async def _generate_chat_response(
        self, gen_request: GenerateReqInput, raw_request: Request, model_name: str
    ) -> OllamaChatResponse:
        """Generate non-streaming chat response."""
        start_time = time.time_ns()

        # Get response from tokenizer manager
        response = await self.tokenizer_manager.generate_request(
            gen_request, raw_request
        ).__anext__()

        end_time = time.time_ns()
        total_duration = end_time - start_time

        output_text = response.get("text", "")

        return OllamaChatResponse(
            model=model_name,
            created_at=self._get_timestamp(),
            message=OllamaMessage(role="assistant", content=output_text),
            done=True,
            done_reason="stop",
            total_duration=total_duration,
            prompt_eval_count=response.get("meta_info", {}).get("prompt_tokens", None),
            eval_count=response.get("meta_info", {}).get("completion_tokens", None),
        )

    async def _stream_chat_response(
        self, gen_request: GenerateReqInput, raw_request: Request, model_name: str
    ) -> StreamingResponse:
        """Generate streaming chat response."""

        async def generate_stream() -> AsyncIterator[bytes]:
            previous_text = ""
            async for chunk in self.tokenizer_manager.generate_request(
                gen_request, raw_request
            ):
                text = chunk.get("text", "")
                is_done = chunk.get("meta_info", {}).get("finish_reason") is not None

                # Calculate delta (new text since last chunk)
                delta = text[len(previous_text) :]
                previous_text = text

                if is_done:
                    # Final chunk
                    response = OllamaChatStreamResponse(
                        model=model_name,
                        created_at=self._get_timestamp(),
                        message=OllamaMessage(role="assistant", content=""),
                        done=True,
                        done_reason="stop",
                    )
                else:
                    response = OllamaChatStreamResponse(
                        model=model_name,
                        created_at=self._get_timestamp(),
                        message=OllamaMessage(role="assistant", content=delta),
                        done=False,
                    )

                yield orjson.dumps(response.model_dump()) + b"\n"

        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson",
        )

    async def handle_generate(
        self, request: OllamaGenerateRequest, raw_request: Request
    ) -> Union[OllamaGenerateResponse, StreamingResponse]:
        """Handle /api/generate endpoint."""
        model_name = self.tokenizer_manager.served_model_name

        # Build prompt
        prompt = request.prompt
        if request.system:
            prompt = f"{request.system}\n\n{prompt}"

        # Convert options to sampling params
        sampling_params = self._convert_options_to_sampling_params(request.options)

        # Create SGLang request
        gen_request = GenerateReqInput(
            text=prompt,
            sampling_params=sampling_params,
            stream=request.stream,
        )

        if request.stream:
            return await self._stream_generate_response(
                gen_request, raw_request, model_name
            )
        else:
            return await self._generate_generate_response(
                gen_request, raw_request, model_name
            )

    async def _generate_generate_response(
        self, gen_request: GenerateReqInput, raw_request: Request, model_name: str
    ) -> OllamaGenerateResponse:
        """Generate non-streaming generate response."""
        start_time = time.time_ns()

        response = await self.tokenizer_manager.generate_request(
            gen_request, raw_request
        ).__anext__()

        end_time = time.time_ns()
        total_duration = end_time - start_time

        output_text = response.get("text", "")

        return OllamaGenerateResponse(
            model=model_name,
            created_at=self._get_timestamp(),
            response=output_text,
            done=True,
            done_reason="stop",
            total_duration=total_duration,
            prompt_eval_count=response.get("meta_info", {}).get("prompt_tokens", None),
            eval_count=response.get("meta_info", {}).get("completion_tokens", None),
        )

    async def _stream_generate_response(
        self, gen_request: GenerateReqInput, raw_request: Request, model_name: str
    ) -> StreamingResponse:
        """Generate streaming generate response."""

        async def generate_stream() -> AsyncIterator[bytes]:
            previous_text = ""
            async for chunk in self.tokenizer_manager.generate_request(
                gen_request, raw_request
            ):
                text = chunk.get("text", "")
                is_done = chunk.get("meta_info", {}).get("finish_reason") is not None

                # Calculate delta (new text since last chunk)
                delta = text[len(previous_text) :]
                previous_text = text

                if is_done:
                    response = OllamaGenerateStreamResponse(
                        model=model_name,
                        created_at=self._get_timestamp(),
                        response="",
                        done=True,
                        done_reason="stop",
                    )
                else:
                    response = OllamaGenerateStreamResponse(
                        model=model_name,
                        created_at=self._get_timestamp(),
                        response=delta,
                        done=False,
                    )

                yield orjson.dumps(response.model_dump()) + b"\n"

        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson",
        )

    def get_tags(self) -> OllamaTagsResponse:
        """Handle /api/tags endpoint - list available models."""
        model_name = self.tokenizer_manager.served_model_name
        model_path = self.tokenizer_manager.model_path

        model_info = OllamaModelInfo(
            name=model_name,
            model=model_name,
            modified_at=self._get_timestamp(),
            size=0,  # We don't track model size
            digest="sha256:sglang0000000000000000000000000000000000000000000000000000000000",
            details={
                "format": "sglang",
                "family": (
                    model_name.split("/")[-1] if "/" in model_name else model_name
                ),
                "parameter_size": "unknown",
            },
        )

        return OllamaTagsResponse(models=[model_info])

    def get_show(self, model: str) -> OllamaShowResponse:
        """Handle /api/show endpoint - show model information."""
        model_config = self.tokenizer_manager.model_config

        return OllamaShowResponse(
            modelfile="",
            parameters="",
            template="",  # Template info not easily accessible
            details={
                "format": "sglang",
                "family": model,
                "parameter_size": "unknown",
            },
            model_info={
                "model_path": self.tokenizer_manager.model_path,
                "context_length": model_config.context_len if model_config else None,
                "is_generation": self.tokenizer_manager.is_generation,
            },
        )
