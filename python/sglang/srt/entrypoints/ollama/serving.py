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

    def _convert_options_to_sampling_params(self, options: dict = None) -> dict:
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

        # Set a reasonable default for max_new_tokens if not specified
        # Ollama users typically expect longer responses than SGLang's default (128)
        if "max_new_tokens" not in sampling_params:
            sampling_params["max_new_tokens"] = 2048

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

        # Handle empty prompt - Ollama CLI sends empty requests on initialization
        if not prompt or not prompt.strip():
            empty_response = OllamaGenerateResponse(
                model=model_name,
                created_at=self._get_timestamp(),
                response="",
                done=True,
                done_reason="stop",
            )
            if request.stream:
                # Return streaming response with done=True
                async def empty_stream() -> AsyncIterator[bytes]:
                    yield orjson.dumps(empty_response.model_dump()) + b"\n"

                return StreamingResponse(
                    empty_stream(),
                    media_type="application/x-ndjson",
                )
            return empty_response

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

        # Extract model family from model name
        model_family = model.split("/")[-1] if "/" in model else model
        # Remove common suffixes to get base family
        for suffix in ["-Instruct", "-Chat", "-Base"]:
            if model_family.endswith(suffix):
                model_family = model_family[: -len(suffix)]
                break

        # Build context length info
        context_len = model_config.context_len if model_config else 4096

        return OllamaShowResponse(
            license="",  # License info not available from SGLang
            modelfile=f"FROM {model}\nPARAMETER num_ctx {context_len}\n",
            parameters=f"num_ctx {context_len}",
            template="",  # Template info not easily accessible
            modified_at=self._get_timestamp(),
            details={
                "parent_model": "",
                "format": "sglang",
                "family": model_family,
                "families": [model_family],
                "parameter_size": "unknown",
                "quantization_level": "",
            },
            model_info={
                "general.architecture": model_family,
                "general.name": model,
                "general.parameter_count": 0,
                f"{model_family}.context_length": context_len,
                f"{model_family}.block_count": 0,
                f"{model_family}.embedding_length": 0,
                f"{model_family}.attention.head_count": 0,
            },
            capabilities=["completion"],
        )
