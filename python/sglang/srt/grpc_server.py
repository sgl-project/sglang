import logging
from concurrent import futures
from typing import Optional

import grpc

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.proto import completion_pb2, completion_pb2_grpc


class CompletionServicer(completion_pb2_grpc.CompletionServiceServicer):
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager

    async def Complete(self, request, context):
        try:
            # Convert request to sampling params
            sampling_params = {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "min_p": request.min_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stop": list(request.stop),
                "ignore_eos": request.ignore_eos,
            }

            # Create GenerateReqInput
            generate_request = GenerateReqInput(
                text=request.prompt,  # Pass as text instead of prompt
                sampling_params=sampling_params,
                stream=request.stream,
            )

            # Generate completion
            async for chunk in self.tokenizer_manager.generate_request(
                generate_request, None
            ):
                yield completion_pb2.CompletionResponse(
                    text=chunk.get("text", ""),
                    finished=chunk.get("finished", False),
                    usage=completion_pb2.Usage(
                        prompt_tokens=chunk.get("usage", {}).get("prompt_tokens", 0),
                        completion_tokens=chunk.get("usage", {}).get(
                            "completion_tokens", 0
                        ),
                        total_tokens=chunk.get("usage", {}).get("total_tokens", 0),
                    ),
                )

        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


def serve_grpc(
    tokenizer_manager: TokenizerManager,
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: Optional[int] = None,
):
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )

    completion_pb2_grpc.add_CompletionServiceServicer_to_server(
        CompletionServicer(tokenizer_manager), server
    )

    server.add_insecure_port(f"{host}:{port}")
    return server
