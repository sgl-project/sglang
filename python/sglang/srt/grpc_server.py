import logging
import traceback
from concurrent import futures
from typing import AsyncGenerator, Optional

import grpc

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.proto import completion_pb2, completion_pb2_grpc


class CompletionServicer(completion_pb2_grpc.CompletionServiceServicer):
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager

    async def Complete(
        self,
        request: completion_pb2.CompletionRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[completion_pb2.CompletionResponse, None]:
        try:
            # Convert gRPC request to internal format
            adapted_request = GenerateReqInput(
                text=request.prompt,
                sampling_params={
                    "max_new_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                    "min_p": request.min_p,
                    "frequency_penalty": request.frequency_penalty,
                    "presence_penalty": request.presence_penalty,
                    "stop": list(request.stop),
                    "ignore_eos": request.ignore_eos,
                },
                stream=request.stream,
            )

            # Process request through tokenizer manager
            async for content in self.tokenizer_manager.generate_request(
                adapted_request
            ):
                # Create response for each token/chunk
                response = completion_pb2.CompletionResponse(
                    text=content["text"],  # Send full text so far
                    finished=False,  # Not finished until last message
                    usage=completion_pb2.Usage(
                        prompt_tokens=content["meta_info"]["prompt_tokens"],
                        completion_tokens=content["meta_info"]["completion_tokens"],
                        # TODO: fix this
                        # total_tokens=content["meta_info"]["total_tokens"],
                    ),
                )
                yield response

            # Send final response with finished flag
            final_response = completion_pb2.CompletionResponse(
                text=content["text"],  # Final complete text
                finished=True,
                usage=completion_pb2.Usage(
                    prompt_tokens=content["meta_info"]["prompt_tokens"],
                    completion_tokens=content["meta_info"]["completion_tokens"],
                    # TODO: fix this
                    # total_tokens=content["meta_info"]["total_tokens"],
                ),
            )
            yield final_response

        except Exception as e:
            # Handle errors consistently
            error_msg = f"Error in gRPC Complete: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            await context.abort(grpc.StatusCode.INTERNAL, error_msg)


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
