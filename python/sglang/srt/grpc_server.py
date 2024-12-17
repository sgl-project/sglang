import traceback
from typing import Any, AsyncGenerator, Callable, Dict

import grpc

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.proto import completion_pb2, completion_pb2_grpc


class CompletionServicer(completion_pb2_grpc.CompletionServiceServicer):
    def __init__(
        self,
        generate_request: Callable[
            [GenerateReqInput], AsyncGenerator[Dict[str, Any], None]
        ],
    ):
        self.generate_request = generate_request

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
            async for content in self.generate_request(adapted_request):
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
