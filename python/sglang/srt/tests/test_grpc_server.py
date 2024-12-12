import asyncio
from typing import AsyncGenerator

import grpc
import pytest

from sglang.srt.proto import completion_pb2, completion_pb2_grpc
from sglang.srt.server import launch_server
from sglang.srt.server_args import ServerArgs


async def test_grpc_completion():
    # Initialize server args with test configuration
    server_args = ServerArgs(
        model_path="meta-llama/Llama-2-7b-chat-hf",
        host="localhost",
        port=12345,
        grpc_port=50051,
        load_format="dummy",  # Use dummy weights for testing
    )

    # Start the server
    server_task = asyncio.create_task(launch_server(server_args))
    await asyncio.sleep(5)  # Wait for server to start

    try:
        # Create gRPC channel and stub
        async with grpc.aio.insecure_channel("localhost:50051") as channel:
            stub = completion_pb2_grpc.CompletionServiceStub(channel)

            # Create completion request
            request = completion_pb2.CompletionRequest(
                model="meta-llama/Llama-2-7b-chat-hf",
                prompt="Hello, how are you?",
                max_tokens=10,
                temperature=0.7,
                top_p=1.0,
                stream=True,
            )

            # Make streaming request
            responses = []
            async for response in stub.Complete(request):
                responses.append(response)

            # Verify responses
            assert len(responses) > 0
            assert all(isinstance(r.text, str) for r in responses)
            assert responses[-1].finished == True
            assert responses[-1].usage.prompt_tokens > 0
            assert responses[-1].usage.completion_tokens > 0
            assert responses[-1].usage.total_tokens == (
                responses[-1].usage.prompt_tokens
                + responses[-1].usage.completion_tokens
            )

    finally:
        # Cleanup
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_grpc_error_handling():
    # Initialize server args with test configuration
    server_args = ServerArgs(
        model_path="meta-llama/Llama-2-7b-chat-hf",
        host="localhost",
        port=12346,
        grpc_port=50052,
        load_format="dummy",
    )

    # Start the server
    server_task = asyncio.create_task(launch_server(server_args))
    await asyncio.sleep(5)  # Wait for server to start

    try:
        # Create gRPC channel and stub
        async with grpc.aio.insecure_channel("localhost:50052") as channel:
            stub = completion_pb2_grpc.CompletionServiceStub(channel)

            # Test invalid temperature
            with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                request = completion_pb2.CompletionRequest(
                    prompt="Hello", temperature=-1.0  # Invalid temperature
                )
                async for _ in stub.Complete(request):
                    pass

            assert exc_info.value.code() == grpc.StatusCode.INTERNAL

            # Test empty prompt
            with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                request = completion_pb2.CompletionRequest(
                    prompt="", temperature=0.7  # Empty prompt
                )
                async for _ in stub.Complete(request):
                    pass

            assert exc_info.value.code() == grpc.StatusCode.INTERNAL

    finally:
        # Cleanup
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(test_grpc_completion())
