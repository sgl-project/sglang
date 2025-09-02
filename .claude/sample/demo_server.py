#!/usr/bin/env python3
"""
Demo gRPC server implementing SGLang scheduler protocol for testing the Rust client.

This is a minimal mock server that implements the basic SGLang scheduler gRPC interface.
It's designed to work with the client_demo.rs example for testing protobuf communication.

Usage:
    python demo_server.py [--port PORT]

Then run the Rust client:
    cargo run --example client_demo --features grpc-client
"""

import asyncio
import argparse
import logging
import time
import random
from typing import AsyncIterator

import grpc
from grpc import aio

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Generate Python protobuf code from the .proto file
# Run this first: python -m grpc_tools.protoc -I src/proto --python_out=examples --grpc_python_out=examples src/proto/sglang_scheduler.proto

try:
    import sglang_scheduler_pb2 as pb2
    import sglang_scheduler_pb2_grpc as pb2_grpc
    logger.info("Successfully imported protobuf modules")
except ImportError as e:
    print(f"Error: Python protobuf files not found! {e}")
    print("Please generate them first with:")
    print("python -m grpc_tools.protoc -I ../src/proto --python_out=. --grpc_python_out=. ../src/proto/sglang_scheduler.proto")
    print("You may need to install grpcio-tools: pip install grpcio grpcio-tools")
    exit(1)

class MockSglangScheduler(pb2_grpc.SglangSchedulerServicer):
    """Mock implementation of SGLang scheduler for testing."""
    
    def __init__(self):
        self.clients = {}
        self.request_count = 0
        self.start_time = time.time()
        logger.info("MockSglangScheduler initialized")
        
    async def Initialize(self, request: pb2.InitializeRequest, context) -> pb2.InitializeResponse:
        """Initialize client connection."""
        logger.info(f"Initialize request from client: {request.client_id}")
        logger.info(f"Client version: {request.client_version}, mode: {request.mode}")
        
        # Store client info
        self.clients[request.client_id] = {
            "version": request.client_version,
            "mode": request.mode,
            "connected_at": time.time()
        }
        
        # Mock model info
        model_info = pb2.ModelInfo(
            model_name="Mock-Llama-3-8B-Instruct",
            max_context_length=8192,
            vocab_size=128256,
            supports_tool_calling=True,
            supports_vision=False,
            special_tokens=["<|begin_of_text|>", "<|end_of_text|>"],
            model_type="llama",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            tokenizer_type="llama",
            eos_token_ids=[128001, 128009],
            pad_token_id=128001,
            bos_token_id=128000
        )
        
        # Mock server capabilities
        capabilities = pb2.ServerCapabilities(
            continuous_batching=True,
            disaggregated_serving=False,
            speculative_decoding=False,
            max_batch_size=64,
            max_num_batched_tokens=4096,
            max_prefill_tokens=2048,
            attention_backend="flashinfer",
            supports_lora=True,
            supports_grammar=True,
            supports_multimodal=False,
            supported_modalities=[],
            supports_custom_logit_processor=False,
            supports_session=True,
            num_gpus=1,
            gpu_type="H100",
            total_gpu_memory=80000000000,  # 80GB
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        
        return pb2.InitializeResponse(
            success=True,
            scheduler_version="0.3.5-mock",
            model_info=model_info,
            capabilities=capabilities
        )
    
    async def Generate(self, request: pb2.GenerateRequest, context) -> AsyncIterator[pb2.GenerateResponse]:
        """Generate text stream."""
        logger.info(f"Generate request: {request.request_id}")
        
        self.request_count += 1
        
        # Extract input text from the oneof field
        input_text = ""
        if request.HasField('input'):
            which_input = request.WhichOneof('input')
            if which_input == 'text':
                input_text = request.text
            elif which_input == 'tokenized':
                input_text = request.tokenized.original_text if request.tokenized else ""
        else:
            input_text = "No input provided"
            
        logger.info(f"Input text: '{input_text[:100]}...' (truncated)")
        
        # Get sampling params
        sampling_params = request.sampling_params
        max_tokens = sampling_params.max_new_tokens if sampling_params else 50
        temperature = sampling_params.temperature if sampling_params else 0.7
        
        # Check if structured generation is requested
        is_structured = False
        if sampling_params and sampling_params.HasField('constraint'):
            constraint_type = sampling_params.WhichOneof('constraint')
            if constraint_type == 'json_schema':
                is_structured = True
                logger.info("JSON schema constraint detected - generating structured output")
            elif constraint_type == 'regex':
                logger.info(f"Regex constraint detected: {sampling_params.regex}")
            elif constraint_type == 'ebnf_grammar':
                logger.info(f"EBNF grammar constraint detected")
        
        # Mock token generation
        if is_structured:
            # Generate a mock JSON response
            mock_tokens = [
                '{"name":', ' "', 'Alice', '", "', 'age', '":', ' 25', ', "', 'city', '":', ' "', 'Paris', '"}'
            ]
        else:
            # Generate mock completion for the input
            if "capital of France" in input_text.lower():
                mock_tokens = [" Paris", ",", " which", " is", " located", " in", " the", " north", " of", " France", "."]
            else:
                mock_tokens = [" This", " is", " a", " mock", " response", " from", " the", " demo", " server", "."]
        
        # Limit tokens based on max_new_tokens
        mock_tokens = mock_tokens[:max_tokens]
        
        # Stream tokens with realistic timing
        prompt_tokens = len(input_text.split())  # Rough estimate
        completion_tokens = 0
        
        start_time = time.time()
        
        for i, token in enumerate(mock_tokens):
            completion_tokens += 1
            
            # Simulate token generation delay based on temperature (lower = faster)
            delay = max(0.01, 0.05 + (temperature - 0.5) * 0.1 + random.uniform(-0.02, 0.02))
            await asyncio.sleep(delay)
            
            # Create streaming chunk
            chunk = pb2.GenerateStreamChunk(
                token_id=1000 + i,  # Mock token ID
                text=token,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=0,
                generation_time=time.time() - start_time,
                queue_time=5  # Mock queue time in ms
            )
            
            yield pb2.GenerateResponse(
                request_id=request.request_id,
                chunk=chunk
            )
        
        # Send completion response
        total_time = time.time() - start_time
        tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
        
        complete = pb2.GenerateComplete(
            output_ids=list(range(1000, 1000 + completion_tokens)),
            output_text=''.join(mock_tokens),
            finish_reason=pb2.GenerateComplete.STOP,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=0,
            total_generation_time=total_time,
            time_to_first_token=0.05,  # Mock TTFT
            tokens_per_second=tokens_per_second
        )
        
        yield pb2.GenerateResponse(
            request_id=request.request_id,
            complete=complete
        )
        
        logger.info(f"Generation completed: {completion_tokens} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
    
    async def Embed(self, request: pb2.EmbedRequest, context) -> pb2.EmbedResponse:
        """Generate embeddings (mock)."""
        logger.info(f"Embed request: {request.request_id}")
        
        # Mock embedding generation
        embedding_dim = 4096
        mock_embedding = [random.uniform(-1.0, 1.0) for _ in range(embedding_dim)]
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        complete = pb2.EmbedComplete(
            embedding=mock_embedding,
            prompt_tokens=len(request.text.split()) if hasattr(request, 'text') else 0,
            cached_tokens=0,
            embedding_dim=embedding_dim,
            generation_time=0.1
        )
        
        return pb2.EmbedResponse(
            request_id=request.request_id,
            complete=complete
        )
    
    async def HealthCheck(self, request: pb2.HealthCheckRequest, context) -> pb2.HealthCheckResponse:
        """Health check endpoint."""
        logger.info("Health check request")
        
        uptime = time.time() - self.start_time
        
        return pb2.HealthCheckResponse(
            healthy=True,
            num_requests_running=random.randint(0, 3),
            num_requests_waiting=random.randint(0, 2),
            gpu_cache_usage=random.uniform(0.3, 0.8),
            gpu_memory_usage=random.uniform(0.6, 0.9),
            kv_cache_total_blocks=1000,
            kv_cache_used_blocks=random.randint(400, 800),
            kv_cache_hit_rate=random.uniform(0.7, 0.95),
            generation_throughput=random.uniform(80, 120),  # tokens/sec
            average_queue_time=random.uniform(0.01, 0.1),   # seconds
            average_generation_time=random.uniform(0.02, 0.08),  # seconds
            cpu_usage=random.uniform(0.2, 0.6),
            memory_usage=random.randint(8000000000, 12000000000),  # bytes
            num_prefill_requests=0,
            num_decode_requests=self.request_count
        )
    
    async def Abort(self, request: pb2.AbortRequest, context) -> pb2.AbortResponse:
        """Abort a request."""
        logger.info(f"Abort request: {request.request_id} - {request.reason}")
        
        return pb2.AbortResponse(
            success=True,
            message=f"Request {request.request_id} aborted successfully"
        )
    
    async def FlushCache(self, request: pb2.FlushCacheRequest, context) -> pb2.FlushCacheResponse:
        """Flush cache."""
        logger.info(f"Flush cache request: flush_all={request.flush_all}")
        
        # Mock cache flush
        entries_flushed = random.randint(50, 200) if request.flush_all else random.randint(10, 50)
        memory_freed = entries_flushed * random.randint(1000000, 5000000)  # bytes
        
        await asyncio.sleep(0.05)  # Simulate flush time
        
        return pb2.FlushCacheResponse(
            success=True,
            num_entries_flushed=entries_flushed,
            memory_freed=memory_freed,
            message=f"Flushed {entries_flushed} cache entries, freed {memory_freed} bytes"
        )

async def serve(port: int):
    """Start the mock gRPC server."""
    # Create server with explicit options
    server = aio.server(options=[
        ('grpc.keepalive_time_ms', 10000),
        ('grpc.keepalive_timeout_ms', 5000),
        ('grpc.keepalive_permit_without_calls', True),
        ('grpc.http2.max_pings_without_data', 0),
        ('grpc.http2.min_time_between_pings_ms', 10000),
        ('grpc.http2.min_ping_interval_without_data_ms', 300000)
    ])
    
    servicer = MockSglangScheduler()
    pb2_grpc.add_SglangSchedulerServicer_to_server(servicer, server)
    logger.info("Added MockSglangScheduler servicer to server")
    
    # Bind to IPv4 address (what the Rust client connects to)
    listen_addr = f'0.0.0.0:{port}'
    
    try:
        actual_port = server.add_insecure_port(listen_addr)
        logger.info(f"Successfully bound to {listen_addr}, actual port: {actual_port}")
    except Exception as e:
        logger.error(f"Failed to bind to {listen_addr}: {e}")
        raise
    
    logger.info(f"Starting mock SGLang scheduler server on port {port}")
    await server.start()
    
    try:
        logger.info("Server started successfully! Waiting for requests...")
        logger.info(f"You can now run the Rust client with: cargo run --example client_demo --features grpc-client")
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        await server.stop(5)

def main():
    parser = argparse.ArgumentParser(description='Mock SGLang scheduler gRPC server for testing')
    parser.add_argument('--port', type=int, default=50051, help='Port to listen on (default: 50051)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        asyncio.run(serve(args.port))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())