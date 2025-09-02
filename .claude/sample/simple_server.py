#!/usr/bin/env python3
"""
Simple synchronous gRPC server for testing.
This uses the standard (non-async) gRPC Python API which might be more reliable.
"""

import time
import logging
import argparse
from concurrent import futures

import grpc

try:
    import sglang_scheduler_pb2 as pb2
    import sglang_scheduler_pb2_grpc as pb2_grpc
except ImportError as e:
    print(f"Error importing protobuf files: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSglangScheduler(pb2_grpc.SglangSchedulerServicer):
    """Simple synchronous implementation."""
    
    def __init__(self):
        logger.info("SimpleSglangScheduler initialized")
    
    def Initialize(self, request, context):
        """Initialize client connection."""
        logger.info(f"Initialize request from client: {request.client_id}")
        logger.info(f"Client version: {request.client_version}, mode: {request.mode}")
        
        # Mock model info
        model_info = pb2.ModelInfo(
            model_name="Simple-Mock-Model",
            max_context_length=8192,
            vocab_size=128256,
            supports_tool_calling=True,
            supports_vision=False
        )
        
        # Mock server capabilities
        capabilities = pb2.ServerCapabilities(
            continuous_batching=True,
            max_batch_size=64
        )
        
        return pb2.InitializeResponse(
            success=True,
            scheduler_version="0.1.0-simple",
            model_info=model_info,
            capabilities=capabilities
        )
    
    def HealthCheck(self, request, context):
        """Health check."""
        logger.info("Health check request")
        return pb2.HealthCheckResponse(
            healthy=True,
            num_requests_running=0,
            num_requests_waiting=0,
            gpu_cache_usage=0.5,
            gpu_memory_usage=0.7
        )

def serve(port):
    """Start the simple gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_SglangSchedulerServicer_to_server(SimpleSglangScheduler(), server)
    
    listen_addr = f'0.0.0.0:{port}'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"Starting simple gRPC server on {listen_addr}")
    server.start()
    logger.info("Server started successfully!")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=30000)
    args = parser.parse_args()
    
    serve(args.port)