#!/usr/bin/env python3
"""
Simple Python gRPC client to test the demo server.
This helps isolate whether the issue is with the Python server or Rust client.
"""

import asyncio
import grpc
import logging

try:
    import sglang_scheduler_pb2 as pb2
    import sglang_scheduler_pb2_grpc as pb2_grpc
except ImportError as e:
    print(f"Error importing protobuf files: {e}")
    print("Make sure to run the protobuf generation first")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_python_client():
    """Test the Python server with a Python client."""
    
    # Connect to the server
    async with grpc.aio.insecure_channel('127.0.0.1:50051') as channel:
        client = pb2_grpc.SglangSchedulerStub(channel)
        
        try:
            # Test Initialize
            logger.info("Testing Initialize request...")
            init_request = pb2.InitializeRequest(
                client_id="python-test-client",
                client_version="0.1.0",
                mode=pb2.InitializeRequest.Mode.REGULAR
            )
            
            response = await client.Initialize(init_request)
            logger.info(f"Initialize response: success={response.success}")
            logger.info(f"Scheduler version: {response.scheduler_version}")
            
            # Test HealthCheck
            logger.info("Testing HealthCheck request...")
            health_request = pb2.HealthCheckRequest(include_detailed_metrics=False)
            health_response = await client.HealthCheck(health_request)
            logger.info(f"Health check: healthy={health_response.healthy}")
            
            logger.info("Python client test completed successfully!")
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(test_python_client())