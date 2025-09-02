#!/usr/bin/env python3
"""
Minimal gRPC server using a simple service to test basic connectivity.
"""

import grpc
from concurrent import futures
import logging

# Create a minimal protobuf definition
import tempfile
import os
import subprocess
import sys

# Create minimal proto file
proto_content = '''
syntax = "proto3";

package test;

service TestService {
  rpc SayHello(HelloRequest) returns (HelloResponse);
}

message HelloRequest {
  string name = 1;
}

message HelloResponse {
  string message = 1;
}
'''

def create_minimal_proto():
    """Create and compile a minimal proto file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.proto', delete=False) as f:
        f.write(proto_content)
        proto_file = f.name
    
    try:
        # Generate Python files
        cmd = [
            sys.executable, '-m', 'grpc_tools.protoc',
            f'--proto_path={os.path.dirname(proto_file)}',
            f'--python_out=.',
            f'--grpc_python_out=.',
            proto_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Protoc failed: {result.stderr}")
            return None
            
        # Import the generated modules
        proto_name = os.path.basename(proto_file).replace('.proto', '_pb2')
        grpc_name = proto_name.replace('_pb2', '_pb2_grpc')
        
        # The files should be generated with temp file names, let's find them
        base_name = os.path.basename(proto_file).replace('.proto', '')
        
        return base_name
        
    finally:
        os.unlink(proto_file)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Simple test: start server on port 30000 and see if we can connect
    def serve():
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # Don't add any servicers, just test if the server starts
        listen_addr = '0.0.0.0:30000'
        server.add_insecure_port(listen_addr)
        
        logger.info(f"Starting minimal server on {listen_addr}")
        server.start()
        logger.info("Minimal server started - testing basic connectivity")
        
        try:
            import time
            time.sleep(5)  # Run for 5 seconds
            logger.info("Minimal server test completed")
        except KeyboardInterrupt:
            logger.info("Server interrupted")
        finally:
            server.stop(5)
    
    serve()