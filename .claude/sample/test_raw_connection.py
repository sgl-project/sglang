#!/usr/bin/env python3
"""
Test raw connection to the gRPC server to verify it's accepting HTTP/2 connections.
"""

import socket
import time

def test_raw_connection():
    """Test basic TCP connection."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        
        print("Attempting TCP connection to 127.0.0.1:30000...")
        start_time = time.time()
        sock.connect(('127.0.0.1', 30000))
        connect_time = time.time() - start_time
        print(f"TCP connection successful in {connect_time:.3f}s")
        
        # Send HTTP/2 connection preface
        preface = b'PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n'
        print("Sending HTTP/2 connection preface...")
        sock.send(preface)
        
        # Try to read response
        sock.settimeout(2)
        try:
            response = sock.recv(1024)
            print(f"Received response: {response}")
        except socket.timeout:
            print("No response received within timeout")
        
        sock.close()
        print("Test completed")
        
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_raw_connection()