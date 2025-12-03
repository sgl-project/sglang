"""
Profile kernel shapes by making requests to a running SGLang server.

This script connects to an existing SGLang server and makes requests while
logging all kernel shapes. This is the RECOMMENDED way to profile a server
since it doesn't require modifying the server process.

Usage:
    # Step 1: Start your server normally (in another terminal)
    python -m sglang.launch_server \\
        --model-path deepseek-ai/DeepSeek-V3 \\
        --tp-size 8 \\
        --port 30000
    
    # Step 2: Run this profiler to capture shapes
    python profile_server_shapes.py \\
        --host http://localhost:30000 \\
        --output-file deepseek_shapes.jsonl \\
        --num-requests 5
    
    # Step 3: Analyze the results
    python analyze_shapes.py deepseek_shapes.jsonl

Note: This approach profiles the CLIENT side operations (tokenization, 
deserialization, etc.). For profiling the actual MODEL inference on the server,
use the engine-based approach with offline_batch_inference.py
"""

import argparse
import json
import sys
import time

import requests

# Add profiler to path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_shape_logger import CompactShapeLogger, ShapeLogger


def test_server_connection(base_url: str) -> bool:
    """Test if server is reachable."""
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return False


def make_chat_request(base_url: str, prompt: str, max_tokens: int = 100) -> dict:
    """Make a chat completion request to the server."""
    url = f"{base_url}/v1/chat/completions"
    
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }
    
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def make_completion_request(base_url: str, prompt: str, max_tokens: int = 100) -> dict:
    """Make a completion request to the server."""
    url = f"{base_url}/v1/completions"
    
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }
    
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="Profile kernel shapes from SGLang server requests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:30000",
        help="Server URL (default: http://localhost:30000)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="server_profile_shapes.jsonl",
        help="Output file for shape logs",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=5,
        help="Number of requests to make (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens per request (default: 100)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        default=True,
        help="Use compact logging (default: True)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print shapes to console",
    )
    parser.add_argument(
        "--request-type",
        choices=["chat", "completion"],
        default="chat",
        help="Type of request to make (default: chat)",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("SGLang Server Shape Profiler (Client-side)")
    print("="*80)
    print(f"Server: {args.host}")
    print(f"Output: {args.output_file}")
    print(f"Requests: {args.num_requests}")
    print(f"Max tokens: {args.max_tokens}")
    print("="*80)
    
    # Test connection
    print("\nTesting server connection...")
    if not test_server_connection(args.host):
        print("ERROR: Cannot connect to server!")
        print(f"Make sure server is running at {args.host}")
        print("\nStart server with:")
        print("  python -m sglang.launch_server --model-path MODEL --tp-size N")
        sys.exit(1)
    
    print("✓ Server is reachable\n")
    
    # Sample prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Explain quantum computing in simple terms:",
        "Write a haiku about mountains:",
        "What is the meaning of life?",
        "Describe the process of photosynthesis:",
    ]
    
    # Select prompts
    selected_prompts = prompts[:args.num_requests]
    
    # Choose logger
    LoggerClass = CompactShapeLogger if args.compact else ShapeLogger
    
    print("Starting shape logging...")
    print("(Note: This profiles CLIENT-side operations, not server-side model inference)\n")
    
    # Run requests with shape logging
    try:
        with LoggerClass(args.output_file, verbose=args.verbose) as logger:
            for i, prompt in enumerate(selected_prompts, 1):
                print(f"[{i}/{len(selected_prompts)}] Processing: {prompt[:50]}...")
                
                start_time = time.time()
                
                try:
                    if args.request_type == "chat":
                        response = make_chat_request(args.host, prompt, args.max_tokens)
                    else:
                        response = make_completion_request(args.host, prompt, args.max_tokens)
                    
                    elapsed = time.time() - start_time
                    
                    # Extract response text
                    if args.request_type == "chat":
                        text = response["choices"][0]["message"]["content"]
                    else:
                        text = response["choices"][0]["text"]
                    
                    print(f"  ✓ Completed in {elapsed:.2f}s")
                    print(f"  Response: {text[:80]}{'...' if len(text) > 80 else ''}\n")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}\n")
        
        # Print summary
        summary = logger.get_summary()
        print("\n" + "="*80)
        print("Profiling Complete")
        print("="*80)
        print(f"Total operations captured: {summary['total_operations']}")
        print(f"Unique operations: {summary['unique_operations']}")
        
        if summary['total_operations'] > 0:
            print(f"\nTop 5 most frequent operations:")
            sorted_ops = sorted(
                summary['operation_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for op_name, count in sorted_ops:
                print(f"  {count:6d} : {op_name}")
        
        print(f"\nShapes written to: {args.output_file}")
        print("\nNOTE: These are CLIENT-SIDE operations (tokenization, etc.)")
        print("For SERVER-SIDE model inference profiling, use:")
        print("  python qwen_shape_logger.py --model YOUR_MODEL")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise


if __name__ == "__main__":
    main()
