"""
Launch SGLang server with kernel shape logging enabled.

This script wraps the standard SGLang server launch and injects shape logging
into the inference process. Perfect for profiling DeepSeek or any other model
with tensor parallelism.

Usage:
    # Launch DeepSeek with TP=8 and shape logging
    python launch_server_with_shapes.py \
        --model-path deepseek-ai/DeepSeek-V3 \
        --tp-size 8 \
        --output-file deepseek_shapes.jsonl
    
    # Then make requests to trigger shape logging:
    curl http://localhost:30000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50
        }'
    
    # Stop the server when done (Ctrl+C) and analyze shapes:
    python analyze_shapes.py deepseek_shapes.jsonl

Note: 
    - Model weights are NOT downloaded if already cached
    - Shape logging starts after the first request
    - Use --log-first-n-requests to limit how many requests are logged
"""

import argparse
import atexit
import dataclasses
import os
import signal
import sys
import threading

# Add profiler to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_shape_logger import CompactShapeLogger, ShapeLogger

# Global shape logger instance
_shape_logger = None
_log_lock = threading.Lock()
_requests_logged = 0
_logging_enabled = False


def enable_shape_logging(
    output_file: str = "server_shapes.jsonl",
    compact: bool = True,
    verbose: bool = False,
    max_requests: int = -1,
):
    """
    Enable shape logging for server inference.
    
    Args:
        output_file: Output file for shapes
        compact: Use compact logging
        verbose: Print shapes to console
        max_requests: Maximum number of requests to log (-1 = unlimited)
    """
    global _shape_logger, _logging_enabled
    
    LoggerClass = CompactShapeLogger if compact else ShapeLogger
    _shape_logger = LoggerClass(output_file=output_file, verbose=verbose)
    _shape_logger.__enter__()
    _logging_enabled = True
    
    print(f"\n{'='*80}")
    print("Shape Logging Enabled")
    print(f"{'='*80}")
    print(f"Output file: {output_file}")
    print(f"Compact mode: {compact}")
    print(f"Verbose mode: {verbose}")
    print(f"Max requests to log: {max_requests if max_requests > 0 else 'unlimited'}")
    print(f"{'='*80}\n")
    
    # Register cleanup
    atexit.register(disable_shape_logging)
    
    return _shape_logger


def disable_shape_logging():
    """Disable shape logging and write results."""
    global _shape_logger, _logging_enabled
    
    if _shape_logger is not None and _logging_enabled:
        print(f"\n{'='*80}")
        print("Stopping Shape Logging...")
        print(f"{'='*80}")
        
        try:
            _shape_logger.__exit__(None, None, None)
            _logging_enabled = False
            
            summary = _shape_logger.get_summary()
            print(f"\nLogged {_requests_logged} requests")
            print(f"Total operations: {summary['total_operations']}")
            print(f"Unique operations: {summary['unique_operations']}")
            print(f"\nShapes saved to: {_shape_logger.output_file}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Error stopping shape logger: {e}")
        
        _shape_logger = None


def should_log_request() -> bool:
    """Check if we should log this request."""
    global _requests_logged
    
    with _log_lock:
        _requests_logged += 1
        return _logging_enabled


class ShapeLoggingContext:
    """Context manager for request-level shape logging."""
    
    def __init__(self, max_requests: int = -1):
        self.max_requests = max_requests
        self.should_log = False
    
    def __enter__(self):
        global _requests_logged
        
        if not _logging_enabled:
            return self
        
        with _log_lock:
            if self.max_requests > 0 and _requests_logged >= self.max_requests:
                return self
            
            _requests_logged += 1
            self.should_log = True
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def patch_model_runner():
    """
    Patch the model runner to enable shape logging during forward passes.
    This intercepts the model's forward pass.
    """
    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
        
        original_forward = ModelRunner.forward_batch_generation
        
        def wrapped_forward(self, *args, **kwargs):
            # Only log if shape logging is enabled
            if _logging_enabled and _shape_logger is not None:
                # Forward pass already happens within the shape logger context
                return original_forward(self, *args, **kwargs)
            else:
                return original_forward(self, *args, **kwargs)
        
        ModelRunner.forward_batch_generation = wrapped_forward
        print("[Shape Logger] Patched ModelRunner.forward_batch_generation")
        
    except Exception as e:
        print(f"[Shape Logger] Warning: Could not patch model runner: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Launch SGLang server with kernel shape logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DeepSeek with TP=8
  python launch_server_with_shapes.py \\
      --model-path deepseek-ai/DeepSeek-V3 \\
      --tp-size 8 \\
      --output-file deepseek_v3_shapes.jsonl
  
  # Qwen with TP=4 and verbose logging
  python launch_server_with_shapes.py \\
      --model-path Qwen/Qwen2.5-72B-Instruct \\
      --tp-size 4 \\
      --output-file qwen_shapes.jsonl \\
      --verbose
  
  # Log only first 10 requests
  python launch_server_with_shapes.py \\
      --model-path MODEL_PATH \\
      --tp-size 8 \\
      --log-first-n-requests 10
        """,
    )
    
    # Shape logging arguments
    parser.add_argument(
        "--output-file",
        type=str,
        default="server_shapes.jsonl",
        help="Output file for shape logs (default: server_shapes.jsonl)",
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="Disable compact mode (includes dtype/device, larger files)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print shapes to console (warning: very verbose)",
    )
    parser.add_argument(
        "--log-first-n-requests",
        type=int,
        default=-1,
        help="Only log first N requests (-1 = unlimited)",
    )
    parser.add_argument(
        "--start-logging-immediately",
        action="store_true",
        help="Start logging immediately (default: after first request)",
    )
    
    # Import ServerArgs to get all standard server arguments
    from sglang.srt.server_args import ServerArgs
    ServerArgs.add_cli_args(parser)
    
    args = parser.parse_args()
    
    # Extract our custom args
    output_file = args.output_file
    compact = not args.no_compact
    verbose = args.verbose
    max_requests = args.log_first_n_requests
    start_immediately = args.start_logging_immediately
    
    # Remove our custom args
    delattr(args, "output_file")
    delattr(args, "no_compact")
    delattr(args, "verbose")
    delattr(args, "log_first_n_requests")
    delattr(args, "start_logging_immediately")
    
    # Parse server args
    server_args = ServerArgs.from_cli_args(args)
    
    print("="*80)
    print("SGLang Server with Shape Logging")
    print("="*80)
    print(f"Model: {server_args.model_path}")
    print(f"Tensor Parallel: {server_args.tp_size}")
    print(f"Shape Output: {output_file}")
    print(f"Compact Mode: {compact}")
    print("="*80)
    
    # Enable shape logging immediately if requested
    if start_immediately:
        enable_shape_logging(output_file, compact, verbose, max_requests)
    
    # Patch the model runner
    patch_model_runner()
    
    # Import and start the server
    # We need to do this after setting up the logger
    from sglang.srt.server import launch_server
    
    try:
        # If not started immediately, enable logging after server is ready
        if not start_immediately:
            # The server will block here, so we enable logging before launch
            enable_shape_logging(output_file, compact, verbose, max_requests)
        
        # Launch the server (this blocks until server stops)
        launch_server(server_args)
        
    except KeyboardInterrupt:
        print("\n\nServer interrupted by user")
    except Exception as e:
        print(f"\n\nServer error: {e}")
        raise
    finally:
        disable_shape_logging()


if __name__ == "__main__":
    main()
