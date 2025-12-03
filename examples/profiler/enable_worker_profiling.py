"""
Enable shape logging in SGLang TP workers by setting environment variables.

This script patches the tp_worker to enable shape logging on rank 0.

Usage:
    # Set environment variable and run
    export SGLANG_PROFILE_SHAPES=1
    export SGLANG_PROFILE_SHAPES_RANK=0
    export SGLANG_PROFILE_SHAPES_FILE=gpu0_shapes.jsonl
    
    python -m sglang.launch_server \\
        --model-path Qwen/Qwen2.5-14B-Instruct \\
        --tp 8 \\
        --port 30000
    
    # Then make requests to trigger shape logging
    curl http://localhost:30000/v1/completions \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "prompt": "Hello, my name is",
            "max_tokens": 50
        }'
"""

import os
import sys

# Check if profiling is enabled
ENABLE_PROFILING = os.environ.get("SGLANG_PROFILE_SHAPES", "0") == "1"
TARGET_RANK = int(os.environ.get("SGLANG_PROFILE_SHAPES_RANK", "0"))
OUTPUT_FILE = os.environ.get("SGLANG_PROFILE_SHAPES_FILE", "gpu0_shapes.jsonl")

if ENABLE_PROFILING:
    print(f"[Shape Profiling] Enabled on rank {TARGET_RANK}, output: {OUTPUT_FILE}")
    
    # Import the logger
    profiler_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, profiler_path)
    from torch_shape_logger_rank import CompactRankAwareShapeLogger
    
    # Global logger instance
    _shape_logger = None
    _shape_logger_active = False
    
    def get_shape_logger():
        global _shape_logger
        if _shape_logger is None:
            from torch_shape_logger_rank import get_current_rank
            current_rank = get_current_rank()
            
            if current_rank == TARGET_RANK:
                _shape_logger = CompactRankAwareShapeLogger(
                    output_file=OUTPUT_FILE,
                    verbose=False,
                    only_rank=TARGET_RANK,
                )
                print(f"[Rank {current_rank}] Shape logger initialized: {OUTPUT_FILE}")
        return _shape_logger
    
    def activate_shape_logger():
        global _shape_logger_active
        if not _shape_logger_active:
            logger = get_shape_logger()
            if logger and logger.should_log:
                logger.__enter__()
                _shape_logger_active = True
                print(f"[Rank {TARGET_RANK}] Shape logger activated")
    
    def deactivate_shape_logger():
        global _shape_logger_active
        if _shape_logger_active:
            logger = get_shape_logger()
            if logger and logger.should_log:
                logger.__exit__(None, None, None)
                _shape_logger_active = False
                print(f"[Rank {TARGET_RANK}] Shape logger deactivated")
    
    print("[Shape Profiling] To use this, patch tp_worker.py to call activate_shape_logger()")
