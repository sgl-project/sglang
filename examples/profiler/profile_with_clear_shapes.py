"""
Profile model kernel shapes with clear, readable output format.

This script captures tensor shapes during model inference and outputs them
in a clear format: kernel name, input shapes, and output shapes.

Usage:
    # Profile DeepSeek V3 with TP=8
    python profile_with_clear_shapes.py \
        --model-path deepseek-ai/DeepSeek-V3 \
        --tp-size 8 \
        --num-prompts 3 \
        --max-tokens 50

    # Profile Qwen with TP=8
    python profile_with_clear_shapes.py \
        --model-path Qwen/Qwen2.5-14B-Instruct \
        --tp-size 8 \
        --num-prompts 3 \
        --max-tokens 50
"""

import argparse
import dataclasses
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Add profiler to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import sglang as sgl
from sglang.srt.server_args import ServerArgs


class ClearShapeLogger(TorchDispatchMode):
    """
    A TorchDispatchMode that logs operations with clear, readable format:
    - Kernel name first
    - All input shapes clearly labeled
    - All output shapes clearly labeled
    """

    def __init__(
        self,
        output_file: str = "kernel_shapes.jsonl",
        verbose: bool = False,
        tp_size: int = 1,
    ):
        """
        Initialize the ClearShapeLogger.

        Args:
            output_file: Path to the output file where shapes will be logged (JSONL format)
            verbose: If True, print shapes to console as well
            tp_size: Tensor parallel size (for metadata)
        """
        super().__init__()
        self.output_file = output_file
        self.verbose = verbose
        self.tp_size = tp_size
        self.call_count = 0
        self.file_handle = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        Intercept every torch operation and log tensor shapes.
        """
        if kwargs is None:
            kwargs = {}

        # Execute the actual operation
        result = func(*args, **kwargs)

        # Log the operation
        self._log_operation(func, args, kwargs, result)

        return result

    def _extract_shapes(self, obj) -> Optional[List]:
        """
        Extract shape information from a tensor or nested structure.
        Returns a flat list of shapes.
        """
        shapes = []
        
        if isinstance(obj, torch.Tensor):
            shapes.append(list(obj.shape))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                item_shapes = self._extract_shapes(item)
                if item_shapes:
                    shapes.extend(item_shapes)
        elif isinstance(obj, dict):
            for value in obj.values():
                value_shapes = self._extract_shapes(value)
                if value_shapes:
                    shapes.extend(value_shapes)
        
        return shapes if shapes else None

    def _log_operation(self, func, args, kwargs, result):
        """
        Log information about a single operation in clear format.
        """
        self.call_count += 1
        
        # Extract kernel/operation name
        op_name = str(func)
        
        # Extract shapes from inputs (args)
        input_shapes = self._extract_shapes(args)
        
        # Extract shapes from keyword arguments
        kwarg_shapes = self._extract_shapes(kwargs)
        
        # Combine all input shapes
        all_input_shapes = []
        if input_shapes:
            all_input_shapes.extend(input_shapes)
        if kwarg_shapes:
            all_input_shapes.extend(kwarg_shapes)
        
        # Extract shapes from outputs
        output_shapes = self._extract_shapes(result)

        # Only log if there are actual tensors involved
        if all_input_shapes or output_shapes:
            log_entry = {
                "call_id": self.call_count,
                "kernel": op_name,
                "input_shapes": all_input_shapes if all_input_shapes else [],
                "output_shapes": output_shapes if output_shapes else [],
                "tp_size": self.tp_size,
            }

            # Write immediately to file
            if self.file_handle:
                try:
                    self.file_handle.write(json.dumps(log_entry) + "\n")
                    self.file_handle.flush()  # Force write to disk
                except Exception as e:
                    print(f"Warning: Failed to write log entry: {e}")

            if self.verbose:
                self._print_operation(log_entry)

    def _print_operation(self, log_entry: Dict[str, Any]):
        """Print operation in clear, readable format."""
        print(f"\n[{log_entry['call_id']}] {log_entry['kernel']}")
        
        if log_entry['input_shapes']:
            print(f"  Inputs:")
            for i, shape in enumerate(log_entry['input_shapes'], 1):
                print(f"    [{i}] {shape}")
        
        if log_entry['output_shapes']:
            print(f"  Outputs:")
            for i, shape in enumerate(log_entry['output_shapes'], 1):
                print(f"    [{i}] {shape}")

    def __enter__(self):
        """Enter the context manager."""
        super().__enter__()
        self.call_count = 0
        
        # Open file for incremental writing
        self.file_handle = open(self.output_file, "w")
        
        print(f"\n{'='*80}")
        print("Clear Shape Logging Started")
        print(f"{'='*80}")
        print(f"Output file: {self.output_file}")
        print(f"TP size: {self.tp_size}")
        print(f"Verbose mode: {self.verbose}")
        print(f"{'='*80}\n")
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close file."""
        super().__exit__(exc_type, exc_val, exc_tb)
        
        # Close file handle
        if self.file_handle:
            try:
                self.file_handle.close()
            except Exception as e:
                print(f"Warning: Failed to close file handle: {e}")
            self.file_handle = None
        
        print(f"\n{'='*80}")
        print("Shape Logging Complete")
        print(f"{'='*80}")
        print(f"Total operations logged: {self.call_count}")
        print(f"Output file: {self.output_file}")
        print(f"{'='*80}\n")
        
        return False


def main(
    server_args: ServerArgs,
    output_file: str = "kernel_shapes.jsonl",
    verbose: bool = False,
    num_prompts: int = 3,
    max_tokens: int = 100,
):
    """
    Profile model with clear shape logging.

    Args:
        server_args: Server configuration arguments
        output_file: Path to output JSONL file for shape logs
        verbose: If True, print shapes to console during execution
        num_prompts: Number of prompts to process
        max_tokens: Maximum tokens to generate per prompt
    """
    print("=" * 80)
    print("Model Kernel Shape Profiler (Clear Format)")
    print("=" * 80)
    print(f"Model: {server_args.model_path}")
    print(f"Tensor Parallel: {server_args.tp_size}")
    print(f"Output file: {output_file}")
    print(f"Verbose mode: {verbose}")
    print(f"Prompts: {num_prompts}")
    print(f"Max tokens: {max_tokens}")
    print("=" * 80)

    # Sample prompts
    all_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "Explain quantum computing in simple terms:",
        "What is the meaning of life?",
    ]
    prompts = all_prompts[:num_prompts]

    # Create sampling params
    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": max_tokens,
    }

    print(f"\nInitializing engine with TP={server_args.tp_size}...")
    llm = sgl.Engine(**dataclasses.asdict(server_args))
    print("✓ Engine initialized successfully!\n")

    # Run inference with shape logging
    print("Running inference with shape logging...")
    print("(This may take a while, especially on first run...)\n")
    
    with ClearShapeLogger(
        output_file=output_file,
        verbose=verbose,
        tp_size=server_args.tp_size,
    ) as logger:
        outputs = llm.generate(prompts, sampling_params)
    
    print("✓ Generation completed!")

    # Print the outputs
    print("\n" + "=" * 80)
    print("Generation Results")
    print("=" * 80)
    for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
        print(f"\n[{i}/{len(prompts)}]")
        print(f"Prompt: {prompt}")
        generated = output['text']
        display_text = generated[:150] + '...' if len(generated) > 150 else generated
        print(f"Generated: {display_text}")

    print("\nShutting down engine...")
    llm.shutdown()
    print("✓ Done.")
    
    # Print analysis hint
    print(f"\nTo analyze the shapes:")
    print(f"  python analyze_clear_shapes.py {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile kernel shapes with clear, readable format",
    )
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=100)

    args = parser.parse_args()

    # Extract custom args
    output_file = args.output_file
    verbose = args.verbose
    num_prompts = args.num_prompts
    max_tokens = args.max_tokens

    # Remove custom args
    delattr(args, "output_file")
    delattr(args, "verbose")
    delattr(args, "num_prompts")
    delattr(args, "max_tokens")

    # Parse server args
    server_args = ServerArgs.from_cli_args(args)

    if server_args.model_path is None:
        print("Error: --model-path is required")
        sys.exit(1)

    # Set default output file with TP size in name
    if output_file is None:
        model_name = server_args.model_path.split('/')[-1].replace('-', '_').lower()
        output_file = f"{model_name}_tp{server_args.tp_size}_clear_shapes.jsonl"

    main(
        server_args=server_args,
        output_file=output_file,
        verbose=verbose,
        num_prompts=num_prompts,
        max_tokens=max_tokens,
    )
