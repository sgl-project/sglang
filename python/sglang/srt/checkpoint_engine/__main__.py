#!/usr/bin/env python3
"""
Main entry point for checkpoint engine update functionality.

This allows the checkpoint engine update to be run as:
python3 -m sglang.checkpoint_engine.update [args]

Which will automatically invoke torchrun with the original update.py script.
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Main entry point that invokes the original update.py with torchrun."""
    
    # Find the original update script in examples directory
    script_dir = Path(__file__).parent
    update_script = script_dir.parent.parent.parent.parent / "examples" / "checkpoint_engine" / "update.py"
    
    if not update_script.exists():
        print(f"Error: Could not find update.py script at {update_script}", file=sys.stderr)
        sys.exit(1)
    
    # Parse inference_parallel_size from command line arguments to determine nproc-per-node
    inference_parallel_size = 8  # default
    args = sys.argv[1:]  # Skip the script name
    
    # Look for --inference-parallel-size in arguments
    for i, arg in enumerate(args):
        if arg == "--inference-parallel-size" and i + 1 < len(args):
            try:
                inference_parallel_size = int(args[i + 1])
            except ValueError:
                pass
            break
        elif arg.startswith("--inference-parallel-size="):
            try:
                inference_parallel_size = int(arg.split("=", 1)[1])
            except ValueError:
                pass
            break
    
    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc-per-node={inference_parallel_size}",
        str(update_script)
    ] + args
    
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    
    # Execute torchrun with the original script
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("Error: torchrun command not found. Please ensure PyTorch is installed.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
