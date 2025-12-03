"""
Torch Dispatch-based Shape Logger

This module provides a PyTorch dispatch mode that logs tensor shapes for all operations.
Inspired by: https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505

Usage:
    from torch_shape_logger import ShapeLogger
    
    with ShapeLogger(output_file="shapes.log") as logger:
        # Your model inference code here
        model(input_data)
    
    # Shapes are automatically written to the file
"""

import atexit
import json
import signal
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class ShapeLogger(TorchDispatchMode):
    """
    A TorchDispatchMode that intercepts all tensor operations and logs their input/output shapes.
    """

    def __init__(self, output_file: str = "kernel_shapes.jsonl", verbose: bool = False, write_incrementally: bool = True):
        """
        Initialize the ShapeLogger.

        Args:
            output_file: Path to the output file where shapes will be logged (JSONL format)
            verbose: If True, print shapes to console as well
            write_incrementally: If True, write each log entry immediately (safer but slower)
        """
        super().__init__()
        self.output_file = output_file
        self.verbose = verbose
        self.write_incrementally = write_incrementally
        self.call_count = 0
        self.op_counts = defaultdict(int)
        self.log_entries: List[Dict[str, Any]] = []
        self.file_handle = None
        self._cleanup_registered = False

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

    def _extract_tensor_info(self, obj) -> Optional[Dict[str, Any]]:
        """
        Extract shape and dtype information from a tensor or nested structure.
        """
        if isinstance(obj, torch.Tensor):
            return {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
                "requires_grad": obj.requires_grad,
            }
        elif isinstance(obj, (list, tuple)):
            tensor_infos = []
            for item in obj:
                info = self._extract_tensor_info(item)
                if info is not None:
                    tensor_infos.append(info)
            return tensor_infos if tensor_infos else None
        elif isinstance(obj, dict):
            tensor_infos = {}
            for key, value in obj.items():
                info = self._extract_tensor_info(value)
                if info is not None:
                    tensor_infos[str(key)] = info
            return tensor_infos if tensor_infos else None
        return None

    def _log_operation(self, func, args, kwargs, result):
        """
        Log information about a single operation.
        """
        self.call_count += 1
        op_name = str(func)
        self.op_counts[op_name] += 1

        # Extract tensor shapes from inputs
        input_shapes = self._extract_tensor_info(args)
        kwarg_shapes = self._extract_tensor_info(kwargs)

        # Extract tensor shapes from outputs
        output_shapes = self._extract_tensor_info(result)

        # Only log if there are actual tensors involved
        if input_shapes or output_shapes:
            log_entry = {
                "call_id": self.call_count,
                "operation": op_name,
                "op_count": self.op_counts[op_name],
                "inputs": input_shapes,
                "kwargs": kwarg_shapes,
                "outputs": output_shapes,
            }

            # Write incrementally if enabled
            if self.write_incrementally and self.file_handle:
                try:
                    self.file_handle.write(json.dumps(log_entry) + "\n")
                    self.file_handle.flush()  # Force write to disk
                except Exception as e:
                    print(f"Warning: Failed to write log entry: {e}")
            else:
                self.log_entries.append(log_entry)

            if self.verbose:
                print(f"[{self.call_count}] {op_name}")
                if input_shapes:
                    print(f"  Inputs: {input_shapes}")
                if output_shapes:
                    print(f"  Outputs: {output_shapes}")

    def __enter__(self):
        """Enter the context manager."""
        super().__enter__()
        self.call_count = 0
        self.op_counts.clear()
        self.log_entries.clear()
        
        # Open file for incremental writing
        if self.write_incrementally:
            self.file_handle = open(self.output_file, "w")
        
        # Register cleanup handlers to ensure data is saved even if killed
        if not self._cleanup_registered:
            atexit.register(self._emergency_cleanup)
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            self._cleanup_registered = True
        
        print(f"Shape logging started. Will write to: {self.output_file}")
        print(f"Write mode: {'incremental (safer)' if self.write_incrementally else 'batch (faster)'}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and write logs to file."""
        super().__exit__(exc_type, exc_val, exc_tb)
        
        # Close file handle if open
        if self.file_handle:
            try:
                self.file_handle.close()
            except Exception as e:
                print(f"Warning: Failed to close file handle: {e}")
            self.file_handle = None
        
        # Write remaining logs if not writing incrementally
        if not self.write_incrementally and self.log_entries:
            self._write_logs()
        
        # Always write summary
        self._write_summary()
        
        print(f"\nShape logging complete!")
        print(f"Total operations: {self.call_count}")
        print(f"Unique operations: {len(self.op_counts)}")
        print(f"Logs written to: {self.output_file}")
        return False

    def _write_logs(self):
        """Write all logged entries to the output file in JSONL format."""
        try:
            with open(self.output_file, "w") as f:
                for entry in self.log_entries:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Error writing logs: {e}")
    
    def _write_summary(self):
        """Write summary file."""
        try:
            summary_file = self.output_file.replace(".jsonl", "_summary.json")
            summary = {
                "total_operations": self.call_count,
                "unique_operations": len(self.op_counts),
                "operation_counts": dict(sorted(self.op_counts.items(), key=lambda x: x[1], reverse=True)),
            }
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Summary written to: {summary_file}")
        except Exception as e:
            print(f"Error writing summary: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of logged operations."""
        return {
            "total_operations": self.call_count,
            "unique_operations": len(self.op_counts),
            "operation_counts": dict(self.op_counts),
        }
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals to save data before exit."""
        print(f"\n\nReceived signal {signum}, saving data before exit...")
        self._emergency_cleanup()
        import sys
        sys.exit(1)
    
    def _emergency_cleanup(self):
        """Emergency cleanup to save data if process is terminated."""
        if self.call_count > 0:
            print(f"\nEmergency cleanup: Saving {self.call_count} operations...")
            try:
                # Close file handle if open
                if self.file_handle:
                    self.file_handle.close()
                    self.file_handle = None
                
                # Write remaining logs if not writing incrementally
                if not self.write_incrementally and self.log_entries:
                    self._write_logs()
                
                # Write summary
                self._write_summary()
                print("Emergency save completed!")
            except Exception as e:
                print(f"Error during emergency cleanup: {e}")


class CompactShapeLogger(ShapeLogger):
    """
    A more compact version that only logs operation names and shapes (no dtype/device).
    Useful for reducing log file size.
    """

    def _extract_tensor_info(self, obj) -> Optional[Any]:
        """Extract only shape information (no dtype/device)."""
        if isinstance(obj, torch.Tensor):
            return list(obj.shape)
        elif isinstance(obj, (list, tuple)):
            shapes = []
            for item in obj:
                info = self._extract_tensor_info(item)
                if info is not None:
                    shapes.append(info)
            return shapes if shapes else None
        elif isinstance(obj, dict):
            shapes = {}
            for key, value in obj.items():
                info = self._extract_tensor_info(value)
                if info is not None:
                    shapes[str(key)] = info
            return shapes if shapes else None
        return None


def analyze_shape_log(log_file: str):
    """
    Analyze a shape log file and print statistics with example shapes.

    Args:
        log_file: Path to the JSONL log file
    """
    print(f"Analyzing {log_file}...")

    op_counts = defaultdict(int)
    shape_patterns = defaultdict(lambda: defaultdict(int))
    first_example = {}  # Store first example for each operation

    with open(log_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                op_name = entry["operation"]
                op_counts[op_name] += 1

                # Track shape patterns for each operation
                if entry.get("outputs"):
                    shape_key = json.dumps(entry["outputs"], sort_keys=True)
                    shape_patterns[op_name][shape_key] += 1
                
                # Store first example
                if op_name not in first_example:
                    first_example[op_name] = entry
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")

    print("\n=== Top 20 Most Frequent Operations (with example shapes) ===")
    for op_name, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        example = first_example.get(op_name, {})
        output_shape = example.get("outputs", "N/A")
        # Format shape compactly
        if isinstance(output_shape, list):
            shape_str = str(output_shape)
        else:
            shape_str = json.dumps(output_shape)
        # Limit length
        if len(shape_str) > 60:
            shape_str = shape_str[:57] + "..."
        print(f"{count:8d} : {op_name}")
        print(f"           Example output shape: {shape_str}")

    print("\n=== Total Statistics ===")
    print(f"Total operations: {sum(op_counts.values())}")
    print(f"Unique operations: {len(op_counts)}")
    print(f"\nFor detailed shape analysis with all shapes, use:")
    print(f"  python analyze_shapes.py {log_file} --show-shapes")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Analyze shape log files")
    parser.add_argument("--analyze", type=str, help="Analyze an existing log file")
    args = parser.parse_args()

    if args.analyze:
        analyze_shape_log(args.analyze)
    else:
        # Demo
        print("Demo: Logging shapes for a simple PyTorch operation")
        with ShapeLogger("demo_shapes.jsonl", verbose=True) as logger:
            x = torch.randn(10, 20)
            y = torch.randn(20, 30)
            z = torch.mm(x, y)
            result = torch.relu(z)

        print("\nSummary:", logger.get_summary())
