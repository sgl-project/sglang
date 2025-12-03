"""
Rank-Aware Torch Dispatch-based Shape Logger

This module provides a PyTorch dispatch mode that logs tensor shapes for all operations
and tracks which GPU/rank each operation runs on. Essential for TP (tensor parallel) profiling.

Usage:
    from torch_shape_logger_rank import RankAwareShapeLogger
    
    # Option 1: Log only on GPU 0
    with RankAwareShapeLogger(output_file="gpu0_shapes.log", only_rank=0) as logger:
        # Your model inference code here
        model(input_data)
    
    # Option 2: Log all ranks with rank information
    with RankAwareShapeLogger(output_file="all_ranks_shapes.log") as logger:
        model(input_data)
"""

import atexit
import json
import os
import signal
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def get_current_rank() -> int:
    """Get the current process rank (GPU ID)."""
    # Try different methods to get rank
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    elif torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        # If not in distributed mode, assume rank 0
        return 0


def get_world_size() -> int:
    """Get the world size (number of GPUs)."""
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    elif torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


class RankAwareShapeLogger(TorchDispatchMode):
    """
    A TorchDispatchMode that intercepts all tensor operations and logs their input/output shapes
    along with the GPU/rank information.
    """

    def __init__(
        self,
        output_file: str = "kernel_shapes.jsonl",
        verbose: bool = False,
        write_incrementally: bool = True,
        only_rank: Optional[int] = None,
        log_device_placement: bool = True,
    ):
        """
        Initialize the RankAwareShapeLogger.

        Args:
            output_file: Path to the output file where shapes will be logged (JSONL format)
            verbose: If True, print shapes to console as well
            write_incrementally: If True, write each log entry immediately (safer but slower)
            only_rank: If set, only log operations on this specific rank (e.g., 0 for GPU 0)
            log_device_placement: If True, log which device each tensor is on
        """
        super().__init__()
        self.output_file = output_file
        self.verbose = verbose
        self.write_incrementally = write_incrementally
        self.only_rank = only_rank
        self.log_device_placement = log_device_placement
        self.call_count = 0
        self.op_counts = defaultdict(int)
        self.log_entries: List[Dict[str, Any]] = []
        self.file_handle = None
        self._cleanup_registered = False
        
        # Get rank info
        self.current_rank = get_current_rank()
        self.world_size = get_world_size()
        
        # Check if we should log on this rank
        self.should_log = (only_rank is None) or (self.current_rank == only_rank)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        Intercept every torch operation and log tensor shapes.
        """
        if kwargs is None:
            kwargs = {}

        # Execute the actual operation
        result = func(*args, **kwargs)

        # Log the operation (only if this rank should log)
        if self.should_log:
            self._log_operation(func, args, kwargs, result)

        return result

    def _extract_tensor_info(self, obj) -> Optional[Dict[str, Any]]:
        """
        Extract shape and dtype information from a tensor or nested structure.
        """
        if isinstance(obj, torch.Tensor):
            info = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
            if self.log_device_placement:
                info["device"] = str(obj.device)
            return info
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
                "rank": self.current_rank,
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
                print(f"[Rank {self.current_rank}][{self.call_count}] {op_name}")
                if input_shapes:
                    print(f"  Inputs: {input_shapes}")
                if output_shapes:
                    print(f"  Outputs: {output_shapes}")

    def __enter__(self):
        """Enter the context manager."""
        super().__enter__()
        
        # Only initialize logging on the appropriate rank
        if self.should_log:
            self.call_count = 0
            self.op_counts.clear()
            self.log_entries.clear()
            
            # Add rank to filename if logging all ranks
            if self.only_rank is None:
                base, ext = os.path.splitext(self.output_file)
                self.output_file = f"{base}_rank{self.current_rank}{ext}"
            
            # Open file for incremental writing
            if self.write_incrementally:
                self.file_handle = open(self.output_file, "w")
            
            # Register cleanup handlers to ensure data is saved even if killed
            if not self._cleanup_registered:
                atexit.register(self._emergency_cleanup)
                signal.signal(signal.SIGTERM, self._signal_handler)
                signal.signal(signal.SIGINT, self._signal_handler)
                self._cleanup_registered = True
            
            print(f"[Rank {self.current_rank}/{self.world_size}] Shape logging started. Will write to: {self.output_file}")
            print(f"[Rank {self.current_rank}] Write mode: {'incremental (safer)' if self.write_incrementally else 'batch (faster)'}")
        else:
            print(f"[Rank {self.current_rank}] Shape logging skipped (only_rank={self.only_rank})")
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and write logs to file."""
        super().__exit__(exc_type, exc_val, exc_tb)
        
        if not self.should_log:
            return False
        
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
        
        print(f"\n[Rank {self.current_rank}] Shape logging complete!")
        print(f"[Rank {self.current_rank}] Total operations: {self.call_count}")
        print(f"[Rank {self.current_rank}] Unique operations: {len(self.op_counts)}")
        print(f"[Rank {self.current_rank}] Logs written to: {self.output_file}")
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
                "rank": self.current_rank,
                "world_size": self.world_size,
                "total_operations": self.call_count,
                "unique_operations": len(self.op_counts),
                "operation_counts": dict(sorted(self.op_counts.items(), key=lambda x: x[1], reverse=True)),
            }
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[Rank {self.current_rank}] Summary written to: {summary_file}")
        except Exception as e:
            print(f"Error writing summary: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of logged operations."""
        return {
            "rank": self.current_rank,
            "world_size": self.world_size,
            "total_operations": self.call_count,
            "unique_operations": len(self.op_counts),
            "operation_counts": dict(self.op_counts),
        }
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals to save data before exit."""
        print(f"\n\n[Rank {self.current_rank}] Received signal {signum}, saving data before exit...")
        self._emergency_cleanup()
        import sys
        sys.exit(1)
    
    def _emergency_cleanup(self):
        """Emergency cleanup to save data if process is terminated."""
        if self.should_log and self.call_count > 0:
            print(f"\n[Rank {self.current_rank}] Emergency cleanup: Saving {self.call_count} operations...")
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
                print(f"[Rank {self.current_rank}] Emergency save completed!")
            except Exception as e:
                print(f"Error during emergency cleanup: {e}")


class CompactRankAwareShapeLogger(RankAwareShapeLogger):
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


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Rank-aware shape logger demo")
    parser.add_argument("--only-rank", type=int, help="Only log on this rank")
    parser.add_argument("--output", type=str, default="rank_shapes.jsonl", help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print(f"Current rank: {get_current_rank()}")
    print(f"World size: {get_world_size()}")
    
    # Demo with simple operations
    with RankAwareShapeLogger(args.output, verbose=args.verbose, only_rank=args.only_rank) as logger:
        x = torch.randn(10, 20)
        y = torch.randn(20, 30)
        z = torch.mm(x, y)
        result = torch.relu(z)

    if logger.should_log:
        print("\nSummary:", logger.get_summary())
