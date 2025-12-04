"""
Rank-Aware Torch Dispatch-based Shape Logger

Minimal logger for profiling tensor shapes in TP workers.
Only logs operations on specified rank to avoid overhead.

Usage in TP worker (automatically activated by environment variables):
    SGLANG_PROFILE_SHAPES=1 SGLANG_PROFILE_SHAPES_RANK=0 \\
    SGLANG_PROFILE_SHAPES_FILE=shapes.jsonl python -m sglang.launch_server ...
    
Or with bench_one_batch_server (recommended):
    SGLANG_PROFILE_SHAPES=1 SGLANG_PROFILE_SHAPES_RANK=0 \\
    SGLANG_PROFILE_SHAPES_FILE=shapes.jsonl python3 -m sglang.bench_one_batch_server ...
"""

import atexit
import inspect
import json
import os
import signal
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

# Try to import CUDA graph and torch.compile detection utilities
try:
    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
    from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
    _CAN_DETECT_CUDA_GRAPH = True
except ImportError:
    _CAN_DETECT_CUDA_GRAPH = False
    def get_is_capture_mode():
        return False
    def is_in_piecewise_cuda_graph():
        return False

# Try to detect torch.compile mode
def _is_torch_compile_mode():
    """Detect if we're inside a torch.compile context."""
    try:
        # Check if torch._dynamo is active
        if hasattr(torch, '_dynamo'):
            return torch._dynamo.is_compiling() if hasattr(torch._dynamo, 'is_compiling') else False
    except Exception:
        pass
    return False


def get_current_rank() -> int:
    """Get the current process rank (GPU ID)."""
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    elif torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class CompactRankAwareShapeLogger(TorchDispatchMode):
    """Compact shape logger that only logs on specified rank."""

    def __init__(
        self,
        output_file: str = "shapes.jsonl",
        verbose: bool = False,
        only_rank: Optional[int] = None,
    ):
        super().__init__()
        self.output_file = output_file
        self.verbose = verbose
        self.only_rank = only_rank
        self.call_count = 0
        self.op_counts = defaultdict(int)
        self.file_handle = None
        self.current_rank = get_current_rank()
        self.should_log = (only_rank is None) or (self.current_rank == only_rank)
        self._forward_pass_count = 0
        self._in_forward_pass = False

    def start_forward_pass(self):
        """Called by tp_worker to mark start of forward pass."""
        if self.should_log and not self._in_forward_pass:
            self._in_forward_pass = True
            self._forward_pass_count += 1
            if self.verbose:
                print(f"[Rank {self.current_rank}] Forward pass #{self._forward_pass_count} started")

    def end_forward_pass(self):
        """Called by tp_worker to mark end of forward pass."""
        if self.should_log and self._in_forward_pass:
            self._in_forward_pass = False
            if self.verbose:
                print(f"[Rank {self.current_rank}] Forward pass #{self._forward_pass_count} ended, ops={self.call_count}")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        # Skip shape logging during CUDA graph capture or torch.compile
        should_skip_logging = False
        if _CAN_DETECT_CUDA_GRAPH:
            should_skip_logging = get_is_capture_mode() or is_in_piecewise_cuda_graph()
        if not should_skip_logging:
            should_skip_logging = _is_torch_compile_mode()
        
        # Extract input shapes with argument names (if profiling and not skipping)
        named_inputs = None
        if self.should_log and self._in_forward_pass and not should_skip_logging:
            named_inputs = self._extract_shapes_with_names(func, args, kwargs)
        
        result = func(*args, **kwargs)
        
        if self.should_log and self._in_forward_pass and not should_skip_logging:
            self._log_operation(func, named_inputs, result)
        
        return result

    def _extract_shapes(self, obj) -> Optional[Any]:
        """Extract only shapes (no dtype/device)."""
        if isinstance(obj, torch.Tensor):
            return list(obj.shape)
        elif isinstance(obj, (list, tuple)):
            shapes = [self._extract_shapes(item) for item in obj]
            return [s for s in shapes if s is not None] or None
        elif isinstance(obj, dict):
            shapes = {str(k): self._extract_shapes(v) for k, v in obj.items()}
            return {k: v for k, v in shapes.items() if v is not None} or None
        return None

    def _extract_shapes_with_names(self, func, args, kwargs) -> Optional[Dict[str, Any]]:
        """Extract shapes with argument names from function signature."""
        try:
            # Try to get function signature to map args to parameter names
            sig = None
            try:
                sig = inspect.signature(func)
            except (ValueError, TypeError):
                # Some torch operations don't have inspectable signatures
                pass
            
            named_args = {}
            
            if sig:
                # Map positional args to parameter names
                param_names = list(sig.parameters.keys())
                bound_args = None
                try:
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                except TypeError:
                    # If binding fails, fall back to positional mapping
                    pass
                
                if bound_args:
                    for param_name, arg_value in bound_args.arguments.items():
                        shape = self._extract_shapes(arg_value)
                        if shape is not None:
                            named_args[param_name] = shape
                else:
                    # Fallback: map positional args by index
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            shape = self._extract_shapes(arg)
                            if shape is not None:
                                named_args[param_names[i]] = shape
            else:
                # No signature available: use positional indices and kwargs
                for i, arg in enumerate(args):
                    shape = self._extract_shapes(arg)
                    if shape is not None:
                        named_args[f"arg{i}"] = shape
            
            # Add kwargs (they already have names)
            for kwarg_name, kwarg_value in kwargs.items():
                shape = self._extract_shapes(kwarg_value)
                if shape is not None:
                    named_args[kwarg_name] = shape
            
            return named_args if named_args else None
        except Exception:
            # Fallback to simple extraction without names
            input_shapes = self._extract_shapes(args)
            kwarg_shapes = self._extract_shapes(kwargs) if kwargs else None
            result = {}
            if input_shapes:
                result["args"] = input_shapes
            if kwarg_shapes:
                result.update(kwarg_shapes)
            return result if result else None

    def _log_operation(self, func, named_inputs, result):
        """Log operation with minimal overhead."""
        output_shapes = self._extract_shapes(result)
        if named_inputs or output_shapes:  # Log if any shapes present
            self.call_count += 1
            op_name = str(func)
            self.op_counts[op_name] += 1
            
            if self.file_handle:
                try:
                    log_entry = {
                        "call_id": self.call_count,
                        "forward_pass": self._forward_pass_count,
                        "operation": op_name,
                        "outputs": output_shapes,
                    }
                    # Add named inputs (includes both positional args with names and kwargs)
                    if named_inputs:
                        log_entry["inputs"] = named_inputs
                    self.file_handle.write(json.dumps(log_entry) + "\n")
                    if self.call_count % 1000 == 0:  # Flush periodically
                        self.file_handle.flush()
                except Exception:
                    pass

    def __enter__(self):
        super().__enter__()
        if self.should_log:
            self.call_count = 0
            self.op_counts.clear()
            self._forward_pass_count = 0
            self._in_forward_pass = False
            self.file_handle = open(self.output_file, "w")
            atexit.register(self._cleanup)
            print(f"[Rank {self.current_rank}] Shape profiling enabled → {self.output_file}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        if self.should_log:
            self._cleanup()
        return False

    def _cleanup(self):
        """Save and close."""
        if self.file_handle:
            try:
                self.file_handle.close()
                self.file_handle = None
                # Write summary
                summary_file = self.output_file.replace(".jsonl", "_summary.json")
                summary = {
                    "rank": self.current_rank,
                    "total_operations": self.call_count,
                    "forward_passes": self._forward_pass_count,
                    "unique_operations": len(self.op_counts),
                    "operation_counts": dict(sorted(self.op_counts.items(), key=lambda x: x[1], reverse=True)),
                }
                with open(summary_file, "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"[Rank {self.current_rank}] Profiled {self.call_count} ops across {self._forward_pass_count} forward passes → {self.output_file}")
            except Exception:
                pass


