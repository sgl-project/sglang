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
        log_first_n_forward_passes: int = 2,
        skip_first_n_forward_passes: int = 2,
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
        self.log_first_n_forward_passes = log_first_n_forward_passes
        self.skip_first_n_forward_passes = skip_first_n_forward_passes
        self._signal_handler_installed = False
    
    def _signal_handler(self, signum, frame):
        """Handle signals to ensure cleanup on crash."""
        print(f"[SHAPE_PROFILER] Signal {signum} received, flushing data...", flush=True)
        self._cleanup()
        # Re-raise the signal
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def start_forward_pass(self):
        """Called by tp_worker to mark start of forward pass."""
        if self.should_log and not self._in_forward_pass:
            self._in_forward_pass = True
            self._forward_pass_count += 1
            effective_pass = self._forward_pass_count - self.skip_first_n_forward_passes
            
            # Add separator markers for prefill and decode
            # Ensure file handle is open (it should be from __enter__, but check just in case)
            if effective_pass > 0 and self.file_handle:
                try:
                    if effective_pass == 1:
                        # First forward pass after warmup = prefill
                        separator = {
                            "call_id": 0,
                            "forward_pass": self._forward_pass_count,
                            "operation": "=" * 80,
                            "marker": "PREFILL_START",
                            "inputs": None,
                            "outputs": None,
                        }
                        self.file_handle.write(json.dumps(separator) + "\n")
                        self.file_handle.write("\n" * 3)  # Large space separator
                        self.file_handle.flush()
                        print(f"[SHAPE_PROFILER] Starting PREFILL pass #{self._forward_pass_count} (effective: {effective_pass})", flush=True)
                    elif effective_pass == 2:
                        # Second forward pass after warmup = decode
                        separator = {
                            "call_id": 0,
                            "forward_pass": self._forward_pass_count,
                            "operation": "=" * 80,
                            "marker": "DECODE_START",
                            "inputs": None,
                            "outputs": None,
                        }
                        self.file_handle.write(json.dumps(separator) + "\n")
                        self.file_handle.write("\n" * 3)  # Large space separator
                        self.file_handle.flush()
                        print(f"[SHAPE_PROFILER] Starting DECODE pass #{self._forward_pass_count} (effective: {effective_pass})", flush=True)
                except Exception as e:
                    if self.verbose:
                        print(f"[Rank {self.current_rank}] Error writing marker: {e}")
            
            if self.verbose:
                print(f"[Rank {self.current_rank}] Forward pass #{self._forward_pass_count} started (effective: {effective_pass})")
            
            # Print status every few passes so user knows profiling is active
            if effective_pass > 0 and effective_pass <= 5:
                print(f"[SHAPE_PROFILER] Logging forward pass {effective_pass} → {self.output_file}", flush=True)

    def end_forward_pass(self):
        """Called by tp_worker to mark end of forward pass."""
        if self.should_log and self._in_forward_pass:
            self._in_forward_pass = False
            if self.verbose:
                print(f"[Rank {self.current_rank}] Forward pass #{self._forward_pass_count} ended, ops={self.call_count}")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        # Skip shape logging during CUDA graph capture (but allow replay) or torch.compile
        should_skip_logging = False
        if _CAN_DETECT_CUDA_GRAPH:
            # Only skip during capture, not during replay (replay is actual inference we want to log)
            should_skip_logging = get_is_capture_mode()
            # Note: We don't skip is_in_piecewise_cuda_graph() because that includes both capture and replay
            # We want to log replay operations but skip capture
        if not should_skip_logging:
            should_skip_logging = _is_torch_compile_mode()
        
        # Extract input shapes with argument names (if profiling and not skipping)
        named_inputs = None
        if self.should_log and self._in_forward_pass and not should_skip_logging:
            named_inputs = self._extract_shapes_with_names(func, args, kwargs)
        
        # Execute the operation with error handling
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            # Log the failed operation, save file, and exit
            print(f"[SHAPE_PROFILER] EXCEPTION CAUGHT in dispatch: {type(e).__name__}: {e}", flush=True)
            print(f"[SHAPE_PROFILER] should_log={self.should_log}, _in_forward_pass={self._in_forward_pass}, should_skip={should_skip_logging}", flush=True)
            
            if self.should_log and self._in_forward_pass and not should_skip_logging:
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
                self._log_operation(func, named_inputs, None, error_info=error_info)
                
                # Save and close the file immediately
                print(f"[SHAPE_PROFILER] Calling cleanup to save file...", flush=True)
                self._cleanup()
                print(f"[SHAPE_PROFILER] File saved to {self.output_file}. Re-raising exception.", flush=True)
            else:
                print(f"[SHAPE_PROFILER] Not logging because: should_log={self.should_log}, in_forward={self._in_forward_pass}, skip={should_skip_logging}", flush=True)
            
            # Re-raise the error
            raise
        
        # Log successful operation
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
            named_args = {}
            
            # Try to get function signature to map args to parameter names
            sig = None
            try:
                sig = inspect.signature(func)
            except (ValueError, TypeError):
                # Some torch operations don't have inspectable signatures
                pass
            
            if sig:
                # Map positional args and kwargs to parameter names
                param_names = list(sig.parameters.keys())
                bound_args = None
                try:
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                except TypeError:
                    # If binding fails, fall back to positional mapping
                    pass
                
                if bound_args:
                    # bound_args.arguments includes both positional args and kwargs
                    for param_name, arg_value in bound_args.arguments.items():
                        shape = self._extract_shapes(arg_value)
                        # Include parameter even if shape is None (for non-tensor args like scalars)
                        # This ensures kwargs names are always visible
                        if shape is not None:
                            named_args[param_name] = shape
                        elif param_name in kwargs:
                            # Include kwargs even if they're not tensors (e.g., scalars, None)
                            # This ensures kwargs names are visible
                            named_args[param_name] = None
                else:
                    # Fallback: map positional args by index, then add kwargs separately
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            shape = self._extract_shapes(arg)
                            if shape is not None:
                                named_args[param_names[i]] = shape
                    # Add kwargs separately in fallback case
                    for kwarg_name, kwarg_value in kwargs.items():
                        shape = self._extract_shapes(kwarg_value)
                        # Always include kwargs names, even if not tensors
                        if shape is not None:
                            named_args[kwarg_name] = shape
                        else:
                            named_args[kwarg_name] = None
            else:
                # No signature available: use positional indices and kwargs
                for i, arg in enumerate(args):
                    shape = self._extract_shapes(arg)
                    if shape is not None:
                        named_args[f"arg{i}"] = shape
                # Add kwargs (they already have names)
                # Always include kwargs names, even if not tensors
                for kwarg_name, kwarg_value in kwargs.items():
                    shape = self._extract_shapes(kwarg_value)
                    if shape is not None:
                        named_args[kwarg_name] = shape
                    else:
                        named_args[kwarg_name] = None
            
            return named_args if named_args else None
        except Exception as e:
            # Fallback to simple extraction without names
            result = {}
            input_shapes = self._extract_shapes(args)
            if input_shapes:
                result["args"] = input_shapes
            # Always include kwargs with their names, even if not tensors
            if kwargs:
                for kwarg_name, kwarg_value in kwargs.items():
                    shape = self._extract_shapes(kwarg_value)
                    if shape is not None:
                        result[kwarg_name] = shape
                    else:
                        # Include kwargs even if not tensors (e.g., scalars, None)
                        result[kwarg_name] = None
            return result if result else None

    def _log_operation(self, func, named_inputs, result, error_info=None):
        """Log operation with minimal overhead."""
        # Skip warmup forward passes
        if self._forward_pass_count <= self.skip_first_n_forward_passes:
            return
        
        # Skip logging if we're limiting to first N forward passes after warmup and this exceeds that
        effective_pass = self._forward_pass_count - self.skip_first_n_forward_passes
        if self.log_first_n_forward_passes > 0 and effective_pass > self.log_first_n_forward_passes:
            return
        
        output_shapes = self._extract_shapes(result) if result is not None else None
        # Log if any shapes present OR if there's an error
        if named_inputs or output_shapes or error_info:
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
                    # Add error info if present
                    if error_info:
                        log_entry["error"] = error_info
                        print(f"[SHAPE_PROFILER] LOGGING ERROR: {error_info}", flush=True)
                    
                    # Write and flush immediately
                    json_str = json.dumps(log_entry) + "\n"
                    self.file_handle.write(json_str)
                    self.file_handle.flush()
                    os.fsync(self.file_handle.fileno())
                    
                    # Print every 100 ops to show progress
                    if self.call_count % 100 == 0 or error_info:
                        print(f"[SHAPE_PROFILER] Logged op #{self.call_count} to {self.output_file}", flush=True)
                        
                except Exception as log_err:
                    print(f"[SHAPE_PROFILER] ERROR logging operation: {log_err}", flush=True)
                    import traceback
                    traceback.print_exc()

    def __enter__(self):
        super().__enter__()
        if self.should_log:
            self.call_count = 0
            self.op_counts.clear()
            self._forward_pass_count = 0
            self._in_forward_pass = False
            # Open file in write mode, we'll flush after every write
            self.file_handle = open(self.output_file, "w")
            atexit.register(self._cleanup)
            # Register signal handlers for cleanup
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGQUIT, self._signal_handler)
            print(f"[Rank {self.current_rank}] Shape profiling enabled → {self.output_file}", flush=True)
            print(f"[SHAPE_PROFILER] File opened: {self.output_file}, skip_warmup={self.skip_first_n_forward_passes}, log_n={self.log_first_n_forward_passes}", flush=True)
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
                print(f"[SHAPE_PROFILER] Flushing and closing {self.output_file}...", flush=True)
                self.file_handle.flush()
                os.fsync(self.file_handle.fileno())  # Force OS write
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
            except Exception as e:
                print(f"[SHAPE_PROFILER] Error during cleanup: {e}", flush=True)
                pass


