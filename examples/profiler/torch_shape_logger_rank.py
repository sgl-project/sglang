"""
Rank-Aware Torch Dispatch-based Shape Logger

Minimal logger for profiling tensor shapes in TP workers.
Only logs operations on specified rank to avoid overhead.

Usage in TP worker (automatically activated by environment variables):
    SGLANG_PROFILE_SHAPES=1 SGLANG_PROFILE_SHAPES_RANK=0 \\
    SGLANG_PROFILE_SHAPES_FILE=shapes.jsonl python -m sglang.launch_server ...
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

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = func(*args, **kwargs)
        if self.should_log:
            self._log_operation(func, result)
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

    def _log_operation(self, func, result):
        """Log operation with minimal overhead."""
        output_shapes = self._extract_shapes(result)
        if output_shapes:
            self.call_count += 1
            op_name = str(func)
            self.op_counts[op_name] += 1
            
            if self.file_handle:
                try:
                    log_entry = {
                        "call_id": self.call_count,
                        "operation": op_name,
                        "outputs": output_shapes,
                    }
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
                    "unique_operations": len(self.op_counts),
                    "operation_counts": dict(sorted(self.op_counts.items(), key=lambda x: x[1], reverse=True)[:50]),
                }
                with open(summary_file, "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"[Rank {self.current_rank}] Profiled {self.call_count} ops → {self.output_file}")
            except Exception:
                pass
