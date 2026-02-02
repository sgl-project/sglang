# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Union
import os
import atexit
import signal
import traceback

import torch
import torch.distributed

from .parallel_state import get_tp_group

# Communication volume tracking
_COMM_STATS = {
    "total_bytes": 0,
    "total_calls": 0,
    "by_shape": {},
    "by_caller": {},  # Track callers for small batches
}
_COMM_DEBUG = os.environ.get("SGLANG_COMM_DEBUG", "0") == "1"
_COMM_PRINT_INTERVAL = int(os.environ.get("SGLANG_COMM_PRINT_INTERVAL", "1000"))  # Print every N calls
_COMM_PRINT_ATEXIT = os.environ.get("SGLANG_COMM_PRINT_ATEXIT", "1") == "1"  # Print summary at exit
_COMM_TRACE_SMALL_BATCH = os.environ.get("SGLANG_COMM_TRACE_SMALL_BATCH", "0") == "1"  # Trace small batch calls
_SUMMARY_PRINTED = False


def reset_comm_stats():
    """Reset communication statistics."""
    global _COMM_STATS, _SUMMARY_PRINTED
    _COMM_STATS = {
        "total_bytes": 0,
        "total_calls": 0,
        "by_shape": {},
        "by_caller": {},
    }
    _SUMMARY_PRINTED = False


def get_comm_stats():
    """Get communication statistics."""
    return _COMM_STATS.copy()


def print_comm_summary(force=False):
    """Print communication summary."""
    global _SUMMARY_PRINTED
    if _SUMMARY_PRINTED and not force:
        return
    stats = _COMM_STATS
    if stats['total_calls'] == 0:
        return
    _SUMMARY_PRINTED = True
    print(f"\n{'='*60}", flush=True)
    print(f"[AllReduce Communication Summary]", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total calls: {stats['total_calls']}", flush=True)
    print(f"Total bytes: {stats['total_bytes']:,} ({stats['total_bytes'] / 1024 / 1024:.2f} MB)", flush=True)
    print(f"\nBy shape:", flush=True)
    for shape, info in sorted(stats['by_shape'].items(), key=lambda x: -x[1]['bytes']):
        print(f"  {shape}: {info['count']} calls, {info['bytes']:,} bytes ({info['bytes'] / 1024 / 1024:.4f} MB), dtype={info['dtype']}", flush=True)
    
    if stats['by_caller']:
        print(f"\nSmall batch callers (batch_size <= 4):", flush=True)
        for caller, count in sorted(stats['by_caller'].items(), key=lambda x: -x[1]):
            print(f"  {caller}: {count} calls", flush=True)
    print(f"{'='*60}\n", flush=True)


def _signal_handler(signum, frame):
    """Handle signals to print summary before exit."""
    print_comm_summary()
    # Re-raise the signal to allow normal termination
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _atexit_print_summary():
    """Print summary at program exit."""
    if _COMM_PRINT_ATEXIT:
        print_comm_summary()


# Register handlers
atexit.register(_atexit_print_summary)
try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
except:
    pass  # May fail in non-main thread


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    global _SUMMARY_PRINTED
    # Track communication volume
    comm_bytes = input_.numel() * input_.element_size()
    shape_key = str(tuple(input_.shape))
    batch_size = input_.shape[0] if len(input_.shape) > 0 else 1
    
    _COMM_STATS["total_bytes"] += comm_bytes
    _COMM_STATS["total_calls"] += 1
    
    if shape_key not in _COMM_STATS["by_shape"]:
        _COMM_STATS["by_shape"][shape_key] = {"count": 0, "bytes": 0, "dtype": str(input_.dtype)}
    _COMM_STATS["by_shape"][shape_key]["count"] += 1
    _COMM_STATS["by_shape"][shape_key]["bytes"] += comm_bytes
    
    # Track callers for small batch sizes
    if _COMM_TRACE_SMALL_BATCH and batch_size <= 4:
        stack = traceback.extract_stack()
        # Get the caller (skip this function and the immediate caller)
        for frame in reversed(stack[:-2]):
            if "sglang" in frame.filename and "communication_op" not in frame.filename:
                caller_key = f"{frame.filename.split('/')[-1]}:{frame.lineno}:{frame.name}"
                _COMM_STATS["by_caller"][caller_key] = _COMM_STATS["by_caller"].get(caller_key, 0) + 1
                break
    
    # Print per-call info if debug enabled
    if _COMM_DEBUG:
        print(f"[AllReduce] shape={tuple(input_.shape)}, dtype={input_.dtype}, bytes={comm_bytes}, MB={comm_bytes / 1024 / 1024:.4f}", flush=True)
    
    # Print periodic summary if interval is set
    if _COMM_PRINT_INTERVAL > 0 and _COMM_STATS["total_calls"] % _COMM_PRINT_INTERVAL == 0:
        _SUMMARY_PRINTED = False  # Allow printing again
        print_comm_summary(force=True)
    
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
