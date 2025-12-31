"""
Memory utilities for GPU memory monitoring and optimization in SGLang.

This module provides utilities for monitoring GPU memory usage,
tracking memory allocations, and implementing memory-efficient
patterns for LLM inference.
"""

from contextlib import contextmanager
from typing import Optional, Dict, List, Generator
import torch


class GPUMemoryTracker:
    """
    Track GPU memory usage with support for multiple GPUs.
    
    This class provides methods to monitor current memory usage,
    peak memory usage, and memory allocation patterns across
    available CUDA devices.
    
    Example:
        >>> tracker = GPUMemoryTracker()
        >>> tracker.start_tracking()
        >>> # ... run your inference ...
        >>> stats = tracker.get_stats()
        >>> print(f"Peak memory: {stats['peak_allocated'] / 1e9:.2f} GB")
    """
    
    def __init__(self):
        self._tracking = False
        self._reset_stats()
    
    def _reset_stats(self) -> None:
        """Reset all memory statistics."""
        self._initial_allocated: Dict[int, int] = {}
        self._initial_reserved: Dict[int, int] = {}
        self._peak_allocated: Dict[int, int] = {}
        self._max_reserved: Dict[int, int] = {}
    
    def start_tracking(self) -> None:
        """
        Start tracking GPU memory usage.
        
        Records the initial memory state on all available CUDA devices
        to establish a baseline for subsequent measurements.
        """
        if not torch.cuda.is_available():
            return
        
        self._reset_stats()
        self._tracking = True
        
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
            self._initial_allocated[i] = torch.cuda.memory_allocated(i)
            self._initial_reserved[i] = torch.cuda.memory_reserved(i)
            self._peak_allocated[i] = self._initial_allocated[i]
            self._max_reserved[i] = self._initial_reserved[i]
    
    def stop_tracking(self) -> Dict[int, Dict[str, float]]:
        """
        Stop tracking and return memory statistics.
        
        Returns:
            Dictionary mapping device ID to memory statistics.
            Each device stats dict contains:
            - 'allocated_gb': Current allocated memory in GB
            - 'reserved_gb': Current reserved memory in GB
            - 'peak_allocated_gb': Peak allocated memory in GB
            - 'max_reserved_gb': Maximum reserved memory in GB
            - 'allocated_delta_gb': Change in allocated memory since start
        """
        if not self._tracking:
            return {}
        
        stats: Dict[int, Dict[str, float]] = {}
        
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
            current_allocated = torch.cuda.memory_allocated(i)
            current_reserved = torch.cuda.memory_reserved(i)
            
            stats[i] = {
                'allocated_gb': current_allocated / (1024**3),
                'reserved_gb': current_reserved / (1024**3),
                'peak_allocated_gb': self._peak_allocated[i] / (1024**3),
                'max_reserved_gb': self._max_reserved[i] / (1024**3),
                'allocated_delta_gb': (current_allocated - self._initial_allocated[i]) / (1024**3),
            }
        
        self._tracking = False
        return stats
    
    def get_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Get current memory statistics without stopping tracking.
        
        Returns:
            Same format as stop_tracking() for current snapshot.
        """
        if not self._tracking:
            return {}
        
        stats: Dict[int, Dict[str, float]] = {}
        
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
            current_allocated = torch.cuda.memory_allocated(i)
            current_reserved = torch.cuda.memory_reserved(i)
            
            # Update peak tracking
            self._peak_allocated[i] = max(self._peak_allocated[i], current_allocated)
            self._max_reserved[i] = max(self._max_reserved[i], current_reserved)
            
            stats[i] = {
                'allocated_gb': current_allocated / (1024**3),
                'reserved_gb': current_reserved / (1024**3),
                'peak_allocated_gb': self._peak_allocated[i] / (1024**3),
                'max_reserved_gb': self._max_reserved[i] / (1024**3),
                'allocated_delta_gb': (current_allocated - self._initial_allocated[i]) / (1024**3),
            }
        
        return stats
    
    def get_summary(self) -> str:
        """
        Get a formatted summary of memory statistics.
        
        Returns:
            Human-readable string summary of current GPU memory usage.
        """
        if not torch.cuda.is_available():
            return "CUDA is not available"
        
        stats = self.get_stats()
        if not stats:
            return "Not currently tracking. Call start_tracking() first."
        
        lines = ["GPU Memory Summary:"]
        for device_id, device_stats in stats.items():
            lines.append(f"  GPU {device_id}:")
            lines.append(f"    Current: {device_stats['allocated_gb']:.2f} GB allocated, "
                        f"{device_stats['reserved_gb']:.2f} GB reserved")
            lines.append(f"    Peak: {device_stats['peak_allocated_gb']:.2f} GB allocated, "
                        f"{device_stats['max_reserved_gb']:.2f} GB reserved")
            lines.append(f"    Delta: {device_stats['allocated_delta_gb']:.2f} GB since tracking started")
        
        return "\n".join(lines)


@contextmanager
def track_memory(
    name: Optional[str] = None,
    verbose: bool = True
) -> Generator[GPUMemoryTracker, None, None]:
    """
    Context manager for tracking GPU memory usage around a code block.
    
    This provides a simple way to measure memory impact of specific
    operations or code sections.
    
    Args:
        name: Optional name for the tracked section (used in output)
        verbose: Whether to print memory summary on exit
    
    Yields:
        GPUMemoryTracker instance for accessing statistics
    
    Example:
        >>> with track_memory("model_loading"):
        ...     model = load_model()
        ...     tokenizer = load_tokenizer()
        >>> # Memory stats available via tracker after context exits
        >>> # Or printed automatically if verbose=True
    """
    tracker = GPUMemoryTracker()
    tracker.start_tracking()
    
    try:
        yield tracker
    finally:
        stats = tracker.stop_tracking()
        
        if verbose:
            if name:
                print(f"\nMemory tracking for '{name}':")
            else:
                print("\nMemory tracking results:")
            
            if not stats:
                print("  CUDA not available")
            else:
                for device_id, device_stats in stats.items():
                    print(f"  GPU {device_id}: "
                          f"Delta = {device_stats['allocated_delta_gb']:.3f} GB, "
                          f"Peak = {device_stats['peak_allocated_gb']:.3f} GB")


def get_available_gpu_memory(gpu_id: int = 0) -> float:
    """
    Get available (free) GPU memory for a specific device.
    
    Args:
        gpu_id: The GPU device ID to query
    
    Returns:
        Available memory in GB, or -1 if CUDA is unavailable
        or the device doesn't exist.
    """
    if not torch.cuda.is_available():
        return -1.0
    
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        return -1.0
    
    torch.cuda.synchronize(gpu_id)
    props = torch.cuda.get_device_properties(gpu_id)
    reserved = torch.cuda.memory_reserved(gpu_id)
    allocated = torch.cuda.memory_allocated(gpu_id)
    free = props.total_memory - reserved
    
    return free / (1024**3)


def get_gpu_memory_utilization(gpu_id: int = 0) -> float:
    """
    Get GPU memory utilization as a percentage.
    
    Args:
        gpu_id: The GPU device ID to query
    
    Returns:
        Utilization percentage (0.0 to 100.0), or -1.0 if
        CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return -1.0
    
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        return -1.0
    
    torch.cuda.synchronize(gpu_id)
    allocated = torch.cuda.memory_allocated(gpu_id)
    props = torch.cuda.get_device_properties(gpu_id)
    
    return (allocated / props.total_memory) * 100.0


def clear_gpu_memory_cache(device: Optional[int] = None) -> None:
    """
    Clear GPU memory cache for specified device or all devices.
    
    This function releases cached memory back to the GPU, which
    can be useful before measuring memory usage or when switching
    between different model configurations.
    
    Args:
        device: Specific GPU ID, or None to clear all devices
    """
    if not torch.cuda.is_available():
        return
    
    if device is not None:
        if device >= 0 and device < torch.cuda.device_count():
            torch.cuda.empty_cache()
    else:
        for _ in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()


def estimate_model_memory_requirements(
    num_parameters: int,
    precision: str = "bf16",
    num_layers: Optional[int] = None,
    num_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> Dict[str, float]:
    """
    Estimate memory requirements for a model based on its parameters.
    
    This provides a rough estimate of memory needed for model weights
    and key activations during inference.
    
    Args:
        num_parameters: Total number of parameters in the model
        precision: Data precision ("fp32", "fp16", "bf16", "int8")
        num_layers: Number of transformer layers (for activation estimate)
        num_heads: Number of attention heads (for KV cache estimate)
        head_dim: Dimension of attention heads (for KV cache estimate)
    
    Returns:
        Dictionary with estimated memory requirements in GB:
        - 'weights_gb': Estimated weight memory
        - 'activations_gb': Estimated activation memory (if layers provided)
        - 'kv_cache_per_token_gb': KV cache per token (if heads and dim provided)
        - 'total_estimate_gb': Sum of applicable estimates
    
    Example:
        >>> # Estimate for a 7B model
        >>> estimates = estimate_model_memory_requirements(
        ...     num_parameters=7_000_000_000,
        ...     precision="bf16"
        ... )
        >>> print(f"Weights: {estimates['weights_gb']:.2f} GB")
    """
    # Bytes per parameter based on precision
    precision_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
    }
    
    bytes_per_param = precision_bytes.get(precision.lower(), 2)
    
    # Weight memory
    weight_memory = (num_parameters * bytes_per_param) / (1024**3)
    
    estimates = {
        "weights_gb": weight_memory,
    }
    
    # Activation memory estimate (rough approximation: ~2x params per layer for bf16)
    if num_layers is not None:
        activation_memory = (num_parameters * bytes_per_param * 2) / (1024**3)
        estimates["activations_gb"] = activation_memory
    
    # KV cache per token estimate
    if num_heads is not None and head_dim is not None:
        # 2 * num_layers * num_heads * head_dim * bytes_per_param
        # Assuming default 32 layers if not specified
        layers_for_kv = num_layers or 32
        kv_per_token = (2 * layers_for_kv * num_heads * head_dim * bytes_per_param) / (1024**3)
        estimates["kv_cache_per_token_gb"] = kv_per_token
    
    # Total estimate
    total = weight_memory
    if "activations_gb" in estimates:
        total += estimates["activations_gb"]
    estimates["total_estimate_gb"] = total
    
    return estimates
