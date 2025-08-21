"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Utilities for memory management and OOM handling."""

import logging
import functools
from typing import Optional, Callable, Any, Dict, Tuple

import torch

logger = logging.getLogger(__name__)


def get_gpu_memory_usage(device: Optional[torch.device] = None) -> Tuple[float, float]:
    """Get current GPU memory usage.
    
    Returns:
        Tuple of (used_memory_gb, total_memory_gb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    if device is None:
        device = torch.cuda.current_device()
    
    used_memory = torch.cuda.memory_allocated(device) / (1024**3)
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    return used_memory, total_memory


def get_memory_pressure_level(device: Optional[torch.device] = None) -> str:
    """Get current memory pressure level.
    
    Returns:
        "low", "medium", "high", or "critical"
    """
    used_memory, total_memory = get_gpu_memory_usage(device)
    if total_memory == 0:
        return "low"
    
    usage_ratio = used_memory / total_memory
    if usage_ratio < 0.7:
        return "low"
    elif usage_ratio < 0.85:
        return "medium"
    elif usage_ratio < 0.95:
        return "high"
    else:
        return "critical"


def suggest_memory_optimizations(
    current_config: Dict[str, Any],
    memory_pressure: str,
    oom_context: str = ""
) -> Dict[str, Any]:
    """Suggest memory optimization parameters based on current state.
    
    Args:
        current_config: Current server configuration
        memory_pressure: Current memory pressure level
        oom_context: Context where OOM occurred (e.g., "prefill", "decode")
        
    Returns:
        Dictionary of suggested parameter adjustments
    """
    suggestions = {}
    
    # Get current values with defaults
    mem_fraction_static = current_config.get("mem_fraction_static", 0.9)
    max_running_requests = current_config.get("max_running_requests", None)
    chunked_prefill_size = current_config.get("chunked_prefill_size", None)
    max_prefill_tokens = current_config.get("max_prefill_tokens", 16384)
    
    if memory_pressure in ["high", "critical"] or "out of memory" in oom_context.lower():
        # Reduce memory fraction for KV cache
        if mem_fraction_static > 0.7:
            suggestions["mem_fraction_static"] = max(0.7, mem_fraction_static - 0.1)
            
        # Reduce max running requests if not set or too high
        if max_running_requests is None or max_running_requests > 64:
            suggestions["max_running_requests"] = min(64, max_running_requests or 128)
        elif max_running_requests > 32:
            suggestions["max_running_requests"] = max(16, max_running_requests // 2)
            
        # Enable chunked prefill for long sequences
        if "prefill" in oom_context.lower():
            if chunked_prefill_size is None or chunked_prefill_size > 4096:
                suggestions["chunked_prefill_size"] = 4096
            elif chunked_prefill_size > 2048:
                suggestions["chunked_prefill_size"] = 2048
                
        # Reduce max prefill tokens
        if max_prefill_tokens > 8192:
            suggestions["max_prefill_tokens"] = 8192
            
    elif memory_pressure == "medium":
        # More conservative adjustments
        if mem_fraction_static > 0.85:
            suggestions["mem_fraction_static"] = 0.8
            
        if "prefill" in oom_context.lower() and chunked_prefill_size is None:
            suggestions["chunked_prefill_size"] = 8192
    
    return suggestions


def create_oom_error_message(
    error_context: str,
    tokens_requested: int,
    available_tokens: int,
    current_config: Dict[str, Any]
) -> str:
    """Create a detailed OOM error message with suggestions.
    
    Args:
        error_context: Context where OOM occurred
        tokens_requested: Number of tokens that were requested
        available_tokens: Number of tokens available
        current_config: Current server configuration
        
    Returns:
        Formatted error message with suggestions
    """
    memory_pressure = get_memory_pressure_level()
    used_memory, total_memory = get_gpu_memory_usage()
    suggestions = suggest_memory_optimizations(current_config, memory_pressure, error_context)
    
    error_msg = f"""
{error_context} out of memory.

Memory Status:
- GPU Memory Usage: {used_memory:.2f}GB / {total_memory:.2f}GB ({used_memory/total_memory*100:.1f}%)
- Memory Pressure: {memory_pressure}
- Tokens Requested: {tokens_requested}
- Tokens Available: {available_tokens}

Current Configuration:
- mem_fraction_static: {current_config.get('mem_fraction_static', 'auto')}
- max_running_requests: {current_config.get('max_running_requests', 'auto')}
- chunked_prefill_size: {current_config.get('chunked_prefill_size', 'disabled')}
- max_prefill_tokens: {current_config.get('max_prefill_tokens', 16384)}

Suggested Optimizations:"""

    if suggestions:
        for param, value in suggestions.items():
            error_msg += f"\n- Set --{param.replace('_', '-')} {value}"
    else:
        error_msg += "\n- Consider reducing batch size or sequence length"
        error_msg += "\n- Try enabling CPU offloading with --cpu-offload-gb"
    
    error_msg += f"""

Quick fixes:
- Restart server with --mem-fraction-static 0.7
- Use --chunked-prefill-size 4096 for long sequences
- Set --max-running-requests 32 to limit concurrency
"""
    
    return error_msg


def handle_cuda_oom_gracefully(func: Callable) -> Callable:
    """Decorator to handle CUDA OOM errors gracefully.
    
    This decorator catches torch.cuda.OutOfMemoryError and provides
    helpful error messages with recovery suggestions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            # Extract context from function name and arguments
            context = f"Function: {func.__name__}"
            if args and hasattr(args[0], '__class__'):
                context = f"{args[0].__class__.__name__}.{func.__name__}"
            
            logger.error(f"CUDA OOM in {context}: {str(e)}")
            
            # Get current memory status
            used_memory, total_memory = get_gpu_memory_usage()
            memory_pressure = get_memory_pressure_level()
            
            # Create enhanced error message
            enhanced_msg = f"""
CUDA Out of Memory Error in {context}

GPU Memory Status:
- Used: {used_memory:.2f}GB / {total_memory:.2f}GB ({used_memory/total_memory*100:.1f}%)
- Pressure Level: {memory_pressure}

Original Error: {str(e)}

Immediate Actions:
1. Reduce batch size or sequence length
2. Enable chunked prefill: --chunked-prefill-size 4096
3. Lower memory fraction: --mem-fraction-static 0.7
4. Limit concurrent requests: --max-running-requests 32

For persistent issues:
- Use smaller model or quantization
- Enable CPU offloading: --cpu-offload-gb 4
- Consider distributed inference across multiple GPUs
"""
            
            # Re-raise with enhanced message
            raise torch.cuda.OutOfMemoryError(enhanced_msg) from e
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Handle other OOM-related RuntimeErrors
                context = f"{func.__name__}"
                logger.error(f"Memory error in {context}: {str(e)}")
                
                enhanced_msg = f"""
Memory allocation error in {context}

Original Error: {str(e)}

This might be a CUDA memory issue. Try:
1. Reducing batch size
2. Using --chunked-prefill-size 4096
3. Setting --mem-fraction-static 0.7
4. Limiting requests with --max-running-requests 32
"""
                raise RuntimeError(enhanced_msg) from e
            else:
                # Re-raise other RuntimeErrors unchanged
                raise
    return wrapper


def log_memory_usage(device: Optional[torch.device] = None, context: str = ""):
    """Log current GPU memory usage."""
    used_memory, total_memory = get_gpu_memory_usage(device)
    pressure = get_memory_pressure_level(device)
    
    if context:
        context = f" [{context}]"
    
    logger.info(
        f"GPU Memory{context}: {used_memory:.2f}GB / {total_memory:.2f}GB "
        f"({used_memory/total_memory*100:.1f}%) - Pressure: {pressure}"
    )


def check_memory_before_allocation(required_gb: float, device: Optional[torch.device] = None) -> bool:
    """Check if there's enough memory before attempting allocation.
    
    Args:
        required_gb: Required memory in GB
        device: GPU device to check
        
    Returns:
        True if allocation is likely to succeed
    """
    if not torch.cuda.is_available():
        return True
    
    if device is None:
        device = torch.cuda.current_device()
    
    available_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    used_memory = torch.cuda.memory_allocated(device) / (1024**3)
    free_memory = available_memory - used_memory
    
    # Keep some safety margin (10% of total memory)
    safety_margin = available_memory * 0.1
    usable_memory = free_memory - safety_margin
    
    return usable_memory >= required_gb