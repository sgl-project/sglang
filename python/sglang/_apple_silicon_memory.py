"""Unified Apple Silicon GPU memory queries for the MLX and torch-MPS runtimes.

SGLang has two runtimes on Apple Silicon: native MLX (``SGLANG_USE_MLX=1``)
and torch-MPS.  Both share unified memory, of which Metal can only keep
``recommendedMaxWorkingSetSize`` (~65-75% of system RAM) resident on the GPU,
but each runtime has its own allocator and cannot see the other's
allocations.  Questions like "how much GPU memory is used / still available"
must therefore be answered with the APIs of the runtime that actually
allocates: MLX when the MLX backend is active, ``torch.mps`` otherwise.
These helpers are that dispatch point (#21443).

All sizes are in bytes; callers convert units.  Only stdlib is imported at
module level so that ``sglang._mps_stub`` (installed from
``sglang/__init__``) and the ``multimodal_gen`` platform layer can depend on
this module without pulling in ``sglang.srt``.
"""

from __future__ import annotations


def _mlx_runtime_active() -> bool:
    """Return True when SGLang runs the native MLX backend."""
    try:
        from sglang.srt.utils.tensor_bridge import use_mlx
    except ImportError:
        # Degraded installs without sglang.srt still get torch-MPS reporting.
        return False
    return use_mlx()


def apple_gpu_working_set_size() -> int:
    """Metal's ``recommendedMaxWorkingSetSize`` in bytes (0 when unknown).

    This is how much unified memory the GPU can keep resident; allocating
    past it causes GPU paging and driver instability.
    """
    if _mlx_runtime_active():
        import mlx.core as mx

        return int(mx.device_info().get("max_recommended_working_set_size", 0))

    import torch

    recommended = getattr(torch.mps, "recommended_max_memory", None)
    if recommended is None:
        return 0
    return max(int(recommended()), 0)


def apple_gpu_allocated_memory() -> int:
    """Bytes of the working set held by the active runtime's allocator."""
    if _mlx_runtime_active():
        import mlx.core as mx

        # Active buffers plus the allocator cache: both stay resident, which
        # matches what torch.mps.driver_allocated_memory() reports.
        return int(mx.get_active_memory()) + int(mx.get_cache_memory())

    import torch

    return int(torch.mps.driver_allocated_memory())


def apple_gpu_empty_cache() -> None:
    """Release cached blocks held by the active runtime's allocator."""
    if _mlx_runtime_active():
        import mlx.core as mx

        mx.clear_cache()
        return

    import torch

    torch.mps.empty_cache()


def apple_gpu_total_memory() -> int:
    """GPU-usable memory in bytes: the working set, else total system RAM.

    The psutil fallback keeps memory planning alive when the working-set
    size is unknown (API missing or reporting 0) rather than advertising a
    zero-sized device.
    """
    working_set = apple_gpu_working_set_size()
    if working_set > 0:
        return working_set

    import psutil

    return int(psutil.virtual_memory().total)


def apple_gpu_available_memory(empty_cache: bool = False) -> int:
    """Bytes the active runtime can still safely allocate on the GPU.

    Reports ``min(system available, working-set headroom)``: unified memory
    is shared with the CPU, so free system RAM bounds any allocation, while
    the remaining working-set headroom bounds what the GPU may keep
    resident.  Falls back to system-available memory when the working-set
    size is unknown.
    """
    if empty_cache:
        apple_gpu_empty_cache()

    import psutil

    available = int(psutil.virtual_memory().available)
    working_set = apple_gpu_working_set_size()
    if working_set > 0:
        headroom = working_set - apple_gpu_allocated_memory()
        available = min(available, max(headroom, 0))
    return available
