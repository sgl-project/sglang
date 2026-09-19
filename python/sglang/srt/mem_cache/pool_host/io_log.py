"""Per-transfer logging for host-pool KV transfers (device <-> host).

A CUDA fault raised by a host<->device KV transfer is usually *asynchronous*: the
copy or kernel launch returns normally and the sticky error only surfaces at some
later synchronization point, so the traceback names an unrelated frame (PyTorch
warns that "the stacktrace below might be incorrect"). `CUDA_LAUNCH_BLOCKING=1`
does localize such a fault, but it serializes every launch in the process and
changes the timing being debugged.

Setting `SGLANG_DEBUG_HOST_POOL_IO=1` logs one line per transfer instead, so the
tail of the log still holds a record of the transfer that faulted. Only metadata
is read (shapes, dtypes, pointers, pool configuration) and no tensor value is ever
touched, so this neither synchronizes nor perturbs the asynchronous behavior it
is meant to explain, and the same record can be frozen into a crash dump.

Fields worth reading first:

* ``per_page_bytes`` -- size of one page's copy. The staged write-back kernel
  switches to the batch copy API at ``kLargeCopyThresholdBytes`` (128 KiB), and
  that API expects device addresses in its destination array.
* ``host_alloc`` -- ``alloc_with_host_register`` (mmap + ``cudaHostRegister``, the
  default) vs ``alloc_with_pin_memory`` (``cudaHostAlloc``). Registered host
  memory can have a device address different from its host virtual address, which
  the batch copy API does not translate.
* ``jit`` / ``wb_jit`` -- which implementation served the transfer (JIT kernel vs
  the AOT ``sgl_kernel.kvcacheio`` path).
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from typing import Any, Callable, Optional, TypeVar

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])

# staged_write_back.cuh switches to cudaMemcpyBatchAsync at this copy size.
BATCH_COPY_THRESHOLD_BYTES = 128 * 1024

_DIRECTION_BY_PREFIX = (("backup", "D2H"), ("load", "H2D"))


def host_pool_io_log_enabled() -> bool:
    """True when the caller asked for per-transfer host-pool logging."""
    return envs.SGLANG_DEBUG_HOST_POOL_IO.get()


def _direction_of(func_name: str) -> str:
    for prefix, direction in _DIRECTION_BY_PREFIX:
        if func_name.startswith(prefix):
            return direction
    return "?"


def _alloc_func_name(pool: Any) -> str:
    """Name of the host allocation function backing this pool.

    ``ALLOC_MEMORY_FUNCS`` is keyed by device *type* string, not ``torch.device``,
    so a ``torch.device`` key would silently report the default.
    """
    device = getattr(getattr(pool, "device_pool", None), "device", None)
    key = device.type if isinstance(device, torch.device) else str(device)
    try:
        from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS

        return ALLOC_MEMORY_FUNCS[key].__name__
    except Exception:  # pragma: no cover - logging must never break a transfer
        return "unknown"


def _tensor_brief(value: Any) -> Optional[str]:
    """Shape/dtype/device of a tensor; never reads its contents."""
    if not isinstance(value, torch.Tensor):
        return None
    return f"{tuple(value.shape)}:{str(value.dtype).replace('torch.', '')}@{value.device.type}"


def _host_base_ptr(pool: Any) -> Optional[str]:
    for name in ("k_buffer", "kv_buffer", "v_buffer"):
        buffer = getattr(pool, name, None)
        if isinstance(buffer, torch.Tensor):
            return f"{hex(buffer.data_ptr())}({name})"
        if isinstance(buffer, (list, tuple)) and buffer and torch.is_tensor(buffer[0]):
            return f"{hex(buffer[0].data_ptr())}({name}[0])"
    return None


def describe_transfer(pool: Any, func_name: str, bound: Any, elapsed_ms: float) -> str:
    """One CPU-only summary line for a single host-pool transfer."""
    page_size = getattr(pool, "page_size", None)
    layer_num = getattr(pool, "layer_num", None)
    stride = getattr(pool, "token_stride_size", None)
    per_page_bytes = getattr(pool, "size_per_token", None)
    if None not in (page_size, layer_num, stride):
        # Same quantity the JIT kernels pass as `first_page_bytes`.
        per_page_bytes = page_size * layer_num * stride

    fields = [
        func_name,
        f"direction={_direction_of(func_name)}",
        f"layout={getattr(pool, 'layout', None)}",
        f"page_size={page_size}",
        f"per_page_bytes={per_page_bytes}",
        f"batch_threshold={per_page_bytes is not None and per_page_bytes >= BATCH_COPY_THRESHOLD_BYTES}",
        f"host_alloc={_alloc_func_name(pool)}",
        f"host_base={_host_base_ptr(pool)}",
        f"jit={getattr(pool, 'can_use_jit', None)}",
        f"wb_jit={getattr(pool, 'can_use_write_back_jit', None)}",
    ]

    indices = []
    for name, value in sorted(bound.arguments.items()):
        brief = _tensor_brief(value)
        if brief is not None:
            indices.append(f"{name}={brief}")
    fields.append("indices[" + ",".join(indices) + "]")

    for name in ("layer_id", "io_backend", "is_draft"):
        if name in bound.arguments:
            fields.append(f"{name}={bound.arguments[name]}")

    fields.append(f"elapsed_ms={elapsed_ms:.3f}")
    return "[host_pool_io] " + " ".join(fields)


def log_host_pool_io(func: _F) -> _F:
    """Log one metadata-only line per call of a host-pool transfer method.

    A returning call logs at INFO -- an asynchronous fault is *not* reported here,
    because the call did return; the fault surfaces later, which is exactly the
    problem this log exists to work around. A raising call is logged at ERROR
    before the exception is propagated.
    """
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not host_pool_io_log_enabled():
            return func(self, *args, **kwargs)

        try:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
        except TypeError:  # pragma: no cover - defensive, signatures are fixed
            bound = None

        start = time.perf_counter()
        try:
            result = func(self, *args, **kwargs)
        except BaseException:
            if bound is not None:
                logger.error(
                    "%s raised",
                    describe_transfer(
                        self, func.__name__, bound, (time.perf_counter() - start) * 1e3
                    ),
                )
            raise

        if bound is not None:
            logger.info(
                "%s",
                describe_transfer(
                    self, func.__name__, bound, (time.perf_counter() - start) * 1e3
                ),
            )
        return result

    return wrapper  # type: ignore[return-value]
