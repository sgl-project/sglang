"""H2D copy speedup during weight loading (temporary pin).

Background: during load, many worker threads concurrently run dst.copy_(src),
where src is a pageable CPU tensor mmap'd from safetensors. For a pageable
source, cudaMemcpyAsync must first memcpy the data on the CPU into a driver
staging pinned buffer (CPU-bound, contends on a global driver lock) before the
DMA. This staging phase is amplified several-fold under the performance
governor and is the real cause of slow loads (GPU transfer and bandwidth are
not the bottleneck).

Approach: temporarily monkey-patch torch.Tensor.copy_ around load_weights,
routing only "CPU src -> CUDA dst" copies through "temporary pin + non_blocking",
released right after.
- Single choke point: covers copy_ in all models and weight loaders, no
  per-module changes needed.
- Temporary pin: pinned then freed right after the copy, so it won't blow up
  page-locked memory like persistently pinning an entire shard would.
- Scoped: active only during load; restored on with-exit, so the inference
  path's copy_ is unaffected.
"""

import contextlib
import functools
import platform
import subprocess
import threading

import torch

_wload_pin_tls = threading.local()


@functools.cache
def _is_grace_cpu_platform():
    if platform.machine().lower() not in ("aarch64", "arm64"):
        return False

    platform_texts = []
    paths = (
        "/sys/class/dmi/id/sys_vendor",
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/board_vendor",
        "/sys/class/dmi/id/board_name",
        "/proc/device-tree/model",
        "/sys/firmware/devicetree/base/model",
        "/proc/cpuinfo",
    )
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="ignore") as file:
                platform_texts.append(file.read().lower())
        except OSError:
            continue

    try:
        platform_texts.append(
            subprocess.check_output(
                ("lscpu",),
                encoding="utf-8",
                errors="ignore",
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).lower()
        )
    except Exception:
        pass

    platform_text = "\n".join(platform_texts)
    return "grace" in platform_text and "nvidia" in platform_text


@contextlib.contextmanager
def pin_h2d_copy_during_load():
    if not torch.cuda.is_available() or not _is_grace_cpu_platform():
        yield
        return

    _orig_copy_ = torch.Tensor.copy_

    def _patched_copy_(self, src, *args, **kwargs):
        # Only intercept "CPU (unpinned) src -> CUDA dst"; everything else
        # passes through unchanged.
        if (
            isinstance(src, torch.Tensor)
            and self.is_cuda
            and src.device.type == "cpu"
            and not src.is_pinned()
        ):
            try:
                device_idx = self.device.index
                if device_idx is None:
                    device_idx = torch.cuda.current_device()

                streams = getattr(_wload_pin_tls, "streams", None)
                if streams is None:
                    streams = _wload_pin_tls.streams = {}

                with torch.cuda.device(device_idx):
                    stream = streams.get(device_idx)
                    if stream is None:
                        stream = streams[device_idx] = torch.cuda.Stream()
                    pinned = src.pin_memory()  # temporary pin, freed after scope
                    with torch.cuda.stream(stream):
                        _orig_copy_(self, pinned, non_blocking=True)
                    stream.synchronize()
                return self
            except Exception:
                # Fall back to the original sync copy on any error for correctness.
                return _orig_copy_(self, src, *args, **kwargs)
        return _orig_copy_(self, src, *args, **kwargs)

    torch.Tensor.copy_ = _patched_copy_
    try:
        yield
    finally:
        torch.Tensor.copy_ = _orig_copy_
