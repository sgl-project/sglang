# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Device, backend, and CUDA/HIP/driver version detection; device memory, stream, and capability queries."""

from __future__ import annotations

import builtins
import importlib
import logging
import os
import platform
import re
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from ctypes import byref
from functools import lru_cache, partial
from importlib.util import find_spec
from typing import Any, Iterator, List, Optional, Tuple

import psutil
import torch

from sglang.srt.environ import envs
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import check_pkg_version_at_least, get_bool_env_var, is_mlu

logger = logging.getLogger(__name__)

__all__ = [
    "is_hip",
    "FP8_E4M3_MAX",
    "FP8_E4M3_MIN",
    "is_cuda",
    "is_cuda_alike",
    "is_hpu",
    "is_xpu",
    "register_xpu_device_properties_for_dynamo",
    "is_npu",
    "is_host_cpu_x86",
    "is_host_cpu_arm64",
    "is_cpu",
    "is_musa",
    "is_mps",
    "is_sail",
    "is_float4_e2m1fn_x2",
    "get_cuda_version",
    "device_context",
    "is_hopper_with_cuda_12_3",
    "is_blackwell_supported",
    "is_blackwell",
    "is_sm120_supported",
    "is_sm100_supported",
    "is_sm100_or_sm110_supported",
    "is_sm110_supported",
    "is_sm80_supported",
    "is_sm90_supported",
    "is_sm120",
    "is_sm121",
    "is_amx_tile_supported",
    "cpu_has_amx_support",
    "use_intel_amx_backend",
    "xpu_has_xmx_support",
    "is_flashinfer_available",
    "is_nvidia_cublas_version_ge_12_9",
    "empty_device_cache",
    "get_available_gpu_memory",
    "is_pin_memory_available",
    "async_h2d",
    "async_h2d_stack",
    "async_d2h",
    "get_dispatch_device_backend",
    "get_device_module",
    "create_device_stream",
    "device_stream_context",
    "is_device_stream_capturing",
    "create_device_event",
    "get_amdgpu_memory_capacity",
    "get_device_sm",
    "get_nvgpu_memory_capacity",
    "get_hpu_memory_capacity",
    "get_npu_memory_capacity",
    "get_instance_memory_status",
    "get_free_cpu_memory",
    "get_mlu_memory_capacity",
    "get_cpu_memory_capacity",
    "get_xpu_memory_capacity",
    "get_mps_memory_capacity",
    "get_mtgpu_memory_capacity",
    "get_device_memory_capacity",
    "get_device_name",
    "is_mnnvl_fabric_device",
    "is_fi_a2a_supported",
    "is_habana_available",
    "get_device",
    "get_device_count",
    "get_device_core_count",
    "get_device_capability",
    "get_compiler_backend",
    "set_cuda_arch",
    "is_gfx95_supported",
    "is_gfx942_supported",
    "is_gfx115_supported",
    "use_gfx115_kernels",
    "is_gfx1250_supported",
    "get_hip_version",
    "get_nvidia_driver_version",
    "get_nvidia_driver_version_str",
    "check_cuda_result",
    "get_cuda_driver_bindings",
    "maybe_reindex_device_id",
    "cached_device_index",
    "get_current_device_stream_fast",
    "parse_lscpu_topology",
    "get_physical_cpus_by_numa_or_block",
    "get_cpu_ids_by_node",
    "is_shm_available",
    "is_confidential_compute",
    "init_cublas",
]


# https://pytorch.org/docs/stable/notes/hip.html#checking-for-hip
@lru_cache(maxsize=1)
def is_hip() -> bool:
    return torch.version.hip is not None and torch.cuda.is_available()


if is_hip():
    HIP_FP8_E4M3_FNUZ_MAX = 224.0
    FP8_E4M3_MAX = HIP_FP8_E4M3_FNUZ_MAX
else:
    FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


FP8_E4M3_MIN = -FP8_E4M3_MAX


builtins.FP8_E4M3_MAX = FP8_E4M3_MAX


builtins.FP8_E4M3_MIN = FP8_E4M3_MIN


@lru_cache(maxsize=1)
def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda is not None


@lru_cache(maxsize=1)
def is_cuda_alike():
    return is_cuda() or is_hip() or is_sail()


@lru_cache(maxsize=1)
def is_hpu() -> bool:
    return hasattr(torch, "hpu") and torch.hpu.is_available()


@lru_cache(maxsize=1)
def is_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def register_xpu_device_properties_for_dynamo() -> None:
    if not is_xpu():
        return

    import torch._dynamo.utils as dynamo_utils

    xpu_props_type = getattr(torch.xpu, "_XpuDeviceProperties", None)
    if xpu_props_type is not None:
        dynamo_utils.common_constant_types.add(xpu_props_type)


@lru_cache(maxsize=1)
def is_npu() -> bool:
    if not hasattr(torch, "npu"):
        return False

    if not torch.npu.is_available():
        raise RuntimeError(
            "torch_npu detected, but NPU device is not available or visible."
        )

    return True


# Cache host CPU capability checks at import time to prevent TorchDynamo
# from graph-breaking on `torch.cpu.is_available()` (torch op returning non-Tensor).
_IS_HOST_CPU_X86 = (
    platform.machine().lower() in ("x86_64", "amd64", "i386", "i686")
    and hasattr(torch, "cpu")
    and torch.cpu.is_available()
)


_IS_HOST_CPU_ARM64 = (
    platform.machine().lower() in ("aarch64", "arm64")
    and hasattr(torch, "cpu")
    and torch.cpu.is_available()
)


@lru_cache(maxsize=1)
def is_host_cpu_x86() -> bool:
    return _IS_HOST_CPU_X86


def is_host_cpu_arm64() -> bool:
    return _IS_HOST_CPU_ARM64


@lru_cache(maxsize=1)
def is_cpu() -> bool:
    if os.getenv("SGLANG_USE_CPU_ENGINE", "0") != "1":
        return False
    return is_host_cpu_x86() or is_host_cpu_arm64()


@lru_cache(maxsize=1)
def is_musa() -> bool:
    if not hasattr(torch.version, "musa") or torch.version.musa is None:
        return False
    try:
        import torchada  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def is_mps() -> bool:
    return hasattr(torch, "mps") and torch.mps.is_available()


@lru_cache(maxsize=1)
def is_sail() -> bool:
    return torch.cuda.is_available() and "ZW" in torch.cuda.get_device_name()


def is_float4_e2m1fn_x2(dtype) -> bool:
    """Check if dtype is float4_e2m1fn_x2 and CUDA is available."""
    target_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return is_cuda() and dtype == target_dtype


def get_cuda_version():
    if torch.version.cuda:
        return tuple(map(int, torch.version.cuda.split(".")))
    return (0, 0)


@contextmanager
def device_context(device: torch.device):
    if device.type == "cpu" and is_cpu():
        with torch.device("cpu"):
            yield
    else:
        module = torch.get_device_module(device)
        if module is not None:
            with module.device(device.index):
                yield
        else:
            raise ValueError(f"Unknown device module: {device}")


def _check_cuda_device_version(
    device_capability_majors: List[int], cuda_version: Tuple[int, int]
):
    if not is_cuda():
        return False
    # get_device_sm() answers from NVML while torch.cuda is uninitialized, so
    # the platform probes evaluated at import time do not create a CUDA context.
    return (
        get_device_sm() // 10 in device_capability_majors
        and tuple(map(int, torch.version.cuda.split(".")[:2])) >= cuda_version
    )


is_hopper_with_cuda_12_3 = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[9], cuda_version=(12, 3)
    )
)


is_blackwell_supported = is_blackwell = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version,
        device_capability_majors=[10, 11, 12],
        cuda_version=(12, 8),
    )
)


is_sm120_supported = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[12], cuda_version=(12, 8)
    )
)


is_sm100_supported = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[10], cuda_version=(12, 8)
    )
)


# Datacenter Blackwell (SM100) plus SM110; excludes consumer Blackwell (SM120).
# This is the arch set flash_attn.cute accepts for the absorbed-MLA qv argument.
is_sm100_or_sm110_supported = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version,
        device_capability_majors=[10, 11],
        cuda_version=(12, 8),
    )
)


is_sm110_supported = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[11], cuda_version=(12, 8)
    )
)


is_sm80_supported = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[8], cuda_version=(11, 0)
    )
)


is_sm90_supported = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[9], cuda_version=(12, 3)
    )
)


# RTX Blackwell. Unlike is_sm120_supported(), this excludes SM121/GB10.
@lru_cache(maxsize=1)
def is_sm120() -> bool:
    return is_cuda() and torch.cuda.get_device_capability() == (12, 0)


# GB10 (DGX Spark and OEM equivalents). Not expressible via
# _check_cuda_device_version, which only matches on the major.
@lru_cache(maxsize=1)
def is_sm121() -> bool:
    return is_cuda() and torch.cuda.get_device_capability() == (12, 1)


@lru_cache(maxsize=1)
def _is_intel_amx_backend_available():
    try:
        import sgl_kernel  # noqa: F401

        return hasattr(torch.ops.sgl_kernel, "convert_weight_packed")
    except Exception:
        return False


try:
    # move torch.cpu._is_amx_tile_supported() from cpu_has_amx_support
    # to support torch compile
    is_amx_tile_supported = torch.cpu._is_amx_tile_supported()
except:
    is_amx_tile_supported = False


def cpu_has_amx_support():
    return is_amx_tile_supported and _is_intel_amx_backend_available()


def use_intel_amx_backend(layer):
    return getattr(layer, "use_intel_amx_backend", False)


def xpu_has_xmx_support():
    # TODO: update with XPU capability query
    if is_xpu():
        # currently only PVC/LNL/BMG supports F64, so we only support these now
        return torch.xpu.get_device_properties().has_fp64
    return False


@lru_cache(maxsize=1)
def is_flashinfer_available():
    """
    Check whether flashinfer is available.
    As of Oct. 6, 2024, it is only available on NVIDIA GPUs.
    """
    if not get_bool_env_var("SGLANG_IS_FLASHINFER_AVAILABLE", default="true"):
        return False
    return importlib.util.find_spec("flashinfer") is not None and is_cuda()


def is_nvidia_cublas_version_ge_12_9():
    """
    temporary fix for issue #11272 (cublas 12.9+)
    """
    for pkg in ("nvidia-cublas", "nvidia-cublas-cu12"):
        if check_pkg_version_at_least(pkg, "12.9"):
            return True
    return False


def empty_device_cache(device_module: Optional[Any] = None) -> bool:
    """Release unused cached blocks from the active device allocator.

    This does not clear SGLang KV/radix/request caches and does not free live
    tensors. It only forwards to the backend allocator's empty_cache hook when
    one is available.
    """

    if device_module is None:
        device_module = torch.get_device_module()

    empty_cache = getattr(device_module, "empty_cache", None)
    if empty_cache is None:
        return False

    empty_cache()
    return True


def get_available_gpu_memory(
    device, gpu_id, distributed=False, empty_cache=True, cpu_group=None
):
    """
    Get available memory for cuda:gpu_id device.
    When distributed is True, the available memory is the minimum available memory of all GPUs.
    """
    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        assert gpu_id < num_gpus

        if torch.cuda.current_device() != gpu_id:
            logger.warning(
                "current device is not %s, but %s, which may cause useless "
                "memory allocation for torch CUDA context.",
                gpu_id,
                torch.cuda.current_device(),
            )

        if empty_cache:
            empty_device_cache(torch.cuda)
        props = torch.cuda.get_device_properties(gpu_id)
        if props.is_integrated:
            # On these devices, which use sysmem as device mem, torch.cuda.mem_get_info()
            # only reports "free" memory, which can be lower than what is actually
            # available due to not including cache memory. So we use the system available
            # memory metric instead.
            free_gpu_memory = psutil.virtual_memory().available
        else:
            free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)

    elif device == "mlu":
        num_gpus = torch.mlu.device_count()
        assert gpu_id < num_gpus

        if torch.mlu.current_device() != gpu_id:
            logger.warning(
                "current device is not %s, but %s, which may cause useless "
                "memory allocation for torch MLU context.",
                gpu_id,
                torch.mlu.current_device(),
            )

        if empty_cache:
            empty_device_cache(torch.mlu)
        free_gpu_memory, _ = torch.mlu.mem_get_info(gpu_id)

    elif device == "xpu":
        num_gpus = torch.xpu.device_count()
        assert gpu_id < num_gpus

        if torch.xpu.current_device() != gpu_id:
            logger.warning(
                "current device is not %s, but %s, which may cause useless "
                "memory allocation for torch XPU context.",
                gpu_id,
                torch.xpu.current_device(),
            )

        if empty_cache:
            empty_device_cache(torch.xpu)
        used_memory = torch.xpu.memory_allocated(gpu_id)
        total_gpu_memory = torch.xpu.get_device_properties(gpu_id).total_memory
        free_gpu_memory = total_gpu_memory - used_memory

    elif device == "hpu":
        num_gpus = torch.hpu.device_count()
        assert gpu_id < num_gpus

        if torch.hpu.current_device() != gpu_id:
            logger.warning(
                "current device is not %s, but %s, which may cause useless "
                "memory allocation for torch HPU context.",
                gpu_id,
                torch.hpu.current_device(),
            )

        free_gpu_memory, total_gpu_memory = torch.hpu.mem_get_info()

    elif device == "cpu":
        # TODO: rename the variables in the current function to be not GPU specific
        total_free_memory = get_free_cpu_memory()
        n_node = len(get_cpu_ids_by_node()) or 1
        free_gpu_memory = round(total_free_memory / n_node, 3)
    elif device == "npu":
        num_gpus = torch.npu.device_count()
        assert gpu_id < num_gpus

        if torch.npu.current_device() != gpu_id:
            logger.warning(
                "current device is not %s, but %s, which may cause useless "
                "memory allocation for torch NPU context.",
                gpu_id,
                torch.npu.current_device(),
            )
        if empty_cache:
            empty_device_cache(torch.npu)
        if envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0:
            import zbal

            if not zbal.is_mix_alloc():
                free_gpu_memory, total_gpu_memory = zbal.zbal_module.mem_get_info()
            else:
                # mix mode fall back into npu mem info since gva may not inited yet
                free_gpu_memory, total_gpu_memory = torch.npu.mem_get_info()
        else:
            free_gpu_memory, total_gpu_memory = torch.npu.mem_get_info()
    elif device == "musa":
        num_gpus = torch.musa.device_count()
        assert gpu_id < num_gpus

        if torch.musa.current_device() != gpu_id:
            logger.warning(
                "current device is not %s, but %s, which may cause useless "
                "memory allocation for torch MUSA context.",
                gpu_id,
                torch.musa.current_device(),
            )
        if empty_cache:
            empty_device_cache(torch.musa)
        props = torch.musa.get_device_properties(gpu_id)
        if props.is_integrated:
            # On these devices, which use sysmem as device mem, torch.musa.mem_get_info()
            # only reports "free" memory, which can be lower than what is actually
            # available due to not including cache memory. So we use the system available
            # memory metric instead.
            free_gpu_memory = psutil.virtual_memory().available
        else:
            free_gpu_memory, _ = torch.musa.mem_get_info(gpu_id)
    elif device == "mps":
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx

        num_gpus = torch.mps.device_count()
        assert gpu_id < num_gpus

        if use_mlx():
            # Torch's allocator does not track MLX Metal allocations.
            free_gpu_memory = psutil.virtual_memory().available
        else:
            if empty_cache:
                empty_device_cache(torch.mps)
            # Bound free memory by host RAM and Metal's working-set limit.
            total_gpu_memory = torch.mps.recommended_max_memory()
            metal_headroom = max(
                0, total_gpu_memory - torch.mps.driver_allocated_memory()
            )
            free_gpu_memory = min(psutil.virtual_memory().available, metal_headroom)
    else:
        if not current_platform.is_out_of_tree():
            raise ValueError(
                f"Unsupported device type: {device!r}. "
                "If this is an OOT platform, ensure it is properly registered "
                "via the 'sglang.platform_plugins' entry point."
            )
        total_mem = current_platform.get_device_total_memory(gpu_id)
        used_mem = current_platform.get_current_memory_usage()
        free_gpu_memory = total_mem - used_mem

    if distributed:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32)
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
        )
        free_gpu_memory = tensor.item()

    return free_gpu_memory / (1 << 30)


def is_pin_memory_available(device=None) -> bool:
    # Under NVIDIA Confidential Computing host<->device copies go through a
    # bounce buffer, so host staging is pageable there.
    if is_confidential_compute():
        return False
    return current_platform.is_pin_memory_available(device)


def async_h2d(args: List, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """A host list on ``device``; staged in pinned memory so the copy is enqueued
    on the current stream instead of synchronizing like a pageable copy."""
    pin = is_pin_memory_available(device)
    return torch.tensor(args, dtype=dtype, pin_memory=pin).to(device, non_blocking=pin)


def async_h2d_stack(
    tensors: List[torch.Tensor], *, device: torch.device
) -> torch.Tensor:
    """``torch.stack(tensors)`` on ``device``; host tensors are stacked straight
    into pinned memory and copied like ``async_h2d``."""
    if not tensors[0].is_cpu:
        return torch.stack(tensors).to(device)
    pin = is_pin_memory_available(device)
    out = torch.empty(
        (len(tensors), *tensors[0].shape), dtype=tensors[0].dtype, pin_memory=pin
    )
    return torch.stack(tensors, out=out).to(device, non_blocking=pin)


def async_d2h(tensor: torch.Tensor) -> torch.Tensor:
    """Enqueue an accelerator-to-pinned-host copy on the current stream."""
    if tensor.device.type == "cpu":
        return tensor
    if not is_pin_memory_available(tensor.device):
        return tensor.to("cpu")
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    host.copy_(tensor, non_blocking=True)
    tensor.record_stream(torch.get_device_module(tensor.device).current_stream())
    return host


def get_dispatch_device_backend():
    if is_cuda_alike():
        dispatch_key = "CUDA"
    elif is_xpu():
        dispatch_key = "XPU"
    elif is_npu():
        dispatch_key = "NPU"
    else:
        raise RuntimeError("No supported accelerator (CUDA/XPU) available")
    return dispatch_key


@lru_cache(maxsize=1)
def get_device_module():
    # Resolve from the platform checks: torch.get_device_module() with no
    # argument initializes the CUDA runtime, which poisons fork() startup.
    if is_cuda() or is_hip():
        return torch.cuda
    if is_npu():
        return torch.npu
    if is_xpu():
        return torch.xpu
    if is_musa():
        return torch.musa
    # From torch 2.14, a bare torch.get_device_module() is torch.cuda on a CUDA wheel
    # even with no usable device; require an available accelerator instead.
    accelerator = torch.accelerator.current_accelerator(check_available=True)
    if accelerator is None:
        return torch.cpu
    return torch.get_device_module(accelerator)


def create_device_stream(device):
    """Create a device stream for the given device type."""
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return torch.get_device_module(device).Stream(device=device)


def device_stream_context(stream):
    """Return the appropriate stream context manager for ``stream``."""
    return torch.get_device_module(stream.device).stream(stream)


def is_device_stream_capturing(device: torch.device) -> bool:
    """Whether ``device``'s current stream is mid graph capture (False if unsupported)."""
    # Every platform declaring capabilities.graph_capture calls
    # device_module.is_current_stream_capturing() during capture, except CPU,
    # whose graph runner compiles instead of capturing a stream.
    if device.type != current_platform.device_type or device.type == "cpu":
        return False
    if not current_platform.capabilities.graph_capture:
        return False
    return torch.get_device_module(device).is_current_stream_capturing()


def create_device_event(device):
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return torch.get_device_module(device).Event()


def get_amdgpu_memory_capacity():
    try:
        # Run rocm-smi and capture the output
        result = subprocess.run(
            [
                "rocminfo | grep 'gfx' -A 100 | grep 'Pool 1' -A 5 | grep 'Size:' | awk '{print $2}'"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"rocm-smi error: {result.stderr.strip()}")

        # Parse the output to extract memory values in MiB
        memory_values = [
            float(mem.split("(")[0].strip()) / 1024
            for mem in result.stdout.strip().split("\n")
        ]

        if not memory_values:
            raise ValueError("No GPU memory values found.")

        # Return the minimum memory value
        return min(memory_values)

    except FileNotFoundError:
        raise RuntimeError(
            "rocm-smi not found. Ensure AMD ROCm drivers are installed and accessible."
        )


def _get_device_sm_via_nvml() -> Optional[int]:
    # Compute capability of torch device 0, read while torch.cuda stays
    # uninitialized; None when NVML cannot answer and the caller falls back.
    try:
        import pynvml
    except ImportError:
        logger.debug("get_device_sm: pynvml is not installed, using torch.cuda")
        return None
    # Private torch API, read defensively: it maps the torch ordinal to the NVML
    # index under CUDA_VISIBLE_DEVICES / MIG; absent or failing -> fall back.
    getter = getattr(torch.cuda, "_get_nvml_device_index", None)
    if getter is None:
        logger.debug(
            "get_device_sm: torch.cuda._get_nvml_device_index is missing, "
            "using torch.cuda"
        )
        return None
    try:
        idx = getter(0)
    except Exception:
        logger.debug(
            "get_device_sm: torch.cuda._get_nvml_device_index(0) failed, "
            "using torch.cuda",
            exc_info=True,
        )
        return None
    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        finally:
            pynvml.nvmlShutdown()
        return major * 10 + minor
    except Exception:
        logger.debug(
            "get_device_sm: NVML query failed, using torch.cuda", exc_info=True
        )
        return None


def get_device_sm():
    if torch.cuda.is_available() or is_musa():
        # Called at import time (e.g. by the DeepGEMM configurer): initializing
        # torch.cuda here would create a context and poison fork() startup.
        if not is_musa() and not torch.cuda.is_initialized():
            sm = _get_device_sm_via_nvml()
            if sm is not None:
                return sm
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    return 0


def _cuda_mem_fallback(reason: str) -> int:
    """Fallback to torch.cuda.mem_get_info() and return total GPU memory in MiB.

    Queries all visible CUDA devices and returns the minimum total memory,
    consistent with the nvidia-smi path that takes min(memory_values).

    Returns the total memory in MiB, or raises RuntimeError if CUDA is
    unavailable or mem_get_info() fails.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(reason)
    try:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            # Include the original failure reason for diagnostics
            raise RuntimeError(f"{reason} No CUDA devices found via torch.cuda.")
        memory_values = []
        for i in range(device_count):
            total = torch.cuda.mem_get_info(i)[1] // 1024 // 1024  # unit: MiB
            memory_values.append(total)
        result = min(memory_values)
        logger.warning(
            f"{reason} Falling back to torch.cuda.mem_get_info(). "
            f"Reported total GPU memory per device (MiB): {memory_values}, "
            f"using min: {result} MiB."
        )
        return result
    except (RuntimeError, ValueError, OSError) as e:
        raise RuntimeError(
            f"{reason} torch.cuda.mem_get_info() fallback also failed: {e}"
        ) from e


def get_nvgpu_memory_capacity():
    try:
        # Run nvidia-smi and capture the output
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            return _cuda_mem_fallback(
                f"nvidia-smi failed (exit code {result.returncode}: {result.stderr.strip()})."
            )

        # Parse the output to extract memory values
        memory_values = [
            float(mem)
            for mem in result.stdout.strip().split("\n")
            if re.match(r"^\d+(\.\d+)?$", mem.strip())
        ]

        if not memory_values:
            # Fallback when nvidia-smi returns no parseable values,
            # typically in NVIDIA MIG mode.
            return _cuda_mem_fallback(
                "Failed to get GPU memory capacity from nvidia-smi."
            )

        # Return the minimum memory value
        return min(memory_values)

    except FileNotFoundError:
        return _cuda_mem_fallback(
            "nvidia-smi not found. Ensure NVIDIA drivers are installed and accessible."
        )


def get_hpu_memory_capacity():
    try:
        # Run hl-smi and capture the output
        result = subprocess.run(
            ["hl-smi --query | grep 'Total'"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"hl-smi error: {result.stderr.strip()}")

        # Parse the output to extract memory values in MiB
        memory_values = [
            float(mem.split(" ")[-2]) for mem in result.stdout.strip().split("\n")
        ]

        if not memory_values:
            raise ValueError("No GPU memory values found.")

        # Return the minimum memory value
        return min(memory_values)

    except FileNotFoundError:
        raise RuntimeError(
            "hl-smi not found. Ensure Habana drivers are installed and accessible."
        )


def get_npu_memory_capacity():
    try:
        import torch_npu  # noqa: F401

        if envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0:
            return envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get()  # unit: MB
        else:
            return torch.npu.mem_get_info()[1] // 1024 // 1024  # unit: MB
    except ImportError:
        raise ImportError("torch_npu is required when run on npu device.")


def get_instance_memory_status(cgroup_file: str) -> Optional[int]:
    try:
        with open(cgroup_file, "r") as f:
            content = f.read().strip()
            # "max" or other non memory size content will be skipped
            if not content.isdigit():
                raise ValueError

            return int(content)
    except (PermissionError, FileNotFoundError, ValueError):
        return None


def get_free_cpu_memory() -> int:
    free_cpu_memory = psutil.virtual_memory().available
    instance_max_cpu_memory = get_instance_memory_status("/sys/fs/cgroup/memory.max")
    instance_current_cpu_memory = get_instance_memory_status(
        "/sys/fs/cgroup/memory.current"
    )
    if instance_max_cpu_memory is not None and instance_current_cpu_memory is not None:
        instance_free_cpu_memory = max(
            0, instance_max_cpu_memory - instance_current_cpu_memory
        )
        free_cpu_memory = min(free_cpu_memory, instance_free_cpu_memory)

    return free_cpu_memory


def get_mlu_memory_capacity():
    if not is_mlu():
        raise RuntimeError("MLU device is not available.")
    return torch.mlu.mem_get_info()[1] // 1024 // 1024  # unit: MB


def get_cpu_memory_capacity():
    # Per-rank memory capacity cannot be determined for customized core settings
    if os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", ""):
        logger.warning(
            "SGLANG_CPU_OMP_THREADS_BIND is set for explicit control of CPU cores for each TP rank."
            "Please beware that in this case the available memory amounts of the ranks cannot be determined in prior."
            "You need to set proper --max-total-tokens to avoid the out-of-memory error."
        )
        return None
    n_node = len(get_cpu_ids_by_node()) or 1
    total_cpu_memory = psutil.virtual_memory().total
    instance_max_cpu_memory = get_instance_memory_status("/sys/fs/cgroup/memory.max")
    if instance_max_cpu_memory is not None:
        total_cpu_memory = min(total_cpu_memory, instance_max_cpu_memory)
    per_node_mem = total_cpu_memory / n_node
    # Retrieved value in Byte, need MB
    return float(per_node_mem // (1 << 20))


def get_xpu_memory_capacity():
    try:
        if torch.xpu.is_available():
            return torch.xpu.mem_get_info()[1] // 1024 // 1024  # unit: MB
        raise ValueError("No GPU memory values found.")
    except AttributeError:
        raise RuntimeError("torch.xpu is not available.")


def get_mps_memory_capacity():
    # Metal's working-set limit is smaller than unified system RAM.
    try:
        if torch.mps.is_available():
            return torch.mps.recommended_max_memory() // 1024 // 1024  # unit: MB
        raise ValueError("No GPU memory values found.")
    except AttributeError:
        raise RuntimeError("torch.mps is not available.")


def get_mtgpu_memory_capacity():
    try:
        # Run mthreads-gmi and capture the output
        result = subprocess.run(
            [
                "mthreads-gmi --query | grep 'FB Memory Usage' -A 2 | grep 'Total' | awk -F':' '{print $2}' | awk '{print $1}' | sed 's/MiB//'"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"mthreads-gmi error: {result.stderr.strip()}")

        # Parse the output to extract memory values
        memory_values = [
            float(mem)
            for mem in result.stdout.strip().split("\n")
            if re.match(r"^\d+(\.\d+)?$", mem.strip())
        ]

        if not memory_values:
            # Fallback to torch.musa.mem_get_info() when failed to get memory capacity from mthreads-gmi.
            if hasattr(torch, "musa") and torch.musa.is_available():
                logger.warning(
                    "Failed to get GPU memory capacity from mthreads-gmi, falling back to torch.musa.mem_get_info()."
                )
                return torch.musa.mem_get_info()[1] // 1024 // 1024  # unit: MB
            raise ValueError("No GPU memory values found.")

        # Return the minimum memory value
        return min(memory_values)

    except FileNotFoundError:
        raise RuntimeError(
            "mthreads-gmi not found. Ensure Moore Threads drivers are installed and accessible."
        )


def get_device_memory_capacity(device: str = None):
    # OOT platforms provide their own memory query via the platform class.
    if current_platform.is_out_of_tree():
        mem_bytes = current_platform.get_device_total_memory()
        if mem_bytes:
            return mem_bytes / (1 << 20)  # bytes -> MiB
        return None
    if is_cuda():
        gpu_mem = get_nvgpu_memory_capacity()
    elif is_hip():
        gpu_mem = get_amdgpu_memory_capacity()
    elif device == "mps":
        gpu_mem = get_mps_memory_capacity()
    elif device == "hpu":
        gpu_mem = get_hpu_memory_capacity()
    elif device == "npu":
        gpu_mem = get_npu_memory_capacity()
    elif device == "cpu":
        gpu_mem = get_cpu_memory_capacity()
    elif device == "xpu":
        gpu_mem = get_xpu_memory_capacity()
    elif device == "musa":
        gpu_mem = get_mtgpu_memory_capacity()
    elif device == "mlu":
        gpu_mem = get_mlu_memory_capacity()
    else:
        # GPU memory is not known yet or no GPU is available.
        gpu_mem = None

    return gpu_mem


def get_device_name(device_id: int = 0) -> str:
    if hasattr(torch, "mps") and torch.mps.is_available():
        return torch.backends.mps.get_name()

    if is_mlu():
        return torch.mlu.get_device_name(device_id)

    if (hasattr(torch, "cuda") and torch.cuda.is_available()) or is_musa():
        return torch.cuda.get_device_name(device_id)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_device_name(device_id)

    if hasattr(torch, "hpu") and torch.hpu.is_available():
        return torch.hpu.get_device_name(device_id)

    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.get_device_name(device_id)


@lru_cache(maxsize=1)
def is_mnnvl_fabric_device() -> bool:
    """Whether the GPU sits on an MNNVL fabric (cross-node NVLink). Used to
    auto-select fabric-dependent communication paths (NCCL cuMem/MNNVL, custom
    all-reduce v2 multinode, DCP fi_a2a).

    The GB200/GB300 superchips are recognized by device name. Other NVLink
    fabric parts (SM107 systems report a generic device name) are recognized
    by NVML: the GPU has completed NVLink fabric registration."""
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        return False
    if is_sail():
        # SAIL is a fabric candidate device; PCCL decides the final communication mode.
        return True
    name = (torch.cuda.get_device_name(0) or "").upper()
    if any(tag in name for tag in ("GB200", "GB300")):
        return True
    from sglang.srt.utils.cuda_vmm_utils import is_gpu_fabric_ready

    return is_gpu_fabric_ready(torch.device("cuda", 0))


def is_fi_a2a_supported(
    *, dcp_size: int, tp_size: int, pp_size: int, nnodes: int
) -> bool:
    if not get_platform().is_sm100:
        return False
    if is_mnnvl_fabric_device():
        return True
    tp_size_per_node = tp_size // max(nnodes // pp_size, 1)
    return tp_size_per_node % dcp_size == 0


@lru_cache(maxsize=1)
def is_habana_available() -> bool:
    return find_spec("habana_frameworks") is not None


@lru_cache(maxsize=8)
def get_device(device_id: Optional[int] = None) -> str:
    if is_cpu():
        if cpu_has_amx_support():
            logger.info("Intel AMX is detected, using CPU with Intel AMX support.")
        else:
            logger.warning(
                "CPU device enabled, using torch native backend, low performance expected."
            )
        return "cpu"

    if is_mlu():
        if device_id is None:
            return "mlu"
        return "mlu:{}".format(device_id)

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if device_id is None:
            return "cuda"
        return "cuda:{}".format(device_id)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        if device_id is None:
            return "xpu"
        return "xpu:{}".format(device_id)

    if is_npu():
        if device_id is None:
            return "npu"
        return "npu:{}".format(device_id)

    if is_habana_available():
        try:
            import habana_frameworks.torch.hpu  # noqa: F401

            if torch.hpu.is_available():
                if device_id is None:
                    return "hpu"
                return "hpu:{}".format(device_id)
        except ImportError:
            raise ImportError(
                "Habana frameworks detected, but failed to import 'habana_frameworks.torch.hpu'."
            )

    if is_musa():
        if device_id is None:
            return "musa"
        return "musa:{}".format(device_id)

    if is_mps():
        if device_id is None:
            return "mps"
        return "mps:{}".format(device_id)

    try:
        return current_platform.get_device(device_id)
    except Exception:
        raise RuntimeError(
            "No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS, MLU) or platform plugin is available."
        )


@lru_cache(maxsize=1)
def get_device_count() -> int:
    if hasattr(torch, "mps") and torch.mps.is_available():
        try:
            return torch.mps.device_count()
        except RuntimeError:
            return 0

    if is_mlu():
        try:
            return torch.mlu.device_count()
        except RuntimeError:
            return 0

    if (hasattr(torch, "cuda") and torch.cuda.is_available()) or is_musa():
        try:
            return torch.cuda.device_count()
        except RuntimeError:
            return 0

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            return torch.xpu.device_count()
        except RuntimeError:
            return 0

    if is_habana_available():
        try:
            import habana_frameworks.torch.hpu  # noqa: F401

            if torch.hpu.is_available():
                return torch.hpu.device_count()
        except (ImportError, RuntimeError):
            return 0

    return 0  # No accelerators available


def get_device_core_count(device_id: int = 0) -> int:
    if (hasattr(torch, "cuda") and torch.cuda.is_available()) or is_musa():
        return torch.cuda.get_device_properties(device_id).multi_processor_count
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_device_properties(device_id).gpu_eu_count

    return 0


def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
    major, minor = None, None
    if (hasattr(torch, "cuda") and torch.cuda.is_available()) or is_musa():
        major, minor = torch.cuda.get_device_capability(device_id)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        major, minor, *_ = torch.xpu.get_device_capability(device_id)["version"].split(
            "."
        )
        # Currently XPU version does not contain capability information.
        major, minor = None, None

    if hasattr(torch, "hpu") and torch.hpu.is_available():
        try:
            # TODO(HandH1998): `get_device_capability` is not supported by `torch.hpu` for now.
            # Update this once the support is available.
            # major, minor = torch.hpu.get_device_capability(device_id)
            major, minor = None, None
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while getting device capability of hpu: {e}."
            ) from e

    return major, minor


def get_compiler_backend(mode=None) -> str:
    # OOT platforms provide their own compile backend.
    if current_platform.is_out_of_tree():
        return current_platform.get_compile_backend(mode)

    if hasattr(torch, "mps") and torch.mps.is_available():
        # MPS has no SGLang graph runner.
        return "eager"

    if hasattr(torch, "hpu") and torch.hpu.is_available():
        return "hpu_backend"

    if hasattr(torch, "npu") and torch.npu.is_available():
        try:
            import torchair
            import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce  # noqa: F401
            from torchair.configs.compiler_config import CompilerConfig
        except ImportError:
            raise ImportError(
                "NPU detected, but torchair package is not installed. "
                "Please install torchair for torch.compile support on NPU."
            )
        compiler_config = CompilerConfig()
        compiler_config.mode = "max-autotune"
        if mode == "npugraph_ex":
            compiler_config.mode = "reduce-overhead"
            compiler_config.debug.run_eagerly = True
        npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
        return npu_backend

    return "inductor"


def set_cuda_arch():
    if is_flashinfer_available():
        capability = torch.cuda.get_device_capability()
        arch = f"{capability[0]}.{capability[1]}"
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = (
            f"{arch}{'a' if capability[0] >= 9 else ''}"
        )


@lru_cache(maxsize=1)
def is_gfx95_supported():
    """Whether the device is an AMD gfx95 GPU (the MX-capable ROCm arch).

    False on every non-HIP build, so callers do not need their own is_hip().
    """
    if is_hip():
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx95"])
    else:
        return False


@lru_cache(maxsize=1)
def is_gfx942_supported():
    """
    Returns whether the current platform is AMD CDNA3 (gfx942 — MI300X / MI325X).
    """
    if is_hip():
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx942"])
    else:
        return False


@lru_cache(maxsize=1)
def is_gfx115_supported():
    """
    Returns whether the current platform is AMD RDNA3.5 (gfx115x, e.g. gfx1150, gfx1151).
    """
    if torch.version.hip and torch.cuda.is_available():
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return gcn_arch.startswith("gfx115")
    else:
        return False


@lru_cache(maxsize=1)
def use_gfx115_kernels() -> bool:
    """gfx115x and the gfx115x kernels not turned off (``SGLANG_DISABLE_GFX115_KERNELS``);
    read once."""
    from sglang.srt.environ import envs

    return is_gfx115_supported() and not envs.SGLANG_DISABLE_GFX115_KERNELS.get()


@lru_cache(maxsize=1)
def is_gfx1250_supported():
    """
    Returns whether the current platform is AMD RDNA4 (gfx1250).
    """
    if is_hip():
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx1250"])
    else:
        return False


def get_hip_version():
    if torch.version.hip:
        return tuple(map(int, torch.version.hip.split("-")[0].split(".")))
    return (0, 0, 0)


@lru_cache(maxsize=1)
def get_nvidia_driver_version() -> tuple:
    """Return the NVIDIA driver version as a tuple of ints, e.g. (595, 58, 3).
    Returns (0,) on failure."""
    version_str = get_nvidia_driver_version_str()
    if version_str is None:
        return (0,)
    try:
        return tuple(int(x) for x in version_str.split("."))
    except ValueError:
        return (0,)


@lru_cache(maxsize=1)
def get_nvidia_driver_version_str() -> str | None:
    """Return the NVIDIA driver version string, e.g. '595.58.03'.
    Returns None on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        version_str = result.stdout.strip().split("\n")[0].strip()
        return version_str if version_str else None
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def check_cuda_result(raw_output):
    import cuda.bindings.runtime as cuda_rt

    err, *results = raw_output
    if err != cuda_rt.cudaError_t.cudaSuccess:
        raise Exception(f"CUDA error: {err}")

    return results


def get_cuda_driver_bindings():
    try:
        from cuda.bindings import driver as cuda_driver
    except ImportError:
        from cuda import cuda as cuda_driver

    return cuda_driver


@contextmanager
def maybe_reindex_device_id(gpu_id: int) -> Iterator[int]:
    if not envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.get():
        yield gpu_id
        return

    with current_platform.reindex_device_id(gpu_id) as reindexed_device_id:
        yield reindexed_device_id


cached_device_index = -1


def get_current_device_stream_fast():
    global cached_device_index
    if cached_device_index == -1:
        cached_device_index = torch.get_device_module().current_device()
    return torch.get_device_module().current_stream(cached_device_index)


def parse_lscpu_topology():
    try:
        # Get CPU topology: CPU,Core,Socket,Node,Cache (L3 identifies CBB).
        output = subprocess.check_output(
            ["lscpu", "-p=CPU,Core,Socket,Node,Cache"], text=True
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error running 'lscpu': {e}")

    cpu_info = []
    for line in output.splitlines():
        # Parse only data lines (skip comments)
        if line.startswith("#"):
            continue
        parts = line.split(",")
        numa_parts = parts[:4]  # CPU, Core, Socket, Node
        cache_str = parts[-1] if len(parts) > 4 else "0"
        cache_parts = cache_str.split(":")
        line_info = list()
        # 5 elements expected: [cpu, core, socket, node, L3]
        for p in numa_parts:
            line_info.append(int(p) if p.isdigit() else 0)
        l3_cache_index = int(cache_parts[-1]) if cache_parts[-1] else 0
        line_info.append(l3_cache_index)
        cpu_info.append(line_info)

    # [(0,0,0,0,0),(1,1,0,0,0),...,(43,43,0,1,0),...,(256,0,0,0,0),...]
    return cpu_info


def get_physical_cpus_by_numa_or_block():
    cpu_info = parse_lscpu_topology()
    n_nodes = len({node for _, _, _, node, _ in cpu_info})
    n_blocks = len({block for _, _, _, _, block in cpu_info})
    use_block = n_blocks > n_nodes
    physical_by_group = defaultdict(dict)

    for cpu, core, socket_id, node, block in cpu_info:
        group = block if use_block else node
        key = (core, socket_id)
        if key not in physical_by_group[group]:
            # Pick the first CPU seen for that physical core.
            physical_by_group[group][key] = cpu

    # Retrieves CPUs that the current process is allowed to run on
    cpus_allowed_list = psutil.Process().cpu_affinity()
    return {
        group: set(core_to_cpu.values()).intersection(cpus_allowed_list)
        for group, core_to_cpu in physical_by_group.items()
    }


# Only physical cores are used. Logical cores are excluded.
@lru_cache(maxsize=1)
def get_cpu_ids_by_node():
    node_to_cpus = get_physical_cpus_by_numa_or_block()
    # Sort by NUMA node index or CBB block index
    # ['0,1,2,3', '4,5,6,7', '8,9,10,11', '12,13,14,15', '16,17,18,19', '20,21,22,23']
    return [
        ",".join(map(str, sorted(cpus)))
        for _, cpus in sorted(node_to_cpus.items())
        if cpus
    ]


def is_shm_available(dtype, world_size, local_size):
    return (
        (cpu_has_amx_support() or is_host_cpu_arm64())
        and dtype in [torch.bfloat16, torch.float16, torch.float]
        and world_size >= 1
        and world_size == local_size
    )


@lru_cache(maxsize=1)
def is_confidential_compute() -> bool:
    """Whether the GPU is running in NVIDIA Confidential Computing (CC) mode.

    Detected once via NVML and cached.
    Overridable with ``SGLANG_FORCE_CONFIDENTIAL_COMPUTE``.
    """
    forced = envs.SGLANG_FORCE_CONFIDENTIAL_COMPUTE.get()
    if forced is not None:
        return forced
    if not is_cuda():
        return False

    try:
        import pynvml
    except ImportError:
        logger.warning("pynvml not available; assuming Confidential Computing is off")
        return False

    cc_enabled = False
    try:
        pynvml.nvmlInit()

        # Hopper and newer supports a more nuanced query of confidential
        # compute settings
        cc_settings = pynvml.c_nvmlSystemConfComputeSettings_v1_t()
        ret = pynvml.nvmlSystemGetConfComputeSettings(byref(cc_settings))
        pynvml._nvmlCheckReturn(ret)
        # PPCIE implies CC
        cc_enabled = (
            cc_settings.ccFeature == pynvml.NVML_CC_SYSTEM_FEATURE_ENABLED
            or cc_settings.multiGpuMode == pynvml.NVML_CC_SYSTEM_MULTIGPU_PROTECTED_PCIE
        )
    except pynvml.NVMLError_NotSupported:
        # Simple query for older GPUs
        try:
            cc_state = pynvml.nvmlSystemGetConfComputeState()
            cc_enabled = cc_state.ccFeature == pynvml.NVML_CC_SYSTEM_FEATURE_ENABLED
        except pynvml.NVMLError as error:
            logger.warning(f"Error querying CC state: {error!s}")
    except pynvml.NVMLError as error:
        logger.warning(f"Error querying CC state: {error!s}")
    finally:
        # Shutdown
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            # Ignore shutdown errors
            pass

    return cc_enabled


def init_cublas():
    """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
    dtype = torch.float16
    device = "cuda"
    a = torch.ones((16, 16), dtype=dtype, device=device)
    b = torch.ones((16, 16), dtype=dtype, device=device)
    c = a @ b
    return c
