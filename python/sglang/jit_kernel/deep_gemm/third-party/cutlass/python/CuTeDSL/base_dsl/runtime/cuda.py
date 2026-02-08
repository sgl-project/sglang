# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

"""
This module provides CUDA Python helper functions
"""


from functools import lru_cache
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import os
import ctypes

import cuda.bindings.driver as cuda
import cuda.bindings.nvrtc as nvrtc

# MLIR imports
from ..._mlir import ir
from ..._mlir.dialects import gpu

# Local module imports
from ..utils.logger import log as _log
from ..common import *
from .jit_arg_adapters import JitArgAdapterRegistry


# =============================================================================
# Utils
# =============================================================================


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise DSLRuntimeError("Unknown error type: {}".format(error))


def _get_gpu_arch_info(major, minor):
    """Get GPU architecture information and compatibility details."""
    gpu_arch_map = {
        (7, 0): ("Volta", "sm_70", ["sm_70"]),  # V100
        (7, 5): ("Turing", "sm_75", ["sm_75"]),  # RTX 20 Series, Quadro RTX
        (8, 0): ("Ampere", "sm_80", ["sm_80"]),  # A100
        (8, 6): ("Ampere", "sm_86", ["sm_86", "sm_80"]),  # RTX 30 Series
        (8, 9): ("Ada", "sm_89", ["sm_89", "sm_86"]),  # RTX 40 Series
        (8, 7): ("Ampere", "sm_87", ["sm_87", "sm_86", "sm_80"]),  # A10, A40
        (9, 0): ("Hopper", "sm_90a", ["sm_90a"]),  # H100
        (10, 0): ("Blackwell", "sm_100a", ["sm_100a"]),  # B200
    }
    return gpu_arch_map.get(
        (major, minor), ("Unknown", f"sm_{major}{minor}", [f"sm_{major}{minor}"])
    )


def get_compute_capability_major_minor(device_id: int = 0):
    """
    Returns the compute capability of the CUDA device as a tuple of (major, minor).
    For example: (8, 0) for Ampere, (9, 0) for Hopper, (10, 0) for Blackwell.
    Returns None on failure.
    """
    try:
        checkCudaErrors(cuda.cuInit(0))
        device = checkCudaErrors(cuda.cuDeviceGet(device_id))
        major = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            )
        )
        minor = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            )
        )
        return major, minor
    except RuntimeError as e:
        _log().info(f"Failed to get CUDA compute capability: {e}")
        return None, None


@dataclass
class DeviceInfo:
    """Data class to store CUDA device information."""

    device_count: int = 0
    current_device: int = 0
    device_name: Optional[str] = None
    major_version: Optional[int] = None
    minor_version: Optional[int] = None
    arch_name: Optional[str] = None
    sm_arch: Optional[str] = None
    compatible_archs: Optional[List[str]] = None
    memory_gb: Optional[float] = None
    target_arch: Optional[str] = None
    error_message: Optional[str] = None
    initialization_failed: bool = False

    def pretty_str(self) -> str:
        """
        Convert DeviceInfo to a formatted string for display.
        """
        info = ""

        if self.initialization_failed:
            return f"{Colors.BOLD}- CUDA initialization failed{Colors.RESET}"

        if self.error_message:
            return f"{Colors.BOLD}- Failed to get GPU info: {self.error_message}{Colors.RESET}"

        if self.device_count > 0:
            info += f"{Colors.BOLD}- CUDA devices available: {self.device_count} (current: {self.current_device})\n"

            if self.major_version is not None and self.minor_version is not None:
                info += f"- Architecture: {Colors.BLUE}{self.arch_name}{Colors.RESET} ({Colors.GREEN}{self.sm_arch}{Colors.RESET})\n"
                info += f"- Compatible SM archs: {Colors.GREEN}{', '.join(self.compatible_archs or [])}{Colors.RESET}\n"

                if self.memory_gb is not None:
                    info += f"- Total Memory: {Colors.BLUE}{self.memory_gb:.2f} GB{Colors.RESET}\n"

            else:
                info += f"- Compute capability: unknown\n"
                info += f"- SM arch: unknown{Colors.RESET}\n"
        else:
            info += f"- No devices available\n"

        return info


def get_device_info() -> DeviceInfo:
    """
    Get detailed information about CUDA devices.
    Returns a DeviceInfo dataclass with device information.
    """
    device_info = DeviceInfo()

    # Initialize CUDA if not already initialized
    try:
        result = cuda.cuInit(0)
        if result[0].value:  # Check for error
            device_info.initialization_failed = True
            return device_info
    except:
        pass

    try:
        # Get device count
        result = cuda.cuDeviceGetCount()
        device_info.device_count = result[1] if result[0].value == 0 else 0

        if device_info.device_count > 0:
            # Get current device
            try:
                result = cuda.cuCtxGetDevice()
                if result[0].value == 0:
                    device_info.current_device = result[1]
            except:
                pass

            # Get device name
            try:
                name_result = cuda.cuDeviceGetName(100, device_info.current_device)
                if name_result[0].value == 0:
                    device_info.device_name = name_result[1]
            except:
                pass

            # Get compute capability and architecture info
            try:
                major, minor = get_compute_capability_major_minor(
                    device_info.current_device
                )

                # Check if we successfully got the compute capability
                if major is not None and minor is not None:
                    device_info.major_version = major
                    device_info.minor_version = minor

                    arch_name, sm_arch, compatible_archs = _get_gpu_arch_info(
                        device_info.major_version, device_info.minor_version
                    )

                    device_info.arch_name = arch_name
                    device_info.sm_arch = sm_arch
                    device_info.compatible_archs = compatible_archs

                    # Get memory info
                    try:
                        total_mem = cuda.cuDeviceGetAttribute(
                            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_MEMORY,
                            device_info.current_device,
                        )
                        if total_mem[0].value == 0:
                            device_info.memory_gb = total_mem[1] / (
                                1024 * 1024 * 1024
                            )  # Convert to GB
                    except:
                        pass

            except Exception as e:
                pass  # Compute capability info will remain None

    except Exception as e:
        device_info.error_message = str(e)

    return device_info


def checkCudaErrors(result):
    """Check CUDA errors and provide detailed error messages."""
    if result[0].value:
        error_code = result[0].value
        error_name = _cudaGetErrorEnum(result[0])

        raise DSLCudaRuntimeError(error_code, error_name)

    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


# =============================================================================
# Driver Helpers
# =============================================================================


@lru_cache(maxsize=1)
def initialize_cuda_context(device_id: int = 0, flags: int = 0):
    """
    Initializes the CUDA context for a specified device.
    """
    # Initialize CUDA Driver API
    _log().info(f"cuInit {flags}")
    checkCudaErrors(cuda.cuInit(flags))
    # Retrieve handle for device
    _log().info(f"cuDeviceGet {device_id}")
    cuDevice = checkCudaErrors(cuda.cuDeviceGet(device_id))
    _log().info(f"{cuDevice} <-- cuDeviceGet")
    # Create context
    _log().info(f"cuCtxCreate {0} {cuDevice}")
    if cuda.CUDA_VERSION >= 13000:
        # Use cuCtxCreate_v4 API with explicit CUctxCreateParams None, since v2
        # and v3 API has been removed from CTK 13.
        # See https://github.com/NVIDIA/cuda-python/pull/792
        context = checkCudaErrors(cuda.cuCtxCreate(None, 0, cuDevice))
    else:
        context = checkCudaErrors(cuda.cuCtxCreate(0, cuDevice))
    _log().info(f"{context} <-- cuCtxCreate")

    return context


def load_cubin_module(cubin_file):
    """
    Loads a CUBIN file and returns the module.
    """
    # Load CUBIN file as binary data
    _log().info(f"read cubin {cubin_file}")
    with open(cubin_file, "rb") as f:
        cubin_data = f.read()
    # Load module data
    _log().info(f"cuModuleLoadData {np.char.array(cubin_data).ctypes.data}")
    module = checkCudaErrors(
        cuda.cuModuleLoadData(np.char.array(cubin_data).ctypes.data)
    )
    return module


def unload_cubin_module(module):
    """
    Unloads a CUBIN module.
    """
    _log().info(f"cuModuleUnload {module}")
    checkCudaErrors(cuda.cuModuleUnload(module))


def load_cubin_module_data(cubin_data):
    """
    Loads a CUBIN from data and returns the module.
    """
    # Load module data
    _log().info(f"cuModuleLoadData {np.char.array(cubin_data).ctypes.data}")
    module = checkCudaErrors(
        cuda.cuModuleLoadData(np.char.array(cubin_data).ctypes.data)
    )
    return module


def get_kernel_function(module, kernel_name):
    """
    Retrieves the kernel function from the module.
    """
    _log().info(f"cuModuleGetFunction {module} {kernel_name}")
    kernel = checkCudaErrors(
        cuda.cuModuleGetFunction(module, bytes(kernel_name, "utf-8"))
    )
    _log().info(f"{kernel} <-- cuModuleGetFunction")
    return kernel


def launch_kernel(kernel, grid_dims, block_dims, stream, smem_size, kernel_args=None):
    """
    Launches the CUDA kernel.
    """
    _log().info(
        f"cuLaunchKernel {kernel} grid={grid_dims} blocks={block_dims} smem_size={smem_size} stream={stream} {kernel_args}"
    )
    checkCudaErrors(
        cuda.cuLaunchKernel(
            kernel,
            grid_dims[0],
            grid_dims[1],
            grid_dims[2],
            block_dims[0],
            block_dims[1],
            block_dims[2],
            smem_size,  # Shared memory size
            stream,
            kernel_args,
            0,  # Extra parameters
        )
    )


def stream_sync(stream):
    """
    Synchronizes the CUDA stream.
    """
    _log().info(f"cuStreamSynchronize {stream}")
    checkCudaErrors(cuda.cuStreamSynchronize(stream))


def stream_create(id=0):
    """
    Creates the CUDA stream.
    """
    _log().info(f"cuStreamCreate {id}")
    stream = checkCudaErrors(cuda.cuStreamCreate(id))
    _log().info(f"{stream} <-- cuStreamCreate")
    return stream


def stream_destroy(stream):
    """
    Destroys the CUDA stream.
    """
    _log().info(f"cuStreamDestroy {stream}")
    checkCudaErrors(cuda.cuStreamDestroy(stream))


def context_destroy(context):
    """
    Destroys the CUDA context.
    """
    _log().info(f"cuCtxDestroy {context}")
    checkCudaErrors(cuda.cuCtxDestroy(context))


def allocate(size_in_bytes: int, stream=None):
    """
    Allocate device memory based on numpy host array size.
    """
    _log().info("Allocate size_in_bytes=[%s] stream=[%s]", size_in_bytes, stream)
    if stream is None:
        device_memory = checkCudaErrors(cuda.cuMemAlloc(size_in_bytes))
    else:
        device_memory = checkCudaErrors(cuda.cuMemAllocAsync(size_in_bytes, stream))
    _log().info("Allocated [%s]", device_memory)
    return device_memory


def deallocate(device_pointer, stream=None):
    """
    Deallocate the specified device memory pointer.
    """
    _log().info(
        "Deallocate device_pointer=[%s] stream=[%s]", hex(int(device_pointer)), stream
    )
    if stream is None:
        checkCudaErrors(cuda.cuMemFree(device_pointer))
    else:
        checkCudaErrors(cuda.cuMemFreeAsync(device_pointer, stream))


def memcpy_h2d(host_pointer, device_pointer, size_in_bytes, stream=None):
    """
    Copy data from host to device memory.
    """
    _log().info(
        "Copy host-to-device host_pointer[%s] device_ptr=[%s] size_in_bytes=[%s] stream=[%s]",
        hex(host_pointer),
        hex(int(device_pointer)),
        size_in_bytes,
        stream,
    )
    if stream is None:
        checkCudaErrors(cuda.cuMemcpyHtoD(device_pointer, host_pointer, size_in_bytes))
    else:
        checkCudaErrors(
            cuda.cuMemcpyHtoDAsync(device_pointer, host_pointer, size_in_bytes, stream)
        )


def memcpy_d2h(host_pointer, device_pointer, size_in_bytes, stream=None):
    """
    Copy data from device to host memory.
    """
    _log().info(
        "Copy device-host-to device_pointer=[%s] host_pointer[%s]  size_in_bytes=[%s] stream=[%s]",
        hex(int(device_pointer)),
        hex(host_pointer),
        size_in_bytes,
        stream,
    )
    if stream is None:
        checkCudaErrors(cuda.cuMemcpyDtoH(host_pointer, device_pointer, size_in_bytes))
    else:
        checkCudaErrors(
            cuda.cuMemcpyDtoHAsync(host_pointer, device_pointer, size_in_bytes, stream)
        )


def default_stream():
    return cuda.CUstream(0)


def get_driver_version():
    """
    Returns the CUDA driver version.
    """
    return checkCudaErrors(cuda.cuDriverGetVersion())


def set_kernel_attribute(kernel, attribute, value):
    """
    Sets a CUDA kernel attribute.
    """
    return checkCudaErrors(cuda.cuFuncSetAttribute(kernel, attribute, value))


@JitArgAdapterRegistry.register_jit_arg_adapter(cuda.CUstream)
class StreamAdapter:
    """
    Convert a CUDA stream to a stream representation for JIT arg generation.
    """

    def __init__(self, arg):
        self._arg = arg
        self._c_pointer = self._arg.getPtr()

    def __new_from_mlir_values__(self, values):
        assert len(values) == 1
        return values[0]

    def __c_pointers__(self):
        return [self._c_pointer]

    def __get_mlir_types__(self):
        return [gpu.AsyncTokenType.get()]
