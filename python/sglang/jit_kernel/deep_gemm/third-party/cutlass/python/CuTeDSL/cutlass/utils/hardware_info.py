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

from cuda.bindings import driver, nvrtc

import cutlass.cute as cute

"""
This class is used to get the hardware info of given GPU device.
It provides methods to get the max active clusters for given cluster size.

Prerequisite:
- CUDA driver is initialized via `driver.cuInit` or other CUDA APIs.
- CUDA context is created via `driver.cuCtxCreate` or other CUDA APIs.

"""


class HardwareInfo:
    """
    device_id: CUDA device ID to get the hardware info.
    """

    def __init__(self, device_id: int = 0):
        count = self._checkCudaErrors(driver.cuDeviceGetCount())
        if device_id >= count:
            raise ValueError(
                f"Device ID {device_id} is out of range for device count {count}"
            )
        self.device_id = device_id
        self.device = self._checkCudaErrors(driver.cuDeviceGet(device_id))
        self.context = self._checkCudaErrors(driver.cuCtxGetCurrent())
        self.driver_version = self._checkCudaErrors(driver.cuDriverGetVersion())

    # Getting the max active clusters for a given cluster size
    def get_max_active_clusters(self, cluster_size: int) -> int:
        self._get_device_function()
        if self._cuda_driver_version_lt(11, 8):
            raise RuntimeError(
                "CUDA Driver version < 11.8, cannot get _max_active_clusters"
            )
        if cluster_size <= 0 or cluster_size > 32:
            raise ValueError(
                f"Cluster size must be between 1 and 32, {cluster_size} is not supported"
            )

        max_shared_memory_per_block = self._checkCudaErrors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                self.device,
            )
        )
        self._checkCudaErrors(
            driver.cuFuncSetAttribute(
                self.kernel,
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                max_shared_memory_per_block,
            )
        )
        max_dynamic_shared_memory = self._checkCudaErrors(
            driver.cuOccupancyAvailableDynamicSMemPerBlock(
                self.kernel, 1, 1  # numBlocks  # blockSize
            )
        )
        max_active_blocks = self._checkCudaErrors(
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                self.kernel, 1, max_dynamic_shared_memory  # blockSize,
            )
        )
        # allow non-portable cluster size to support detection of non-portable cluster size
        self._checkCudaErrors(
            driver.cuFuncSetAttribute(
                self.kernel,
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                1,
            )
        )
        # prepare launch configuration
        launch_config = driver.CUlaunchConfig()
        launch_config.blockDimX = 128
        launch_config.blockDimY = 1
        launch_config.blockDimZ = 1
        launch_config.sharedMemBytes = max_dynamic_shared_memory
        launch_config.numAttrs = 1
        # max possible cluster size is 32
        cluster_dims_attr = driver.CUlaunchAttribute()
        cluster_dims_attr.id = (
            driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        )
        value = driver.CUlaunchAttributeValue()
        value.clusterDim.x = cluster_size
        value.clusterDim.y = 1
        value.clusterDim.z = 1
        cluster_dims_attr.value = value
        launch_config.attrs = [cluster_dims_attr]
        launch_config.gridDimX = cluster_size
        launch_config.gridDimY = max_active_blocks
        launch_config.gridDimZ = 1

        num_clusters = self._checkCudaErrors(
            driver.cuOccupancyMaxActiveClusters(self.kernel, launch_config)
        )
        return num_clusters

    def get_l2_cache_size_in_bytes(self) -> int:
        return self._checkCudaErrors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
                self.device,
            )
        )

    def get_device_multiprocessor_count(self) -> int:
        return self._checkCudaErrors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                self.device,
            )
        )

    def _checkCudaErrors(self, result) -> None:
        if result[0].value:
            raise RuntimeError(
                "CUDA error code={}({})".format(
                    result[0].value, self._cudaGetErrorEnum(result[0])
                )
            )
        # CUDA APIs always return the status as the first element of the result tuple
        if len(result) == 1:
            return None
        elif len(result) == 2:
            return result[1]
        else:
            return result[1:]

    def _cudaGetErrorEnum(self, error) -> str:
        if isinstance(error, driver.CUresult):
            err, name = driver.cuGetErrorName(error)
            return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
        elif isinstance(error, nvrtc.nvrtcResult):
            return nvrtc.nvrtcGetErrorString(error)[1]
        else:
            raise RuntimeError("Unknown error type: {}".format(error))

    def _cuda_driver_version_ge(self, major: int, minor: int) -> bool:
        return self.driver_version >= (major * 1000 + 10 * minor)

    def _cuda_driver_version_lt(self, major: int, minor: int) -> bool:
        return not self._cuda_driver_version_ge(major, minor)

    @cute.kernel
    def _empty_kernel(self):
        return

    @cute.jit
    def _host_function(self):
        self._empty_kernel().launch(
            grid=[1, 1, 1],
            block=[1, 1, 1],
        )

    # get a empty kernel to compute occupancy
    def _get_device_function(self) -> None:
        self.compiled_kernel = cute.compile(self._host_function)
        self.module = next(iter(self.compiled_kernel.cuda_modules.modules)).cuda_module
        self.kernel = next(iter(self.compiled_kernel.cuda_modules.modules)).kernel_ptr
