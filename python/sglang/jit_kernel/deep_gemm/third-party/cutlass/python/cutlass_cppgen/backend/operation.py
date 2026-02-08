#################################################################################################
#
# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

import ctypes
from cutlass_cppgen.utils.lazy_import import lazy_import
cuda = lazy_import("cuda.cuda")

from cutlass_cppgen.backend.utils.device import device_cc

_supports_cluster_launch = None


def supports_cluster_launch():
    from cuda import __version__ 
    _version_splits = [int(x) for x in __version__.split("rc")[0].split(".post")[0].split(".")]
    global _supports_cluster_launch
    if _supports_cluster_launch is None:
        major, minor = _version_splits[0], _version_splits[1]
        _supports_cluster_launch = device_cc() in [90, 100, 101, 103] and (major > 11 or (major == 11 and minor >= 8))
    return _supports_cluster_launch


class LaunchConfiguration:
    def __init__(self, grid=[1, 1, 1], block=[1, 1, 1], smem=0):
        self.grid = grid
        self.block = block
        self.shared_memory_capacity = smem


class ExecutableOperation:
    def __init__(self, operation):
        self.operation = operation
        self.module = None
        self.kernel = None

    def name(self):
        return self.operation.procedural_name()

    def emit(self):
        return ""

    def can_implement(self, configuration, arguments):
        raise NotImplementedError()

    def get_host_workspace_size(self, arguments):
        raise NotImplementedError()

    def get_device_workspace_size(self, arguments):
        raise NotImplementedError()

    def plan(self, arguments):
        raise NotImplementedError()

    def initialize(self, host_workspace, device_workspace, launch_config, arguments, stream=None):
        raise NotImplementedError()

    def run_with_clusters(self, launch_config, kernel_params, stream=None):
        if not stream:
            stream = cuda.CUstream(0)
        if hasattr(self.operation, "tile_description") and hasattr(self.operation.tile_description, "cluster_shape"):
            attr = cuda.CUlaunchAttribute()
            attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = self.operation.tile_description.cluster_shape
            attr.id = cuda.CUstreamAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            attrs = [attr]

            # Allow for non-portable cluster sizes
            err, = cuda.cuFuncSetAttribute(
                self.kernel, cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
            if err != cuda.CUresult.CUDA_SUCCESS:
                return err
        else:
            attrs = []

        config = cuda.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = launch_config.grid
        config.blockDimX, config.blockDimY, config.blockDimZ = launch_config.block
        config.blockDimZ = launch_config.block[2]
        config.sharedMemBytes = launch_config.shared_memory_capacity
        config.hStream = stream
        config.attrs = attrs
        config.numAttrs = len(attrs)

        err, = cuda.cuLaunchKernelEx(
            config, f=self.kernel, kernelParams=kernel_params, extra=0)
        return err

    def run_without_clusters(self, launch_config, kernel_params, stream=None):
        if not stream:
            stream = cuda.CUstream(0)
        err, = cuda.cuLaunchKernel(
            self.kernel,
            launch_config.grid[0], launch_config.grid[1], launch_config.grid[2],
            launch_config.block[0], launch_config.block[1], launch_config.block[2],
            launch_config.shared_memory_capacity,
            stream,
            kernel_params,
            0)

        return err

    def run(self, host_workspace, device_workspace, launch_config, stream=None):
        if not stream:
            stream = cuda.CUstream(0)
        cArg = (ctypes.c_char * len(host_workspace)).from_buffer(host_workspace)
        packed = (ctypes.c_void_p * 1)()
        packed[0] = ctypes.addressof(cArg)

        if supports_cluster_launch():
            return self.run_with_clusters(launch_config, packed, stream)
        else:
            return self.run_without_clusters(launch_config, packed, stream)
