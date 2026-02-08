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

import numpy as np

import cutlass_cppgen
from cutlass_cppgen.utils.datatypes import is_numpy_tensor
from cutlass_cppgen.utils.lazy_import import lazy_import

if cutlass_cppgen.use_rmm:
    import rmm
else:
    cudart = lazy_import("cuda.cudart")


class PoolMemoryManager:
    def __init__(self, init_pool_size: int, max_pool_size: int) -> None:
        self.pool = rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(),
            initial_pool_size=init_pool_size,
            maximum_pool_size=max_pool_size
        )
        self.mr = rmm.mr.TrackingResourceAdaptor(self.pool)
        rmm.mr.set_current_device_resource(self.mr)

    def pool_size(self):
        return self.pool.pool_size()


class DevicePtrWrapper:
    """
    Wrapper around a pointer to device memory to provide a uniform interface with the RMM DeviceBuffer
    (at least in terms of the interface used by the CUTLASS Python interface)
    """
    def __init__(self, dev_ptr):
        self.dev_ptr = dev_ptr

    @property
    def ptr(self):
        return self.dev_ptr


def _todevice(host_data):
    """
    Helper for transferring host data to device memory
    """
    if cutlass_cppgen.use_rmm:
        return rmm.DeviceBuffer.to_device(host_data.tobytes())
    else:
        nbytes = len(host_data.tobytes())
        dev_ptr_wrapper = device_mem_alloc(nbytes)
        err, = cudart.cudaMemcpy(
            dev_ptr_wrapper.ptr,
            host_data.__array_interface__['data'][0],
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise Exception(f"cudaMemcpy failed with error {err}")
        return dev_ptr_wrapper


def todevice(host_data, dtype=np.float32):
    """
    Pass the host_data to device memory
    """
    if isinstance(host_data, list):
        return _todevice(np.array(host_data, dtype=dtype))
    elif is_numpy_tensor(host_data):
        return _todevice(host_data)


def device_mem_alloc(size):
    if cutlass_cppgen.use_rmm:
        return rmm.DeviceBuffer(size=size)
    else:
        err, ptr = cudart.cudaMalloc(size)
        if err != cudart.cudaError_t.cudaSuccess:
            raise Exception(f"cudaMalloc failed with error {err}")
        return DevicePtrWrapper(ptr)


def align_size(size, alignment=256):
    return ((size + alignment - 1) // alignment) * alignment


def create_memory_pool(init_pool_size=0, max_pool_size=2 ** 34):
    if cutlass_cppgen.use_rmm:
        memory_pool = PoolMemoryManager(init_pool_size=init_pool_size, max_pool_size=max_pool_size)
        return memory_pool
    else:
        return None
