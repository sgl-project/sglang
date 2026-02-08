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

"""
Utility functions for interacting with the device
"""
from __future__ import annotations

from cutlass_cppgen.utils.lazy_import import lazy_import
cuda = lazy_import("cuda.cuda")
cudart =  lazy_import("cuda.cudart")

import cutlass_cppgen
from cutlass_cppgen.utils.datatypes import is_cupy_tensor, is_numpy_tensor, is_torch_tensor


def check_cuda_errors(result: list):
    """
    Checks whether `result` contains a CUDA error raises the error as an exception, if so. Otherwise,
    returns the result contained in the remaining fields of `result`.

    :param result: the results of the `cudart` method, consisting of an error code and any method results
    :type result: list

    :return: non-error-code results from the `results` parameter
    """
    # `result` is of the format : (cudaError_t, result...)
    err = result[0]
    if err.value:
        raise RuntimeError("CUDA error: {}".format(cudart.cudaGetErrorName(err)))

    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def device_cc(device: int = -1) -> int:
    """
    Returns the compute capability of the device with ID `device`.

    :param device: ID of the device to query
    :type device: int

    :return: compute capability of the queried device (e.g., 80 for SM80)
    :rtype: int
    """
    if device == -1:
        device = cutlass_cppgen.device_id()

    deviceProp = check_cuda_errors(cudart.cudaGetDeviceProperties(device))
    major = str(deviceProp.major)
    minor = str(deviceProp.minor)
    return int(major + minor)


def device_sm_count(device: int = -1):
    if device == -1:
        device = cutlass_cppgen.device_id()
    err, device_sm_count = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device
    )
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise Exception(
            "Failed to retireve SM count. "
            f"cuDeviceGetAttribute() failed with error: {cuda.cuGetErrorString(err)[1]}"
        )

    return device_sm_count


def to_device_ptr(tensor) -> cuda.CUdeviceptr:
    """
    Converts a tensor to a CUdeviceptr

    :param tensor: tensor to convert
    :type tensor: np.ndarray | torch.Tensor | cp.ndarray | int

    :return: device pointer
    :rtype: cuda.CUdeviceptr
    """
    if is_numpy_tensor(tensor):
        ptr = cuda.CUdeviceptr(tensor.__array_interface__["data"][0])
    elif is_torch_tensor(tensor):
        ptr = cuda.CUdeviceptr(tensor.data_ptr())
    elif is_cupy_tensor(tensor):
        ptr = cuda.CUdeviceptr(int(tensor.data.ptr))
    elif isinstance(tensor, cuda.CUdeviceptr):
        ptr = tensor
    elif isinstance(tensor, int):
        ptr = cuda.CUdeviceptr(tensor)
    else:
        raise NotImplementedError(tensor)

    return ptr
