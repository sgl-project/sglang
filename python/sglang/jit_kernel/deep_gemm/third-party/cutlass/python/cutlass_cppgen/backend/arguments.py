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

from math import prod
from typing import Union

from cutlass_cppgen.utils.lazy_import import lazy_import

cuda = lazy_import("cuda.cuda")
cudart = lazy_import("cuda.cudart")
import numpy as np

import cutlass_cppgen
from cutlass_cppgen.backend.frontend import CupyFrontend, NumpyFrontend, TorchFrontend
from cutlass_cppgen.backend.memory_manager import DevicePtrWrapper
from cutlass_cppgen.utils.datatypes import is_cupy_tensor, is_numpy_tensor, is_torch_tensor


class ArgumentBase:
    """
    Base class for operation arguments
    """

    def __init__(
        self,
        A: "Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor, cp.ndarray]",
        B: "Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor, cp.ndarray]",
        C: "Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor, cp.ndarray]",
        D: "Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor, cp.ndarray]",
        **kwargs,
    ) -> None:
        # tensor_C can be interpreted as the bias with bias=True in keyword args
        self.bias = kwargs.get("bias", False)

        self.stream = kwargs.get("stream", cuda.CUstream(0))

        # RMM buffers used to track tensor lifetime
        self.buffers = {}
        # Host tensor to copy the computed result back
        self.host_tensors = {}

        self.ptr_A = self.tensor_to_ptr(A, "A")
        self.ptr_B = self.tensor_to_ptr(B, "B")
        self.ptr_C = self.tensor_to_ptr(C, "C")
        self.ptr_D = self.tensor_to_ptr(D, "D", is_output=True)
        if C is not None:
            if not isinstance(C, cuda.CUdeviceptr):
                self.tensor_c_numel = prod(C.shape)

    def tensor_to_ptr(self, tensor, name, is_output=False):
        """
        Convert and remember the input tensor to cuda.CUdeviceptr used by cuda python
        For numpy.ndarray, it also remembers the host buffer for synchronization
        """
        if tensor is None:
            return cuda.CUdeviceptr(0)
        if is_numpy_tensor(tensor):
            if is_output:
                assert name
            self.buffers[name] = NumpyFrontend.argument(tensor, is_output)
            if is_output:
                self.host_tensors[name] = tensor
            return self.buffers[name].ptr
        elif is_torch_tensor(tensor):
            return TorchFrontend.argument(tensor)
        elif isinstance(tensor, cuda.CUdeviceptr):
            return tensor
        elif is_cupy_tensor(tensor):
            return CupyFrontend.argument(tensor)
        else:
            raise TypeError("Unsupported Frontend. Only support numpy and torch")

    def sync(self, stream_sync=True):
        if stream_sync:
            (err,) = cudart.cudaDeviceSynchronize()
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("CUDA Error %s" % str(err))

        for key in self.host_tensors.keys():
            host_tensor = self.host_tensors[key]
            (err,) = cuda.cuMemcpyDtoH(
                host_tensor,
                self.buffers[key].ptr,
                host_tensor.size * host_tensor.itemsize,
            )
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("CUDA Error %s" % str(err))

        self.free()

    def free(self):
        """
        Frees allocated device-side memory
        """
        # Free any device memory allocated manually
        if not cutlass_cppgen.use_rmm:
            for name, buf in self.buffers.items():
                if isinstance(buf, DevicePtrWrapper):
                    err, = cudart.cudaFree(buf.ptr)
                    if err != cudart.cudaError_t.cudaSuccess:
                        raise RuntimeError(f"cudaFree failed with error {err}")

            if hasattr(self, "workspace_buffer") and isinstance(self.workspace_buffer, DevicePtrWrapper):
                err, = cudart.cudaFree(self.workspace_buffer.ptr)
                if err != cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaFree failed with error {err}")
                del self.workspace_buffer
