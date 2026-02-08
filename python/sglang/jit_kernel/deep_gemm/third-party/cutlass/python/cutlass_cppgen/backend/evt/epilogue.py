#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Epilogue Visitor interface for compiling, and running visitor-based epilogue.
"""

import ctypes

from cutlass_cppgen.utils.lazy_import import lazy_import
cuda = lazy_import("cuda.cuda")
from cutlass_library import DataType
import numpy as np

from cutlass_cppgen.backend.epilogue import EpilogueFunctorBase
import cutlass_cppgen.backend.evt.backend
from cutlass_cppgen.backend.frontend import TensorFrontend
from cutlass_cppgen.utils.datatypes import is_numpy_tensor
from cutlass_cppgen.backend.evt.passes.util import cc_map


class EpilogueFunctorVisitor(EpilogueFunctorBase):
    """
    Apply an epilogue functor described by the epilogue EVT

    :param cc: compute capability
    :param visitor_frontend: user-provide visitor frontend

    """
    def __init__(self, cc: int, visitor, element_compute=DataType.f32) -> None:
        # Type of Emitter based on CC
        self.emit_cls = getattr(cutlass_cppgen.backend.evt.backend, f"Sm{cc_map[cc]}Emitter")

        # Visitor Types
        self.visitor = visitor
        self.graph = visitor.dag_ir

        # Data types
        self.element_epilogue = element_compute # element compute
        self.element_output = self.graph.get_node_meta('D').underlying_impl.element

        # Epilogue Thread Type
        epilogue_thread_type = self.visitor.epilogue_thread_type
        if cc_map[cc] in [90, 100]:
            self.arg_c_type = self.visitor.arg_c_type
            self.arg_d_type = self.visitor.arg_d_type
        output_names = self.visitor.return_names
        reduction_names = self.visitor.reduction_names

        # Epilogue stages specialized for sm80 kernel
        if cc == 80:
            if hasattr(self.visitor, "epilogue_stages"):
                self.epilogue_stages = self.visitor.epilogue_stages
                assert self.epilogue_stages <= 2, "Only supports Stages <=2 in SM80 Epilogue"

        # Epilogue Argument Type
        class _Arguments(ctypes.Structure):
            """
            Concepts:
            class _EpilogueArguments(ctypes.Structure):
                _fields_ = [
                    ("epilogue", _Arguments), <- this class
                    ("ptr_C", ctypes.c_void_p),
                    ("stride_C", StrideBatched_),
                    ("ptr_D", ctypes.c_void_p),
                    ("stride_D", StrideBatched_)
                ]
            """
            _fields_ = [
                ("output_op", epilogue_thread_type)
            ]

            def __init__(self, kwargs: dict) -> None:
                # The user-input kwargs is a dict of (name: tensors)
                # We first convert all of them to device pointers
                ptr_kwargs = {}
                for key in kwargs.keys():
                    is_output = key in output_names and key not in reduction_names
                    ptr_kwargs[key] = self.get_tensor_ptr(key, kwargs, is_output)
                # Initialize the thread arguments
                self.output_op = epilogue_thread_type(ptr_kwargs)

            def get_tensor_ptr(self, tensor_name, kwargs, is_output=False):
                """
                Helper function for extracting device pointer
                """
                # Skip the special tensors
                if cc in [90, 100]:
                    if tensor_name in ["C", "D"]:
                        return 0
                if tensor_name not in kwargs.keys():
                    raise ValueError(f"Tensor {tensor_name} is not provided.")
                tensor = kwargs[tensor_name]

                # For float scalar constant, directly return the value
                if isinstance(tensor, float):
                    return tensor

                # The tensor frontend returns a device buffer for np.ndarray
                # and device ptr for other frontends
                buffer_or_ptr = TensorFrontend.argument(tensor, is_output)
                if is_numpy_tensor(tensor):
                    # Remember the host tensor for later synchronization
                    setattr(self, f"{tensor_name}_buffer", buffer_or_ptr)
                    setattr(self, f"{tensor_name}_host", tensor)
                    return int(buffer_or_ptr.ptr)
                else:
                    return int(buffer_or_ptr)

            def sync(self):
                """
                Synchronize the results from device to host
                """
                for name in output_names:
                    if hasattr(self, f"{name}_host"):
                        host_tensor = getattr(self, f"{name}_host")
                        tensor_ptr = getattr(self, f"{name}_buffer").ptr
                        (err,) = cuda.cuMemcpyDtoH(
                            host_tensor,
                            tensor_ptr,
                            host_tensor.size * host_tensor.itemsize,
                        )
                        if err != cuda.CUresult.CUDA_SUCCESS:
                            raise RuntimeError("CUDA Error %s" % str(err))

        self.epilogue_type = _Arguments

    def emit(self, operation):
        """
        Emit the C++ code
        """
        emitter = self.emit_cls(operation, self.graph)
        return emitter.emit()

    def get_smem_size(self, tile_description):
        """
        Get the shared memory size in bytes
        """
        return self.visitor.get_smem_size(tile_description)
