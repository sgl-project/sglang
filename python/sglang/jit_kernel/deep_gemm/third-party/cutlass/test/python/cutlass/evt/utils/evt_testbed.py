################################################################################
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
################################################################################

"""
Testbed classes of EVT
"""

import torch
import unittest

import cutlass_cppgen
from cutlass_cppgen import Tensor
import cutlass_cppgen.backend.evt
from cutlass_cppgen.shape import GemmCoord
from cutlass_cppgen.utils.datatypes import torch_type
from cutlass_cppgen.utils.profiler import CUDAEventProfiler


class EVTReferenceModule:
    def __init__(self, layout_A, layout_B, layout_C, epilogue_visitor):
        self.layout_A = layout_A
        self.layout_B = layout_B
        self.layout_C = layout_C
        self.epilogue_visitor = epilogue_visitor

    def run(self, A, B, C, problem_size, alpha, beta, batch=1):
        if self.layout_A == cutlass_cppgen.LayoutType.RowMajor:
            A_row = A.view((batch, problem_size.m, problem_size.k))
        else:
            A_col = A.view((batch, problem_size.k, problem_size.m))
            A_row = torch.permute(A_col, (0, 2, 1))

        if self.layout_B == cutlass_cppgen.LayoutType.RowMajor:
            B_row = B.view((batch, problem_size.k, problem_size.n))
        else:
            B_col = B.view((batch, problem_size.n, problem_size.k))
            B_row = torch.permute(B_col, (0, 2, 1))

        if self.layout_C == cutlass_cppgen.LayoutType.RowMajor:
            C_row = C.view((batch, problem_size.m, problem_size.n))
        else:
            C_col = C.view((batch, problem_size.n, problem_size.m))
            C_row = torch.permute(C_col, (0, 2, 1))

        out_row = torch.matmul(A_row, B_row) * alpha + C_row * beta

        if self.layout_C == cutlass_cppgen.LayoutType.ColumnMajor:
            out = torch.permute(out_row, (0, 2, 1))
        else:
            out = out_row

        return torch.flatten(out)

    def __call__(self, A, B, C, problem_size, batch=1, epilogue_args=None):
        # Running the mainloop
        accum = self.run(
            A, B, C, problem_size, 1.0, 0.0, batch=batch
        ).reshape(batch, problem_size.m, problem_size.n)
        
        # Running the epilogue
        epilogue_args["accum"] = accum
        references = self.epilogue_visitor(**epilogue_args)
        
        # Return the results
        if not isinstance(references, tuple):
            references = (references,)
        return references
        

class EVTTestBed:
    """
    Epilogue Visitor Testbed
    """
    def __init__(self, element, evt_fn, example_inputs, profile=False, **kwargs) -> None:
        self.element = element
        layout = cutlass_cppgen.LayoutType.RowMajor
        self.example_inputs = example_inputs
        
        # Create the Gemm plan
        self.plan = cutlass_cppgen.op.Gemm(element=element, layout=layout, element_accumulator=torch.float32)
        
        if "tile_description" in kwargs:
            self.plan.tile_description = kwargs["tile_description"]
        
        if "swizzling_functor" in kwargs:
            self.plan.swizzling_functor = kwargs["swizzling_functor"]
        
        # Compile the epilogue visitor
        epilogue_visitor = cutlass_cppgen.epilogue.trace(evt_fn, example_inputs)
        if "epilogue_stages" in kwargs:
            epilogue_visitor.epilogue_stages = kwargs["epilogue_stages"]
        self.plan.epilogue_visitor = epilogue_visitor
        
        # Reference model
        self.reference_fn = EVTReferenceModule(layout, layout, layout, epilogue_visitor)
        
        self.profile = profile

    def get_torch_tensor(self, shape, dtype=None, fill=None):
        if dtype is None:
            dtype = self.element
        
        dtype = torch_type(dtype)
        if fill is None:
            return torch.ceil(
                torch.empty(size=shape, dtype=dtype, device="cuda").uniform_(-4.5, 3.5)
            )
        else:
            return torch.full(shape, fill, dtype=dtype, device="cuda")
    
    def verify(self, problem_size, input_keys, result_keys, batch_count=1):
        """
        Verify the results
        """
        problem_size = GemmCoord(*problem_size)

        # Initiate the GEMM arguments
        tensor_A = self.get_torch_tensor((batch_count, problem_size.m, problem_size.k))
        tensor_B = self.get_torch_tensor((batch_count, problem_size.k, problem_size.n))
        
        # Initialize the epilogue args
        epilogue_args = {}
        for key in self.example_inputs.keys():
            if key in input_keys:
                tensor = self.example_inputs[key]
                if isinstance(tensor, Tensor):
                    epilogue_args[key] = self.get_torch_tensor(tensor.shape, tensor.element)
                else:
                    epilogue_args[key] = tensor
            elif key in result_keys:
                tensor = self.example_inputs[key]
                if isinstance(tensor, Tensor):
                    if "max" in key:
                        fill = -1000
                    else:
                        fill = 0
                    epilogue_args[key] = self.get_torch_tensor(tensor.shape, tensor.element, fill=fill)
                else:
                    epilogue_args[key] = tensor
        
        tensor_D = epilogue_args["D"]
        if "C" in epilogue_args:
            tensor_C = epilogue_args["C"]
        else:
            tensor_C = tensor_D
        # Run the device kernel
        self.plan.run(tensor_A, tensor_B, tensor_C, tensor_D, visitor_args=epilogue_args)
        
        # Run the host reference
        evt_args_inputs = {}
        for key in input_keys:
            evt_args_inputs[key] = epilogue_args[key]
        
        reference_results = self.reference_fn(
            tensor_A, tensor_B, tensor_C, problem_size, batch_count, evt_args_inputs)
        
        # Compare the results
        for result, ref in zip(result_keys, reference_results):
            assert torch.equal(
                epilogue_args[result].flatten(), 
                ref.masked_fill(torch.isnan(ref), float('inf')).flatten())
        
        # Run profile
        if self.profile:
            profiler = CUDAEventProfiler(
                self.plan, 100, 100, tensor_A, tensor_B, tensor_C, tensor_D,
                visitor_args = epilogue_args
            )
            print(f"Cutlass Python Duration: {profiler()}")


class EVTTestCaseBase(unittest.TestCase):
    """
    Base class for EVT Unittest
    """
    def __init__(self, methodName: str = "runTest", lmnk=(6, 512, 256, 128)) -> None:
        super().__init__(methodName)
        
        self.element = cutlass_cppgen.DataType.f16
        self.l, self.m, self.n, self.k = lmnk
        
        self.problem_size = (self.m, self.n, self.k)
        
        torch.random.manual_seed(42)
    
    def fake_tensor(self, element, shape, stride=None):
        if stride is None:
            return Tensor(element=element, shape=shape, layout_tag=cutlass_cppgen.LayoutType.RowMajor)
        else:
            return Tensor(element=element, shape=shape, stride=stride)
    
    def get_problem_sizes(self, alignment, k=None, batch_count=[3,]):
        k = k if k else self.k
        problem_size_m = [alignment, 512 - 3 * alignment]
        problem_size_n = [alignment, 512 - alignment]
        if alignment % 8 == 0:
            problem_size_m.append(768)
            problem_size_n.append(768)
        problem_size_l = batch_count
        problem_sizes = []
        for m in problem_size_m:
            for n in problem_size_n:
                for l in problem_size_l:
                    problem_sizes.append((m, n, k, l))
        
        return problem_sizes
