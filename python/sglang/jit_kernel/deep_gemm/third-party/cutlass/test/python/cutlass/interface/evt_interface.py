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
Test the EVT interface
"""

import numpy as np
import unittest

import cutlass_cppgen
from cutlass_cppgen import LayoutType, Tensor
from cutlass_cppgen.backend.utils.device import device_cc
from cutlass_cppgen.epilogue import reshape, permute

from utils import ExpectException


@unittest.skipIf(device_cc() not in [80, 90], "This unittest is for Sm80 and Sm90 only")
class EVTErrorTests(unittest.TestCase):
    """
    Tests various error scenarios that arise with the EVT interface
    """
    @unittest.skipIf(device_cc() != 90, "Only Sm90 EVT requires root node be 'D'")
    def test_root_not_d(self):
        """
        Test when "D" does not exist in Sm90 EVT
        """
        def evt_root_not_d(accum, alpha):
            F = accum * alpha
            return F
        
        example_tensors = {
            "accum": self.fake_tensor(np.float16, (6, 512, 512)),
            "alpha": 1.2,
            "F": self.fake_tensor(np.float16, (6, 512, 512))
        }
        
        with ExpectException(device_cc() == 90, 
            "SyntaxError: Sm90 EVT requires the epilogue to have a returned tensor D, "
            "but the variable 'D' is not found in the return values.", True):
            
            cutlass_cppgen.epilogue.trace(evt_root_not_d, example_tensors)

    def test_no_accum(self):
        """
        Test when "accum" is not in input arguments
        """
        def evt_no_accum(alpha, C):
            D = alpha * C
            return D
        
        example_tensors = {
            "C": self.fake_tensor(np.float16, (6, 512, 512)),
            "alpha": 1.2,
            "D": self.fake_tensor(np.float16, (6, 512, 512))
        }
        
        with ExpectException(True, "SyntaxError: Cannot find 'accum' in the argument list.", True):
            cutlass_cppgen.epilogue.trace(evt_no_accum, example_tensors)
    
    @unittest.skipIf(device_cc() != 90, "Only Sm90 EVT has concern on smem size")
    def test_too_much_shared_memory(self):
        """
        Test when the epilogue consumes too much shared memory
        """
        def evt_too_much_shared_memory(accum, C1, C2, C3, C4, C5, C6, C7, C8):
            D1 = accum + C1
            D2 = D1 + C2
            D3 = D2 + C3
            D4 = D3 + C4
            D5 = D4 + C5
            D6 = D5 + C6
            D7 = D6 + C7
            D = D7 + C8
            return D, D1, D2, D3, D4, D5, D6, D7
        
        example_tensors = {
            "accum": self.fake_tensor(np.float16, (6, 512, 512)),
            "C1": self.fake_tensor(np.float16, (6, 512, 512)),
            "C2": self.fake_tensor(np.float16, (6, 512, 512)),
            "C3": self.fake_tensor(np.float16, (6, 512, 512)),
            "C4": self.fake_tensor(np.float16, (6, 512, 512)),
            "C5": self.fake_tensor(np.float16, (6, 512, 512)),
            "C6": self.fake_tensor(np.float16, (6, 512, 512)),
            "C7": self.fake_tensor(np.float16, (6, 512, 512)),
            "C8": self.fake_tensor(np.float16, (6, 512, 512)),
            "D1": self.fake_tensor(np.float16, (6, 512, 512)),
            "D2": self.fake_tensor(np.float16, (6, 512, 512)),
            "D3": self.fake_tensor(np.float16, (6, 512, 512)),
            "D4": self.fake_tensor(np.float16, (6, 512, 512)),
            "D5": self.fake_tensor(np.float16, (6, 512, 512)),
            "D6": self.fake_tensor(np.float16, (6, 512, 512)),
            "D7": self.fake_tensor(np.float16, (6, 512, 512)),
            "D": self.fake_tensor(np.float16, (6, 512, 512))
        }
        
        epilogue_visitor = cutlass_cppgen.epilogue.trace(evt_too_much_shared_memory, example_tensors)
        
        plan = cutlass_cppgen.op.Gemm(
            element=np.float16, layout=cutlass_cppgen.LayoutType.RowMajor,
            element_accumulator=np.float32
        )
        
        with ExpectException(True, 
            "RuntimeError: The epilogue consumes too much shared memory. " 
            "No valid tile description is found in the generator.", True):
            plan.epilogue_visitor = epilogue_visitor
    
    def test_not_ssa(self):
        """
        Test when the epilogue is not in SSA
        """
        def evt_redefine(accum, C, alpha):
            F = accum + C
            F = F * alpha
            D = F
            return D, F

        example_tensors = {
            "accum": self.fake_tensor(np.float16, (6, 512, 512)),
            "C": self.fake_tensor(np.float16, (6, 512, 512)),
            "alpha": 1.5,
            "D": self.fake_tensor(np.float16, (6, 512, 512)),
            "F": self.fake_tensor(np.float16, (6, 512, 512))
        }
        
        with ExpectException(True, "SyntaxError: Variable 'F' cannot be defined twice.", True):
            cutlass_cppgen.epilogue.trace(evt_redefine, example_tensors)

        def evt_undefine(accum, alpha):
            F = accum + C
            D = F * alpha
            return D, F
        
        example_tensors = {
            "accum": self.fake_tensor(np.float16, (6, 512, 512)),
            "alpha": 1.5,
            "D": self.fake_tensor(np.float16, (6, 512, 512)),
            "F": self.fake_tensor(np.float16, (6, 512, 512))
        }
        
        with ExpectException(True, "SyntaxError: Variable 'C' is undefined.", True):
            cutlass_cppgen.epilogue.trace(evt_undefine, example_tensors)
    
    def test_missing_example_tensor(self):
        """
        Test when the example tensor of an input/output variable is not provided
        """
        def evt_missing_example_tensor(accum, C):
            D = accum + C
            return D
        
        example_tensors = {
            "accum": self.fake_tensor(np.float16, (6, 512, 512)),
            "C": self.fake_tensor(np.float16, (6, 512, 512)),
        }
        
        with ExpectException(True, "RuntimeError: Example input for D is not provided.", True):
            cutlass_cppgen.epilogue.trace(evt_missing_example_tensor, example_tensors)
        
        example_tensors = {
            "accum": self.fake_tensor(np.float16, (6, 512, 512)),
            "D": self.fake_tensor(np.float16, (6, 512, 512)),
        }
        
        with ExpectException(True, "RuntimeError: Example input for C is not provided.", True):
            cutlass_cppgen.epilogue.trace(evt_missing_example_tensor, example_tensors)
        
    def test_return_expression(self):
        """
        Test when the return value is an expression
        """
        def evt_return_expr(accum, C):
            return accum + C
        
        example_tensors = {
            "accum": self.fake_tensor(np.float16, (6, 512, 512)),
            "C": self.fake_tensor(np.float16, (6, 512, 512)),
        }
        
        with ExpectException(True, "SyntaxError: Return value cannot be an expression", True):
            cutlass_cppgen.epilogue.trace(evt_return_expr, example_tensors)
    
    def test_incompatible_shape(self):
        """
        Test when the shape of example tensors are incompatible
        """
        def evt_incompatible_shape(accum, C):
            D = accum + C
            return D
        
        example_tensors = {
            "accum": self.fake_tensor(np.float16, (6, 256, 512)),
            "C": self.fake_tensor(np.float16, (6, 512, 512)),
            "D": self.fake_tensor(np.float16, (6, 512, 512))
        }
        
        with ExpectException(True, 
            "RuntimeError: Dimension mismatch between accum(6, 256, 512), C(6, 512, 512).", True):
            cutlass_cppgen.epilogue.trace(evt_incompatible_shape, example_tensors)
    
    def test_no_matching_impl(self):
        def evt_no_matching_impl(accum, bias):
            D = accum + reshape(permute(bias, indices=(1, 0)), new_shape=(512, 1))
            return D

        example_tensors = {
            "accum": self.fake_tensor(np.float16, (6, 512, 256)),
            "bias": self.fake_tensor(np.float16, (16, 32)),
            "D": self.fake_tensor(np.float16, (6, 512, 256))
        }
        
        with ExpectException(True, "NotImplementedError: No matching op for node bias with stride (0, (1, 32), 0).", True):
            cutlass_cppgen.epilogue.trace(evt_no_matching_impl, example_tensors)
    #
    # Helper functions
    #
    
    def fake_tensor(self, element, shape):
        return Tensor(element=element, shape=shape, layout_tag=LayoutType.RowMajor)


if __name__ == '__main__':
    unittest.main()
