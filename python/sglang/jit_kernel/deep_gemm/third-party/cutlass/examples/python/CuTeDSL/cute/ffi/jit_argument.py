# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

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

"""Example of accessing POD (Plain Old Data) from C or other languages via LLVM operations.

This example demonstrates a basic approach to building customized interfaces as C-structures between user code
and JIT compiled functions. It provides a minimal-cost solution for calling JIT functions
and can be used to build AOT (Ahead-of-Time) launchers for JIT compiled functions.

The C-structure is defined as:

.. code-block:: c

    struct Tensor {
        void *ptr;          // Pointer to tensor data
        int32_t shape[3];   // Tensor dimensions
        int32_t strides[3]; // Memory strides for each dimension
    };

The example defines Tensor and TensorValue classes that wrap C structs for view of a tensor with its data pointer,
shape, and strides, enabling efficient data passing between different language boundaries.

.. note::
   Future development may include automated code generation flows.
"""

import cutlass
import cutlass.cute as cute

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
import cutlass._mlir.extras.types as T


class ExampleTensorValue(ir.Value):
    """A wrapper class for tensor values in MLIR.

    This class extends ir.Value to provide convenient access to tensor data pointer,
    shape, and strides through MLIR operations.

    :type: ir.Value
    """

    def __init__(self, v):
        """Initialize a new TensorValue.

        :param v: The underlying MLIR value to wrap
        :type v: ir.Value
        """
        super().__init__(v)

    @property
    def data_ptr(self, *, loc=None, ip=None):
        """Get the data pointer from the tensor value.

        Extracts the data pointer (first field) from the LLVM struct value.

        :param loc: Optional location information for MLIR operations
        :type loc: Optional[ir.Location]
        :param ip: Optional insertion point for MLIR operations
        :type ip: Optional[ir.InsertionPoint]
        :return: An integer value representing the data pointer
        :rtype: ir.Value
        """
        # Extract the data pointer from the LLVM struct value
        # The data pointer is the first field (index 0) in the struct

        # Use llvm.extractvalue to get the pointer field from the struct
        ptr_val = llvm.extractvalue(
            llvm.PointerType.get(),
            self,
            [0],  # Extract the first field (index 0)
            loc=loc,
            ip=ip,
        )

        return cute.make_ptr(cutlass.Float32, ptr_val)

    @property
    def shape(self):
        """Get the shape of the tensor.

        Extracts the shape (second field) from the LLVM struct value.

        :return: A tuple of integers representing the tensor dimensions
        :rtype: tuple[ir.Value, ...]
        """
        i32_type = ir.IntegerType.get_signless(32)
        # Extract the shape field from the LLVM struct value
        # The shape is the second field (index 1) in the struct
        shape_val = llvm.extractvalue(
            llvm.StructType.get_literal([i32_type] * 3),
            self,
            [1],  # Extract the second field (index 1)
        )

        # Extract each dimension from the shape struct
        return tuple(llvm.extractvalue(i32_type, shape_val, [i]) for i in range(3))

    @property
    def stride(self):
        """Get the strides of the tensor.

        Extracts the strides (third field) from the LLVM struct value.

        :return: A tuple of integers representing the tensor strides
        :rtype: tuple[ir.Value, ...]
        """
        i32_type = ir.IntegerType.get_signless(32)
        # Extract the strides field from the LLVM struct value
        # The strides are the third field (index 2) in the struct
        strides_val = llvm.extractvalue(
            llvm.StructType.get_literal([i32_type] * 3),
            self,
            [2],  # Extract the third field (index 2)
        )

        # Extract each dimension from the strides struct
        return tuple(llvm.extractvalue(i32_type, strides_val, [i]) for i in range(3))


class ExampleTensor:
    """A class representing a tensor with its data pointer, shape, and strides.

    This class provides a Python interface to create and manipulate tensor structures
    that can be passed to CUTE JIT compiled functions.

    :ivar _c_struct_p: The C struct pointer for the tensor
    :ivar _rank: The number of dimensions in the tensor
    """

    def __init__(self, c_struct_p, rank):
        """Initialize a new Tensor.

        :param c_struct_p: The C struct pointer for the tensor
        :type c_struct_p: int
        :param rank: The number of dimensions in the tensor
        :type rank: int
        """
        self._c_struct_p = c_struct_p
        self._rank = rank

    def __get_mlir_types__(self):
        """Get the MLIR types for this tensor.

        Creates an LLVM structure type representing a C-structure with:

        .. code-block:: c

            struct Tensor {
                void *ptr;
                int32_t shape[3];
                int32_t strides[3];
            };

        :return: A list containing the MLIR struct type
        :rtype: list[llvm.StructType]

        Create an LLVM structure type that represents a C-structure like:
        """

        # Get the number of dimensions from the shape
        ndim = self._rank

        # Create the pointer type (void*)
        ptr_type = llvm.PointerType.get()

        # Create array types for shape and strides (int32_t[ndim])
        int32_type = ir.IntegerType.get_signless(32)
        shape_type = llvm.StructType.get_literal([int32_type] * ndim)
        strides_type = llvm.StructType.get_literal([int32_type] * ndim)

        # Create the structure type
        struct_type = llvm.StructType.get_literal([ptr_type, shape_type, strides_type])

        return [struct_type]

    def __new_from_mlir_values__(self, values):
        """Create a new TensorValue from MLIR values.

        :param values: A list of MLIR values
        :type values: list[ir.Value]
        :return: A new TensorValue instance
        :rtype: TensorValue
        """
        return ExampleTensorValue(values[0])

    def __c_pointers__(self):
        """Get the C pointers for this tensor.

        :return: A list containing the C struct pointer
        :rtype: list[int]
        """
        return [self._c_struct_p]


@cute.jit
def foo(tensor):
    """Example JIT function that prints tensor information.

    :param tensor: A Tensor instance to print information about
    :type tensor: Tensor
    """
    cute.printf("data_ptr: {}", tensor.data_ptr)
    cute.printf("shape: {}", tensor.shape)
    cute.printf("stride: {}", tensor.stride)

    mA = cute.make_tensor(
        tensor.data_ptr, cute.make_layout(tensor.shape, stride=tensor.stride)
    )
    cute.print_tensor(mA)


import sys
import os
import subprocess
import shutil
import tempfile
import torch


def run_test(tmpdir=None):
    # Skip cleanup if user provides tmpdir
    cleanup = tmpdir is None
    # Initialize temporary build directory
    tmpdir = tmpdir or tempfile.mkdtemp()

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))

        subprocess.run(["cmake", "-B", tmpdir, current_dir], check=True)
        subprocess.run(["cmake", "--build", tmpdir], check=True)

        sys.path.append(tmpdir)

        from tensor import make_tensor, pycapsule_get_pointer

        # Mock test tensor and corresponding C structure for this example
        # In production, this may come from external library
        x = torch.arange(2 * 8 * 4).to(torch.float32).reshape(2, 8, 4)
        c_struct = make_tensor(x.data_ptr(), x.shape, x.stride())
        c_struct_p = pycapsule_get_pointer(c_struct)

        # Initialize tensor wrapper and compile test function
        tensor = ExampleTensor(c_struct_p, len(x.shape))
        compiled_func = cute.compile(foo, tensor)

        # Benchmark pointer access performance
        from time import time

        start = time()
        # Measure performance of critical path pointer access
        # get C pointers is on critical path to call JIT compiled function
        for _ in range(1000):
            tensor.__c_pointers__()
        end = time()
        print(f"__c_pointers__: {(end - start) * 1000} us")

        # Execute compiled function
        compiled_func(tensor)
    except Exception as e:
        print(e)
    finally:
        if cleanup:
            # Clean up the temporary directory
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Set temporary directory for building C modules"
    )
    parser.add_argument(
        "--tmp-dir", type=str, help="Temporary directory path for building C modules"
    )
    args = parser.parse_args()

    run_test(args.tmp_dir)
