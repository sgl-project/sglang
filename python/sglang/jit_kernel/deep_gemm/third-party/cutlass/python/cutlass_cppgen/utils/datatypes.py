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
Utility functions for converting between frontend datatypes and CUTLASS datatypes
"""

import cutlass_cppgen
from cutlass_library import (
    DataTypeSize,
    MathOperation,
    MathInstruction
)
from cutlass_cppgen.backend.library import (
    TileDescription,
)

bfloat16_available = None
cupy_available = None
numpy_available = None
torch_available = None
_library_to_cupy_dict = None
_library_to_numpy_dict = None
_library_to_torch_dict = None
_torch_to_library_dict = None


def is_numpy_available():
    global numpy_available, _library_to_numpy_dict
    if numpy_available is None:
        try:
            import numpy as np

            numpy_available = True
            _library_to_numpy_dict = {
                cutlass_cppgen.DataType.f16: np.float16,
                cutlass_cppgen.DataType.f32: np.float32,
                cutlass_cppgen.DataType.f64: np.float64,
                cutlass_cppgen.DataType.s8: np.int8,
                cutlass_cppgen.DataType.s32: np.int32,
            }
        except ImportError:
            numpy_available = False
            _library_to_numpy_dict = {}
    return numpy_available


def is_numpy_tensor(inp) -> bool:
    if is_numpy_available():
        import numpy as np
        return isinstance(inp, np.ndarray)
    return False


def numpy_library_type(inp) -> cutlass_cppgen.DataType:
    if is_numpy_available():
        import numpy as np
        if inp == np.float16:
            return cutlass_cppgen.DataType.f16
        elif inp == np.float32:
            return cutlass_cppgen.DataType.f32
        elif inp == np.float64:
            return cutlass_cppgen.DataType.f64
        elif inp == np.int8:
            return cutlass_cppgen.DataType.s8
        elif inp == np.int32:
            return cutlass_cppgen.DataType.s32
    return None


def numpy_type(inp):
    return _library_to_numpy_dict.get(inp, None)


def is_cupy_available():
    global cupy_available
    if cupy_available is None:
        try:
            import cupy as cp

            cupy_available = True
            _library_to_cupy_dict = {
                cutlass_cppgen.DataType.f16: cp.float16,
                cutlass_cppgen.DataType.f32: cp.float32,
                cutlass_cppgen.DataType.f64: cp.float64,
                cutlass_cppgen.DataType.s8: cp.int8,
                cutlass_cppgen.DataType.s32: cp.int32,
            }
        except ImportError:
            cupy_available = False
            _library_to_cupy_dict = {}
    return cupy_available


def is_cupy_tensor(inp) -> bool:
    if is_cupy_available():
        import cupy as cp
        return isinstance(inp, cp.ndarray)
    return False


def cupy_library_type(inp) -> cutlass_cppgen.DataType:
    if is_cupy_available():
        import cupy as cp
        if inp == cp.float16:
            return cutlass_cppgen.DataType.f16
        elif inp == cp.float32:
            return cutlass_cppgen.DataType.f32
        elif inp == cp.float64:
            return cutlass_cppgen.DataType.f64
    return None


def cupy_type(inp):
    return _library_to_cupy_dict.get(inp, None)


def is_torch_available():
    global torch_available, _library_to_torch_dict, _torch_to_library_dict
    if torch_available is None:
        try:
            import torch

            torch_available = True
            _torch_to_library_dict = {
                torch.half: cutlass_cppgen.DataType.f16,
                torch.float16: cutlass_cppgen.DataType.f16,
                torch.bfloat16: cutlass_cppgen.DataType.bf16,
                torch.float: cutlass_cppgen.DataType.f32,
                torch.float32: cutlass_cppgen.DataType.f32,
                torch.double: cutlass_cppgen.DataType.f64,
                torch.float64: cutlass_cppgen.DataType.f64,
                torch.int8: cutlass_cppgen.DataType.s8,
                torch.int32: cutlass_cppgen.DataType.s32,
                torch.uint8: cutlass_cppgen.DataType.u8,
            }

            _library_to_torch_dict = {
                cutlass_cppgen.DataType.f16: torch.half,
                cutlass_cppgen.DataType.f16: torch.float16,
                cutlass_cppgen.DataType.bf16: torch.bfloat16,
                cutlass_cppgen.DataType.f32: torch.float,
                cutlass_cppgen.DataType.f32: torch.float32,
                cutlass_cppgen.DataType.f64: torch.double,
                cutlass_cppgen.DataType.f64: torch.float64,
                cutlass_cppgen.DataType.s8: torch.int8,
                cutlass_cppgen.DataType.s32: torch.int32,
                cutlass_cppgen.DataType.u8: torch.uint8,
            }

            def possibly_add_type(torch_type_name, cutlass_type):
                # Only try adding the type if the version of torch being used supports it
                if hasattr(torch, torch_type_name):
                    torch_type = getattr(torch, torch_type_name)
                    _torch_to_library_dict[torch_type] = cutlass_type
                    _library_to_torch_dict[cutlass_type] = torch_type

            possibly_add_type("float8_e4m3fn", cutlass_cppgen.DataType.e4m3)
            possibly_add_type("float8_e5m2", cutlass_cppgen.DataType.e5m2)

        except ImportError:
            torch_available = False
            _torch_to_library_dict = {}
            _library_to_torch_dict = {}
    return torch_available


def is_torch_tensor(inp) -> bool:
    if is_torch_available():
        import torch
        return isinstance(inp, torch.Tensor)
    return False


def torch_library_type(inp) -> cutlass_cppgen.DataType:
    return _torch_to_library_dict.get(inp, None)


def torch_type(inp):
    return _library_to_torch_dict.get(inp, None)


def is_bfloat16_available():
    global bfloat16_available

    if bfloat16_available is None:
        try:
            import bfloat16

            bfloat16_available = True
        except ImportError:
            bfloat16_available = False
    return bfloat16_available


def bfloat16_library_type(inp) -> cutlass_cppgen.DataType:
    if is_bfloat16_available():
        import bfloat16
        if inp == bfloat16.bfloat16:
            return cutlass_cppgen.DataType.bf16


def bfloat16_type(inp):
    if is_bfloat16_available():
        import bfloat16
        if inp == cutlass_cppgen.DataType.bf16:
            return bfloat16.bfloat16


def library_type(inp):
    if inp in DataTypeSize:
        return inp

    for cvt_fn in [
        bfloat16_library_type,
        cupy_library_type,
        numpy_library_type,
        torch_library_type,
    ]:
        out = cvt_fn(inp)
        if out is not None:
            return out

    raise Exception(f"No available conversion from type {inp} to a library type.")


def _tensor_from_numpy(np_tensor):
    dtype = library_type(np_tensor.dtype)
    if np_tensor.flags.c_contiguous:
        layout = cutlass_cppgen.LayoutType.RowMajor
    elif np_tensor.flags.f_contiguous:
        layout = cutlass_cppgen.LayoutType.ColumnMajor
    return (dtype, layout)


def _tensor_from_torch(pt_tensor):
    dtype = library_type(pt_tensor.dtype)
    return (dtype, cutlass_cppgen.LayoutType.RowMajor)


def get_datatype_and_layout(tensor):
    if (is_numpy_tensor(tensor) or is_cupy_tensor(tensor)):
        return _tensor_from_numpy(tensor)
    elif is_torch_tensor(tensor):
        return _tensor_from_torch(tensor)
    elif isinstance(tensor, float) or isinstance(tensor, int):
        return (cutlass_cppgen.DataType.f32, cutlass_cppgen.LayoutType.RowMajor)
    else:
        raise Exception(f"Unable to convert tensor of type {type(tensor)} to Python-bound CUTLASS datatype and layout.")


def get_tensor_shape(tensor, op="GEMM"):
    if (is_numpy_tensor(tensor) or is_cupy_tensor(tensor)):
        return tensor.shape
    elif is_torch_tensor(tensor):
        size = tensor.size()
        if op == "CONV":
            # PyTorch Tensors have shape NCHW
            return (size[0], size[2], size[3], size[1])
        else:
            return tuple(tensor.size())
    elif isinstance(tensor, float) or isinstance(tensor, int):
        return (1,)
    else:
        raise Exception(f"Unable to convert tensor of type {type(tensor)} to Python-bound CUTLASS datatype and layout.")


_math_operation_value_map = {x.value: x for x in MathOperation}


def backend_math_operation(math_op: MathOperation):
    if math_op.value not in _math_operation_value_map.keys():
        raise Exception(f"Unable to convert math operation of type {math_op} to backend math operation.")
    return _math_operation_value_map[math_op.value]


def construct_backend_td(td: cutlass_cppgen.TileDescription,
                         kernel_schedule: cutlass_cppgen.KernelScheduleType,
                         epilogue_schedule: cutlass_cppgen.EpilogueScheduleType,
                         tile_scheduler: cutlass_cppgen.TileSchedulerType) -> TileDescription:
    mi = td.math_instruction
    backend_mi = MathInstruction(
        mi.instruction_shape,
        mi.element_a,
        mi.element_b,
        mi.element_accumulator,
        mi.opcode_class,
        backend_math_operation(mi.math_operation)
    )
    cluster_shape = td.cluster_shape if hasattr(td, "cluster_shape") else [1, 1, 1]
    return TileDescription(td.threadblock_shape, td.stages, td.warp_count,
                           backend_mi, cluster_shape, kernel_schedule, epilogue_schedule, tile_scheduler)


def td_from_profiler_op(op) -> TileDescription:
    """
    Converts the profiler's TileDescription in ``op`` into the backend TileDescription

    :param op: profiler Operation

    :returns: backend TileDescription
    :rtype: cutlass_cppgen.backend.TileDescription
    """
    kschedule = op.kernel_schedule if hasattr(op, 'kernel_schedule') else None
    eschedule = op.epilogue_schedule if hasattr(op, 'epilogue_schedule') else None
    tschedule = op.tile_scheduler if hasattr(op, 'tile_scheduler') else None
    return construct_backend_td(op.tile_description, kschedule, eschedule, tschedule)


def td_from_profiler_td(td: TileDescription) -> TileDescription:
    """
    Converts the profiler's TileDescription into the backend TileDescription

    :param td: profiler TileDescription
    :type td: cutlass_cppgen.TileDescription

    :returns: backend TileDescription
    :rtype: cutlass_cppgen.backend.TileDescription
    """
    return construct_backend_td(td, kernel_schedule=None, epilogue_schedule=None, tile_scheduler=None)


def to_camel_case(snake_str):
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def getattr_enum(obj, attr_name):
    # The attr_name is under the snake_case
    camel_attr = to_camel_case(attr_name)
    if hasattr(obj, camel_attr):
        return getattr(obj, camel_attr)
    else:
        raise Exception(f"Invalid option: {attr_name}")
