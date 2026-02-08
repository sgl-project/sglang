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

from cutlass_library import (
    DataType,
    KernelScheduleType,
    TileSchedulerType
)
from cutlass_cppgen.backend.library import DataTypeSizeBytes


class GemmCoord_(ctypes.Structure):
    _fields_ = [
        ("m", ctypes.c_int),
        ("n", ctypes.c_int),
        ("k", ctypes.c_int)
    ]

    def __init__(self, m, n, k) -> None:
        self.m = m
        self.n = n
        self.k = k


class GemmCoordBatched_(ctypes.Structure):
    """
    Wrapper around a GemmCoord that also contains batch count. This is used for encoding
    batched GEMM inputs to CUTLASS 3 GEMMs.
    """

    _fields_ = [
        ("m", ctypes.c_int),
        ("n", ctypes.c_int),
        ("k", ctypes.c_int),
        ("batch_count", ctypes.c_int)
    ]

    def __init__(self, gemm_coord, batch_count) -> None:
        self.m = gemm_coord.m
        self.n = gemm_coord.n
        self.k = gemm_coord.k
        self.batch_count = batch_count


class MatrixCoord_(ctypes.Structure):
    _fields_ = [
        ("row", ctypes.c_int),
        ("column", ctypes.c_int)
    ]


class dim3_(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("z", ctypes.c_int)
    ]


class StrideBatched_(ctypes.Structure):
    """
    CUTLASS 3.0 strides for operands contain one static dimension and two variable dimensions. The
    variable dimensions represent the stride along non-unit-stride dimension of the row/column major
    layout, and the batch stride. This structure encodes the two variable dimensions.
    """
    _fields_ = [
        ("major_stride", ctypes.c_int64),
        ("batch_stride", ctypes.c_int64)
    ]



class GenericMainloopArguments3x_(ctypes.Structure):
    """
    Structure representing the superset of possible mainloop arguments.
    This structure should not be passed to kernels directly, but, rather,
    be used as an input to one of the more specific schedule arguments, which
    will each select those arguments relevant to the particular schedule.
    """
    _fields_ = [
        ("ptr_A", ctypes.c_void_p),
        ("stride_A", StrideBatched_),
        ("ptr_B", ctypes.c_void_p),
        ("stride_B", StrideBatched_),
        ("mma_promotion_interval", ctypes.c_int)
    ]


class _PersistentTileSchedulerArguments(ctypes.Structure):
    _fields_ = [
        ("max_swizzle_size", ctypes.c_int),
        ("raster_order_option", ctypes.c_int),
    ]


class _PersistentTileSchedulerStreamKArguments(ctypes.Structure):
    _fields_ = [
        ("splits", ctypes.c_int),
        ("max_swizzle_size", ctypes.c_int),
        ("raster_order_option", ctypes.c_int),
        ("reduction_mode", ctypes.c_int),
        ("decomposition_mode", ctypes.c_int),
    ]


def get_tile_scheduler_arguments_3x(
    tile_scheduler: TileSchedulerType,
    splits: int = 1):
    max_swizzle_size = 1
    raster_order_option = 0 # Heuristic
    if tile_scheduler in [TileSchedulerType.Default, TileSchedulerType.Persistent]:
        return _PersistentTileSchedulerArguments(
            max_swizzle_size,
            raster_order_option,
        )
    elif tile_scheduler == TileSchedulerType.StreamK:
        reduction_mode = 0 # Deterministic
        decomposition_mode = 0 # Heuristic
        return _PersistentTileSchedulerStreamKArguments(
            splits,
            max_swizzle_size,
            raster_order_option,
            reduction_mode,
            decomposition_mode,
        )


def get_mainloop_arguments_3x(
    kernel_schedule: KernelScheduleType,
    element_A,
    element_B,
    alignment_A: int,
    alignment_B: int) -> ctypes.Structure:
    """
    Returns the ctypes structure to be used for the 3.x kernel's mainloop parameters.

    :param kernel_schedule: type of kernel schedule to be used in the mainloop
    :type kernel_schedule: cutlass_library.KernelScheduleType
    :param element_A: data type of operand A
    :param element_B: data type of operand B
    :param alignment_A: alignment of operand A
    :type alignment_A: int
    :param alignment_B: alignment of operand B
    :type alignment_B: int

    :returns: ctypes structure to be used for the 3.x kernel's mainloop parameters
    :rtype: ctypes.Structure
    """
    class _MainloopArgumentsTma(ctypes.Structure):
        _fields_ = [
            ("ptr_A", ctypes.c_void_p),
            ("stride_A", StrideBatched_),
            ("ptr_B", ctypes.c_void_p),
            ("stride_B", StrideBatched_),
            ("mma_promotion_interval", ctypes.c_int)
        ]

        @staticmethod
        def from_generic_mainloop_args(args: GenericMainloopArguments3x_):
            return _MainloopArgumentsTma(
                args.ptr_A, args.stride_A, args.ptr_B, args.stride_B,
                args.mma_promotion_interval
            )

    class _MainloopArgumentsMultistage(ctypes.Structure):
        _fields_ = [
            ("ptr_A", ctypes.c_void_p),
            ("stride_A", StrideBatched_),
            ("ptr_B", ctypes.c_void_p),
            ("stride_B", StrideBatched_),
        ]

        @staticmethod
        def from_generic_mainloop_args(args: GenericMainloopArguments3x_):
            return _MainloopArgumentsMultistage(
                args.ptr_A, args.stride_A, args.ptr_B, args.stride_B,
            )

    # Currently all 3.x kernels (CpAsync and Tma) have the same argument structure.
    # Should that become not the case, this is the place to return custom ctypes
    # structures based on selected kernel schedule.
    return _MainloopArgumentsTma


def get_gemm_arguments_3x(mainloop_arguments, epilogue_functor, scheduler_args, default_epilogue):
    if not default_epilogue and hasattr(epilogue_functor, "epilogue_type_evt"):
        _EpilogueOutputOpParams = epilogue_functor.epilogue_type_evt
    else:
        _EpilogueOutputOpParams = epilogue_functor.epilogue_type

    if hasattr(epilogue_functor, "visitor"):
        class _EpilogueArguments(ctypes.Structure):
            _fields_ = [
                ("epilogue", _EpilogueOutputOpParams),
                ("arg_C", epilogue_functor.arg_c_type),
                ("arg_D", epilogue_functor.arg_d_type)
            ]

            def __init__(self, output_op, ptr_c, stride_c, ptr_d, stride_d) -> None:
                self.epilogue = output_op
                self.arg_C = epilogue_functor.arg_c_type(ptr_c)
                self.arg_D = epilogue_functor.arg_d_type(ptr_d)
    else:
        class _EpilogueArguments(ctypes.Structure):
            _fields_ = [
                ("epilogue", _EpilogueOutputOpParams),
                ("ptr_C", ctypes.c_void_p),
                ("stride_C", StrideBatched_),
                ("ptr_D", ctypes.c_void_p),
                ("stride_D", StrideBatched_),
            ]

    class _HardwareInfo(ctypes.Structure):
        _fields_ = [
            ("device_id", ctypes.c_int),
            ("sm_count", ctypes.c_int),
            ("max_active_clusters", ctypes.c_int),
            ("cluster_shape", dim3_),
            ("cluster_shape_fallback", dim3_),
        ]

    class _GemmArguments(ctypes.Structure):
        _fields_ = [
            ("mode", ctypes.c_int),
            ("problem_size", GemmCoordBatched_),
            ("mainloop", mainloop_arguments),
            ("epilogue", _EpilogueArguments),
            ("hw_info", _HardwareInfo),
            ("scheduler", type(scheduler_args)),
        ]

    return _GemmArguments, _EpilogueArguments, _EpilogueOutputOpParams, _HardwareInfo


def get_gemm_arguments(epilogue_functor):
    _EpilogueOutputOpParams = epilogue_functor.epilogue_type

    class _GemmArguments(ctypes.Structure):
        _fields_ = [
            # Arguments from UniversalArgumentsBase
            ("mode", ctypes.c_int),
            ("problem_size", GemmCoord_),
            ("batch_count", ctypes.c_int),
            ("batch_stride_D", ctypes.c_longlong),
            # Remaining arguments
            ("epilogue", _EpilogueOutputOpParams),
            ("ptr_A", ctypes.c_void_p),
            ("ptr_B", ctypes.c_void_p),
            ("ptr_C", ctypes.c_void_p),
            ("ptr_D", ctypes.c_void_p),
            ("batch_stride_A", ctypes.c_longlong),
            ("batch_stride_B", ctypes.c_longlong),
            ("batch_stride_C", ctypes.c_longlong),
            ("stride_a", ctypes.c_longlong),
            ("stride_b", ctypes.c_longlong),
            ("stride_c", ctypes.c_longlong),
            ("stride_d", ctypes.c_longlong),
            ("lda", ctypes.c_longlong),
            ("ldb", ctypes.c_longlong),
            ("ldc", ctypes.c_longlong),
            ("ldd", ctypes.c_longlong),
            ("ptr_gather_A_indices", ctypes.c_void_p),
            ("ptr_gather_B_indices", ctypes.c_void_p),
            ("ptr_scatter_D_indices", ctypes.c_void_p)
        ]

    return _GemmArguments, _EpilogueOutputOpParams


def get_gemm_arguments_streamk(epilogue_functor):
    _EpilogueOutputOpParams = epilogue_functor.epilogue_type

    class _GemmArguments(ctypes.Structure):
        _fields_ = [
            ("mode", ctypes.c_int),
            ("problem_size", GemmCoord_),
            ("batch_count", ctypes.c_int),
            ("epilogue", _EpilogueOutputOpParams),
            ("ptr_A", ctypes.c_void_p),
            ("ptr_B", ctypes.c_void_p),
            ("ptr_C", ctypes.c_void_p),
            ("ptr_D", ctypes.c_void_p),
            ("batch_stride_A", ctypes.c_longlong),
            ("batch_stride_B", ctypes.c_longlong),
            ("batch_stride_C", ctypes.c_longlong),
            ("batch_stride_D", ctypes.c_longlong),
            ("stride_a", ctypes.c_longlong),
            ("stride_b", ctypes.c_longlong),
            ("stride_c", ctypes.c_longlong),
            ("stride_d", ctypes.c_longlong),
            ("lda", ctypes.c_longlong),
            ("ldb", ctypes.c_longlong),
            ("ldc", ctypes.c_longlong),
            ("ldd", ctypes.c_longlong),
            ("avail_sms", ctypes.c_int)
        ]

    return _GemmArguments, _EpilogueOutputOpParams


###########################################################################################
# GEMM Grouped
###########################################################################################


def get_gemm_grouped_arguments(epilogue_functor):
    _EpilogueOutputOpParams = epilogue_functor.epilogue_type

    class _GEMMGroupedArguments(ctypes.Structure):
        _fields_ = [
            ("problem_sizes", ctypes.c_void_p),
            ("problem_count", ctypes.c_int),
            ("threadblock_count", ctypes.c_int),
            ("output_op", _EpilogueOutputOpParams),
            ("ptr_A", ctypes.c_void_p),
            ("ptr_B", ctypes.c_void_p),
            ("ptr_C", ctypes.c_void_p),
            ("ptr_D", ctypes.c_void_p),
            ("lda", ctypes.c_void_p),
            ("ldb", ctypes.c_void_p),
            ("ldc", ctypes.c_void_p),
            ("ldd", ctypes.c_void_p),
            ("host_problem_sizes", ctypes.c_void_p)
        ]

    return _GEMMGroupedArguments, _EpilogueOutputOpParams


############################################################################################
# Convolution2D
############################################################################################


class Conv2DProblemSize_(ctypes.Structure):
    _fields_ = [
        ("N", ctypes.c_int),
        ("H", ctypes.c_int),
        ("W", ctypes.c_int),
        ("C", ctypes.c_int),
        ("P", ctypes.c_int),
        ("Q", ctypes.c_int),
        ("K", ctypes.c_int),
        ("R", ctypes.c_int),
        ("S", ctypes.c_int),
        ("pad_h", ctypes.c_int),
        ("pad_w", ctypes.c_int),
        ("stride_h", ctypes.c_int),
        ("stride_w", ctypes.c_int),
        ("dilation_h", ctypes.c_int),
        ("dilation_w", ctypes.c_int),
        ("mode", ctypes.c_int),  # kCrossCorrelation: 0, kConvolution: 1
        ("split_k_slices", ctypes.c_int),
        ("groups", ctypes.c_int)
    ]

    def __init__(self, problem_size) -> None:
        for field_name, _ in self._fields_:
            setattr(self, field_name, getattr(problem_size, field_name))


class Layout4D(ctypes.Structure):
    _fields_ = [("stride", ctypes.c_int * 3)]

    def __init__(self, tensor_ref):
        stride = tensor_ref.stride()
        setattr(self, "stride", (stride.at(0), stride.at(1), stride.at(2)))


class TensorRef_(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("layout", Layout4D)
    ]

    def __init__(self, tensor_ref):
        setattr(self, "ptr", tensor_ref.data())
        setattr(self, "layout", Layout4D(tensor_ref.layout()))


class TensorRef2D_(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("stride", ctypes.c_int)
    ]


def get_conv2d_arguments(epilogue_functor):
    _EpilogueOutputOpParams = epilogue_functor.epilogue_type

    class _Conv2dArguments(ctypes.Structure):
        _fields_ = [
            ("conv_kind", ctypes.c_int),
            ("problem_size", Conv2DProblemSize_),
            ("ptr_A", ctypes.c_void_p),
            ("ptr_B", ctypes.c_void_p),
            ("ptr_C", ctypes.c_void_p),
            ("ptr_D", ctypes.c_void_p),
            ("tensor_C_numel", ctypes.c_int),
            ("output_op", _EpilogueOutputOpParams),
            ("split_k_mode", ctypes.c_int)
        ]

    return _Conv2dArguments, _EpilogueOutputOpParams


############################################################################################
# Reduction
############################################################################################


def get_reduction_params(epilogue_functor):
    _EpilogueOutputParams = epilogue_functor.epilogue_type

    class _ReductionParams(ctypes.Structure):
        _fields_ = [
            ("problem_size", MatrixCoord_),
            ("partitions", ctypes.c_int),
            ("partition_stride", ctypes.c_longlong),
            ("workspace", TensorRef2D_),
            ("destination", TensorRef2D_),
            ("source", TensorRef2D_),
            ("output_op", _EpilogueOutputParams),
        ]

    return _ReductionParams, _EpilogueOutputParams


###########################################################################################
# Epilogue Visitor Type Factory
###########################################################################################

class Empty(ctypes.Structure):
    _fields_ = []

    def __init__(self, *arg) -> None:
        pass

class EmptyByte(ctypes.Structure):
    _fields_ = [
        ("byte", ctypes.c_byte)
    ]

    def __init__(self, *arg) -> None:
        pass

class EBO:
    def __init__(self, index: int, type) -> None:
        self.index = index
        self.type = type

    def __eq__(self, other) -> bool:
        if isinstance(other, EBO):
            return self.index == other.index and self.type == other.type
        return False

    def __hash__(self) -> int:
        return hash((self.index, self.type))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self) -> str:
        return f"<{self.index}, {self.type}>"


def tuple_factory_(input_tuple, dtype, constants=[0,1]):
    """
    The factory function generating cute::Tuple with input tuple
    :param input_tuple: the input tuple
    :type input_tuple: tuple
    :param dtype: the data type for non-constant values
    :type dtype: str, "int32_t", "int", "int64_t"
    :param constant: the values that will be treated as constants
    :type constant: list[int]

    :return: ctype structure representing the cute::Tuple
    :return: the empty base classes of the tuple
    """

    # The empty base classes of the current tuple
    empty_bases = []
    # The first non empty base class
    first_non_empty_base = None
    # The ctype fields of the current tuple
    ctype_fields = []

    for idx, entry in enumerate(input_tuple):
        # For nested tuples
        if isinstance(entry, tuple):
            sub_tuple_ctype, sub_empty_bases = tuple_factory_(entry, dtype, constants)
            if ctypes.sizeof(sub_tuple_ctype) == 0:
                # The empty tuple base class is also an empty EBO
                empty_bases.append(EBO(idx, entry))
            else:
                if first_non_empty_base is None:
                    first_non_empty_base = sub_empty_bases
            ctype_fields.append((f"entry_{idx}", sub_tuple_ctype))
        else:
            if entry in constants:
                empty_bases.append(EBO(idx, entry))
                ctype_fields.append((f"entry_{idx}", Empty))
            else:
                ctype_fields.append((f"entry_{idx}", dtype))
                if first_non_empty_base is None:
                    first_non_empty_base = []

    # Create the ctype tuple
    class TupleType(ctypes.Structure):
        _fields_ = ctype_fields

        def __init__(self, args) -> None:
            fields = self._fields_

            assert len(fields) == len(args)
            for field, arg in zip(fields, args):
                name = field[0]
                field_type = field[1]
                setattr(self, name, field_type(arg))

    return TupleType, empty_bases

def tuple_factory(input_tuple, dtype: str, constants=[0,1]):
    """
    The factory function generating cute::Tuple with input tuple
    :param input_tuple: the input tuple
    :type input_tuple: tuple
    :param dtype: the data type for non-constant values
    :type dtype: str, "int32_t", "int", "int64_t"
    :param constant: the values that will be treated as constants
    :type constant: list[int]

    :return: ctype structure representing the cute::Tuple
    :return: the empty base classes of the tuple
    """
    # Step 1: convert the dtype
    if dtype == "int64_t":
        dtype = ctypes.c_longlong
    elif dtype in ["int", "int32_t"]:
        dtype = ctypes.c_int32
    else:
        raise NotImplementedError(f"Type {dtype} is not supported")

    tuple_type, _ = tuple_factory_(input_tuple, dtype, constants)

    if ctypes.sizeof(tuple_type) == 0:
        return EmptyByte
    return tuple_type


def visitor_factory(node_types, node_names):
    """
    Creates the argument type of epilogue visitor type

    :param node_types: list of argument types under ctypes
    :param node_names: list of argument names under str

    :return: tuple type in ctypes.Structure
    """
    ctypes_field = []
    # Struct is used when number of nodes < 4
    # Because the Sm90VisitorImplBase has specification up to 4 nodes
    # in `include/cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp`
    if len(node_types) <= 4:
        for idx, node_type in enumerate(node_types):
            if ctypes.sizeof(node_type) == 0:
                # Special case for empty struct
                # 1 byte placeholder is used for correct alignment
                ctypes_field.append((node_names[idx], ctypes.c_byte))
            else:
                ctypes_field.append((node_names[idx], node_type))

        class VisitorType(ctypes.Structure):
            _fields_ = ctypes_field

            def __init__(self, kwargs) -> None:
                for field in self._fields_:
                    fname, ftype = field
                    if ftype != ctypes.c_byte:
                        setattr(self, fname, ftype(kwargs))

    # For cases with more than 4 nodes, tuple is used
    else:
        for idx, node_type in enumerate(node_types):
            ctypes_field.append((node_names[idx], node_type))

        class VisitorType(ctypes.Structure):
            _fields_ = ctypes_field

            def __init__(self, kwargs) -> None:
                for field in self._fields_:
                    fname, ftype = field
                    setattr(self, fname, ftype(kwargs))

    return VisitorType
