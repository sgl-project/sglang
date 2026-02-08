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

from cutlass_library import SubstituteTemplate
import numpy as np

from cutlass_library import DataType, DataTypeTag
from cutlass_cppgen.backend.c_types import MatrixCoord_, tuple_factory
from cutlass_cppgen.backend.frontend import NumpyFrontend
from cutlass_cppgen.backend.library import ActivationOp, ActivationOpTag
from cutlass_cppgen.utils.datatypes import is_numpy_tensor, is_torch_available, is_torch_tensor

dtype2ctype = {
    DataType.f16: ctypes.c_uint16,
    DataType.bf16: ctypes.c_uint16,
    DataType.f32: ctypes.c_float,
    DataType.f64: ctypes.c_double,
    DataType.s8: ctypes.c_int8,
    DataType.s32: ctypes.c_int32
}

if is_torch_available():
    import torch
    import torch.nn.functional as F


def get_scalar(value):
    """
    Returns a scalar value from a container (e.g., np.ndarray)
    """
    if is_numpy_tensor(value):
        if value.size != 1:
            raise Exception("Scalars used in epilogue must be of size 1")
        return value.reshape(-1)[0]
    elif is_torch_tensor(value):
        if value.size != 1:
            raise Exception("Scalars used in epilogue must be of size 1")
        return value.reshape(-1)[0]
    else:
        return value


def to_ctype_value(value, dtype):
    """
    Converts ``value`` to the corresponding storage needed for the ctype that
    will store ``value``.
    """
    scalar = get_scalar(value)
    if dtype == DataType.f16:
        # Convert f16 value into an integer
        return int.from_bytes(np.float16(scalar).tobytes(), "little")
    else:
        return scalar


#################################################################################################
#
# Epilogue Functors
#
#################################################################################################


class EpilogueFunctorBase:
    """
    Base class for thread-level epilogue functors
    """

    def __init__(self) -> None:
        pass

    def emit(self, tag, template_argument):
        template = """${tag}<${arguments}>"""
        arguments = ""
        for idx, arg in enumerate(template_argument):
            arguments += arg
            if idx < len(template_argument) - 1:
                arguments += ", "
        values = {
            "tag": tag,
            "arguments": arguments,
        }

        return SubstituteTemplate(template, values)


class LinearCombination(EpilogueFunctorBase):
    """
    Apply a linear combination operator to an array of elements
    D = alpha * accumulator + beta * source

    :param element_output: data type used to load and store tensors

    :param epilogue_vector_length: number of elements computed per operation.
    Usually it is 128/sizeof_bits_v<ElementOutput_>, but we use 64 and 32 sometimes
    when there are not enough data to store

    :param element_accumulator: Accumulator data type

    :param element_epilogue: data type used to compute linear combination
    """

    tag = "cutlass::epilogue::thread::LinearCombination"

    def __init__(
        self, element_output, epilogue_vector_length,
        element_accumulator=None, element_epilogue=None) -> None:
        super().__init__()

        if element_accumulator is None:
            element_accumulator = element_output
        if element_epilogue is None:
            element_epilogue = element_output

        self.element_output = element_output
        self.element_accumulator = element_accumulator
        self.element_epilogue = element_epilogue
        self.epilogue_vector_length = epilogue_vector_length

        self.template_arguments = [
            DataTypeTag[element_output],
            str(epilogue_vector_length),
            DataTypeTag[element_accumulator],
            DataTypeTag[element_epilogue],
        ]

        c_element_epilogue = dtype2ctype[self.element_epilogue]
        element_epilogue = self.element_epilogue

        class _EpilogueOutputOpParamsEVT(ctypes.Structure):
            """
            Epilogue params when using the default linear combination of EVT, which
            does not currently use {alpha,beta}_ptr_array
            """

            stride_type = tuple_factory((0,0,1), "int64_t", [0])
            _fields_ = [
                ("alpha", c_element_epilogue),
                ("beta", c_element_epilogue),
                ("alpha_ptr", ctypes.c_void_p),
                ("beta_ptr", ctypes.c_void_p),
                ("dalpha", stride_type),
                ("dbeta", stride_type),
            ]

            def __init__(self, alpha, beta, *args) -> None:
                self.alpha = to_ctype_value(alpha, element_epilogue)
                self.beta = to_ctype_value(beta, element_epilogue)

        class _EpilogueOutputOpParams(ctypes.Structure):
            _fields_ = [
                ("alpha", c_element_epilogue),
                ("beta", c_element_epilogue),
                ("alpha_ptr", ctypes.c_void_p),
                ("beta_ptr", ctypes.c_void_p),
                ("alpha_ptr_array", ctypes.c_void_p),
                ("beta_ptr_array", ctypes.c_void_p),
            ]

            def __init__(self, alpha, beta, *args) -> None:
                self.alpha = to_ctype_value(alpha, element_epilogue)
                self.beta = to_ctype_value(beta, element_epilogue)

            def to_evt_params(self) -> _EpilogueOutputOpParamsEVT:
                return _EpilogueOutputOpParamsEVT(self.alpha, self.beta)

        self.epilogue_type = _EpilogueOutputOpParams
        self.epilogue_type_evt = _EpilogueOutputOpParamsEVT

    def emit(self):
        return super().emit(self.tag, self.template_arguments)


class LinearCombinationClamp(LinearCombination):
    """
    Applies a linear combination operator to an array of elements then clamps
    the output before converting to the output element type.

    D = alpha * accumulator + beta * source + uniform

    :param element_output: data type used to load and store tensors

    :param epilogue_vector_length: number of elements computed per operation.
    Usually it is 128/sizeof_bits_v<ElementOutput_>, but we use 64 and 32 sometimes
    when there are not enough data to store

    :param element_accumulator: Accumulator data type

    :param element_epilogue: data type used to compute linear combination
    """

    tag = "cutlass::epilogue::thread::LinearCombinationClamp"

    def __init__(
        self, element_output, epilogue_vector_length,
        element_accumulator=None, element_epilogue=None) -> None:
        # Base constructor
        super().__init__(
            element_output,
            epilogue_vector_length,
            element_accumulator,
            element_epilogue,
        )

        c_element_epilogue = dtype2ctype[self.element_epilogue]
        element_epilogue = self.element_epilogue

        class _EpilogueOutputOpParams(ctypes.Structure):
            _fields_ = [
                ("alpha", c_element_epilogue),
                ("beta", c_element_epilogue),
                ("alpha_ptr", ctypes.c_void_p),
                ("beta_ptr", ctypes.c_void_p),
            ]

            def __init__(self, alpha, beta, *args) -> None:
                self.alpha = to_ctype_value(alpha, element_epilogue)
                self.beta = to_ctype_value(beta, element_epilogue)

        self.epilogue_type = _EpilogueOutputOpParams


class FastLinearCombinationClamp(EpilogueFunctorBase):
    """
    Applies a linear combination operator to an array of elements then clamps
    the output before converting to the output element type.

    D = alpha * accumulator + beta * source

    Note: The below method only when problem_size_K <= 256 for signed int8 gemm
    or problem_size_K <= 128 for unsigned int8 gemm. The default approach is
    above.

    :param element_output: data type used to load and store tensors

    :param epilogue_vector_length: number of elements computed per operation.
    Usually it is 128/sizeof_bits_v<ElementOutput_>, but we use 64 and 32 sometimes
    when there are not enough data to store
    """

    tag = "cutlass::epilogue::thread::FastLinearCombinationClamp"

    def __init__(self, element_output, epilogue_vector_length, *args) -> None:
        super().__init__()

        self.template_arguments = [
            DataTypeTag[element_output], str(epilogue_vector_length)
        ]

        self.element_accumulator = DataType.s32
        self.element_epilogue = DataType.f32

        # get epilogue output op
        c_element_epilogue = dtype2ctype[self.element_epilogue]
        element_epilogue = self.element_epilogue

        class _EpilogueOutputOpParams(ctypes.Structure):
            _fields_ = [
                ("alpha", c_element_epilogue),
                ("beta", c_element_epilogue),
                ("alpha_ptr", ctypes.c_void_p),
                ("beta_ptr", ctypes.c_void_p),
            ]

            def __init__(self, alpha, beta, *args) -> None:
                self.alpha = to_ctype_value(alpha, element_epilogue)
                self.beta = to_ctype_value(beta, element_epilogue)

        self.epilogue_type = _EpilogueOutputOpParams

    def emit(self):
        return super().emit(self.tag, self.template_arguments)


class LinearCombinationGeneric(LinearCombination):
    """
    Applies a linear combination operator followed by an activation function
    to an array of elements.

    D = activation(alpha * accumulator + beta * source)

    :param activation_functor: input activation functor

    :param element_output: data type used to load and store tensors

    :param epilogue_vector_length: number of elements computed per operation.
    Usually it is 128/sizeof_bits_v<ElementOutput_>, but we use 64 and 32 sometimes
    when there are not enough data to store

    :param element_accumulator: Accumulator data type

    :param element_epilogue: data type used to compute linear combination
    """

    tag = "cutlass::epilogue::thread::LinearCombinationGeneric"

    def __init__(
        self, activation_functor,
        element_output, epilogue_vector_length,
        element_accumulator=None, element_epilogue=None) -> None:
        super().__init__(
            element_output,
            epilogue_vector_length,
            element_accumulator,
            element_epilogue,
        )

        self.template_arguments = [
            activation_functor.emit()] + self.template_arguments

        self.activation_functor = activation_functor
        self.element_epilogue = element_epilogue

        # get epilogue output op
        self.epilogue_type = self.activation_functor.epilogue_output_op(self.element_epilogue)


class ActivationFunctor:
    """
    Base class for frequently used activation functions
    """

    @staticmethod
    def numpy(x: np.ndarray):
        raise NotImplementedError()

    @classmethod
    def emit(cls):
        return ActivationOpTag[cls.binding_type]

    @staticmethod
    def epilogue_output_op(element_epilogue):
        c_element_epilogue = dtype2ctype[element_epilogue]

        class _EpilogueOutputOpParams(ctypes.Structure):
            _fields_ = [
                ("alpha", c_element_epilogue),
                ("beta", c_element_epilogue),
                ("alpha_ptr", ctypes.c_void_p),
                ("beta_ptr", ctypes.c_void_p),
            ]

            def __init__(self, alpha, beta, *args) -> None:
                self.alpha = to_ctype_value(alpha, element_epilogue)
                self.beta = to_ctype_value(beta, element_epilogue)

        return _EpilogueOutputOpParams

class ActivationMeta(type):
    @classmethod
    def __call__(cls, x, *args):
        if is_numpy_tensor(x):
            return cls.numpy(x, *args)
        elif is_torch_tensor(x):
            return cls.torch(x, *args)
        else:
            raise NotImplementedError("Unsupported tensor type")

    @classmethod
    def numpy(cls, *args):
        raise NotImplementedError(f"Numpy reference for {cls.__name__[:-4]} is not implemented.")

    @classmethod
    def torch(cls, *args):
        raise NotImplementedError(f"PyTorch reference for {cls.__name__[:-4]} is not implemented.")

##############################################################################
# identity operator
class identityMeta(ActivationMeta):
    @classmethod
    def numpy(cls, x):
        return x

    @classmethod
    def torch(cls, x):
        return x

class identity(ActivationFunctor, metaclass=identityMeta):
    binding_type = ActivationOp.Identity


##############################################################################
# ReLu operator
class reluMeta(ActivationMeta):
    @classmethod
    def numpy(cls, x):
        return np.where(x > 0, x, 0)

    @classmethod
    def torch(cls, x):
        return F.relu(x)

class relu(ActivationFunctor, metaclass=reluMeta):
    binding_type = ActivationOp.ReLU


##############################################################################
# Leaky ReLu operator
class leakyReLUMeta(ActivationMeta):
    @classmethod
    def numpy(cls, x, leaky_alpha):
        return np.maximum(x, 0) + np.minimum(x, 0) * leaky_alpha

    @classmethod
    def torch(cls, x, leaky_alpha):
        return F.leaky_relu(x, leaky_alpha)

class leaky_relu(ActivationFunctor, metaclass=leakyReLUMeta):
    binding_type = ActivationOp.LeakyReLU

    @staticmethod
    def epilogue_output_op(element_epilogue):
        c_element_epilogue = dtype2ctype[element_epilogue]

        class _EpilogueOutputOpParams(ctypes.Structure):
            _fields_ = [
                ("alpha", c_element_epilogue),
                ("beta", c_element_epilogue),
                ("alpha_ptr", ctypes.c_void_p),
                ("beta_ptr", ctypes.c_void_p),
                ("leaky_alpha", c_element_epilogue)
            ]

            def __init__(self, alpha, beta, leaky_alpha=0.2, *args) -> None:
                self.alpha = to_ctype_value(alpha, element_epilogue)
                self.beta = to_ctype_value(beta, element_epilogue)
                self.alpha_ptr = 0
                self.beta_ptr = 0
                self.leaky_alpha = to_ctype_value(leaky_alpha, element_epilogue)

        return _EpilogueOutputOpParams


##############################################################################
# Tanh operator
class tanhMeta(ActivationMeta):
    @classmethod
    def numpy(cls, x):
        return np.tanh(x)

    @classmethod
    def torch(cls, x):
        return torch.tanh(x)

class tanh(ActivationFunctor, metaclass=tanhMeta):
    binding_type = ActivationOp.Tanh


##############################################################################
# Sigmoid operator
class sigmoidMeta(ActivationMeta):
    @classmethod
    def numpy(cls, x):
        return 1.0 / (1.0 + np.exp(-x))

    @classmethod
    def torch(cls, x):
        return F.sigmoid(x)

class sigmoid(ActivationFunctor, metaclass=sigmoidMeta):
    binding_type = ActivationOp.Sigmoid


##############################################################################
# SiLu operator
class siluMeta(ActivationMeta):
    @classmethod
    def numpy(cls, x):
        return x * sigmoidMeta.numpy()

    @classmethod
    def silu(cls, x):
        return F.silu(x)


class silu(ActivationFunctor, metaclass=siluMeta):
    binding_type = ActivationOp.SiLU


##############################################################################
# Hardswish operator
class hardswishMeta(ActivationMeta):
    @classmethod
    def numpy(cls, x):
        relu6 = np.minimum(np.maximum(x + 3.0, 0), 6.0)
        return x * relu6 / 6.0

    @classmethod
    def torch(cls, x):
        return F.hardswish(x)


class hardswish(ActivationFunctor, metaclass=hardswishMeta):
    binding_type = ActivationOp.HardSwish


##############################################################################
# GELU operator
class geluMeta(ActivationMeta):
    @classmethod
    def numpy(cls, x):
        from scipy.special import erf
        return 0.5 * x * (1 + erf(x / np.sqrt(2.0)))

    @classmethod
    def torch(cls, x):
        return F.gelu(x)


class gelu(ActivationFunctor, metaclass=geluMeta):
    binding_type = ActivationOp.Gelu
