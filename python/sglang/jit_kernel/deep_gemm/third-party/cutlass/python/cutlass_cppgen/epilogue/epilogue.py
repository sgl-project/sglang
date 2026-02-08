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
Registry of elementwise epilogues

Elementwise epilogues can be added to many CUTLASS kernels in the CUTLAS Python interface via
code like the following for GEMM:

.. highlight:: python
.. code-block:: python

    plan = cutlass_cppgen.op.Gemm(element=cutlass_cppgen.DataType.f32, layout=cutlass_cppgen.LayoutType.RowMajor)
    plan.activation = cutlass_cppgen.epilogue.relu
"""

from cutlass_cppgen.backend import epilogue, device_cc


gelu = epilogue.gelu
hardswish = epilogue.hardswish
identity = epilogue.identity
leaky_relu = epilogue.leaky_relu
relu = epilogue.relu
sigmoid = epilogue.sigmoid
silu = epilogue.silu
tanh = epilogue.tanh


_activations = [gelu, hardswish, identity, leaky_relu, relu, sigmoid, silu, tanh]


def get_activations() -> list:
    """
    Returns a list of available activation functions

    :return: list of available activation functions
    :rtype: list
    """
    return _activations


def get_activation_epilogue(
    activation,
    element_output,
    elements_per_access,
    element_accumulator,
    element_compute,
):
    """
    Return an epilogue corresponding to the activation function, data types, and alignment
    used in the kernel

    :param activation: elementwise activation function to use
    :param element_output: data type of the output
    :param elements_per_access: alignment of operand C of the kernel
    :type elements_per_access: int
    :param element_accumulator: data type of the accumulated output C
    :param element_compute: data type in which compute operations should be performed

    :return: epilogue functor
    """
    if activation not in _activations:
        raise Exception(
            f"Unsupported activation type {activation}. Available activations are: {_activations}"
        )

    if activation == identity:
        return epilogue.LinearCombination(
            element_output, elements_per_access, element_accumulator, element_compute
        )
    else:
        return epilogue.LinearCombinationGeneric(
            activation,
            element_output,
            elements_per_access,
            element_accumulator,
            element_compute,
        )


"""
Frontend for EVT that generates epilogue functor through tracing the input function
"""
from cutlass_cppgen.backend.evt.frontend import PythonASTFrontend


def trace(fn, example_tensors, **kwargs):
    """
    Trace `fn(**example_tensors)` and generates epilogue visitor

    :param fn or str: Python callable or string of the epilogue function
    :param example_tensors: example inputs for fn
    :type example_tensors: dict

    .. hightlight:: python
    .. code-block:: python
        import cutlass_cppgen.backend.evt

        # Define epilogue function as Python callable
        def example_fn(accum, C, alpha, beta, gamma):
            D = ((accum + C) * alpha - gamma) / beta
            return D

        # Define the example tensors
        example_inputs = {
            "accum": torch.empty(size=(6, 512, 512), dtype=torch.float16, device="cuda"),
            "C": torch.empty(size=(6, 512, 512), dtype=torch.float16, device="cuda"),
            "alpha": 1.5,
            "beta": 0.5,
            "gamma": 2.5,
            "D": torch.empty(size=(6, 512, 512), dtype=torch.float16, device="cuda")
        }

        # Generate the epilogue functor
        epilogue_visitor = cutlass_cppgen.epilogue.trace(example_fn, example_inputs)
    """
    if callable(fn):
        class EpilogueFunctor(PythonASTFrontend):
            def __init__(self, cc=None, **kwargs):
                if not cc:
                    cc = device_cc()
                super().__init__(cc, **kwargs)
            pass
        setattr(EpilogueFunctor, "__call__", staticmethod(fn))

        epilogue_functor = EpilogueFunctor(**kwargs)
        epilogue_functor.trace(example_tensors)
        return epilogue_functor
    elif isinstance(fn, str):
        class EpilogueFunctor(PythonASTFrontend):
            def __init__(self, cc=None, **kwargs):
                self.source = textwrap.dedent(fn)
                if not cc:
                    cc = device_cc()
                super().__init__(cc, **kwargs)

            def parse(self, example_inputs) -> None:
                self.example_inputs = example_inputs
                self.ast = ast.parse(self.source)
                self.visit(self.ast)

        epilogue_functor = EpilogueFunctor(**kwargs)
        epilogue_functor.trace(example_tensors)
        return epilogue_functor
    else:
        raise NotImplementedError("Expect a callable Python function")
