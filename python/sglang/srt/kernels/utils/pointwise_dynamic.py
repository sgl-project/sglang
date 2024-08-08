"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Adapted from https://github.com/FlagOpen/FlagGems/blob/4f6ea4cc2fc004d806d48a7d8f21dcb6a79a6dc5/src/flag_gems/utils/pointwise_dynamic.py

import importlib
import os
import threading
from typing import Any, Callable, List, Mapping, Optional, Tuple

import torch
import torch._prims_common as utils
import triton
from triton import language as tl
from triton.runtime.jit import JITFunction

from sglang.srt.kernels.utils.triton_utils import (
    IndentedBuffer,
    NameSpace,
    broadcast_shapes,
    cache_dir,
)


# ------------------ Operation Description ---------------------------
def _type_name(type) -> str:
    "Render typename as string, work for both (bool, int, float, str) and torch.dtype object"
    if type in (bool, int, float, str):
        return type.__name__
    if isinstance(type, torch.dtype):
        return str(type)
    return str(type)


def _check_typed_list(container, type):
    for item in container:
        assert isinstance(item, type)


def _check_sized_list(container, size):
    assert len(container) == size


class OPDesc:
    _num_inputs: int
    _is_tensor: List[bool]
    _dtypes: List[Optional[type]]

    _num_input_tensors: int
    _num_non_tensor_inputs: int

    _num_outputs: int
    _promotion_methods: List[Tuple[int, ...]]

    def __init__(
        self,
        *,
        num_inputs: Optional[int] = None,
        is_tensor: Optional[List[bool]] = None,
        dtypes: Optional[List[Optional[type]]] = None,
        num_outputs: Optional[int] = None,
        promotion_methods: Optional[List[Tuple[int, ...]]] = None,
    ):
        if is_tensor is not None:
            _check_typed_list(is_tensor, bool)
        if dtypes is not None:
            _check_typed_list(dtypes, (type, type(None)))
        if promotion_methods is None:
            raise ValueError(
                "No type promotion method provided! You must provide type promotion method for each output!"
            )
        else:
            self._promotion_methods = promotion_methods

        if num_inputs is not None:
            self._num_inputs = num_inputs
            if is_tensor is not None:
                _check_sized_list(is_tensor, num_inputs)
                self._is_tensor = is_tensor
            else:
                self._is_tensor = [True] * num_inputs

            if dtypes is not None:
                _check_sized_list(dtypes, num_inputs)
                self._dtypes = dtypes
            else:
                self._dtypes = [None] * num_inputs
        elif is_tensor is not None:
            self._num_inputs = len(is_tensor)
            self._is_tensor = is_tensor
            if dtypes is not None:
                _check_sized_list(dtypes, self._num_inputs)
                self._dtypes = dtypes
            else:
                self._dtypes = [None] * self._num_inputs
        elif dtypes is not None:
            self._num_inputs = len(dtypes)
            self._dtypes = dtypes
            if is_tensor is not None:
                _check_sized_list(is_tensor, self._num_inputs)
                self._is_tensor = is_tensor
            else:
                self._is_tensor = [item is None for item in dtypes]
        else:
            raise ValueError(
                "Cannot make OPDesc when none of (num_inputs, is_tensor, dtypes) is specified."
            )

        if num_outputs is not None:
            self._num_outputs = num_outputs
            _check_sized_list(promotion_methods, num_outputs)
        else:
            self._num_outputs = len(promotion_methods)

        assert self._num_inputs >= 1
        assert self._num_outputs >= 1

        self._num_input_tensors = sum(self._is_tensor)
        self._num_non_tensor_inputs = self._num_inputs - self._num_input_tensors

    def num_inputs(self):
        # num of arguments, outputs not included
        return self._num_inputs

    def num_outputs(self):
        return self._num_outputs

    def is_tensor(self, arg_id: int) -> bool:
        return self._is_tensor[arg_id]

    def input_type(self, arg_id) -> Optional[type]:
        return self._dtypes[arg_id]

    def num_input_tensors(self) -> int:
        return self._num_input_tensors

    def num_output_tensors(self) -> int:
        return self._num_outputs

    def num_non_tensor_args(self) -> int:
        return self._num_non_tensor_inputs

    def type_promotion_methods(self) -> List[Tuple[int, ...]]:
        return self._promotion_methods

    def _match_enum_by_string(
        self, input_str: str
    ) -> utils.ELEMENTWISE_TYPE_PROMOTION_KIND:
        for kind in utils.ELEMENTWISE_TYPE_PROMOTION_KIND:
            if input_str.lower() == kind.name.lower():
                return kind
        raise ValueError(f"No matching enum member found for input: {input_str}")

    def ith_type_promotion_args(self, i) -> List[int]:
        return self._promotion_methods[i][:-1]

    def ith_type_promotion_kind(self, i) -> utils.ELEMENTWISE_TYPE_PROMOTION_KIND:
        return self._match_enum_by_string(self._promotion_methods[i][-1])

    def signature(self, outputs_in_arg: bool = False):
        input_types = []
        for is_tensor, dtype in zip(self._is_tensor, self._dtypes):
            if is_tensor:
                input_types.append("Tensor")
            else:
                if dtype is None:
                    input_types.append("scalar")
                else:
                    input_types.append(_type_name(dtype))

        output_types = []
        for _ in range(self.num_outputs()):
            output_types.append("Tensor")
        if outputs_in_arg:
            input_types.extend(output_types)
        sig = f'Pointwise: ({", ".join(input_types)}) -> ({", ".join(output_types)})'
        return sig

    def __str__(self) -> str:
        return self.signature(outputs_in_arg=False)


# --------------------------- pointwise wrapper genration -----------------------------------
def parameter_for_wrapper(op_desc: OPDesc, include_outputs: bool = False) -> str:
    """Generate parameter declaration with type annotation for wrapper function.
    Example: in0: torch.Tensor, val0: float, out0: torch.Tensor
    """
    parameters: List[str] = []

    input_tensor_index = 0
    non_tensor_index = 0
    for i in range(op_desc.num_inputs()):
        if op_desc._is_tensor[i]:
            parameters.append(f"in{input_tensor_index}: torch.Tensor")
            input_tensor_index += 1
        else:
            if op_desc.input_type(i) is not None:
                parameters.append(
                    f"val{non_tensor_index}: {_type_name(op_desc.input_type(i))}"
                )
            else:
                parameters.append(f"val{non_tensor_index}")
            non_tensor_index += 1

    if include_outputs:
        output_tensor_index = 0
        for i in range(op_desc.num_outputs()):
            parameters.append(f"out{output_tensor_index}: torch.Tensor")
            output_tensor_index += 1

    parameters.append("**kwargs")

    return ", ".join(parameters)


def ith_parameter_for_type_promotion(op_desc: OPDesc, ith: int) -> str:
    """Generate parameter reference for i-th type promotion rule
    Example: in0, val0, out0
    """
    parameters: List[str] = []

    input_tensor_index = 0
    non_tensor_index = 0
    for i in range(op_desc.num_inputs()):
        if i not in op_desc.ith_type_promotion_args(ith):
            if op_desc._is_tensor[i]:
                input_tensor_index += 1
            else:
                non_tensor_index += 1
            continue
        if op_desc._is_tensor[i]:
            parameters.append(f"in{input_tensor_index}")
            input_tensor_index += 1
        else:
            parameters.append(f"val{non_tensor_index}")
            non_tensor_index += 1

    return ", ".join(parameters)


def parameter_ref_for_wrapper(
    op_desc: OPDesc,
    include_outputs: bool = False,
    include_offset: bool = False,
    include_kwargs: bool = False,
) -> str:
    """Generate parameter reference for wrapper function.
    Example: in0, val0, out0, out0_offset
    """
    parameters: List[str] = []

    input_tensor_index = 0
    non_tensor_index = 0
    for i in range(op_desc.num_inputs()):
        if op_desc._is_tensor[i]:
            parameters.append(f"in{input_tensor_index}")
            input_tensor_index += 1
        else:
            parameters.append(f"val{non_tensor_index}")
            non_tensor_index += 1

    if include_outputs:
        output_tensor_index = 0
        for i in range(op_desc.num_outputs()):
            parameters.append(f"out{output_tensor_index}")
            if include_offset:
                parameters.append(f"out{output_tensor_index}_offset")
            output_tensor_index += 1

    if include_kwargs:
        parameters.append("**kwargs")

    return ", ".join(parameters)


def output_ref_for_wrapper(op_desc: OPDesc) -> str:
    """Generate output variable refernece for wrapper function.
    Example: out0, out1
    """
    parameters: List[str] = [f"out{i}" for i in range(op_desc.num_outputs())]
    return ", ".join(parameters)


def docstring_for_functional_wrapper(op_desc: OPDesc):
    doc = f'"""Generated wrapper function with {str(op_desc)}"""'
    return doc


def docstring_for_destination_passing_wrapper(op_desc: OPDesc):
    doc = f'"""Generated wrapper function with {op_desc.signature(outputs_in_arg=True)}"""'
    return doc


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import math")
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("from triton import language as tl")
    code.newline()
    code.writeline("from sglang.srt.kernels.utils.triton_utils import (")
    code.writeline("    broadcast_shapes,")
    code.writeline("    broadcasted_stride,")
    code.writeline("    c_contiguous_stride,")
    code.writeline("    volume,")
    code.writeline("    libentry,")
    code.writeline("    type_promotion,")
    code.writeline(")")
    code.writeline("import torch._prims_common as utils")
    code.newline()
    code.newline()
    return code


def generate_functional_pointwise_wrapper(
    op_desc: OPDesc,
    wrapper_name: str,
    destination_passing_func_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # wrapper signature
    parameters: str = parameter_for_wrapper(op_desc, include_outputs=False)
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        # docstring
        wrapper_docstring = docstring_for_functional_wrapper(op_desc)
        code.writeline(wrapper_docstring)

        shapes_str = ", ".join(
            f"in{i}.shape" for i in range(op_desc.num_input_tensors())
        )
        code.writeline(f"shape = broadcast_shapes([{shapes_str}])")

        # output allocation
        num_output_tensor_index = 0
        for i in range(op_desc.num_outputs()):
            type_promotion_args = ith_parameter_for_type_promotion(op_desc, i)
            k_type_promotion = op_desc.ith_type_promotion_kind(i)
            code.writeline(
                (
                    f"out{num_output_tensor_index} = "
                    f"torch.empty(shape, dtype=type_promotion"
                    f"({type_promotion_args}, type_promotion=utils.{k_type_promotion})[1], "
                    f"device=in0.device)"
                )
            )
            num_output_tensor_index += 1

        # call destination_passing_func
        output_names: str = output_ref_for_wrapper(op_desc)
        call_str = (
            f"{output_names} = {destination_passing_func_name}"
            f"({parameter_ref_for_wrapper(op_desc, include_outputs=True, include_offset=False, include_kwargs=True)})"
        )
        code.writeline(call_str)

        return_str = f"return {output_names}"
        code.writeline(return_str)
        code.newline()
        code.newline()
    return code


def generate_destination_passing_pointwise_wrapper(
    op_desc: OPDesc,
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # wrapper signature
    parameters: str = parameter_for_wrapper(op_desc, include_outputs=True)
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        # docstring
        wrapper_docstring = docstring_for_destination_passing_wrapper(op_desc)
        code.writeline(wrapper_docstring)

        if rank > 0:
            code.writeline("shape = out0.shape")
            code.writeline("num_tasks = volume(shape)")

        if rank > 0:
            code.writeline("tile_size = min(512, triton.next_power_of_2(num_tasks))")
            code.writeline("num_warps = 4")
            code.writeline("num_ctas = min(65535, triton.cdiv(num_tasks, tile_size))")
            code.writeline(
                "tiles_per_cta = triton.cdiv(num_tasks, tile_size * num_ctas)"
            )
        else:
            code.writeline("num_warps = 1")
            code.writeline("num_ctas = 1")
        code.writeline("grid = (num_ctas, 1, 1)")
        code.newline()

        # input strides for each input tensor w.r.t. the task index space
        if rank > 0:
            code.writeline("# strides of each tensor argument w.r.t the task space")
            for i in range(op_desc.num_input_tensors()):
                code.writeline(
                    f"in{i}_strides = broadcasted_stride(in{i}.shape, in{i}.stride(), shape)"
                )
            for i in range(op_desc.num_output_tensors()):
                code.writeline(f"if 'out{i}_offset' in kwargs:")
                with code.indent():
                    code.writeline(f"out{i}_offset = kwargs['out{i}_offset']")
                code.writeline("else:")
                with code.indent():
                    code.writeline(f"out{i}_offset = 0")

                code.writeline(f"if 'out{i}_strides' in kwargs:")
                with code.indent():
                    code.writeline(f"out{i}_strides = kwargs['out{i}_strides']")
                code.writeline("else:")
                with code.indent():
                    code.writeline(f"out{i}_strides = out{i}.stride()")
        else:
            for i in range(op_desc.num_output_tensors()):
                code.writeline(f"out{i}_offset = 0")
        code.newline()

        # grid
        code.writeline("# kernel launch")

        # launch kernel
        code.writeline("with torch.cuda.device(in0.device.index):")
        with code.indent():
            kernel_launch: str = f"{kernel_name}[grid]("
            code.writeline(kernel_launch)

            with code.indent():
                code.writeline(
                    "{},".format(
                        parameter_ref_for_wrapper(
                            op_desc,
                            include_outputs=True,
                            include_offset=True,
                            include_kwargs=False,
                        )
                    )
                )

                if rank > 0:
                    for i in range(op_desc.num_input_tensors()):
                        s = ", ".join(f"in{i}_strides[{j}]" for j in range(rank))
                        code.writeline(f"{s}, # stride for in{i}")

                    for i in range(op_desc.num_output_tensors()):
                        s = ", ".join(f"out{i}_strides[{j}]" for j in range(rank))
                        code.writeline(f"{s}, # stride for out{i}")

                    shape_args: str = ", ".join(f"shape[{i}]" for i in range(rank))
                    code.writeline(f"{shape_args}, # task indexing space")
                    code.writeline("num_tasks, # num tasks")
                    code.writeline("tiles_per_cta=tiles_per_cta, # tiles_per_cta")
                    code.writeline("tile_size=tile_size,")
                    code.writeline("one_tile_per_cta=tiles_per_cta==1,")
                code.writeline("num_warps=num_warps,")
            code.writeline(")")

        # return
        code.writeline(f"return {output_ref_for_wrapper(op_desc)}")
        code.newline()
        code.newline()
    return code


def generate_pointwise_kernel(
    op_desc: OPDesc,
    scalar_fn: JITFunction,
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # make the inlined function visible in the context
    fn_name = scalar_fn.__name__
    code.writeline(f"from {scalar_fn.__module__} import {fn_name}")
    code.writeline(f"inlined_f = {fn_name}._scalar_fn")
    code.newline()

    # the decorators
    code.writeline("@libentry()")
    if op_desc.num_non_tensor_args() > 0:
        # we do not specialize non tensor args since they are passed into the inlined function
        # which means that their values may not deserve specialization
        non_specialize_arg_names = [
            f"val{i}" for i in range(op_desc.num_non_tensor_args())
        ]
        code.writeline(f"@triton.jit(do_not_specialize={non_specialize_arg_names})")
    else:
        code.writeline("@triton.jit")

    # signature
    code.writeline(f"def {kernel_name}(")
    function_ns = NameSpace()
    with code.indent():
        input_tensor_index = 0
        non_tensor_index = 0
        output_tensor_index = 0
        # signature: inputs ptrs & non tensor inputs
        for i in range(op_desc.num_inputs()):
            if op_desc.is_tensor(i):
                code.writeline(
                    f"in{input_tensor_index}_ptr: tl.tensor, # of tl.pointer_type"
                )
                function_ns.create_name(f"in{input_tensor_index}_ptr")
                input_tensor_index += 1
            else:
                if op_desc.input_type(i) is not None:
                    code.writeline(
                        f"val{non_tensor_index}: {_type_name(op_desc.input_type(i))},"
                    )
                else:
                    code.writeline(f"val{non_tensor_index},")
                function_ns.create_name(f"val{non_tensor_index}")
                non_tensor_index += 1

        # signature: output ptrs
        for i in range(op_desc.num_outputs()):
            code.writeline(
                f"out{output_tensor_index}_ptr: tl.tensor, # of tl.pointer_type"
            )
            code.writeline(f"out{output_tensor_index}_offset: int,")
            function_ns.create_name(f"out{output_tensor_index}_ptr")
            function_ns.create_name(f"out{output_tensor_index}_offset")
            output_tensor_index += 1

        # signature: strides, for each tensor arguments
        # only add this arguments when rank > 0
        if rank > 0:
            # strides for inputs
            for i in range(op_desc.num_input_tensors()):
                for j in range(rank):
                    function_ns.create_name(f"in{i}_stride{j}")
                stride_args = ", ".join(f"in{i}_stride{j}: int" for j in range(rank))
                code.writeline(f"{stride_args}, # strides for in{i}")

            # strides for outputs
            for i in range(op_desc.num_output_tensors()):
                for j in range(rank):
                    function_ns.create_name(f"out{i}_stride{j}")
                stride_args = ", ".join(f"out{i}_stride{j}: int" for j in range(rank))
                code.writeline(f"{stride_args}, # strides for out{i}")

            # task space, used to reconstruct multi index
            task_space_args = ", ".join(f"s{i}: int" for i in range(rank))
            for i in range(rank):
                function_ns.create_name(f"s{i}")
            code.writeline(f"{task_space_args}, # task_space")

            # number of tasks, used to compute mask
            code.writeline("num_tasks: int,")
            function_ns.create_name("num_tasks")

        # tile size & tiles_per_cta, gsl style
        if rank > 0:
            code.writeline("tiles_per_cta,")
            function_ns.create_name("tiles_per_cta")

            code.writeline("tile_size: tl.constexpr,")
            function_ns.create_name("tile_size")

            code.writeline("one_tile_per_cta: tl.constexpr,")
            function_ns.create_name("one_tile_per_cta")
    code.writeline("):")

    # input & output names
    inputs_to_scalar_fn = []
    input_tensor_index = 0
    non_tensor_index = 0
    for i in range(op_desc.num_inputs()):
        if op_desc.is_tensor(i):
            inputs_to_scalar_fn.append(f"in{input_tensor_index}")
            input_tensor_index += 1
        else:
            inputs_to_scalar_fn.append(f"val{non_tensor_index}")
            non_tensor_index += 1
    inputs_to_scalar_fn: str = ", ".join(inputs_to_scalar_fn)

    outputs_to_scalar_fn = [f"out{i}" for i in range(op_desc.num_outputs())]
    outputs_to_scalar_fn: str = ", ".join(outputs_to_scalar_fn)

    # function body for rank-0
    if rank == 0:
        with code.indent():
            code.writeline("# loads")
            for i in range(op_desc.num_input_tensors()):
                ptrs_expr: str = f"in{i}_ptr"
                load_stmt: str = f"in{i} = tl.load({ptrs_expr})"
                function_ns.create_name(f"in{i}")  # add to the namespace
                code.writeline(load_stmt)
            code.newline()

            code.writeline("# compute")
            code.writeline(f"{outputs_to_scalar_fn} = inlined_f({inputs_to_scalar_fn})")
            code.newline()

            code.writeline("# stores")
            for i in range(op_desc.num_output_tensors()):
                ptrs_expr: str = f"out{i}_ptr + out{i}_offset"
                store_stmt: str = f"tl.store({ptrs_expr}, out{i})"
                code.writeline(store_stmt)
            code.newline()
            return code

    with code.indent():
        # get pid
        code.writeline("# task id & masking")
        pid_stmt = "pid = tl.program_id(0)"
        code.writeline(pid_stmt)
        function_ns.create_name("pid")

        code.writeline("num_ctas = tl.num_programs(0)")
        function_ns.create_name("num_ctas")

        # get tid (a.k.a task id)
        tid_stmt = "init_tid = pid * tile_size + tl.arange(0, tile_size)"
        code.writeline(tid_stmt)
        function_ns.create_name("init_tid")

        # one-tile-per-cta, monolithic kernel style
        code.writeline("if one_tile_per_cta: # monolitic kernel style")
        with code.indent():
            tid_stmt = "tid = init_tid"
            code.writeline(tid_stmt)
            function_ns.create_name("tid")

            # only apply masking when rank > 0
            # since we only load a value instead of a block of values when the rank is 0
            mask_stmt: str = "mask = tid < num_tasks"
            code.writeline(mask_stmt)
            function_ns.create_name("mask")
            code.newline()

            # reconstruct multi index
            code.writeline("# multi index recontruction")
            for i in reversed(range(rank)):
                if i > 0:
                    code.writeline(f"i{i} = tid % s{i}")
                    code.writeline(f"tid //= s{i}")
                else:
                    code.writeline(f"i{i} = tid")
                function_ns.create_name(f"{i}")
            code.newline()

            # loads
            code.writeline("# loads")
            for i in range(op_desc.num_input_tensors()):
                ptrs_expr: str = " + ".join(
                    f"i{j} * in{i}_stride{j}" for j in range(rank)
                )
                ptrs_expr: str = f"in{i}_ptr + {ptrs_expr}"
                load_stmt: str = f"in{i} = tl.load({ptrs_expr}, mask=mask)"
                function_ns.create_name(f"in{i}")  # add to the namespace
                code.writeline(load_stmt)
            code.newline()

            # compute
            code.writeline("# compute")
            code.writeline(f"{outputs_to_scalar_fn} = inlined_f({inputs_to_scalar_fn})")
            code.newline()

            # stores
            code.writeline("# stores")
            for i in range(op_desc.num_output_tensors()):
                ptrs_expr: str = " + ".join(
                    f"i{j} * out{i}_stride{j}" for j in range(rank)
                )
                ptrs_expr: str = f"out{i}_ptr + out{i}_offset + {ptrs_expr}"
                store_stmt: str = f"tl.store({ptrs_expr}, out{i}, mask=mask)"
                code.writeline(store_stmt)

        # https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
        code.writeline("else: # grid-stride-loop style kernel")
        with code.indent():
            code.writeline("for j in range(0, tiles_per_cta):")
            function_ns.create_name("j")
            with code.indent():
                tid_stmt = "tid = init_tid + j * tile_size * num_ctas"
                code.writeline(tid_stmt)
                function_ns.create_name("tid")

                # only apply masking when rank > 0
                # since we only load a value instead of a block of values when the rank is 0
                mask_stmt: str = "mask = tid < num_tasks"
                code.writeline(mask_stmt)
                function_ns.create_name("mask")
                code.newline()

                # reconstruct multi index
                code.writeline("# multi index recontruction")
                for i in reversed(range(rank)):
                    if i > 0:
                        code.writeline(f"i{i} = tid % s{i}")
                        code.writeline(f"tid //= s{i}")
                    else:
                        code.writeline(f"i{i} = tid")
                    function_ns.create_name(f"{i}")
                code.newline()

                # loads
                code.writeline("# loads")
                for i in range(op_desc.num_input_tensors()):
                    ptrs_expr: str = " + ".join(
                        f"i{j} * in{i}_stride{j}" for j in range(rank)
                    )
                    ptrs_expr: str = f"in{i}_ptr + {ptrs_expr}"
                    load_stmt: str = f"in{i} = tl.load({ptrs_expr}, mask=mask)"
                    function_ns.create_name(f"in{i}")  # add to the namespace
                    code.writeline(load_stmt)
                code.newline()

                # compute
                code.writeline("# compute")
                code.writeline(
                    f"{outputs_to_scalar_fn} = inlined_f({inputs_to_scalar_fn})"
                )
                code.newline()

                # stores
                code.writeline("# stores")
                for i in range(op_desc.num_output_tensors()):
                    ptrs_expr: str = " + ".join(
                        f"i{j} * out{i}_stride{j}" for j in range(rank)
                    )
                    ptrs_expr: str = f"out{i}_ptr + out{i}_offset + {ptrs_expr}"
                    store_stmt: str = f"tl.store({ptrs_expr}, out{i}, mask=mask)"
                    code.writeline(store_stmt)
                code.newline()
    return code


def generate_code(
    op_desc: OPDesc,
    scalar_fn: JITFunction,
    inputs: Tuple[Any],
    wrapper_name: str,
    destination_passing_func_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    assert (
        len(inputs) == op_desc.num_inputs()
    ), "the number of inputs does not match {str(op_desc)}"
    input_tensor_ids = [i for i in range(op_desc.num_inputs()) if op_desc.is_tensor(i)]
    tensor_shapes = [inputs[i].shape for i in input_tensor_ids]
    shape = broadcast_shapes(tensor_shapes)
    rank = len(shape)

    # the only runtime determined factor is the rank of the task space
    code = generate_imports(code)
    code = generate_functional_pointwise_wrapper(
        op_desc, wrapper_name, destination_passing_func_name, code
    )
    code = generate_destination_passing_pointwise_wrapper(
        op_desc, rank, destination_passing_func_name, kernel_name, code
    )
    code = generate_pointwise_kernel(op_desc, scalar_fn, rank, kernel_name, code)
    return code


class PointwiseDynamicFunction:
    """Utility to generate function for general pointwise operation. It generate wrapper & JITFunction
    which are specialized according to the rank of the task space(the broadcasted shape of all input tensors).
    The generated code are written out to the cache directory.
    """

    def __init__(self, op_desc: OPDesc, scalar_fn: JITFunction):
        self._op_desc = op_desc

        assert isinstance(scalar_fn, JITFunction)
        self._scalar_fn = scalar_fn
        self._scalar_fn_cache_key = scalar_fn.cache_key
        self.pid = os.getpid()
        self.lock = threading.Lock()

        # instantiated & cached overloads
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        # note: kwargs should not be used in JITFunction directly
        key = f"{self.arg_key(*args)}"
        cache = self.overloads
        lock = self.lock

        while key not in cache:
            # generate file & import it
            with lock:
                if key in cache:
                    break
                code = IndentedBuffer()
                code = generate_code(
                    self._op_desc,
                    self._scalar_fn,
                    args,
                    "_wrapper",
                    "_wrapper_out",
                    "_jit_function",
                    code,
                )

                file_name = f"pointwise_dynamic_{self._scalar_fn_cache_key}_rank_{key}_pid_{self.pid}.py"

                with open(
                    os.path.join(cache_dir(), file_name), "wt", encoding="utf-8"
                ) as f:
                    f.write(code.getvalue())

                # load
                spec = importlib.util.spec_from_file_location(
                    f"_gen_module_{self._scalar_fn_cache_key}_rank_{key}_pid_{self.pid}",
                    f.name,
                )
                m = importlib.util.module_from_spec(spec)
                # do not expose it to sys.modules
                # sys.modules["_add_module"] = m
                spec.loader.exec_module(m)
                overload = getattr(m, "_wrapper")
                cache[key] = overload

        overload = self.overloads[key]
        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


def pointwise_dynamic(
    f: Optional[JITFunction] = None,
    *,
    num_inputs: Optional[int] = None,
    is_tensor: Optional[List[bool]] = None,
    dtypes: Optional[List[Optional[type]]] = None,
    num_outputs: Optional[int] = None,
    promotion_methods: Optional[Tuple[int, ...]] = None,
):
    def decorator(fn):
        nonlocal num_inputs
        if (num_inputs is None) and (is_tensor is None) and (dtypes is None):
            num_inputs = len(fn.arg_names)
        op_desc = OPDesc(
            num_inputs=num_inputs,
            is_tensor=is_tensor,
            dtypes=dtypes,
            num_outputs=num_outputs,
            promotion_methods=promotion_methods,
        )
        return PointwiseDynamicFunction(op_desc, fn)

    if f is not None:
        return decorator(f)
    return decorator
