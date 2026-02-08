# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

import functools
import inspect
import logging
import os
from enum import Enum
from inspect import isclass
from itertools import product
from time import time
from typing import Any, Callable, Dict, List, Optional, Type, Union

import cuda.bindings.driver as cuda_driver
import cuda.bindings.runtime as cuda_runtime
import numpy as np

import cutlass._mlir.ir as ir
import cutlass.base_dsl.jit_executor
import cutlass.cute as cute
from cutlass._mlir.dialects import builtin, cf, nvvm, vector
from cutlass.cute import core, nvgpu
from cutlass.cutlass_dsl import Constexpr, CuTeDSL, T, t, dsl_user_op


@dsl_user_op
def assert_(cond, msg=None, *, loc=None, ip=None):
    cf.assert_(t.Boolean(cond).ir_value(), msg if msg else "", loc=loc, ip=ip)


def _maybe_recast_tensor_from_f4(src: core.Tensor, tv_layout: core.Layout):
    if src.element_type.width == 4:
        tv_layout = core.recast_layout(8, 4, tv_layout)
        src = core.recast_tensor(src, dtype=t.Int8)
    return src, tv_layout


def _maybe_recast_to_f4(input: core.TensorSSA, dtype: Type[core.Numeric]):
    """Conditionally recasts the tensor to 4-bit type if the destination type is 4-bit.

    :param input: The input tensor to recast.
    :param dtype: The target numeric type to potentially recast to.
    :raises TypeError: If dtype is not a subclass of Numeric.
    :return: A new tensor recast to 4-bit if dtype is 4-bit, otherwise returns self unchanged.
    """
    if not isclass(dtype) or not issubclass(dtype, core.Numeric):
        raise TypeError(f"dst_ty must be a type of Numeric, but got {dtype}")

    if dtype.width == 4:
        recast_shape = core.recast_layout(4, 8, core.make_layout(input.shape)).shape
        i4_vec = vector.bitcast(
            T.vector(input.type.shape[0] * 2, T.i(4)), input.maybe_downcast()
        )
        res_vect = builtin.unrealized_conversion_cast(
            [T.vector(i4_vec.type.shape[0], dtype.mlir_type)], [i4_vec]
        )
        return core.TensorSSA(res_vect, recast_shape, dtype)
    return input


def _maybe_recast_from_f4(input: core.TensorSSA, src_dtype: Type[core.Numeric]):
    """Conditionally recasts the tensor from 4-bit type if the source type is 4-bit.

    :param input: The input tensor to recast.
    :param src_dtype: The source numeric type to potentially recast from.
    :raises TypeError: If src_dtype is not a subclass of Numeric.
    :return: A new tensor recast from 4-bit if src_dtype is 4-bit, otherwise returns self unchanged.
    """
    if not isclass(src_dtype) or not issubclass(src_dtype, core.Numeric):
        raise TypeError(f"src_ty must be a type of Numeric, but got {src_dtype}")

    if src_dtype.width == 4:
        recast_shape = core.recast_layout(8, 4, core.make_layout(input.shape)).shape
        i4_vec = builtin.unrealized_conversion_cast(
            [T.vector(input.type.shape[0], T.i(4))], [input.maybe_downcast()]
        )
        res_vect = vector.bitcast(T.vector(i4_vec.type.shape[0] // 2, T.i8()), i4_vec)
        return core.TensorSSA(res_vect, recast_shape, core.Int8)
    return input


@CuTeDSL.kernel
def _convert_kernel(
    gSrc: core.Tensor,
    gDst: core.Tensor,
    cSrc: core.Tensor,
    src_tv_layout: core.Layout,
    dst_tv_layout: core.Layout,
    src_shape: core.Shape,
    src_ty,
    dst_ty,
):
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())

    cta_coord = (None, bidx)
    # logical idx -> address
    ctaSrc = gSrc[cta_coord]  # (...,TileV,...)
    ctaDst = gDst[cta_coord]  # (...,TileV,...)
    ctaCSrc = cSrc[cta_coord]  # (...,TileV,...)
    # print(f"ctaSrc = {ctaSrc.type}")

    # compose with CTA TV layout
    # tid, vid -> address
    tidfrgSrc = core.composition(ctaSrc, src_tv_layout)  # (T,V)
    tidfrgDst = core.composition(ctaDst, dst_tv_layout)  # (T,V)
    tidfrgCSrc = core.composition(ctaCSrc, src_tv_layout)  # (T,V)
    # print(f"tidfrgSrc = {tidfrgSrc.type}")

    # slice for threads
    thr_coord = (tidx, None)
    thrSrc = tidfrgSrc[thr_coord]  # (V)
    thrDst = tidfrgDst[thr_coord]  # (V)
    thrCSrc = tidfrgCSrc[thr_coord]  # (V)
    # print(f"thrSrc = {thrSrc.type}")

    # predicate
    if core.elem_less(thrCSrc[0], src_shape):
        # allocate fragments for gmem->rmem
        frgSrc = core.make_fragment(
            core.get(src_tv_layout, mode=[1]), gSrc.element_type
        )  # (V)
        frgDst = core.make_fragment(
            core.get(dst_tv_layout, mode=[1]), gDst.element_type
        )  # (V)
        # print(f"frgSrc = {frgSrc.type}")

        # Move data to reg address space
        copy_atom_load = core.make_copy_atom(nvgpu.CopyUniversalOp(), gSrc.element_type)
        core.copy(copy_atom_load, thrSrc, frgSrc)

        vec_src = frgSrc.load()
        vec_src = _maybe_recast_to_f4(vec_src, src_ty)
        vec_dst = vec_src.to(dst_ty)
        vec_dst = _maybe_recast_from_f4(vec_dst, dst_ty)
        frgDst.store(vec_dst)

        # Copy the results back to c
        copy_atom_stg = core.make_copy_atom(nvgpu.CopyUniversalOp(), gDst.element_type)
        core.copy(copy_atom_stg, frgDst, thrDst)


@CuTeDSL.jit(preprocess=False)
def _convert(
    src: core.Tensor,
    dst: core.Tensor,
    leading_mode: Constexpr,
    elem_per_copy: Constexpr,
):

    # Step 1. figure proper tv_layout
    src_ty = src.element_type
    dst_ty = dst.element_type

    tv_layout = core.make_layout((128, elem_per_copy), stride=(elem_per_copy, 1))

    # Step 2. maybe recast from f4 tensor
    src, src_tv_layout = _maybe_recast_tensor_from_f4(src, tv_layout)
    dst, dst_tv_layout = _maybe_recast_tensor_from_f4(dst, tv_layout)
    src_shape = src.shape
    # predicate tensor
    idA = core.make_identity_tensor(src.shape)

    # Step 3. select a proper tiling pattern as (...,TileV, ...)
    src_cta_tiler = [
        1,
    ] * core.rank(src.layout)
    src_cta_tiler[leading_mode] = core.size(src_tv_layout)  # (...,TileV,...)
    dst_cta_tiler = [
        1,
    ] * core.rank(dst.layout)
    dst_cta_tiler[leading_mode] = core.size(dst_tv_layout)  # (...,TileV,...)

    # Step 4. partition input and output tensor by cta tiler.
    gS = core.zipped_divide(
        src, tuple(src_cta_tiler)
    )  # ((...,TileV,...),(...,RestV,...))
    cS = core.zipped_divide(
        idA, tuple(src_cta_tiler)
    )  # ((...,TileV,...),(...,RestV,...))
    gD = core.zipped_divide(
        dst, tuple(dst_cta_tiler)
    )  # ((...,TileV,...),(...,RestV,...))
    # print(f"{gS.type=}")

    _convert_kernel(
        gS,
        gD,
        cS,
        src_tv_layout,
        dst_tv_layout,
        src_shape,
        src_ty,
        dst_ty,
    ).launch(
        grid=[core.size(gS, mode=[1]), 1, 1],
        block=[core.size(src_tv_layout, mode=[0]), 1, 1],
    )


# Converts from src tensor to dst tensor, their logical shape are required to be the same.
# And when src or dst dtype is narrow precision(Float4E2M1FN/Float8E8M0FNU/Float8E4M3FN), the shape of
# their leading dimension should be 4(fp8)/8(fp4) element align. (nvgpu.cvt_fptrunc/cvt_fpext
# needs 32-bits aligned input/output)
def convert(src: core.Tensor, dst: core.Tensor):
    assert len(src.shape) == len(
        dst.shape
    ), "Shape of src and dst tensors should be the same rank."
    # find leading mode
    leading_mode = [
        idx
        for idx, (shape, stride) in enumerate(zip(src.shape, src.stride))
        if shape > 1 and stride == 1
    ]
    if len(leading_mode) != 1:
        raise ValueError(f"Leading mode should be unique, but got {leading_mode}")
    leading_mode = leading_mode[0]

    elem_per_copy = 2

    if src.element_type.width == 4 or dst.element_type.width == 4:
        elem_per_copy = 8
    elif src.element_type.width == 8 or dst.element_type.width == 8:
        elem_per_copy = 4
    assert (
        src.shape[leading_mode] % elem_per_copy == 0
        and dst.shape[leading_mode] % elem_per_copy == 0
    )
    _convert(src, dst, leading_mode, elem_per_copy)


#########################################
# Testing utilities
#########################################


def sample_pytest(rand_cfg=None):
    """
    Decorator to randomly sample pytest parametrized tests.
    rand_cfg: Tuple[int, float] - (random_seed, sample_ratio)
    Sampling is disabled when:
    - A specific test is selected (via -k or direct test path)
    - Not running under pytest
    """
    import functools
    import os
    import random
    import sys

    import pytest

    seed, sample_ratio = rand_cfg
    random.seed(seed)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if rand_cfg is not None and "PYTEST_CURRENT_TEST" in os.environ:
                # Check if test was explicitly selected like ::test_name[param1-param2-...]
                if "-k" in sys.argv or any(".py::" in arg for arg in sys.argv):
                    # Test was explicitly selected, don't skip
                    return func(*args, **kwargs)

                if random.uniform(0.0, 1.0) > sample_ratio:
                    pytest.skip(f"Randomly skipped (sampling ratio: {sample_ratio})")
            return func(*args, **kwargs)

        return wrapper

    return decorator


#########################################
# Benchmarking utilities
#########################################


class JitArguments:
    """
    A type to hold both args and kwargs for passing to a kernel while benchmarking.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _cuda_success(
    err: Union[tuple, cuda_runtime.cudaError_t, cuda_driver.CUresult], message: str
):
    """
    Helper function to check CUDA API errors.
    """
    if isinstance(err, tuple):
        _cuda_success(err[0], message)
    elif isinstance(err, cuda_runtime.cudaError_t):
        error_message = cuda_runtime.cudaGetErrorString(err)[1].decode("utf-8")
        if err != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{message} : {error_message}")
    elif isinstance(err, cuda_driver.CUresult):
        if err != cuda_driver.CUresult.CUDA_SUCCESS:
            error_message = cuda_driver.cuGetErrorString(err)[1].decode("utf-8")
            raise RuntimeError(f"{message} : {error_message}")
    else:
        raise TypeError(
            f"{err} is an unexpected type : it should be a cudaError_t or CUresult"
        )


def _does_kernel_use_stream(
    kernel: Callable, stream: cuda_driver.CUstream, *args, **kwargs
):
    """
    This function checks if the kernel uses the provided non-default stream.
    It does this by capturing the stream and then checking if any kernels were launched.
    :param kernel: The kernel to check
    :type kernel: Callable
    :param stream: The stream to check
    :type stream: cuda_driver.CUstream
    :return: True if the kernel uses the stream, False otherwise
    :rtype: bool
    """

    assert int(stream) != int(
        cuda_driver.CUstream_flags.CU_STREAM_DEFAULT
    ), "Stream must be a non-default stream"

    err = cuda_runtime.cudaStreamBeginCapture(
        stream, cuda_runtime.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
    )
    _cuda_success(err, "Error on stream capture")

    kernel(*args, **kwargs)

    err, graph = cuda_runtime.cudaStreamEndCapture(stream)
    _cuda_success(err, "Error on stream capture")

    # Get number of nodes in warmup graph to check it matches what is expected
    err, _, num_nodes = cuda_runtime.cudaGraphGetNodes(graph)
    _cuda_success(err, "Error on querying graph")
    return num_nodes > 0


def benchmark(
    callable: Callable,
    *,
    warmup_iterations: int = 10,
    iterations: int = 100,
    stream: Optional[cuda_driver.CUstream] = None,
    kernel_arguments: Optional[JitArguments] = None,
    workspace_generator: Optional[Callable[[], JitArguments]] = None,
    workspace_count: int = 1,
    use_cuda_graphs: bool = False,
) -> float:
    """Benchmarks a callable function with the specified parameters.

    For example,
    .. code-block:: python

        from cutlass.cute.testing import benchmark

        @cute.jit
        def user_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda_driver.CUstream):
            # contents of the function
            pass

        time_us = benchmark(user_function, kernel_arguments=JitArguments(a, b, c, stream)
                            warmup_iterations=10, iterations=100
                            stream=stream)

    To prevent skewing results by repeately accessing the L2 cache, use the workspace_count and workspace_generator
    parameters to cycle through a number of different workspaces.

    .. code-block:: python

        from cutlass.cute.testing import benchmark

        @cute.jit
        def user_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
            # contents of the function
            pass

        def workspace_generator():
            # create a, b, and c
            return JitArguments(a, b, c)

        time_us = benchmark(user_function,
                            workspace_generator=workspace_generator,
                            workspace_count=10,
                            warmup_iterations=10000,
                            iterations=1000)

    To benchmark you may always configure the function being profiled (callable), the warmup iterations, and
    the number of profiling iterations.

    Whenever the kernel being benchmarked runs in a non-default stream, the stream must be provided through the stream parameter.

    To use CUDA graphs, the callable must be a compiled @cute.jit annotated function.
    When using CUDA graphs, the kernel must be launched in a non-default stream.

    :param callable: The function to benchmark
    :type callable: Callable
    :param warmup_iterations: Number of warmup iterations, defaults to 10
    :type warmup_iterations: int, optional
    :param iterations: Number of benchmark iterations, defaults to 100
    :type iterations: int, optional
    :param stream: Stream kernel is launched in, defaults to CUDA stream default
    :type stream: CUstream, None
    :param kernel_arguments: Kernel arguments to launch callable with, defaults to None
    :type kernel_arguments: JitArguments, None
    :param workspace_generator: Function that returns kernel arguments, defaults to None
    :type workspace_generator: Callable
    :param workspace_count: Number of workspaces (arguments) to loop through, looping through enough workspaces will keep the L2 cache cold
    :type workspace_count: int, optional
    :param use_cuda_graphs: Whether to use cuda graphs, defaults to False
    :type use_cuda_graphs: bool, optional

    :return: The benchmark time in microseconds
    :rtype: float
    """

    if stream is None:
        stream = cuda_driver.CUstream(cuda_driver.CUstream_flags.CU_STREAM_DEFAULT)

    if workspace_count < 1:
        raise ValueError("workspace_count must be at least 1")

    time_us = float("nan")
    if workspace_generator == None:
        # If no workspace generator is provided, we need a single workspace
        if workspace_count != 1:
            raise ValueError("Need a single workspace if not providing a generator")

        # If no workspace generator is provided, we need a kernel_argument
        if kernel_arguments == None:
            raise ValueError(
                "Please pass a kernel argument if not providing a generator"
            )
        workspace_generator = lambda: kernel_arguments

    workspaces = [workspace_generator() for _ in range(workspace_count)]

    for workspace in workspaces:
        if type(workspace) != JitArguments:
            raise TypeError(
                "workspace_generator and/or kernel_arguments should use JitArguments type"
            )

    def _loop_and_call_kernel(iterations: int, workspace_index: int = 0):
        for _ in range(iterations):
            current_workspace = workspaces[workspace_index]
            callable(*current_workspace.args, **current_workspace.kwargs)
            workspace_index = (workspace_index + 1) % workspace_count
        return workspace_index

    # Create CUDA events for timing
    err, start_event = cuda_driver.cuEventCreate(
        cuda_driver.CUevent_flags.CU_EVENT_DEFAULT
    )
    _cuda_success(err, "Error on creating event")
    err, end_event = cuda_driver.cuEventCreate(
        cuda_driver.CUevent_flags.CU_EVENT_DEFAULT
    )
    _cuda_success(err, "Error on creating event")

    elapsed_time = float("nan")

    if use_cuda_graphs:
        # Check if the callable is a JitExecutor
        if not isinstance(callable, cutlass.base_dsl.jit_executor.JitExecutor):
            raise TypeError("Function must be precompiled to be used with CUDA Graphs")

        # Check if the stream is a non-default stream
        if int(stream) == int(cuda_driver.CUstream_flags.CU_STREAM_DEFAULT):
            raise ValueError(
                "Measuring with CUDA Graphs requires executing in a non-default stream"
            )

        workspace_index = 0

        # Capture warmup graph
        err = cuda_runtime.cudaStreamBeginCapture(
            stream, cuda_runtime.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
        )
        _cuda_success(err, "Error on stream capture")

        workspace_index = _loop_and_call_kernel(warmup_iterations)
        err, gwarm = cuda_runtime.cudaStreamEndCapture(stream)
        _cuda_success(err, "Error on stream capture")

        # Get number of nodes in warmup graph to check it matches what is expected
        err, _, num_nodes = cuda_runtime.cudaGraphGetNodes(gwarm)
        _cuda_success(err, "Error on querying graph")
        # Assertion is >= since we may launch multiple kernels in one host function
        if num_nodes < warmup_iterations:
            raise ValueError(
                f"CUDA stream passed to benchmark does not match the stream the kernel was launched in"
            )

        # Capture profiling graph
        err = cuda_runtime.cudaStreamBeginCapture(
            stream, cuda_runtime.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
        )
        _cuda_success(err, "Error on stream capture")
        _loop_and_call_kernel(iterations, workspace_index)
        err, gprofile = cuda_runtime.cudaStreamEndCapture(stream)
        _cuda_success(err, "Error on stream capture")

        # Instantiate graphs
        err, gwarm = cuda_runtime.cudaGraphInstantiate(gwarm, 0)
        _cuda_success(err, "Error on graph instantiation")
        err, gprofile = cuda_runtime.cudaGraphInstantiate(gprofile, 0)
        _cuda_success(err, "Error on graph instantiation")

        # Launch warmup graph
        err = cuda_runtime.cudaGraphLaunch(gwarm, stream)
        _cuda_success(err, "Error on graph launch")

        # Record start time
        err = cuda_driver.cuEventRecord(start_event, stream)
        _cuda_success(err, "Error on recording event")

        # Launch profiling graph
        err = cuda_runtime.cudaGraphLaunch(gprofile, stream)
        _cuda_success(err, "Error on graph launch")

        # Record end time
        err = cuda_driver.cuEventRecord(end_event, stream)
        _cuda_success(err, "Error on recording event")
        err = cuda_driver.cuEventSynchronize(end_event)
        _cuda_success(err, "Error on synchronizing event")

        # Get elapsed time
        err, elapsed_time = cuda_driver.cuEventElapsedTime(start_event, end_event)
        _cuda_success(err, "Error on querying event")

        # Destroy graphs
        err = cuda_runtime.cudaGraphExecDestroy(gwarm)
        _cuda_success(err, "Error on destroying graph")
        err = cuda_runtime.cudaGraphExecDestroy(gprofile)
        _cuda_success(err, "Error on destroying graph")

    else:

        if int(stream) != int(
            cuda_driver.CUstream_flags.CU_STREAM_DEFAULT
        ) and not _does_kernel_use_stream(
            callable, stream, *workspaces[0].args, **workspaces[0].kwargs
        ):
            raise ValueError(
                "CUDA stream passed to benchmark does not match the stream the kernel was launched in"
            )

        # Not using graphs
        # Warmup
        workspace_index = _loop_and_call_kernel(warmup_iterations)
        # Record start event
        err = cuda_driver.cuEventRecord(start_event, stream)
        _cuda_success(err, "Error on recording event")
        _loop_and_call_kernel(iterations, workspace_index)
        # Record end event
        err = cuda_driver.cuEventRecord(end_event, stream)
        _cuda_success(err, "Error on recording event")
        # Synchronize end event
        err = cuda_driver.cuEventSynchronize(end_event)
        _cuda_success(err, "Error on synchronizing event")
        err, elapsed_time = cuda_driver.cuEventElapsedTime(start_event, end_event)
        _cuda_success(err, "Error on querying event")

    # Destroy events
    err = cuda_driver.cuEventDestroy(start_event)
    _cuda_success(err, "Error on destroying event")
    err = cuda_driver.cuEventDestroy(end_event)
    _cuda_success(err, "Error on destroying event")

    return elapsed_time / iterations * 1e3


def get_workspace_count(
    one_workspace_bytes: int, warmup_iterations: int, iterations: int
) -> int:
    """Calculate the number of workspaces needed to fill L2 cache.

    :param one_workspace_bytes: Size of one workspace in bytes
    :type one_workspace_bytes: int
    :param warmup_iterations: Number of warmup iterations
    :type warmup_iterations: int
    :param iterations: Number of iterations
    :type iterations: int
    :return: Number of workspaces needed
    :rtype: int
    """
    num_l2_cache_bytes = cutlass.utils.HardwareInfo().get_l2_cache_size_in_bytes()
    return max(
        1,
        min(
            warmup_iterations + iterations,  # Don't create more workspaces than needed
            (num_l2_cache_bytes + one_workspace_bytes - 1)
            // one_workspace_bytes,  # Ceiling division
        ),
    )

