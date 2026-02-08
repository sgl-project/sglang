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

"""
This module provides jit executor related classes
"""
import ctypes
import inspect
import io
from typing import get_origin

import numpy as np

# MLIR modules imports
from .._mlir import ir

# Local modules imports
from . import typing as t
from .common import DSLRuntimeError
from .runtime import cuda as cuda_helpers
from .runtime.jit_arg_adapters import JitArgAdapterRegistry, is_arg_spec_constexpr
from .typing import get_c_pointers
from .utils.logger import log
from .utils.timer import timer


class CudaSingleModule:
    def __init__(self, cuda_module, kernel_ptr):
        self.cuda_module = cuda_module
        self.kernel_ptr = kernel_ptr


class CudaModules:
    def __init__(self, modules, args):
        # list of CudaSingleModule
        self.modules = modules
        # extra kernel ptr arguments for launch
        self.args = args


class JitExecutor:
    def __init__(
        self,
        dsl,
        engine,
        capi_func,
        ir_module,
        args_spec,
        function_name,
        cuda_modules: CudaModules = None,
        jit_time_profiling=False,
    ):
        self.dsl = dsl
        self.engine = engine
        self.capi_func = capi_func
        self.ir_module = ir_module
        self.args_spec = args_spec
        self.function_name = function_name
        if args_spec is not None:
            self.original_args_spec = args_spec
            self.args_spec = self.filter_runtime_arg_spec(args_spec)
        # cuda kernels
        self.cuda_modules = cuda_modules
        self.jit_time_profiling = jit_time_profiling

    def filter_runtime_arg_spec(self, arg_spec: inspect.FullArgSpec):
        runtime_args = []
        runtime_annotations = {}
        runtime_defaults = []

        # Calculate the offset where defaults start in the original args
        if arg_spec.defaults:
            defaults_start_idx = len(arg_spec.args) - len(arg_spec.defaults)
        else:
            defaults_start_idx = len(arg_spec.args)

        # Filter arguments and maintain their properties
        for i, arg_name in enumerate(arg_spec.args):
            arg_type = arg_spec.annotations.get(arg_name, None)

            # Skip compile-time arguments
            if is_arg_spec_constexpr(arg_type, arg_name, i, self.function_name):
                continue

            # Keep runtime arguments
            runtime_args.append(arg_name)
            if arg_name in arg_spec.annotations:
                runtime_annotations[arg_name] = arg_type

            # Keep corresponding default if it exists
            if i >= defaults_start_idx:
                default_idx = i - defaults_start_idx
                runtime_defaults.append(arg_spec.defaults[default_idx])

        # Filter kwonlyargs and their defaults
        runtime_kwonlyargs = []
        runtime_kwonlydefaults = {}

        if arg_spec.kwonlyargs:
            for kwarg in arg_spec.kwonlyargs:
                arg_type = arg_spec.annotations.get(kwarg, None)

                # Apply same filtering logic
                if is_arg_spec_constexpr(arg_type, kwarg, i, self.function_name):
                    continue

                runtime_kwonlyargs.append(kwarg)
                if kwarg in arg_spec.annotations:
                    runtime_annotations[kwarg] = arg_type
                if arg_spec.kwonlydefaults and kwarg in arg_spec.kwonlydefaults:
                    runtime_kwonlydefaults[kwarg] = arg_spec.kwonlydefaults[kwarg]

        # Convert runtime_defaults to tuple if not empty (as expected by FullArgSpec)
        runtime_defaults = tuple(runtime_defaults) if runtime_defaults else None

        return inspect.FullArgSpec(
            args=runtime_args,
            varargs=arg_spec.varargs,  # Keep original varargs
            varkw=arg_spec.varkw,  # Keep original varkw
            defaults=runtime_defaults,
            kwonlyargs=runtime_kwonlyargs,
            kwonlydefaults=runtime_kwonlydefaults if runtime_kwonlydefaults else None,
            annotations=runtime_annotations,
        )

    def __del__(self):
        if self.cuda_modules:
            cuda_modules = [module.cuda_module for module in self.cuda_modules.modules]
            for module in set(cuda_modules):
                cuda_helpers.unload_cubin_module(module)

    def get_constexpr_args(self) -> list[dict[str, int | str]]:
        """
        This function returns the constexpr args that have been pruned from the original function signature.
        The return type is a list of dicts, each dict contains the argument index (argument_index) and argument name (argument_name).

        :return: list of dicts, each dict contains the argument index (argument_index) and argument name (argument_name).
        :rtype: list[dict[str, int | str]]
        """
        if self.original_args_spec is None:
            return list()
        constexpr_args = list()
        for i, arg_name in enumerate(self.original_args_spec.args):
            if arg_name not in self.args_spec.args:
                constexpr_args.append({"argument_index": i, "argument_name": arg_name})

        if self.original_args_spec.kwonlyargs:
            for kwarg in self.original_args_spec.kwonlyargs:
                if kwarg not in self.args_spec.kwonlyargs:
                    constexpr_args.append(
                        {"argument_index": None, "argument_name": kwarg}
                    )
        return constexpr_args

    def generate_execution_args(self, args, kwargs, args_spec: inspect.FullArgSpec):
        """
        This function is the prune version of `generate_mlir_function_types` which only generates execution args
        to get rid of mlir context.
        """

        # Process positional arguments with defaults
        rectified_args = list(args)
        if args_spec.defaults and len(args) < len(args_spec.args):
            rectified_args.extend(args_spec.defaults[len(args) - len(args_spec.args) :])
        for k, v in kwargs.items():
            if k in args_spec.args:
                idx = args_spec.args.index(k)
                if idx < len(rectified_args):
                    rectified_args[idx] = v
                else:
                    rectified_args.append(v)

        # Process keyword arguments
        rectified_kwargs = {k: v for k, v in kwargs.items() if k not in args_spec.args}
        if args_spec.kwonlydefaults and len(rectified_kwargs) < len(
            args_spec.kwonlyargs
        ):
            rectified_kwargs.update(args_spec.kwonlydefaults)

        # args/kwargs must match arg_specs
        if len(rectified_args) != len(args_spec.args) or len(rectified_kwargs) != len(
            args_spec.kwonlyargs
        ):
            raise DSLRuntimeError(
                "input args/kwargs length does not match runtime function signature!",
                context={
                    "input args length": len(rectified_args),
                    "input kwargs length": len(rectified_kwargs),
                    "function signature args length": len(args_spec.args),
                    "function signature kwonlyargs length": len(args_spec.kwonlyargs),
                },
            )

        exe_args = []
        adapted_args = []
        input_args = rectified_args + list(rectified_kwargs.values())
        input_arg_names = args_spec.args + args_spec.kwonlyargs
        for arg, arg_name in zip(input_args, input_arg_names):
            # short-cut for args already converted
            if hasattr(arg, "__c_pointers__"):
                exe_args.extend(arg.__c_pointers__())
                continue

            arg_type = args_spec.annotations.get(arg_name, None)

            # Implicit cast to NumericMeta
            if isinstance(arg_type, t.NumericMeta):
                arg = t.cast(arg, arg_type)
            else:
                # If not any known type, try registered adapter to do the conversion
                adapter = JitArgAdapterRegistry.get_registered_adapter(type(arg))
                if adapter:
                    arg = adapter(arg)
                    adapted_args.append(arg)

            exe_args.extend(get_c_pointers(arg))

        return exe_args, adapted_args

    def __call__(self, *args, **kwargs):
        exe_args, adapted_args = self.generate_execution_args(
            args, kwargs, self.args_spec
        )

        self.run_compiled_program(exe_args)

    # Assume each execution args has type `c_void_p` to reduce the overhead of `ctypes.cast`.
    def get_invoke_packed_args(self, exe_args):
        if self.cuda_modules:
            exe_args += self.cuda_modules.args
        packed_args = (ctypes.c_void_p * len(exe_args))()
        for argNum in range(len(exe_args)):
            packed_args[argNum] = exe_args[argNum]
        return packed_args

    def run_compiled_program(self, exe_args):
        if self.jit_time_profiling:
            profiler = timer(enable=True)
            try:
                packed_args = profiler(self.get_invoke_packed_args)(exe_args)
                profiler(self.capi_func)(packed_args)
            except Exception as e:
                raise DSLRuntimeError(f"💥💥💥 Runtime Crash 💥💥💥", cause=e)
        else:
            try:
                packed_args = self.get_invoke_packed_args(exe_args)
                self.capi_func(packed_args)
            except Exception as e:
                raise DSLRuntimeError(f"💥💥💥 Runtime Crash 💥💥💥", cause=e)

    def update_jit_cuda_modules(self, kernel_symbols):
        # preload cuda module from compiled cubin in ir and store to jit_executor.kernels.
        if len(kernel_symbols) > 0:
            extra_args = []
            module = self.ir_module
            cuda_kernel_cache = dict()
            cuda_driver_version = cuda_helpers.get_driver_version()
            for sym in kernel_symbols:
                if sym not in cuda_kernel_cache:
                    log().debug(f"Loading CUDA module for symbol: {sym}")

                    # load cuda module/get function pointer from module and cache
                    def walk_callback(sym, func_sym, cubin_data):
                        cubin_module = cuda_helpers.load_cubin_module_data(cubin_data)
                        kernel_ptr = cuda_helpers.get_kernel_function(
                            cubin_module, func_sym
                        )
                        # Enable non-portable cluster size for CUDA version 11.8 or higher.
                        if cuda_driver_version >= 11080:
                            cuda_helpers.set_kernel_attribute(
                                kernel_ptr,
                                cuda_helpers.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                                1,
                            )
                        cuda_kernel_cache[sym] = CudaSingleModule(
                            cubin_module, kernel_ptr
                        )

                    self.walk_module_and_get_cubin_data(module, sym, walk_callback)
                else:
                    log().debug(f"Symbol {sym} already in cache")
                # check if kernel is empty.
                if sym in cuda_kernel_cache:
                    extra_args.append(
                        ctypes.c_void_p(cuda_kernel_cache[sym].kernel_ptr.getPtr())
                    )
            # store to the jit result if jit result is cached.
            self.cuda_modules = CudaModules(cuda_kernel_cache.values(), extra_args)

        return self

    def _get_escaped_cubin_bytes(self, cubin_data):
        """This function escapes cubin data from mlir raw bytecode to executable binary bytes"""

        def ishex(inp):
            return (
                inp in range(0x30, 0x3A)
                or inp in range(0x61, 0x67)
                or inp in range(0x41, 0x47)
            )

        converted = bytearray()
        idx = 0
        while idx < len(cubin_data):
            # escape the original bytes
            if cubin_data[idx] == 0x5C:
                # if data of idx is b'\\'
                if ishex(cubin_data[idx + 1]) and ishex(cubin_data[idx + 2]):
                    converted += bytearray.fromhex(
                        cubin_data[idx + 1 : idx + 3].decode()
                    )
                    idx += 3
                elif cubin_data[idx + 1] == 0x5C:
                    converted.append(cubin_data[idx])
                    idx += 2
            else:
                # no escape, directly write
                converted.append(cubin_data[idx])
                idx += 1
        return bytes(converted)

    def walk_module_and_get_cubin_data(self, module, sym, callback):
        """This function is used to walk gpu binary op, extract the cubin inside, and process cubin data with callback."""

        def walk_gpu_binary_op(op):
            if op.name != "gpu.binary":
                return ir.WalkResult.ADVANCE
            s = io.BytesIO()
            op.write_bytecode(s)
            cubin_data = s.getvalue()
            if sym.encode() not in cubin_data:
                return ir.WalkResult.ADVANCE

            if (
                "kernels" != op.opview.sym_name.value
                and sym != op.opview.sym_name.value
            ):
                return ir.WalkResult.ADVANCE
            # function symbol of kernel(gpu.launch_func) is equal to sym name in mlir
            func_sym = sym
            if sym == op.opview.sym_name.value and not sym.endswith("_kernel"):
                func_sym = sym.rsplit("_", 1)[0]

            cubin_data = cubin_data.split(b'bin = "')[1].split(b'">')[0]
            cubin_data = self._get_escaped_cubin_bytes(cubin_data)
            callback(sym, func_sym, cubin_data)
            return ir.WalkResult.ADVANCE

        module.operation.walk(walk_gpu_binary_op)
