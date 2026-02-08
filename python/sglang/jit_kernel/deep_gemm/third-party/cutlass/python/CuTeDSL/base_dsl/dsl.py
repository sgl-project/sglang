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
This module provides a main DSL class for any Dialect.
The DSL should be inherited as a new class, and its initialization requires dialects.
It handles most of the mechanics for the DSL in an agnostic way,
for example, it can handle various dialect-specific tasks.
"""


# Standard library imports
from dataclasses import dataclass, field
import atexit
import os
import io
import sys
import errno
import ctypes
import re
import inspect
import argparse
import hashlib
from functools import lru_cache, wraps
from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Any, Union, Tuple, get_origin, get_args, List
from types import FunctionType, SimpleNamespace
import warnings

from . import typing as t
from .env_manager import EnvironmentVarManager
from .compiler import CompileOptions
from .ast_helpers import DSLOptimizationWarning

# =============================================================================
# CUDA Python
# =============================================================================

from ..base_dsl._mlir_helpers.arith import const

# =============================================================================
# Local module imports
# =============================================================================

from .cache_helpers import *
from .jit_executor import JitExecutor
from .utils.timer import timer
from .utils.logger import setup_log, log
from .utils.stacktrace import filter_exception, walk_to_top_module, filter_stackframe
from .runtime.jit_arg_adapters import is_argument_constexpr, JitArgAdapterRegistry

from .ast_preprocessor import DSLPreprocessor
from .common import *
from .typing import (
    get_c_pointers,
    get_mlir_types,
)

# =============================================================================
# MLIR modules
# =============================================================================

from .._mlir import ir
from .._mlir import runtime as rt
from .._mlir.extras import types as T
from .._mlir.dialects import arith, math, func

# =============================================================================
# Global Variables
# =============================================================================

MLIR_DYNAMIC = -9223372036854775808

# =============================================================================
# Codegen Utils
# =============================================================================


def _numpy_type_to_mlir_type(dtype):
    if dtype == np.float64:
        return T.f64()
    if dtype == np.float16:
        return T.f16()
    if dtype == np.float32:
        return T.f32()
    if dtype == np.int64:
        return T.i64()
    if dtype == np.int32:
        return T.i32()
    if dtype == np.int16:
        return T.i16()
    if dtype == np.int8:
        return T.i8()
    if dtype == np.uint64:
        return T.ui64()
    if dtype == np.uint32:
        return T.ui32()
    if dtype == np.uint16:
        return T.ui16()
    if dtype == np.uint8:
        return T.ui8()
    if dtype == np.bool_:
        return T.bool()
    if dtype == f8E5M2:
        return T.f8E5M2()
    if dtype == f8E4M3FN:
        return T.f8E4M3FN()
    if dtype == f8E8M0FNU:
        return T.f8E8M0FNU()
    if dtype == f6E3M2FN:
        return T.f6E3M2FN()
    if dtype == f6E2M3FN:
        return T.f6E2M3FN()
    if dtype == f4E2M1FN:
        return T.f4E2M1FN()
    assert False, f"Unknown type {type}"


def _mlir_type_to_numpy_type(type):
    if type == T.f64():
        return np.float64
    if type == T.f16():
        return np.float16
    if type == T.f32():
        return np.float32
    if type == T.i64():
        return np.int64
    if type == T.i32():
        return np.int32
    if type == T.i16():
        return np.int16
    if type == T.i8():
        return np.int8
    if type == T.ui64():
        return np.uint64
    if type == T.ui32():
        return np.uint32
    if type == T.ui16():
        return np.uint16
    if type == T.ui8():
        return np.uint8
    if type == T.bool():
        return np.bool_
    assert False, f"Unknown type {type}"


# =============================================================================
# Main DSL Class
# =============================================================================


def is_dynamic_expression(value):
    """
    Given the `value`, check if itself is an IR value or recursively go through it to check if it contains IR value
    """
    if isinstance(value, (tuple, list)):
        for x in value:
            if is_dynamic_expression(x):
                return True
    elif isinstance(value, (ir.Value, ir.BlockArgumentList)) or hasattr(
        value, "__extract_mlir_values__"
    ):
        return True
    return False


def extract_mlir_values(obj):
    """
    Given the `obj`, recursively go through it to extract all contained IR values as list of MLIR values
    """
    res = []
    if hasattr(obj, "__extract_mlir_values__"):
        res = obj.__extract_mlir_values__()
    elif isinstance(obj, (tuple, list)):
        res = sum((extract_mlir_values(x) for x in obj), [])
    elif isinstance(obj, SimpleNamespace):
        res = []
        for k, v in obj.__dict__.items():
            res.extend(extract_mlir_values(v))
    # Can't call is_dynamic_expression as _is_dynamic_expression depends on extract_mlir_values
    elif isinstance(obj, set):
        raise DSLRuntimeError(
            "Sets are not supported in extract_mlir_values to ensure order preservation",
            context="The DSL attempted to generate JIT function argument(s) for an argument of type set but failed.",
            suggestion="Consider using a list or tuple instead",
        )
    elif isinstance(obj, ir.Value):
        res = [obj]
    elif isinstance(obj, ir.BlockArgumentList):
        res = list(obj)  # type: ignore

    return res


def new_from_mlir_values(obj, values):
    """
    Create a new python object by populating containing MLIR values with list of new values
    """
    if hasattr(obj, "__new_from_mlir_values__"):
        return obj.__new_from_mlir_values__(values)
    elif isinstance(obj, (tuple, list)):
        res = []
        for x in obj:
            n_items = len(get_mlir_types(x))
            res.append(new_from_mlir_values(x, values[:n_items]))
            values = values[n_items:]
        obj_ty = type(obj)
        return obj_ty(res)
    elif isinstance(obj, SimpleNamespace):
        res = SimpleNamespace()
        for k, v in obj.__dict__.items():
            n_items = len(get_mlir_types(v))
            res.__dict__[k] = new_from_mlir_values(v, values[:n_items])
            values = values[n_items:]
        return res
    elif isinstance(obj, set):
        raise DSLRuntimeError(
            "Sets are not supported in new_from_mlir_values to ensure order preservation",
            context="The DSL attempted to generate JIT function argument(s) for an argument of type set but failed.",
            suggestion="Consider using a list or tuple instead",
        )
    elif is_dynamic_expression(obj):

        if len(values) == 0:
            return obj

        assert len(values) == 1
        return values[0]
    else:
        assert len(values) == 0, f"{obj} expects 0 values, but got {values}"
        return obj


class DSLCallable:
    """
    Wrapper class for a callable object used within the DSL.

    DSLCallable is designed to wrap a function and provide additional
    introspection utilities such as retrieving the argument specification
    and signature. It ensures that the wrapped function can only be called
    once, after which the reference to the function is cleared to prevent
    further invocations. This is useful in scenarios where a function should
    only be executed a single time within the DSL's execution model.

    Attributes:
        func (callable): The function to be wrapped and managed.

    Methods:
        __call__(*args, **kwargs): Calls the wrapped function and clears it.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        ret = self.__func__(*args, **kwargs)
        self.func = None
        return ret

    @property
    def __func__(self):
        assert self.func is not None, "DSLCallable is already called"
        return self.func

    @property
    def __signature__(self):
        return inspect.signature(self.__func__)

    @property
    def __name__(self):
        return self.__func__.__name__


class BaseDSL:
    gpu_module = None

    def __init__(
        self,
        *,
        name: str,
        dsl_package_name: List[str],
        compiler_provider: Any,
        pass_sm_arch_name: str,
        device_compilation_only=False,
        preprocess=False,
    ):
        """
        Constructor for initializing the class with required providers and environment settings.

        Parameters:
        - name (str): Name of DSL, used for environment variables and logging.
        - package_name (str): Name of the package, used for the preprocessor.
        - compiler_provider (MLIR dialect): Provider for compiler.
        - pass_sm_arch_name (str): The keyword name of the SM.
        - device_compilation_only (bool) : Only device code, and call it via cuda driver
        - preprocess (bool): Enable AST transformation.

        This constructs a DSL instance and sets up environment management,
        warning configurations, and logging functionalities. It reads
        environment variables using `EnvironmentVarManager` and configures
        a logger with settings from the environment. If environment warnings
        are detected, they are escalated to errors to ensure strict handling.
        """
        # Enforcing initialization of instance variables
        if not all([name, compiler_provider, pass_sm_arch_name]):
            raise DSLRuntimeError(
                "All required parameters must be provided and non-empty"
            )

        self.name = name
        self.compiler_provider = compiler_provider
        self.pass_sm_arch_name = pass_sm_arch_name
        self.frame = None
        self.no_cache = False
        self.device_compilation_only = device_compilation_only
        self.num_kernels = 0
        # Read environment variables
        self.envar = EnvironmentVarManager(self.name)
        self.enable_preprocessor = preprocess
        # This cache uses hash of original ir and env as key, allows dump/load to/from file. Enabled by default
        self.jit_cache = (
            dict()
            if self.envar.disable_file_caching
            else load_cache_from_path(self.name, self.envar.file_caching_capacity)
        )
        self.host_jit_decorator_name = f"@{BaseDSL.jit.__name__}"
        self.device_jit_decorator_name = f"@{BaseDSL.kernel.__name__}"

        # set warning
        if not self.envar.enable_optimization_warnings:
            # By default, optimization warnings are disabled
            warnings.filterwarnings("ignore", category=DSLOptimizationWarning)
        if self.envar.warnings_as_errors:
            warnings.filterwarnings("error")
        if self.envar.warnings_ignore:
            warnings.filterwarnings("ignore")

        # Initialize logger
        if self.envar.log_to_console == False and self.envar.jitTimeProfiling:
            self.envar.log_to_console = True
            self.envar.log_level = 20  # info level
        setup_log(
            self.name,
            self.envar.log_to_console,
            self.envar.log_to_file,
            f"{self.name}.log",
            self.envar.log_level,
        )

        # kernel symbols are temporary symbol string variables, their values are valid until the compilation is done.
        self.kernel_symbols = []
        # used to generate unique name for gpu.launch
        self.launch_inner_count = 0
        # initialize default compile options
        self.compile_options = CompileOptions()

        if preprocess:
            self.preprocessor = DSLPreprocessor(dsl_package_name)
        log().info(f"Initializing {name} DSL")
        log().debug(f"Logger initialized for {self.name}")

        # Hook excepthook
        if self.envar.filterStacktrace:
            origin_excepthook = sys.excepthook
            module_dir = walk_to_top_module(os.path.dirname(os.path.abspath(__file__)))

            def excepthook(excep_type, value, traceback):
                filter_exception(value, module_dir)
                if hasattr(value, "__traceback__"):
                    origin_excepthook(excep_type, value, value.__traceback__)
                else:
                    origin_excepthook(
                        excep_type, value, filter_stackframe(traceback, module_dir)
                    )

            sys.excepthook = excepthook

            # Restore original excepthook
            def restore_excepthook(hook):
                sys.excepthook = hook

            atexit.register(restore_excepthook, origin_excepthook)

    def dump_cache(self):
        if not self.envar.disable_file_caching:
            dump_cache_to_path(
                self.name, self.jit_cache, self.envar.file_caching_capacity
            )

    @lru_cache(maxsize=1)
    def print_warning_once(self, message):
        log().warning(f"Warning: {message}")
        warnings.warn(message, UserWarning)

    def print_warning(self, message):
        log().warning(f"Warning: {message}")
        warnings.warn(message, UserWarning)

    @classmethod
    @lru_cache(maxsize=1)
    def _get_dsl(cls):
        # Instantiate the DSL Class once
        main_dsl = cls()
        if not main_dsl.no_cache:
            # register atexit callback
            atexit.register(main_dsl.dump_cache)
        return main_dsl

    @staticmethod
    def _can_preprocess(**dkwargs):
        """
        Check if AST transformation is enabled or not for `jit` and `kernel` decorators.
        """
        return dkwargs.pop("preprocess", True)

    @staticmethod
    def _get_original_function(fcn_ptr, name):
        """
        Get the original function from the decorated function
        """
        while fcn_ptr.__name__ != name:
            # If the function is wrapped with functools, get from __wrapped__
            if hasattr(fcn_ptr, "__wrapped__"):
                fcn_ptr = fcn_ptr.__wrapped__
            # If the function is wrapped manually, it's the first in clousure
            elif callable(fcn_ptr.__closure__[0].cell_contents):
                fcn_ptr = fcn_ptr.__closure__[0].cell_contents
            else:
                raise DSLRuntimeError(
                    f"Cannot find the original function {name} in the closure chain"
                )
        return fcn_ptr

    @staticmethod
    def _preprocess_and_execute(func):
        """
        Run ast transformation and return the materialized function pointer
        """
        if hasattr(func, "_transformed_ast"):
            # If the function ptr is already materialized, use the existing one
            func._dsl_object.frame = func._decorator_frame
            if func._transformed_ast is None:
                func._transformed_ast = func._dsl_object.run_preprocessor(func)
                if func._transformed_ast is None:
                    del func._transformed_ast
                    func._dsl_object.frame = None
                    return func

            fcn_ptr = func._dsl_object.get_function_ptr(func)
            # If the function is decorated, de-decorate it
            fcn_ptr = BaseDSL._get_original_function(fcn_ptr, func.__name__)
            func._dsl_object.frame = None
            return DSLCallable(fcn_ptr)
        return func

    def jit_runner(self, executor, frame, *dargs, **dkwargs):
        """
        Decorator to mark a function for JIT compilation.
        """
        log().info("jit_runner")

        def jit_runner_decorator(func):
            func._dsl_object = self
            # Run preprocessor that alters AST
            if self.enable_preprocessor and BaseDSL._can_preprocess(**dkwargs):
                # For an annotated function, add some DSL attributes
                # When materializing the AST, we need decorator's frame
                func._decorator_frame = frame
                # No transformed ast at this point
                func._transformed_ast = None

            @wraps(func)
            def jit_wrapper(*args, **kwargs):
                func_ptr = BaseDSL._preprocess_and_execute(func)
                return executor(func_ptr, *args, **kwargs)

            return jit_wrapper

        if len(dargs) == 1 and callable(dargs[0]):
            return jit_runner_decorator(dargs[0])
        else:
            return jit_runner_decorator

    @classmethod
    def jit(cls, *dargs, **dkwargs):
        """
        Decorator to mark a function for JIT compilation for Host code.
        """
        frame = inspect.currentframe().f_back
        # Instantiate the DSL Class
        main_dsl = cls._get_dsl()
        return main_dsl.jit_runner(main_dsl._func, frame, *dargs, **dkwargs)

    @classmethod
    def kernel(cls, *dargs, **dkwargs):
        """
        Decorator to mark a function for JIT compilation for GPU.
        """
        frame = inspect.currentframe().f_back
        # Instantiate the DSL Class
        main_dsl = cls._get_dsl()
        return main_dsl.jit_runner(main_dsl._kernel_helper, frame, *dargs, **dkwargs)

    @abstractmethod
    def _kernel_helper(self, func, *args, **kwargs):
        """
        Helper function to handle kernel generation logic
        """
        pass

    @abstractmethod
    def _build_gpu_module(self, attrs):
        """
        Build the module op that contains the kernels.
        """
        pass

    @abstractmethod
    def _get_pipeline(self, pipeline):
        """
        Get the pipeline from the other configuration options.
        """
        if pipeline != None:
            return pipeline
        return None

    @staticmethod
    def log_additions(func_type, operands=None, types=None, arg_attrs=None):
        if operands is not None and operands != []:
            log().debug(
                f"Added {func_type} operands: [%s]", ", ".join(map(str, operands))
            )
        if types is not None:
            log().debug(
                f"Added {func_type} arg_types: [%s]", ", ".join(map(str, types))
            )
        if arg_attrs is not None:
            log().debug(
                f"Added {func_type} arg_attrs: [%s]", ", ".join(map(str, arg_attrs))
            )

    def mangle_name(self, function_name, args, args_spec: inspect.FullArgSpec):
        """Does simple name mangling"""

        for spec_arg, arg in zip(args_spec.args, args):
            spec_ty = args_spec.annotations.get(spec_arg, None)
            if spec_ty != None:
                if issubclass(type(spec_ty), (t.IRValue, t.IRVariadic)):
                    continue
                if isinstance(spec_ty, (ir.Type, ir.Value)):
                    continue
            if isinstance(arg, (ir.Type, ir.Value, ir.OpResult)):
                continue
            if isinstance(type(arg), (ir.Type, ir.Value, ir.OpResult)):
                continue
            if self._is_tensor_descriptor(arg):
                continue
            if inspect.isclass(spec_ty):
                class_name = str(arg).replace("class", "")
                class_name = class_name.replace(" ", "")
                function_name = f"{function_name}_{class_name}"
            elif isinstance(arg, (list, tuple)):
                function_name = f"{function_name}_{'_'.join(map(str, arg))}"
            else:
                function_name = f"{function_name}_{arg}"
        # we would need a dedicated MR to follow up
        unwanted_chars = r"'-![]#,.<>()\":{}=%?@;"
        translation_table = str.maketrans("", "", unwanted_chars)
        function_name = function_name.translate(translation_table)
        # identify address and drop
        function_name = re.sub(r"0x[a-f0-9]{8,16}", "", function_name)
        function_name = re.sub(r"\s+", " ", function_name)
        function_name = function_name.replace(" ", "_")
        function_name = function_name.replace("\n", "_")
        # max fname is 256 character, leave space
        function_name = function_name[:180]
        log().info(f"Final mangled function name: {function_name}")
        return function_name

    def _generate_execution_arguments_for_known_types(
        self, arg, arg_spec, arg_name, i, fop_args, iv_block_args
    ):
        """
        Generate MLIR arguments for known types.

        Sub-DSLs can override this method to handle types that are not
        natively supported by the Base DSL.
        """
        ir_arg = []
        if is_argument_constexpr(arg, arg_spec, arg_name, i, func):
            ir_arg.append(arg)

        return ir_arg, iv_block_args

    def generate_execution_arguments(
        self,
        args,
        kwargs,
        fop,
        args_spec: inspect.FullArgSpec,
    ):
        """Create list of arguments that will be passed to MLIR's func.func op"""

        def gen_exec_args(input_args, arg_names, annotations, fop_args):
            assert len(input_args) == len(arg_names)

            ir_args = []
            iv_block_args = 0
            for i, arg in enumerate(input_args):
                arg_name = arg_names[i]
                arg_spec = annotations.get(arg_name, None)
                log().debug("Processing [%d] Argument [%s : %s]", i, arg_name, arg_spec)

                # Implicit cast to NumericMeta
                if isinstance(arg_spec, t.NumericMeta) and not isinstance(
                    arg, arg_spec
                ):
                    arg = t.cast(arg, arg_spec)

                ir_arg, iv_block_args = (
                    self._generate_execution_arguments_for_known_types(
                        arg, arg_spec, arg_name, i, fop_args, iv_block_args
                    )
                )

                if not ir_arg:
                    # If it's not a known type, try JIT argument adapter
                    # to convert the argument if possible
                    adapter = JitArgAdapterRegistry.get_registered_adapter(type(arg))
                    arg = adapter(arg) if adapter else arg

                    n_args = len(get_mlir_types(arg))
                    blk_args = fop_args[iv_block_args : iv_block_args + n_args]
                    ir_arg.append(new_from_mlir_values(arg, blk_args))
                    iv_block_args += n_args

                self.log_additions(ir_arg)
                ir_args.extend(ir_arg)

            return ir_args, iv_block_args

        fop_args = list(fop.regions[0].blocks[0].arguments)
        ir_args, iv_block_args = gen_exec_args(
            args, args_spec.args, args_spec.annotations, fop_args
        )
        ir_kwargs, _ = gen_exec_args(
            [kwargs[arg] for arg in args_spec.kwonlyargs],
            args_spec.kwonlyargs,
            args_spec.annotations,
            fop_args[iv_block_args:],
        )
        ir_kwargs = {k: v for k, v in zip(args_spec.kwonlyargs, ir_kwargs)}

        log().debug("execution args: %s", ", ".join(map(str, ir_args)))
        log().debug("execution kwargs: %s", ", ".join(map(str, ir_kwargs)))
        return ir_args, ir_kwargs

    @abstractmethod
    def _generate_mlir_type_for_tensor_descriptor(self, tensor):
        """
        Generate MLIR type for the tensor descriptor.
        """
        pass

    @abstractmethod
    def _generate_executable_arg_for_tensor_descriptor(
        self, mlir_value=None, ptr_tensor_ty=None, tensor=None
    ):
        """
        Generates executable value for the given tensor descriptor.
        """
        pass

    def _get_globals(self):
        """
        Combines global and local variables from the current context and the
        caller's frame comes. This includes the current module's globals, the
        global variables from the caller's frame, and the local variables from
        the caller's frame.

        "self.frame" is used to fetch the caller's frame.

        AST preprocessor generates a new python code, so the resulting globals
        dictionary is used to execute the python code.
        """
        all_globals = {}
        if self.frame:
            all_globals.update(self.frame.f_globals)
            all_globals.update(self.frame.f_locals)
        return all_globals

    @abstractmethod
    def _is_tensor_descriptor(self, maybe_tensor_descriptor) -> bool:
        pass

    @abstractmethod
    def _handle_tensor_descriptor(
        self, maybe_tensor, arg_name: str, need_gpu_memory: bool
    ) -> Any:
        pass

    def _validate_arg(self, arg, arg_index, arg_name, arg_spec):
        """
        Validates if the arg is really of the annotated type for type safety.

        The default implementation is empty. Subclasses can override this method to add more validation logic.
        Returns None if validation passes, otherwise returns an error derived from DSLBaseError.
        """
        pass

    def _generate_jit_func_args_for_known_types(
        self,
        func,
        arg,
        arg_name,
        arg_spec,
        arg_index,
        *,
        is_host=True,
    ):
        """
        Generate JIT function arguments for known types.

        Sub-DSLs can override this method to handle types that are not
        natively supported by the Base DSL.
        """

        jit_arg_type, jit_arg_attr, jit_exec_arg = [], [], []
        default_attr = ir.DictAttr.get({})

        if is_argument_constexpr(arg, arg_spec, arg_name, arg_index, func):
            jit_exec_arg = jit_arg_type = jit_arg_attr = None

        return jit_exec_arg, jit_arg_type, jit_arg_attr

    def _generate_jit_func_args(
        self,
        func,
        function_name,
        args,
        kwargs,
        args_spec: inspect.FullArgSpec,
        *,
        is_host=True,
    ):
        """Generate JIT function arguments."""

        assert len(args) == len(args_spec.args) and len(kwargs) == len(
            args_spec.kwonlyargs
        ), (
            f"Input args {len(args)=} and kwargs {len(kwargs)=} must match arg_spec.args "
            f"{len(args_spec.args)=} and arg_spec.kwonlyargs {len(args_spec.kwonlyargs)=}"
        )

        jit_arg_types, jit_arg_attrs, jit_exec_args = [], [], []
        jit_adapted_args = []
        default_attr = ir.DictAttr.get({})

        input_args = [*args, *kwargs.values()]
        input_arg_names = [*args_spec.args, *args_spec.kwonlyargs]
        for i, (arg_name, arg) in enumerate(zip(input_arg_names, input_args)):
            spec_ty = args_spec.annotations.get(arg_name, None)
            log().debug("Processing [%d] Argument [%s : %s]", i, arg_name, spec_ty)

            # Implicitly convert into Numeric type if possible
            if isinstance(spec_ty, t.NumericMeta) and not isinstance(arg, spec_ty):
                arg = t.cast(arg, spec_ty)

            # Type safety check
            if spec_ty is not None:
                err = self._validate_arg(arg, i, arg_name, spec_ty)
                if err is not None:
                    raise err

            jit_exec_arg, jit_arg_type, jit_arg_attr = (
                self._generate_jit_func_args_for_known_types(
                    func,
                    arg,
                    arg_name,
                    spec_ty,
                    i,
                    is_host=is_host,
                )
            )

            if jit_arg_type is not None and len(jit_arg_type) == 0:
                # If not any known type, try JIT argument adapter
                # to convert the argument
                adapter = JitArgAdapterRegistry.get_registered_adapter(type(arg))
                if adapter:
                    arg = adapter(arg)
                    jit_adapted_args.append(arg)

                if is_host:
                    jit_exec_arg.extend(get_c_pointers(arg))
                    jit_arg_type.extend(get_mlir_types(arg))
                else:
                    dyn_vals = extract_mlir_values(arg)
                    jit_exec_arg.extend(dyn_vals)
                    jit_arg_type.extend([v.type for v in dyn_vals])

                if not jit_arg_type or not jit_exec_arg:
                    if (is_host and hasattr(arg, "__c_pointers__")) or (
                        not is_host
                        and hasattr(arg, "__extract_mlir_values__")
                        and hasattr(arg, "__new_from_mlir_values__")
                    ):
                        pass
                    else:
                        raise DSLRuntimeError(
                            f"failed to generate argument #{i+1} ({arg_name}) for JIT function '{function_name}'.",
                            context={
                                f"Argument {arg_name}": "The DSL attempted to convert it into Dynamic Expression (aka MLIR values) but failed.",
                                f"Call-site argument value": arg,
                                f"Call-site argument type": type(arg),
                            },
                            suggestion=f"Consider annotating the argument with `{arg_name} : Constexpr` "
                            "if it's a value known at compile-time. "
                            f"Otherwise, implement the {'`JitArgument`' if is_host else '`DynamicExpression`'} "
                            f"protocol or register a custom JIT argument adapter for type `{type(arg)}` to "
                            "enable dynamic value conversion at runtime.",
                        )

                jit_arg_attr.extend([default_attr] * len(jit_arg_type))

            if jit_arg_type is not None:
                jit_exec_args.extend(jit_exec_arg)
                jit_arg_types.extend(jit_arg_type)
                jit_arg_attrs.extend(jit_arg_attr)

        return jit_exec_args, jit_arg_types, jit_arg_attrs, jit_adapted_args

    def generate_mlir_function_types(
        self, func, function_name, input_args, kwargs, args_spec: inspect.FullArgSpec
    ):
        """Convert input arguments to MLIR function signature also convert numpy arrays to memref."""

        exe_args, types, attrs, adapted_args = self._generate_jit_func_args(
            func, function_name, input_args, kwargs, args_spec, is_host=True
        )

        log().debug("Execution Arguments: %s", ", ".join(map(str, exe_args)))
        log().debug("Types: %s", ", ".join(map(str, types)))

        assert len(exe_args) == len(
            types
        ), "expects the same number of arguments and function parameters"

        return exe_args, types, adapted_args

    @dataclass
    class LaunchConfig:
        cluster: list = None
        grid: list = field(default_factory=lambda: [1, 1, 1])
        block: list = field(default_factory=lambda: [1, 1, 1])
        smem: int = None
        async_deps: list = field(default_factory=list)
        has_cluster: bool = False
        min_blocks_per_mp: int = 0
        auto_smem: bool = False

        def __post_init__(self):
            if len(self.grid) != 3:
                raise DSLRuntimeError(f"Expect 3d grid!")
            if len(self.block) != 3:
                raise DSLRuntimeError(f"Expect 3d block!")

            if self.smem is None:
                self.smem = 0
                self.auto_smem = True

            self.has_cluster = self.cluster is not None
            if self.cluster is None:
                self.cluster = [None, None, None]
            elif len(self.cluster) != 3:
                raise DSLRuntimeError(f"Expect 3d cluster!")

    def diagnostic(self):
        """Check command line parameters and enables diagnostic"""
        # Check command line arguments "-diagnostic"
        parser = argparse.ArgumentParser(description="Process diagnostic status.")
        parser.add_argument(
            "-diagnostic",
            nargs="?",
            const="all",
            choices=["all", "fail", "success", "info", "suggestion"],
            help="Set diagnostic status (fail, success, info, suggestion).",
        )

        args, _ = parser.parse_known_args()
        ctx = ir.Context.current

        def callback(d):
            print(f"  [{self.name} Diagnostic] : {d.message}")

        ctx.attach_diagnostic_handler(callback)

        # Early return, don't enable diagnostics
        if args.diagnostic is None:
            return

        # Enable MLIR Flags
        ctx.emit_error_diagnostics = True
        ir._GlobalDebug.flag = True
        if args.diagnostic == "all":
            ir._GlobalDebug.set_types("diagnostic")
        else:
            ir._GlobalDebug.set_types(f"diagnostic-{args.diagnostic}")

    def get_location(self):
        """
        Get python location information and generate MLIR location
        """

        if self.frame is None:
            log().debug("Frame is None")
            return None

        file_loc = ir.Location.file(
            self.frame.f_code.co_filename, self.frame.f_lineno, 0
        )

        loc = ir.Location.name(self.frame.f_code.co_name, childLoc=file_loc)
        return loc

    def compile_and_jit(self, module, pipeline, shared_libs, function_name=""):
        """
        Compile and JIT an MLIR module.
        """

        try:
            self.diagnostic()

            orig_stdout = sys.stdout
            orig_stderr = sys.stderr
            sys.stderr = redirect_stderr = io.StringIO()
            sys.stdout = redirect_stdout = io.StringIO()

            try:
                kernel = self.compiler_provider.compile_and_jit(
                    module,
                    pipeline,
                    shared_libs=shared_libs,
                    cuda_toolkit=self.envar.cuda_toolkit,
                    arch=self.envar.arch,
                )

            finally:
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
                ir._GlobalDebug.flag = False

            # Print captured output.
            print(redirect_stdout.getvalue(), file=sys.stdout, end="")
            print(redirect_stderr.getvalue(), file=sys.stderr, end="")

            return kernel

        except Exception as e:
            raise DSLRuntimeError("🧊🧊🧊 ICE 🧊🧊🧊", cause=e)
        finally:
            pass

    def preprocess_pipeline(self, pipeline, arch) -> str:

        if self.envar.cuda_toolkit is None:
            self.print_warning(
                "CUDA_TOOLKIT_PATH environment variable is not set. Cannot set toolkitPath."
            )

        options = {
            "toolkitPath": self.envar.cuda_toolkit if self.envar.cuda_toolkit else None,
            self.pass_sm_arch_name: arch,
        }

        opt_str = ""
        for k, v in options.items():
            if v:
                opt_str += f"{k}={v} "

        if opt_str:
            # Automatically append the pipeline options if any is specified through env var
            pattern = re.compile(r"{(.+)}")
            match = pattern.search(pipeline)
            if match:
                opt_str = f"{{{match[1]} {opt_str}}}"
                pipeline = re.sub(r"{.+}", opt_str, pipeline)
            else:
                pipeline = pipeline.rstrip(")") + f"{{{opt_str}}})"
        log().debug(f"Using pipeline = {pipeline}")
        return pipeline

    def get_shared_libs(self) -> list:
        shared_libs = []
        support_libs = self.envar.shared_libs
        if support_libs is not None:
            _libs = support_libs.split(":")
            for lib in _libs:
                if not os.path.exists(lib):
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), lib
                    )
                shared_libs.append(lib)
        else:
            self.print_warning(f"{self.name}_LIBS environment variable is not set")

        return shared_libs

    @lru_cache(maxsize=1)
    def get_version(self):
        version_hash = hashlib.sha256()

        return version_hash

    def get_module_hash(self, module, function_name):
        s = io.BytesIO()
        module.operation.write_bytecode(s)
        for attr, value in self.envar.__dict__.items():
            if value is not None:
                s.write(str(value).encode())
        # Add compile options to the hash
        s.write(self.compile_options.to_str().encode())
        module_hash = self.get_version().copy()
        module_hash.update(s.getvalue())
        module_hash = module_hash.hexdigest()

        log().debug("Bytecode=[%s]", s.getvalue().hex())
        log().debug("Version=[%s]", self.get_version().hexdigest())
        log().info(
            "Function=[%s] Computed module_hash=[%s]", function_name, module_hash
        )
        return module_hash

    def build_module(self, module, function_name: str):
        """
        Build the MLIR module, verify and return the module
        """

        # Save IR in a file
        if self.envar.keepIR:
            save_ir(self.name, module, function_name)

        if self.envar.printIR:
            print("\n//===--- ------ Generated IR ------ ---====\n")
            module.operation.print(
                enable_debug_info=self.envar.generate_source_location
            )
            print("\n//===--- --- End of Generated IR -- ---====\n")

        # Verify the module
        try:
            module.operation.verify()
        except Exception as e:
            raise DSLRuntimeError(f"🧊🧊🧊 ICE IR Verification Failed 🧊🧊🧊", cause=e)

        return module

    def generate_original_ir(
        self,
        ir,
        func,
        funcBody,
        kwargs,
        function_name,
        func_types,
        gpu_module_attrs,
        args,
        args_spec,
    ):
        # This location is set to None for now; otherwise, calls to the same
        # function on different lines would produce different line numbers,
        # which would break the cache.
        loc = None  # self.get_location()

        def build_ir_module():
            module = ir.Module.create(loc=loc)
            unit_attr = ir.UnitAttr.get()
            module.operation.attributes["gpu.container_module"] = unit_attr

            with ir.InsertionPoint(module.body):
                # Always generate gpu module. It's canonicalized by the compiler when it's not used.
                self._build_gpu_module(gpu_module_attrs)

                fop = func.FuncOp(function_name, (func_types, []), loc=loc)
                fop.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
                log().debug("Generated Function OP [%s]", fop)
                with ir.InsertionPoint(fop.add_entry_block()):
                    ir_args, ir_kwargs = self.generate_execution_arguments(
                        args, kwargs, fop, args_spec
                    )
                    # Call user function body
                    try:
                        result = funcBody(*ir_args, **ir_kwargs)
                        func.ReturnOp([])
                    except NameError as name_error:
                        raise DSLRuntimeError(
                            f"💥💥💥 Error during runtime code generation for function `{funcBody.__name__}` 💥💥💥",
                            cause=name_error,
                            suggestion="Using variables defined in dynamic control flow is not supported. Please give an initial value before control flow.",
                        )
                    except DSLRuntimeError as dsl_error:
                        # Throw it's already a DSL error
                        raise dsl_error
            return module, result

        # Build IR module
        profiler = timer(enable=self.envar.jitTimeProfiling)
        module, result = profiler(build_ir_module)()
        module_hash = self.get_module_hash(module, function_name)

        module = self.build_module(module, function_name)

        return module, module_hash, result

    def compile_and_cache(
        self, module, module_hash, function_name, pipeline, args_spec, no_cache
    ):
        arch = self.envar.arch
        pipeline = self.preprocess_pipeline(self._get_pipeline(pipeline), arch)
        shared_libs = self.get_shared_libs()
        profiler = timer(enable=self.envar.jitTimeProfiling)
        if (
            no_cache
            or module_hash not in self.jit_cache
            or self.jit_cache[module_hash].ir_module is None
        ):
            log().info(
                "JIT cache miss function=[%s] module_hash=[%s]",
                function_name,
                module_hash,
            )
            # Compile and JIT MLIR module
            engine = profiler(self.compile_and_jit)(
                module, pipeline, shared_libs, function_name=function_name
            )
        else:
            log().info(
                "JIT cache hit IN-FILE function=[%s] module_hash=[%s]",
                function_name,
                module_hash,
            )
            module = self.jit_cache[module_hash].ir_module
            engine = self.compiler_provider.jit(module, shared_libs=shared_libs)
        capi_func = profiler(engine.lookup)(function_name)
        jit_executor = JitExecutor(
            self,
            engine,
            capi_func,
            module,
            args_spec,
            function_name,
            jit_time_profiling=self.envar.jitTimeProfiling,
        )
        jit_executor = jit_executor.update_jit_cuda_modules(self.kernel_symbols)

        if not no_cache:
            # module stored in cache is compiled.
            self.jit_cache[module_hash] = jit_executor

        return jit_executor

    def post_compilation_cleanup(self):
        """Clean up some internal state after one compilation is completed."""
        # clear the kernel symbols after the compilation is done.
        self.kernel_symbols = []
        self.launch_inner_count = 0
        # reset num_kernels to 0 for next compilation.
        self.num_kernels = 0
        # reset the compile options after the compilation is done.
        self.compile_options = CompileOptions()

    def generate_mlir(
        self,
        funcBody,
        kwargs,
        function_name,
        gpu_module_attrs,
        args,
        args_spec,
        pipeline,
        no_cache,
        compile_only,
        loc=None,
    ):
        """Generate MLIR module and compile iself.T_provider."""
        with ir.Context(), ir.Location.unknown():
            # Convert input arguments to MLIR arguments
            exe_args, func_types, adapted_args = self.generate_mlir_function_types(
                funcBody, function_name, args, kwargs, args_spec
            )

            # Generate original ir module and its hash value.
            module, module_hash, result = self.generate_original_ir(
                ir,
                func,
                funcBody,
                kwargs,
                function_name,
                func_types,
                gpu_module_attrs,
                args,
                args_spec,
            )

            # dryrun is used to only generate IR
            if self.envar.dryrun:
                return result

            if (
                no_cache
                or module_hash not in self.jit_cache
                or self.jit_cache[module_hash].capi_func is None
            ):
                # no cache or cache miss, do ir generation/compilation/jit engine
                jit_executor = self.compile_and_cache(
                    module, module_hash, function_name, pipeline, args_spec, no_cache
                )
            else:
                # cache hit
                log().info(
                    "JIT cache hit IN-MEMORY function=[%s] module_hash=[%s]",
                    function_name,
                    module_hash,
                )
                jit_executor = self.jit_cache[module_hash]

            self.post_compilation_cleanup()
        # If compile_only is set, bypass execution return the jit_executor directly
        if compile_only:
            return jit_executor
        # Run the compiled program
        jit_executor.run_compiled_program(exe_args)

        return result

    def run_preprocessor(self, funcBody):
        if not hasattr(funcBody, "_preprocessed"):
            function_name = funcBody.__name__
            self.funcBody = funcBody
            log().info("Started preprocessing [%s]", function_name)
            exec_globals = self._get_globals()
            transformed_ast = self.preprocessor.transform(funcBody, exec_globals)
            if self.envar.print_after_preprocessor:
                log().info(
                    f"# Printing unparsed AST after preprocess of func=`{function_name}` id=`{id(funcBody)}`"
                )
                DSLPreprocessor.print_ast(transformed_ast)
            funcBody._preprocessed = True
            return transformed_ast
        return None

    def get_function_ptr(self, original_function):
        file_name = inspect.getsourcefile(original_function)
        code_object = compile(
            original_function._transformed_ast, filename=file_name, mode="exec"
        )
        return self.preprocessor.exec(
            original_function.__name__,
            original_function,
            code_object,
            self._get_globals(),
        )

    def _get_function_bound_args(self, sig, func_name, *args, **kwargs):
        """
        Binds provided arguments to a function's signature and applies default values.

        E.g. given a function signature `def foo(a, b=2, c=3)`, and at call-site if we do
        `foo(a=1, c=4)`, the returned BoundArguments object will have args = `[1]`
        and kwargs = `{'b': 2, 'c': 4}`

        An exception will be raised if binding fails.
        """
        try:
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
        except Exception as e:
            raise DSLRuntimeError(
                f"Failed to bind arguments to function `{func_name}` with signature `{sig}`",
                cause=e,
            )
        return bound_args

    def _canonicalize_args(self, sig, *args, **kwargs):
        """
        Canonicalize the input arguments so that returned args only contain
        positional arguments and kwargs only contain keyword arguments.
        """
        function_name = self.funcBody.__name__
        bound_args = self._get_function_bound_args(sig, function_name, *args, **kwargs)
        canonicalized_args = bound_args.args
        canonicalized_kwargs = bound_args.kwargs
        return canonicalized_args, canonicalized_kwargs

    def _check_arg_count(self, *args, **kwargs):
        if not self.funcBody:
            raise DSLRuntimeError("Function body is not set.")

        # Pass the actual function object to inspect.signature to get the signature.
        sig = inspect.signature(self.funcBody)

        function_name = self.funcBody.__name__

        bound_args = self._get_function_bound_args(sig, function_name, *args, **kwargs)

        # Check if all non-default arguments are provided
        for param in sig.parameters.values():
            if (
                param.default is inspect.Parameter.empty
                and param.name not in bound_args.arguments
            ):
                raise DSLRuntimeError(
                    f"Missing required argument in `{function_name}`: '{param.name}'"
                )

        return sig

    def _func(self, funcBody, *args, **kwargs):
        """Decorator for MLIR functions.
        It cuts the boilerplate code, does the following:
            1. Generates `func.func`
            2. Types translation (numpy arrays -> cute.memref, float -> <f32>, etc.)
            3. Compiles and JITs the MLIR module
            4. Invokes the generated function
            5. Operator overloading (a + b --> arith.addi a, b)
            6. Generates GPU kernel function with GPU module and kernel attributes baked
        """
        if ir.Context.current is None:
            pass
        elif ir.InsertionPoint.current is not None:
            return funcBody(*args, **kwargs)

        function_name = funcBody.__name__
        self.funcBody = funcBody

        pipeline = kwargs.pop("pipeline", None)
        gpu_module_attrs = kwargs.pop("gpu_module_attrs", {})

        # Disable cache
        no_cache = kwargs.pop("no_cache", False)

        # Always compile(disable cache) and return the result jit_executor
        compile_only = kwargs.pop("compile_only", False)

        if not no_cache and compile_only:
            no_cache = True
            self.print_warning("Cache is disabled as user wants to compile only.")

        # Check the number of arguments
        sig = self._check_arg_count(*args, **kwargs)

        args_spec = inspect.getfullargspec(funcBody)

        # Canonicalize the input arguments
        canonicalized_args, canonicalized_kwargs = self._canonicalize_args(
            sig, *args, **kwargs
        )

        # Simple name mangling
        function_name = self.mangle_name(function_name, canonicalized_args, args_spec)

        # Generate MLIR Context and start generating IR
        log().debug(f"Generating MLIR for function '{function_name}'")
        result = self.generate_mlir(
            funcBody,
            canonicalized_kwargs,
            function_name,
            gpu_module_attrs,
            canonicalized_args,
            args_spec,
            pipeline,
            no_cache,
            compile_only,
        )

        return result

    class _KernelGenHelper(ABC):
        def __init__(self):
            self.func_op = None
            self.func_type = None

        @abstractmethod
        def generate_func_op(self, arg_types, arg_attrs, kernel_name, loc=None):
            assert arg_types is not None, "Invalid arg_types!"
            assert kernel_name is not None, "kernel name is empty"
            pass

        @abstractmethod
        def generate_func_ret_op(self):
            pass

        @abstractmethod
        def generate_launch_op(self, *args, **kwargs):
            pass

        @abstractmethod
        def get_func_body_start(self):
            pass

    @abstractmethod
    def enter_gpu_module(module):
        """Compute the insertion point into the given module."""
        pass

    @lru_cache(maxsize=1)
    def _get_default_stream(self):
        """Returns the default stream 0"""
        from .runtime import cuda as cuda_helpers

        return cuda_helpers.stream_create()

    def _execute_cuda(
        self, fname_cubin, kernel_name, grid_size, block_size, smem_size, stream=None
    ):
        """
        Executes a specified CUDA kernel from a cubin file, handling module loading,
        kernel retrieval, stream creation, kernel launch, and synchronization.
        """
        from .runtime import cuda as cuda_helpers

        # Step 1. Load CUDA Module
        module = cuda_helpers.load_cubin_module(fname_cubin)
        # Step 2. Find CUDA function
        kernel_ptr = cuda_helpers.get_kernel_function(module, kernel_name)

        sync_execution_default = False
        if stream is None:
            stream = self._get_default_stream()
            sync_execution_default = True

        # Step 4. Launch the kernel
        cuda_helpers.launch_kernel(
            kernel_ptr,
            grid_size,
            block_size,
            stream,
            smem_size=smem_size,
            kernel_args=self.exe_args,
        )

        if sync_execution_default:
            # Step 5. Optional Sync cuda stream
            cuda_helpers.stream_sync(stream)

    def _execute_by_cuda_driver(
        self,
        kernel_generator,
        generate_cubin,
        grid_size,
        block_size,
        smem_size,
        stream=None,
    ):
        """
        This function builds IR and execute the module using cuda driver.
        It doesn't use mlir's cuda runtime
        """
        ret = None

        # Step 1. Build IR
        with ir.Context(), ir.Location.unknown():
            loc = self.get_location()
            module = ir.Module.create(loc=loc)
            unit_attr = ir.UnitAttr.get()
            module.operation.attributes["gpu.container_module"] = unit_attr
            with ir.InsertionPoint(module.body):
                self._build_gpu_module()
                ret, kernel_name = kernel_generator()
                log().debug(
                    f"Kernel generator returned: ret={ret}, kernel_name={kernel_name}"
                )

        module = self.build_module(module, kernel_name)

        # dryrun is used to only generate IR
        if self.envar.dryrun:
            return ret

        # Generate cubin
        fname_cubin = generate_cubin(module, kernel_name)

        # Execute a cuda kernel from cubin
        self._execute_cuda(
            fname_cubin, kernel_name, grid_size, block_size, smem_size, stream
        )

        return ret

    def generate_kernel_operands_and_types(
        self, kernel_func, kernel_name, args_spec, args, kwargs
    ):
        """
        Generate the operands and types for the kernel function
        """

        kernel_operands, kernel_arg_types, kernel_arg_attrs = [], [], []

        log().debug(
            "Processing GPU kernel call in [%s] mode",
            (
                f"Only {self.device_jit_decorator_name}"
                if self.device_compilation_only
                else f"{self.host_jit_decorator_name} + {self.device_jit_decorator_name}"
            ),
        )

        if self.device_compilation_only:
            return kernel_operands, kernel_arg_types, kernel_arg_attrs

        kernel_operands, kernel_arg_types, kernel_arg_attrs, _ = (
            self._generate_jit_func_args(
                kernel_func, kernel_name, args, kwargs, args_spec, is_host=False
            )
        )

        log().debug("Final kernel_operands: %s", ", ".join(map(str, kernel_operands)))
        log().debug("Final kernel_arg_types: %s", ", ".join(map(str, kernel_arg_types)))
        log().debug("Final kernel_arg_attrs: %s", ", ".join(map(str, kernel_arg_attrs)))

        assert (
            len(kernel_operands) == len(kernel_arg_types) == len(kernel_arg_attrs)
        ), "Size of kernel_operands, kernel_arg_types and kernel_arg_attrs must be equal"

        return kernel_operands, kernel_arg_types, kernel_arg_attrs

    def kernel_launcher(self, *dargs, **dkwargs):
        def decorator(funcBody):
            @wraps(funcBody)
            def kernel_wrapper(*args, **kwargs):
                """
                Base decorator for generating kernel function

                This decorator provides a template for kernel function generation
                including kernel function header/body and kernel launch op at call site

                Optional arguments (with default value in <>):
                  - requiredArgs <[]>:      specifies the mandatory arguments that must present in kernel function signature
                                            the args will be validated and collected as a namedtuple
                  - optionalArgs <[]>:      specifies the optional arguments that might present in kernel function signature
                                            the args will be collected (if present) as a namedtuple
                  - unitAttrNames <[]>:     specifies the name(s) of ir.UnitAttr to be set for kernel function op
                  - valueAttrDict <{}>:     specifies the name(s) and value(s) of ir.Attribute to be set for kernel function op
                  - kernelGenHelper <None>: specifies the mandatory customized kernel generation helper class (derived from _KernelGenHelper)

                Return value:
                  A namedtuple "KernelReturns" is returned with following fields:
                  - kernel_func_ret: the return of the kernel function
                  - launch_op_ret:   the return of the launch op
                """

                requiredArgs = dkwargs.get("requiredArgs", [])
                optionalArgs = dkwargs.get("optionalArgs", [])
                unitAttrNames = dkwargs.get("unitAttrNames", [])
                valueAttrDict = dkwargs.get("valueAttrDict", {})
                kernelGenHelper = dkwargs.get("kernelGenHelper", None)

                kernel_name = funcBody.__name__
                args_spec = inspect.getfullargspec(funcBody)
                self.funcBody = funcBody

                # Give each kernel a unique name. (The same kernel may be
                # called multiple times, resulting in multiple kernel traces.)
                # The mangled name of Python function is part of the name to
                # improve readability.
                kernel_name = f"kernel_{self.mangle_name(kernel_name, args, args_spec)}_{self.num_kernels}"
                self.num_kernels += 1

                # Step 0. Preprocess the arguments
                def extract_args(argNames, assertIfNone=False) -> list:
                    extracted = []
                    for name in argNames:
                        value = kwargs.pop(name, None)
                        if assertIfNone and value is None:
                            raise DSLRuntimeError(
                                f"{name} is required for {kernel_name}"
                            )
                        extracted.append(value)

                    return extracted

                RequiredArgs = namedtuple("RequiredArgs", requiredArgs)
                req_args = (
                    RequiredArgs._make(extract_args(requiredArgs, assertIfNone=True))
                    if requiredArgs
                    else None
                )
                OptionalArgs = namedtuple("OptionalArgs", optionalArgs)
                opt_args = (
                    OptionalArgs._make(extract_args(optionalArgs))
                    if optionalArgs
                    else None
                )
                assert (
                    kernelGenHelper is not None
                ), "kernelGenHelper should be explicitly specified!"

                # check arguments
                sig = self._check_arg_count(*args, **kwargs)

                # Canonicalize the input arguments
                canonicalized_args, canonicalized_kwargs = self._canonicalize_args(
                    sig, *args, **kwargs
                )

                kernel_operands, kernel_types, kernel_arg_attrs = (
                    self.generate_kernel_operands_and_types(
                        funcBody,
                        kernel_name,
                        args_spec,
                        canonicalized_args,
                        canonicalized_kwargs,
                    )
                )

                with self._enter_gpu_module():
                    log().debug("Generating device kernel")
                    if self.device_compilation_only:
                        log().debug("Generating cuda-python arguments")
                        # Convert input arguments to MLIR arguments
                        self.exe_args, kernel_types, _ = (
                            self.generate_mlir_function_types(
                                funcBody,
                                kernel_name,
                                canonicalized_args,
                                canonicalized_kwargs,
                                args_spec,
                            )
                        )

                    helper = kernelGenHelper()
                    loc = self.get_location()
                    fop = helper.generate_func_op(
                        kernel_types, kernel_arg_attrs, kernel_name, loc
                    )
                    log().debug(f"Kernel function op: {fop}")
                    for attr in unitAttrNames:
                        fop.attributes[attr] = ir.UnitAttr.get()
                    for key, val in valueAttrDict.items():
                        fop.attributes[key] = val

                    fop.sym_visibility = ir.StringAttr.get("public")
                    with ir.InsertionPoint(helper.get_func_body_start()):
                        ir_args, ir_kwargs = self.generate_execution_arguments(
                            canonicalized_args, canonicalized_kwargs, fop, args_spec
                        )
                        log().debug(
                            f"IR arguments - args: {ir_args} ; kwargs: {ir_kwargs}"
                        )
                        # Call user function body
                        kernel_ret = funcBody(*ir_args, **ir_kwargs)
                        helper.generate_func_ret_op()

                # Step 3. Generate call site `launch_func`
                kernel_sym = ir.SymbolRefAttr.get(["kernels", kernel_name])
                launch_ret = helper.generate_launch_op(
                    kernelSym=kernel_sym,
                    kernelOperands=kernel_operands,
                    requiredArgs=req_args,
                    optionalArgs=opt_args,
                )

                KernelReturns = namedtuple(
                    "KernelReturns", ["kernel_func_ret", "launch_op_ret"]
                )
                result = KernelReturns(
                    kernel_func_ret=kernel_ret, launch_op_ret=launch_ret
                )
                log().debug(f"Kernel result: {result}, kernel name: {kernel_name}")
                return result, kernel_name

            return kernel_wrapper

        if len(dargs) == 1 and callable(dargs[0]):
            return decorator(dargs[0])
        else:
            return decorator
