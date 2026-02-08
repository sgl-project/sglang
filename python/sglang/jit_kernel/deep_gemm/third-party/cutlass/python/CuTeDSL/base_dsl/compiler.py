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
This module provides a class that compiles generated IR using MLIR's PassManager
and executes it using MLIR's ExecutionEngine.

"""

from typing import Sequence, Optional, Tuple
import os
import sys
import inspect
import argparse
from .common import DSLRuntimeError
from .utils.logger import log

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from .._mlir import ir


# =============================================================================
# Compiler Class
# =============================================================================


class CompilationError(RuntimeError):
    """Custom error class for compilation failures"""

    # Add ANSI color codes
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def __init__(
        self,
        message: str,
        nvvm_error: Optional[str] = None,
        ir_context: Optional[str] = None,
        cuda_toolkit: Optional[str] = None,
        arch: Optional[str] = None,
    ):
        self.nvvm_error = nvvm_error
        self.ir_context = ir_context
        self.cuda_toolkit = cuda_toolkit
        self.arch = arch
        # Call parent with formatted error to avoid showing class name
        super().__init__("")  # Empty string to avoid class name
        # Store formatted error for str() representation
        self._formatted_error = self._format_error()

    def __str__(self) -> str:
        """Override string representation to avoid showing class name"""
        return self._formatted_error

    def __repr__(self) -> str:
        """Override repr representation to avoid showing class name"""
        return self._formatted_error

    def _format_error(self) -> str:
        if not self.nvvm_error:
            return str(self.args[0])

        return f"""NVVM Compilation Error:
----------------------

{self.BLUE}⚙️  Current Settings:{self.RESET}
{self.BOLD}- CUDA Toolkit Path: {self.cuda_toolkit or "Not Set"}
- Target Architecture: {self.arch}{self.RESET}

IR Context (truncated):
{self.ir_context}

{self.YELLOW}💡 Possible Solutions:{self.RESET}
{self.GREEN}1. Check if CUDA_TOOLKIT_PATH is set correctly
2. Verify target architecture ({self.arch}) is supported by your CUDA toolkit
3. Make sure CUDA toolkit version matches the target architecture requirements{self.RESET}"""


class Compiler:
    """Compiler class for compiling and building MLIR modules."""

    def __init__(self, passmanager, execution_engine):
        self.passmanager = passmanager
        self.execution_engine = execution_engine

    def __call__(self, module):
        """Convenience application method."""
        self.compile(module)

    def _process_error(self, error_msg: str) -> Tuple[Optional[str], Optional[str]]:
        """Process error message to extract NVVM error and IR context"""
        nvvm_error = None
        ir_msg = ""

        if "NVVM_ERROR" in error_msg:
            # Extract the specific NVVM error
            nvvm_error = (
                error_msg.split("libNVVM extra log:")[1].strip()
                if "libNVVM extra log:" in error_msg
                else error_msg
            )

            # Extract IR context
            if "see current operation:" in error_msg:
                # Get the IR section
                ir_section = error_msg.split("see current operation:")[1].strip()
                # Remove duplicate IR section
                ir_section = ir_section.split("error: unknown: Failed translating")[
                    0
                ].strip()

                # Get first few lines and last few lines of the IR
                ir_lines = ir_section.split("\n")
                if len(ir_lines) > 10:
                    ir_msg = "\n".join(ir_lines[:5] + ["  ..."] + ir_lines[-5:])
                else:
                    ir_msg = ir_section

        return nvvm_error, ir_msg

    def compile(
        self,
        module,
        pipeline: str,
        cuda_toolkit: str = "",
        arch: str = "",
        enable_verifier=False,
    ):
        """Compiles the module by invoking the pipeline."""
        try:
            pm = self.passmanager.PassManager.parse(pipeline)
            pm.enable_verifier(enable_verifier)
            pm.run(module.operation)
        except Exception as e:
            error_msg = str(e)
            nvvm_error, ir_msg = self._process_error(error_msg)

            if nvvm_error:
                raise CompilationError(
                    error_msg,
                    nvvm_error=nvvm_error,
                    ir_context=ir_msg,
                    cuda_toolkit=cuda_toolkit,
                    arch=arch,
                ) from e
            raise e

    def jit(self, module, opt_level: int = 2, shared_libs: Sequence[str] = ()):
        """Wraps the module in a JIT execution engine."""
        return self.execution_engine.ExecutionEngine(
            module, opt_level=opt_level, shared_libs=shared_libs
        )

    def compile_and_jit(
        self,
        module,
        pipeline: str,
        shared_libs: Sequence[str] = (),
        opt_level: int = 2,
        cuda_toolkit: str = "",
        arch: str = "",
    ):
        """Compiles and jits the module."""
        self.compile(
            module,
            pipeline,
            cuda_toolkit,
            arch,
        )
        return self.jit(module, opt_level, shared_libs)


class CompileOptions:
    def __init__(self, options: str = ""):
        """
        This class encapsulates all compilation options relevant to function compilation.
        It provides a convenient way to manage and pass compilation options,
        particularly for controlling compilation settings.
        By centralizing these options, it ensures consistent and flexible configuration of
        compilation parameters such as optimization level, debugging control, etc.

        :param options: The options for the function. Will be parsed by argparse.
        :type options: str
        """
        if not isinstance(options, str):
            raise DSLRuntimeError(
                f"Invalid compilation `options`: {options}, it should be a string"
            )
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument("--opt-level", nargs="?", type=int, default=3)
        self._parser.add_argument(
            "--enable-device-assertions", action="store_true", default=False
        )
        self._parser.add_argument("--link-libraries", type=str, default="")

        try:
            self._options = self._parser.parse_args(options.split())
        except SystemExit as e:
            # catch argparse error and raise as DSLRuntimeError
            raise DSLRuntimeError(
                f"Invalid compile options: '{options}'. Please check the option values and format."
            )
        log().info("`cute.compile` CompileOptions: options=" + options)

    def to_str(self):
        """
        Generate a string representation of all compilation options
        which will be used in pipeline options.
        """
        option_strings = []
        for key, value in vars(self._options).items():
            hyphen_key = key.replace("_", "-")
            if isinstance(value, bool):
                formatted_value = "true" if value else "false"
            else:
                formatted_value = str(value)
            option_strings.append(f"{hyphen_key}={formatted_value}")

        return " ".join(option_strings)


def compile(func, *args, **kwargs):
    """
    This function is used to compile a `cute.jit` decorated function.
    It will process the compile options and input parameters, do explicit compilation and return  the jit executor.

    :param func: The function to compile. It can be a regular function, a method or a class instance.
    :param args: The arguments to pass to the function.
    :param kwargs: The keyword arguments to pass to the function. It can contain `options` like
    `opt_level` to control the compilation flags.

    :return: The jit executor.

    :raises: DSLRuntimeError if the function is not decorated with `cute.jit` or is not callable.
    """
    if func is None:
        raise DSLRuntimeError("Function is not set or invalid.")

    if not callable(func):
        raise DSLRuntimeError("Object is not callable.")

    kwargs["compile_only"] = True
    kwargs["no_cache"] = True

    if inspect.isfunction(func):
        # regular function
        pass
    elif inspect.ismethod(func):
        # if it's a method, add the instance to the first argument
        args = [func.__self__] + list(args)
        func = func.__func__
    elif inspect.isclass(type(func)) and hasattr(func, "__call__"):
        # If it's a class instance, get the class's __call__ method
        args = [func] + list(args)
        # Get the actual function from the class definition
        func = func.__call__.__func__
    else:
        raise DSLRuntimeError(
            "Invalid function type, only function, method and module are supported, but got",
            func,
        )

    # If it's a wrapped function created by jit decorator, get the original function
    if hasattr(func, "__wrapped__"):
        func = func.__wrapped__

    if not hasattr(func, "_dsl_object"):
        raise DSLRuntimeError("Function is not decorated with jit decorator.")

    # process compile options, extract the options and remove them from the kwargs
    options = kwargs.pop("options", "")
    func._dsl_object.compile_options = CompileOptions(options)
    fcn_ptr = func._dsl_object._preprocess_and_execute(func)
    return func._dsl_object._func(fcn_ptr, *args, **kwargs)
