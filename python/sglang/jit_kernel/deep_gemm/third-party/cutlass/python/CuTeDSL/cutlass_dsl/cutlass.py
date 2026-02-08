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
This module provides a DSL for Cutlass Dialects. It also includes utils with
regarding to that dialect.
"""

# Local module imports
from itertools import chain
from types import GenericAlias, SimpleNamespace, UnionType
from typing import Callable, Union, Type, List, Union, Sequence, ForwardRef, Any
import functools
import pkgutil
from dataclasses import is_dataclass, fields
from collections.abc import Sequence
import builtins

from ..base_dsl import *
from ..base_dsl import compiler
from ..base_dsl.dsl import is_dynamic_expression, extract_mlir_values
from ..base_dsl.typing import *
from ..base_dsl.typing import DynamicExpression, get_mlir_types
from ..base_dsl.runtime.jit_arg_adapters import is_arg_spec_constexpr

from ..base_dsl.ast_helpers import const_expr

# MLIR Imports
from cutlass._mlir import ir, execution_engine, passmanager
from cutlass._mlir.dialects import arith, func, gpu, scf, cute, gpu as cutlass_gpu
from cutlass._mlir.dialects._ods_common import (
    get_op_result_or_op_results as _get_op_result_or_op_results,
)
from cutlass._mlir.extras import types as T

# Helpers
from ..base_dsl._mlir_helpers import arith as cutlass_arith
from ..base_dsl._mlir_helpers import lru_cache_ir

from ..base_dsl.ast_helpers import (
    loop_selector,
    executor,
    if_selector,
    if_executor,
    while_selector,
    while_executor,
    assert_executor,
    const_expr,
    dynamic_expr,
    bool_cast,
    compare_executor,
    any_executor,
    all_executor,
    range_value_check,
    range_perf_warning,
    cf_symbol_check,
)

from .cutlass_ast_decorators import (
    _loop_execute_range_dynamic,
    _if_execute_dynamic,
    _while_execute_dynamic,
)

from .tree_utils import (
    is_constexpr_field,
    tree_flatten,
    tree_unflatten,
    PyTreeDef,
    is_frozen_dataclass,
    DSLTreeFlattenError,
)
from ..base_dsl.runtime.jit_arg_adapters import JitArgAdapterRegistry


# =============================================================================
# Cutlass DSL Base Abstract Class
# =============================================================================


# Return a ctype class that represents the in-memory layout expected
# for a CuTe hierarchical tuple type.
def get_sparse_tuple_ctype(dyn):
    # When there is a single dynamic value, the sparse CuTe
    # representation is a single integer.
    if isinstance(dyn, int):
        return ctypes.c_int32

    # For zero or greater than 1 dynamic values, the tuple
    # representation will be a struct with a field for each dynamic
    # value. The representation is flattened, even for hierarchical CuTe
    # profiles (although we are only dealing with depth 1 inputs here).
    class TupleDescriptor(ctypes.Structure):
        _fields_ = [(f"x{idx}", ctypes.c_int32) for idx in range(len(dyn))]

        def __str__(self):
            return f"struct<{str(self._fields_)}>"

    return TupleDescriptor


def is_cute_algebra_type(arg_spec):
    # Walk through the arg_spec to check if it's a cute algebra type
    _cute_algebra_type_aliases = (
        "Shape",
        "Stride",
        "Coord",
        "Tile",
        "IntTuple",
    )

    origin = get_origin(arg_spec)
    if origin is Union:
        for sub_ty in get_args(arg_spec):
            sub_origin = get_origin(sub_ty)
            if sub_origin is Tuple or (
                type(sub_origin) is type and issubclass(sub_origin, tuple)
            ):
                tuple_arg0 = get_args(sub_ty)[0]
                if isinstance(
                    tuple_arg0, ForwardRef
                ) and tuple_arg0.__forward_arg__ in (_cute_algebra_type_aliases):
                    return True
    return False


def _get_c_pointers_cutlass(obj):
    """
    This is an extended version of `get_c_pointers` that supports dataclasses, SimpleNamespace, and dict.
    """
    if hasattr(obj, "__c_pointers__"):
        return obj.__c_pointers__()
    elif isinstance(obj, (tuple, list)):
        return list(chain.from_iterable(_get_c_pointers_cutlass(x) for x in obj))
    elif isinstance(obj, SimpleNamespace):
        return list(
            chain.from_iterable(
                _get_c_pointers_cutlass(x) for x in obj.__dict__.values()
            )
        )
    elif isinstance(obj, dict):
        return list(
            chain.from_iterable(_get_c_pointers_cutlass(x) for x in obj.values())
        )
    elif is_dataclass(obj):
        return list(
            chain.from_iterable(
                _get_c_pointers_cutlass(getattr(obj, f.name))
                for f in fields(obj)
                if not is_constexpr_field(f)
            )
        )
    elif isinstance(obj, set):
        raise DSLRuntimeError(
            "Sets are not supported in get_c_pointers to ensure order preservation",
            context="The DSL attempted to generate JIT function argument(s) for an argument of type set but failed.",
            suggestion="Consider using a list or tuple instead",
        )
    else:
        # Try get adapter
        adapter = JitArgAdapterRegistry.get_registered_adapter(type(obj))
        if adapter is not None:
            return _get_c_pointers_cutlass(adapter(obj))
    return []


class CutlassBaseDSL(BaseDSL):
    """This abstract class provides a DSL for Cutlass."""

    def __init__(
        self,
        name: str,
        compiler_provider: Any,
        pass_sm_arch_name: str,
        device_compilation_only: bool = False,
        preprocess: bool = False,
    ):
        super().__init__(
            name=name,
            dsl_package_name=["cutlass"],
            compiler_provider=compiler_provider,
            pass_sm_arch_name=pass_sm_arch_name,
            device_compilation_only=device_compilation_only,
            preprocess=preprocess,
        )
        self._smem_usage_tracker: tuple = None

    # this method is not useful for cutlass_dsl, so we only provide a dummy implementation.
    def _is_tensor_descriptor(self, maybe_tensor_descriptor) -> bool:
        return False

    # this method is not useful for cutlass_dsl, so we only provide a dummy implementation.
    def _handle_tensor_descriptor(
        self, maybe_tensor, arg_name: str, need_gpu_memory: bool
    ) -> Any:
        return False

    def _build_gpu_module(self, attrs):
        self.gpu_module = gpu.GPUModuleOp(ir.StringAttr.get("kernels"))
        with ir.InsertionPoint(self.gpu_module.bodyRegion.blocks.append(*[])):
            pass

        for attr_name in attrs:
            self.gpu_module.attributes[attr_name] = ir.Attribute.parse(attrs[attr_name])

    def _get_pipeline(self, pipeline):
        pipeline = super()._get_pipeline(pipeline)
        if pipeline == None:
            # cubin format is required to be cubin as we launch cuda module at python level.
            return (
                "builtin.module(cute-to-nvvm{cubin-format=bin "
                + self.compile_options.to_str()
                + "})"
            )

        return pipeline

    def preprocess_pipeline(self, pipeline, arch) -> str:
        pipeline = super().preprocess_pipeline(pipeline, arch)
        pipeline = pipeline.rstrip(")") + ",external-kernel-for-gpu-launch)"
        return pipeline

    def _enter_gpu_module(self):
        return ir.InsertionPoint(self.gpu_module.bodyRegion.blocks[0])

    def _generate_kernel_attrs(self, config: BaseDSL.LaunchConfig) -> dict:
        assert isinstance(
            config, BaseDSL.LaunchConfig
        ), f"Expect LaunchConfig for @kernel, but got {type(config)}"

        ret = {}
        # generate launch bound attr from LaunchConfig
        max_threads = ", ".join(map(str, config.block))
        ret["nvvm.reqntid"] = ir.Attribute.parse(f"array<i32 : {max_threads}>")
        # min_blocks_per_mp is optional for kernel
        min_blocks = config.min_blocks_per_mp
        if min_blocks > 0:
            ret["nvvm.minctasm"] = ir.Attribute.parse(f"{min_blocks} : i32")
        return ret

    @lru_cache(maxsize=1)
    def get_version(self):
        """
        Get the version of cutlass dsl, used for computing the hash key of the cache.
        Including source python files and the shared library.
        """
        dsl_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # get the version hash of the cutlass shared library
        version_hash = hashlib.sha256()
        # update the version hash of the source python files
        for lib in pkgutil.walk_packages([dsl_path], prefix="cutlass."):
            try:
                with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
                    version_hash.update(f.read())
            except Exception:
                raise DSLRuntimeError(
                    f"Failed to read module file {lib.name}. The file may not exist or may not be readable."
                    "Please re-install the package."
                )
        try:
            # update the version hash of the cutlass shared library
            with open(
                os.path.join(dsl_path, "_mlir/_mlir_libs/libCutlassIRPythonCAPI.so"),
                "rb",
            ) as f:
                while True:
                    chunk = f.read(1024**2)
                    if not chunk:
                        break
                    version_hash.update(chunk)
        except Exception:
            raise DSLRuntimeError(
                f"Failed to read the shared library file libCutlassIRPythonCAPI.so."
                "The file may not exist or may not be readable."
                "Please re-install the package."
            )

        return version_hash

    @staticmethod
    def track_smem_allocator(allocator, callback):
        """
        Tracks shared memory usage for kernel functions.
        Find and set allocator to its parent dsl object.
        """
        frame = inspect.currentframe().f_back
        while frame:
            obj = frame.f_locals.get("self", None)
            if obj and isinstance(obj, CutlassBaseDSL):
                obj._set_smem_tracking(allocator, callback)
                return
            frame = frame.f_back
        warnings.warn("Cannot find parent dsl for allocator!", UserWarning)

    def _set_smem_tracking(self, allocator, callback):
        # Registers an allocator and callback for current dsl
        self._smem_usage_tracker = (allocator, callback)

    def _reset_smem_tracking(self):
        # Clear an allocator and callback for current dsl
        self._smem_usage_tracker = None

    def _get_smem_usage(self) -> int:
        # Treat final allocated bytes of allocator as smem usage
        if not self._smem_usage_tracker:
            return 0
        allocator, callback = self._smem_usage_tracker
        return callback(allocator)

    def _kernel_helper(self, funcBody, *args, **kwargs):
        class _CutlassIrKernelGenHelper(BaseDSL._KernelGenHelper):
            def __init__(self, dsl: CutlassBaseDSL):
                super().__init__()
                self.dsl = dsl
                self.dsl._reset_smem_tracking()

            def generate_func_op(self, arg_types, arg_attrs, kernel_name, loc=None):
                super().generate_func_op(arg_types, arg_attrs, kernel_name)
                self.func_op = func.FuncOp(
                    kernel_name, ir.FunctionType.get(arg_types, []), loc=loc
                )
                if arg_attrs is not None:
                    log().debug(arg_attrs)
                    self.func_op.arg_attrs = arg_attrs
                return self.func_op

            def generate_func_ret_op(self):
                return func.ReturnOp([])

            def get_func_body_start(self):
                assert self.func_op is not None, "Invalid func_op is not expected!"
                return self.func_op.add_entry_block()

            def generate_launch_op(self, *args, **kwargs):
                # Extract args and do validation
                kernelSym = kwargs.get("kernelSym", None)
                kernelOperands = kwargs.get("kernelOperands", None)
                requiredArgs = kwargs.get("requiredArgs", None)
                assert kernelSym is not None, "kernelSym being None is not expected!"
                assert (
                    requiredArgs is not None
                ), "requiredArgs being None is not expected!"
                assert (
                    kernelOperands is not None
                ), "kernelOperands being None is not expected!"
                assert isinstance(
                    requiredArgs.config, BaseDSL.LaunchConfig
                ), f"Expect LaunchConfig for @kernel, but got {type(requiredArgs.config)}"

                cfg = requiredArgs.config

                # Apply to grid, block, and cluster if present
                cfg.grid = [to_index(size) for size in cfg.grid]
                cfg.block = [to_index(size) for size in cfg.block]
                if cfg.has_cluster:
                    cfg.cluster = [to_index(size) for size in cfg.cluster]

                smem_usage = self.dsl._get_smem_usage()
                if any(not isinstance(x, int) for x in [cfg.smem, smem_usage]):
                    pass  # cannot compare dynamic value inside kernel to launch op in py
                elif cfg.auto_smem:
                    cfg.smem = smem_usage
                elif smem_usage > cfg.smem:
                    warnings.warn(
                        f"Potential error: specified kernel launch smem bytes "
                        f"({cfg.smem}) is smaller than kernel usage ({smem_usage})!",
                        UserWarning,
                    )
                cfg.smem = const(cfg.smem)

                if not isinstance(cfg.async_deps, (list, tuple)):
                    cfg.async_deps = [cfg.async_deps]
                is_async = len(cfg.async_deps) > 0
                token = gpu.launch_func(
                    gpu.AsyncTokenType.get() if is_async else None,
                    cfg.async_deps,
                    kernelSym,
                    *cfg.grid,
                    *cfg.block,
                    kernelOperands,
                    **dict(
                        zip(
                            ("cluster_size_x", "cluster_size_y", "cluster_size_z"),
                            tuple(cfg.cluster),
                        )
                    ),
                    dynamic_shared_memory_size=cfg.smem,
                )
                return token if is_async else None

        return KernelLauncher(
            self,
            lambda: _CutlassIrKernelGenHelper(self),
            funcBody,
            *args,
            **kwargs,
        )

    def _preprocess_launch_config_args(self, args, kwargs):
        """Helper to preprocess args and kwargs for LaunchConfig"""
        if "stream" in kwargs:
            kwargs["async_deps"] = kwargs.pop("stream")

    def mangle_name(self, function_name, args, args_spec: inspect.FullArgSpec):
        """Mangle the name of the function to avoid conflicts with other functions"""
        function_name = "cutlass_" + function_name
        return super().mangle_name(function_name, args, args_spec)

    def _validate_arg(self, arg, arg_index, arg_name, arg_annotation):
        """
        Validates if the arg is really of the annotated type.
        """

        if (
            is_arg_spec_constexpr(arg_annotation, arg_name, arg_index, None)
            or arg_annotation is Any
        ):
            pass
        else:
            origin = get_origin(arg_annotation)
            # Handle special case where annotation is Type[X] but arg is an actual type
            if origin is type and isinstance(arg, type):
                # Get the expected base type from Type[X]
                expected_base = get_args(arg_annotation)[0]
                if not issubclass(arg, expected_base):
                    return DSLRuntimeError(
                        f"expects argument #{arg_index+1} ({arg_name}) to be Type[{expected_base}], but got {arg}"
                    )
            # Handle Union types and generic types
            elif origin is Union or isinstance(arg_annotation, UnionType):
                # For Union types, check if arg matches any of the allowed types
                allowed_types = get_args(arg_annotation)
                if not any(
                    (ty is Any)
                    or (isinstance(ty, type) and isinstance(arg, ty))
                    or (get_origin(ty) is tuple and isinstance(arg, tuple))
                    for ty in allowed_types
                ):
                    return DSLRuntimeError(
                        f"expects argument #{arg_index+1} ({arg_name}) to be one of {allowed_types}, but got {type(arg)}"
                    )
            elif isinstance(arg_annotation, type):
                # Handle simple type annotations
                if not isinstance(arg, arg_annotation) and arg is not None:
                    return DSLRuntimeError(
                        f"expects argument #{arg_index+1} ({arg_name}) to be {arg_annotation}, but got {type(arg)}"
                    )
        # Everything looks good if we are here
        return None

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
        jit_arg_type, jit_arg_attr, jit_exec_arg = [], [], []
        default_attr = ir.DictAttr.get({})

        (
            jit_exec_arg,
            jit_arg_type,
            jit_arg_attr,
        ) = super()._generate_jit_func_args_for_known_types(
            func, arg, arg_name, arg_spec, arg_index, is_host=is_host
        )

        if jit_arg_type is not None and len(jit_arg_type) == 0:
            # Handle DSL specific types
            if is_cute_algebra_type(arg_spec):
                dyn_vals = extract_mlir_values(arg)
                if dyn_vals:
                    # Handle dynamic types
                    jit_arg_type.extend([v.type for v in dyn_vals])
                    jit_arg_attr.extend([default_attr] * len(dyn_vals))
                    jit_exec_arg.extend(get_c_pointers(arg) if is_host else dyn_vals)
                else:
                    jit_exec_arg = jit_arg_type = jit_arg_attr = None
            elif not hasattr(arg, "__extract_mlir_values__") and not hasattr(
                arg, "__new_from_mlir_values__"
            ):
                # Try tree_flatten
                try:
                    dyn_vals, _ = tree_flatten(arg)
                except DSLTreeFlattenError:
                    # If fails, just return the original arg
                    return jit_exec_arg, jit_arg_type, jit_arg_attr

                if dyn_vals:
                    jit_arg_type.extend([v.type for v in dyn_vals])
                    jit_arg_attr.extend([default_attr] * len(dyn_vals))
                    jit_exec_arg.extend(
                        _get_c_pointers_cutlass(arg) if is_host else dyn_vals
                    )
                else:
                    # If tree flatten yields empty list, treat it as a constexpr thing
                    # Like a dataclass with all fields are constexpr, or an empty tuple or list
                    jit_exec_arg = jit_arg_type = jit_arg_attr = None
        return jit_exec_arg, jit_arg_type, jit_arg_attr

    def _generate_execution_arguments_for_known_types(
        self, arg, arg_spec, arg_name, i, fop_args, iv_block_args
    ):
        ir_arg, iv_block_args = super()._generate_execution_arguments_for_known_types(
            arg, arg_spec, arg_name, i, fop_args, iv_block_args
        )
        if not ir_arg:
            # Handling DSL specific types
            if is_cute_algebra_type(arg_spec):
                n_args = len(get_mlir_types(arg))
                blk_args = fop_args[iv_block_args : iv_block_args + n_args]
                ir_arg.append(new_from_mlir_values(arg, blk_args))
                iv_block_args += n_args
            elif not hasattr(arg, "__extract_mlir_values__") and not hasattr(
                arg, "__new_from_mlir_values__"
            ):
                # Try tree_unflatten
                try:
                    dyn_vals, tree_def = tree_flatten(arg)
                    block_args = fop_args[iv_block_args : iv_block_args + len(dyn_vals)]
                    ir_arg.append(tree_unflatten(tree_def, block_args))
                    iv_block_args += len(dyn_vals)
                except DSLTreeFlattenError:
                    return ir_arg, iv_block_args

        return ir_arg, iv_block_args


# =============================================================================
# Cute DSL Class
# =============================================================================


class CuTeDSL(CutlassBaseDSL):
    """
    This is a concrete DSL subclass for the CuTe dialect.
    """

    def __init__(self):
        name = "CUTE_DSL"
        compiler_provider = compiler.Compiler(passmanager, execution_engine)
        pass_sm_arch_name = "cubin-chip"

        super().__init__(name, compiler_provider, pass_sm_arch_name, preprocess=True)


# =============================================================================
# KernelLauncher
# =============================================================================


class KernelLauncher:
    """
    This class is used to launch a kernel function.
    Usage:
        ```python
        @cute.kernel
        def kernel(arg1, arg2, ...):
            ...

        @cute.jit
        def launch_kernel():
            kernel(arg1, arg2, ...).launch(grid=[1, 1, 1], block=[1, 1, 1], ...)
            # or
            kernel(arg1, arg2, ...)(grid=[1, 1, 1], block=[1, 1, 1], ...)
        ```
    """

    def __init__(
        self,
        dsl: "CutlassBaseDSL",
        kernelGenHelper: BaseDSL._KernelGenHelper,
        funcBody,
        *func_args,
        **func_kwargs,
    ):
        self.dsl = dsl
        self.kernelGenHelper = kernelGenHelper
        self.funcBody = funcBody
        self.func_args = func_args
        self.func_kwargs = func_kwargs

        self._check_func_args(funcBody, *func_args, **func_kwargs)

    def _check_func_args(self, funcBody, *func_args, **func_kwargs):
        # Get function signature
        sig = inspect.signature(funcBody)

        # func_args and func_kwargs should match funcBody's signature,
        # no extra or missing arguments.
        try:
            sig.bind(*func_args, **func_kwargs)
        except TypeError as e:
            raise DSLRuntimeError(
                f"Failed to bind arguments to function `{funcBody.__name__}` with signature `{sig}`",
                cause=e,
            )

    def smem_usage(self) -> int:
        """
        Check smem usage for this kernel, only available after `launch`
        """
        return self.dsl._get_smem_usage()

    def launch(self, *args, **kwargs):
        self.dsl.frame = inspect.currentframe().f_back
        self.dsl._preprocess_launch_config_args(args, kwargs)
        config = self.dsl.LaunchConfig(*args, **kwargs)

        kernel_generator = self.dsl.kernel_launcher(
            requiredArgs=["config"],
            unitAttrNames=["gpu.kernel", "cute.kernel"],
            valueAttrDict=self.dsl._generate_kernel_attrs(config),
            kernelGenHelper=self.kernelGenHelper,
        )(self.funcBody)

        ret, name = kernel_generator(*self.func_args, **self.func_kwargs, config=config)
        self.dsl.kernel_symbols.append(name)
        self.dsl.frame = None
        return ret.launch_op_ret

    def __call__(self, *args, **kwargs):
        return self.launch(*args, **kwargs)


# =============================================================================
# Utils
# =============================================================================
def _filter_readonly_frozen_dataclass(
    iter_args: List[Any], items_to_filter: List[Any], full_write_args_count: int
) -> List[Any]:
    """
    Filter items based on whether corresponding iter_args are frozen dataclasses.

    This function filters items (which can be values or names) based on the same
    logic: keep items if they correspond to full-write arguments (index < full_write_args_count)
    or if the corresponding iter_arg is not a frozen dataclass.

    Args:
        iter_args: List of arguments to check for frozen dataclass status
        items_to_filter: List of items to filter (values or names)
        full_write_args_count: Number of arguments that are always written (not read-only)

    Returns:
        Filtered list of items

    Examples:
        # Filter values (original remove_read_only_frozen_dataclass behavior)
        filtered_values = _filter_readonly_frozen_dataclass(iter_args, iter_args, full_write_args_count)

        # Filter names (original filter_readonly_frozen_dataclass_names behavior)
        filtered_names = _filter_readonly_frozen_dataclass(iter_args, iter_args_names, full_write_args_count)
    """
    return [
        item
        for i, item in enumerate(items_to_filter)
        if i < full_write_args_count or not is_frozen_dataclass(iter_args[i])
    ]


def remove_read_only_frozen_dataclass(
    iter_args: List[Any], full_write_args_count: int
) -> List[Any]:
    """Filter out frozen dataclass arguments that are not full-write arguments."""
    return _filter_readonly_frozen_dataclass(
        iter_args, iter_args, full_write_args_count
    )


def filter_readonly_frozen_dataclass_names(
    iter_args: List[Any], iter_args_names: List[str], full_write_args_count: int
) -> List[str]:
    """Filter names based on whether corresponding iter_args are frozen dataclasses."""
    return _filter_readonly_frozen_dataclass(
        iter_args, iter_args_names, full_write_args_count
    )


def insert_read_only_frozen_dataclass(
    iter_args: List[Any], original_iter_args: List[Any], full_write_args_count: int
) -> List[Any]:
    """
    Insert read-only frozen dataclass arguments back into the iteration arguments.

    This function takes the new iteration arguments and the original arguments,
    and preserves frozen dataclass instances from the original arguments while
    using the new arguments for non-frozen dataclass instances.

    Args:
        iter_args: New iteration arguments to use for non-frozen dataclass instances
        original_iter_args: Original iteration arguments to preserve frozen dataclass instances from
        full_write_args_count: Number of arguments that are always written (not read-only)

    Returns:
        List of arguments with frozen dataclass instances preserved from original
    """
    # Take full-write arguments from new iter_args
    full_write_args = (
        iter_args[:full_write_args_count] if full_write_args_count > 0 else []
    )

    # Process remaining arguments: preserve frozen dataclass from original, use new for others
    remaining_original = original_iter_args[full_write_args_count:]
    remaining_new = iter_args[full_write_args_count:]

    def process_remaining_arg(original_arg, new_arg_iter):
        """Process a single remaining argument, preserving frozen dataclass if present"""
        return original_arg if is_frozen_dataclass(original_arg) else next(new_arg_iter)

    # Use zip to pair original args with new args, then map the processing function
    new_arg_iter = iter(remaining_new)
    processed_remaining = [
        process_remaining_arg(orig_arg, new_arg_iter) for orig_arg in remaining_original
    ]

    return full_write_args + processed_remaining


def unpack_to_irvalue(
    mixed_values: List[Any], body_name: str, full_write_args_count: int
) -> Tuple[List[ir.Value], PyTreeDef]:
    log().debug("===--- Values UNPack")
    for idx, packed in enumerate(mixed_values):
        log().debug("[%d]: will-unpacked: [type:%s] %s", idx, type(packed), packed)

    try:
        unpacked_values, treedef = tree_flatten(
            remove_read_only_frozen_dataclass(mixed_values, full_write_args_count)
        )
    except DSLTreeFlattenError as e:
        raise DSLRuntimeError(
            f"The '{body_name}' statement encountered a user-defined Python object, which cannot be automatically converted into an dynamic expression.",
            context={
                e.message: (
                    f"All expressions within '{body_name}' must be dynamic expressions, "
                    "mixing Python objects and dynamic expressions is not supported. "
                    "The DSL failed to convert the Python object into dynamic expressions."
                )
            },
            suggestion=(
                f"Please ensure '{e.type_str}' implements the '{DynamicExpression.__name__}' or mark with `dataclass`, "
                f"so it can be treated as a valid dynamic expression or mark '{body_name}' as a constant expression if conditions are Python objects."
            ),
        )

    log().debug("------------------ ")
    for idx, unpacked in enumerate(unpacked_values):
        log().debug("[%d]: unpacked values: %s", idx, unpacked)
    log().debug("treedef: %s", treedef)
    log().debug("------------------ ")

    return unpacked_values, treedef


def pack_from_irvalue(
    ir_values: List["ir.Value"],
    pytree_def: PyTreeDef,
    mixed_values: List[Any],
    full_write_args_count: int,
) -> List[Any]:
    """
    Packs MLIR values into a list of mixed values.
    """
    log().debug("===--- Values Pack (%d)", len(ir_values))
    for idx, value in enumerate(ir_values):
        log().debug("[%d]: will-packed: %s", idx, value)
    log().debug("treedef: %s", pytree_def)
    log().debug("------------------ ")

    unflattened = tree_unflatten(pytree_def, ir_values)
    return insert_read_only_frozen_dataclass(
        unflattened, mixed_values, full_write_args_count
    )


def to_index(value):
    """Converts a value to an index, either by casting or coercing to int."""
    if is_dynamic_expression(value):
        if isinstance(value, Numeric):
            value = value.ir_value()
        assert ir.IntegerType.isinstance(
            value.type
        ), f"expects integer type, but got {value.type}"
        res = arith.index_cast(T.index(), value)
    else:
        res = const(int(value), ty=T.index())

    return res


def _validate_iter_args_structure(iter_args, ir_values):
    """
    Validates that iter_args structure contains the same number of atomic values
    as there are IR values.

    Args:
        iter_args: Original iteration arguments, possibly nested sequences
        ir_values: Flattened MLIR values extracted from iter_args

    Returns:
        bool: True if the number of atomic values in iter_args matches
              the number of values in ir_values
    """
    # Handle non-sequence case
    if not isinstance(iter_args, (tuple, list, set)):
        return not isinstance(ir_values, (tuple, list, set)) or len(ir_values) == 1

    # If we have a sequence but ir_values isn't one, there's a mismatch
    if not isinstance(ir_values, (tuple, list, set)):
        return False

    # Count all non-sequence values recursively
    def count_values(args):
        if not isinstance(args, (tuple, list, set)):
            return 1
        else:
            return sum(count_values(arg) for arg in args)

    return count_values(iter_args) == len(ir_values)



# =============================================================================
# DSL implementation of Python Build-in Operators
# =============================================================================


def _minmax(op, *args, loc=None, ip=None):
    """Computes the minimum or maximum value from the provided arguments."""
    from ..base_dsl.typing import _binary_op, _binary_op_type_promote

    # AST Traversal doesn't support early exit in if executor
    x = None
    res = None
    if len(args) == 1:
        # Handle case for min([a, b, c, d, ..])
        if hasattr(args[0], "__iter__"):
            x = op(*tuple(args[0]))
        # Handle case for min(a)
        else:
            x = args[0]
    # Handle case for min(a, b, c, ...) and min([x, y], [b]) and min(a, (x, y, z))
    elif len(args) > 1:
        res, *xs = tuple(args)
        for x in xs:
            lhs = as_numeric(op(res, loc=loc, ip=ip))
            rhs = as_numeric(op(x, loc=loc, ip=ip))
            emitter = getattr(cutlass_arith, f"_{op.__name__}")

            lhs, rhs, res_type = _binary_op_type_promote(lhs, rhs, promote_bool=True)

            if isinstance(lhs.value, cutlass_arith.ArithValue) and isinstance(
                lhs, Integer
            ):
                lhs_val = lhs.value.with_signedness(lhs.signed)
            else:
                lhs_val = lhs.value

            if isinstance(rhs.value, cutlass_arith.ArithValue) and isinstance(
                rhs, Integer
            ):
                rhs_val = rhs.value.with_signedness(rhs.signed)
            else:
                rhs_val = rhs.value

            res = res_type(emitter(lhs_val, rhs_val), loc=loc, ip=ip)
        x = res
    else:
        raise DSLNotImplemented(f"{type(args)} is not supported")
    return x


def min(*args, loc=None, ip=None):
    """Computes the minimum value from the provided arguments.

    This function differs from Python's built-in min() in that the return type
    is determined by the static types of the inputs, not their dynamic values.

    :param args: One or more values or iterables to find the minimum of
    :type args: tuple
    :param loc: Source location for MLIR operation tracking
    :type loc: object, optional
    :param ip: Insertion point for MLIR operation
    :type ip: object, optional
    :return: The minimum value among all inputs
    :rtype: Numeric
    :raises DSLNotImplemented: If the input type is not supported

    Supports multiple calling patterns:

    - min(a): Returns a
    - min([a, b, c, ...]): Returns minimum of all elements in the iterable
    - min(a, b, c, ...): Returns minimum of all arguments
    - min([x, y], [b]): Returns minimum across all elements in all iterables
    - min(a, (x, y, z)): Returns minimum across all elements

    Examples:

    .. code-block:: python

        # Find minimum of two values
        result = min(x, y)

        # Find minimum of multiple values
        result = min(a, b, c, d)

        # Find minimum of values in a list
        values = [a, b, c, d]
        result = min(values)

        # Find minimum across mixed arguments
        result = min(x, [y, z])

    Difference from Python's built-in min():

    .. code-block:: python

        # In Python, the return type depends on the dynamic values:
        a = 5
        b = 3.14
        result = min(a, b)  # Returns 3.14 (float)

        # In this DSL implementation, the return type is determined statically:
        a = Int32(5)
        b = Float32(3.14)
        result = min(a, b)  # Return type is determined by the type of operands, not values
    """
    return _minmax(min, *args, loc=loc, ip=ip)


def max(*args, loc=None, ip=None):
    """Computes the maximum value from the provided arguments.

    This function differs from Python's built-in max() in that the return type
    is determined by the static types of the inputs, not their dynamic values.

    :param args: One or more values or iterables to find the maximum of
    :type args: tuple
    :param loc: Source location for MLIR operation tracking
    :type loc: object, optional
    :param ip: Insertion point for MLIR operation
    :type ip: object, optional
    :return: The maximum value among all inputs
    :rtype: Numeric
    :raises DSLNotImplemented: If the input type is not supported

    Supports multiple calling patterns:

    - max(a): Returns a
    - max([a, b, c, ...]): Returns maximum of all elements in the iterable
    - max(a, b, c, ...): Returns maximum of all arguments
    - max([x, y], [b]): Returns maximum across all elements in all iterables
    - max(a, (x, y, z)): Returns maximum across all elements

    Examples:

    .. code-block:: python

        # Find maximum of two values
        result = max(x, y)

        # Find maximum of multiple values
        result = max(a, b, c, d)

        # Find maximum of values in a list
        values = [a, b, c, d]
        result = max(values)

        # Find maximum across mixed arguments
        result = max(x, [y, z])

    Difference from Python's built-in max():

    .. code-block:: python

        # In Python, the return type depends on the dynamic values:
        a = 5
        b = 3.14
        result = max(a, b)  # Returns 5 (int)

        # In this DSL implementation, the return type is determined statically:
        a = Int32(5)
        b = Float32(3.14)
        result = max(a, b)  # Return type is determined by the type of operands, not values
    """
    return _minmax(max, *args, loc=loc, ip=ip)


def and_(*args, loc=None, ip=None):
    """AND operation for value in DSL numeric types.

    :param *args: One or more numeric values to AND together
    :type *args: Numeric
    :param loc: Source location for MLIR operation tracking
    :type loc: object, optional
    :param ip: Insertion point for MLIR operation
    :type ip: object, optional
    :return: The result of the logical AND operation
    :rtype: Numeric
    :raises ValueError: If no arguments are provided

    Supports multiple calling patterns:

    - and_(a): Returns a
    - and_(a, b, c, ...): if a is truthy, returns and_(b, c, ...), otherwise returns a

    All arguments must be of the same type.

    Examples:

    .. code-block:: python

        # In Python, 'and' returns the second operand if the first is truthy,
        # otherwise it returns the first operand
        a = 5
        b = 3
        result = a and b  # Returns 3

        # In this DSL implementation, the behavior is similar but works with DSL types
        a = Int32(5)
        b = Int32(3)
        result = and_(a, b)  # Returns b
    """
    if len(args) == 0:
        raise ValueError("and_() requires at least one argument")

    if len(args) == 1:
        return args[0]

    def and_op(lhs, rhs):
        if not isinstance(lhs, (Numeric, cutlass_arith.ArithValue, int, float, bool)):
            raise DSLNotImplemented(f"{type(lhs)} is not supported")
        elif isinstance(lhs, (int, float, bool)) and isinstance(
            rhs, (int, float, bool)
        ):
            return lhs and rhs
        else:
            return as_numeric(lhs).__dsl_and__(as_numeric(rhs))

    return functools.reduce(and_op, args[1:], args[0])


def or_(*args, loc=None, ip=None):
    """Logical OR operation for DSL numeric types.

    :param *args: One or more numeric values to OR together
    :type *args: Numeric
    :param loc: Source location for MLIR operation tracking
    :type loc: object, optional
    :param ip: Insertion point for MLIR operation
    :type ip: object, optional
    :return: The result of the logical OR operation
    :rtype: Numeric
    :raises ValueError: If no arguments are provided

    Supports multiple calling patterns:

    - or_(a): Returns a
    - or_(a, b, c, ...): if a is truthy, returns a, otherwise returns or_(b, c, ...)

    Examples:

    .. code-block:: python

        # In Python, 'or' returns the first operand if it's truthy,
        # otherwise it returns the second operand
        a = 5
        b = 3
        result = a or b  # Returns 5

        # In this DSL implementation, the behavior is similar but works with DSL types
        a = Int32(5)
        b = Int32(3)
        result = or_(a, b)  # Returns a
    """
    if len(args) == 0:
        raise ValueError("or_() requires at least one argument")

    if len(args) == 1:
        return args[0]

    def or_op(lhs, rhs):
        if not isinstance(lhs, (Numeric, cutlass_arith.ArithValue, int, float, bool)):
            raise DSLNotImplemented(f"{type(lhs)} is not supported")
        elif isinstance(lhs, (int, float, bool)) and isinstance(
            rhs, (int, float, bool)
        ):
            return lhs or rhs
        else:
            return as_numeric(lhs).__dsl_or__(as_numeric(rhs))

    return functools.reduce(or_op, args[1:], args[0])


def all_(iterable):
    """Logical AND operation for all elements in an iterable.

    Returns True if all elements in the iterable are truthy, otherwise False.
    This is the DSL equivalent of Python's built-in all() function.

    :param iterable: An iterable containing values to check
    :type iterable: Iterable
    :return: True if all elements are truthy, False otherwise
    :rtype: Boolean

    Examples:

    .. code-block:: python

        # Check if all values are non-zero
        values = [Int32(1), Int32(2), Int32(3)]
        result = all_(values)  # Returns True

        # Check if all conditions are met
        conditions = [a > 0, b < 10, c != 0]
        result = all_(conditions)  # Returns True if all conditions are met
    """
    bool_iterable = [Boolean(i) for i in iterable]
    return functools.reduce(
        lambda lhs, rhs: lhs.__dsl_and__(rhs) if hasattr(lhs, "__dsl_and__") else lhs,
        bool_iterable,
        Boolean(True),
    )


def any_(iterable):
    """Logical OR operation for any element in an iterable.

    Returns True if any element in the iterable is truthy, otherwise False.
    This is the DSL equivalent of Python's built-in any() function.

    :param iterable: An iterable containing values to check
    :type iterable: Iterable
    :return: True if any element is truthy, False otherwise
    :rtype: Boolean

    Examples:

    .. code-block:: python

        # Check if any value is non-zero
        values = [Int32(0), Int32(0), Int32(3)]
        result = any_(values)  # Returns True

        # Check if any condition is met
        conditions = [a > 10, b < 0, c != 0]
        result = any_(conditions)  # Returns True if any condition is met
    """
    bool_iterable = [Boolean(i) for i in iterable]
    return functools.reduce(
        lambda lhs, rhs: lhs.__dsl_or__(rhs) if hasattr(lhs, "__dsl_or__") else lhs,
        bool_iterable,
        Boolean(False),
    )


# =============================================================================
# Conditional Expression
# =============================================================================


def select_(cond, if_value, else_value):
    def _as_scalar(value):
        if isinstance(value, list):
            if len(value) == 1:
                return value[0]
            else:
                raise DSLRuntimeError(
                    "Conditional expression must have exactly one value in all expressions"
                )
        return value

    if not is_dynamic_expression(cond):
        raise DSLRuntimeError("Conditional expression must be dynamic")

    # Extract MLIR values
    cond = extract_mlir_values(cond)
    if is_dynamic_expression(if_value):
        if_value = extract_mlir_values(if_value)
    else:
        if_value = const(if_value)
    if is_dynamic_expression(else_value):
        else_value = extract_mlir_values(else_value)
    else:
        else_value = const(else_value)

    return arith.SelectOp(
        _as_scalar(cond), _as_scalar(if_value), _as_scalar(else_value)
    ).result


# =============================================================================
# Terminator
# =============================================================================


def yield_out(args=[], loc=None, ip=None):
    """
    Generate a yield operation. It it used to return values from a loop, if-else, or while region.
    """
    scf.yield_(extract_mlir_values(args), loc=loc, ip=ip)


# =============================================================================
# For Loop
# =============================================================================


class LoopUnroll(ir.Attribute):
    def __init__(self, **kwargs):
        valid_keys = set(["count", "full"])
        def to_mlir_attr(val):
            if isinstance(val, bool):
                return "true" if val else "false"
            elif isinstance(val, int):
                return f"{val} : i32"
            else:
                raise DSLNotImplemented(f"{type(val)} is not supported")

        cfg = {key: to_mlir_attr(kwargs[key]) for key in valid_keys if key in kwargs}
        if kwargs.get("count", None) == 1:
            cfg["disable"] = "true"

        unroll = "<" + ", ".join(f"{key} = {value}" for key, value in cfg.items()) + ">"

        super().__init__(
            ir.Attribute.parse(f"#llvm.loop_annotation<unroll = {unroll}>")
        )


def for_generate(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[ir.Value]] = None,
    *,
    unroll: LoopUnroll = None,
    prefetch_stages=None,
    loc=None,
    ip=None,
):
    """
    scf.for with yield support
    """

    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0
    start = const(start)
    params = [start, stop, step]
    for i, p in enumerate(params):
        if isinstance(p, int):
            p = const(p)
        elif isinstance(p, float):
            raise DSLRuntimeError(f"{p=} must be int.")
        elif isinstance(p, Integer):
            p = p.ir_value()
        params[i] = p

    start, stop, step = params

    def _createI32Attr(value):
        if not isinstance(value, int):
            raise DSLRuntimeError(f"value must be int.")
        return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)

    ir_iter_args = extract_mlir_values(iter_args) if iter_args is not None else None
    if not _validate_iter_args_structure(iter_args, ir_iter_args):
        raise DSLRuntimeError("iter_args: Elements should be extractable as ir.Value.")
    for_op = scf.ForOp(start, stop, step, ir_iter_args, loc=loc, ip=ip)
    if unroll is not None:
        for_op.attributes["loop_annotation"] = unroll

    if prefetch_stages is not None:
        for_op.attributes["cutlass.pipelining"] = _createI32Attr(prefetch_stages)

    iv = for_op.induction_variable
    new_results = new_from_mlir_values(iter_args, for_op.results)
    new_iter_args = new_from_mlir_values(iter_args, for_op.inner_iter_args)
    new_iter_args = () if new_iter_args is None else tuple(new_iter_args)

    with ir.InsertionPoint(for_op.body):
        if len(new_iter_args) > 1:
            yield iv, new_iter_args, new_results
        elif len(new_iter_args) == 1:
            yield iv, new_iter_args[0], new_results[0]
        else:
            yield iv


# =============================================================================
# Logical Operators
# =============================================================================


def not_(lhs: Union[ir.Value, bool], *, loc=None, ip=None):
    """
    Logical Not
    """
    res = None
    # Handle Python bool first to prevent infinite recursion
    if type(lhs) == bool:
        res = lhs ^ True
    elif hasattr(lhs, "__dsl_not__"):
        res = lhs.__dsl_not__(loc=loc, ip=ip)
    elif is_dynamic_expression(lhs):
        # If lhs is MLIR value, compute not using xor
        res = arith.XOrIOp(lhs, const(1, lhs.type)).result
    else:
        res = bool(lhs) ^ True

    return res


# =============================================================================
# If/Else
# =============================================================================


def if_generate(
    cond: Boolean,
    then_body: Callable,
    else_body: Optional[Callable] = None,
    input_args: List[DslType] = None,
    return_types: List[DslType] = None,
    *,
    loc=None,
    ip=None,
) -> List:
    """
    Generate an IfOp with optional else branch and return values.

    Args:
        cond: The condition expression
        then_body: Function to execute in then branch
        else_body: Optional function to execute in else branch
        input_args: Arguments to pass to branch bodies
        return_types: Expected return types for the operation
        loc: Optional location information
        ip: Optional insertion point

    Returns:
        List of DSL typed results
    """
    input_args = input_args or []
    mlir_return_types = []

    # Validate and collect MLIR return types (if provided).
    if return_types is not None:
        for t in return_types:
            if not isinstance(t, DslType):
                raise DSLRuntimeError(f"{t=} must be a DslType.")
            mlir_return_types.append(t.mlir_type)

    # Determine whether there's an else branch.
    has_else = else_body is not None

    # Create the IfOp.
    if_op = scf.IfOp(
        Boolean(cond).ir_value(), mlir_return_types, hasElse=has_else, loc=loc, ip=ip
    )

    def _execute_and_yield_out(body, input_args):
        yield_vals = body(*input_args)
        if return_types is not None:
            if not isinstance(yield_vals, Iterable):
                # body only return single element
                yield_vals = [yield_vals]

            yield_vals = [t(r) for t, r in zip(return_types, yield_vals)]
        yield_out(yield_vals)

    # Generate the body for 'then'.
    with ir.InsertionPoint(if_op.then_block):
        _execute_and_yield_out(then_body, input_args)

    # Generate the body for 'else' if provided.
    if has_else:
        with ir.InsertionPoint(if_op.else_block):
            _execute_and_yield_out(else_body, input_args)

    # Collect MLIR results.
    mlir_results = _get_op_result_or_op_results(if_op)

    if not isinstance(mlir_results, list):
        mlir_results = [mlir_results]

    # Wrap the results with their DSL types.
    if return_types is None:
        return []

    vals = [t(r) for t, r in zip(return_types, mlir_results)]

    if len(vals) == 1:
        return vals[0]

    return vals


# =============================================================================
# While Loop
# =============================================================================


class WhileLoopContext:
    """
    Context manager for a dynamic while loop.
    """

    def __init__(
        self,
        inputs: Sequence[Union[ir.Value, Numeric]],
        condition: Callable[[Sequence[ir.Value]], ir.Value],
        *,
        loc=None,
        ip=None,
    ):
        # Keep original inputs and allow recover original type information
        self.inputs = inputs

        self.input_ir_values = extract_mlir_values(inputs)

        if not _validate_iter_args_structure(inputs, self.input_ir_values):
            raise DSLRuntimeError("inputs: Elements should be extractable as ir.Value.")

        self.condition = condition
        self.input_ir_types = [i.type for i in self.input_ir_values]
        self.while_op = scf.WhileOp(
            self.input_ir_types, self.input_ir_values, loc=loc, ip=ip
        )

        self.before_region = self.while_op.before
        self.after_region = self.while_op.after

        self.before_region.blocks.append(*self.input_ir_types)
        self.before_block = self.before_region.blocks[0]

        self.after_region.blocks.append(*self.input_ir_types)
        self.after_block = self.after_region.blocks[0]

    def __enter__(self):
        with ir.InsertionPoint(self.before_block):
            args = new_from_mlir_values(self.inputs, self.before_block.arguments)
            cond = self.condition(*args)
            cond_ir_val = extract_mlir_values(cond)
            scf.ConditionOp(cond_ir_val[0], [*self.before_block.arguments])
        self.ipoint_op = ir.InsertionPoint(self.after_block)
        self.ipoint_op.__enter__()
        return new_from_mlir_values(self.inputs, self.after_block.arguments)

    def __exit__(self, exc_type, exc_value, traceback):
        self.ipoint_op.__exit__(exc_type, exc_value, traceback)

    @property
    def results(self):
        return new_from_mlir_values(self.inputs, self.while_op.results_)


def while_generate(
    inputs: Sequence[Union[ir.Value, Numeric]],
    condition: Callable[[Sequence[Union[ir.Value, Numeric]]], Union[ir.Value, Numeric]],
    *,
    loc=None,
    ip=None,
) -> WhileLoopContext:
    """
    Generate a WhileLoopContext for a dynamic loop.
    """
    return WhileLoopContext(inputs, condition, loc=loc, ip=ip)


def equal(lhs, rhs):
    if not is_dynamic_expression(lhs) and not is_dynamic_expression(rhs):
        return lhs == rhs

    # Both sequence
    if isinstance(lhs, Sequence) and isinstance(rhs, Sequence):
        # Short-circuit for unequal length
        if len(lhs) != len(rhs):
            return False
        return all_(equal(l, r) for l, r in zip(lhs, rhs))
    return lhs == rhs


def not_equal(lhs, rhs):
    if not is_dynamic_expression(lhs) and not is_dynamic_expression(rhs):
        return lhs != rhs

    # Both sequence
    if isinstance(lhs, Sequence) and isinstance(rhs, Sequence):
        # Short-circuit for unequal length
        if len(lhs) != len(rhs):
            return True
        return any_(not_equal(l, r) for l, r in zip(lhs, rhs))

    if hasattr(lhs, "__ne__"):
        return lhs != rhs
    elif hasattr(rhs, "__ne__"):
        return rhs != lhs
    else:
        return not_(equal(lhs, rhs))


def in_(lhs, rhs):
    if not is_dynamic_expression(lhs) and not is_dynamic_expression(rhs):
        return lhs in rhs

    if not isinstance(rhs, Sequence):
        raise DSLRuntimeError(
            f"'in' not supported between instances of {type(lhs)} and {type(rhs)}"
        )

    return any_(equal(lhs, r) for r in rhs)


def _lte_gte(lhs, rhs, op):
    def native_lte_gte(lhs, rhs, op):
        match op:
            case "<":
                return lhs < rhs
            case "<=":
                if hasattr(lhs, "__le__"):
                    return lhs <= rhs
                else:
                    return not_(lhs > rhs)
            case ">":
                return lhs > rhs
            case ">=":
                if hasattr(lhs, "__ge__"):
                    return lhs >= rhs
                else:
                    return not_(lhs < rhs)
            case _:
                raise DSLRuntimeError(f"Unsupported comparison operator: {op}")

    if not is_dynamic_expression(lhs) and not is_dynamic_expression(rhs):
        return native_lte_gte(lhs, rhs, op)

    # Both sequence, comparisons other than == and != do not allow mixing different types of sequences
    if (
        isinstance(lhs, Sequence)
        and isinstance(rhs, Sequence)
        and type(lhs) == type(rhs)
    ):
        unequal_found = False
        comp_results = []
        mask = []
        for l, r in zip(lhs, rhs):
            is_equal = equal(l, r)
            mask.append(not_(or_(is_equal, unequal_found)))
            unequal_found = not_(is_equal)
            comp_results.append(_lte_gte(l, r, op))

        result = any_(and_(r, m) for r, m in zip(comp_results, mask))

        if len(lhs) != len(rhs):
            # Ref https://docs.python.org/3/tutorial/datastructures.html#comparing-sequences-and-other-types
            # If one sequence is an initial sub-sequence of the other, the shorter sequence is the smaller (lesser) one
            has_valid_mask = any_(mask)
            match op:
                case "<":
                    length_result = len(lhs) < len(rhs)
                case ">":
                    length_result = len(lhs) > len(rhs)
                case "<=":
                    length_result = len(lhs) <= len(rhs)
                case ">=":
                    length_result = len(lhs) >= len(rhs)
            if type(has_valid_mask) == bool:
                return result if has_valid_mask else length_result
            else:
                return select_(has_valid_mask, result, length_result)
        else:
            if op in {"<=", ">="}:
                # If no unequal, return True
                return select_(unequal_found, result, True)
            else:
                return result
    else:
        return native_lte_gte(lhs, rhs, op)


def greater_than(lhs, rhs):
    return _lte_gte(lhs, rhs, ">")


def greater_equal(lhs, rhs):
    return _lte_gte(lhs, rhs, ">=")


def less_than(lhs, rhs):
    return _lte_gte(lhs, rhs, "<")


def less_equal(lhs, rhs):
    return _lte_gte(lhs, rhs, "<=")


def _compare_dispatch(lhs, rhs, op):
    """
    Dispatches the comparison operation between lhs and rhs based on the given operator.

    :param lhs: The left-hand side operand for the comparison.
    :param rhs: The right-hand side operand for the comparison.
    :param op: The comparison operator as a string. Supported operators are:
        - "is", "is not": Python identity comparisons.
        - "in", "not in": Membership tests.
        - "==", "!=": Equality and inequality.
        - "<", ">", "<=", ">=": Relational comparisons.
    :return: The result of the comparison, which may be a boolean or a DSL-specific type.
    :raises DSLRuntimeError: If the operator is not supported.
    """
    match op:
        # 'is' and 'is not' are pure python operators
        case "is":
            return lhs is rhs
        case "is not":
            return lhs is not rhs
        case "in":
            return in_(lhs, rhs)
        case "not in":
            return not_(in_(lhs, rhs))
        case "==":
            return equal(lhs, rhs)
        case "!=":
            return not_equal(lhs, rhs)
        case "<":
            return less_than(lhs, rhs)
        case ">":
            return greater_than(lhs, rhs)
        case ">=":
            return greater_equal(lhs, rhs)
        case "<=":
            return less_equal(lhs, rhs)
        case _:
            raise DSLRuntimeError(f"Unsupported comparison operator: {op}")


def _compare_executor(left, comparators, ops):
    # Fast path for single comparison
    if len(comparators) == 1:
        return _compare_dispatch(left, comparators[0], ops[0])

    # Chain comparison, dispatch in a loop
    result = True
    current = left
    for comparator, op in zip(comparators, ops):
        cmp_result = _compare_dispatch(current, comparator, op)
        result = and_(result, cmp_result)
        current = comparator

    return result


def _builtin_redirector(fcn):
    if fcn == builtins.max:
        return max
    elif fcn == builtins.min:
        return min
    elif fcn == builtins.any:
        return any_
    elif fcn == builtins.all:
        return all_
    else:
        raise DSLRuntimeError(f"Unsupported built-in function: {fcn}")


# =============================================================================
# Set the AST decorator
# =============================================================================

# Set the DSL specific functions
executor.set_functions(
    is_dynamic_expression=is_dynamic_expression,
    loop_execute_range_dynamic=_loop_execute_range_dynamic,
    if_dynamic=_if_execute_dynamic,
    while_dynamic=_while_execute_dynamic,
    compare_executor=_compare_executor,
    any_executor=any_,
    all_executor=all_,
    builtin_redirector=_builtin_redirector,
)
