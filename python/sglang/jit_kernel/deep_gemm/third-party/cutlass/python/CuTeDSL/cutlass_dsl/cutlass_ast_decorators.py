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

from typing import List, Tuple
from types import NoneType
from cutlass._mlir import ir
from cutlass._mlir.dialects import scf, arith
from cutlass._mlir.extras import types as T
from collections.abc import Sequence

from ..base_dsl.dsl import is_dynamic_expression
from ..base_dsl.ast_helpers import *
from ..base_dsl.utils.logger import log
from ..base_dsl import typing as t
from ..base_dsl.typing import (
    Int32,
    Float32,
    Boolean,
    Numeric,
    get_mlir_types,
    as_numeric,
)
from . import cutlass as cutlass_dsl
from .tree_utils import PyTreeDef, check_tree_equal

# =============================================================================
# AST Helpers
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


class ScfGenerator:
    """
    Encapsulates common scf dialect functionality: pack, unpack, and SCF execution.
    """

    def __init__(self):
        pass

    @staticmethod
    def _normalize_region_result_to_list(region_result: Any) -> List[Any]:
        """
        Convert region_result to a list if it is not already a list
        If region_result is a list, return it as is.
        If region_result is None, return an empty list.
        If region_result is not a list, return a list containing region_result as the only element.
        """
        if region_result is None:
            region_result_list = []
        elif not isinstance(region_result, list):
            region_result_list = [region_result]
        else:
            region_result_list = region_result
        return region_result_list

    @staticmethod
    def _check_region_result(original_value, region_value, arg_name, op_type_name):
        """
        Validate that a region result maintains the same type as the original value.

        This method checks for type consistency between the original value passed to a dynamic
        SCF operation (like for, if, while) and the value returned from the operation's region.

        Args:
            original_value: The value before entering the SCF operation region
            region_value: The value returned from the SCF operation region
            arg_name: Name of the argument being checked (for error reporting)
            op_type_name: Type of SCF operation (e.g., 'for', 'if', 'while') for error reporting

        Raises:
            DSLRuntimeError: If the region value has a different type than the original value.
                The error includes suggestions for using compile-time control flow instead.

        Note:
            This method performs relaxed type checking that allows inheritance relationships.
            For example, a child class can be returned where a parent class was expected.
            However, fundamental type changes (like None to non-None, different sequence types,
            or different numeric types) are not allowed in dynamic SCF operations.
        """

        def get_type_name(value):
            if isinstance(value, NoneType):
                return "None"
            elif isinstance(value, Sequence):
                return f"{type(value).__name__}<{len(value)}>"
            else:
                return type(value).__name__

        # Check for type mismatches
        type_mismatch = False
        old_type_name = None
        new_type_name = None

        # Handle None type changes
        if isinstance(original_value, NoneType) != isinstance(region_value, NoneType):
            type_mismatch = True
            old_type_name = get_type_name(original_value)
            new_type_name = get_type_name(region_value)
        # Handle sequence type/length changes
        elif isinstance(original_value, Sequence) and isinstance(
            region_value, Sequence
        ):
            if type(original_value) != type(region_value) or len(original_value) != len(
                region_value
            ):
                type_mismatch = True
                old_type_name = get_type_name(original_value)
                new_type_name = get_type_name(region_value)
        # Handle numeric type changes
        elif isinstance(
            original_value, (Numeric, ArithValue, ir.Value, int, float, bool)
        ) or isinstance(
            region_value, (Numeric, ArithValue, ir.Value, int, float, bool)
        ):
            try:
                original_numeric = as_numeric(original_value)
                region_numeric = as_numeric(region_value)
                if original_numeric.dtype != region_numeric.dtype:
                    type_mismatch = True
                    old_type_name = original_numeric.dtype.__name__
                    new_type_name = region_numeric.dtype.__name__
            except Exception:
                pass
        # Handle general type changes (relaxed for inheritance)
        elif type(original_value) != type(region_value):
            old_type = type(original_value)
            new_type = type(region_value)
            if not (issubclass(old_type, new_type) or issubclass(new_type, old_type)):
                type_mismatch = True
                old_type_name = old_type.__name__
                new_type_name = new_type.__name__

        if type_mismatch:
            raise DSLRuntimeError(
                f"`{arg_name}` is {old_type_name} prior to this `{op_type_name}`, "
                f"and update to {new_type_name} inside of this `{op_type_name}` is not supported.",
                suggestion=(
                    f"Please avoid changing type inside a dynamic `{op_type_name}`, "
                    f"or change to compile-time control flow by marking this `{op_type_name}` with "
                    f"`{'range_constexpr' if op_type_name == 'for' else 'const_expr'}`."
                ),
            )

    def scf_execute_dynamic(
        self,
        op_type_name: str,
        mix_iter_args: List[Any],
        full_write_args_count: int,
        mix_iter_arg_names: List[str],
        create_op_func: Callable[[List[ir.Value]], ir.Operation],
        region_builders: List[
            Callable[
                [
                    "ir.Operation",
                    List["ir.Value"],  # block_args
                    List["ir.Value"],  # dyn_yield_ops
                    PyTreeDef,
                    List[Any],
                    int,
                ],
                Any,
            ]
        ],
        # block_term_op_builder[region_builder] = scf_op_builder
        # e.g. scf.ConditionOp for while loop
        block_term_op_builder: Dict[Callable, Callable] = {},
    ) -> Any:
        # 1) Unpack
        ir_values, pytree_def = cutlass_dsl.unpack_to_irvalue(
            mix_iter_args, op_type_name, full_write_args_count
        )
        # 2) Create the SCF op
        op = create_op_func(ir_values)
        log().debug("Generated scf.%s \n[%s]", op_type_name, op)

        # 3) Build the regions
        for i, builder in enumerate(region_builders):
            region = op.regions[i]
            block = region.blocks[0]
            with ir.InsertionPoint(block):
                block_args = list(block.arguments)
                region_result = builder(
                    op,
                    block_args,
                    ir_values,
                    pytree_def,
                    mix_iter_args,
                    full_write_args_count,
                )

                # Use custom terminator if provided for this builder, otherwise use default YieldOp
                if builder in block_term_op_builder:
                    # Use the provided terminator generator
                    block_term_op_builder[builder](region_result, full_write_args_count)
                else:
                    # Normalize region_result
                    region_result_list = ScfGenerator._normalize_region_result_to_list(
                        region_result
                    )
                    # For standard yield op, check result
                    for arg, result, name in zip(
                        mix_iter_args,
                        region_result_list,
                        mix_iter_arg_names,
                    ):
                        ScfGenerator._check_region_result(
                            arg, result, name, op_type_name
                        )

                    # Default behavior - generate YieldOp
                    region_values, yield_pytree_def = cutlass_dsl.unpack_to_irvalue(
                        region_result_list, op_type_name, full_write_args_count
                    )

                    mismatch = check_tree_equal(pytree_def, yield_pytree_def)
                    if mismatch != -1:
                        # Get arg name
                        filterd_arg_names = (
                            cutlass_dsl.filter_readonly_frozen_dataclass_names(
                                mix_iter_args, mix_iter_arg_names, full_write_args_count
                            )
                        )

                        raise DSLRuntimeError(
                            f"`{filterd_arg_names[mismatch]}` is structured different after this `{op_type_name}`.",
                            suggestion=(
                                f"Please avoid changing type structure inside a dynamic `{op_type_name}`, "
                                f"or change to compile-time control flow by marking this `{op_type_name}` with "
                                f"`{'range_constexpr' if op_type_name == 'for' else 'const_expr'}`."
                            ),
                        )

                    scf.YieldOp(region_values)

        log().debug("Completed scf.%s \n[%s]", op_type_name, op)

        # 4) Pack final results
        final_results = cutlass_dsl.pack_from_irvalue(
            op.results, pytree_def, mix_iter_args, full_write_args_count
        )

        # 5) Return in a nice pattern
        if not final_results:
            return
        if len(final_results) == 1:
            return final_results[0]
        return final_results


def _attr_const_check(attr, expected_type, attr_name):
    # Use strict type equality to prevent `bool` being accepted where `int` is required.
    if is_dynamic_expression(attr) or type(attr) is not expected_type:
        raise DSLRuntimeError(
            f"loop attribute `{attr_name}` must be a Python value of type `{expected_type.__name__}`, got `{type(attr).__name__}`."
        )


def _loop_execute_range_dynamic(
    func: Callable,
    start: Any,
    stop: Any,
    step: Any,
    mix_iter_args: List[Any] = [],
    full_write_args_count: int = 0,
    mix_iter_arg_names: List[str] = [],
    unroll: int = -1,
    unroll_full: bool = False,
    prefetch_stages: int = None,
):
    """
    Example: build an scf.for with optional unroll, using our universal helper.
    """
    scf_gen = ScfGenerator()

    def create_for_op(dyn_yield_ops: List[ir.Value]):
        for d in dyn_yield_ops:
            if not isinstance(d, ir.Value):
                raise DSLRuntimeError(
                    f"Invalid dyn_yield_ops: {dyn_yield_ops} \n\tExpected ir.Value, got {type(d)}"
                )

        # Convert Python ints or values to IR constants if needed
        start_ = t.as_numeric(start)
        stop_ = t.as_numeric(stop)
        step_ = t.as_numeric(step)
        assert start_ is not t.Int32, "Start is required for scf.for"
        assert stop_ is not t.Int32, "Stop is required for scf.for"
        assert step_ is not t.Int32, "Step is required for scf.for"
        start_ = start_.ir_value()
        stop_ = stop_.ir_value()
        step_ = step_.ir_value()

        # Attributes must be pure Python value, add a check
        _attr_const_check(unroll, int, "unroll")
        _attr_const_check(unroll_full, bool, "unroll_full")

        # Possibly attach unroll attributes
        unroll_attr = None
        if unroll_full:
            unroll_attr = LoopUnroll(full=True)
        elif unroll != -1:
            unroll_attr = LoopUnroll(count=unroll)
        log().debug("Unroll attribute: %s", unroll_attr)

        prefetch_stages_attr = None
        if prefetch_stages is not None:
            _attr_const_check(prefetch_stages, int, "prefetch_stages")
            if prefetch_stages >= 0:
                prefetch_stages_attr = ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(32), prefetch_stages
                )
            else:
                raise DSLRuntimeError(
                    f"loop attribute `prefetch_stages` must be non-negative, got `{prefetch_stages}`."
                )
        log().debug("prefetch_stages attribute: %s", prefetch_stages_attr)

        log().debug(
            "Creating scf.ForOp \n\t\tstart=%s: type : %s\n\t\tstop=%s: type : %s\n\t\tstep=%s: type : %s",
            start_,
            type(start_),
            stop_,
            type(stop_),
            step_,
            type(step_),
        )
        # Create scf.ForOp, passing iteration args if any
        try:
            if not dyn_yield_ops:
                for_op = scf.ForOp(start_, stop_, step_)
            else:
                for_op = scf.ForOp(start_, stop_, step_, list(dyn_yield_ops))
        except Exception as e:
            yield_ops = "\n".join(
                f"\t\t{i} => {d} : type : {type(d)}"
                for i, d in enumerate(dyn_yield_ops)
            )
            raise DSLRuntimeError(
                f"Failed to create scf.ForOp \n\t\tstart={start_}: type : {type(start_)}"
                f"\n\t\tstop={stop_}: type : {type(stop_)}\n\t\tstep={step_}: type : {type(step_)}"
                f", \n\tdyn_yield_ops:\n{yield_ops}"
            ) from e

        if unroll_attr is not None:
            for_op.attributes["loop_annotation"] = unroll_attr

        if prefetch_stages_attr is not None:
            for_op.attributes["cutlass.pipelining"] = prefetch_stages_attr

        return for_op

    def for_body_builder(
        op,
        block_args,
        _,
        pytree_def,
        mix_iter_args,
        full_write_args_count,
    ):
        # scf.ForOp block_args are typically [induction_var, iter_args...]
        # But MLIR also gives you op.induction_variable
        iv = t.as_numeric(op.induction_variable)
        log().debug(
            "For body builder: %s block_args: %s full_write_args_count: %s",
            iv,
            block_args,
            full_write_args_count,
        )
        # block_args[1:] are iteration variables
        func_args = []
        func_args.extend(
            cutlass_dsl.pack_from_irvalue(
                block_args[1:], pytree_def, mix_iter_args, full_write_args_count
            )
        )
        if not func_args:
            # No iteration arguments, or only the induction var
            func(iv)
            return []  # yield nothing
        else:
            updated_func_args = func(iv, *func_args)
            return updated_func_args

    # Now call the universal SCF executor with a single region builder
    return scf_gen.scf_execute_dynamic(
        op_type_name="for",
        mix_iter_args=mix_iter_args,
        full_write_args_count=full_write_args_count,
        mix_iter_arg_names=mix_iter_arg_names,
        create_op_func=create_for_op,
        region_builders=[for_body_builder],
    )


def _if_execute_dynamic(
    pred: "ir.Value",
    then_block: Callable,
    else_block: Callable = None,
    mix_yield_args: List[Any] = [],
    full_write_args_count: int = 0,
    mix_yield_arg_names: List[str] = [],
    if_constexpr=None,  # ignoring for brevity
):
    """
    Build an scf.if with optional else, using our universal helper.
    """
    scf_gen = ScfGenerator()

    def create_if_op(dyn_yield_ops: List[ir.Value]):
        # Assume final result types match the dynamic yields
        result_types = [arg.type for arg in dyn_yield_ops]

        pred_ = Boolean(pred)

        try:
            if_op = scf.IfOp(
                pred_.ir_value(),
                hasElse=(else_block is not None),
                results_=result_types,
            )
        except Exception as e:
            raise DSLRuntimeError(
                f"Failed to create scf.IfOp \n\t\tpred={pred_}: type : {type(pred_)}"
            ) from e
        return if_op

    def then_builder(
        if_op,
        _,
        dyn_yield_ops,
        pytree_def,
        mix_iter_args,
        full_write_args_count,
    ):
        flat_args = []
        flat_args.extend(
            cutlass_dsl.pack_from_irvalue(
                dyn_yield_ops, pytree_def, mix_iter_args, full_write_args_count
            )
        )
        return then_block(*flat_args)

    region_builders = [then_builder]

    if else_block is not None:

        def else_builder(
            if_op,
            _,
            dyn_yield_ops,
            pytree_def,
            mix_iter_args,
            full_write_args_count,
        ):
            flat_args = []
            flat_args.extend(
                cutlass_dsl.pack_from_irvalue(
                    dyn_yield_ops, pytree_def, mix_iter_args, full_write_args_count
                )
            )
            return else_block(*flat_args)

        region_builders.append(else_builder)

    return scf_gen.scf_execute_dynamic(
        op_type_name="if",
        mix_iter_args=mix_yield_args,
        full_write_args_count=full_write_args_count,
        mix_iter_arg_names=mix_yield_arg_names,
        create_op_func=create_if_op,
        region_builders=region_builders,
    )


def _while_execute_dynamic(
    while_before_block: Callable,
    while_after_block: Callable = None,
    write_args=[],
    full_write_args_count=0,
    write_args_names=[],
):
    """
    Create and return an SCF WhileOp for dynamic loops.
    Generate the dynamic loop body using SCF WhileOp.

    Args:
        while_before_block: Function that returns (condition, updated_values)
        while_after_block: Function that returns updated values
        write_args: Values that are updated in the loop

    See create_while_function in ast_preprocessor.py for details on the input structure.
    """
    log().debug("_while_execute_dynamic")
    while_op_type_name = "while"
    scf_gen = ScfGenerator()

    def create_while_op(dyn_yield_ops: List[ir.Value]):
        # Create the while operation with the types from yield_args
        result_types = [arg.type for arg in dyn_yield_ops]
        try:
            while_op = scf.WhileOp(result_types, dyn_yield_ops)
            while_op.before.blocks.append(*result_types)
            while_op.after.blocks.append(*result_types)
            log().debug("[%s]", while_op)
            return while_op
        except Exception as e:
            yield_ops = "\n".join(
                f"\t\t{i} => {d} : type : {type(d)}"
                for i, d in enumerate(dyn_yield_ops)
            )
            raise DSLRuntimeError(
                f"Failed to create scf.WhileOp with yield_ops:\n{yield_ops}"
            ) from e

    def before_block_builder(
        op,
        block_args,
        _,
        pytree_def,
        mix_iter_args,
        full_write_args_count,
    ):
        # Build the before (condition) block
        flat_args = []
        flat_args.extend(
            cutlass_dsl.pack_from_irvalue(
                block_args, pytree_def, mix_iter_args, full_write_args_count
            )
        )

        log().debug("before block args: %s", flat_args)

        cond, before_results = while_before_block(*flat_args)

        if not isinstance(before_results, (list, ir.OpResultList)):
            before_results = [before_results]

        log().debug("cond [%s]", cond)
        log().debug(
            "before_results [%s]",
            before_results,
        )

        return cond, before_results

    def before_block_terminator(cond_and_results, full_write_args_count):
        # Generate a condition op instead of yield op
        cond = cond_and_results[0]
        before_result_list = ScfGenerator._normalize_region_result_to_list(
            cond_and_results[1]
        )
        ir_cond = as_numeric(cond).ir_value()
        ir_results_list, pytree_def = cutlass_dsl.unpack_to_irvalue(
            before_result_list, while_op_type_name, full_write_args_count
        )
        log().debug(
            "creating scf.ConditionOp with [%s], [%s]",
            ir_cond,
            ir_results_list,
        )
        scf.ConditionOp(ir_cond, ir_results_list)

    def after_block_builder(
        op,
        block_args,
        _,
        pytree_def,
        mix_iter_args,
        full_write_args_count,
    ):
        # Build the after (body) block
        flat_args = []
        flat_args.extend(
            cutlass_dsl.pack_from_irvalue(
                block_args, pytree_def, mix_iter_args, full_write_args_count
            )
        )

        log().debug("after block args: %s", flat_args)

        after_results = while_after_block(*flat_args)

        if not isinstance(after_results, (list, ir.OpResultList)):
            after_results = [after_results]

        log().debug(
            "after_results [%s]",
            after_results,
        )

        return after_results

    # Call the universal SCF executor with two region builders
    return scf_gen.scf_execute_dynamic(
        op_type_name=while_op_type_name,
        mix_iter_args=write_args,
        full_write_args_count=full_write_args_count,
        mix_iter_arg_names=write_args_names,
        create_op_func=create_while_op,
        region_builders=[before_block_builder, after_block_builder],
        block_term_op_builder={
            before_block_builder: before_block_terminator
        },  # Only customize the before block
    )
