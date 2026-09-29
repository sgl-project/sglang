# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import builtins
import operator
from dataclasses import dataclass

import torch
from torch import fx


@dataclass(frozen=True)
class NormSemantics:
    epsilon: float
    zero_centered: bool
    cast_before_weight: bool


def _op(node, name):
    if not isinstance(node, fx.Node):
        return False
    return (node.op == "call_method" and node.target == name) or (
        node.op == "call_function"
        and node.target in {getattr(torch, name, None), getattr(operator, name, None)}
    )


def _float_input(node, input_node):
    if _op(node, "float"):
        return node.args == (input_node,)
    return _op(node, "to") and node.args == (input_node, torch.float32)


def _input_cast(node, input_node):
    if _op(node, "type_as") and node.args[1:] == (input_node,):
        return node.args[0]
    if _op(node, "to") and len(node.args) == 2:
        dtype = node.args[1]
        if (
            isinstance(dtype, fx.Node)
            and dtype.op == "call_function"
            and dtype.target is builtins.getattr
            and dtype.args == (input_node, "dtype")
        ):
            return node.args[0]
    return None


def _weight(node):
    if isinstance(node, fx.Node) and node.op == "get_attr" and node.target == "weight":
        return True
    return _op(node, "float") and _weight(node.args[0])


def _scale(node):
    if _weight(node):
        return False
    if _op(node, "add") and len(node.args) == 2:
        for weight, offset in (node.args, node.args[::-1]):
            if isinstance(offset, (int, float)) and offset == 1 and _weight(weight):
                return True
    return None


def _normalization(node, input_node):
    if not _op(node, "mul") or len(node.args) != 2:
        return None
    for value, inv_std in (node.args, node.args[::-1]):
        if not _float_input(value, input_node) or not _op(inv_std, "rsqrt"):
            continue
        variance_epsilon = inv_std.args[0]
        if not _op(variance_epsilon, "add") or len(variance_epsilon.args) != 2:
            continue
        for mean, epsilon in (variance_epsilon.args, variance_epsilon.args[::-1]):
            if not isinstance(epsilon, (int, float)) or not _op(mean, "mean"):
                continue
            dimension = mean.args[1] if len(mean.args) > 1 else mean.kwargs.get("dim")
            keepdim = (
                mean.args[2]
                if len(mean.args) > 2
                else mean.kwargs.get("keepdim", False)
            )
            if dimension not in (-1, (-1,), [-1]) or not keepdim:
                continue
            squared = mean.args[0]
            if (
                (_op(squared, "pow") and squared.args == (value, 2))
                or (_op(squared, "square") and squared.args == (value,))
                or (_op(squared, "mul") and squared.args == (value, value))
            ):
                return float(epsilon)
    return None


def match_rms_norm(module):
    try:
        graph = fx.symbolic_trace(module).graph
    except (RuntimeError, TypeError, ValueError, fx.proxy.TraceError):
        return None
    inputs = [node for node in graph.nodes if node.op == "placeholder"]
    if len(inputs) != 1:
        return None
    if any(not node.users and node.op != "output" for node in graph.nodes):
        return None
    input_node = inputs[0]
    output = next(node for node in graph.nodes if node.op == "output").args[0]
    final_cast = _input_cast(output, input_node)
    tail = output if final_cast is None else final_cast
    weight = getattr(module, "weight", None)
    if weight is None:
        epsilon = _normalization(tail, input_node)
        if epsilon is not None and final_cast is not None:
            return NormSemantics(epsilon, False, False)
        return None
    if not _op(tail, "mul") or len(tail.args) != 2:
        return None
    for value, scale in (tail.args, tail.args[::-1]):
        zero_centered = _scale(scale)
        if zero_centered is None:
            continue
        early_cast = _input_cast(value, input_node)
        normalized = value if early_cast is None else early_cast
        epsilon = _normalization(normalized, input_node)
        if epsilon is None:
            continue
        if zero_centered:
            if early_cast is not None or final_cast is None:
                continue
            scale_weight = next(
                argument for argument in scale.args if isinstance(argument, fx.Node)
            )
            if weight.dtype != torch.float32 and not _op(scale_weight, "float"):
                continue
        else:
            if (early_cast is None) == (final_cast is None):
                continue
            if early_cast is not None and not (
                isinstance(scale, fx.Node) and scale.op == "get_attr"
            ):
                continue
        return NormSemantics(epsilon, zero_centered, early_cast is not None)
    return None
