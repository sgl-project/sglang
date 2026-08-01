# SPDX-License-Identifier: Apache-2.0
"""Safe predicate expressions for conditional DAG routing.

Route predicates are evaluated on the orchestrator against a flat context of
per-request scalar metadata (``height``, ``width``, ``generate_audio``, and
anything else that survives the JSON hop).  Expressions use ordinary Python
syntax restricted to an AST allow-list: comparisons, boolean/arithmetic
operators, names, literals and subscripts.

Evaluation is a small recursive interpreter rather than ``eval``, so a
predicate has no way to reach outside its context dict even if the allow-list
were bypassed.
"""

from __future__ import annotations

import ast
import logging
import operator
from typing import Any

logger = logging.getLogger(__name__)


class PredicateError(ValueError):
    """Raised when a predicate is syntactically invalid or uses banned syntax."""


_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Subscript,
    ast.Tuple,
    ast.List,
    ast.IfExp,
)

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_CMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_CONSTANT_NAMES: dict[str, Any] = {"True": True, "False": False, "None": None}


def compile_predicate(expr: str) -> ast.Expression:
    """Parse and validate a predicate, returning an AST ready for evaluation.

    Raises ``PredicateError`` for syntax errors or disallowed constructs.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise PredicateError(f"Invalid predicate syntax {expr!r}: {e}") from None

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise PredicateError(
                f"Predicate {expr!r} uses disallowed syntax "
                f"{type(node).__name__}; only comparisons, boolean and "
                f"arithmetic operators, names, literals and subscripts are permitted"
            )
    return tree


def predicate_names(expr: str) -> set[str]:
    """Return the free variable names a predicate reads from its context."""
    tree = compile_predicate(expr)
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in _CONSTANT_NAMES
    }


def evaluate_predicate(
    expr: str | ast.Expression,
    context: dict[str, Any],
    *,
    default: bool = False,
) -> bool:
    """Evaluate a predicate against *context*.

    Names absent from the context resolve to ``None`` rather than raising, so
    a route whose predicate references a field the request never set is simply
    not taken.  Any other evaluation error falls back to *default*.
    """
    tree = compile_predicate(expr) if isinstance(expr, str) else expr
    try:
        return bool(_eval_node(tree.body, context))
    except Exception as e:
        logger.warning("Predicate evaluation failed (%s), defaulting to %s", e, default)
        return default


def _eval_node(node: ast.AST, ctx: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in _CONSTANT_NAMES:
            return _CONSTANT_NAMES[node.id]
        return ctx.get(node.id)

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for value in node.values:
                result = _eval_node(value, ctx)
                if not result:
                    return result
            return result
        result = False
        for value in node.values:
            result = _eval_node(value, ctx)
            if result:
                return result
        return result

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, ctx)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        return +operand

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise PredicateError(f"Unsupported operator {type(node.op).__name__}")
        return op(_eval_node(node.left, ctx), _eval_node(node.right, ctx))

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, ctx)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _CMPOPS.get(type(op_node))
            if op is None:
                raise PredicateError(f"Unsupported comparison {type(op_node).__name__}")
            right = _eval_node(comparator, ctx)
            if not op(left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.IfExp):
        if _eval_node(node.test, ctx):
            return _eval_node(node.body, ctx)
        return _eval_node(node.orelse, ctx)

    if isinstance(node, ast.Subscript):
        container = _eval_node(node.value, ctx)
        key = _eval_node(node.slice, ctx)
        if container is None:
            return None
        try:
            return container[key]
        except (KeyError, IndexError, TypeError):
            return None

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(e, ctx) for e in node.elts)

    if isinstance(node, ast.List):
        return [_eval_node(e, ctx) for e in node.elts]

    raise PredicateError(f"Unsupported predicate node {type(node).__name__}")
