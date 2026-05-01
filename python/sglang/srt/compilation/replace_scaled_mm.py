"""Rewrite eligible aten::_scaled_mm nodes to sgl_kernel.fp8_scaled_mm.

The rewrite is intentionally graph-local. Non-matching aten::_scaled_mm calls are
left in the graph, so they still lower through PyTorch's native implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import fx

try:
    from torch.fx.experimental.symbolic_shapes import definitely_true
except ImportError:
    from torch.fx.experimental.symbolic_shapes import (
        statically_known_true as definitely_true,
    )

from sglang.srt.compilation.inductor_pass import SGLangInductorPass

logger = logging.getLogger(__name__)

_ATEN_SCALED_MM = torch.ops.aten._scaled_mm.default


def _node_val(arg: Any) -> Any:
    return arg.meta.get("val") if isinstance(arg, fx.Node) else arg


def _arg(
    node: fx.Node,
    pos: int,
    name: str,
    default: Any = None,
) -> Any:
    if name in node.kwargs:
        return node.kwargs[name]
    if len(node.args) > pos:
        return node.args[pos]
    return default


def _is_cuda_2d_tensor(val: Any) -> bool:
    return (
        isinstance(val, torch.Tensor) and val.device.type == "cuda" and val.dim() == 2
    )


def _is_static_multiple(value: Any, multiple: int) -> bool:
    if not isinstance(value, int):
        return False
    return value % multiple == 0


def _same_dim(lhs: Any, rhs: Any) -> bool:
    return definitely_true(lhs == rhs)


def _is_one(value: Any) -> bool:
    return definitely_true(value == 1)


def _scale_shape_matches(scale: torch.Tensor, dim: Any) -> bool:
    if scale.dim() == 0:
        return True
    if scale.dim() == 1:
        return _is_one(scale.shape[0]) or _same_dim(scale.shape[0], dim)
    if scale.dim() == 2:
        return (
            (_is_one(scale.shape[0]) and _is_one(scale.shape[1]))
            or (_same_dim(scale.shape[0], dim) and _is_one(scale.shape[1]))
            or (_is_one(scale.shape[0]) and _same_dim(scale.shape[1], dim))
        )
    return False


def _cutlass_eligible(node: fx.Node) -> bool:
    mat_a = _node_val(_arg(node, 0, "self"))
    mat_b = _node_val(_arg(node, 1, "mat2"))
    scale_a = _node_val(_arg(node, 2, "scale_a"))
    scale_b = _node_val(_arg(node, 3, "scale_b"))
    bias = _node_val(_arg(node, 4, "bias"))
    scale_result = _arg(node, 5, "scale_result")
    out_dtype = _arg(node, 6, "out_dtype")
    use_fast_accum = _arg(node, 7, "use_fast_accum", False)

    if not (
        _is_cuda_2d_tensor(mat_a)
        and _is_cuda_2d_tensor(mat_b)
        and isinstance(scale_a, torch.Tensor)
        and isinstance(scale_b, torch.Tensor)
        and mat_a.dtype == torch.float8_e4m3fn
        and mat_b.dtype == torch.float8_e4m3fn
        and mat_a.stride(1) == 1
        and mat_b.stride(0) == 1
        and _same_dim(mat_a.shape[1], mat_b.shape[0])
        and _is_static_multiple(mat_b.shape[0], 16)
        and _is_static_multiple(mat_b.shape[1], 16)
        and scale_a.dtype == torch.float32
        and scale_b.dtype == torch.float32
        and scale_a.is_contiguous()
        and scale_b.is_contiguous()
        and _scale_shape_matches(scale_a, mat_a.shape[0])
        and _scale_shape_matches(scale_b, mat_b.shape[1])
        and out_dtype in (torch.float16, torch.bfloat16)
        and scale_result is None
        and not use_fast_accum
    ):
        return False

    if bias is not None and not (
        isinstance(bias, torch.Tensor)
        and bias.is_contiguous()
        and bias.dtype == out_dtype
        and (bias.dim() == 1 and _same_dim(bias.shape[0], mat_b.shape[1]))
    ):
        return False

    return True


class ReplaceScaledMMWithCutlassPass(SGLangInductorPass):
    """Replace eligible aten::_scaled_mm.default nodes with sgl_kernel CUTLASS."""

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_replace_scaled_mm")

        # Import lazily: it registers the sgl_kernel operator namespace, but only
        # when this pass actually sees a graph that may use the op.
        cutlass_op: Optional[torch._ops.OpOverload] = None
        count = 0

        for node in graph.nodes:
            if node.op != "call_function" or node.target != _ATEN_SCALED_MM:
                continue
            if not _cutlass_eligible(node):
                continue

            if cutlass_op is None:
                import sgl_kernel  # noqa: F401

                cutlass_op = torch.ops.sgl_kernel.fp8_scaled_mm.default

            with graph.inserting_before(node):
                replacement = graph.call_function(
                    cutlass_op,
                    args=(
                        _arg(node, 0, "self"),
                        _arg(node, 1, "mat2"),
                        _arg(node, 2, "scale_a"),
                        _arg(node, 3, "scale_b"),
                        _arg(node, 6, "out_dtype"),
                        _arg(node, 4, "bias"),
                        None,
                    ),
                )
                replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            count += 1

        graph.lint()
        logger.debug("Rewrote %s aten::_scaled_mm nodes to fp8_scaled_mm", count)
        self.dump_graph(graph, "after_replace_scaled_mm")
        self.end_and_log()
