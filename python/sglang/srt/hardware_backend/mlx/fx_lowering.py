"""Architecture-independent FX planning for whole-forward MLX execution.

Torch Dynamo already turns an SGLang model forward into an FX graph. This
module classifies that graph by operation rather than model architecture. It
is intentionally a validation backend for now: a production executor can be
provided once every admitted operation has an MLX lowering and serving-state
contract.
"""

from __future__ import annotations

import json
import logging
import math
import operator
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Optional

import msgspec
import torch
import torch.fx
import torch.nn.functional as F
from torch.export.graph_signature import OutputKind

logger = logging.getLogger(__name__)


class UnsupportedMlxFxGraphError(RuntimeError):
    """Raised when one FX graph cannot be lowered as a single MLX region."""


def _torch_entry_point(path: str) -> Any:
    """Resolve a semi-private Torch entry point the lowering depends on.

    These names are stable across the Torch >= 2.13 floor but carry no public
    guarantee; failing here names the missing symbol instead of surfacing as
    an unrelated lookup error at first use.
    """
    value: Any = torch
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            raise RuntimeError(
                f"torch.{path} is not available in torch {torch.__version__}; "
                "the MLX FX lowering requires torch >= 2.13"
            )
    return value


_AUTO_FUNCTIONALIZED = _torch_entry_point("ops.higher_order.auto_functionalized_v2")
_NN_LINEAR = _torch_entry_point("_C._nn.linear")
_NN_SDPA = _torch_entry_point("_C._nn.scaled_dot_product_attention")


class MlxLowering(msgspec.Struct, frozen=True):
    """One MLX lowering: its implementation and the FX targets that select it.

    ``fn(mx, args, kwargs)`` runs inside the MLX graph on resolved values.
    ``admit(args, kwargs)``, when given, runs at plan time on the node's
    literal arguments (tensors appear as ``torch.fx.Node``) and returns why
    the node cannot be lowered, or None; that routes inadmissible graphs to
    the Torch fallback before any executor exists.
    """

    name: str
    fn: Any
    aten: tuple = ()
    functions: tuple = ()
    methods: tuple = ()
    admit: Any = None


@dataclass(frozen=True)
class MlxFxNodePlan:
    node_name: str
    node_op: str
    target: Any
    lowering: Optional[str]
    implementation: Any = None
    reason: Optional[str] = None

    @property
    def supported(self) -> bool:
        return self.lowering is not None


@dataclass(frozen=True)
class MlxFxGraphPlan:
    """One captured forward and its operation-level MLX coverage."""

    graph_module: torch.fx.GraphModule
    nodes: tuple[MlxFxNodePlan, ...]

    @property
    def unsupported(self) -> tuple[MlxFxNodePlan, ...]:
        return tuple(node for node in self.nodes if not node.supported)

    @property
    def fully_supported(self) -> bool:
        return not self.unsupported

    def describe_unsupported(self) -> str:
        return ", ".join(
            f"{node.node_op}:{node.target}"
            + (f" ({node.reason})" if node.reason else "")
            for node in self.unsupported
        )

    def require_fully_supported(self) -> None:
        if self.fully_supported:
            return
        raise UnsupportedMlxFxGraphError(
            "FX forward cannot run as one MLX region; unsupported nodes: "
            + self.describe_unsupported()
        )


class MlxFxLoweringRegistry:
    """Maps FX targets to MLX lowerings, implementation included.

    The standard tables are derived from the ``@mlx_lowering`` registrations
    below: each lowering function declares the ATen overloads, Torch
    functions, and tensor methods it implements, so adding an op is one
    self-contained function and never a second table edit. Every entry binds
    an implementation, so a target the plan admits always has code behind it.
    """

    def __init__(self) -> None:
        self._functions: dict[Any, MlxLowering] = {}
        self._functionalized: dict[Any, MlxLowering] = {}
        self._methods: dict[str, MlxLowering] = {}

    def register_function(self, target: Any, lowering: Any) -> None:
        """Bind a ``call_function`` target to a lowering.

        ``lowering`` is a :class:`MlxLowering`, a callable ``fn(mx, args,
        kwargs)``, or the name of a built-in lowering; a name that has no
        implementation is rejected here rather than at first execution.
        """
        self._functions[target] = _as_lowering(lowering, target)

    def register_functionalized(self, custom_target: Any, lowering: Any) -> None:
        """Bind the ``auto_functionalized`` spelling of a mutating custom op.

        ``torch.export`` wraps a custom op that mutates an input in
        ``auto_functionalized_v2(op, ...)``; the lowering receives that
        wrapper's arguments, with the op as ``args[0]``, and must return the
        functional outputs the wrapper declares.
        """
        bound = _as_lowering(lowering, custom_target)
        self._functionalized[custom_target] = msgspec.structs.replace(
            bound, name=f"functionalized_{bound.name}"
        )

    def register_method(self, target: str, lowering: Any) -> None:
        self._methods[target] = _as_lowering(lowering, target)

    def resolve(self, node: torch.fx.Node) -> Optional[MlxLowering]:
        if node.op == "call_function":
            if node.target is _AUTO_FUNCTIONALIZED:
                return self._functionalized.get(node.args[0])
            return self._functions.get(node.target)
        if node.op == "call_method":
            return self._methods.get(str(node.target))
        return None

    @classmethod
    def standard_decoder(cls) -> MlxFxLoweringRegistry:
        """Return lowerings used by ordinary dense decoder graphs."""
        registry = cls()
        for lowering in _unique_lowerings():
            for target in lowering.functions:
                registry.register_function(target, lowering)
            for target in lowering.methods:
                registry.register_method(target, lowering)
        return registry

    @classmethod
    def standard_export_decoder(cls) -> MlxFxLoweringRegistry:
        """Return canonical ATen lowerings produced by ``torch.export``."""
        registry = cls()
        for lowering in _unique_lowerings():
            for target in lowering.aten:
                registry.register_function(target, lowering)
        return registry


def _as_lowering(lowering: Any, target: Any) -> MlxLowering:
    if isinstance(lowering, MlxLowering):
        return lowering
    if isinstance(lowering, str):
        built_in = _LOWERINGS.get(lowering)
        if built_in is None:
            raise ValueError(
                f"no built-in MLX lowering named {lowering!r} to bind to {target}; "
                "pass the implementation itself"
            )
        return built_in
    if callable(lowering):
        return MlxLowering(
            name=getattr(lowering, "__name__", repr(lowering)), fn=lowering
        )
    raise TypeError(f"cannot bind {lowering!r} as an MLX lowering for {target}")


def _unique_lowerings() -> tuple[MlxLowering, ...]:
    # Aliases ("view" -> the reshape lowering) map to one object; register
    # each implementation once.
    seen: dict[int, MlxLowering] = {}
    for lowering in _LOWERINGS.values():
        seen.setdefault(id(lowering), lowering)
    return tuple(seen.values())


def build_mlx_fx_plan(
    graph_module: torch.fx.GraphModule,
    registry: MlxFxLoweringRegistry,
) -> MlxFxGraphPlan:
    """Classify a captured forward without inspecting its model architecture."""
    return MlxFxGraphPlan(
        graph_module=graph_module,
        nodes=tuple(_plan_node(node, registry) for node in graph_module.graph.nodes),
    )


def _plan_node(node: torch.fx.Node, registry: MlxFxLoweringRegistry) -> MlxFxNodePlan:
    if node.op in {"placeholder", "get_attr", "output"}:
        return MlxFxNodePlan(node.name, node.op, node.target, lowering=node.op)
    lowering = registry.resolve(node)
    if lowering is None:
        return MlxFxNodePlan(
            node.name, node.op, node.target, lowering=None, reason="no MLX lowering"
        )
    reason = lowering.admit(node.args, node.kwargs) if lowering.admit else None
    if reason is not None:
        return MlxFxNodePlan(
            node.name, node.op, node.target, lowering=None, reason=reason
        )
    return MlxFxNodePlan(
        node.name,
        node.op,
        node.target,
        lowering=lowering.name,
        implementation=lowering.fn,
    )


def _resolve_fx_value(value: Any, values: dict[torch.fx.Node, Any]) -> Any:
    if isinstance(value, torch.fx.Node):
        return values[value]
    if isinstance(value, tuple):
        return tuple(_resolve_fx_value(item, values) for item in value)
    if isinstance(value, list):
        return [_resolve_fx_value(item, values) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_fx_value(item, values) for key, item in value.items()}
    return value


_LOWERINGS: dict[str, MlxLowering] = {}


def mlx_lowering(
    name: str,
    *,
    aten: Sequence[Any] = (),
    functions: Sequence[Any] = (),
    methods: Sequence[str] = (),
    aliases: Sequence[str] = (),
    admit: Optional[Callable[..., Optional[str]]] = None,
):
    """Register ``fn`` as the built-in MLX lowering ``name`` for its FX targets.

    ``aten`` are the canonical overloads ``torch.export`` emits, ``functions``
    and ``methods`` the eager-FX spellings; ``aliases`` register the same
    lowering under additional names (``reshape``/``view``); ``admit`` is the
    plan-time admissibility predicate (see :class:`MlxLowering`).
    """

    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        lowering = MlxLowering(
            name=name,
            fn=fn,
            aten=tuple(aten),
            functions=tuple(functions),
            methods=tuple(methods),
            admit=admit,
        )
        for key in (name, *aliases):
            if key in _LOWERINGS:
                raise ValueError(f"duplicate MLX lowering: {key}")
            _LOWERINGS[key] = lowering
        return fn

    return register


def _lower_mlx_node(
    lowering: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run one built-in lowering by name; the catalogue entry for tests."""
    import mlx.core as mx

    spec = _LOWERINGS.get(lowering)
    if spec is None:
        raise UnsupportedMlxFxGraphError(f"missing MLX lowering: {lowering}")
    return spec.fn(mx, args, kwargs)


def _arg(args, kwargs, index, name, default=None):
    return args[index] if len(args) > index else kwargs.get(name, default)


@mlx_lowering("assert_metadata", aten=(torch.ops.aten._assert_tensor_metadata.default,))
def _lower_assert_metadata(mx, args, kwargs):
    return None


@mlx_lowering("alias", aten=(torch.ops.aten.alias.default,))
def _lower_alias(mx, args, kwargs):
    return args[0]


@mlx_lowering("concatenate", aten=(torch.ops.aten.cat.default,))
def _lower_concatenate(mx, args, kwargs):
    return mx.concatenate(tuple(args[0]), axis=_arg(args, kwargs, 1, "dim", 0))


# aten.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse): the
# trailing flags only affect backward, so every forward is admissible.
@mlx_lowering("embedding", aten=(torch.ops.aten.embedding.default,))
def _lower_embedding(mx, args, kwargs):
    weight, input_ids = args[:2]
    return mx.take(weight, input_ids, axis=0)


def _admit_functional_embedding(args, kwargs) -> Optional[str]:
    if _arg(args, kwargs, 3, "max_norm") is not None:
        return "embedding max_norm renormalizes the weight in place"
    return None


# F.embedding(input, weight, padding_idx, max_norm, ...): the eager spelling
# takes ids first, so it is its own lowering rather than a shape guess.
@mlx_lowering(
    "embedding_functional",
    functions=(F.embedding,),
    admit=_admit_functional_embedding,
)
def _lower_functional_embedding(mx, args, kwargs):
    input_ids, weight = args[:2]
    return mx.take(weight, input_ids, axis=0)


def _admit_rms_norm(args, kwargs) -> Optional[str]:
    if _arg(args, kwargs, 2, "weight") is None:
        return "MLX RMSNorm requires an explicit weight"
    return None


@mlx_lowering(
    "rms_norm",
    aten=(torch.ops.aten.rms_norm.default,),
    functions=(torch.rms_norm,),
    admit=_admit_rms_norm,
)
def _lower_rms_norm(mx, args, kwargs):
    value = args[0]
    weight = _arg(args, kwargs, 2, "weight")
    epsilon = _arg(args, kwargs, 3, "eps")
    if weight is None:
        raise UnsupportedMlxFxGraphError("MLX RMSNorm requires an explicit weight")
    if epsilon is None:
        epsilon = mx.finfo(value.dtype).eps
    return mx.fast.rms_norm(value, weight, float(epsilon))


def _admit_layer_norm(args, kwargs) -> Optional[str]:
    normalized_shape = _arg(args, kwargs, 1, "normalized_shape")
    if isinstance(normalized_shape, Sequence) and len(normalized_shape) != 1:
        return (
            "MLX layer_norm only normalizes the last axis, found "
            f"normalized_shape={tuple(normalized_shape)}"
        )
    return None


@mlx_lowering(
    "layer_norm", aten=(torch.ops.aten.layer_norm.default,), admit=_admit_layer_norm
)
def _lower_layer_norm(mx, args, kwargs):
    value, normalized_shape = args[:2]
    weight = _arg(args, kwargs, 2, "weight")
    bias = _arg(args, kwargs, 3, "bias")
    epsilon = _arg(args, kwargs, 4, "eps", 1e-5)
    if len(normalized_shape) != 1 or normalized_shape[0] != value.shape[-1]:
        raise UnsupportedMlxFxGraphError(
            "MLX layer_norm only normalizes the last axis, found "
            f"normalized_shape={tuple(normalized_shape)}"
        )
    return mx.fast.layer_norm(value, weight, bias, float(epsilon))


@mlx_lowering(
    "linear",
    aten=(torch.ops.aten.linear.default,),
    functions=(_NN_LINEAR,),
)
def _lower_linear(mx, args, kwargs):
    value, weight = args[:2]
    bias = _arg(args, kwargs, 2, "bias")
    output = value @ mx.swapaxes(weight, -1, -2)
    return output if bias is None else output + bias


@mlx_lowering("matmul", aten=(torch.ops.aten.matmul.default, torch.ops.aten.mm.default))
def _lower_matmul(mx, args, kwargs):
    return args[0] @ args[1]


def _admit_dtype_kwarg(args, kwargs) -> Optional[str]:
    dtype = kwargs.get("dtype")
    if dtype is not None and dtype not in _MLX_DTYPES:
        return f"Torch dtype has no MLX Metal lowering: {dtype}"
    return None


@mlx_lowering("mean", aten=(torch.ops.aten.mean.dim,), admit=_admit_dtype_kwarg)
def _lower_mean(mx, args, kwargs):
    axis = _arg(args, kwargs, 1, "dim")
    keepdims = _arg(args, kwargs, 2, "keepdim", False)
    dtype = kwargs.get("dtype")
    value = args[0] if dtype is None else args[0].astype(_mlx_dtype(dtype, mx))
    return mx.mean(value, axis=axis, keepdims=keepdims)


def _admit_sdpa(args, kwargs) -> Optional[str]:
    if _arg(args, kwargs, 4, "dropout_p", 0.0) not in (None, 0, 0.0):
        return "SDPA dropout is unsupported"
    return None


@mlx_lowering(
    "sdpa",
    aten=(torch.ops.aten.scaled_dot_product_attention.default,),
    functions=(_NN_SDPA,),
    admit=_admit_sdpa,
)
def _lower_sdpa(mx, args, kwargs):
    query, key, value = args[:3]
    attention_mask = _arg(args, kwargs, 3, "attn_mask")
    dropout = _arg(args, kwargs, 4, "dropout_p", 0.0)
    causal = _arg(args, kwargs, 5, "is_causal", False)
    scale = _arg(args, kwargs, 6, "scale")
    enable_gqa = _arg(args, kwargs, 7, "enable_gqa", False)
    if dropout not in (None, 0, 0.0):
        raise UnsupportedMlxFxGraphError("SDPA dropout is unsupported")
    if enable_gqa:
        if any(tensor.ndim < 3 for tensor in (query, key, value)):
            raise UnsupportedMlxFxGraphError("SDPA GQA requires a head dimension")
        query_heads, key_heads, value_heads = (
            tensor.shape[-3] for tensor in (query, key, value)
        )
        if (
            key_heads == 0
            or value_heads == 0
            or query_heads % key_heads
            or query_heads % value_heads
        ):
            raise UnsupportedMlxFxGraphError(
                "SDPA GQA key/value head counts must divide query heads"
            )
        if query_heads != key_heads:
            key = mx.repeat(key, query_heads // key_heads, axis=-3)
        if query_heads != value_heads:
            value = mx.repeat(value, query_heads // value_heads, axis=-3)
    scale = float(scale) if scale is not None else 1.0 / sqrt(query.shape[-1])
    scores = (query @ mx.swapaxes(key, -1, -2)) * scale
    if causal:
        rows, columns = scores.shape[-2:]
        # ATen SDPA uses upper-left causal alignment even when L != S.
        # Cached decode's lower-right convention belongs to its own op.
        causal_mask = mx.triu(
            mx.full((rows, columns), -float("inf"), dtype=scores.dtype),
            k=1,
        )
        scores = scores + causal_mask
    if attention_mask is not None:
        if attention_mask.dtype == mx.bool_:
            scores = mx.where(attention_mask, scores, -float("inf"))
        else:
            scores = scores + attention_mask
    # Torch SDPA returns zero for fully masked rows. Avoid softmax(-inf, ...)
    # without replacing unrelated NaNs that should still propagate.
    fully_masked = mx.all(scores == -float("inf"), axis=-1, keepdims=True)
    probabilities = mx.softmax(mx.where(fully_masked, 0, scores), axis=-1)
    probabilities = mx.where(fully_masked, 0, probabilities)
    return probabilities @ value


@mlx_lowering("silu", aten=(torch.ops.aten.silu.default,), functions=(F.silu,))
def _lower_silu(mx, args, kwargs):
    return mx.sigmoid(args[0]) * args[0]


@mlx_lowering("sigmoid", aten=(torch.ops.aten.sigmoid.default,))
def _lower_sigmoid(mx, args, kwargs):
    return mx.sigmoid(args[0])


@mlx_lowering("relu", aten=(torch.ops.aten.relu.default,))
def _lower_relu(mx, args, kwargs):
    return mx.maximum(args[0], 0)


def _admit_gelu(args, kwargs) -> Optional[str]:
    approximate = kwargs.get("approximate", "none")
    if approximate not in ("none", "tanh"):
        return f"unsupported GELU approximation: {approximate}"
    return None


@mlx_lowering("gelu", aten=(torch.ops.aten.gelu.default,), admit=_admit_gelu)
def _lower_gelu(mx, args, kwargs):
    value = args[0]
    approximate = kwargs.get("approximate", "none")
    if approximate == "tanh":
        return (
            0.5
            * value
            * (1 + mx.tanh(sqrt(2 / math.pi) * (value + 0.044715 * value**3)))
        )
    if approximate != "none":
        raise UnsupportedMlxFxGraphError(
            f"unsupported GELU approximation: {approximate}"
        )
    return value * 0.5 * (1 + mx.erf(value / sqrt(2)))


@mlx_lowering("tanh", aten=(torch.ops.aten.tanh.default,))
def _lower_tanh(mx, args, kwargs):
    return mx.tanh(args[0])


@mlx_lowering("stack", aten=(torch.ops.aten.stack.default,))
def _lower_stack(mx, args, kwargs):
    return mx.stack(tuple(args[0]), axis=_arg(args, kwargs, 1, "dim", 0))


@mlx_lowering("getitem", aten=(operator.getitem,), functions=(operator.getitem,))
def _lower_getitem(mx, args, kwargs):
    return args[0][args[1]]


@mlx_lowering(
    "add", aten=(torch.ops.aten.add.Tensor,), functions=(operator.add, torch.add)
)
def _lower_add(mx, args, kwargs):
    alpha = _arg(args, kwargs, 2, "alpha", 1)
    if alpha == 1:
        return args[0] + args[1]
    return args[0] + args[1] * alpha


@mlx_lowering(
    "multiply", aten=(torch.ops.aten.mul.Tensor,), functions=(operator.mul, torch.mul)
)
def _lower_multiply(mx, args, kwargs):
    return args[0] * args[1]


@mlx_lowering(
    "subtract", aten=(torch.ops.aten.sub.Tensor,), functions=(operator.sub, torch.sub)
)
def _lower_subtract(mx, args, kwargs):
    alpha = _arg(args, kwargs, 2, "alpha", 1)
    if alpha == 1:
        return args[0] - args[1]
    return args[0] - args[1] * alpha


@mlx_lowering("chunk", aten=(torch.ops.aten.chunk.default,), methods=("chunk",))
def _lower_chunk(mx, args, kwargs):
    value, chunks = args[:2]
    axis = _arg(args, kwargs, 2, "dim", 0)
    width = value.shape[axis]
    chunk_size = (width + chunks - 1) // chunks
    indices = tuple(range(chunk_size, width, chunk_size))
    return tuple(mx.split(value, indices, axis=axis))


@mlx_lowering("split", aten=(torch.ops.aten.split.Tensor,))
def _lower_split(mx, args, kwargs):
    value, split_size = args[:2]
    axis = _arg(args, kwargs, 2, "dim", 0)
    width = value.shape[axis]
    indices = tuple(range(split_size, width, split_size))
    return tuple(mx.split(value, indices, axis=axis))


@mlx_lowering("split_with_sizes", aten=(torch.ops.aten.split_with_sizes.default,))
def _lower_split_with_sizes(mx, args, kwargs):
    value, sizes = args[:2]
    axis = _arg(args, kwargs, 2, "dim", 0)
    indices = []
    offset = 0
    for size in sizes[:-1]:
        offset += size
        indices.append(offset)
    return tuple(mx.split(value, tuple(indices), axis=axis))


@mlx_lowering(
    "reshape",
    aten=(torch.ops.aten.reshape.default, torch.ops.aten.view.default),
    methods=("reshape",),
    aliases=("view",),
)
def _lower_reshape(mx, args, kwargs):
    shape = args[1] if len(args) == 2 and isinstance(args[1], Sequence) else args[1:]
    return mx.reshape(args[0], tuple(shape))


@mlx_lowering("flatten", aten=(torch.ops.aten.flatten.using_ints,))
def _lower_flatten(mx, args, kwargs):
    value = args[0]
    start = _arg(args, kwargs, 1, "start_dim", 0)
    end = _arg(args, kwargs, 2, "end_dim", -1)
    if end < 0:
        end += value.ndim
    flattened = 1
    for width in value.shape[start : end + 1]:
        flattened *= width
    shape = (*value.shape[:start], flattened, *value.shape[end + 1 :])
    return mx.reshape(value, shape)


@mlx_lowering("index_select", aten=(torch.ops.aten.index_select.default,))
def _lower_index_select(mx, args, kwargs):
    return mx.take(args[0], args[2], axis=args[1])


@mlx_lowering("numpy_transpose", aten=(torch.ops.aten.numpy_T.default,))
def _lower_numpy_transpose(mx, args, kwargs):
    return mx.transpose(args[0], axes=tuple(reversed(range(args[0].ndim))))


@mlx_lowering("power", aten=(torch.ops.aten.pow.Tensor_Scalar,))
def _lower_power(mx, args, kwargs):
    return mx.power(args[0], args[1])


@mlx_lowering("rsqrt", aten=(torch.ops.aten.rsqrt.default,))
def _lower_rsqrt(mx, args, kwargs):
    return mx.rsqrt(args[0])


def _admit_to_dtype(args, kwargs) -> Optional[str]:
    dtype = _arg(args, kwargs, 1, "dtype")
    if isinstance(dtype, torch.dtype) and dtype not in _MLX_DTYPES:
        return f"Torch dtype has no MLX Metal lowering: {dtype}"
    return None


@mlx_lowering("to_dtype", aten=(torch.ops.aten.to.dtype,), admit=_admit_to_dtype)
def _lower_to_dtype(mx, args, kwargs):
    return args[0].astype(_mlx_dtype(args[1], mx))


@mlx_lowering("unsqueeze", aten=(torch.ops.aten.unsqueeze.default,))
def _lower_unsqueeze(mx, args, kwargs):
    return mx.expand_dims(args[0], axis=args[1])


@mlx_lowering(
    "empty_like", aten=(torch.ops.aten.empty_like.default,), admit=_admit_dtype_kwarg
)
def _lower_empty_like(mx, args, kwargs):
    dtype = kwargs.get("dtype")
    if dtype is None:
        return mx.empty_like(args[0])
    return mx.empty(args[0].shape, dtype=_mlx_dtype(dtype, mx))


@mlx_lowering("slice", aten=(torch.ops.aten.slice.Tensor,))
def _lower_slice(mx, args, kwargs):
    value = args[0]
    axis = _arg(args, kwargs, 1, "dim", 0)
    start = _arg(args, kwargs, 2, "start", 0)
    stop = _arg(args, kwargs, 3, "end")
    step = _arg(args, kwargs, 4, "step", 1)
    if start is None:
        start = 0
    if stop is None:
        stop = value.shape[axis]
    width = value.shape[axis]
    if start >= 0:
        start = min(start, width)
    if stop >= 0:
        stop = min(stop, width)
    slices = [slice(None)] * value.ndim
    slices[axis] = slice(start, stop, step)
    return value[tuple(slices)]


@mlx_lowering("transpose", methods=("transpose",))
def _lower_transpose(mx, args, kwargs):
    return mx.swapaxes(args[0], args[1], args[2])


@mlx_lowering("contiguous", methods=("contiguous",))
def _lower_contiguous(mx, args, kwargs):
    return mx.contiguous(args[0])


_MLX_DTYPES = frozenset(
    {
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }
)


def _mlx_dtype(dtype: torch.dtype, mx: Any) -> Any:
    mapping = {
        torch.bool: mx.bool_,
        torch.int8: mx.int8,
        torch.int16: mx.int16,
        torch.int32: mx.int32,
        torch.int64: mx.int64,
        torch.float16: mx.float16,
        torch.bfloat16: mx.bfloat16,
        torch.float32: mx.float32,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise UnsupportedMlxFxGraphError(
            f"Torch dtype has no MLX Metal lowering: {dtype}"
        ) from exc


def _resolve_graph_attrs(
    graph_module: torch.fx.GraphModule, attr_nodes: Sequence[torch.fx.Node]
) -> tuple[tuple[Any, str, Any], ...]:
    """Resolve ``get_attr`` targets once: (owner module, attribute, value).

    FX ``get_attr`` nodes are not limited to parameters and registered
    buffers: a module may hang constant tensors or scalars directly on
    itself, and ``torch.export``'s ``module()`` re-registers lifted tensor
    constants the same way. The owner/attribute pair is what a later step
    re-reads to notice a replaced tensor without walking the module again.
    """
    parameters = dict(graph_module.named_parameters())
    buffers = dict(graph_module.named_buffers())
    resolved = []
    for node in attr_nodes:
        target = str(node.target)
        module_path, _, attr_name = target.rpartition(".")
        owner = graph_module.get_submodule(module_path) if module_path else graph_module
        if target in parameters:
            value = parameters[target]
        elif target in buffers:
            value = buffers[target]
        else:
            value = getattr(owner, attr_name)
        resolved.append((owner, attr_name, value))
    return tuple(resolved)


def make_mlx_fx_executor(
    plan: MlxFxGraphPlan,
    example_inputs: list[Any],
) -> Callable[..., Any]:
    """Build one compiled MLX callable for a fully admitted FX graph.

    The graph is compiled with static shapes: lowerings read Python-level
    shapes (causal masks, chunk and flatten arithmetic), which a shapeless
    trace cannot carry, so a new input shape is a new compile.
    """
    import mlx.core as mx

    from sglang.srt.utils.tensor_bridge import MlxTensorView, mlx_call_multi

    plan.require_fully_supported()
    attr_nodes = tuple(
        node for node in plan.graph_module.graph.nodes if node.op == "get_attr"
    )
    attrs = _resolve_graph_attrs(plan.graph_module, attr_nodes)
    # Parameters, buffers, and constant-tensor attributes ride as borrowed
    # views; non-tensor attributes (scalars, shapes) are captured by value.
    tensor_attrs = tuple(
        (node, owner, attr_name)
        for node, (owner, attr_name, value) in zip(attr_nodes, attrs)
        if isinstance(value, torch.Tensor)
    )
    constant_attr_values = {
        node: value
        for node, (_, _, value) in zip(attr_nodes, attrs)
        if not isinstance(value, torch.Tensor)
    }
    attr_views = [
        MlxTensorView(getattr(owner, attr_name)) for _, owner, attr_name in tensor_attrs
    ]
    placeholder_nodes = tuple(
        node for node in plan.graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholder_nodes) != len(example_inputs):
        raise UnsupportedMlxFxGraphError(
            "FX placeholder and example-input counts do not match"
        )
    tensor_positions = tuple(
        index
        for index, value in enumerate(example_inputs)
        if isinstance(value, torch.Tensor)
    )
    scalar_positions = tuple(
        index for index in range(len(example_inputs)) if index not in tensor_positions
    )
    used_scalars = tuple(
        placeholder_nodes[index]
        for index in scalar_positions
        if placeholder_nodes[index].users
    )
    if used_scalars:
        names = ", ".join(node.name for node in used_scalars)
        raise UnsupportedMlxFxGraphError(
            "whole-graph MLX lowering requires normalization of used symbolic "
            f"scalar inputs: {names}"
        )
    tensor_placeholder_nodes = tuple(
        placeholder_nodes[index] for index in tensor_positions
    )
    tensor_attr_nodes = tuple(node for node, _, _ in tensor_attrs)

    def mlx_graph(*arrays):
        runtime_arrays = arrays[: len(tensor_placeholder_nodes)]
        captured_arrays = arrays[len(tensor_placeholder_nodes) :]
        values: dict[torch.fx.Node, Any] = dict(
            zip(tensor_placeholder_nodes, runtime_arrays)
        )
        values.update(zip(tensor_attr_nodes, captured_arrays))
        values.update(constant_attr_values)
        for node, node_plan in zip(plan.graph_module.graph.nodes, plan.nodes):
            if node.op in {"placeholder", "get_attr"}:
                continue
            args = _resolve_fx_value(node.args, values)
            kwargs = _resolve_fx_value(node.kwargs, values)
            if node.op == "output":
                result = args[0]
                return tuple(result) if isinstance(result, (tuple, list)) else (result,)
            values[node] = node_plan.implementation(mx, args, kwargs)
        raise UnsupportedMlxFxGraphError("FX graph has no output node")

    compiled_graph = mx.compile(mlx_graph, shapeless=False)

    def execute(*torch_inputs):
        if len(torch_inputs) != len(placeholder_nodes):
            raise RuntimeError(
                "compiled MLX graph input count changed: "
                f"expected {len(placeholder_nodes)}, found {len(torch_inputs)}"
            )
        if any(
            not isinstance(torch_inputs[index], torch.Tensor)
            or torch_inputs[index].device.type != "mps"
            for index in tensor_positions
        ):
            raise RuntimeError("compiled MLX graph requires Torch MPS tensors")
        # A view borrows storage; a replaced parameter or buffer (new
        # storage, same attribute) needs a new borrow. In-place updates keep
        # the storage and need nothing.
        for index, (_, owner, attr_name) in enumerate(tensor_attrs):
            tensor = getattr(owner, attr_name)
            if not attr_views[index].matches(tensor):
                attr_views[index] = MlxTensorView(tensor)
        return mlx_call_multi(
            compiled_graph,
            *(torch_inputs[index] for index in tensor_positions),
            *attr_views,
            device="mps",
        )

    return execute


@dataclass(frozen=True)
class MlxExportPlan:
    """Functional Export IR plus explicit serving-state mutations."""

    exported_program: Any
    functional_program: Any
    graph_plan: MlxFxGraphPlan
    mutated_user_inputs: tuple[str, ...]


def export_mlx_plan(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    registry: MlxFxLoweringRegistry,
    *,
    kwargs: Optional[dict[str, Any]] = None,
    dynamic_shapes: Any = None,
) -> MlxExportPlan:
    """Capture and functionalize a model forward before MLX lowering."""
    exported = torch.export.export(
        module,
        args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )
    functional = exported.run_decompositions()
    graph_plan = build_mlx_fx_plan(functional.graph_module, registry)
    mutations = tuple(
        str(spec.target)
        for spec in functional.graph_signature.output_specs
        if spec.kind is OutputKind.USER_INPUT_MUTATION
    )
    return MlxExportPlan(
        exported_program=exported,
        functional_program=functional,
        graph_plan=graph_plan,
        mutated_user_inputs=mutations,
    )


class MlxFxCaptureBackend:
    """Dynamo backend that validates whole-graph MLX lowering coverage.

    The default executor returns the captured Torch graph so the capture path
    can be tested without pretending that MLX execution is implemented. A
    production ``executor_factory`` receives the complete admitted graph once
    and must return one callable containing the Torch/MLX bridge boundary.
    Dynamo owns the compiled callables; the backend keeps at most
    ``max_plans`` recent plans for reports and tests.
    """

    def __init__(
        self,
        registry: Optional[MlxFxLoweringRegistry] = None,
        *,
        executor_factory: Optional[
            Callable[[MlxFxGraphPlan, list[Any]], Callable[..., Any]]
        ] = None,
        fallback_to_torch: bool = False,
        report_path: Optional[str] = None,
        max_plans: int = 32,
    ) -> None:
        self.registry = registry or MlxFxLoweringRegistry.standard_decoder()
        self.executor_factory = executor_factory
        self.fallback_to_torch = fallback_to_torch
        self.report_path = report_path
        self.plans: deque[MlxFxGraphPlan] = deque(maxlen=max_plans)

    def __call__(
        self, graph_module: torch.fx.GraphModule, example_inputs: list[Any]
    ) -> Callable[..., Any]:
        plan = build_mlx_fx_plan(graph_module, self.registry)
        self.plans.append(plan)
        if self.report_path is not None:
            Path(self.report_path).write_text(
                json.dumps(
                    {
                        "fully_supported": plan.fully_supported,
                        "nodes": [
                            {
                                "name": node.node_name,
                                "op": node.node_op,
                                "target": str(node.target),
                                "lowering": node.lowering,
                                "reason": node.reason,
                            }
                            for node in plan.nodes
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        if not plan.fully_supported:
            if self.fallback_to_torch:
                logger.warning(
                    "MLX region cannot lower this graph; it runs on Torch. "
                    "Unsupported nodes: %s",
                    plan.describe_unsupported(),
                )
                return graph_module.forward
            plan.require_fully_supported()
        if self.executor_factory is None:
            return graph_module.forward
        return self.executor_factory(plan, example_inputs)


__all__ = [
    "MlxExportPlan",
    "MlxFxCaptureBackend",
    "MlxFxGraphPlan",
    "MlxFxLoweringRegistry",
    "MlxFxNodePlan",
    "MlxLowering",
    "UnsupportedMlxFxGraphError",
    "build_mlx_fx_plan",
    "export_mlx_plan",
    "make_mlx_fx_executor",
    "mlx_lowering",
]
