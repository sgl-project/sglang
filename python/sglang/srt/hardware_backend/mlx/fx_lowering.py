"""Architecture-independent FX planning for whole-forward MLX execution.

Torch Dynamo already turns an SGLang model forward into an FX graph. This
module classifies that graph by operation rather than model architecture. It
is intentionally a validation backend for now: a production executor can be
provided once every admitted operation has an MLX lowering and serving-state
contract.
"""

from __future__ import annotations

import json
import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.fx
import torch.nn.functional as F


class UnsupportedMlxFxGraphError(RuntimeError):
    """Raised when one FX graph cannot be lowered as a single MLX region."""


@dataclass(frozen=True)
class MlxFxNodePlan:
    node_name: str
    node_op: str
    target: Any
    lowering: Optional[str]

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

    def require_fully_supported(self) -> None:
        unsupported = self.unsupported
        if not unsupported:
            return
        details = ", ".join(f"{node.node_op}:{node.target}" for node in unsupported)
        raise UnsupportedMlxFxGraphError(
            "FX forward cannot run as one MLX region; unsupported nodes: " + details
        )


class MlxFxLoweringRegistry:
    """Maps FX operations to reusable MLX lowering names.

    The standard tables are derived from the ``@_lowering`` registrations
    below: each lowering function declares the ATen overloads, Torch
    functions, and tensor methods it implements, so adding an op is one
    self-contained function and never a second table edit.
    """

    def __init__(self) -> None:
        self._functions: dict[Any, str] = {}
        self._methods: dict[str, str] = {}

    def register_function(self, target: Any, lowering: str) -> None:
        self._functions[target] = lowering

    def register_method(self, target: str, lowering: str) -> None:
        self._methods[target] = lowering

    def resolve(self, node: torch.fx.Node) -> Optional[str]:
        if node.op in {"placeholder", "get_attr", "output"}:
            return node.op
        if node.op == "call_function":
            if node.target is torch.ops.higher_order.auto_functionalized_v2:
                custom_target = node.args[0]
                lowering = self._functions.get(custom_target)
                return None if lowering is None else f"functionalized_{lowering}"
            return self._functions.get(node.target)
        if node.op == "call_method":
            return self._methods.get(str(node.target))
        return None

    @classmethod
    def standard_decoder(cls) -> MlxFxLoweringRegistry:
        """Return lowerings used by ordinary dense decoder graphs."""
        registry = cls()
        for name, spec in _LOWERINGS.items():
            for target in spec.functions:
                registry.register_function(target, name)
            for target in spec.methods:
                registry.register_method(target, name)
        return registry

    @classmethod
    def standard_export_decoder(cls) -> MlxFxLoweringRegistry:
        """Return canonical ATen lowerings produced by ``torch.export``."""
        registry = cls()
        for name, spec in _LOWERINGS.items():
            for target in spec.aten:
                registry.register_function(target, name)
        return registry


def build_mlx_fx_plan(
    graph_module: torch.fx.GraphModule,
    registry: MlxFxLoweringRegistry,
) -> MlxFxGraphPlan:
    """Classify a captured forward without inspecting its model architecture."""
    return MlxFxGraphPlan(
        graph_module=graph_module,
        nodes=tuple(
            MlxFxNodePlan(
                node_name=node.name,
                node_op=node.op,
                target=node.target,
                lowering=registry.resolve(node),
            )
            for node in graph_module.graph.nodes
        ),
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


@dataclass(frozen=True)
class _LoweringSpec:
    """One MLX lowering and the FX targets that select it."""

    fn: Callable[..., Any]
    aten: tuple[Any, ...]
    functions: tuple[Any, ...]
    methods: tuple[str, ...]


_LOWERINGS: dict[str, _LoweringSpec] = {}


def _lowering(
    name: str,
    *,
    aten: Sequence[Any] = (),
    functions: Sequence[Any] = (),
    methods: Sequence[str] = (),
    aliases: Sequence[str] = (),
):
    """Register ``fn`` as MLX lowering ``name`` for the given FX targets.

    ``aten`` are the canonical overloads ``torch.export`` emits, ``functions``
    and ``methods`` the eager-FX spellings; ``aliases`` register the same
    function under additional lowering names (``reshape``/``view``).
    """

    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = _LoweringSpec(fn, tuple(aten), tuple(functions), tuple(methods))
        for key in (name, *aliases):
            if key in _LOWERINGS:
                raise ValueError(f"duplicate MLX lowering: {key}")
            _LOWERINGS[key] = spec
        return fn

    return register


def _lower_mlx_node(
    lowering: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    import mlx.core as mx

    spec = _LOWERINGS.get(lowering)
    if spec is None:
        raise UnsupportedMlxFxGraphError(f"missing MLX lowering: {lowering}")
    return spec.fn(mx, args, kwargs)


def _arg(args, kwargs, index, name, default=None):
    return args[index] if len(args) > index else kwargs.get(name, default)


@_lowering("assert_metadata", aten=(torch.ops.aten._assert_tensor_metadata.default,))
def _lower_assert_metadata(mx, args, kwargs):
    return None


@_lowering("alias", aten=(torch.ops.aten.alias.default,))
def _lower_alias(mx, args, kwargs):
    return args[0]


@_lowering("concatenate", aten=(torch.ops.aten.cat.default,))
def _lower_concatenate(mx, args, kwargs):
    return mx.concatenate(tuple(args[0]), axis=_arg(args, kwargs, 1, "dim", 0))


@_lowering(
    "embedding", aten=(torch.ops.aten.embedding.default,), functions=(F.embedding,)
)
def _lower_embedding(mx, args, kwargs):
    if isinstance(args[0], mx.array) and args[0].ndim >= 2:
        weight, input_ids = args[:2]
    else:
        input_ids, weight = args[:2]
    if len(args) > 3 and args[3] is not None:
        raise UnsupportedMlxFxGraphError("embedding max_norm is unsupported")
    return mx.take(weight, input_ids, axis=0)


@_lowering(
    "rms_norm", aten=(torch.ops.aten.rms_norm.default,), functions=(torch.rms_norm,)
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


@_lowering("layer_norm", aten=(torch.ops.aten.layer_norm.default,))
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


@_lowering(
    "linear",
    aten=(torch.ops.aten.linear.default,),
    functions=(torch._C._nn.linear,),
)
def _lower_linear(mx, args, kwargs):
    value, weight = args[:2]
    bias = _arg(args, kwargs, 2, "bias")
    output = value @ mx.swapaxes(weight, -1, -2)
    return output if bias is None else output + bias


@_lowering("matmul", aten=(torch.ops.aten.matmul.default, torch.ops.aten.mm.default))
def _lower_matmul(mx, args, kwargs):
    return args[0] @ args[1]


@_lowering("mean", aten=(torch.ops.aten.mean.dim,))
def _lower_mean(mx, args, kwargs):
    axis = _arg(args, kwargs, 1, "dim")
    keepdims = _arg(args, kwargs, 2, "keepdim", False)
    dtype = kwargs.get("dtype")
    value = args[0] if dtype is None else args[0].astype(_mlx_dtype(dtype, mx))
    return mx.mean(value, axis=axis, keepdims=keepdims)


@_lowering(
    "sdpa",
    aten=(torch.ops.aten.scaled_dot_product_attention.default,),
    functions=(torch._C._nn.scaled_dot_product_attention,),
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
    if enable_gqa and query.shape[-3] != key.shape[-3]:
        repeats = query.shape[-3] // key.shape[-3]
        key = mx.repeat(key, repeats, axis=-3)
        value = mx.repeat(value, repeats, axis=-3)
    scale = float(scale) if scale is not None else 1.0 / sqrt(query.shape[-1])
    scores = (query @ mx.swapaxes(key, -1, -2)) * scale
    if causal:
        rows, columns = scores.shape[-2:]
        offset = columns - rows
        causal_mask = mx.triu(
            mx.full((rows, columns), -float("inf"), dtype=scores.dtype),
            k=1 + offset,
        )
        scores = scores + causal_mask
    if attention_mask is not None:
        scores = scores + attention_mask
    return mx.softmax(scores, axis=-1) @ value


@_lowering("silu", aten=(torch.ops.aten.silu.default,), functions=(F.silu,))
def _lower_silu(mx, args, kwargs):
    return mx.sigmoid(args[0]) * args[0]


@_lowering("sigmoid", aten=(torch.ops.aten.sigmoid.default,))
def _lower_sigmoid(mx, args, kwargs):
    return mx.sigmoid(args[0])


@_lowering("relu", aten=(torch.ops.aten.relu.default,))
def _lower_relu(mx, args, kwargs):
    return mx.maximum(args[0], 0)


@_lowering("gelu", aten=(torch.ops.aten.gelu.default,))
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


@_lowering("tanh", aten=(torch.ops.aten.tanh.default,))
def _lower_tanh(mx, args, kwargs):
    return mx.tanh(args[0])


@_lowering("stack", aten=(torch.ops.aten.stack.default,))
def _lower_stack(mx, args, kwargs):
    return mx.stack(tuple(args[0]), axis=_arg(args, kwargs, 1, "dim", 0))


@_lowering("getitem", aten=(operator.getitem,), functions=(operator.getitem,))
def _lower_getitem(mx, args, kwargs):
    return args[0][args[1]]


@_lowering("add", aten=(torch.ops.aten.add.Tensor,), functions=(operator.add,))
def _lower_add(mx, args, kwargs):
    alpha = _arg(args, kwargs, 2, "alpha", 1)
    if alpha == 1:
        return args[0] + args[1]
    return args[0] + args[1] * alpha


@_lowering("multiply", aten=(torch.ops.aten.mul.Tensor,), functions=(operator.mul,))
def _lower_multiply(mx, args, kwargs):
    return args[0] * args[1]


@_lowering("subtract", aten=(torch.ops.aten.sub.Tensor,))
def _lower_subtract(mx, args, kwargs):
    alpha = _arg(args, kwargs, 2, "alpha", 1)
    if alpha == 1:
        return args[0] - args[1]
    return args[0] - args[1] * alpha


@_lowering("chunk", aten=(torch.ops.aten.chunk.default,), methods=("chunk",))
def _lower_chunk(mx, args, kwargs):
    value, chunks = args[:2]
    axis = _arg(args, kwargs, 2, "dim", 0)
    width = value.shape[axis]
    chunk_size = (width + chunks - 1) // chunks
    indices = tuple(range(chunk_size, width, chunk_size))
    return tuple(mx.split(value, indices, axis=axis))


@_lowering("split", aten=(torch.ops.aten.split.Tensor,))
def _lower_split(mx, args, kwargs):
    value, split_size = args[:2]
    axis = _arg(args, kwargs, 2, "dim", 0)
    width = value.shape[axis]
    indices = tuple(range(split_size, width, split_size))
    return tuple(mx.split(value, indices, axis=axis))


@_lowering("split_with_sizes", aten=(torch.ops.aten.split_with_sizes.default,))
def _lower_split_with_sizes(mx, args, kwargs):
    value, sizes = args[:2]
    axis = _arg(args, kwargs, 2, "dim", 0)
    indices = []
    offset = 0
    for size in sizes[:-1]:
        offset += size
        indices.append(offset)
    return tuple(mx.split(value, tuple(indices), axis=axis))


@_lowering(
    "reshape",
    aten=(torch.ops.aten.reshape.default, torch.ops.aten.view.default),
    methods=("reshape",),
    aliases=("view",),
)
def _lower_reshape(mx, args, kwargs):
    shape = args[1] if len(args) == 2 and isinstance(args[1], Sequence) else args[1:]
    return mx.reshape(args[0], tuple(shape))


@_lowering("flatten", aten=(torch.ops.aten.flatten.using_ints,))
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


@_lowering("index_select", aten=(torch.ops.aten.index_select.default,))
def _lower_index_select(mx, args, kwargs):
    return mx.take(args[0], args[2], axis=args[1])


@_lowering("numpy_transpose", aten=(torch.ops.aten.numpy_T.default,))
def _lower_numpy_transpose(mx, args, kwargs):
    return mx.transpose(args[0], axes=tuple(reversed(range(args[0].ndim))))


@_lowering("power", aten=(torch.ops.aten.pow.Tensor_Scalar,))
def _lower_power(mx, args, kwargs):
    return mx.power(args[0], args[1])


@_lowering("rsqrt", aten=(torch.ops.aten.rsqrt.default,))
def _lower_rsqrt(mx, args, kwargs):
    return mx.rsqrt(args[0])


@_lowering("to_dtype", aten=(torch.ops.aten.to.dtype,))
def _lower_to_dtype(mx, args, kwargs):
    return args[0].astype(_mlx_dtype(args[1], mx))


@_lowering("unsqueeze", aten=(torch.ops.aten.unsqueeze.default,))
def _lower_unsqueeze(mx, args, kwargs):
    return mx.expand_dims(args[0], axis=args[1])


@_lowering("empty_like", aten=(torch.ops.aten.empty_like.default,))
def _lower_empty_like(mx, args, kwargs):
    dtype = kwargs.get("dtype")
    if dtype is None:
        return mx.empty_like(args[0])
    return mx.empty(args[0].shape, dtype=_mlx_dtype(dtype, mx))


@_lowering("slice", aten=(torch.ops.aten.slice.Tensor,))
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


@_lowering("transpose", methods=("transpose",))
def _lower_transpose(mx, args, kwargs):
    return mx.swapaxes(args[0], args[1], args[2])


@_lowering("contiguous", methods=("contiguous",))
def _lower_contiguous(mx, args, kwargs):
    return mx.contiguous(args[0])


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


def _resolve_graph_attr(graph_module: torch.fx.GraphModule, target: str) -> Any:
    """Resolve a ``get_attr`` target: parameter, buffer, or plain attribute.

    FX ``get_attr`` nodes are not limited to parameters and registered
    buffers: a module may hang constant tensors or scalars directly on
    itself, and ``torch.export``'s ``module()`` re-registers lifted tensor
    constants the same way.
    """
    if target in dict(graph_module.named_parameters()):
        return graph_module.get_parameter(target)
    if target in dict(graph_module.named_buffers()):
        return graph_module.get_buffer(target)
    module_path, _, attr_name = target.rpartition(".")
    owner = graph_module.get_submodule(module_path) if module_path else graph_module
    return getattr(owner, attr_name)


def make_mlx_fx_executor(
    plan: MlxFxGraphPlan,
    example_inputs: list[Any],
    *,
    shapeless: bool = False,
) -> Callable[..., Any]:
    """Build one compiled MLX callable for a fully admitted FX graph."""
    import mlx.core as mx

    from sglang.srt.utils.tensor_bridge import MlxTensorView, mlx_call_multi

    plan.require_fully_supported()
    attr_nodes = tuple(
        node for node in plan.graph_module.graph.nodes if node.op == "get_attr"
    )
    attr_values = tuple(
        _resolve_graph_attr(plan.graph_module, str(node.target)) for node in attr_nodes
    )
    # Parameters, buffers, and constant-tensor attributes ride as borrowed
    # views; non-tensor attributes (scalars, shapes) are captured by value.
    tensor_attr_nodes = tuple(
        node
        for node, value in zip(attr_nodes, attr_values)
        if isinstance(value, torch.Tensor)
    )
    constant_attr_values = {
        node: value
        for node, value in zip(attr_nodes, attr_values)
        if not isinstance(value, torch.Tensor)
    }
    attr_views = tuple(
        MlxTensorView(value) for value in attr_values if isinstance(value, torch.Tensor)
    )
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
            if node_plan.lowering is None:
                raise UnsupportedMlxFxGraphError(
                    f"missing lowering for admitted node {node.target}"
                )
            values[node] = _lower_mlx_node(node_plan.lowering, args, kwargs)
        raise UnsupportedMlxFxGraphError("FX graph has no output node")

    compiled_graph = mx.compile(mlx_graph, shapeless=shapeless)

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
        for node, view in zip(tensor_attr_nodes, attr_views):
            tensor = _resolve_graph_attr(plan.graph_module, str(node.target))
            if not view.matches(tensor):
                view.refresh(tensor)
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
        if str(spec.kind).endswith("USER_INPUT_MUTATION")
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
    ) -> None:
        self.registry = registry or MlxFxLoweringRegistry.standard_decoder()
        self.executor_factory = executor_factory
        self.fallback_to_torch = fallback_to_torch
        self.report_path = report_path
        self.plans: list[MlxFxGraphPlan] = []

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
                return graph_module.forward
            plan.require_fully_supported()
        if self.executor_factory is None:
            return graph_module.forward
        return self.executor_factory(plan, example_inputs)


__all__ = [
    "MlxFxCaptureBackend",
    "MlxFxGraphPlan",
    "MlxFxLoweringRegistry",
    "MlxFxNodePlan",
    "MlxExportPlan",
    "UnsupportedMlxFxGraphError",
    "build_mlx_fx_plan",
    "export_mlx_plan",
    "make_mlx_decode_export_executor",
    "make_mlx_fx_executor",
    "make_mlx_prefill_export_executor",
    "run_torch_decode_export_reference",
    "run_torch_prefill_export_reference",
]
