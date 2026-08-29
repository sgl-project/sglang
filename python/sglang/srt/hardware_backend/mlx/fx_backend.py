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

from sglang.srt.environ import envs


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
    """Maps FX operations to reusable MLX lowering names."""

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
        for target, lowering in (
            (F.embedding, "embedding"),
            (torch.rms_norm, "rms_norm"),
            (torch._C._nn.linear, "linear"),
            (torch._C._nn.scaled_dot_product_attention, "sdpa"),
            (F.silu, "silu"),
            (operator.getitem, "getitem"),
            (operator.add, "add"),
            (operator.mul, "multiply"),
        ):
            registry.register_function(target, lowering)
        for target in ("chunk", "reshape", "view", "transpose", "contiguous"):
            registry.register_method(target, target)
        return registry

    @classmethod
    def standard_export_decoder(cls) -> MlxFxLoweringRegistry:
        """Return canonical ATen lowerings produced by ``torch.export``."""
        registry = cls()
        for target, lowering in (
            (torch.ops.aten._assert_tensor_metadata.default, "assert_metadata"),
            (torch.ops.aten.alias.default, "alias"),
            (torch.ops.aten.cat.default, "concatenate"),
            (torch.ops.aten.chunk.default, "chunk"),
            (torch.ops.aten.embedding.default, "embedding"),
            (torch.ops.aten.empty_like.default, "empty_like"),
            (torch.ops.aten.flatten.using_ints, "flatten"),
            (torch.ops.aten.gelu.default, "gelu"),
            (torch.ops.aten.index_select.default, "index_select"),
            (torch.ops.aten.layer_norm.default, "layer_norm"),
            (torch.ops.aten.linear.default, "linear"),
            (torch.ops.aten.matmul.default, "matmul"),
            (torch.ops.aten.mean.dim, "mean"),
            (torch.ops.aten.mm.default, "linear"),
            (torch.ops.aten.numpy_T.default, "numpy_transpose"),
            (torch.ops.aten.pow.Tensor_Scalar, "power"),
            (torch.ops.aten.reshape.default, "reshape"),
            (torch.ops.aten.relu.default, "relu"),
            (torch.ops.aten.rms_norm.default, "rms_norm"),
            (torch.ops.aten.rsqrt.default, "rsqrt"),
            (torch.ops.aten.scaled_dot_product_attention.default, "sdpa"),
            (torch.ops.aten.silu.default, "silu"),
            (torch.ops.aten.sigmoid.default, "sigmoid"),
            (torch.ops.aten.stack.default, "stack"),
            (torch.ops.aten.tanh.default, "tanh"),
            (torch.ops.aten.add.Tensor, "add"),
            (torch.ops.aten.mul.Tensor, "multiply"),
            (torch.ops.aten.slice.Tensor, "slice"),
            (torch.ops.aten.split.Tensor, "split"),
            (torch.ops.aten.split_with_sizes.default, "split_with_sizes"),
            (torch.ops.aten.sub.Tensor, "subtract"),
            (torch.ops.aten.to.dtype, "to_dtype"),
            (torch.ops.aten.unsqueeze.default, "unsqueeze"),
            (torch.ops.aten.view.default, "reshape"),
            (operator.getitem, "getitem"),
        ):
            registry.register_function(target, lowering)
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


def _lower_mlx_node(
    lowering: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    import mlx.core as mx

    if lowering == "assert_metadata":
        return None
    if lowering == "alias":
        return args[0]
    if lowering == "concatenate":
        axis = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        return mx.concatenate(tuple(args[0]), axis=axis)

    if lowering == "embedding":
        if isinstance(args[0], mx.array) and args[0].ndim >= 2:
            weight, input_ids = args[:2]
        else:
            input_ids, weight = args[:2]
        if len(args) > 3 and args[3] is not None:
            raise UnsupportedMlxFxGraphError("embedding max_norm is unsupported")
        return mx.take(weight, input_ids, axis=0)
    if lowering == "rms_norm":
        value = args[0]
        weight = args[2] if len(args) > 2 else kwargs.get("weight")
        epsilon = args[3] if len(args) > 3 else kwargs.get("eps")
        if weight is None:
            raise UnsupportedMlxFxGraphError("MLX RMSNorm requires an explicit weight")
        if epsilon is None:
            epsilon = mx.finfo(value.dtype).eps
        return mx.fast.rms_norm(value, weight, float(epsilon))
    if lowering == "layer_norm":
        value, normalized_shape = args[:2]
        weight = args[2] if len(args) > 2 else kwargs.get("weight")
        bias = args[3] if len(args) > 3 else kwargs.get("bias")
        epsilon = args[4] if len(args) > 4 else kwargs.get("eps", 1e-5)
        if len(normalized_shape) != 1 or normalized_shape[0] != value.shape[-1]:
            raise UnsupportedMlxFxGraphError(
                "MLX layer_norm only normalizes the last axis, found "
                f"normalized_shape={tuple(normalized_shape)}"
            )
        return mx.fast.layer_norm(value, weight, bias, float(epsilon))
    if lowering == "linear":
        value, weight = args[:2]
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        output = value @ mx.swapaxes(weight, -1, -2)
        return output if bias is None else output + bias
    if lowering == "matmul":
        return args[0] @ args[1]
    if lowering == "mean":
        axis = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdims = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        dtype = kwargs.get("dtype")
        value = args[0] if dtype is None else args[0].astype(_mlx_dtype(dtype, mx))
        return mx.mean(value, axis=axis, keepdims=keepdims)
    if lowering == "sdpa":
        query, key, value = args[:3]
        attention_mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
        dropout = args[4] if len(args) > 4 else kwargs.get("dropout_p", 0.0)
        causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)
        scale = args[6] if len(args) > 6 else kwargs.get("scale")
        enable_gqa = args[7] if len(args) > 7 else kwargs.get("enable_gqa", False)
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
    if lowering == "silu":
        return mx.sigmoid(args[0]) * args[0]
    if lowering == "sigmoid":
        return mx.sigmoid(args[0])
    if lowering == "relu":
        return mx.maximum(args[0], 0)
    if lowering == "gelu":
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
    if lowering == "tanh":
        return mx.tanh(args[0])
    if lowering == "stack":
        axis = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        return mx.stack(tuple(args[0]), axis=axis)
    if lowering == "getitem":
        return args[0][args[1]]
    if lowering == "add":
        return args[0] + args[1]
    if lowering == "multiply":
        return args[0] * args[1]
    if lowering == "subtract":
        alpha = kwargs.get("alpha", 1)
        return args[0] - args[1] * alpha
    if lowering == "chunk":
        value, chunks = args[:2]
        axis = args[2] if len(args) > 2 else kwargs.get("dim", 0)
        width = value.shape[axis]
        chunk_size = (width + chunks - 1) // chunks
        indices = tuple(range(chunk_size, width, chunk_size))
        return tuple(mx.split(value, indices, axis=axis))
    if lowering == "split":
        value, split_size = args[:2]
        axis = args[2] if len(args) > 2 else kwargs.get("dim", 0)
        width = value.shape[axis]
        indices = tuple(range(split_size, width, split_size))
        return tuple(mx.split(value, indices, axis=axis))
    if lowering == "split_with_sizes":
        value, sizes = args[:2]
        axis = args[2] if len(args) > 2 else kwargs.get("dim", 0)
        indices = []
        offset = 0
        for size in sizes[:-1]:
            offset += size
            indices.append(offset)
        return tuple(mx.split(value, tuple(indices), axis=axis))
    if lowering in {"reshape", "view"}:
        shape = (
            args[1] if len(args) == 2 and isinstance(args[1], Sequence) else args[1:]
        )
        return mx.reshape(args[0], tuple(shape))
    if lowering == "flatten":
        value = args[0]
        start = args[1] if len(args) > 1 else 0
        end = args[2] if len(args) > 2 else -1
        if end < 0:
            end += value.ndim
        flattened = 1
        for width in value.shape[start : end + 1]:
            flattened *= width
        shape = (*value.shape[:start], flattened, *value.shape[end + 1 :])
        return mx.reshape(value, shape)
    if lowering == "index_select":
        return mx.take(args[0], args[2], axis=args[1])
    if lowering == "numpy_transpose":
        return mx.transpose(args[0], axes=tuple(reversed(range(args[0].ndim))))
    if lowering == "power":
        return mx.power(args[0], args[1])
    if lowering == "rsqrt":
        return mx.rsqrt(args[0])
    if lowering == "to_dtype":
        return args[0].astype(_mlx_dtype(args[1], mx))
    if lowering == "unsqueeze":
        return mx.expand_dims(args[0], axis=args[1])
    if lowering == "empty_like":
        dtype = kwargs.get("dtype")
        if dtype is None:
            return mx.empty_like(args[0])
        return mx.empty(args[0].shape, dtype=_mlx_dtype(dtype, mx))
    if lowering == "slice":
        value = args[0]
        axis = args[1] if len(args) > 1 else 0
        start = args[2] if len(args) > 2 else 0
        stop = args[3] if len(args) > 3 else value.shape[axis]
        step = args[4] if len(args) > 4 else 1
        width = value.shape[axis]
        if start >= 0:
            start = min(start, width)
        if stop >= 0:
            stop = min(stop, width)
        slices = [slice(None)] * value.ndim
        slices[axis] = slice(start, stop, step)
        return value[tuple(slices)]
    if lowering == "transpose":
        return mx.swapaxes(args[0], args[1], args[2])
    if lowering == "contiguous":
        return mx.contiguous(args[0])
    raise UnsupportedMlxFxGraphError(f"missing MLX lowering: {lowering}")


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
    attr_views = tuple(
        (
            MlxTensorView(plan.graph_module.get_parameter(str(node.target)))
            if str(node.target) in dict(plan.graph_module.named_parameters())
            else MlxTensorView(plan.graph_module.get_buffer(str(node.target)))
        )
        for node in attr_nodes
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
        values.update(zip(attr_nodes, captured_arrays))
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
        for node, view in zip(attr_nodes, attr_views):
            target = str(node.target)
            try:
                tensor = plan.graph_module.get_parameter(target)
            except AttributeError:
                tensor = plan.graph_module.get_buffer(target)
            if not view.matches(tensor):
                view.refresh(tensor)
        return mlx_call_multi(
            compiled_graph,
            *(torch_inputs[index] for index in tensor_positions),
            *attr_views,
            device="mps",
        )

    return execute


def _make_mlx_export_executor(
    exported_program: Any,
    example_inputs: tuple[Any, ...],
    *,
    mode: str,
) -> Callable[..., torch.Tensor]:
    """Lower one admitted strict serving export to MLX plus deferred KV commit."""
    import os

    import mlx.core as mx

    from sglang.kernels.ops.attention.mlx_kv_commit import (
        commit_deferred_kv,
        verify_deferred_kv_commit,
    )
    from sglang.kernels.ops.attention.mlx_radix_attention import (
        DeferredAttentionSpec,
        causal_gqa,
        radix_decode_deferred,
        radix_prefill_deferred,
    )
    from sglang.srt.utils.tensor_bridge import MlxTensorView, mlx_call_multi

    graph_module = exported_program.module()
    for node in tuple(graph_module.graph.nodes):
        if node.op == "call_module" and str(node.target) == "_guards_fn":
            if node.users:
                raise UnsupportedMlxFxGraphError("export guard unexpectedly has users")
            graph_module.graph.erase_node(node)
    graph_module.recompile()

    registry = MlxFxLoweringRegistry.standard_export_decoder()
    attention_target = torch.ops.sglang.unified_attention_with_output.default
    registry.register_function(attention_target, "radix_decode")
    plan = build_mlx_fx_plan(graph_module, registry)
    plan.require_fully_supported()

    attr_nodes = tuple(
        node for node in graph_module.graph.nodes if node.op == "get_attr"
    )
    attr_tensors = tuple(_get_graph_tensor(graph_module, node) for node in attr_nodes)
    attr_views = tuple(MlxTensorView(tensor) for tensor in attr_tensors)
    placeholder_nodes = tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholder_nodes) != len(example_inputs):
        raise UnsupportedMlxFxGraphError(
            "decode export placeholder and example-input counts do not match"
        )
    if mode not in {"decode", "prefill"}:
        raise ValueError(f"unsupported MLX serving export mode: {mode}")
    tensor_positions = tuple(
        index
        for index, value in enumerate(example_inputs)
        if isinstance(value, torch.Tensor)
    )
    tensor_placeholders = tuple(placeholder_nodes[index] for index in tensor_positions)
    used_non_tensors = tuple(
        placeholder_nodes[index]
        for index, value in enumerate(example_inputs)
        if not isinstance(value, torch.Tensor) and placeholder_nodes[index].users
    )
    if used_non_tensors:
        raise UnsupportedMlxFxGraphError(
            "decode export contains used non-tensor runtime inputs"
        )

    attention_nodes = tuple(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target is attention_target
    )
    k_pools = tuple(
        _get_graph_tensor(graph_module, node.kwargs["k_cache"])
        for node in attention_nodes
    )
    v_pools = tuple(
        _get_graph_tensor(graph_module, node.kwargs["v_cache"])
        for node in attention_nodes
    )
    if k_pools:
        # Loud launch-time failure beats silently serving a pool whose
        # commits vanish (compile_shader write loss near the working-set
        # limit); probes the real buffers through the same kernel.
        verify_deferred_kv_commit(
            list(k_pools),
            list(v_pools),
            num_kv_heads=int(k_pools[0].shape[1]),
            head_dim=int(k_pools[0].shape[2]),
        )
    from sglang.srt.hardware_backend.mlx.export_validation import ServingForwardArg

    out_cache_position = ServingForwardArg.OUT_CACHE_LOC
    prefix_lens_position = ServingForwardArg.EXTEND_PREFIX_LENS
    debug_attention = envs.SGLANG_DEBUG_MLX_EXPORT_ATTENTION.get()

    def make_mlx_graph(prefill_attention: str):
        def mlx_graph(*arrays):
            return _run_mlx_graph(prefill_attention, *arrays)

        return mlx_graph

    def _run_mlx_graph(prefill_attention, *arrays):
        runtime_arrays = arrays[: len(tensor_placeholders)]
        captured_arrays = arrays[len(tensor_placeholders) :]
        values: dict[torch.fx.Node, Any] = dict(
            zip(tensor_placeholders, runtime_arrays)
        )
        values.update(zip(attr_nodes, captured_arrays))
        new_keys = []
        new_values = []
        attention_outputs = []
        first_query = None
        first_key = None
        first_value = None
        for node, node_plan in zip(graph_module.graph.nodes, plan.nodes):
            if node.op in {"placeholder", "get_attr"}:
                continue
            args = _resolve_fx_value(node.args, values)
            kwargs = _resolve_fx_value(node.kwargs, values)
            if node.op == "output":
                result = args[0]
                outputs = (
                    tuple(result) if isinstance(result, (tuple, list)) else (result,)
                )
                if debug_attention:
                    return (
                        *outputs,
                        mx.stack(attention_outputs),
                        first_query,
                        first_key,
                        first_value,
                        mx.stack(new_keys),
                        mx.stack(new_values),
                    )
                return (*outputs, mx.stack(new_keys), mx.stack(new_values))
            if node.target is attention_target:
                query, key, value, output = args[:4]
                k_cache = kwargs["k_cache"]
                num_kv_heads = k_cache.shape[1]
                head_dim = k_cache.shape[2]
                spec = DeferredAttentionSpec(
                    num_q_heads=query.shape[-1] // head_dim,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
                query = query.reshape(query.shape[0], spec.num_q_heads, head_dim)
                key = key.reshape(key.shape[0], num_kv_heads, head_dim)
                value = value.reshape(value.shape[0], num_kv_heads, head_dim)
                if mode == "decode":
                    query = mx.contiguous(query)
                    key = mx.contiguous(key)
                    value = mx.contiguous(value)
                    attention = radix_decode_deferred(
                        query,
                        key,
                        value,
                        k_cache,
                        kwargs["v_cache"],
                        kwargs["req_to_token"],
                        kwargs["req_pool_indices"],
                        kwargs["seq_lens"],
                        spec=spec,
                    )
                elif prefill_attention == "causal":
                    # Single request, no cached prefix: plain causal SDPA is
                    # exactly the same math and ~15x faster than the radix
                    # kernel at prefill shapes. Selection happens per call in
                    # execute(); this graph is only run when it applies.
                    query = mx.contiguous(query)
                    key = mx.contiguous(key)
                    value = mx.contiguous(value)
                    attention = causal_gqa(query, key, value, spec=spec)
                else:
                    query = mx.contiguous(query)
                    key = mx.contiguous(key)
                    value = mx.contiguous(value)
                    attention = radix_prefill_deferred(
                        query,
                        key,
                        value,
                        k_cache,
                        kwargs["v_cache"],
                        kwargs["req_to_token"],
                        kwargs["req_pool_indices"],
                        kwargs["extend_prefix_lens"],
                        kwargs["extend_seq_lens"],
                        spec=spec,
                    )
                output[:] = attention.reshape(output.shape)
                attention_outputs.append(attention)
                if first_query is None:
                    first_query = query
                    first_key = key
                if first_value is None:
                    first_value = value
                new_keys.append(key)
                new_values.append(value)
                values[node] = None
                continue
            if node_plan.lowering is None:
                raise UnsupportedMlxFxGraphError(
                    f"missing lowering for admitted node {node.target}"
                )
            values[node] = _lower_mlx_node(node_plan.lowering, args, kwargs)
        raise UnsupportedMlxFxGraphError("decode export has no output node")

    compiled_graph = mx.compile(make_mlx_graph("radix"), shapeless=False)
    # Packed multi-request batches and radix-prefix reuse need the custom
    # kernel; the plain-causal fast path is only valid for one request with
    # no cached prefix, so it exists only for batch-size-1 exports.
    causal_prefill_graph = None
    if (
        mode == "prefill"
        and example_inputs[ServingForwardArg.REQ_POOL_INDICES].shape[0] == 1
    ):
        causal_prefill_graph = mx.compile(make_mlx_graph("causal"), shapeless=False)

    def execute(*runtime_inputs):
        tensor_inputs = tuple(runtime_inputs[index] for index in tensor_positions)
        graph = compiled_graph
        if (
            causal_prefill_graph is not None
            and int(runtime_inputs[prefix_lens_position].max()) == 0
        ):
            graph = causal_prefill_graph
        results = mlx_call_multi(
            graph,
            *tensor_inputs,
            *attr_views,
            device="mps",
        )
        expected_results = 7 if debug_attention else 3
        if len(results) != expected_results:
            raise RuntimeError(
                "MLX graph expected logits, optional debug attention, and two KV "
                f"deltas; got {len(results)}"
            )
        if debug_attention:
            logits, all_attention, first_query, first_key, first_value, new_k, new_v = (
                results
            )
        else:
            logits, new_k, new_v = results
        if envs.SGLANG_DEBUG_MLX_KV_DELTAS.get():
            # new_k/new_v are zero-copy views of MLX-owned buffers; MLX may
            # reuse those buffers on the next region run, so a stash held
            # across runs must own its storage.
            execute.last_new_k = new_k.clone()
            execute.last_new_v = new_v.clone()
        first_k_pool = k_pools[0]
        spec = DeferredAttentionSpec(
            num_q_heads=(
                attention_nodes[0].args[0].meta["val"].shape[-1]
                // first_k_pool.shape[2]
            ),
            num_kv_heads=first_k_pool.shape[1],
            head_dim=first_k_pool.shape[2],
        )
        commit_deferred_kv(
            new_k,
            new_v,
            runtime_inputs[out_cache_position],
            k_pools,
            v_pools,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
        )
        return (
            (logits, all_attention, first_query, first_key, first_value)
            if debug_attention
            else logits
        )

    return execute


def make_mlx_decode_export_executor(
    exported_program: Any,
    example_inputs: tuple[Any, ...],
) -> Callable[..., torch.Tensor]:
    """Build the single-region MLX executor for one decode export bucket."""
    return _make_mlx_export_executor(exported_program, example_inputs, mode="decode")


def make_mlx_prefill_export_executor(
    exported_program: Any,
    example_inputs: tuple[Any, ...],
) -> Callable[..., torch.Tensor]:
    """Build the single-region MLX executor for one packed Radix prefill bucket."""
    return _make_mlx_export_executor(exported_program, example_inputs, mode="prefill")


def _get_graph_tensor(
    graph_module: torch.fx.GraphModule, node: torch.fx.Node
) -> torch.Tensor:
    if node.op != "get_attr":
        raise UnsupportedMlxFxGraphError(
            f"expected a bound graph tensor, found {node.op}:{node.target}"
        )
    target = str(node.target)
    try:
        return graph_module.get_parameter(target)
    except AttributeError:
        pass
    try:
        return graph_module.get_buffer(target)
    except AttributeError:
        pass
    # Non-strict export stores lifted closure constants (`lifted_tensor_*`)
    # as plain module attributes, reachable only by attribute traversal.
    value = graph_module
    for part in target.split("."):
        value = getattr(value, part)
    if not isinstance(value, torch.Tensor):
        raise UnsupportedMlxFxGraphError(
            f"graph attribute {target} is not a tensor: {type(value)}"
        )
    return value


def run_torch_decode_export_reference(
    exported_program: Any,
    runtime_inputs: tuple[Any, ...],
    *,
    return_attention: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute the same decode export with a small Torch attention reference."""
    graph_module = exported_program.module()
    attention_target = torch.ops.sglang.unified_attention_with_output.default
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target is attention_target:
            node.target = _torch_radix_decode_reference
    graph_module.recompile()
    _TORCH_DECODE_DEBUG_ATTENTION.clear()
    logits = graph_module(*runtime_inputs)
    if return_attention:
        return logits, torch.stack(_TORCH_DECODE_DEBUG_ATTENTION)
    return logits


_TORCH_DECODE_DEBUG_ATTENTION: list[torch.Tensor] = []
_TORCH_PREFILL_DEBUG_ATTENTION: list[torch.Tensor] = []
_TORCH_PREFILL_DEBUG_QKV: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []


def run_torch_prefill_export_reference(
    exported_program: Any,
    runtime_inputs: tuple[Any, ...],
    *,
    return_first_attention: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute a prefill export with the deferred Radix attention contract."""
    graph_module = exported_program.module()
    attention_target = torch.ops.sglang.unified_attention_with_output.default
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target is attention_target:
            node.target = _torch_radix_prefill_reference
    graph_module.recompile()
    _TORCH_PREFILL_DEBUG_ATTENTION.clear()
    _TORCH_PREFILL_DEBUG_QKV.clear()
    logits = graph_module(*runtime_inputs)
    if return_first_attention:
        return (
            logits,
            torch.stack(_TORCH_PREFILL_DEBUG_ATTENTION),
            _TORCH_PREFILL_DEBUG_QKV[0],
        )
    return logits


def _torch_radix_prefill_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    out_cache_loc: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    **_: Any,
) -> None:
    del layer_id
    num_kv_heads, head_dim = k_cache.shape[1:]
    num_q_heads = query.shape[-1] // head_dim
    q = query.reshape(query.shape[0], num_q_heads, head_dim)
    current_k = key.reshape(key.shape[0], num_kv_heads, head_dim)
    current_v = value.reshape(value.shape[0], num_kv_heads, head_dim)
    _TORCH_PREFILL_DEBUG_QKV.append(
        (q.detach().clone(), current_k.detach().clone(), current_v.detach().clone())
    )
    q_per_kv = num_q_heads // num_kv_heads
    rows = []
    request_start = 0
    for batch in range(req_pool_indices.shape[0]):
        request = int(req_pool_indices[batch].item())
        prefix_length = int(extend_prefix_lens[batch].item())
        extension_length = int(extend_seq_lens[batch].item())
        prefix_slots = req_to_token[request, :prefix_length].long()
        cached_k = k_cache[prefix_slots]
        cached_v = v_cache[prefix_slots]
        for offset in range(extension_length):
            output_index = request_start + offset
            keys = torch.cat(
                (cached_k, current_k[request_start : output_index + 1])
            ).repeat_interleave(q_per_kv, dim=1)
            values = torch.cat(
                (cached_v, current_v[request_start : output_index + 1])
            ).repeat_interleave(q_per_kv, dim=1)
            scores = torch.einsum("hd,thd->ht", q[output_index].float(), keys.float())
            probabilities = torch.softmax(scores * (head_dim**-0.5), dim=-1)
            rows.append(
                torch.einsum("ht,thd->hd", probabilities, values.float()).to(q.dtype)
            )
        request_start += extension_length
    output.copy_(torch.stack(rows).reshape_as(output))
    _TORCH_PREFILL_DEBUG_ATTENTION.append(output.detach().clone())
    if save_kv_cache:
        k_cache.index_copy_(0, out_cache_loc, current_k)
        v_cache.index_copy_(0, out_cache_loc, current_v)


def _torch_radix_decode_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    **_: Any,
) -> None:
    del layer_id
    num_kv_heads, head_dim = k_cache.shape[1:]
    num_q_heads = query.shape[-1] // head_dim
    q = query.reshape(query.shape[0], num_q_heads, head_dim)
    current_k = key.reshape(key.shape[0], num_kv_heads, head_dim)
    current_v = value.reshape(value.shape[0], num_kv_heads, head_dim)
    rows = []
    q_per_kv = num_q_heads // num_kv_heads
    for batch in range(q.shape[0]):
        sequence_length = int(seq_lens[batch].item())
        request = int(req_pool_indices[batch].item())
        prefix_slots = req_to_token[request, : sequence_length - 1].long()
        keys = torch.cat((k_cache[prefix_slots], current_k[batch : batch + 1]))
        values = torch.cat((v_cache[prefix_slots], current_v[batch : batch + 1]))
        keys = keys.repeat_interleave(q_per_kv, dim=1)
        values = values.repeat_interleave(q_per_kv, dim=1)
        scores = torch.einsum("hd,thd->ht", q[batch].float(), keys.float())
        probabilities = torch.softmax(scores * (head_dim**-0.5), dim=-1)
        rows.append(
            torch.einsum("ht,thd->hd", probabilities, values.float()).to(q.dtype)
        )
    output.copy_(torch.stack(rows).reshape_as(output))
    _TORCH_DECODE_DEBUG_ATTENTION.append(output.detach().clone())
    if save_kv_cache:
        k_cache.index_copy_(0, out_cache_loc, current_k)
        v_cache.index_copy_(0, out_cache_loc, current_v)


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
