"""Torch-owned exported inference graphs compiled directly by MLX."""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable, Mapping

import mlx.core as mx
import torch
from torch.export.graph_signature import InputKind, OutputKind

from sglang.srt.hardware_backend.mps.compiled_region import UnsupportedMlxRegion
from sglang.srt.utils.tensor_bridge import MlxTensorView

logger = logging.getLogger(__name__)

_SUPPORTED_OPS = frozenset(
    """
    aten.view.default aten.reshape.default aten._unsafe_view.default
    aten.clone.default aten.contiguous.default aten.detach.default aten.alias.default
    aten._assert_tensor_metadata.default aten.to.dtype aten._to_copy.default
    aten.linear.default aten.mm.default aten.bmm.default aten.matmul.default
    aten.addmm.default aten.t.default aten.transpose.int aten.permute.default
    aten.embedding.default aten.index_select.default aten.index.Tensor
    aten.select.int aten.slice.Tensor aten.unsqueeze.default aten.squeeze.dim
    aten.cat.default aten.stack.default aten.split.Tensor aten.split_with_sizes.default
    aten.expand.default aten.add.Tensor aten.add.Scalar aten.sub.Tensor aten.sub.Scalar
    aten.mul.Tensor aten.mul.Scalar aten.div.Tensor aten.div.Scalar
    aten.pow.Tensor_Scalar aten.rsqrt.default aten.sqrt.default aten.cos.default
    aten.sin.default aten.neg.default aten.sigmoid.default aten.silu.default
    aten.mean.dim aten.rms_norm.default aten.layer_norm.default
    aten.native_layer_norm.default aten.gelu.default aten.relu.default aten.tanh.default
    """.split()
)


def host_alias(tensor: torch.Tensor) -> torch.Tensor:
    """Borrow contiguous MPS storage; the caller owns both synchronization fences."""
    if not tensor.is_contiguous():
        raise ValueError("Compiled MLX requires contiguous CPU or MPS tensors")
    if tensor.device.type == "cpu":
        return tensor.detach()
    if tensor.device.type != "mps":
        raise ValueError("Compiled MLX requires contiguous CPU or MPS tensors")
    try:
        from torch.mps import _host_alias_storage
    except ImportError as error:
        raise RuntimeError(
            "Compiled MLX requires Torch with torch.mps._host_alias_storage "
            "(tested with Torch 2.13.0)"
        ) from error
    return torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
        _host_alias_storage(tensor.untyped_storage()),
        tensor.storage_offset(),
        tensor.shape,
        tensor.stride(),
    )


def _signature(tensor):
    return tensor.shape, tensor.dtype, tensor.stride(), tensor.device


def _dtype(dtype):
    names = {
        torch.float32: mx.float32,
        torch.float16: mx.float16,
        torch.bfloat16: mx.bfloat16,
        torch.int64: mx.int64,
        torch.int32: mx.int32,
        torch.bool: mx.bool_,
    }
    return names[dtype]


def _lower(target, args, kw):
    if target is operator.getitem:
        return args[0][args[1]]
    name = str(target)
    x = args[0] if args else None
    if name in (
        "aten.view.default",
        "aten.reshape.default",
        "aten._unsafe_view.default",
    ):
        return mx.reshape(x, args[1])
    if name in (
        "aten.clone.default",
        "aten.contiguous.default",
        "aten.detach.default",
        "aten.alias.default",
    ):
        return x
    if name == "aten._assert_tensor_metadata.default":
        return None
    if name in ("aten.to.dtype", "aten._to_copy.default"):
        dtype = kw.get("dtype") if name == "aten._to_copy.default" else args[1]
        return x if dtype is None else x.astype(_dtype(dtype))
    if name == "aten.linear.default":
        out = x @ args[1].T
        return out if len(args) < 3 or args[2] is None else out + args[2]
    if name in ("aten.mm.default", "aten.bmm.default", "aten.matmul.default"):
        return x @ args[1]
    if name == "aten.addmm.default":
        return kw.get("beta", 1) * x + kw.get("alpha", 1) * (args[1] @ args[2])
    if name == "aten.t.default":
        return x.T
    if name == "aten.transpose.int":
        return mx.swapaxes(x, args[1], args[2])
    if name == "aten.permute.default":
        return mx.transpose(x, args[1])
    if name == "aten.embedding.default":
        return x[args[1].astype(mx.int32)]
    if name == "aten.index_select.default":
        return mx.take(x, args[2].astype(mx.int32), axis=args[1])
    if name == "aten.index.Tensor":
        return x[
            tuple(slice(None) if i is None else i.astype(mx.int32) for i in args[1])
        ]
    if name == "aten.select.int":
        index = [slice(None)] * x.ndim
        index[args[1]] = args[2]
        return x[tuple(index)]
    if name == "aten.slice.Tensor":
        index = [slice(None)] * x.ndim
        dim = args[1] if len(args) > 1 else 0
        index[dim] = slice(
            args[2] if len(args) > 2 else None,
            min(args[3], x.shape[dim]) if len(args) > 3 else None,
            args[4] if len(args) > 4 else 1,
        )
        return x[tuple(index)]
    if name == "aten.unsqueeze.default":
        return mx.expand_dims(x, args[1])
    if name == "aten.squeeze.dim":
        return mx.squeeze(x, axis=args[1]) if x.shape[args[1]] == 1 else x
    if name in ("aten.cat.default", "aten.stack.default"):
        fn = mx.concatenate if name == "aten.cat.default" else mx.stack
        return fn(x, axis=args[1] if len(args) > 1 else 0)
    if name in ("aten.split.Tensor", "aten.split_with_sizes.default"):
        dim = args[2] if len(args) > 2 else 0
        if isinstance(args[1], int):
            points = list(range(args[1], x.shape[dim], args[1]))
        else:
            points, total = [], 0
            for size in args[1][:-1]:
                total += size
                points.append(total)
        return mx.split(x, points, axis=dim)
    if name == "aten.expand.default":
        padded = (1,) * (len(args[1]) - x.ndim) + x.shape
        shape = [old if new == -1 else new for old, new in zip(padded, args[1])]
        return mx.broadcast_to(x, shape)
    if name in ("aten.add.Tensor", "aten.add.Scalar"):
        return x + args[1] * kw.get("alpha", 1)
    if name in ("aten.sub.Tensor", "aten.sub.Scalar"):
        return x - args[1] * kw.get("alpha", 1)
    if name in ("aten.mul.Tensor", "aten.mul.Scalar"):
        return x * args[1]
    if name in ("aten.div.Tensor", "aten.div.Scalar"):
        return x / args[1]
    if name == "aten.pow.Tensor_Scalar":
        return mx.power(x, args[1])
    unary = {
        "aten.rsqrt.default": mx.rsqrt,
        "aten.sqrt.default": mx.sqrt,
        "aten.cos.default": mx.cos,
        "aten.sin.default": mx.sin,
        "aten.neg.default": mx.negative,
        "aten.sigmoid.default": mx.sigmoid,
    }
    if name in unary:
        return unary[name](x)
    if name == "aten.silu.default":
        value = x.astype(mx.float32)
        return (value * mx.sigmoid(value)).astype(x.dtype)
    if name == "aten.relu.default":
        return mx.maximum(x, 0)
    if name == "aten.tanh.default":
        return mx.tanh(x)
    if name == "aten.gelu.default":
        value = x.astype(mx.float32)
        if kw.get("approximate", "none") == "tanh":
            out = (
                0.5
                * value
                * (
                    1
                    + mx.tanh(
                        (2 / 3.141592653589793) ** 0.5 * (value + 0.044715 * value**3)
                    )
                )
            )
        else:
            out = 0.5 * value * (1 + mx.erf(value / 2**0.5))
        return out.astype(x.dtype)
    if name in ("aten.layer_norm.default", "aten.native_layer_norm.default"):
        axes = tuple(range(x.ndim - len(args[1]), x.ndim))
        value = x.astype(mx.float32)
        mean = mx.mean(value, axis=axes, keepdims=True)
        variance = mx.mean(mx.square(value - mean), axis=axes, keepdims=True)
        eps = args[4] if len(args) > 4 else 1e-5
        inverse = mx.rsqrt(variance + eps)
        out = (value - mean) * inverse
        if len(args) > 2 and args[2] is not None:
            out = out * args[2].astype(mx.float32)
        if len(args) > 3 and args[3] is not None:
            out = out + args[3].astype(mx.float32)
        out = out.astype(x.dtype)
        if name == "aten.native_layer_norm.default":
            stats_dtype = args[2].dtype if args[2] is not None else x.dtype
            return out, mean.astype(stats_dtype), inverse.astype(stats_dtype)
        return out
    if name == "aten.mean.dim":
        return mx.mean(
            x, axis=tuple(args[1]), keepdims=args[2] if len(args) > 2 else False
        )
    if name == "aten.rms_norm.default":
        if tuple(args[1]) != (x.shape[-1],):
            raise ValueError("Direct MLX RMSNorm requires a single normalized axis")
        weight = args[2] if len(args) > 2 else None
        eps = args[3] if len(args) > 3 else None
        return mx.fast.rms_norm(
            x, weight, mx.finfo(x.dtype).eps if eps is None else eps
        )
    raise ValueError(f"No direct MLX lowering for {target}")


def _rms_pattern(node):
    """Recognize x * rsqrt(mean(x**2, -1, keepdim=True) + eps)."""
    if str(node.target) != "aten.mul.Tensor":
        return None
    for x, inverse in (node.args, node.args[::-1]):
        if (
            not isinstance(inverse, torch.fx.Node)
            or str(inverse.target) != "aten.rsqrt.default"
        ):
            continue
        add = inverse.args[0]
        if (
            not isinstance(add, torch.fx.Node)
            or str(add.target) not in ("aten.add.Tensor", "aten.add.Scalar")
            or add.kwargs.get("alpha", 1) != 1
        ):
            continue
        mean, eps = add.args[:2]
        if (
            not isinstance(eps, (float, int))
            or not isinstance(mean, torch.fx.Node)
            or str(mean.target) != "aten.mean.dim"
        ):
            continue
        power = mean.args[0]
        if (
            isinstance(power, torch.fx.Node)
            and isinstance(x, torch.fx.Node)
            and x.meta["val"].dtype == torch.float32
            and str(power.target) == "aten.pow.Tensor_Scalar"
            and power.args == (x, 2)
            and list(mean.args[1]) == [-1]
            and len(mean.args) > 2
            and mean.args[2]
        ):
            return x, eps
    return None


class CompiledMlxGraph:
    """Fixed-shape, read-only export with retained, replaceable Torch storage."""

    @torch.no_grad()
    def __init__(
        self,
        *,
        model,
        example_inputs,
        attention=None,
        custom_lowerings: Mapping[torch._ops.OpOverload, Callable] | None = None,
    ):
        self.model = model
        self.attention = attention
        self.custom_lowerings = dict(custom_lowerings or {})
        self.input_signatures = tuple(_signature(x) for x in example_inputs)
        exported = torch.export.export(model, example_inputs, strict=False)
        exported = exported.run_decompositions({})
        if any(
            s.kind != OutputKind.USER_OUTPUT
            for s in exported.graph_signature.output_specs
        ):
            raise UnsupportedMlxRegion(
                "Direct MLX graphs must not mutate inputs or buffers"
            )
        self.nodes = tuple(exported.graph.nodes)
        for node in self.nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node.op != "call_function":
                raise UnsupportedMlxRegion(
                    f"Unsupported direct MLX graph node: {node.op}"
                )
            if node.target is operator.getitem or node.target in self.custom_lowerings:
                continue
            if str(node.target) == "sglang.mlx_radix_decode.default":
                if self.attention is not None:
                    continue
                raise UnsupportedMlxRegion("Radix attention lowering was not provided")
            if str(node.target) not in _SUPPORTED_OPS:
                raise UnsupportedMlxRegion(f"No direct MLX lowering for {node.target}")
        self.bindings = []
        self.attribute_signatures = {}
        self.views = {}
        self.closed = False
        user_index = 0
        for spec in exported.graph_signature.input_specs:
            if spec.kind == InputKind.USER_INPUT:
                self.bindings.append((spec.kind, user_index))
                user_index += 1
            elif spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                self.bindings.append((spec.kind, spec.target))
                tensor = (
                    model.get_parameter(spec.target)
                    if spec.kind == InputKind.PARAMETER
                    else model.get_buffer(spec.target)
                )
                self.attribute_signatures[spec.target] = _signature(tensor)
            elif spec.kind == InputKind.CONSTANT_TENSOR:
                self.bindings.append((spec.kind, exported.constants[spec.target]))
            else:
                raise ValueError(f"Unsupported graph input kind: {spec.kind}")
        self.norms = {
            node: pattern for node in self.nodes if (pattern := _rms_pattern(node))
        }
        logger.info(
            "Direct MLX export: %d nodes, %d fused RMSNorms",
            len(self.nodes),
            len(self.norms),
        )
        self.compiled = {
            tail: mx.compile(self._function(tail), shapeless=False)
            for tail in (False, True)
        }
        self.execution_count = 0

    def _function(self, tail):
        def forward(*flat):
            values = {}
            position = layer = 0
            for node in self.nodes:
                if node.op == "placeholder":
                    values[node] = flat[position]
                    position += 1
                    continue

                def resolve(arg):
                    return torch.fx.node.map_arg(arg, values.__getitem__)

                if node.op == "output":
                    result = resolve(node.args[0])
                    return (
                        tuple(result)
                        if isinstance(result, (tuple, list))
                        else (result,)
                    )
                if node.op != "call_function":
                    raise ValueError(f"Unsupported direct MLX graph node: {node.op}")
                args, kw = resolve(node.args), resolve(node.kwargs)
                if node in self.norms:
                    x, eps = self.norms[node]
                    values[node] = mx.fast.rms_norm(values[x], None, eps)
                elif str(node.target) == "sglang.mlx_radix_decode.default":
                    if self.attention is None:
                        raise ValueError("Radix attention lowering was not provided")
                    tails = (flat[-2][layer], flat[-1][layer]) if tail else None
                    values[node] = self.attention(*args, tails=tails)
                    layer += 1
                elif node.target in self.custom_lowerings:
                    values[node] = self.custom_lowerings[node.target](*args, **kw)
                else:
                    values[node] = _lower(node.target, args, kw)
            raise ValueError("Direct MLX graph has no output")

        return forward

    def bind(self, inputs):
        if self.closed:
            raise RuntimeError("Direct MLX graph is closed")
        if tuple(_signature(x) for x in inputs) != self.input_signatures:
            raise ValueError("Direct MLX input shape, dtype, stride or device changed")
        # Export may bind an alias path rather than the first name of a tied tensor.
        parameters = dict(self.model.named_parameters(remove_duplicate=False))
        buffers = dict(self.model.named_buffers(remove_duplicate=False))
        arrays = []
        for index, (kind, target) in enumerate(self.bindings):
            if kind == InputKind.USER_INPUT:
                tensor = inputs[target]
            elif kind == InputKind.PARAMETER:
                tensor = parameters[target]
            elif kind == InputKind.BUFFER:
                tensor = buffers[target]
            else:
                tensor = target
            if (
                kind in (InputKind.PARAMETER, InputKind.BUFFER)
                and _signature(tensor) != self.attribute_signatures[target]
            ):
                raise ValueError(f"Direct MLX attribute metadata changed: {target}")
            cached = self.views.get(index)
            if tensor.device.type == "mps":
                if cached is None or not cached.matches(tensor):
                    cached = MlxTensorView(tensor, synchronize=False)
                    self.views[index] = cached
                arrays.append(cached.array)
            else:
                arrays.append(mx.array(tensor))
        return tuple(arrays)

    def launch(self, arrays, *, tails=None):
        if self.closed:
            raise RuntimeError("Direct MLX graph is closed")
        outputs = self.compiled[tails is not None](*arrays, *(tails or ()))
        mx.async_eval(*outputs)
        self.execution_count += 1
        return outputs

    def close(self):
        mx.synchronize()
        self.compiled.clear()
        self.views.clear()
        self.closed = True
