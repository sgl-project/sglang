"""Torch-owned exported inference graphs compiled directly by MLX."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import mlx.core as mx
import torch
from torch.export.graph_signature import InputKind, OutputKind

from sglang.srt.hardware_backend.mlx.fx_lowering import (
    MlxFxLoweringRegistry,
    UnsupportedMlxFxGraphError,
    build_mlx_fx_plan,
)
from sglang.srt.hardware_backend.mps.compiled_region import UnsupportedMlxRegion
from sglang.srt.utils.tensor_bridge import MlxTensorView

logger = logging.getLogger(__name__)


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


def _weighted_rms_pattern(node, norms):
    if str(node.target) != "aten.mul.Tensor":
        return None
    for normalized, weight in (node.args, node.args[::-1]):
        if normalized not in norms or not isinstance(weight, torch.fx.Node):
            continue
        x, eps = norms[normalized]
        if tuple(weight.meta["val"].shape) == (x.meta["val"].shape[-1],):
            return x, weight, eps
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
        registry = MlxFxLoweringRegistry.standard_export_decoder()
        for target, lowering in (custom_lowerings or {}).items():
            registry.register_function(
                target,
                lambda mx, args, kwargs, fn=lowering: fn(*args, **kwargs),
            )
        if attention is not None:
            from sglang.kernels.ops.attention.mlx.radix_attention_export import (
                radix_decode,
            )

            registry.register_function(
                radix_decode._opoverload,
                lambda mx, args, kwargs: attention(*args, **kwargs),
            )
        self.plan = build_mlx_fx_plan(exported.graph_module, registry)
        try:
            self.plan.require_fully_supported()
        except UnsupportedMlxFxGraphError as error:
            raise UnsupportedMlxRegion(str(error)) from error
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
        self._refresh_binding_owners()
        self.norms = {
            node: pattern for node in self.nodes if (pattern := _rms_pattern(node))
        }
        self.weighted_norms = {
            node: pattern
            for node in self.nodes
            if (pattern := _weighted_rms_pattern(node, self.norms))
        }
        logger.info(
            "Direct MLX export: %d nodes, %d fused RMSNorms (%d weighted)",
            len(self.nodes),
            len(self.norms),
            len(self.weighted_norms),
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
            for node, node_plan in zip(self.nodes, self.plan.nodes):
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
                if node in self.weighted_norms:
                    x, weight, eps = self.weighted_norms[node]
                    values[node] = mx.fast.rms_norm(values[x], values[weight], eps)
                elif node in self.norms:
                    x, eps = self.norms[node]
                    values[node] = mx.fast.rms_norm(values[x], None, eps)
                elif str(node.target) == "sglang.mlx_radix_decode.default":
                    if self.attention is None:
                        raise ValueError("Radix attention lowering was not provided")
                    tails = (flat[-2][layer], flat[-1][layer]) if tail else None
                    values[node] = self.attention(*args, **kw, tails=tails)
                    layer += 1
                else:
                    values[node] = node_plan.implementation(mx, args, kw)
            raise ValueError("Direct MLX graph has no output")

        return forward

    def _refresh_binding_owners(self):
        modules = dict(self.model.named_modules(remove_duplicate=False))
        self._module_links = tuple(
            (modules[path.rpartition(".")[0]], path.rpartition(".")[2], module)
            for path, module in modules.items()
            if path
        )
        self._binding_owners = {
            target: (modules[target.rpartition(".")[0]], target.rpartition(".")[2])
            for kind, target in self.bindings
            if kind in (InputKind.PARAMETER, InputKind.BUFFER)
        }

    def bind(self, inputs):
        if self.closed:
            raise RuntimeError("Direct MLX graph is closed")
        if tuple(_signature(x) for x in inputs) != self.input_signatures:
            raise ValueError("Direct MLX input shape, dtype, stride or device changed")
        if any(
            parent._modules[name] is not child
            for parent, name, child in self._module_links
        ):
            self._refresh_binding_owners()
        arrays = []
        for index, (kind, target) in enumerate(self.bindings):
            if kind == InputKind.USER_INPUT:
                tensor = inputs[target]
            elif kind == InputKind.PARAMETER:
                owner, name = self._binding_owners[target]
                tensor = owner._parameters[name]
            elif kind == InputKind.BUFFER:
                owner, name = self._binding_owners[target]
                tensor = owner._buffers[name]
            else:
                tensor = target
            if (
                kind in (InputKind.PARAMETER, InputKind.BUFFER)
                and _signature(tensor) != self.attribute_signatures[target]
            ):
                raise ValueError(f"Direct MLX attribute metadata changed: {target}")
            cached = self.views.get(index)
            if tensor.device.type == "mps":
                # Shape, dtype, stride and device were checked above.
                if (
                    cached is None
                    or tensor.data_ptr() != cached.torch_tensor.data_ptr()
                ):
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
