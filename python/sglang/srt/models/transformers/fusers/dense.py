# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import ast
import logging
import re
import types
from dataclasses import dataclass

from torch import nn
from transformers.activations import ACT2FN

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.runtime_context import get_parallel

from ..layers import (
    HFCompatibleMergedColumnParallelLinear,
    HFCompatibleQKVParallelLinear,
)
from .source import (
    ReplaceNodes,
    compile_forward,
    projection_calls,
    read_forward,
    self_attribute,
)

logger = logging.getLogger(__name__)
_SILU_TYPES = {nn.SiLU, type(ACT2FN["silu"])}


@dataclass(frozen=True)
class FusionResult:
    stacked_mapping: dict[str, tuple[str, str | int]]
    packed_modules_mapping: dict[str, list[str]]


def _colwise(names, prefix, tp_plan):
    if get_parallel().tp_size == 1:
        return True
    if tp_plan is None:
        return False
    return all(
        next(
            (
                style
                for pattern, style in tp_plan.items()
                if re.fullmatch(pattern, f"{prefix}.{name}")
            ),
            None,
        )
        == "colwise"
        for name in names
    )


def _compatible(projections):
    return (
        all(type(projection) is nn.Linear for projection in projections)
        and len({projection.in_features for projection in projections}) == 1
        and len({projection.bias is None for projection in projections}) == 1
        and len({projection.weight.dtype for projection in projections}) == 1
    )


def _create_fused(factory, names, merged_name, quant_config, packed_modules_mapping):
    previous = (packed_modules_mapping or {}).get(merged_name)
    if previous is not None and previous != list(names):
        return None
    if quant_config is None:
        return factory()
    original = dict(quant_config.packed_modules_mapping)
    previous = original.get(merged_name)
    if previous is not None and previous != list(names):
        return None
    quant_config.update_packed_modules_mapping({**original, merged_name: list(names)})
    try:
        return factory()
    except (ValueError, NotImplementedError, AssertionError) as error:
        quant_config.update_packed_modules_mapping(original)
        logger.debug("Cannot fuse %s: %s", merged_name, error)
        return None


def _install(module, prefix, names, merged_name, merged, forward, shard_ids):
    setattr(module, merged_name, merged)
    for name in names:
        delattr(module, name)
    module.forward = types.MethodType(forward, module)
    return FusionResult(
        {
            f"{prefix}.{name}": (f"{prefix}.{merged_name}", shard)
            for name, shard in zip(names, shard_ids)
        },
        {merged_name: list(names)},
    )


def _safe_projection_statement(statement, calls, names, module):
    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return False
    targets = (
        statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    )
    if any(not isinstance(target, ast.Name) for target in targets):
        return False
    input_name = next(iter(calls.values())).args[0].id
    if any(target.id == input_name for target in targets):
        return False
    allowed_methods = {"view", "reshape", "transpose", "contiguous"}
    for node in ast.walk(statement.value):
        if not isinstance(node, ast.Call):
            continue
        if node in calls.values():
            continue
        name = self_attribute(node.func)
        if name and name not in names:
            norm = getattr(module, name, None)
            if norm is not None and type(norm).__name__.endswith("RMSNorm"):
                continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in allowed_methods:
            continue
        return False
    return True


def _fuse_qkv(
    module, prefix, quant_config, tp_plan, function, original, packed_modules_mapping
):
    names = next(
        (
            candidate
            for candidate in (("q_proj", "k_proj", "v_proj"), ("query", "key", "value"))
            if all(
                isinstance(getattr(module, name, None), nn.Linear) for name in candidate
            )
        ),
        None,
    )
    if (
        names is None
        or hasattr(module, "qkv_proj")
        or not _colwise(names, prefix, tp_plan)
    ):
        return None
    projections = [getattr(module, name) for name in names]
    if not _compatible(projections):
        return None
    q, k, v = projections
    head_dim = getattr(module, "head_dim", getattr(module, "attention_head_size", None))
    if (
        not isinstance(head_dim, int)
        or head_dim <= 0
        or k.out_features != v.out_features
    ):
        return None
    if q.out_features % head_dim or k.out_features % head_dim:
        return None
    num_heads, kv_heads = q.out_features // head_dim, k.out_features // head_dim
    tp_size = get_parallel().attn_tp_size
    if num_heads % tp_size or (
        kv_heads % tp_size if kv_heads >= tp_size else tp_size % kv_heads
    ):
        return None
    calls = projection_calls(function, names)
    if calls is None:
        return None
    statements = [
        index
        for index, statement in enumerate(function.body)
        if any(call in ast.walk(statement) for call in calls.values())
    ]
    if len(statements) != 3 or statements != list(
        range(statements[0], statements[0] + 3)
    ):
        return None
    if not all(
        _safe_projection_statement(function.body[index], calls, names, module)
        for index in statements
    ):
        return None
    temporary_names = [f"_sglang_{shard}_projection" for shard in ("q", "k", "v")]
    if set(temporary_names) & {
        node.id for node in ast.walk(function) if isinstance(node, ast.Name)
    }:
        return None
    source = f"{', '.join(temporary_names)} = self.qkv_proj({calls[names[0]].args[0].id}).split(self._sglang_qkv_sizes, dim=-1)"
    ReplaceNodes(
        {
            id(calls[name]): ast.Name(id=temporary, ctx=ast.Load())
            for name, temporary in zip(names, temporary_names)
        }
    ).visit(function)
    function.body.insert(statements[0], ast.parse(source).body[0])
    forward = compile_forward(function, original)
    merged = _create_fused(
        lambda: HFCompatibleQKVParallelLinear(
            hidden_size=q.in_features,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=kv_heads,
            bias=q.bias is not None,
            params_dtype=q.weight.dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            tp_rank=get_parallel().attn_tp_rank,
            tp_size=tp_size,
        ),
        names,
        "qkv_proj",
        quant_config,
        packed_modules_mapping,
    )
    if merged is None:
        return None
    module._sglang_qkv_sizes = (
        merged.q_proj_shard_size,
        merged.kv_proj_shard_size,
        merged.v_proj_shard_size,
    )
    return _install(module, prefix, names, "qkv_proj", merged, forward, ("q", "k", "v"))


def _fuse_glu(
    module, prefix, quant_config, tp_plan, function, original, packed_modules_mapping
):
    if hasattr(module, "gate_up_proj"):
        return None
    for node in ast.walk(function):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            continue
        for activation, up in ((node.left, node.right), (node.right, node.left)):
            if (
                not isinstance(activation, ast.Call)
                or len(activation.args) != 1
                or activation.keywords
            ):
                continue
            activation_name = self_attribute(activation.func)
            if type(getattr(module, activation_name or "", None)) not in _SILU_TYPES:
                continue
            gate = activation.args[0]
            if not all(isinstance(call, ast.Call) for call in (gate, up)):
                continue
            names = tuple(self_attribute(call.func) for call in (gate, up))
            if None in names or names[0] == names[1]:
                continue
            projections = [getattr(module, name, None) for name in names]
            if not _compatible(projections) or not _colwise(names, prefix, tp_plan):
                continue
            gate_projection, up_projection = projections
            if (
                gate_projection.out_features != up_projection.out_features
                or gate_projection.out_features % (8 * get_parallel().tp_size)
            ):
                continue
            calls = projection_calls(function, names)
            if calls is None or {id(call) for call in calls.values()} != {
                id(gate),
                id(up),
            }:
                continue
            if (
                sum(
                    self_attribute(item) == activation_name
                    for item in ast.walk(function)
                )
                != 1
            ):
                continue
            if getattr(getattr(module, activation_name), "inplace", False):
                continue
            replacement = ast.parse(
                f"self.{activation_name}(self.gate_up_proj({gate.args[0].id}))",
                mode="eval",
            ).body
            ReplaceNodes({id(node): replacement}).visit(function)
            forward = compile_forward(function, original)
            merged = _create_fused(
                lambda: HFCompatibleMergedColumnParallelLinear(
                    input_size=gate_projection.in_features,
                    output_sizes=[
                        gate_projection.out_features,
                        up_projection.out_features,
                    ],
                    bias=gate_projection.bias is not None,
                    params_dtype=gate_projection.weight.dtype,
                    quant_config=quant_config,
                    prefix=f"{prefix}.gate_up_proj",
                ),
                names,
                "gate_up_proj",
                quant_config,
                packed_modules_mapping,
            )
            if merged is None:
                return None
            setattr(module, activation_name, SiluAndMul())
            return _install(
                module, prefix, names, "gate_up_proj", merged, forward, (0, 1)
            )
    return None


def fuse_module(
    module,
    prefix,
    quant_config=None,
    tp_plan=None,
    *,
    packed_modules_mapping=None,
    disabled_fusions=(),
):
    if not any(type(child) is nn.Linear for child in module.children()):
        return None
    for name, fuser in (("qkv", _fuse_qkv), ("mlp", _fuse_glu)):
        if name in disabled_fusions:
            continue
        try:
            function, original = read_forward(module)
            result = fuser(
                module,
                prefix,
                quant_config,
                tp_plan,
                function,
                original,
                packed_modules_mapping,
            )
        except (OSError, TypeError, SyntaxError, ValueError) as error:
            logger.debug("Skipping fusion of %s: %s", prefix, error)
            continue
        if result is not None:
            logger.debug("Fused projections at %s: %s", prefix, result.stacked_mapping)
            return result
    return None


def load_fused_weights(
    module,
    weights,
    mapping,
    loaded_names,
    *,
    ignore_unexpected_suffixes=(),
    require_complete=True,
):
    parameters = dict(module.named_parameters())
    expected_shards = {}
    for target, shard in mapping.values():
        expected_shards.setdefault(target, set()).add(shard)
    loaded_shards = {}
    fully_loaded = set()
    for name, weight in weights:
        source, separator, suffix = name.rpartition(".")
        if separator and source in mapping:
            target, shard = mapping[source]
            target_name = f"{target}.{suffix}"
            parameter = parameters.get(target_name)
            if parameter is None:
                if any(name.endswith(suffix) for suffix in ignore_unexpected_suffixes):
                    continue
                raise ValueError(
                    f"Fused checkpoint parameter {name!r} maps to missing {target_name!r}"
                )
            loader = getattr(parameter, "weight_loader", None)
            if loader is None:
                raise ValueError(f"Fused parameter {target_name!r} has no shard loader")
            loader(parameter, weight, shard)
            partitioned = (
                getattr(parameter, "output_dim", None) is not None
                or getattr(parameter, "needs_scalar_to_array", False)
                or getattr(parameter, "is_metadata", False)
                or getattr(parameter, "is_gguf_weight", False)
                or isinstance(parameter, PerTensorScaleParameter)
            )
            if partitioned:
                seen = loaded_shards.setdefault(target_name, set())
                seen.add(shard)
                if not require_complete or seen == expected_shards[target]:
                    loaded_names.add(target_name)
            else:
                loaded_names.add(target_name)
        else:
            if source in expected_shards and name in parameters:
                fully_loaded.add(name)
            yield name, weight
    if not require_complete:
        return
    for name, seen in loaded_shards.items():
        if name in fully_loaded:
            continue
        target = name.rpartition(".")[0]
        missing = expected_shards[target] - seen
        if missing:
            raise ValueError(
                f"Incomplete fused checkpoint parameter {name!r}: "
                f"missing shards {sorted(missing, key=str)!r}"
            )
