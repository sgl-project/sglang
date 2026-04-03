from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

LORA_A_SUFFIXES = (".lora_A", ".lora_A.weight")
LORA_B_SUFFIXES = (".lora_B", ".lora_B.weight")
FUSED_EXPERT_RUNTIME_BACKEND_GENERIC = "generic"
FUSED_EXPERT_RUNTIME_BACKEND_FLASHINFER_TRTLLM = "flashinfer_trtllm"


def apply_lora_merge_from_tensors(
    model,
    named_tensors: List[tuple[str, torch.Tensor]],
    loader_metadata: Optional[Dict[str, Any]] = None,
):
    """Merge LoRA factors into the live model weights in place."""
    loader_metadata = loader_metadata or {}
    tensor_dict = dict(named_tensors)
    inferred_targets = _infer_targets_from_tensor_names(tensor_dict)
    target_specs = _normalize_target_specs(loader_metadata, inferred_targets)

    params_dict = dict(model.named_parameters(remove_duplicate=False))
    for target_spec in target_specs:
        target_name = target_spec["target_name"]
        if target_name not in params_dict:
            # This is expected on PP ranks that do not own the target parameter.
            continue

        target_param = params_dict[target_name]
        _apply_target_update(
            target_param,
            target_spec,
            tensor_dict=tensor_dict,
            loader_metadata=loader_metadata,
        )


def _apply_target_update(
    target_param: torch.nn.Parameter,
    target_spec: Dict[str, Any],
    *,
    tensor_dict: Dict[str, torch.Tensor],
    loader_metadata: Dict[str, Any],
) -> None:
    fused_expert_runtime_backend = _resolve_fused_expert_runtime_backend(
        target_param,
        target_spec,
    )
    if fused_expert_runtime_backend == FUSED_EXPERT_RUNTIME_BACKEND_FLASHINFER_TRTLLM:
        _apply_flashinfer_fused_expert_update(
            target_param,
            target_spec,
            tensor_dict=tensor_dict,
            loader_metadata=loader_metadata,
        )
        return
    if fused_expert_runtime_backend not in (
        None,
        FUSED_EXPERT_RUNTIME_BACKEND_GENERIC,
    ):
        raise NotImplementedError(
            f"Unsupported fused expert runtime backend {fused_expert_runtime_backend!r}."
        )

    local_delta = _compile_local_lora_delta(
        target_param,
        target_spec,
        tensor_dict=tensor_dict,
        loader_metadata=loader_metadata,
    )
    target_param.data.add_(local_delta)


def _compile_local_lora_delta(
    target_param: torch.nn.Parameter,
    target_spec: Dict[str, Any],
    *,
    tensor_dict: Dict[str, torch.Tensor],
    loader_metadata: Dict[str, Any],
) -> torch.Tensor:
    """Compile one or more LoRA factor pairs into a local runtime delta tensor."""
    base_weight = target_param.data
    total_delta = torch.zeros_like(base_weight)
    for component_spec in _iter_target_components(target_spec):
        lora_a_name = component_spec["lora_a_name"]
        lora_b_name = component_spec["lora_b_name"]
        if lora_a_name not in tensor_dict:
            raise KeyError(
                f"Missing LoRA A tensor {lora_a_name!r} for {target_spec['target_name']!r}."
            )
        if lora_b_name not in tensor_dict:
            raise KeyError(
                f"Missing LoRA B tensor {lora_b_name!r} for {target_spec['target_name']!r}."
            )

        scaling = _resolve_scaling(component_spec, target_spec, loader_metadata)
        component_delta = _compute_component_delta(
            tensor_dict[lora_a_name],
            tensor_dict[lora_b_name],
            target_device=base_weight.device,
            target_dtype=base_weight.dtype,
            scaling=scaling,
        )
        total_delta.add_(
            _place_component_delta(
                target_param,
                base_weight,
                component_delta,
                component_spec,
            )
        )

    return total_delta


def _apply_flashinfer_fused_expert_update(
    target_param: torch.nn.Parameter,
    target_spec: Dict[str, Any],
    *,
    tensor_dict: Dict[str, torch.Tensor],
    loader_metadata: Dict[str, Any],
) -> None:
    layer = _get_weight_loader_owner(target_param)
    if layer is None:
        raise ValueError(
            f"Unable to resolve fused expert layer for {target_spec.get('target_name')!r}."
        )

    target_attr_name = _get_fused_expert_param_attr_name(layer, target_param)
    if target_attr_name is None:
        raise ValueError(
            f"Unable to resolve fused expert parameter for {target_spec.get('target_name')!r}."
        )

    canonical_shape = _get_fused_expert_canonical_shape(layer, target_attr_name)
    packed_expert_shape = tuple(target_param.data.shape[1:])
    temp_param = _make_scratch_param_like(
        target_param,
        shape=(1, *canonical_shape[1:]),
    )
    for component_spec in _iter_target_components(target_spec):
        lora_a_name = component_spec["lora_a_name"]
        lora_b_name = component_spec["lora_b_name"]
        if lora_a_name not in tensor_dict:
            raise KeyError(
                f"Missing LoRA A tensor {lora_a_name!r} for {target_spec['target_name']!r}."
            )
        if lora_b_name not in tensor_dict:
            raise KeyError(
                f"Missing LoRA B tensor {lora_b_name!r} for {target_spec['target_name']!r}."
            )

        shard_id = _normalize_shard_id(component_spec.get("shard_id"))
        if shard_id is None:
            raise ValueError(
                f"fused_experts target {target_spec['target_name']!r} requires shard_id."
            )

        scaling = _resolve_scaling(component_spec, target_spec, loader_metadata)
        lora_a = tensor_dict[lora_a_name].to(
            device=target_param.data.device,
            dtype=target_param.data.dtype,
        )
        lora_b = tensor_dict[lora_b_name].to(
            device=target_param.data.device,
            dtype=target_param.data.dtype,
        )
        num_component_experts = _get_component_num_experts(lora_a, lora_b)
        for expert_id in range(num_component_experts):
            local_expert_id = _map_global_expert_id_to_local_expert_id_for_hot_update(
                layer,
                target_param,
                expert_id,
            )
            if local_expert_id is None:
                continue

            temp_param.data.zero_()
            expert_delta = _compute_expert_component_delta(
                lora_a,
                lora_b,
                expert_id=expert_id,
                scaling=scaling,
            )
            _load_single_fused_expert_delta(
                temp_param,
                expert_delta,
                component_spec,
                shard_id=shard_id,
            )
            packed_delta = _pack_flashinfer_fused_expert_single_expert(
                layer,
                target_attr_name,
                temp_param.data[0],
            )
            if tuple(packed_delta.shape) != packed_expert_shape:
                raise ValueError(
                    "FlashInfer fused expert packed delta shape mismatch: "
                    f"expected {packed_expert_shape}, got {tuple(packed_delta.shape)}."
                )
            target_param.data[local_expert_id].add_(packed_delta)


def _compute_component_delta(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    target_device: torch.device,
    target_dtype: torch.dtype,
    scaling: float,
) -> torch.Tensor:
    """Compile a single LoRA factor pair into a delta tensor."""
    if lora_a.ndim not in (2, 3) or lora_b.ndim not in (2, 3):
        raise ValueError(
            "LoRA merge currently expects rank-2 or rank-3 A/B tensors, "
            f"got A.ndim={lora_a.ndim}, B.ndim={lora_b.ndim}."
        )
    if lora_a.shape[-2] != lora_b.shape[-1]:
        raise ValueError(
            "LoRA rank mismatch: "
            f"A shape {tuple(lora_a.shape)} is incompatible with B shape {tuple(lora_b.shape)}."
        )

    delta = torch.matmul(
        lora_b.to(device=target_device, dtype=target_dtype),
        lora_a.to(device=target_device, dtype=target_dtype),
    )
    return delta * scaling


def _place_component_delta(
    target_param: torch.nn.Parameter,
    base_weight: torch.Tensor,
    delta: torch.Tensor,
    component_spec: Dict[str, Any],
) -> torch.Tensor:
    shard_id = _normalize_shard_id(component_spec.get("shard_id"))
    weight_loader = getattr(target_param, "weight_loader", None)
    if component_spec.get("fused_experts"):
        if weight_loader is None:
            raise ValueError(
                f"Target parameter {component_spec.get('target_name')!r} does not expose "
                "a weight_loader required for shard placement."
            )
        return _materialize_fused_expert_delta(
            target_param,
            delta,
            component_spec,
            shard_id=shard_id,
        )

    if weight_loader is not None:
        placed_param = _make_scratch_param_like(target_param)
        if shard_id is None:
            delta = _reshape_delta_for_target(delta, base_weight)
            weight_loader(placed_param, delta)
        else:
            weight_loader(placed_param, delta, shard_id)
        if placed_param.data.shape != base_weight.shape:
            raise ValueError(
                "Localized LoRA delta shape mismatch: "
                f"expected {tuple(base_weight.shape)}, got {tuple(placed_param.data.shape)}."
            )
        return placed_param.data

    if shard_id is not None:
        raise ValueError(
            f"Target parameter {component_spec.get('target_name')!r} does not expose "
            "a weight_loader required for shard placement."
        )
    delta = _reshape_delta_for_target(delta, base_weight)
    if delta.shape != base_weight.shape:
        raise ValueError(
            "Merged LoRA delta shape mismatch: "
            f"expected {tuple(base_weight.shape)}, got {tuple(delta.shape)}."
        )
    return delta


def _reshape_delta_for_target(
    delta: torch.Tensor, base_weight: torch.Tensor
) -> torch.Tensor:
    if delta.shape == base_weight.shape:
        return delta
    if (
        base_weight.ndim == 3
        and base_weight.shape[1] == 1
        and delta.ndim == 2
        and delta.shape[0] == base_weight.shape[0]
        and delta.shape[1] == base_weight.shape[2]
    ):
        return delta.unsqueeze(1)
    return delta


def _make_scratch_param_like(
    target_param: torch.nn.Parameter,
    *,
    shape: Optional[tuple[int, ...]] = None,
) -> torch.nn.Parameter:
    temp_param = torch.nn.Parameter(
        torch.zeros(
            shape if shape is not None else tuple(target_param.data.shape),
            device=target_param.data.device,
            dtype=target_param.data.dtype,
        ),
        requires_grad=False,
    )
    for attr_name, attr_value in vars(target_param).items():
        if attr_name.startswith("_"):
            continue
        setattr(temp_param, attr_name, attr_value)
    return temp_param


def _load_fused_expert_delta(
    placed_param: torch.nn.Parameter,
    delta: torch.Tensor,
    component_spec: Dict[str, Any],
    *,
    shard_id: Any,
) -> None:
    if delta.ndim != 3:
        raise ValueError(
            "fused_experts placement expects a rank-3 delta with an expert dimension, "
            f"got shape {tuple(delta.shape)}."
        )

    target_name = component_spec.get("target_name")
    if not target_name:
        raise ValueError("fused_experts placement requires component_spec.target_name.")

    weight_loader = placed_param.weight_loader
    for expert_id, expert_delta in enumerate(delta):
        weight_loader(
            placed_param,
            expert_delta,
            target_name,
            shard_id,
            expert_id,
        )


def _materialize_fused_expert_delta(
    target_param: torch.nn.Parameter,
    delta: torch.Tensor,
    component_spec: Dict[str, Any],
    *,
    shard_id: Any,
) -> torch.Tensor:
    placed_param = _make_scratch_param_like(target_param)
    _load_fused_expert_delta(
        placed_param,
        delta,
        component_spec,
        shard_id=shard_id,
    )
    return placed_param.data


def _get_weight_loader_owner(target_param: torch.nn.Parameter):
    weight_loader = getattr(target_param, "weight_loader", None)
    return getattr(weight_loader, "__self__", None)


def _get_fused_expert_param_attr_name(layer, target_param: torch.nn.Parameter) -> Optional[str]:
    for attr_name in ("w13_weight", "w2_weight"):
        if getattr(layer, attr_name, None) is target_param:
            return attr_name
    return None


def _get_fused_expert_canonical_shape(layer, param_attr_name: str) -> tuple[int, int, int]:
    if param_attr_name == "w13_weight":
        rows = (
            2 * layer.intermediate_size_per_partition
            if layer.moe_runner_config.is_gated
            else layer.intermediate_size_per_partition
        )
        return (layer.num_local_experts, rows, layer.hidden_size)
    if param_attr_name == "w2_weight":
        return (
            layer.num_local_experts,
            layer.hidden_size,
            layer.intermediate_size_per_partition,
        )
    raise ValueError(f"Unsupported fused expert parameter {param_attr_name!r}.")


def _resolve_fused_expert_runtime_backend(
    target_param: torch.nn.Parameter,
    target_spec: Dict[str, Any],
) -> Optional[str]:
    if not _has_fused_expert_components(target_spec):
        return None
    layer = _get_weight_loader_owner(target_param)
    if layer is not None and getattr(layer, "use_flashinfer_trtllm_moe", False):
        return FUSED_EXPERT_RUNTIME_BACKEND_FLASHINFER_TRTLLM
    return FUSED_EXPERT_RUNTIME_BACKEND_GENERIC


def _has_fused_expert_components(target_spec: Dict[str, Any]) -> bool:
    return any(
        component.get("fused_experts") for component in _iter_target_components(target_spec)
    )


def _pack_flashinfer_fused_expert_single_expert(
    layer,
    target_attr_name: str,
    expert_tensor: torch.Tensor,
) -> torch.Tensor:
    if expert_tensor.dtype != torch.bfloat16:
        raise NotImplementedError(
            "FlashInfer fused expert hot updates currently expect BF16 weights."
        )

    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    epilogue_tile_m = 128
    block_k = 128
    cache = getattr(layer.quant_method, "_cache_permute_indices", {})

    if target_attr_name == "w13_weight":
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            expert_tensor.view(torch.uint8),
            epilogue_tile_m,
        )
    elif target_attr_name == "w2_weight":
        permute_indices = get_w2_permute_indices_with_cache(
            cache,
            expert_tensor.view(torch.uint8),
            epilogue_tile_m,
        )
    else:
        raise ValueError(f"Unsupported FlashInfer fused expert target {target_attr_name!r}.")

    packed_weight = (
        expert_tensor.clone()
        .view(torch.uint8)[permute_indices.to(expert_tensor.device)]
        .contiguous()
    )
    packed_weight = convert_to_block_layout(
        packed_weight.view(torch.uint8),
        block_k,
    )
    return packed_weight.view(torch.bfloat16).contiguous()


def _get_component_num_experts(lora_a: torch.Tensor, lora_b: torch.Tensor) -> int:
    a_experts = lora_a.shape[0] if lora_a.ndim == 3 else 1
    b_experts = lora_b.shape[0] if lora_b.ndim == 3 else 1
    return max(a_experts, b_experts)


def _compute_expert_component_delta(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    expert_id: int,
    scaling: float,
) -> torch.Tensor:
    expert_a = _select_expert_factor(lora_a, expert_id)
    expert_b = _select_expert_factor(lora_b, expert_id)
    if expert_a.ndim != 2 or expert_b.ndim != 2:
        raise ValueError(
            "FlashInfer fused expert hot updates expect per-expert rank-2 factors, "
            f"got A shape {tuple(expert_a.shape)}, B shape {tuple(expert_b.shape)}."
        )
    if expert_a.shape[0] != expert_b.shape[1]:
        raise ValueError(
            "LoRA rank mismatch: "
            f"A shape {tuple(expert_a.shape)} is incompatible with B shape {tuple(expert_b.shape)}."
        )
    return torch.matmul(expert_b, expert_a) * scaling


def _select_expert_factor(factor: torch.Tensor, expert_id: int) -> torch.Tensor:
    if factor.ndim == 2:
        return factor
    if factor.ndim != 3:
        raise ValueError(
            "FlashInfer fused expert hot updates expect rank-2 or rank-3 factors, "
            f"got shape {tuple(factor.shape)}."
        )
    if factor.shape[0] == 1:
        return factor[0]
    if expert_id >= factor.shape[0]:
        raise IndexError(
            f"Expert index {expert_id} is out of range for factor shape {tuple(factor.shape)}."
        )
    return factor[expert_id]


def _load_single_fused_expert_delta(
    placed_param: torch.nn.Parameter,
    expert_delta: torch.Tensor,
    component_spec: Dict[str, Any],
    *,
    shard_id: Any,
) -> None:
    target_name = component_spec.get("target_name")
    if not target_name:
        raise ValueError("fused_experts placement requires component_spec.target_name.")
    placed_param.weight_loader(
        placed_param,
        expert_delta,
        target_name,
        shard_id,
        0,
    )


def _map_global_expert_id_to_local_expert_id_for_hot_update(
    layer,
    target_param: torch.nn.Parameter,
    expert_id: int,
) -> Optional[int]:
    if getattr(target_param, "_sglang_require_global_experts", False):
        return expert_id
    if not hasattr(layer, "_map_global_expert_id_to_local_expert_id"):
        return expert_id
    local_expert_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
    if local_expert_id == -1:
        return None
    return local_expert_id


def _normalize_target_specs(
    loader_metadata: Dict[str, Any],
    inferred_targets: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    target_specs = loader_metadata.get("targets")
    if target_specs is None:
        if not inferred_targets:
            raise ValueError(
                "loader_metadata.targets is required when target names cannot be inferred "
                "from tensor names."
            )
        target_specs = [
            {"target_name": target_name, **tensor_names}
            for target_name, tensor_names in inferred_targets.items()
        ]
    elif isinstance(target_specs, dict):
        target_specs = [
            {"target_name": target_name, **(target_spec or {})}
            for target_name, target_spec in target_specs.items()
        ]
    elif not isinstance(target_specs, list):
        raise TypeError("loader_metadata.targets must be a list or dict.")

    normalized_specs: List[Dict[str, Any]] = []
    for target_spec in target_specs:
        normalized_spec = dict(target_spec)
        target_name = normalized_spec["target_name"]
        inferred_tensor_names = inferred_targets.get(target_name, {})
        if "components" in normalized_spec:
            normalized_components = []
            for component_spec in normalized_spec["components"]:
                normalized_component = dict(component_spec)
                normalized_component.setdefault(
                    "target_name",
                    target_name,
                )
                normalized_components.append(normalized_component)
            normalized_spec["components"] = normalized_components
        else:
            normalized_spec.setdefault(
                "lora_a_name",
                inferred_tensor_names.get("lora_a_name", f"{target_name}.lora_A"),
            )
            normalized_spec.setdefault(
                "lora_b_name",
                inferred_tensor_names.get("lora_b_name", f"{target_name}.lora_B"),
            )
        normalized_specs.append(normalized_spec)

    return normalized_specs


def _resolve_scaling(
    component_spec: Dict[str, Any],
    target_spec: Dict[str, Any],
    loader_metadata: Dict[str, Any],
) -> float:
    scaling = component_spec.get("scaling")
    if scaling is None:
        scaling = target_spec.get("scaling")
    if scaling is not None:
        return float(scaling)

    rank = (
        component_spec.get("rank")
        or component_spec.get("r")
        or target_spec.get("rank")
        or target_spec.get("r")
        or loader_metadata.get("rank")
        or loader_metadata.get("r")
    )
    alpha = (
        component_spec.get("lora_alpha")
        or target_spec.get("lora_alpha")
        or loader_metadata.get("lora_alpha")
    )
    if alpha is not None and rank is not None:
        return float(alpha) / float(rank)
    return 1.0


def _iter_target_components(target_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    components = target_spec.get("components")
    if components is None:
        return [target_spec]
    if not isinstance(components, list) or not components:
        raise TypeError("target_spec.components must be a non-empty list.")
    return [dict(component_spec) for component_spec in components]


def _normalize_shard_id(shard_id: Any) -> Any:
    if isinstance(shard_id, list):
        return tuple(shard_id)
    return shard_id


def _infer_targets_from_tensor_names(
    tensor_dict: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, str]]:
    inferred_targets: Dict[str, Dict[str, str]] = {}
    for tensor_name in tensor_dict:
        target_name, factor_kind = _parse_factor_tensor_name(tensor_name)
        if target_name is None or factor_kind is None:
            continue
        inferred_targets.setdefault(target_name, {})
        inferred_targets[target_name][factor_kind] = tensor_name
    return {
        target_name: tensor_names
        for target_name, tensor_names in inferred_targets.items()
        if "lora_a_name" in tensor_names and "lora_b_name" in tensor_names
    }


def _parse_factor_tensor_name(tensor_name: str) -> tuple[Optional[str], Optional[str]]:
    for suffix in LORA_A_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)], "lora_a_name"
    for suffix in LORA_B_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)], "lora_b_name"
    return None, None
