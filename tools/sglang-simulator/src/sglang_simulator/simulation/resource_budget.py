"""Optional InferCast resource budgets for the declared target deployment."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_simulator.simulation.types import SchedulerConfig
    from sglang_simulator.spec.accelerator import AcceleratorInfo
    from sglang_simulator.spec.model import ModelInfo

_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CONFIG_FIELDS = {
    "provider",
    "provider_revision",
    "model_id",
    "model_revision",
    "policy",
    "shared_expert_tp_size",
}
_STRUCTURAL_FIELDS = {
    "architectures",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "attention_head_dim",
    "v_head_dim",
    "intermediate_size",
    "vocab_size",
    "tie_word_embeddings",
    "attention_bias",
    "mlp_bias",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "kv_lora_rank",
    "q_lora_rank",
    "moe_intermediate_size",
    "n_routed_experts",
    "n_shared_experts",
    "first_k_dense_replace",
    "moe_layer_freq",
    "moe_router_dtype",
    "router_fp32",
    "topk_method",
    "quantization_config",
}


def _dtype_name(value) -> str:
    name = str(getattr(value, "value", value)).removeprefix("torch.").lower()
    return {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(name, name)


def _validate_loaded_model(
    model: ModelInfo, resources, config_path: Path, target_config: SchedulerConfig
) -> None:
    payload = config_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != resources.provenance.config_sha256:
        raise ValueError(
            "local config.json changed during InferCast resource estimation"
        )
    source = json.loads(payload)
    effective = model.hf_config
    if callable(getattr(effective, "to_dict", None)):
        effective = effective.to_dict()
    if not isinstance(effective, Mapping):
        raise ValueError("memory_budget requires materialized model hf_config")  # noqa: TRY004
    mismatches = {
        name: {"loaded": effective.get(name), "source": source[name]}
        for name in _STRUCTURAL_FIELDS & source.keys()
        if effective.get(name) != source[name]
    }
    if effective.get("quantization_config") != source.get("quantization_config"):
        mismatches["quantization_config"] = {
            "loaded": effective.get("quantization_config"),
            "source": source.get("quantization_config"),
        }
    for name in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "v_head_dim",
        "vocab_size",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "qk_nope_head_dim",
    ):
        value = getattr(model, name)
        if name in source and value is not None and value != source[name]:
            mismatches[name] = {"loaded": value, "source": source[name]}
    if mismatches:
        raise ValueError(
            f"loaded model resource geometry differs from config.json: {mismatches}"
        )
    expected_arch = {"LLAMA": "MHA", "DEEPSEEK": "MLA"}.get(resources.family)
    if expected_arch is not None and model.attention_arch != expected_arch:
        raise ValueError(
            "loaded attention layout differs from InferCast resource layout"
        )
    embedding = next(
        tensor for tensor in resources.tensors if tensor.role == "embedding"
    )
    if _dtype_name(model.torch_dtype) != embedding.storage.data_dtype:
        raise ValueError(
            "loaded model torch_dtype differs from InferCast resource storage"
        )
    if (
        target_config.data_type is not None
        and _dtype_name(target_config.data_type) != embedding.storage.data_dtype
    ):
        raise ValueError("target data_type differs from InferCast resource storage")


def estimate_infercast_budget(
    model: ModelInfo,
    hw: AcceleratorInfo,
    target_config: SchedulerConfig,
    budget_config: Mapping,
) -> tuple[int, dict]:
    """Return logical token capacity and a compact, attributable budget summary."""
    if not isinstance(budget_config, Mapping):
        raise ValueError("memory_budget must be an object")  # noqa: TRY004
    unknown = set(budget_config) - _CONFIG_FIELDS
    if unknown:
        raise ValueError(f"unknown memory_budget fields: {sorted(unknown)}")
    if budget_config.get("provider") != "infercast":
        raise ValueError("memory_budget.provider must be 'infercast'")
    model_id = budget_config.get("model_id")
    if not isinstance(model_id, str) or not model_id or model_id != model_id.strip():
        raise ValueError("memory_budget.model_id must be a nonempty trimmed string")
    revision = budget_config.get("provider_revision")
    if not isinstance(revision, str) or not _FULL_COMMIT.fullmatch(revision):
        raise ValueError(
            "memory_budget.provider_revision must be a full lowercase commit"
        )
    model_revision = budget_config.get("model_revision")
    if model_revision is not None and (
        not isinstance(model_revision, str)
        or not _FULL_COMMIT.fullmatch(model_revision)
    ):
        raise ValueError(
            "memory_budget.model_revision must be null or a full lowercase commit"
        )
    if not isinstance(budget_config.get("policy"), Mapping):
        raise ValueError("memory_budget.policy must be an explicit object")  # noqa: TRY004
    if type(target_config.dp_size) is not int or target_config.dp_size != 1:
        raise ValueError("InferCast memory budgets require target dp_size=1")
    if type(target_config.cp_size) is not int or target_config.cp_size != 1:
        raise ValueError("InferCast memory budgets require target cp_size=1")
    if (
        not isinstance(model.model_path, (str, Path))
        or not str(model.model_path).strip()
    ):
        raise ValueError(
            "InferCast memory budgets require a local model config directory"
        )
    kv_dtype = getattr(target_config.kv_cache_data_type, "value", None)
    if not isinstance(kv_dtype, str):
        raise ValueError("InferCast memory budgets require explicit kv_cache_data_type")  # noqa: TRY004

    capacity = hw.hbm_capacity_gb
    try:
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, (int, float))
            or not math.isfinite(capacity)
            or capacity <= 0
        ):
            raise ValueError
        # Preserve the existing simulator's GiB convention, independent of host GPU.
        capacity_bytes = int(capacity * (1 << 30))
        if capacity_bytes <= 0:
            raise ValueError
    except (ValueError, OverflowError):
        raise ValueError(
            "target accelerator hbm_capacity_gb must be finite and positive"
        ) from None

    try:
        import infercast

        actual_revision = getattr(infercast, "__revision__", None)
        if actual_revision != revision:
            raise ValueError(
                "memory_budget.provider_revision does not match installed InferCast "
                f"(configured={revision!r}, installed={actual_revision!r})"
            )
        from infercast.sdk import (
            MemoryBudgetPolicy,
            ResourceTopology,
            build_model_resources,
            estimate_memory_budget,
        )
    except ImportError as error:
        raise ValueError(
            "InferCast with the resource-budget SDK is required"
        ) from error

    topology = ResourceTopology(
        tp_size=target_config.tp_size,
        ep_size=target_config.ep_size,
        pp_size=target_config.pp_size,
    )
    policy = MemoryBudgetPolicy.from_dict(dict(budget_config["policy"]))
    page_size = target_config.page_size or 1
    config_path = Path(model.model_path).expanduser() / "config.json"
    resources = build_model_resources(
        model_id,
        path=config_path,
        topology=topology,
        kv_cache_dtype=kv_dtype,
        kv_page_size=page_size,
        shared_expert_tp_size=budget_config.get("shared_expert_tp_size"),
        revision=model_revision,
    )
    _validate_loaded_model(model, resources, config_path, target_config)
    budget = estimate_memory_budget(
        resources, device_capacity_bytes=capacity_bytes, policy=policy
    )
    max_tokens = budget.max_cache_tokens
    if (
        budget.status != "fits_declared_budget"
        or type(max_tokens) is not int
        or max_tokens <= 0
    ):
        raise ValueError(
            "InferCast memory budget leaves no usable KV capacity "
            f"(status={budget.status!r}, max_cache_tokens={max_tokens!r})"
        )
    metrics = {
        "provider": "infercast",
        "provider_revision": revision,
        "model_id": model_id,
        "resource_digest": resources.digest,
        "provenance": resources.provenance.to_dict(),
        "policy": policy.to_dict(),
        "topology": topology.to_dict(),
        "kv_cache_dtype": kv_dtype,
        "kv_page_size": page_size,
        "shared_expert_tp_size": budget_config.get("shared_expert_tp_size"),
        "max_total_tokens": max_tokens,
        "ranks": [
            {key: value for key, value in rank.to_dict().items() if key != "tensors"}
            for rank in budget.ranks
        ],
    }
    return max_tokens, metrics
