# SPDX-License-Identifier: Apache-2.0
"""Build compatibility keys for weight-cache requests and snapshot lookups."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.runtime_context import get_exec, get_parallel

from .protocol import (
    CacheConfig,
    compute_env_stamp,
    get_quant_method_name,
    hash_quant_config,
)
from .schema import LoadSpec
from .weight_layout import flashinfer_cutedsl_swap_ab, normalize_init_expert_location

if TYPE_CHECKING:
    import torch
    from transformers import PretrainedConfig

    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig


def resolve_cache_quantization(
    model_config: ModelConfig, requested_method: Optional[str] = None
) -> tuple[str, Any]:
    """Resolve the method and config used by IPC checks and snapshot identity."""
    config = getattr(model_config.hf_config, "quantization_config", None)
    method = get_quant_method_name(requested_method or model_config.quantization)
    return method or get_quant_method_name(config), config


def build_cache_config(
    model_config: ModelConfig,
    *,
    quant_method: str,
    quant_config: Any,
    model_path: str,
    revision: Optional[str],
    ep_size: int,
) -> CacheConfig:
    """Build the fingerprint before loading can mutate the quantization config."""
    parallel = get_parallel()
    architectures = model_config.hf_config.architectures
    return CacheConfig(
        model_path=model_path,
        model_arch=architectures[0] if architectures else "",
        tp_size=parallel.tp_size,
        tp_rank=parallel.tp_rank,
        pp_size=parallel.pp_size,
        pp_rank=parallel.pp_rank,
        dp_size=parallel.dp_size,
        ep_size=ep_size,
        # MoE-DP can alias a wider attention-CP communicator. Compare its
        # configured width, as recorded in the daemon protocol.
        moe_dp_size=parallel.moe_dp_size,
        moe_dp_rank=parallel.moe_dp_rank,
        moe_ep_rank=parallel.moe_ep_rank,
        attn_dp_size=parallel.attn_dp_size,
        enable_dp_lm_head=parallel.enable_dp_lm_head,
        attn_cp_size=parallel.attn_cp_size,
        moe_dense_tp_size=parallel.moe_dense_tp_size,
        moe_a2a_backend=get_exec().moe.moe_a2a_backend,
        quant_method=quant_method,
        quant_config_hash=hash_quant_config(quant_config),
        dtype=str(model_config.dtype),
        revision=revision or "",
        **compute_env_stamp(),
    )


def _hf_config_hash(hf_config: PretrainedConfig) -> str:
    # Composite HF models can have their own locator on each nested config.
    def omit_local_paths(fields):
        fields.pop("_name_or_path", None)
        fields.pop("name_or_path", None)
        return fields

    fields = json.loads(
        hf_config.to_json_string(use_diff=False), object_hook=omit_local_paths
    )
    encoded = json.dumps(fields, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_load_spec(
    *,
    model_config: ModelConfig,
    checkpoint_id: str,
    load_config: LoadConfig,
    target_device: torch.device,
) -> LoadSpec:
    """Capture this rank's model and resolved runtime settings before loading.

    The caller identifies the checkpoint contents (e.g. a pinned Hub commit or
    a local checkpoint digest). Two checkpoints can have identical model configs.
    """
    from sglang.srt.layers.dp_attention import get_moe_cp_size
    from sglang.srt.layers.moe.utils import get_moe_runner_backend
    from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
    from sglang.srt.layers.quantization.fp8_utils import get_fp8_gemm_runner_backend
    from sglang.srt.layers.quantization.unquant import get_bf16_gemm_backend

    quant_method, quant_config = resolve_cache_quantization(model_config)
    cache_config = build_cache_config(
        model_config,
        quant_method=quant_method,
        quant_config=quant_config,
        model_path=model_config.model_path,
        revision=model_config.revision,
        ep_size=get_parallel().moe_ep_size,
    )
    execution = get_exec()
    return LoadSpec(
        **cache_config.to_dict(),
        checkpoint_id=checkpoint_id,
        hf_config_hash=_hf_config_hash(model_config.hf_config),
        draft_model_idx=load_config.draft_model_idx,
        moe_runner_backend=get_moe_runner_backend().value,
        fp4_gemm_runner_backend=get_fp4_gemm_runner_backend().value,
        fp8_gemm_runner_backend=get_fp8_gemm_runner_backend().value,
        bf16_gemm_backend=get_bf16_gemm_backend().value,
        moe_cp_size=get_moe_cp_size(),
        enable_fp32_lm_head=execution.features.enable_fp32_lm_head,
        ep_num_redundant_experts=execution.moe.ep_num_redundant_experts,
        enable_eplb=execution.moe.enable_eplb,
        init_expert_location=normalize_init_expert_location(
            execution.moe.init_expert_location
        ),
        flashinfer_cutedsl_swap_ab=flashinfer_cutedsl_swap_ab(target_device),
    )
