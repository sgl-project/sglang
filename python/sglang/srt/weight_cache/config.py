# SPDX-License-Identifier: Apache-2.0
"""Build snapshot lookup keys from the resolved model and runtime configuration."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from sglang.srt.layers import dp_attention
from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import use_cutedsl_moe_swap_ab
from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
from sglang.srt.layers.quantization.fp8_utils import get_fp8_gemm_runner_backend
from sglang.srt.layers.quantization.unquant import get_bf16_gemm_backend
from sglang.srt.model_loader.loader import PreshardedModelLoader
from sglang.srt.runtime_context import get_exec, get_parallel

from .protocol import build_cache_config, resolve_cache_quantization
from .schema import LoadSpec

if TYPE_CHECKING:
    import torch
    from transformers import PretrainedConfig

    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig


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
    moe_runner = get_moe_runner_backend()
    a2a_backend = get_moe_a2a_backend()
    swap_ab = None
    if moe_runner.is_flashinfer_cutedsl() and (
        a2a_backend.is_none() or a2a_backend.is_flashinfer()
    ):
        swap_ab = use_cutedsl_moe_swap_ab(target_device)

    return LoadSpec(
        **cache_config.to_dict(),
        checkpoint_id=checkpoint_id,
        hf_config_hash=_hf_config_hash(model_config.hf_config),
        draft_model_idx=load_config.draft_model_idx,
        moe_runner_backend=moe_runner.value,
        fp4_gemm_runner_backend=get_fp4_gemm_runner_backend().value,
        fp8_gemm_runner_backend=get_fp8_gemm_runner_backend().value,
        bf16_gemm_backend=get_bf16_gemm_backend().value,
        moe_cp_size=dp_attention.get_moe_cp_size(),
        enable_fp32_lm_head=execution.features.enable_fp32_lm_head,
        ep_num_redundant_experts=execution.moe.ep_num_redundant_experts,
        enable_eplb=execution.moe.enable_eplb,
        init_expert_location=PreshardedModelLoader._normalize_init_expert_location(
            execution.moe.init_expert_location
        ),
        flashinfer_cutedsl_swap_ab=swap_ab,
    )
