from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.model_config import get_nsa_index_head_dim, is_deepseek_nsa
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.utils.common import is_float4_e2m1fn_x2

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def get_cell_size_per_token(mr: ModelRunner, num_layers: int) -> int:
    # args to config cell size
    model_config = mr.model_config
    kv_cache_dtype = mr.kv_cache_dtype
    use_mla_backend = mr.use_mla_backend

    kv_size = torch._utils._element_size(kv_cache_dtype)
    if use_mla_backend:
        cell_size = (
            (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
            * num_layers
            * kv_size
        )
        if is_float4_e2m1fn_x2(kv_cache_dtype):
            # kv_scale_buffer
            scale_block_size = 16
            cell_size = (cell_size // 2) + (
                (
                    (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
                    // scale_block_size
                )
                * num_layers
                * kv_size
            )

        # Add indexer KV cache overhead for NSA models (DeepSeek V3.2)
        if is_deepseek_nsa(model_config.hf_config):
            index_head_dim = get_nsa_index_head_dim(model_config.hf_config)
            indexer_size_per_token = (
                index_head_dim + index_head_dim // NSATokenToKVPool.quant_block_size * 4
            )
            element_size = torch._utils._element_size(
                NSATokenToKVPool.index_k_with_scale_buffer_dtype
            )
            cell_size += indexer_size_per_token * num_layers * element_size
    else:
        if model_config.is_hybrid_swa:
            full_layers_num = len(model_config.full_attention_layer_ids)
            swa_layers_num = len(model_config.swa_attention_layer_ids)

            full_per_token = model_config.get_num_kv_heads(get_attention_tp_size()) * (
                model_config.head_dim + model_config.v_head_dim
            )

            swa_per_token = model_config.get_swa_num_kv_heads(
                get_attention_tp_size()
            ) * (model_config.swa_head_dim + model_config.swa_v_head_dim)

            cell_size = (
                full_per_token * full_layers_num + swa_per_token * swa_layers_num
            ) * kv_size
        else:
            cell_size = (
                model_config.get_num_kv_heads(get_attention_tp_size())
                * (model_config.head_dim + model_config.v_head_dim)
                * num_layers
                * kv_size
            )

        if is_float4_e2m1fn_x2(kv_cache_dtype):
            # kv_scale_buffer
            scale_block_size = 16

            n = model_config.get_num_kv_heads(get_attention_tp_size())
            k = model_config.head_dim
            cell_size = (cell_size // 2) + (
                (n * k * num_layers * 2 * kv_size) // scale_block_size
            )
    return cell_size


def resolve_hybrid_swa_tokens(
    mr: ModelRunner, token_capacity: int
) -> tuple[int, int, int]:
    """Split token_capacity into full/swa pools.

    Returns (effective_capacity, full_max_total_num_tokens, swa_max_total_num_tokens).
    """
    model_config = mr.model_config
    page_size = mr.server_args.page_size
    swa_full_tokens_ratio = mr.server_args.swa_full_tokens_ratio

    full_layers_num = len(model_config.full_attention_layer_ids)
    swa_layers_num = len(model_config.swa_attention_layer_ids)
    assert swa_layers_num > 0, "Hybrid SWA model must have at least one SWA layer"

    def align_page_size(x: int) -> int:
        return (x // page_size) * page_size

    if full_layers_num == 0:
        # all layers are SWA
        swa_tokens = align_page_size(token_capacity)
        logger.info(
            f"Use sliding window memory pool (all SWA). swa_layer_tokens={swa_tokens}"
        )
        return swa_tokens, 0, swa_tokens

    # Use unified memory-based allocation for all hybrid SWA models.
    #
    # Let:
    #   F = Full layer per-token memory
    #   S = SWA layer per-token memory (may differ from F)
    #   r = swa_full_tokens_ratio = swa_tokens / full_tokens
    #
    # The profile phase computed:
    #   cell_size = F * n_full + S * n_swa
    #   token_capacity = rest_memory / cell_size
    #   => total_memory = token_capacity * (F * n_full + S * n_swa)
    #
    # We need to solve:
    #   full_tokens * F * n_full + swa_tokens * S * n_swa = total_memory
    #   swa_tokens = full_tokens * r
    #
    # Solution:
    #   full_tokens = total_memory / (F * n_full + r * S * n_swa)
    #               = token_capacity * (F * n_full + S * n_swa) / (F * n_full + r * S * n_swa)

    kv_size = torch._utils._element_size(mr.kv_cache_dtype)

    # Full layer per-token memory
    full_per_token = (
        model_config.get_num_kv_heads(get_attention_tp_size())
        * (model_config.head_dim + model_config.v_head_dim)
        * kv_size
    )

    # SWA layer per-token memory
    swa_per_token = (
        model_config.get_swa_num_kv_heads(get_attention_tp_size())
        * (model_config.swa_head_dim + model_config.swa_v_head_dim)
        * kv_size
    )

    # Total memory available from profile
    total_memory = token_capacity * (
        full_per_token * full_layers_num + swa_per_token * swa_layers_num
    )

    # Solve the equations
    denominator = (
        full_per_token * full_layers_num
        + swa_full_tokens_ratio * swa_per_token * swa_layers_num
    )
    assert (
        denominator > 0
    ), f"Invalid denominator={denominator} for memory-based allocation. full_per_token={full_per_token}, full_layers_num={full_layers_num}, swa_per_token={swa_per_token}, swa_layers_num={swa_layers_num}, swa_full_tokens_ratio={swa_full_tokens_ratio}"

    full_tokens = align_page_size(int(total_memory / denominator))
    swa_tokens = align_page_size(int(full_tokens * swa_full_tokens_ratio))

    logger.info(
        f"Use sliding window memory pool. full_layer_tokens={full_tokens}, swa_layer_tokens={swa_tokens}"
    )
    return full_tokens, full_tokens, swa_tokens
