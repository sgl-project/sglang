"""Automatic pipeline-parallel layer partition for memory-heterogeneous models.

The default PP split counts layers, but per-layer memory cost is not uniform:
full-attention layers carry KV pool, linear-attention layers carry mamba state,
and with speculative decoding the last stage additionally hosts the draft model
and its KV pool. For hybrid models (e.g. Qwen3.5) an even split can leave the
last stage far heavier than the rest and starve the global KV budget, which is
why SGLANG_PP_LAYER_PARTITION had to be tuned by hand.

This module computes a balanced partition with a small DP over contiguous
layer ranges, minimizing the heaviest stage's estimated bytes. The estimate is
a deterministic pure function of (model_config, server_args, pp_size), so every
rank derives the identical partition without any collective — the same pattern
the mamba pool's max-stage billing already relies on.

Engagement is deliberately narrow: only hybrid (mixed linear/full attention)
models with pp_size > 1 and no explicit SGLANG_PP_LAYER_PARTITION. Uniform
models keep the historical even split, where it is already optimal.
"""

import logging
import math
import os
from typing import List, Optional

import torch

from sglang.srt.configs.hybrid_arch import mambaish_config

logger = logging.getLogger(__name__)

# Byte size of each --kv-cache-dtype choice, mirroring
# mem_cache/kv_cache_dtype.py's resolution. Bytes are all the estimator needs;
# "auto" resolves to the model dtype at the call site.
_KV_DTYPE_STR_TO_BYTES = {
    "bf16": 2,
    "bfloat16": 2,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "mxfp8": 1,
    "nvfp4": 0.5,
    "fp4_mx_block16": 0.5,
}

# Process-level cache consulted by get_pp_indices. A worker process builds one
# target model, so a single slot suffices; the draft worker runs at pp_size=1
# and never reads it.
_AUTO_PP_PARTITION: Optional[List[int]] = None


def get_auto_pp_partition() -> Optional[List[int]]:
    return _AUTO_PP_PARTITION


def _set_auto_pp_partition(partition: Optional[List[int]]) -> None:
    global _AUTO_PP_PARTITION
    _AUTO_PP_PARTITION = partition


def compute_balanced_partition(
    *,
    num_layers: int,
    pp_size: int,
    full_attention_layer_ids: List[int],
    weight_bytes_per_layer: float,
    kv_bytes_per_token_per_full_layer: float,
    mamba_bytes_per_slot_per_linear_layer: float,
    first_stage_extra_bytes: float,
    last_stage_extra_bytes: float,
    draft_kv_bytes_per_token: float,
    reference_num_tokens: float,
    reference_num_slots: float,
) -> List[int]:
    """Contiguous-layer DP minimizing the heaviest stage's estimated bytes.

    Variable terms use reference capacities; the optimum is robust to their
    exact values since every stage's cost is monotone in them. Returns a list
    of per-stage layer counts summing to ``num_layers``.
    """
    assert num_layers >= pp_size > 1

    is_full = [False] * num_layers
    for lid in full_attention_layer_ids:
        if 0 <= lid < num_layers:
            is_full[lid] = True

    # Prefix sums over per-layer fixed (weight) and per-layer variable costs.
    weight_prefix = [0.0] * (num_layers + 1)
    var_prefix = [0.0] * (num_layers + 1)
    for i in range(num_layers):
        weight_prefix[i + 1] = weight_prefix[i] + weight_bytes_per_layer
        if is_full[i]:
            var = reference_num_tokens * kv_bytes_per_token_per_full_layer
        else:
            var = reference_num_slots * mamba_bytes_per_slot_per_linear_layer
        var_prefix[i + 1] = var_prefix[i] + var

    def stage_cost(stage: int, start: int, end: int) -> float:
        cost = (weight_prefix[end] - weight_prefix[start]) + (
            var_prefix[end] - var_prefix[start]
        )
        if stage == 0:
            cost += first_stage_extra_bytes
        if stage == pp_size - 1:
            cost += last_stage_extra_bytes
            cost += reference_num_tokens * draft_kv_bytes_per_token
        return cost

    # f[j] = best (minimized) max-stage cost covering layers [0, j) with the
    # stages processed so far; parent[j] = split point that achieved it.
    INF = math.inf
    f = [INF] * (num_layers + 1)
    f[0] = 0.0
    parents: List[List[int]] = []
    for stage in range(pp_size):
        remaining_stages = pp_size - stage - 1
        g = [INF] * (num_layers + 1)
        parent = [-1] * (num_layers + 1)
        # j must leave at least one layer per remaining stage.
        for j in range(stage + 1, num_layers - remaining_stages + 1):
            for i in range(stage, j):
                if f[i] == INF:
                    continue
                cand = max(f[i], stage_cost(stage, i, j))
                if cand < g[j]:
                    g[j] = cand
                    parent[j] = i
        f = g
        parents.append(parent)

    # Walk back from the full-cover optimum.
    partition = [0] * pp_size
    j = num_layers
    for stage in range(pp_size - 1, -1, -1):
        i = parents[stage][j]
        assert i >= 0, "DP failed to cover all layers"
        partition[stage] = j - i
        j = i
    return partition


def maybe_set_auto_pp_partition(
    model_config, ps, total_gpu_bytes: Optional[int] = None
) -> None:
    """Compute and cache the automatic partition, if engaged.

    Engaged only when all of: pp_size > 1, SGLANG_PP_LAYER_PARTITION unset,
    the model is a hybrid linear/full-attention model. Every input is either a
    shared config or identical hardware, so all ranks derive the same
    partition with no collective — and anything that could disagree (a draft
    config that fails to load) fails loudly instead of diverging.

    ``total_gpu_bytes`` is injectable for tests; production callers leave it
    None to query the local device.
    """
    if ps.pp_size <= 1:
        return
    if os.getenv("SGLANG_PP_LAYER_PARTITION", None) is not None:
        return

    from sglang.srt.runtime_context import (
        get_model,
        get_schedule,
        get_server_args,
        get_spec,
    )

    # SWA hybrids publish full-attention ids on ModelConfig; the GDN families
    # (Qwen3.5, Qwen3-Next, ...) publish them on the mambaish config instead.
    mamba_cfg = mambaish_config(model_config)
    full_ids = list(
        getattr(model_config, "full_attention_layer_ids", None)
        or getattr(mamba_cfg, "full_attention_layer_ids", None)
        or []
    )
    num_layers = model_config.num_hidden_layers
    if not full_ids or len(full_ids) == num_layers:
        # Not a hybrid model; the even split is already optimal.
        return

    if total_gpu_bytes is None and not torch.cuda.is_available():
        # Auto partition estimation needs a GPU budget; keep the even split
        # on non-CUDA platforms (checked before the fallible spec block).
        return

    text_config = model_config.hf_text_config
    dtype_bytes = torch.tensor([], dtype=model_config.dtype).element_size()
    tp_size = max(ps.attn_tp_size, 1)

    # Per-layer weight estimate. MoE/MLP dominates and is identical across
    # layer types in the hybrid families, so a uniform per-layer figure is
    # enough for balancing; fall back to 0 (balance KV/mamba only) when the
    # config does not expose the dims.
    hidden = getattr(text_config, "hidden_size", 0)
    num_experts = getattr(text_config, "num_experts", 0) or 0
    moe_inter = getattr(text_config, "moe_intermediate_size", 0) or 0
    dense_inter = getattr(text_config, "intermediate_size", 0) or 0
    if hidden and num_experts and moe_inter:
        mlp_params = 3 * num_experts * hidden * moe_inter
    elif hidden and dense_inter:
        mlp_params = 3 * hidden * dense_inter
    else:
        mlp_params = 0
    weight_bytes_per_layer = (
        (mlp_params + 4 * hidden * hidden) * dtype_bytes / tp_size if hidden else 0.0
    )

    # KV bytes per token for one full-attention layer, per GPU. "auto" resolves
    # to the model dtype here; an FP8 kv-quant algo is only known after the
    # model loads and would merely halve this estimate.
    num_kv_heads = model_config.get_num_kv_heads(tp_size)
    head_dim = getattr(model_config, "head_dim", None) or (
        model_config.hidden_size // model_config.num_attention_heads
    )
    v_head_dim = getattr(text_config, "v_head_dim", None) or head_dim
    kv_cache_dtype_str = get_model().kv_cache_dtype
    kv_dtype_bytes = (
        dtype_bytes
        if kv_cache_dtype_str == "auto"
        else _KV_DTYPE_STR_TO_BYTES[kv_cache_dtype_str]
    )
    kv_per_token = num_kv_heads * (head_dim + v_head_dim) * kv_dtype_bytes

    # Mamba state bytes per request slot for one linear layer.
    mamba_per_slot = 0.0
    if mamba_cfg is not None and mamba_cfg.mamba2_cache_params.layers:
        mamba_per_slot = mamba_cfg.mamba2_cache_params.mamba_cache_per_req / len(
            mamba_cfg.mamba2_cache_params.layers
        )

    # Endpoint extras: embedding on the first stage, lm_head on the last
    # (mirrored when embeddings are tied).
    vocab = getattr(text_config, "vocab_size", 0)
    embed_bytes = vocab * hidden * dtype_bytes / tp_size if vocab and hidden else 0.0
    tied = bool(getattr(model_config.hf_config, "tie_word_embeddings", False))
    first_extra = embed_bytes
    last_extra = 0.0 if tied else embed_bytes

    # Speculative decoding: the last stage alone hosts the draft — its weights,
    # its own embedding copy (the target's sits on the first stage), and a
    # full-size draft KV pool.
    draft_kv_per_token = 0.0
    if get_spec().speculative_algorithm:
        if get_spec().speculative_draft_model_path:
            # Hard-fail rather than guess: a rank that cannot read the draft
            # config while its peers can would silently pick a different
            # partition.
            from sglang.srt.configs.model_config import ModelConfig

            draft_cfg = ModelConfig.from_server_args(
                get_server_args(),
                model_path=get_spec().speculative_draft_model_path,
                model_revision=get_spec().speculative_draft_model_revision,
                is_draft_model=True,
            )
            draft_layers = draft_cfg.num_hidden_layers
            # The draft's own KV geometry, not the target's.
            draft_head_dim = getattr(draft_cfg, "head_dim", None) or (
                draft_cfg.hidden_size // draft_cfg.num_attention_heads
            )
            draft_v_head_dim = (
                getattr(draft_cfg.hf_text_config, "v_head_dim", None) or draft_head_dim
            )
            draft_geometry = draft_cfg.get_num_kv_heads(tp_size) * (
                draft_head_dim + draft_v_head_dim
            )
        else:
            draft_layers = (
                getattr(text_config, "mtp_num_hidden_layers", None)
                or getattr(text_config, "num_nextn_predict_layers", None)
                or 1
            )
            # Embedded MTP layers share the target's attention geometry.
            draft_geometry = num_kv_heads * (head_dim + v_head_dim)
        # The draft pool follows --speculative-draft-kv-cache-dtype when set,
        # else the target's --kv-cache-dtype (mirrors configure_kv_cache_dtype).
        draft_kv_dtype = get_spec().speculative_draft_kv_cache_dtype
        draft_pool_bytes = (
            _KV_DTYPE_STR_TO_BYTES[draft_kv_dtype]
            if draft_kv_dtype and draft_kv_dtype != "auto"
            else kv_dtype_bytes
        )
        draft_kv_per_token_one_layer = draft_geometry * draft_pool_bytes
        last_extra += draft_layers * weight_bytes_per_layer + embed_bytes
        draft_kv_per_token = draft_layers * draft_kv_per_token_one_layer

    # Reference capacities: what each variable term would hold if the whole
    # per-GPU budget went to it. Only orders of magnitude matter. (The
    # non-CUDA case already returned above.)
    if total_gpu_bytes is None:
        total_gpu_bytes = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory
    budget = total_gpu_bytes * get_schedule().mem_fraction_static
    mean_kv_rate = kv_per_token * len(full_ids) / ps.pp_size
    num_linear = num_layers - len(full_ids)
    mean_mamba_rate = mamba_per_slot * num_linear / ps.pp_size
    ref_tokens = budget / mean_kv_rate if mean_kv_rate > 0 else 0.0
    ref_slots = budget / mean_mamba_rate if mean_mamba_rate > 0 else 0.0

    partition = compute_balanced_partition(
        num_layers=num_layers,
        pp_size=ps.pp_size,
        full_attention_layer_ids=full_ids,
        weight_bytes_per_layer=weight_bytes_per_layer,
        kv_bytes_per_token_per_full_layer=kv_per_token,
        mamba_bytes_per_slot_per_linear_layer=mamba_per_slot,
        first_stage_extra_bytes=first_extra,
        last_stage_extra_bytes=last_extra,
        draft_kv_bytes_per_token=draft_kv_per_token,
        reference_num_tokens=ref_tokens,
        reference_num_slots=ref_slots,
    )
    _set_auto_pp_partition(partition)
    logger.info(
        f"Auto PP layer partition: {partition} "
        f"(set SGLANG_PP_LAYER_PARTITION to override)"
    )
