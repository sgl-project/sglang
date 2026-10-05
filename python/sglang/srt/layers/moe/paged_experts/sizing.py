"""How many experts per MoE layer stay resident on the GPU (K) when the user does not set it.

The K-slot table counts as model weights in sglang's memory accounting, so K is sized from the
same budget the KV pool comes from: what ``--mem-fraction-static`` leaves after the non-expert
weights, minus a KV reserve for the declared concurrency.
"""

from __future__ import annotations

import functools
import glob
import json
import logging
import os
import re
import struct
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Allocations the runtime makes before KV profiling (loader and repack workspaces,
# fragmentation) on top of the exact weight bytes.
_RUNTIME_RESERVE_BYTES = 0.5e9
# Hybrid (mamba / linear-attention) models size their per-request state cache from what is left
# after weights. One in-flight request needs one slot; the mamba radix cache keeps retired states
# for prefix reuse, about 5x that. Capped so a high --max-running-requests cannot turn the whole
# budget into state.
_HYBRID_STATE_SLOTS_PER_REQUEST = 5
_HYBRID_STATE_MAX_SLOTS = 32
_LAYER_INDEX = re.compile(r"layers\.(\d+)\.")


def num_resident_for_budget(
    *,
    free_bytes: float,
    mem_fraction: float,
    nonexpert_bytes: float,
    reserve_bytes: float,
    per_expert_bytes: float,
    top_k: int,
    num_experts: int,
) -> int:
    """Largest K whose experts (``per_expert_bytes`` each, over all MoE layers) fit the static
    budget, clamped to ``[top_k, num_experts]``."""
    budget = free_bytes * mem_fraction - nonexpert_bytes - reserve_bytes
    return max(top_k, min(num_experts, int(budget // per_expert_bytes)))


@functools.lru_cache(maxsize=None)
def auto_num_resident(num_experts: int, top_k: int) -> int:
    """K for the current model, from the checkpoint and the published config. Cached: every
    MoE layer of a model gets the same K, sized against free memory when the first one is
    built."""
    from sglang.srt.runtime_context import (
        get_model,
        get_schedule,
        process_model_config,
    )

    expert_bytes, nonexpert_bytes = _checkpoint_bytes(
        model_path=get_model().model_path,
        revision=get_model().revision,
        num_layers=process_model_config().num_hidden_layers,
    )
    num_resident = num_resident_for_budget(
        # Free memory before model loading, as sglang's KV sizing measures it: the layers
        # built so far are part of nonexpert_bytes.
        free_bytes=torch.cuda.mem_get_info()[0] + torch.cuda.memory_allocated(),
        mem_fraction=get_schedule().mem_fraction_static,
        nonexpert_bytes=nonexpert_bytes,
        reserve_bytes=_RUNTIME_RESERVE_BYTES + _state_reserve_bytes(),
        per_expert_bytes=expert_bytes / num_experts,
        top_k=top_k,
        num_experts=num_experts,
    )
    logger.info(
        "Paged experts: %d of %d experts per layer resident (auto)",
        num_resident,
        num_experts,
    )
    return num_resident


def _state_reserve_bytes() -> float:
    """KV cache for the declared concurrency (one request when unset) at full context, capped at
    --max-total-tokens like the real pool; for hybrid models, the per-request state cache
    instead when that binds (both are carved from the same leftover)."""
    from sglang.srt.configs.hybrid_arch import mambaish_config
    from sglang.srt.configs.model_config import AttentionArch
    from sglang.srt.runtime_context import (
        get_model,
        get_parallel,
        get_schedule,
        process_model_config,
    )

    mc = process_model_config()
    schedule = get_schedule()
    hybrid = mambaish_config(mc)
    # On hybrid models only the full-attention layers hold token KV.
    kv_layers = len(hybrid.full_attention_layer_ids) if hybrid else mc.num_hidden_layers
    kv_elt = 1 if "fp8" in str(get_model().kv_cache_dtype) else 2
    if mc.attention_arch == AttentionArch.MLA:
        cell = (mc.kv_lora_rank + mc.qk_rope_head_dim) * kv_elt
    else:
        kv_heads = mc.get_num_kv_heads(get_parallel().tp_size)
        cell = kv_heads * (mc.head_dim + mc.v_head_dim) * kv_elt
    requests = max(1, schedule.max_running_requests or 1)
    tokens = requests * mc.context_len
    if schedule.max_total_tokens:
        tokens = min(tokens, schedule.max_total_tokens)
    reserve = tokens * cell * kv_layers
    if hybrid is not None:
        slots = schedule.max_mamba_cache_size or min(
            _HYBRID_STATE_MAX_SLOTS, requests * _HYBRID_STATE_SLOTS_PER_REQUEST
        )
        # The state pool carries one padding slot.
        state = hybrid.mamba2_cache_params.mamba_cache_per_req * (slots + 1)
        reserve = max(reserve, state)
    return reserve


def _checkpoint_bytes(
    model_path: str, revision: Optional[str], num_layers: int
) -> Tuple[int, int]:
    """Exact (routed expert, everything else) bytes from the safetensors headers; shared
    experts count as non-expert. Draft (MTP / nextn) layers, numbered from ``num_layers`` on,
    are not loaded for serving and count as neither."""
    folder = model_path
    if not os.path.isdir(folder):
        from huggingface_hub import snapshot_download

        folder = snapshot_download(
            model_path,
            revision=revision,
            local_files_only=True,
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
        )
    files = glob.glob(os.path.join(folder, "*.safetensors"))
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            shards = set(json.load(f)["weight_map"].values())
        # A partially downloaded checkpoint would undercount silently.
        if not shards <= {os.path.basename(path) for path in files}:
            files = []
    if not files:
        raise RuntimeError(
            f"Paged experts: no complete safetensors checkpoint under {folder} to size K "
            "from; set --paged-experts-num-resident"
        )
    expert = nonexpert = 0
    for path in files:
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            layer = _LAYER_INDEX.search(name)
            if (
                "mtp" in name
                or "nextn" in name
                or (layer is not None and int(layer.group(1)) >= num_layers)
            ):
                continue
            begin, end = entry["data_offsets"]
            if ".experts." in name:
                expert += end - begin
            else:
                nonexpert += end - begin
    return expert, nonexpert
