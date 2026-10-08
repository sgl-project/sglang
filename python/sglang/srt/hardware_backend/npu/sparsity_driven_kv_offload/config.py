"""Configuration and validation for sparsity-driven KV offload."""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.configs.model_config import (
    get_dsa_index_head_dim,
    get_dsa_index_topk,
    is_deepseek_dsa,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    attention_backends,
    get_disagg,
    get_schedule,
    process_model_config,
    uses_mla_backend,
)
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


SPARSE_KV_DEVICE_CACHE_UNIT_SIZE = 2048
SPARSE_KV_DEVICE_CACHE_CAPACITIES = tuple(
    factor * SPARSE_KV_DEVICE_CACHE_UNIT_SIZE for factor in range(1, 5)
)


def get_sparsity_driven_kv_offload_device_cache_capacity(
    *, sparse_context_len: int, enable_lru: bool
) -> int:
    """Resolve the device capacity and validate the selected cache policy."""
    if sparse_context_len <= 0:
        raise ValueError("Sparse KV offload requires a positive DSA index_topk.")
    if not enable_lru:
        return sparse_context_len
    if sparse_context_len != SPARSE_KV_DEVICE_CACHE_UNIT_SIZE:
        raise ValueError(
            "SGLANG_NPU_SPARSE_KV_ENABLE_LRU requires DSA index_topk=2048, "
            f"got {sparse_context_len}. Disable LRU to use dynamic top-k."
        )
    env_field = envs.SGLANG_NPU_SPARSE_KV_DEVICE_CACHE_FACTOR
    env_name = env_field.name
    raw_factor = os.getenv(env_name, str(env_field.default))
    try:
        factor = env_field.parse(raw_factor)
    except ValueError as exc:
        raise ValueError(
            f"{env_name} must be an integer in [1, 4], got {raw_factor!r}."
        ) from exc
    if factor < 1 or factor > 4:
        raise ValueError(f"{env_name} must be an integer in [1, 4], got {factor}.")
    return factor * SPARSE_KV_DEVICE_CACHE_UNIT_SIZE


class SparseKVOffloadMode(str, Enum):
    DISABLED = "disabled"
    LOCAL_OFFLOAD = "local_offload"
    PD_PREFILL_NATIVE = "pd_prefill_native"
    PD_DECODE_OFFLOAD = "pd_decode_offload"

    @property
    def uses_host_kv_offload(self) -> bool:
        return self in (
            SparseKVOffloadMode.LOCAL_OFFLOAD,
            SparseKVOffloadMode.PD_DECODE_OFFLOAD,
        )

    @property
    def uses_pd_decode_staging(self) -> bool:
        return self is SparseKVOffloadMode.PD_DECODE_OFFLOAD


def resolve_sparse_kv_offload_mode(
    *,
    model_config: Optional[ModelConfig] = None,
    use_mla_backend: Optional[bool] = None,
) -> SparseKVOffloadMode:
    if not envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.get():
        return SparseKVOffloadMode.DISABLED

    # The NPU MLA pool has no ModelConfig argument; use the published process
    # configuration there. Callers holding a model config pass it explicitly.
    if model_config is None:
        model_config = process_model_config()
    if use_mla_backend is None:
        use_mla_backend = uses_mla_backend()

    prefill_attention_backend, decode_attention_backend = attention_backends()
    if not (
        is_npu()
        and prefill_attention_backend == "ascend"
        and decode_attention_backend == "ascend"
        and use_mla_backend
        and is_deepseek_dsa(model_config.hf_config)
    ):
        raise ValueError(
            "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD requires an NPU "
            "DSA-family MLA model "
            "(for example DeepSeek V3.2 or GLM-5.x) using the Ascend MLA "
            "attention backend."
        )
    if get_schedule().max_running_requests is None:
        raise ValueError(
            "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD requires max_running_requests "
            "to be set to bound the per-process host KV allocation."
        )

    disagg = get_disagg()
    if disagg.disaggregation_mode == "null":
        return SparseKVOffloadMode.LOCAL_OFFLOAD
    if disagg.disaggregation_mode in ("prefill", "decode"):
        if disagg.disaggregation_transfer_backend != "ascend":
            raise ValueError(
                "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD with PD disaggregation "
                "requires disaggregation_transfer_backend='ascend'; got "
                f"{disagg.disaggregation_transfer_backend!r}."
            )
        if disagg.disaggregation_mode == "prefill":
            return SparseKVOffloadMode.PD_PREFILL_NATIVE
        return SparseKVOffloadMode.PD_DECODE_OFFLOAD
    raise ValueError(
        "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD received unsupported "
        f"disaggregation_mode={disagg.disaggregation_mode!r}."
    )


def get_sparsity_driven_kv_offload_sparse_context_len(
    *,
    model_config: ModelConfig,
) -> int:
    """Return the per-request on-device sparse KV window size."""
    sparse_context_len = int(get_dsa_index_topk(model_config.hf_config))
    if sparse_context_len <= 0:
        raise ValueError(
            "Sparsity-driven KV offload requires a positive DSA index_topk, "
            f"got {sparse_context_len}."
        )
    return sparse_context_len


def get_sparsity_driven_kv_offload_index_head_dim(
    *,
    model_config: ModelConfig,
) -> int:
    index_head_dim = getattr(model_config, "index_head_dim", None)
    if index_head_dim is None:
        index_head_dim = get_dsa_index_head_dim(model_config.hf_config)
    index_head_dim = int(index_head_dim)
    if index_head_dim <= 0:
        raise ValueError(
            "Sparsity-driven KV offload requires a positive DSA index_head_dim, "
            f"got {index_head_dim}."
        )
    return index_head_dim


def get_sparsity_driven_kv_offload_cell_size(
    *,
    model_config: ModelConfig,
    use_mla_backend: bool,
    num_layers: int,
    element_size: int,
) -> Optional[int]:
    mode = resolve_sparse_kv_offload_mode(
        model_config=model_config,
        use_mla_backend=use_mla_backend,
    )
    if not mode.uses_host_kv_offload:
        return None

    index_head_dim = get_sparsity_driven_kv_offload_index_head_dim(
        model_config=model_config
    )
    return index_head_dim * num_layers * element_size


def get_sparsity_driven_kv_offload_fixed_memory_size(
    *,
    model_config: ModelConfig,
    use_mla_backend: bool,
    num_layers: int,
    element_size: int,
    max_running_requests_per_worker: int,
) -> Optional[int]:
    """Return the fixed device-KV allocation made by the sparse KV manager.

    In addition to the token-scaled index pool, ``SparseKVCacheManager`` keeps
    a full-MLA-KV cache for every request and layer. With LRU its capacity is
    ``SGLANG_NPU_SPARSE_KV_DEVICE_CACHE_FACTOR * 2048``; otherwise it matches
    the model's ``index_topk``. The request-to-token
    pool includes a padding row and, on PD decode, preallocated transfer rows.
    The manager allocates a device cache for all of them.
    """
    mode = resolve_sparse_kv_offload_mode(
        model_config=model_config,
        use_mla_backend=use_mla_backend,
    )
    if not mode.uses_host_kv_offload:
        return None

    max_running_requests_per_worker = int(max_running_requests_per_worker)
    if max_running_requests_per_worker <= 0:
        raise ValueError(
            "Sparsity-driven KV offload requires a positive per-worker "
            "max_running_requests, got "
            f"{max_running_requests_per_worker}."
        )

    sparse_context_len = get_sparsity_driven_kv_offload_sparse_context_len(
        model_config=model_config
    )
    device_cache_capacity = get_sparsity_driven_kv_offload_device_cache_capacity(
        sparse_context_len=sparse_context_len,
        enable_lru=envs.SGLANG_NPU_SPARSE_KV_ENABLE_LRU.get(),
    )
    kv_head_dim = int(model_config.kv_lora_rank) + int(model_config.qk_rope_head_dim)
    request_capacity = max_running_requests_per_worker + 1
    if mode.uses_pd_decode_staging:
        request_capacity += get_disagg().disaggregation_decode_extra_slots
    return (
        request_capacity
        * device_cache_capacity
        * kv_head_dim
        * num_layers
        * element_size
    )
