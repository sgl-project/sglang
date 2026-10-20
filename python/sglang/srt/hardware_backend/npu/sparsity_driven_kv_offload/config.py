"""Configuration and validation for sparsity-driven KV offload."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.configs.model_config import (
    get_dsa_index_head_dim,
    get_dsa_index_topk,
    is_deepseek_dsa,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.budget import (
    SparseKVBudget,
    resolve_sparse_kv_budget,
)
from sglang.srt.runtime_context import (
    attention_backends,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


def is_sparsity_driven_kv_offload_enabled(
    *,
    model_config: ModelConfig,
    use_mla_backend: bool,
) -> bool:
    indexer_impl = get_sparse_kv_indexer_impl()
    if not envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.get():
        if indexer_impl != "legacy":
            raise ValueError(
                "LIM/reference indexer requires SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1."
            )
        return False

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
    if indexer_impl != "legacy":
        if int(get_dsa_index_topk(model_config.hf_config)) != 2048:
            raise ValueError(
                "LIM/reference integration requires model index_topk=2048."
            )
        if get_schedule().page_size != 128:
            raise ValueError("LIM/reference integration requires --page-size 128.")
        if get_parallel().pp_size != 1 or get_parallel().dcp_enabled:
            raise ValueError(
                "LIM/reference integration currently requires PP=1 and DCP disabled."
            )
        if (get_spec().speculative_algorithm or "NONE").upper() != "NONE":
            raise ValueError(
                "LIM/reference framework integration currently supports Q1 decode without speculation."
            )
    return True


def get_sparse_kv_indexer_impl() -> str:
    value = envs.SGLANG_NPU_SPARSE_KV_INDEXER.get()
    if value not in ("legacy", "li_reference", "lim"):
        raise ValueError(
            "SGLANG_NPU_SPARSE_KV_INDEXER must be legacy, li_reference or lim; "
            f"got {value!r}."
        )
    return value


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
    if not is_sparsity_driven_kv_offload_enabled(
        model_config=model_config,
        use_mla_backend=use_mla_backend,
    ):
        return None

    index_head_dim = get_sparsity_driven_kv_offload_index_head_dim(
        model_config=model_config
    )
    return index_head_dim * num_layers * element_size


def get_sparse_kv_budget(
    *,
    request_rows: int,
    context_len: int,
    num_layers: int,
    top_k: int,
    kv_bytes_per_token: int,
) -> SparseKVBudget:
    return resolve_sparse_kv_budget(
        request_rows=request_rows,
        context_len=context_len,
        num_layers=num_layers,
        top_k=top_k,
        kv_bytes_per_token=kv_bytes_per_token,
        cache_tokens=envs.SGLANG_NPU_SPARSE_KV_CACHE_SIZE.get(),
        hbm_budget_bytes=envs.SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB.get() * 1024**2,
        indexer_impl=get_sparse_kv_indexer_impl(),
    )
