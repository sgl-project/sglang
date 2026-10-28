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
    sparse_batch_capacity,
)
from sglang.srt.runtime_context import (
    attention_backends,
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.utils.common import (
    get_cuda_graph_max_batch_size,
    get_eager_max_batch_size,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


def is_copy_sfa_decode_graph_eligible(seq_lens_cpu) -> bool:
    """Require every live row to have enough tokens to populate its cache.

    C defaults to K=2048. Larger native LIM caches use host gather until C
    visible tokens exist, so those batches must also stay outside this graph.
    """
    if not envs.SGLANG_NPU_ENABLE_COPY_SFA.get():
        return True
    capacity = envs.SGLANG_NPU_SPARSE_KV_CACHE_SIZE.get() or 2048
    return seq_lens_cpu is not None and not any(
        0 < int(length) < capacity for length in seq_lens_cpu.tolist()
    )


def is_sparsity_driven_kv_offload_enabled(
    *,
    model_config: ModelConfig,
    use_mla_backend: bool,
) -> bool:
    indexer_impl = get_sparse_kv_indexer_impl()
    if (
        envs.SGLANG_NPU_ENABLE_COPY_SFA.get()
        and not envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.get()
    ):
        raise ValueError(
            "SGLANG_NPU_ENABLE_COPY_SFA requires SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD."
        )
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
    algorithm = (get_spec().speculative_algorithm or "NONE").upper()
    if algorithm != "NONE" and (
        algorithm not in ("EAGLE", "NEXTN")
        or get_spec().speculative_eagle_topk != 1
        or get_parallel().pp_size != 1
        or get_parallel().dcp_enabled
    ):
        raise ValueError(
            "Sparse KV MTP requires EAGLE/NEXTN with "
            "--speculative-eagle-topk 1, PP=1 and DCP disabled."
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
    copy_sfa_query_heads: int = 128,
    graph_batch_capacity: Optional[int] = None,
) -> SparseKVBudget:
    if graph_batch_capacity is None:
        # Reserve before admission determines the final request-pool size.
        # Capture buckets may round that size up for attention parallelism.
        graph = get_exec().graph
        config = graph.cuda_graph_config
        decode = config.decode if config is not None else None
        graph_batch_capacity = (
            min(max(decode.bs), get_cuda_graph_max_batch_size(request_rows))
            if not graph.disable_cuda_graph and decode is not None and decode.bs
            else 0
        )
    return resolve_sparse_kv_budget(
        request_rows=request_rows,
        context_len=context_len,
        num_layers=num_layers,
        top_k=top_k,
        kv_bytes_per_token=kv_bytes_per_token,
        cache_tokens=envs.SGLANG_NPU_SPARSE_KV_CACHE_SIZE.get(),
        hbm_budget_bytes=envs.SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB.get() * 1024**2,
        indexer_impl=get_sparse_kv_indexer_impl(),
        copy_sfa=envs.SGLANG_NPU_ENABLE_COPY_SFA.get(),
        copy_sfa_query_heads=copy_sfa_query_heads,
        mtp_full_visible=envs.SGLANG_ENABLE_NPU_LIM_MTP_FULL_VISIBLE.get(),
        fused_copy_sfa_metadata=envs.SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA.get(),
        fused_pre_metadata=envs.SGLANG_NPU_FUSED_LIM_PRE_METADATA.get(),
        graph_batch_capacity=sparse_batch_capacity(
            request_rows,
            max(graph_batch_capacity, get_eager_max_batch_size(request_rows - 1)),
        ),
    )
