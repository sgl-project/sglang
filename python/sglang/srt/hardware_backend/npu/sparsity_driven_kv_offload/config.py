"""Configuration and validation for sparsity-driven KV offload."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.configs.model_config import (
    dsa_layer_skips_topk,
    get_dsa_index_head_dim,
    get_dsa_index_topk,
    is_deepseek_dsa,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
    BudgetPlan,
    resolve_sparse_kv_budget,
)
from sglang.srt.runtime_context import (
    attention_backends,
    get_disagg,
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.utils.common import (
    get_cuda_graph_max_batch_size,
    get_eager_max_batch_size,
)
from sglang.srt.utils.device import is_npu

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


def use_fused_sparse_kv_offload() -> bool:
    return envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.get()


def is_sparsity_driven_kv_offload_enabled(
    *,
    model_config: ModelConfig,
    use_mla_backend: bool,
    is_draft_worker: bool = False,
) -> bool:
    # Offload is a target-model policy. Draft attention keeps its native KV
    # pool even though the process inherits the target's environment flags.
    if is_draft_worker:
        return False
    if not envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.get():
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
    if not use_fused_sparse_kv_offload():
        return True
    if int(get_dsa_index_topk(model_config.hf_config)) != 2048:
        raise ValueError("Native LIM integration requires model index_topk=2048.")
    if get_schedule().page_size != 128:
        raise ValueError("Native LIM integration requires --page-size 128.")
    if get_parallel().pp_size != 1 or get_parallel().dcp_enabled:
        raise ValueError(
            "Native LIM integration currently requires PP=1 and DCP disabled."
        )
    algorithm = (get_spec().speculative_algorithm or "NONE").upper()
    if algorithm != "NONE" and (
        algorithm not in ("EAGLE", "NEXTN") or get_spec().speculative_eagle_topk != 1
    ):
        raise ValueError(
            "Sparse KV MTP requires EAGLE/NEXTN with --speculative-eagle-topk 1."
        )
    return True


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
    start_layer: int = 0,
    is_draft_worker: bool = False,
    indexer_layer_ids: Optional[Sequence[int]] = None,
    kv_cache_dtype: Optional[torch.dtype] = None,
) -> Optional[int]:
    if not is_sparsity_driven_kv_offload_enabled(
        model_config=model_config,
        use_mla_backend=use_mla_backend,
        is_draft_worker=is_draft_worker,
    ):
        return None

    index_head_dim = get_sparsity_driven_kv_offload_index_head_dim(
        model_config=model_config
    )
    if indexer_layer_ids is not None:
        num_indexer_layers = len(indexer_layer_ids)
    else:
        from sglang.srt.mem_cache.kv_cache_configurator import (
            _should_elide_dsa_index_k,
        )

        if _should_elide_dsa_index_k(is_draft_worker=is_draft_worker):
            num_indexer_layers = sum(
                not dsa_layer_skips_topk(model_config.hf_config, layer_id)
                for layer_id in range(start_layer, start_layer + num_layers)
            )
        else:
            num_indexer_layers = num_layers
    layer_bytes = index_head_dim * element_size
    if kv_cache_dtype == torch.float8_e4m3fn:
        # NPU FP8 index keys have one FP32 scale per token and indexer layer.
        layer_bytes += torch.float32.itemsize
    return layer_bytes * num_indexer_layers


def get_sparse_kv_budget(
    *,
    model_config: ModelConfig,
    request_rows: int,
    context_len: int,
    num_layers: int,
    top_k: int,
    kv_bytes_per_token: int,
    copy_sfa_query_heads: Optional[int] = None,
    graph_batch_capacity: Optional[int] = None,
    kv_dtype: torch.dtype = torch.bfloat16,
) -> BudgetPlan:
    if not use_fused_sparse_kv_offload():
        raise ValueError("Sparse KV budgeting requires fused LIM and CopySFA.")
    if copy_sfa_query_heads is None:
        query_heads = model_config.num_attention_heads // get_parallel().attn_tp_size
        if query_heads <= 0:
            raise ValueError("CopySFA requires a positive per-rank query head count.")
        copy_sfa_query_heads = 1 << (query_heads - 1).bit_length()
    disagg = get_disagg()
    prefill_rows = request_rows - 1
    if disagg.disaggregation_mode == "decode" and not disagg.enable_pd_role_switch:
        prefill_rows = 0
    elif get_schedule().prefill_max_requests is not None:
        prefill_rows = min(prefill_rows, get_schedule().prefill_max_requests)
    page_size = get_schedule().page_size
    aligned_context = (context_len + page_size - 1) // page_size * page_size
    if graph_batch_capacity is None:
        # Reserve before admission determines the final request-pool size.
        # Capture buckets may round that size up for attention parallelism.
        graph = get_exec().graph
        config = graph.cuda_graph_config
        decode = config.decode if config is not None else None
        graph_batch_capacity = 0
        if not graph.disable_cuda_graph:
            graph_batch_capacity = get_cuda_graph_max_batch_size(request_rows)
            if decode is not None and decode.bs:
                graph_batch_capacity = min(max(decode.bs), graph_batch_capacity)
    return resolve_sparse_kv_budget(
        model_config=model_config,
        page_size=page_size,
        request_rows=request_rows,
        context_len=context_len,
        num_layers=num_layers,
        top_k=top_k,
        kv_bytes_per_token=kv_bytes_per_token,
        cache_tokens=envs.SGLANG_NPU_SPARSE_KV_CACHE_SIZE.get(),
        hbm_budget_bytes=envs.SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB.get() * 1024**2,
        kv_dtype=kv_dtype,
        prefill_workspace_tokens=prefill_rows * aligned_context,
        max_query_width=get_sparse_kv_max_query_width(),
        copy_sfa_query_heads=copy_sfa_query_heads,
        graph_batch_capacity=max(
            request_rows,
            graph_batch_capacity,
            get_eager_max_batch_size(request_rows - 1),
        ),
    )


def get_sparse_kv_max_query_width() -> int:
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
        MAX_QUERY_WIDTH,
    )

    spec = get_spec()
    width = int(spec.speculative_num_draft_tokens or 1)
    if not 1 <= width <= MAX_QUERY_WIDTH:
        raise ValueError(
            f"Native LIM supports chain query widths 1..{MAX_QUERY_WIDTH}."
        )
    return width
