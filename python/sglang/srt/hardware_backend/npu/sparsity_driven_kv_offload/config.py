"""Configuration and validation for sparsity-driven KV offload."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

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
    fit_sparse_running_requests,
    get_sparse_kv_cache_tokens,
    resolve_sparse_kv_budget,
)
from sglang.srt.mem_cache.allocation_sizing import get_req_to_token_row_width
from sglang.srt.runtime_context import (
    attention_backends,
    get_disagg,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.utils.common import ceil_div
from sglang.srt.utils.device import is_npu

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


def use_fused_sparse_kv_offload() -> bool:
    return envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.get()


def _sparse_kv_offload_enabled_for_model(
    *,
    model_config: ModelConfig,
    use_mla_backend: bool,
) -> bool:
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
    return True


def _require_sparse_kv_request_limit() -> None:
    if get_schedule().max_running_requests is None:
        raise ValueError(
            "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD requires max_running_requests "
            "to be set to bound the per-process host KV allocation."
        )


def is_sparsity_driven_kv_offload_enabled(
    *,
    model_config: ModelConfig,
    use_mla_backend: bool,
) -> bool:
    if not _sparse_kv_offload_enabled_for_model(
        model_config=model_config, use_mla_backend=use_mla_backend
    ):
        return False
    _require_sparse_kv_request_limit()
    return True


def is_fused_sparse_kv_offload_enabled(
    *,
    model_config: ModelConfig,
    use_mla_backend: bool,
    is_draft_worker: bool = False,
) -> bool:
    """Return whether this worker uses fused LIM and Copy-SFA offload."""
    # Draft attention keeps its full KV pool even when it inherits the flags.
    if not use_fused_sparse_kv_offload() or is_draft_worker:
        return False
    if not _sparse_kv_offload_enabled_for_model(
        model_config=model_config, use_mla_backend=use_mla_backend
    ):
        return False

    disagg = get_disagg()
    if disagg.disaggregation_mode == "prefill":
        # Prefill publishes full KV pages to the decode worker.
        _require_sparse_kv_request_limit()
        return False
    if disagg.enable_pd_role_switch:
        raise ValueError("Fused sparse KV offload does not support PD role switching.")
    parallel = get_parallel()
    if parallel.pp_size != 1 or parallel.dcp_enabled:
        raise ValueError(
            "Native LIM integration currently requires PP=1 and DCP disabled."
        )
    if disagg.disaggregation_mode == "decode":
        if parallel.attn_cp_size != 1:
            raise ValueError("PD split KV requires context parallelism disabled.")
        if disagg.disaggregation_transfer_backend not in ("ascend", "mooncake"):
            raise ValueError(
                "PD split KV requires the Ascend or Mooncake transfer backend."
            )
        if disagg.disaggregation_decode_host_receive_threshold < 1:
            raise ValueError("PD split KV requires host receive threshold 1.")
    algorithm = (get_spec().speculative_algorithm or "NONE").upper()
    if algorithm != "NONE" and (
        algorithm not in ("EAGLE", "NEXTN") or get_spec().speculative_eagle_topk != 1
    ):
        raise ValueError(
            "Sparse KV MTP requires EAGLE/NEXTN with --speculative-eagle-topk 1."
        )
    return True


def get_sparse_kv_offload_mode(
    *,
    model_config: ModelConfig,
    use_mla_backend: bool,
    is_draft_worker: bool = False,
) -> Literal["legacy", "fused"] | None:
    """Resolve the offload mode used by shared setup and memory accounting."""
    if use_fused_sparse_kv_offload():
        if is_fused_sparse_kv_offload_enabled(
            model_config=model_config,
            use_mla_backend=use_mla_backend,
            is_draft_worker=is_draft_worker,
        ):
            return "fused"
    elif is_sparsity_driven_kv_offload_enabled(
        model_config=model_config,
        use_mla_backend=use_mla_backend,
    ):
        return "legacy"
    return None


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
) -> Optional[int]:
    if (
        get_sparse_kv_offload_mode(
            model_config=model_config,
            use_mla_backend=use_mla_backend,
            is_draft_worker=is_draft_worker,
        )
        is None
    ):
        return None

    index_head_dim = get_sparsity_driven_kv_offload_index_head_dim(
        model_config=model_config
    )
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
    return index_head_dim * num_indexer_layers * element_size


def build_sparse_kv_budget_options(
    *,
    model_config: ModelConfig,
    kv_dtype: torch.dtype,
    num_layers: int,
    attn_dp_size: int,
) -> dict[str, Any]:
    """Resolve the initial sparse budget inputs for this worker."""
    return dict(
        model_config=model_config,
        attn_dp_size=attn_dp_size,
        context_len=get_req_to_token_row_width(model_config.context_len),
        top_k=get_sparsity_driven_kv_offload_sparse_context_len(
            model_config=model_config
        ),
        kv_bytes_per_token=model_config.get_total_num_kv_heads(compressed_mla=True)
        * (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
        * torch._utils._element_size(kv_dtype),
        num_layers=num_layers,
        kv_dtype=kv_dtype,
    )


def get_sparse_kv_budget(
    *,
    model_config: ModelConfig,
    context_len: int,
    num_layers: int,
    top_k: int,
    kv_bytes_per_token: int,
    available_bytes: int,
    index_k_and_draft_bytes_per_token: int,
    attn_dp_size: int = 1,
    kv_dtype: torch.dtype = torch.bfloat16,
) -> BudgetPlan:
    schedule = get_schedule()
    disagg = get_disagg()
    page_size = schedule.page_size
    aligned_context = ceil_div(context_len, page_size) * page_size
    max_query_width = get_sparse_kv_max_query_width()
    cache_tokens = get_sparse_kv_cache_tokens(
        envs.SGLANG_NPU_SPARSE_KV_CACHE_SIZE.get(),
        max_query_width,
        top_k,
    )
    running_rows = fit_sparse_running_requests(
        available_bytes,
        max_total_tokens=schedule.max_total_tokens,
        cache_tokens=cache_tokens,
        context_len=aligned_context,
        num_layers=num_layers,
        kv_bytes_per_token=kv_bytes_per_token,
        index_k_and_draft_bytes_per_token=index_k_and_draft_bytes_per_token,
    )
    if schedule.max_running_requests is not None:
        running_rows = min(
            running_rows, max(1, schedule.max_running_requests // attn_dp_size)
        )
    resident_rows = running_rows + 1
    if disagg.disaggregation_mode == "decode":
        request_rows = resident_rows + (disagg.disaggregation_decode_extra_slots or 0)
        prefill_rows = 0
    else:
        request_rows = resident_rows
        prefill_rows = schedule.prefill_max_requests or 1

    query_heads = model_config.num_attention_heads // get_parallel().attn_tp_size
    # Native SFA accepts 1..64 or 128 query heads per KV head.
    copy_sfa_query_heads = 128 if 64 < query_heads < 128 else query_heads
    return resolve_sparse_kv_budget(
        model_config=model_config,
        page_size=page_size,
        request_rows=request_rows,
        resident_rows=resident_rows,
        context_len=context_len,
        num_layers=num_layers,
        top_k=top_k,
        kv_bytes_per_token=kv_bytes_per_token,
        cache_tokens=cache_tokens,
        kv_dtype=kv_dtype,
        prefill_workspace_tokens=prefill_rows * aligned_context,
        max_query_width=max_query_width,
        copy_sfa_query_heads=copy_sfa_query_heads,
    )


def get_sparse_kv_max_query_width() -> int:
    spec = get_spec()
    # The budget resolver validates the resulting native query geometry.
    return spec.speculative_num_draft_tokens or 1
