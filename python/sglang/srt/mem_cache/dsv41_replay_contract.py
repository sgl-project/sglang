"""Contracts for the CP>1 extension of upstream V4.1 encoder replay.

CP1 keeps the upstream folded extend, kernels, and cache identity. Resolve
parallelism before using these predicates; target and draft runners must not
share a process-global replay policy.
"""

import hashlib
import json


def is_cp_encoder_replay(cfg) -> bool:
    return cfg.enable_encoder_swa_bounded_replay and cfg.attn_cp_size > 1


def validate_cp_encoder_replay(cfg, *, is_cuda: bool, is_sm90: bool) -> None:
    """Constrain the new path without changing any upstream CP1 recipe."""
    if not is_cp_encoder_replay(cfg):
        return
    if not is_cuda or not is_sm90:
        raise ValueError("CP encoder replay currently requires SM90 CUDA")
    if any(
        getattr(cfg, name) != 1
        for name in ("pp_size", "dp_size", "dcp_size", "attn_dp_size")
    ):
        raise ValueError("CP encoder replay requires PP=1, DP=1 and DCP=1")
    if not cfg.enable_prefill_cp or cfg.cp_strategy != "interleave":
        raise ValueError("CP encoder replay requires interleave prefill CP")
    if cfg.attn_cp_size != cfg.tp_size:
        raise ValueError("CP encoder replay requires attention TP=1")
    if cfg.cuda_graph_config.prefill.backend != "disabled":
        raise ValueError("CP encoder replay requires disabled prefill CUDA graphs")
    if cfg.disaggregation_mode == "decode":
        raise ValueError("encoder replay is a Prefill feature; Decode requires CP=1")
    if cfg.disaggregation_mode == "prefill" and (
        cfg.disaggregation_transfer_backend != "mooncake"
        or str(cfg.speculative_algorithm).upper() != "DSPARK"
    ):
        raise ValueError("CP encoder replay PD requires Mooncake and static DSpark")
    # The existing DSpark validator checks the static verify mode and wire layout.
    if cfg.moe_a2a_backend not in (None, "none"):
        raise ValueError("CP encoder replay currently requires native MoE dispatch")
    if cfg.enable_kv_cache_sharding:
        raise ValueError("CP encoder replay does not support KV cache sharding yet")
    if cfg.enable_dsa_cache_layer_split or cfg.enable_cp_tp_group_sharing:
        raise ValueError("CP encoder replay requires replicated per-layer KV pools")
    if cfg.enable_hierarchical_cache:
        if cfg.hicache_host_memory_mode != "cache":
            raise ValueError("CP encoder replay requires cache host memory mode")
        if cfg.hicache_storage_backend not in (None, "mooncake"):
            raise ValueError("CP encoder replay L3 currently requires Mooncake Store")


def sync_restorable_prefix(
    operation, storage_hit_count, *, page_size, pools, reduce_min
):
    """Intersect restore endpoints on the caller's synchronization groups.

    Scheduler probes and background prefetch use distinct groups. Every peer
    reduces the same header, including misses, query failures and cancellation.
    Only a shared positive bound with endpoint evidence adds a mask reduction.
    """
    import torch

    restorable = getattr(operation, "restorable_prefix_pages", None)
    local = getattr(operation, "query_pool_hit_pages", {})
    header = torch.tensor(
        [storage_hit_count, int(restorable is None)]
        + [local.get(pool, -1) for pool in pools],
        dtype=torch.int,
    )
    reduce_min(header)
    count, all_legacy, *verdicts = header.tolist()
    if hasattr(operation, "query_pool_hit_pages"):
        operation.query_pool_hit_pages = {
            pool: pages
            for pool, pages in zip(pools, verdicts, strict=True)
            if pages >= 0
        }
    if count <= 0 or all_legacy:
        return count
    upper = count // page_size
    mask = (
        torch.ones(upper + 1, dtype=torch.int)
        if restorable is None
        else torch.zeros(upper + 1, dtype=torch.int)
    )
    mask[0] = 1
    if restorable is not None:
        endpoints = [p for p in restorable if 0 < p <= upper]
        if endpoints:
            mask[endpoints] = 1
    reduce_min(mask)
    return int(torch.nonzero(mask, as_tuple=True)[0][-1]) * page_size


def replay_storage_tag(pool, *, producer):
    """Identity for CP>1 producers; CP1 never calls this function.

    The supported pools are replicated across CP; existing component keys and
    the single-writer policy address them. Request windows, draft rings, hidden
    states, and pool capacities are not persisted. Cross-CP-size sharing is not
    assumed when native numerical schedules differ.
    """
    layout = {
        "version": 1,
        "execution": "upstream-folded-extend-cp-v1",
        "producer": producer,
        "kv_layout": pool.kv_layout.value,
        "compressed_kv_layouts": [
            [r, pool.compressed_kv_layout(r).value]
            for r in sorted(set(pool.compression_ratios) - {0})
        ],
        "page_size": pool.page_size,
        "compression_ratios": list(pool.compression_ratios),
        "kv_sources": list(pool.kv_source_layers),
    }
    payload = json.dumps(
        layout, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return "dsv41-cp-replay-v1-" + hashlib.sha256(payload).hexdigest()[:24]


def runtime_replay_producer():
    """Snapshot the target recipe once during CP cache construction."""
    import importlib.metadata

    import torch

    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
    from sglang.srt.layers.quantization.fp8_utils import get_fp8_gemm_runner_backend
    from sglang.srt.runtime_context import get_exec, get_parallel, process_model_config

    parallel, model = get_parallel(), process_model_config()
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    versions = {}
    for package in ("torch", "sglang-kernel", "flashinfer-python", "sgl-deep-gemm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "device": [
            device.name,
            device.major,
            device.minor,
            device.multi_processor_count,
        ],
        "parallel": {
            name: getattr(parallel, name)
            for name in (
                "tp_size",
                "ep_size",
                "pp_size",
                "dp_size",
                "attn_dp_size",
                "attn_cp_size",
                "attn_tp_size",
                "cp_strategy",
            )
        },
        "fp8_gemm": get_fp8_gemm_runner_backend().value,
        "moe_runner": get_moe_runner_backend().value,
        "moe_a2a": get_moe_a2a_backend().value,
        "moe_precision": get_exec().moe.flashinfer_mxfp4_moe_precision,
        "dtype": str(model.dtype),
        "quantization": model.quantization,
        "hf_quant_config": model.hf_quant_config,
        "versions": versions,
        "options": {
            name: getattr(envs, name).get()
            for name in (
                "SGLANG_OPT_HOPPER_BLOCK_FP8_BF16",
                "SGLANG_DSV4_DENSE_MXSCALE",
                "SGLANG_DSV4_DENSE_MXSCALE_SHAPES",
                "SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE",
                "SGLANG_DSV41_FUSED_WO_A",
                "SGLANG_OPT_FP8_WO_A_GEMM",
                "SGLANG_OPT_FP8_WO_A_FUSED_INVROPE",
                "SGLANG_OPT_FUSE_WQA_WKV",
                "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE",
                "SGLANG_OPT_USE_FLASHINFER_MHC",
                "SGLANG_OPT_USE_TILELANG_MHC_PRE",
                "SGLANG_OPT_USE_TILELANG_MHC_POST",
                "SGLANG_OPT_FUSE_MHC_POST_PRE",
                "SGLANG_OPT_DEEPGEMM_HC_PRENORM",
                "SGLANG_OPT_DSV41_MEGA_MHC_PREFILL",
            )
        },
    }


def with_storage_layout_tag(extra_config, tag):
    config = dict(extra_config or {})
    original = config.get("extra_backend_tag")
    config["extra_backend_tag"] = f"{original}_{tag}" if original is not None else tag
    return config
