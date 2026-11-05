"""Optional framework binding and early reservation for KVServe's external runtime."""

import json
import logging
import os
import threading
from dataclasses import replace
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_runtimes = {}
_lock = threading.Lock()
_DEFAULT_CHUNK_TOKENS = 1024


def is_enabled():
    path = os.environ.get("SGLANG_KVSERVE_CONFIG")
    if not path:
        return False
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("SGLANG_KVSERVE_CONFIG must contain a JSON object")
    enabled = values.get("enable-compression-pipeline", False)
    if type(enabled) is not bool:
        raise ValueError("enable-compression-pipeline must be a boolean")
    return enabled


def parse_compression_config(values):
    """Apply SGLang defaults and reject unsupported or non-compressing profiles."""
    if not isinstance(values, dict):
        raise ValueError("SGLANG_KVSERVE_CONFIG must contain a JSON object")
    from kvserve_v1.compression.runtime import CompressionConfig

    normalized = dict(values)
    if normalized.get("enable-compression-pipeline") is True:
        normalized.setdefault("logical_blocks_per_chunk", _DEFAULT_CHUNK_TOKENS)
    config = CompressionConfig.from_extra_config({"compression": normalized})
    if not config.enabled:
        return config
    if config.uses_aggregate or not (config.uses_quantizer or config.uses_codec):
        raise ValueError("KVServe requires a quantizer or codec stage")
    if config.slot_count not in (1, 2, 4, 8):
        raise ValueError("KVServe supports 1, 2, 4 or 8 bounded slots")
    if config.uses_quantizer:
        if "tensor" in (config.axis_key, config.axis_value):
            raise ValueError(
                "KVServe C++/CUDA quantization does not support tensor axis"
            )
        if config.split_type == "head" and not config.score_path:
            raise ValueError("KVServe head split requires a score_path")
    if config.score_path and not (
        config.uses_quantizer and config.split_type == "head"
    ):
        raise ValueError("KVServe score_path requires a quantizer with head split")
    if "hybrid_ratio" in normalized and not (
        config.uses_quantizer and config.split_type == "head"
    ):
        raise ValueError("KVServe hybrid_ratio requires a quantizer with head split")
    return config


def select_execution_backend(config, requested_backend):
    """Keep the default codec fused; use the standard CUDA path for other profiles."""
    from kvserve_v1.compression.runtime import CompressionConfig

    if requested_backend not in ("auto", "standard", "fused"):
        raise ValueError("Unknown KVServe execution backend")
    default = CompressionConfig.from_extra_config(
        {"compression": {"enable-compression-pipeline": True}}
    )
    is_default = (
        replace(
            config,
            slot_count=default.slot_count,
            logical_blocks_per_chunk=default.logical_blocks_per_chunk,
        )
        == default
    )
    if not is_default:
        if requested_backend == "fused":
            raise ValueError(
                "KVServe fused execution requires the default codec profile"
            )
        return "standard"
    return requested_backend


def prepare_runtime(kvc):
    """Allocate codec memory before KV sizing, so pre/post-capture free memory charges it."""
    path = os.environ.get("SGLANG_KVSERVE_CONFIG")
    if not is_enabled():
        return
    from sglang.srt.runtime_context import get_disagg, get_parallel

    disagg = get_disagg()
    mode = disagg.disaggregation_mode
    if mode not in ("prefill", "decode"):
        raise ValueError("KVServe compression requires prefill or decode mode")
    if disagg.disaggregation_transfer_backend != "mooncake":
        raise ValueError("KVServe compression requires the Mooncake transfer backend")

    from kvserve_v1.compression.runtime import (
        CacheSpec,
        build_memory_plan,
    )

    parallel = get_parallel()
    if (
        kvc.is_draft_worker
        or kvc.use_mla_backend
        or kvc.is_hybrid_swa
        or kvc.pp_size != 1
        or parallel.attn_tp_size != 1
        or parallel.attn_dcp_size != 1
        or kvc.kv_cache_dtype != torch.bfloat16
    ):
        raise ValueError("KVServe early reservation requires BF16 TP1 full attention")
    if kvc.mambaish_config is not None:
        from sglang.srt.runtime_context import get_memory

        if get_memory().enable_unified_memory:
            raise ValueError("KVServe hybrid attention does not support unified memory")
    with _lock:
        if kvc.gpu_id in _runtimes:
            return
        config = parse_compression_config(json.loads(Path(path).read_text()))
        backend = select_execution_backend(
            config, os.environ.get("SGLANG_KVSERVE_EXECUTION_BACKEND", "auto")
        )
        if kvc.page_size < 1 or config.logical_blocks_per_chunk % kvc.page_size:
            raise ValueError(
                "KVServe chunk tokens must be divisible by native page size"
            )
        layer_ids = range(kvc.layer_info.start_layer, kvc.layer_info.end_layer)
        if kvc.mambaish_config is not None:
            full_layers = set(kvc.mambaish_config.full_attention_layer_ids)
            layer_ids = [i for i in layer_ids if i in full_layers]
        if not layer_ids:
            raise ValueError("KVServe requires at least one full-attention KV layer")
        specs = {
            f"layer.{i}": CacheSpec(
                i,
                1,
                kvc.model_config.get_num_kv_heads(1),
                kvc.model_config.head_dim,
                kvc.kv_cache_dtype,
                kvc.model_config.v_head_dim,
            )
            for i in layer_ids
        }
        device = torch.device("cuda", kvc.gpu_id)
        plan = build_memory_plan(
            config,
            specs,
            blocks_per_logical=1,
            tp_size=1,
            tp_rank=0,
            num_layers=kvc.model_config.num_hidden_layers,
            device=device,
            warm_cpp_backend=True,
        )
        from kvserve_v1.compression.runtime.fused_runtime import create_runtime

        execution = {"prefill": "encode", "decode": "decode"}[mode]
        banks = 4 if mode == "prefill" else 0
        runtime = create_runtime(
            plan,
            "HND",
            1,
            device,
            backend=backend,
            output_slots=banks,
            execution=execution,
            standard_backend="cpp",
        )
        logger.info(
            "KVServe execution=%s capability=%s output_slots=%d pipeline=%s",
            type(runtime).__name__,
            execution,
            banks,
            config.pipeline,
        )
        _runtimes[kvc.gpu_id] = runtime
        logger.info(
            "KVServe reserved before KV sizing: owned=%d driver_estimate=%d bulk=%d",
            runtime.plan.reserved_bytes - runtime.plan.retained_driver_bytes,
            runtime.plan.retained_driver_bytes,
            runtime.bulk.numel(),
        )


def take_runtime(device):
    with _lock:
        return _runtimes.pop(device.index, None)
