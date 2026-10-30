"""Persistent cache identities for CP encoder replay producers."""

import hashlib
import json


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
