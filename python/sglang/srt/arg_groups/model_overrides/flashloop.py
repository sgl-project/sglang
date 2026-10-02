# SPDX-License-Identifier: Apache-2.0
"""Constrain the opt-in FlashLoop prototype to its supported execution paths."""

from sglang.srt.arg_groups.model_override_base import _register_for, resolving_view
from sglang.srt.configs.flashloop import component_options, validate_runtime_options
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    with_phase,
)
from sglang.srt.runtime_context import get_platform


@_register_for("FlashLoopOuroForCausalLM")
def _flashloop_overrides(server_args, hf_config):
    cfg = resolving_view(server_args)
    if not get_platform().is_cuda:
        raise ValueError("FlashLoop currently requires NVIDIA CUDA")
    _, _, quantized = component_options(hf_config)
    # Read immutable input choices; resolution applies the returned declarations.
    names = (
        "tp_size",
        "pp_size",
        "dp_size",
        "dtype",
        "kv_cache_dtype",
        "speculative_algorithm",
        "quantization",
        "prefill_attention_backend",
        "decode_attention_backend",
        "enable_deterministic_inference",
        "enable_hierarchical_cache",
        "enable_memory_saver",
        "enable_unified_memory",
        "enable_hisparse",
        "enable_lora",
        "enable_torch_compile",
        "prefill_only_disable_kv_cache",
        "enable_pdmux",
        "disaggregation_mode",
    )
    options = {name: getattr(cfg, name) for name in names}
    options["dtype"] = "bfloat16" if options["dtype"] == "auto" else options["dtype"]
    options["attention_backend"] = cfg.attention_backend or "triton"
    validate_runtime_options(options)
    if cfg.dcp_size != 1:
        raise ValueError("FlashLoop currently requires DCP=1")
    graph = cfg.cuda_graph_config or CudaGraphConfig()
    if graph.decode.backend not in (Backend.FULL, Backend.DISABLED):
        raise ValueError("FlashLoop supports full decode CUDA graphs or eager decode")
    overrides = dict(
        attention_backend="triton",
        dtype="bfloat16",
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        disable_overlap_schedule=True,
        cuda_graph_config=with_phase(graph, Phase.PREFILL, backend=Backend.DISABLED),
    )
    if quantized:
        if cfg.page_size not in (None, 1, 64):
            raise ValueError("FlashLoop INT4 KV requires 64-token pages")
        overrides["page_size"] = 64
        if cfg.max_running_requests is None:
            overrides["max_running_requests"] = 4
    return overrides
