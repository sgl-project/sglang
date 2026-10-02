# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FlashLoop contributors
"""Checkpoint metadata adapter (does not import checkpoint modeling code)."""

import math


def adapt_config(
    data,
    decode_fraction=0.1,
    prefill_fractions=(1.0, 1.0),
    enable_token_sparse_prefill=True,
    enable_sparse_decode=True,
    quantize_cross_loop_kv=False,
):
    data = dict(data)
    if data.get("model_type") != "ouro":
        raise ValueError("FlashLoop SGLang currently supports Ouro only")
    if not math.isfinite(decode_fraction) or not 0 < decode_fraction <= 1:
        raise ValueError("decode_key_fraction must be in (0, 1]")
    if (
        len(prefill_fractions) != 2
        or not 0 < prefill_fractions[1] <= prefill_fractions[0] <= 1
    ):
        raise ValueError("Prefill fractions require 0 < loop4 <= loop3 <= 1")
    if data.get("total_ut_steps", 4) != 4:
        raise ValueError("FlashLoop SGLang requires four recurrent steps")
    if data["num_attention_heads"] != data["num_key_value_heads"]:
        raise ValueError("This implementation requires equal Q and KV head counts")
    data["flashloop_physical_layers"] = data.get(
        "flashloop_physical_layers", data["num_hidden_layers"]
    )
    data["num_hidden_layers"] = (
        data["flashloop_physical_layers"] * 4
    )  # independent paged KV slots for each recurrence
    data["architectures"] = ["FlashLoopOuroForCausalLM"]
    data["flashloop_decode_fraction"] = decode_fraction if enable_sparse_decode else 1.0
    data["flashloop_prefill_fractions"] = list(
        prefill_fractions if enable_token_sparse_prefill else (1.0, 1.0)
    )
    data["flashloop_components"] = dict(
        token_sparse_prefill=enable_token_sparse_prefill,
        sparse_decode=enable_sparse_decode,
        kv_residual_quantization=quantize_cross_loop_kv,
    )
    data["head_dim"] = data["hidden_size"] // data["num_attention_heads"]
    data["layer_types"] = ["full_attention"] * data["num_hidden_layers"]
    return data


def validate_runtime_options(options):
    """Fail early for paths that need different KV or graph contracts."""
    for name in ("tp_size", "pp_size", "dp_size"):
        if options.get(name, 1) != 1:
            raise ValueError(f"Initial FlashLoop SGLang backend requires {name}=1")
    if options.get("attention_backend", "triton") != "triton":
        raise ValueError("Initial FlashLoop SGLang backend requires Triton attention")
    for name in ("prefill_attention_backend", "decode_attention_backend"):
        if options.get(name) not in (None, "triton"):
            raise ValueError(f"{name} must be Triton")
    if options.get("kv_cache_dtype", "auto") not in ("auto", "bfloat16"):
        raise ValueError(
            "Use auto/BF16 cache dtype; recurrent INT4 is controlled by quantize_cross_loop_kv"
        )
    if options.get("dtype", "bfloat16") != "bfloat16":
        raise ValueError("Initial FlashLoop SGLang backend requires BF16")
    if (
        options.get("speculative_algorithm") is not None
        or options.get("quantization") is not None
    ):
        raise ValueError("Speculation and quantized weights are not integrated yet")
    if (
        not options.get("disable_radix_cache", True)
        or options.get("chunked_prefill_size", -1) != -1
    ):
        raise ValueError(
            "Disable prefix cache and chunked prefill for cross-loop token selection"
        )
    for name in (
        "enable_deterministic_inference",
        "enable_hierarchical_cache",
        "enable_memory_saver",
        "enable_unified_memory",
        "enable_hisparse",
        "enable_lora",
        "enable_torch_compile",
        "prefill_only_disable_kv_cache",
        "enable_pdmux",
    ):
        if options.get(name, False):
            raise ValueError(f"{name} is not supported by the recurrent cache")
    if options.get("disaggregation_mode", "null") not in (None, "null"):
        raise ValueError("Disaggregated serving is not integrated")
    if not options.get("disable_overlap_schedule", True):
        raise ValueError(
            "Overlap scheduling is not validated for the shared loop state"
        )


def component_options(config):
    """Resolve independent switches after JSON model overrides have been applied."""
    components = getattr(config, "flashloop_components", {})
    names = ("token_sparse_prefill", "sparse_decode", "kv_residual_quantization")
    if not isinstance(components, dict) or set(components) - set(names):
        raise ValueError("Unknown FlashLoop component; expected " + ", ".join(names))
    if any(type(value) is not bool for value in components.values()):
        raise ValueError("FlashLoop component switches must be booleans")
    prefill = tuple(config.flashloop_prefill_fractions)
    decode = config.flashloop_decode_fraction
    if len(prefill) != 2 or not 0 < prefill[1] <= prefill[0] <= 1:
        raise ValueError("Require 0 < loop4 fraction <= loop3 fraction <= 1")
    if not math.isfinite(decode) or not 0 < decode <= 1:
        raise ValueError("Decode fraction must be in (0, 1]")
    return (
        prefill if components.get(names[0], False) else (1.0, 1.0),
        decode if components.get(names[1], False) else 1.0,
        components.get(names[2], False),
    )


def register_parser():
    from transformers import PretrainedConfig

    from sglang.srt.configs.model_config_parser_registry import (
        ModelConfigParserBase,
        register_model_config_parser,
    )

    @register_model_config_parser("flashloop")
    class FlashLoopConfigParser(ModelConfigParserBase):
        def parse(self, model, trust_remote_code, revision=None, **kwargs):
            data, _ = PretrainedConfig.get_config_dict(model, revision=revision)
            adapted = adapt_config(data, prefill_fractions=(0.25, 0.1))
            adapted["flashloop_components"] = {}
            return PretrainedConfig.from_dict(adapted)
