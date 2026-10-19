# SPDX-License-Identifier: Apache-2.0
"""Contracts for the SM90 mixed NVFP4/BF16 DSA cache."""


def validate_dsa_nvfp4(cfg, hf_config, *, is_sm90: bool) -> None:
    """Reject combinations whose writers/consumers do not implement this ABI."""
    if not is_sm90:
        raise ValueError("DSA NVFP4 KV cache requires SM90 (Hopper)")
    if getattr(cfg, "dtype", "auto") not in ("auto", "bfloat16", "bf16"):
        raise ValueError("DSA NVFP4 attention requires bfloat16 model activations")
    geometry = (
        getattr(hf_config, "kv_lora_rank", None),
        getattr(hf_config, "qk_rope_head_dim", None),
    )
    if geometry != (512, 64):
        raise ValueError(
            f"DSA NVFP4 requires latent/RoPE dimensions (512, 64), got {geometry}"
        )
    if getattr(hf_config, "index_kpool", 1) != 1:
        raise ValueError("DSA NVFP4 does not support pooled indexers (index_kpool > 1)")
    if getattr(hf_config, "learnable_sink", False):
        raise ValueError("DSA NVFP4 does not support attention sinks")
    if getattr(hf_config, "index_topk", 2048) != 2048:
        raise ValueError("DSA NVFP4 currently requires index_topk=2048")
    for name, expected in (
        ("dsa_prefill_backend", "flashmla_sparse"),
        ("dsa_decode_backend", "flashmla_kv"),
    ):
        if getattr(cfg, name, None) not in (None, expected):
            raise ValueError(f"DSA NVFP4 requires {name}={expected}")
    for name in (
        "enable_hisparse",
        "enable_dsa_cache_layer_split",
        "enable_prefill_cp",
        "enable_two_batch_overlap",
        "enable_unified_memory",
        "enable_hierarchical_cache",
        "enable_lmcache",
    ):
        if getattr(cfg, name, False):
            raise ValueError(f"DSA NVFP4 does not yet support {name}")
    for name in ("pp_size", "attn_cp_size", "dcp_size"):
        if getattr(cfg, name, 1) not in (None, 1):
            raise ValueError(f"DSA NVFP4 currently requires {name}=1")
    if getattr(cfg, "speculative_algorithm", None) is not None:
        raise ValueError("DSA NVFP4 speculative decoding is not yet validated")
    if getattr(cfg, "disaggregation_mode", None) not in (None, "null"):
        raise ValueError("DSA NVFP4 does not yet support prefill/decode disaggregation")
    if getattr(cfg, "page_size", None) not in (None, 64):
        raise ValueError("DSA NVFP4 requires page_size=64")
