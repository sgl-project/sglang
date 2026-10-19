# SPDX-License-Identifier: Apache-2.0
"""Contracts for the SM90 DSA NVFP4 cache layouts and consumers."""


def is_glm5_next_nvfp4(hf_config) -> bool:
    return "Glm5NextForConditionalGeneration" in getattr(hf_config, "architectures", [])


def dsa_nvfp4_cache_dim(kv_lora_rank: int, qk_rope_head_dim: int) -> int:
    """Physical bytes per row, including block scales and optional BF16 RoPE."""
    if kv_lora_rank != 512 or qk_rope_head_dim not in (0, 64):
        raise ValueError("DSA NVFP4 requires latent/RoPE dimensions (512, 0 or 64)")
    return kv_lora_rank // 2 + kv_lora_rank // 16 + qk_rope_head_dim * 2


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
    is_glm5_next = is_glm5_next_nvfp4(hf_config)
    expected_geometry = (512, 0) if is_glm5_next else (512, 64)
    if geometry != expected_geometry:
        raise ValueError(
            f"DSA NVFP4 requires latent/RoPE dimensions {expected_geometry}, got {geometry}"
        )
    if is_glm5_next:
        if (
            getattr(hf_config, "index_kpool", 1) != 4
            or not getattr(hf_config, "index_kpool_compress", False)
            or not getattr(hf_config, "index_kpool_always_select_tail", False)
        ):
            raise ValueError(
                "GLM5Next NVFP4 requires index_kpool=4, compression and tail selection"
            )
    elif getattr(hf_config, "index_kpool", 1) != 1:
        raise ValueError("DSA NVFP4 does not support pooled indexers (index_kpool > 1)")
    if getattr(hf_config, "learnable_sink", False):
        raise ValueError("DSA NVFP4 does not support attention sinks")
    if getattr(hf_config, "index_topk", 2048) != 2048:
        raise ValueError("DSA NVFP4 currently requires index_topk=2048")
    backends = ("fa3", "fa3") if is_glm5_next else ("flashmla_sparse", "flashmla_kv")
    for name, expected in zip(("dsa_prefill_backend", "dsa_decode_backend"), backends):
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
    # KPool pages contain 64 pooled index entries and 64 * kpool main tokens.
    page_size = 256 if is_glm5_next else 64
    if getattr(cfg, "page_size", None) not in (None, page_size):
        raise ValueError(f"DSA NVFP4 requires page_size={page_size}")
