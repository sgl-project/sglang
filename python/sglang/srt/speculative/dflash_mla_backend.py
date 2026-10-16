"""Backend selection and startup checks for compressed DFlash MLA drafts."""

from __future__ import annotations

from typing import Optional, Sequence

DFLASH_MLA_ATTENTION_BACKENDS = frozenset({"triton", "flashinfer", "tokenspeed_mla"})


def dflash_mla_prefill_causal(
    *, layer_types: Optional[Sequence[str]], is_causal: Optional[bool]
) -> bool:
    # Match DFlash's legacy ENCODER_ONLY default when layer_types is absent.
    # FlashInfer supports only full_attention layers, validated at startup.
    return layer_types is not None and bool(is_causal)


def resolve_dflash_mla_attention_backend(backend: Optional[str]) -> str:
    # Target and draft have independent attention shapes and masking requirements.
    # In particular, a target's decode backend is not a safe draft default.
    backend = backend or "triton"
    if backend not in DFLASH_MLA_ATTENTION_BACKENDS:
        raise ValueError(
            "DFLASH MLA drafts support --speculative-draft-attention-backend "
            f"in {sorted(DFLASH_MLA_ATTENTION_BACKENDS)}, got {backend!r}."
        )
    return backend


def validate_dflash_mla_backend_config(
    *,
    backend: str,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_local_heads: int,
    model_dtype: str,
    kv_cache_dtype: str,
    kv_cache_dtype_str: str,
    page_size: int,
    device: str,
    is_hip: bool,
    is_blackwell: bool,
    device_capability: tuple[int, int],
    has_sliding_attention: bool,
    dcp_size: int = 1,
    tp_size: int = 1,
    attn_tp_size: int = 1,
) -> None:
    """Validate the actual draft runtime, independently of target settings."""
    resolve_dflash_mla_attention_backend(backend)
    if device.split(":", 1)[0] != "cuda":
        raise ValueError("DFLASH MLA draft attention requires CUDA or HIP.")
    if dcp_size != 1:
        raise ValueError("DFLASH MLA drafts do not support decode context parallelism.")
    if tp_size != attn_tp_size:
        raise ValueError(
            "DFLASH MLA drafts require matching tensor-parallel and attention "
            "tensor-parallel sizes; DP attention is not supported."
        )
    if model_dtype not in {"torch.bfloat16", "torch.float16"}:
        raise ValueError("DFLASH MLA draft model dtype must be float16 or bfloat16.")
    if kv_cache_dtype_str in {"mxfp8", "nvfp4", "fp4_mx_block16"}:
        raise ValueError(
            "DFLASH MLA does not support block-scaled draft KV dtype "
            f"{kv_cache_dtype_str!r}; use bf16 or fp8_e4m3."
        )
    if kv_cache_dtype not in {
        "torch.bfloat16",
        "torch.float16",
        "torch.float8_e4m3fn",
        "torch.float8_e4m3fnuz",
        "torch.float8_e5m2",
        "torch.float8_e5m2fnuz",
    }:
        raise ValueError(
            f"DFLASH MLA does not support draft KV dtype {kv_cache_dtype!r}; "
            "use bf16 or fp8_e4m3."
        )
    if num_local_heads < 1:
        raise ValueError("DFLASH MLA requires at least one local query head.")
    if page_size < 1:
        raise ValueError("DFLASH MLA requires a positive page_size.")
    if not 16 <= kv_lora_rank <= 512:
        raise ValueError("DFLASH MLA supports kv_lora_rank between 16 and 512.")
    if not 2 <= qk_rope_head_dim <= 64 or qk_rope_head_dim % 2:
        raise ValueError(
            "DFLASH MLA supports an even qk_rope_head_dim between 2 and 64."
        )
    if backend == "triton":
        return
    if is_hip:
        raise ValueError(f"DFLASH MLA backend {backend!r} requires NVIDIA CUDA.")
    if device_capability[0] < 8:
        raise ValueError(
            f"DFLASH MLA backend {backend!r} requires an SM80 or newer GPU."
        )
    if has_sliding_attention:
        raise ValueError(
            f"DFLASH MLA backend {backend!r} does not support sliding attention; "
            "select --speculative-draft-attention-backend triton."
        )
    if (kv_lora_rank, qk_rope_head_dim) != (512, 64):
        raise ValueError(
            f"DFLASH MLA backend {backend!r} requires kv_lora_rank=512 and "
            "qk_rope_head_dim=64; select triton for other dimensions."
        )
    if backend == "flashinfer" and kv_cache_dtype != model_dtype:
        # The existing FlashInfer adapter casts the entire compressed pool to
        # the query dtype on every layer. Avoid that hidden allocation here;
        # native FP8 MLA also has architecture-specific support and scale APIs.
        raise ValueError(
            "DFLASH flashinfer requires draft KV dtype to match the float16 or "
            "bfloat16 model dtype. Set --speculative-draft-kv-cache-dtype bf16 "
            "for a bfloat16 draft, or auto for a float16 draft."
        )
    if backend == "tokenspeed_mla":
        if not is_blackwell or device_capability not in {(10, 0), (10, 3)}:
            raise ValueError(
                "DFLASH tokenspeed_mla requires a Blackwell SM100/SM103 GPU."
            )
        if model_dtype != "torch.bfloat16":
            # TokenSpeed's FP8 decode kernel returns BF16, including when its
            # input queries originated from a different model dtype.
            raise ValueError("DFLASH tokenspeed_mla requires a bfloat16 draft model.")
        if kv_cache_dtype != "torch.float8_e4m3fn":
            raise ValueError(
                "DFLASH tokenspeed_mla requires "
                "--speculative-draft-kv-cache-dtype fp8_e4m3."
            )
        if page_size not in (32, 64):
            raise ValueError("DFLASH tokenspeed_mla requires --page-size 32 or 64.")
        if not 1 <= num_local_heads <= 128:
            raise ValueError("DFLASH tokenspeed_mla requires 1 to 128 local heads.")


def validate_dflash_mla_runner(model_runner, backend: str) -> None:
    config = model_runner.model_config
    if not getattr(config, "is_dflash_mla_draft", False):
        return
    from sglang.srt.runtime_context import get_parallel, get_platform
    from sglang.srt.speculative.dflash_utils import get_dflash_layer_types
    from sglang.srt.utils import is_flashinfer_available, is_tokenspeed_mla_available

    parallel = get_parallel()
    platform = get_platform()
    validate_dflash_mla_backend_config(
        backend=backend,
        kv_lora_rank=config.kv_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
        num_local_heads=config.num_attention_heads // parallel.tp_size,
        model_dtype=str(model_runner.dtype),
        kv_cache_dtype=str(model_runner.kv_cache_dtype),
        kv_cache_dtype_str=model_runner.kv_cache_dtype_str,
        page_size=model_runner.page_size,
        device=str(model_runner.device),
        is_hip=bool(platform.is_hip),
        is_blackwell=bool(platform.is_blackwell),
        device_capability=platform.device_capability,
        has_sliding_attention="sliding_attention"
        in (get_dflash_layer_types(config.hf_config) or ()),
        dcp_size=parallel.attn_dcp_size,
        tp_size=parallel.tp_size,
        attn_tp_size=parallel.attn_tp_size,
    )
    if backend in {"flashinfer", "tokenspeed_mla"} and not is_flashinfer_available():
        raise ValueError(f"DFLASH MLA backend {backend!r} requires FlashInfer.")
    if backend == "tokenspeed_mla" and not is_tokenspeed_mla_available():
        raise ValueError("DFLASH tokenspeed_mla requires the tokenspeed_mla package.")
