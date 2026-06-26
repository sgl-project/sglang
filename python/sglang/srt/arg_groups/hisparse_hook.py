from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv"},
}
HISPARSE_ROCM_DSA_BACKENDS = {"tilelang", "aiter"}
HISPARSE_KV_CACHE_DTYPES = ("bfloat16", "fp8_e4m3")


def _is_hip() -> bool:
    from sglang.srt.server_args import is_hip

    return is_hip()


def _hisparse_default_backend(kv_cache_dtype: str) -> str:
    if _is_hip():
        return "tilelang"
    return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"


def _hisparse_allowed_backends(kv_cache_dtype: str) -> set[str]:
    if _is_hip():
        return HISPARSE_ROCM_DSA_BACKENDS
    return HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE.get(
        kv_cache_dtype, {"flashmla_sparse", "flashmla_kv"}
    )


def apply_hisparse_dsa_backend_defaults(
    server_args: ServerArgs,
    user_set_prefill: bool,
    user_set_decode: bool,
    kv_cache_dtype: str,
) -> bool:
    """Pick DSA backends for --enable-hisparse based on KV dtype.

    CUDA uses dtype-specific FlashMLA backends; ROCm uses TileLang. Returns
    True if hisparse handled backend selection.
    """
    if not server_args.enable_hisparse:
        return False

    backend = _hisparse_default_backend(kv_cache_dtype)
    if not user_set_prefill:
        server_args.dsa_prefill_backend = backend
    if not user_set_decode:
        server_args.dsa_decode_backend = backend
    logger.warning(
        f"HiSparse enabled ({kv_cache_dtype}): using DSA backends "
        f"prefill={server_args.dsa_prefill_backend}, decode={server_args.dsa_decode_backend}."
    )
    return True


def validate_hisparse_dsa_backend(
    server_args: ServerArgs, attr: str, label: str
) -> None:
    backend = getattr(server_args, attr)
    allowed_backends = _hisparse_allowed_backends(server_args.kv_cache_dtype)
    if backend is not None and backend not in allowed_backends:
        raise ValueError(
            f"HiSparse supports DSA {label} backend(s) {sorted(allowed_backends)} "
            f"on this platform with --kv-cache-dtype={server_args.kv_cache_dtype}, "
            f"but got --dsa-{label}-backend={backend}. "
            f"Please use --dsa-{label}-backend="
            f"{_hisparse_default_backend(server_args.kv_cache_dtype)} "
            "or omit it."
        )


def validate_hisparse_kv_cache_dtype(server_args: ServerArgs) -> None:
    if server_args.kv_cache_dtype in HISPARSE_KV_CACHE_DTYPES:
        return

    choices = " or ".join(
        f"--kv-cache-dtype={dtype}" for dtype in HISPARSE_KV_CACHE_DTYPES
    )
    raise ValueError(
        f"HiSparse requires one of {HISPARSE_KV_CACHE_DTYPES} KV cache dtypes, "
        f"but got --kv-cache-dtype={server_args.kv_cache_dtype}. Please use {choices}."
    )


def _validate_dsv4_hisparse_online_c128_mtp(server_args: ServerArgs) -> None:
    """Validate the DSV4 HiSparse + online C128 MTP combination early."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
        return

    if server_args.speculative_algorithm is None:
        return

    if not envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
        raise ValueError(
            "DSV4 HiSparse online C128 with speculative decode requires "
            "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1."
        )

    if server_args.speculative_algorithm != "EAGLE":
        raise ValueError(
            "DSV4 HiSparse online C128 MTP is currently only validated for "
            f"EAGLE, got {server_args.speculative_algorithm!r}."
        )

    if server_args.speculative_eagle_topk not in (None, 1):
        raise ValueError(
            "DSV4 HiSparse online C128 MTP requires "
            f"--speculative-eagle-topk 1, got {server_args.speculative_eagle_topk}."
        )

    speculative_num_steps = int(server_args.speculative_num_steps or 0)
    if speculative_num_steps > 2:
        raise ValueError(
            "DSV4 HiSparse online C128 MTP is currently validated only for "
            f"EAGLE step1/step2, got speculative_num_steps={speculative_num_steps}."
        )

    if not envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
        raise ValueError(
            "DSV4 HiSparse online C128 MTP requires "
            "SGLANG_OPT_USE_COMPRESSOR_V2=1 because the v2 compressor carries "
            "online state-slot metadata."
        )

    logger.warning(
        "DSV4 HiSparse online C128 MTP enabled: C4 stays on the HiSparse "
        "host mirror; C128 uses online EAGLE state banks; eagle_steps=%d, "
        "draft_tokens=%s.",
        speculative_num_steps,
        server_args.speculative_num_draft_tokens,
    )


def validate_hisparse(server_args: ServerArgs) -> None:
    """Validate --enable-hisparse constraints (model class, radix cache, DSA backend)."""
    if not server_args.enable_hisparse:
        return

    from sglang.srt.configs.model_config import (
        is_deepseek_dsa,
        is_deepseek_v4,
    )

    hf_config = server_args.get_model_config().hf_config
    is_v4_hisparse = is_deepseek_v4(hf_config)
    is_hip = _is_hip()
    assert is_deepseek_dsa(hf_config) or is_v4_hisparse, (
        "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) "
        "models (e.g., DeepSeek V3.2, GLM-5) and DeepSeek V4 now. "
    )

    assert (
        server_args.disable_radix_cache
    ), "Hierarchical sparse attention currently requires --disable-radix-cache."

    # DSv4 hisparse handles its own dtype/backend pairing elsewhere; the dtype-
    # aware checks below only apply to the DSA hisparse path.
    if is_v4_hisparse:
        _validate_dsv4_hisparse_online_c128_mtp(server_args)

    if is_hip and is_v4_hisparse:
        # TEMPORARY GUARD: DSv4 HiSparse is not supported on the unified-KV path.
        # In unified-KV mode c4_kv_pool is None, so DeepSeekV4HiSparseTokenToKVPoolAllocator
        # cannot attach and pool init dies with a cryptic AssertionError. Fail fast
        # at startup with a clear message instead. Remove once unified-KV HiSparse lands.
        from sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            raise ValueError(
                "--enable-hisparse is not supported with the unified-KV path on ROCm"
                "(SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton) for DeepSeek-V4: "
                "HiSparse currently requires the separate packed KV layout. "
                "Either set SGLANG_HACK_FLASHMLA_BACKEND=triton, or run without "
                "--enable-hisparse."
            )
        return

    if server_args.kv_cache_dtype not in ("bfloat16", "auto", "fp8_e4m3"):
        validate_hisparse_kv_cache_dtype(server_args)

    for attr, label in [
        ("dsa_prefill_backend", "prefill"),
        ("dsa_decode_backend", "decode"),
    ]:
        validate_hisparse_dsa_backend(server_args, attr, label)
