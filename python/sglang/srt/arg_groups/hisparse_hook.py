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


# The hisparse DSA backend defaults moved to the resolution pipeline
# (arg_groups/overrides.py: _dsa_split_backend_resolution, hisparse arm).


def validate_hisparse_dsa_backend(
    server_args: ServerArgs, attr: str, label: str
) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    # Invoked after the DSA kv-cache-dtype / split-backend declarations:
    # read the resolving state through the view.
    view = resolved_view(server_args)
    backend = getattr(view, attr)
    kv_cache_dtype = view.kv_cache_dtype
    allowed_backends = _hisparse_allowed_backends(kv_cache_dtype)
    if backend is not None and backend not in allowed_backends:
        raise ValueError(
            f"HiSparse supports DSA {label} backend(s) {sorted(allowed_backends)} "
            f"on this platform with --kv-cache-dtype={kv_cache_dtype}, "
            f"but got --dsa-{label}-backend={backend}. "
            f"Please use --dsa-{label}-backend="
            f"{_hisparse_default_backend(kv_cache_dtype)} "
            "or omit it."
        )


def validate_hisparse_kv_cache_dtype(server_args: ServerArgs) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    kv_cache_dtype = resolved_view(server_args).kv_cache_dtype
    if kv_cache_dtype in HISPARSE_KV_CACHE_DTYPES:
        return

    choices = " or ".join(
        f"--kv-cache-dtype={dtype}" for dtype in HISPARSE_KV_CACHE_DTYPES
    )
    raise ValueError(
        f"HiSparse requires one of {HISPARSE_KV_CACHE_DTYPES} KV cache dtypes, "
        f"but got --kv-cache-dtype={kv_cache_dtype}. Please use {choices}."
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
    if is_hip and is_v4_hisparse:
        # DSv4 HiSparse manages its own dtype/backend pairing for both the
        # separate packed-KV and the unified-KV layouts on ROCm (the unified-KV
        # path now wires its own HiSparse C4 device/host pool), so skip the
        # DSA-only dtype checks below.
        return

    from sglang.srt.arg_groups.overrides import resolved_view

    if resolved_view(server_args).kv_cache_dtype not in (
        "bfloat16",
        "auto",
        "fp8_e4m3",
    ):
        validate_hisparse_kv_cache_dtype(server_args)

    for attr, label in [
        ("dsa_prefill_backend", "prefill"),
        ("dsa_decode_backend", "decode"),
    ]:
        validate_hisparse_dsa_backend(server_args, attr, label)
