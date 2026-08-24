from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv", "flashinfer_sparse_mla"},
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
        kv_cache_dtype, {"flashmla_sparse", "flashmla_kv", "flashinfer_sparse_mla"}
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
            f"Please use one of {sorted(allowed_backends)}, or omit the option "
            "to let SGLang pick a backend for this platform."
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
    """Validate --enable-hisparse against the backing the cache config resolves to.

    The resolution itself rejects a cache configuration HiSparse cannot back (see
    `mem_cache/sparsity/factory.py`); what is left here is per-backing: which
    model classes, parallelism and feature combinations that backing supports.
    """
    from sglang.srt.mem_cache.sparsity.factory import (
        HiSparseBacking,
        hisparse_backing,
    )

    backing = hisparse_backing(server_args)
    if backing is None:
        return

    hf_config = server_args.get_model_config().hf_config
    if backing is HiSparseBacking.PRIVATE_HOST:
        if _validate_private_host_backing(server_args, hf_config):
            return
    else:
        _validate_hicache_backing(server_args, hf_config)

    _validate_dsa_dtype_and_backends(server_args)


def _validate_private_host_backing(server_args: ServerArgs, hf_config) -> bool:
    """Constraints for the private-host backing. True = skip the DSA pairing checks."""
    from sglang.srt.configs.model_config import is_deepseek_dsa, is_deepseek_v4

    is_v4_hisparse = is_deepseek_v4(hf_config)
    assert is_deepseek_dsa(hf_config) or is_v4_hisparse, (
        "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) "
        "models (e.g., DeepSeek V3.2, GLM-5) and DeepSeek V4 now. "
    )

    # DSv4 hisparse handles its own dtype/backend pairing elsewhere; the dtype-
    # aware checks only apply to the DSA hisparse path.
    if _is_hip() and is_v4_hisparse:
        # TEMPORARY GUARD: DSv4 HiSparse is not supported on the unified-KV path.
        # In unified-KV mode c4_kv_pool is None, so DeepSeekV4HiSparseTokenToKVPoolAllocator
        # cannot attach and pool init dies with a cryptic AssertionError. Fail fast
        # at startup with a clear message instead. Remove once unified-KV HiSparse lands.
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
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
        return True

    return False


def _validate_hicache_backing(server_args: ServerArgs, hf_config) -> None:
    """Constraints for the HiCache backing.

    Everything rejected here is a path that reaches HiSparse's decode without
    going through its admission, its swap-in, or its eviction bookkeeping -- so
    it would read masked or reused KV rather than fail.
    """
    from sglang.srt.configs.model_config import is_deepseek_dsa

    if not is_deepseek_dsa(hf_config):
        raise ValueError(
            "--enable-hisparse on the HiCache backing only supports DSA models "
            "(e.g. DeepSeek V3.2, GLM-5). DeepSeek V4 is private-host only: add "
            "--disable-radix-cache."
        )

    if server_args.speculative_algorithm is not None:
        raise ValueError(
            "--enable-hisparse on the HiCache backing does not support "
            f"speculative decoding (--speculative-algorithm={server_args.speculative_algorithm}) "
            "yet: the draft worker shares req_to_token with the target, so it "
            "would read the eviction sentinel or a reused slot for any prefix "
            "the tree evicted. Use --disable-radix-cache for the private-host "
            "backing, or drop speculative decoding."
        )

    if server_args.enable_mixed_chunk:
        raise ValueError(
            "--enable-hisparse on the HiCache backing does not support "
            "--enable-mixed-chunk (mixed batches route decode through "
            "forward_extend, which lacks the dual-source swap-in)."
        )

    if server_args.enable_streaming_session or server_args.enable_session_radix_cache:
        raise ValueError(
            "--enable-hisparse on the HiCache backing does not support streaming "
            "sessions (session lock ownership conflicts with releasing the tree "
            "lock at admission)."
        )

    if server_args.disaggregation_mode != "null":
        raise ValueError(
            "--enable-hisparse on the HiCache backing does not support PD "
            "disaggregation (the decode side bypasses admission). Use "
            "--disable-radix-cache for the private-host backing, which has a "
            "direct-to-host admission path."
        )

    if server_args.pp_size > 1:
        raise ValueError(
            "--enable-hisparse on the HiCache backing does not support pipeline "
            "parallelism (pp_size > 1) yet."
        )

    if server_args.hicache_mem_layout not in ("page_first", "layer_first"):
        raise ValueError(
            "--enable-hisparse requires --hicache-mem-layout page_first or "
            f"layer_first (got '{server_args.hicache_mem_layout}'; "
            "page_first_direct breaks the swap-in host addressing)."
        )


def _validate_dsa_dtype_and_backends(server_args: ServerArgs) -> None:
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
