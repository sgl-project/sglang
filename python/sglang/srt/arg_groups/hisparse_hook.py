import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# Backend/dtype pairing: flashmla_sparse only takes BF16 KV;
# flashmla_kv only supports FP8 (it always reads KV as FP8 via
# is_fp8_kvcache=True, inline-quantizing BF16 would defeat HiSparse).
_HISPARSE_ALLOWED_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv"},
}


def _hisparse_default_backend(kv_cache_dtype: str) -> str:
    return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"


def apply_hisparse_nsa_backend_defaults(
    server_args: "ServerArgs",
    user_set_prefill: bool,
    user_set_decode: bool,
    kv_cache_dtype: str,
) -> bool:
    """Pick NSA backends for --enable-hisparse based on KV dtype.

    BF16 KV -> flashmla_sparse, FP8 KV -> flashmla_kv. Returns True if hisparse
    handled backend selection (caller should skip its own default logic).
    """
    if not server_args.enable_hisparse:
        return False

    backend = _hisparse_default_backend(kv_cache_dtype)
    if not user_set_prefill:
        server_args.nsa_prefill_backend = backend
    if not user_set_decode:
        server_args.nsa_decode_backend = backend
    logger.warning(
        f"HiSparse enabled ({kv_cache_dtype}): using NSA backends "
        f"prefill={server_args.nsa_prefill_backend}, decode={server_args.nsa_decode_backend}."
    )
    return True


def validate_hisparse(server_args: "ServerArgs") -> None:
    """Validate --enable-hisparse constraints (model class, radix cache, NSA backend)."""
    if not server_args.enable_hisparse:
        return

    from sglang.srt.configs.model_config import (
        is_deepseek_nsa,
        is_deepseek_v4,
    )

    hf_config = server_args.get_model_config().hf_config
    is_v4_hisparse = is_deepseek_v4(hf_config)
    assert is_deepseek_nsa(hf_config) or is_v4_hisparse, (
        "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) "
        "models (e.g., DeepSeek V3.2, GLM-5) and DeepSeek V4 now. "
    )

    assert (
        server_args.disable_radix_cache
    ), "Hierarchical sparse attention currently requires --disable-radix-cache."

    # DSv4 hisparse handles its own dtype/backend pairing elsewhere; the dtype-
    # aware checks below only apply to the DSA hisparse path.
    if is_v4_hisparse:
        return

    if server_args.kv_cache_dtype not in ("bfloat16", "auto", "fp8_e4m3"):
        raise ValueError(
            f"HiSparse requires bfloat16 or fp8_e4m3 KV cache, "
            f"but got --kv-cache-dtype={server_args.kv_cache_dtype}. "
            f"Please use --kv-cache-dtype=bfloat16 or fp8_e4m3."
        )

    allowed_backends = _HISPARSE_ALLOWED_BACKENDS_BY_DTYPE.get(
        server_args.kv_cache_dtype, {"flashmla_sparse", "flashmla_kv"}
    )
    for attr, label in [
        ("nsa_prefill_backend", "prefill"),
        ("nsa_decode_backend", "decode"),
    ]:
        backend = getattr(server_args, attr)
        if backend is not None and backend not in allowed_backends:
            raise ValueError(
                f"HiSparse with --kv-cache-dtype={server_args.kv_cache_dtype} requires "
                f"--nsa-{label}-backend in {sorted(allowed_backends)}, "
                f"but got {backend}."
            )
