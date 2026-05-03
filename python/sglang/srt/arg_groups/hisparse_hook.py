import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def apply_hisparse_nsa_backend_defaults(
    server_args: "ServerArgs",
    user_set_prefill: bool,
    user_set_decode: bool,
) -> bool:
    """Force flashmla_sparse as NSA backend when --enable-hisparse is set.

    Returns True if hisparse handled backend selection (caller should skip its
    own default logic), False otherwise.
    """
    # HiSparse requires flashmla_sparse for both prefill and decode
    if server_args.enable_hisparse:
        if not user_set_prefill:
            server_args.nsa_prefill_backend = "flashmla_sparse"
        if not user_set_decode:
            server_args.nsa_decode_backend = "flashmla_sparse"
        logger.warning(
            f"HiSparse enabled: using flashmla_sparse NSA backends "
            f"(prefill={server_args.nsa_prefill_backend}, decode={server_args.nsa_decode_backend})."
        )
        return True
    return False


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

    if not is_v4_hisparse:
        for attr, label in [
            ("nsa_prefill_backend", "prefill"),
            ("nsa_decode_backend", "decode"),
        ]:
            backend = getattr(server_args, attr)
            if backend is not None and backend != "flashmla_sparse":
                raise ValueError(
                    f"HiSparse requires flashmla_sparse NSA {label} backend, "
                    f"but got --nsa-{label}-backend={backend}. "
                    f"Please use --nsa-{label}-backend=flashmla_sparse or omit it."
                )
