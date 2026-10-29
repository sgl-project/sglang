"""Required Unlimited-OCR v2 configuration overrides.

DASH maps each sparse block to one KV page, whose counters radix reuse loses.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import _register_for, resolving_view
from sglang.srt.runtime_context import attn_dp_enabled_of

logger = logging.getLogger(__name__)


def _reject_unsupported_serving_modes(cfg: Any) -> None:
    """Raise for the serving modes the Unlimited-OCR v2 path does not implement."""
    if cfg.speculative_algorithm is not None:
        raise ValueError(
            "Unlimited-OCR v2 does not support speculative decoding "
            f"(--speculative-algorithm={cfg.speculative_algorithm})."
        )
    if cfg.pp_size > 1:
        raise ValueError(
            "Unlimited-OCR v2 does not support pipeline parallelism "
            f"(--pp-size={cfg.pp_size}); use --pp-size 1."
        )
    if cfg.disaggregation_mode != "null":
        raise ValueError(
            "Unlimited-OCR v2 does not support PD disaggregation "
            f"(--disaggregation-mode={cfg.disaggregation_mode})."
        )
    if not cfg.disable_uocr_doc_split and (cfg.dp_size > 1 or attn_dp_enabled_of(cfg)):
        raise ValueError(
            "Unlimited-OCR v2 document splitting needs a single DP rank "
            f"(got --dp-size={cfg.dp_size}, --attn-dp-size={cfg.attn_dp_size}): "
            "the windows of one document would be routed to different ranks, "
            "and a window parked for its predecessor on another rank is never "
            "released. Pass --disable-uocr-doc-split to serve with data parallelism."
        )


@_register_for("UnlimitedOCRV2ForCausalLM")
def _unlimited_ocr_v2_overrides(server_args: Any, hf_config: Any) -> dict:
    # Deferred to avoid a module import cycle.
    from sglang.srt.configs.model_config import get_unlimited_ocr_v2_sparse_config

    cfg = resolving_view(server_args)
    _reject_unsupported_serving_modes(cfg)
    sparse_cfg = get_unlimited_ocr_v2_sparse_config(hf_config)
    block_size = int(sparse_cfg["sparse_block_size"])
    overrides: Dict[str, Any] = {}

    if cfg.page_size != block_size:
        if cfg.page_size is not None:
            logger.warning(
                "Unlimited-OCR v2 requires page_size == sparse_block_size (%d); "
                "changing page_size from %s.",
                block_size,
                cfg.page_size,
            )
        overrides["page_size"] = block_size

    for field in (
        "attention_backend",
        "prefill_attention_backend",
        "decode_attention_backend",
    ):
        current = getattr(cfg, field)
        if current not in (None, "unlimited_ocr_v2"):
            raise ValueError(
                f"--{field.replace('_', '-')}={current!r} cannot serve "
                "Unlimited-OCR v2: only the 'unlimited_ocr_v2' backend implements "
                "the DASH entmax router and the bias-aware sparse attend."
            )
    if cfg.attention_backend is None:
        overrides["attention_backend"] = "unlimited_ocr_v2"

    if not cfg.disable_radix_cache:
        logger.warning(
            "Unlimited-OCR v2: disabling the radix cache. A request resuming "
            "from a cached prefix never runs its first chunk, so its <PAGE> / "
            "image-token counters would start at zero and both the page "
            "isolation and the page2d coordinates would diverge from training."
        )
        overrides["disable_radix_cache"] = True

    # CUDA graph defaults run earlier because they are not model-overridable.
    return overrides
