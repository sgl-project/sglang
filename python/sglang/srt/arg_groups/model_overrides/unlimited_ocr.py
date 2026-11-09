"""Ascend execution defaults for Unlimited-OCR reference sliding windows."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import _register_for, resolving_view
from sglang.srt.runtime_context import get_platform


@_register_for("UnlimitedOCRForCausalLM")
def _unlimited_ocr_overrides(server_args: Any, hf_config: Any) -> dict:
    if not get_platform().is_npu:
        return {}

    cfg = resolving_view(server_args)
    if cfg.speculative_algorithm is not None:
        raise ValueError(
            "Unlimited-OCR on Ascend does not support speculative decoding."
        )

    # Ascend's pure-SWA allocator retains references and reclaims generated
    # pages through the existing scheduler's protected eviction frontier.
    return {}
