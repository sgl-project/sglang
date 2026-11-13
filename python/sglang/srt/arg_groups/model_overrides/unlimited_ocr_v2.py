"""Required Unlimited-OCR v2 configuration overrides.

DASH maps each sparse block to one KV page, whose counters radix reuse loses.
"""

import logging
from typing import Any, Dict, Optional

from sglang.srt.arg_groups.model_override_base import _register_for, resolving_view
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.configs.unlimited_ocr_v2 import UocrDocumentConfig
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.runtime_context import attn_dp_enabled_of
from sglang.srt.utils.device import get_device_memory_capacity

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
    if cfg.enable_unified_memory:
        raise ValueError(
            "Unlimited-OCR v2 does not support --enable-unified-memory: the "
            "unified pool serves hybrid-Mamba and hybrid sliding-window models "
            "only."
        )
    if cfg.uocr_doc_split and (cfg.dp_size > 1 or attn_dp_enabled_of(cfg)):
        raise ValueError(
            "Unlimited-OCR v2 document splitting needs a single DP rank "
            f"(got --dp-size={cfg.dp_size}, --attn-dp-size={cfg.attn_dp_size}): "
            "the windows of one document would be routed to different ranks, "
            "and a window parked for its predecessor on another rank is never "
            "released. Omit --uocr-doc-split to serve with data parallelism."
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


# Two windows from 10 GiB, covering 16 GB Macs below the measured M5 (an M1 reports
# 10,922 MiB); each 2 GiB above the M5, about twice a window's KV, adds one.
_MPS_TWO_WINDOW_MIN_MIB = 10 * 1024
_MPS_MEASURED_LIMIT_MIB = 12124
_MPS_MIB_PER_EXTRA_WINDOW = 2 * 1024
_MPS_MAX_DEFAULT_WINDOWS = 8


def _mps_window_budget(gpu_mem_mib: Optional[float]) -> int:
    if gpu_mem_mib is None or gpu_mem_mib < _MPS_TWO_WINDOW_MIN_MIB:
        return 1
    extra = max(0, int(gpu_mem_mib) - _MPS_MEASURED_LIMIT_MIB)
    return min(2 + extra // _MPS_MIB_PER_EXTRA_WINDOW, _MPS_MAX_DEFAULT_WINDOWS)


def mps_window_tokens(
    doc_cfg: UocrDocumentConfig, *, chunked_prefill_size: Optional[int], page_size: int
) -> int:
    """KV tokens of one document window: its context plus one prefill chunk."""
    # The chunk holds the retained tail a follow-up copies; 2048 if chunking is off.
    tokens = doc_cfg.default_context_length() + max(chunked_prefill_size or 0, 2048)
    return (tokens + page_size - 1) // page_size * page_size


def apply_unlimited_ocr_v2_mps_kv_pool_default(server_args: Any) -> None:
    """Default the KV pool of a split Unlimited-OCR v2 server on MPS to whole
    document windows, scaled by the Metal working-set limit."""
    cfg = resolving_view(server_args)
    if cfg.device != "mps" or use_mlx() or cfg.max_total_tokens is not None:
        return
    doc_cfg = UocrDocumentConfig.from_args(cfg)
    if not doc_cfg.uocr_doc_split or doc_cfg.uocr_max_page_tokens == 0:
        return
    try:
        gpu_mem_mib = get_device_memory_capacity("mps")
    except (RuntimeError, ValueError):
        gpu_mem_mib = None
    # handle_gpu_memory_settings has resolved the chunk size by now.
    window = mps_window_tokens(
        doc_cfg,
        chunked_prefill_size=cfg.chunked_prefill_size,
        page_size=int(cfg.page_size or 32),
    )
    declare_resolution(
        server_args,
        "apply_unlimited_ocr_v2_mps_kv_pool_default",
        max_total_tokens=_mps_window_budget(gpu_mem_mib) * window,
    )
