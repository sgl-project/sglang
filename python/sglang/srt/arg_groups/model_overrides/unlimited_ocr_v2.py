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


# Document windows the default KV pool holds, by the Metal working-set limit
# (torch.mps.recommended_max_memory): one up to 14 GiB, which covers a 16 GB Mac
# (12 GiB limit, ~6 GB of it BF16 weights plus the vision workspace), and one
# more for each further 8 GiB, up to eight.
_MPS_BASE_WORKING_SET_MIB = 14 * 1024
_MPS_MIB_PER_EXTRA_WINDOW = 8 * 1024
_MPS_MAX_DEFAULT_WINDOWS = 8


def _mps_window_budget(gpu_mem_mib: Optional[float]) -> int:
    if gpu_mem_mib is None or gpu_mem_mib <= _MPS_BASE_WORKING_SET_MIB:
        return 1
    extra = int(gpu_mem_mib - _MPS_BASE_WORKING_SET_MIB) // _MPS_MIB_PER_EXTRA_WINDOW
    return min(1 + extra, _MPS_MAX_DEFAULT_WINDOWS)


def apply_unlimited_ocr_v2_mps_defaults(server_args: Any) -> None:
    """Default MPS warmup and memory-scaled window KV cap for Unlimited-OCR v2."""
    cfg = resolving_view(server_args)
    if cfg.device != "mps" or use_mlx():
        return
    updates: Dict[str, Any] = {}
    if not cfg.skip_server_warmup and cfg.warmups is None:
        updates["warmups"] = "unlimited_ocr_v2"
    if cfg.max_total_tokens is None:
        doc_cfg = UocrDocumentConfig.from_args(cfg)
        if doc_cfg.uocr_doc_split and doc_cfg.uocr_max_page_tokens > 0:
            page_size = int(cfg.page_size or 32)
            try:
                gpu_mem_mib = get_device_memory_capacity("mps")
            except (RuntimeError, ValueError):
                gpu_mem_mib = None
            windows = _mps_window_budget(gpu_mem_mib)
            window_tokens = doc_cfg.default_context_length()
            # Plus one prefill chunk for the retained tail a follow-up copies.
            # handle_gpu_memory_settings has resolved the chunk size by now;
            # 2048, its smallest default, also covers chunking turned off (-1).
            chunk = max(cfg.chunked_prefill_size or 0, 2048)
            per_window = (
                (window_tokens + chunk + page_size - 1) // page_size * page_size
            )
            updates["max_total_tokens"] = windows * per_window
    if updates:
        declare_resolution(
            server_args,
            "apply_unlimited_ocr_v2_mps_defaults",
            **updates,
        )
