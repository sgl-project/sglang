"""Ascend execution defaults for Unlimited-OCR reference sliding windows."""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import _register_for, resolving_view
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


@_register_for("UnlimitedOCRForCausalLM")
def _unlimited_ocr_overrides(server_args: Any, hf_config: Any) -> dict:
    if not get_platform().is_npu:
        return {}

    cfg = resolving_view(server_args)
    if cfg.speculative_algorithm is not None:
        raise ValueError(
            "Unlimited-OCR on Ascend does not support speculative decoding."
        )

    # Ordinary SWA eviction would discard the globally visible prompt. Keep
    # its KV in the full pool; attention still limits the generated-token window.
    # This also avoids the mixed SWA allocator's zero-sized full pool for this
    # all-SWA architecture on Ascend.
    graph_config = cfg.cuda_graph_config
    for phase in Phase.ALL:
        graph_config = with_phase(graph_config, phase, backend=Backend.DISABLED)
    logger.warning(
        "Unlimited-OCR on Ascend uses full KV storage and eager reference "
        "sliding-window attention; hybrid SWA eviction and graph capture are disabled."
    )
    return {
        "disable_hybrid_swa_memory": True,
        "disable_radix_cache": True,
        "cuda_graph_config": graph_config,
    }
