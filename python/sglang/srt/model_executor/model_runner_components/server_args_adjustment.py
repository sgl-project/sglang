from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def model_specific_adjustment(
    *, server_args: ServerArgs, model_config: ModelConfig
) -> None:
    from sglang.srt.model_executor.model_runner import (
        CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS,
    )

    if model_config.is_multimodal:
        if not model_config.is_multimodal_chunked_prefill_supported:
            server_args.chunked_prefill_size = -1
            logger.info(
                f"Automatically turn off --chunked-prefill-size as it is not supported for "
                f"{model_config.hf_config.model_type}"
            )

    use_mla_backend = model_config.attention_arch == AttentionArch.MLA
    if (
        not use_mla_backend
        or server_args.attention_backend
        not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
    ):
        server_args.disable_chunked_prefix_cache = True

    if not server_args.disable_chunked_prefix_cache:
        log_info_on_rank0(logger, "Chunked prefix cache is turned on.")
