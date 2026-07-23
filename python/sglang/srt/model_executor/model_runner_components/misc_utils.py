from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
from sglang.srt.server_args import CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def maybe_disable_chunked_prefix_cache(
    *, server_args: ServerArgs, use_mla_backend: bool, is_draft_worker: bool
) -> None:
    # Chunked prefix caching requires an MLA model on a backend whose
    # kernels read that layout. This is a load-time gate, not a
    # resolution-time one: out-of-tree platforms register their supported
    # backends in init_backend(), which runs when this module is imported
    # — after ServerArgs.__post_init__. Target runner only: a draft
    # model's (often non-MLA) config must not flip the shared setting.
    if is_draft_worker:
        return
    if (
        not use_mla_backend
        or server_args.attention_backend
        not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
    ):
        if not server_args.disable_chunked_prefix_cache:
            server_args.override(
                "model_runner.chunked_prefix_cache_gate",
                disable_chunked_prefix_cache=True,
            )
    if not server_args.disable_chunked_prefix_cache:
        logger.info("Chunked prefix cache is turned on.")


def create_msprobe_debugger(server_args: ServerArgs) -> Optional[Any]:
    if server_args.msprobe_dump_config is None:
        return None

    try:
        from msprobe.pytorch import PrecisionDebugger, seed_all
    except ImportError:
        logger.warning(
            "Please install msprobe for tensor data dump: pip install mindstudio-probe --pre, "
            "see https://gitcode.com/Ascend/msprobe for details."
        )
        return None

    seed_all(mode=True)
    return PrecisionDebugger(config_path=server_args.msprobe_dump_config)


def resolve_pp_proxy_topk_size(
    *, model_config: ModelConfig, pp_size: int, pp_rank: int, start_layer: int
) -> Optional[int]:
    hf_config = model_config.hf_text_config
    if (
        pp_size <= 1
        or pp_rank == 0
        or not is_deepseek_dsa(hf_config)
        or not dsa_layer_skips_topk(hf_config, start_layer)
    ):
        return None
    return getattr(hf_config, "index_topk", None)
