from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


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
