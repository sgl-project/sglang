from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var, is_hip

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()


def check_quantized_moe_compatibility(
    *,
    model_config: ModelConfig,
    tp_size: int,
    moe_ep_size: int,
    moe_dp_size: int,
) -> None:
    if (
        quantization_config := getattr(
            model_config.hf_config, "quantization_config", None
        )
    ) is not None and (
        weight_block_size := quantization_config.get("weight_block_size", None)
    ) is not None:
        weight_block_size_n = weight_block_size[0]

        if tp_size % moe_ep_size != 0:
            raise ValueError(
                f"tp_size {tp_size} must be divisible by ep_size {moe_ep_size}"
            )
        moe_tp_size = tp_size // moe_ep_size // moe_dp_size

        moe_intermediate_size = getattr(
            model_config.hf_text_config, "moe_intermediate_size", None
        )
        if moe_intermediate_size is None:
            return

        if moe_intermediate_size % moe_tp_size != 0:
            raise ValueError(
                f"moe_intermediate_size {moe_intermediate_size} must be divisible by moe_tp_size ({moe_tp_size}) which is tp_size ({tp_size}) divided by moe_ep_size ({moe_ep_size})."
            )

        if (
            not envs.SGLANG_SHARED_EXPERT_TP1.get()
            and (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0
            and not _use_aiter
        ):
            raise ValueError(
                f"For quantized MoE models, please make sure ({moe_intermediate_size=} / {moe_tp_size=}) % {weight_block_size_n=} == 0 "
                f"where moe_tp_size is equal to tp_size ({tp_size}) divided by ep_size ({moe_ep_size}). "
                f"You can fix this by setting arguments `--tp` and `--ep` correctly."
            )
