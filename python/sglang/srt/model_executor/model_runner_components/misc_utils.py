from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
from sglang.srt.distributed import get_world_group
from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var, is_hip

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()


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


def get_healthy_expert_location_src_rank(
    *, invoked_in_elastic_ep_rejoin_path: bool
) -> int:
    world_group = get_world_group()
    # NOTE: do not key off `self.server_args.elastic_ep_rejoin` here.
    # A rank that was started as a rejoin rank may later act as a healthy
    # rank in a subsequent recovery cycle.
    local_rejoin_flag = bool(invoked_in_elastic_ep_rejoin_path)
    gathered_rejoin_flags = world_group.all_gather_object(local_rejoin_flag)

    for rank_in_group, is_rejoin_rank in enumerate(gathered_rejoin_flags):
        if not is_rejoin_rank:
            return world_group.ranks[rank_in_group]

    raise RuntimeError(
        "No healthy rank found for broadcasting expert location metadata. "
        "All ranks are marked as elastic_ep_rejoin."
    )


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
