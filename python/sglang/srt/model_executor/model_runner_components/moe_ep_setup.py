from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def prepare_moe_topk(
    *,
    model,
    model_config: ModelConfig,
    server_args: ServerArgs,
    moe_ep_size: int,
    moe_ep_rank: int,
) -> None:
    balancer_cls = None
    num_prepared = 0
    num_routed_experts = None
    for module in model.modules():
        if not isinstance(module, (TopK, HashTopK)):
            continue
        if not module.enable_waterfill or module.waterfill_balancer is not None:
            continue
        if num_routed_experts is None:
            num_routed_experts = getattr(
                model_config.hf_config, "n_routed_experts", None
            )
            if num_routed_experts is None:
                raise ValueError("Waterfill requires model config n_routed_experts.")
        if balancer_cls is None:
            from sglang.srt.layers.moe.waterfill import WaterfillBalancer

            balancer_cls = WaterfillBalancer
        # Static EPLB remaps TopK ids to physical expert ids before Waterfill.
        # Redundant experts therefore need to be included in the per-rank
        # expert count used for Waterfill's shared-expert slot remapping.
        num_physical_routed_experts = (
            num_routed_experts + server_args.ep_num_redundant_experts
        )
        if isinstance(module, TopK):
            routed_scaling_factor = module.topk_config.routed_scaling_factor
        else:
            routed_scaling_factor = module.routed_scaling_factor
        module.waterfill_balancer = balancer_cls(
            num_routed_experts=num_physical_routed_experts,
            world_size=moe_ep_size,
            rank=moe_ep_rank,
            layer_id=module.layer_id,
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
        )
        num_prepared += 1
    if num_prepared:
        log_info_on_rank0(logger, f"Prepared {num_prepared} Waterfill TopK modules.")
