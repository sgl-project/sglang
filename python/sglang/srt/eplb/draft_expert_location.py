"""The DSpark draft's expert placement family: its MoE stages get their own
placement and recorder, rebalanced in the target's EPLB pause."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionRecorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    DRAFT_EXPERT_LOCATION_FAMILY,
    ExpertLocationFamily,
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
    set_global_expert_location_metadata,
)
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

_DSPARK_V4_ARCHITECTURE = "DeepseekV4ForCausalLMDSpark"


def draft_eplb_refusal(*, draft_architecture: str) -> Optional[str]:
    """Why the draft's MoE stages keep the trivial placement; None when they can
    follow the draft family."""
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    moe = get_exec().moe
    if not moe.enable_eplb:
        return "--enable-eplb is off"
    if draft_architecture != _DSPARK_V4_ARCHITECTURE:
        return f"draft {draft_architecture} is not a DeepSeek-V4 DSpark MoE"
    if is_hip() or not get_moe_a2a_backend().is_megamoe():
        return "the draft does not run DeepGEMM MegaMoE"
    if moe.ep_dispatch_algorithm not in ("static", "dynamic"):
        return f"ep_dispatch_algorithm={moe.ep_dispatch_algorithm}"
    if moe.elastic_ep_backend is not None or (
        get_parallel().elastic_ep_initial_size is not None
    ):
        return "elastic EP"
    if moe.expert_distribution_recorder_mode != "stat":
        return (
            f"expert_distribution_recorder_mode={moe.expert_distribution_recorder_mode}"
        )
    return None


def init_draft_expert_location(
    *, model_config: ModelConfig, moe_ep_rank: int
) -> Optional[ExpertLocationFamily]:
    """Publish the draft family's trivial placement and recorder before the draft
    is built; returns the draft runner's family, None when it stays opted out."""
    if not envs.SGLANG_DSPARK_ENABLE_DRAFT_EPLB.get():
        return None
    reason = draft_eplb_refusal(
        draft_architecture=model_config.hf_config.architectures[0]
    )
    if reason is not None:
        logger.warning(f"Draft EPLB stays off: {reason}.")
        return None
    _check_draft_slots(model_config)
    metadata = ExpertLocationMetadata.init_trivial(model_config, moe_ep_rank)
    set_global_expert_location_metadata(metadata, family=DRAFT_EXPERT_LOCATION_FAMILY)
    set_global_expert_distribution_recorder(
        ExpertDistributionRecorder.init_new(
            metadata, rank=moe_ep_rank, report_balancedness=False
        ),
        family=DRAFT_EXPERT_LOCATION_FAMILY,
    )
    logger.info(
        f"Draft EPLB on: {metadata.num_layers} stages, "
        f"{metadata.num_logical_experts} experts in "
        f"{metadata.num_physical_experts} slots over EP {metadata.ep_size}."
    )
    return DRAFT_EXPERT_LOCATION_FAMILY


def _check_draft_slots(model_config: ModelConfig) -> None:
    num_logical = ModelConfigForExpertLocation.from_model_config(
        model_config
    ).num_logical_experts
    num_redundant = get_exec().moe.ep_num_redundant_experts
    ep_size = get_parallel().ep_size
    if (num_logical + num_redundant) % ep_size != 0:
        raise ValueError(
            f"Draft EPLB places the draft's {num_logical} experts plus the "
            f"target's {num_redundant} redundant slots (--ep-num-redundant-experts) "
            f"over EP {ep_size}, but {num_logical + num_redundant} slots do not "
            f"divide evenly; pick a redundant count with "
            f"({num_logical} + R) % {ep_size} == 0 or unset "
            "SGLANG_DSPARK_ENABLE_DRAFT_EPLB."
        )
