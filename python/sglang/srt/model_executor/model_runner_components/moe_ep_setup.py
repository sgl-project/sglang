from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.eplb.lplb_solver import (
    LPLBSolver,
    assert_lplb_supported_model,
    clear_global_lplb_solvers,
    set_global_lplb_solver,
)
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import get_bool_env_var, is_hip, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()


def prepare_moe_topk(
    *,
    model,
    model_config: ModelConfig,
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
            num_routed_experts + get_exec().moe.ep_num_redundant_experts
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


def prebuild_deepep_v2_buffers(
    *,
    model,
    disaggregation_mode: str,
    chunked_prefill_size: int,
    attn_tp_size: int,
) -> None:
    """Build every deepep_v2 dispatcher's ElasticBuffer at deployment time.

    No-op unless the a2a backend is deepep_v2. On a prefill node, first validate
    the per-rank cap against the chunk slice (buffer preallocates per that cap).
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import DeepEPv2Dispatcher
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    if not get_moe_a2a_backend().is_deepep_v2():
        return

    min_tokens_per_rank = None
    if (
        disaggregation_mode == "prefill"
        and chunked_prefill_size
        and chunked_prefill_size > 0
        and attn_tp_size > 0
    ):
        min_tokens_per_rank = chunked_prefill_size // attn_tp_size

    num_prebuilt = 0
    for module in model.modules():
        if not isinstance(module, FusedMoE):
            continue
        dispatcher = module.dispatcher
        if not isinstance(dispatcher, DeepEPv2Dispatcher):
            continue
        if (
            min_tokens_per_rank is not None
            and dispatcher.num_max_dispatch_tokens_per_rank < min_tokens_per_rank
        ):
            raise ValueError(
                "SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK="
                f"{dispatcher.num_max_dispatch_tokens_per_rank} is too small for the "
                "prefill DeepEP-V2 ElasticBuffer: it must be >= the per-rank chunk "
                f"slice (chunked_prefill_size / attn_tp_size = {min_tokens_per_rank}). "
                "DeepEP-V2 preallocates buffer space assuming all of a rank's tokens "
                "may hit one expert, so raise the cap to at least "
                f"{min_tokens_per_rank}."
            )
        dispatcher.prebuild()
        num_prebuilt += 1
    if num_prebuilt:
        log_info_on_rank0(
            logger, f"Prebuilt {num_prebuilt} DeepEP-V2 ElasticBuffer(s) at startup."
        )


def init_lplb_solvers(*, model_config: ModelConfig) -> None:
    """Initialize per-layer LPLB solvers from current expert location metadata."""
    from sglang.srt.distributed import get_moe_ep_group

    # Gate: refuse LP for non-DeepSeek MoE families whose empty-token paths
    # don't participate in the EP all-reduce (would deadlock under DP-
    # attention). Failure here happens before any forward pass.
    architectures = getattr(model_config.hf_config, "architectures", None)
    if architectures:
        assert_lplb_supported_model(architectures[0])

    metadata = get_global_expert_location_metadata()
    if metadata is None:
        return
    clear_global_lplb_solvers()
    ep_group = get_moe_ep_group()
    for lid in range(metadata.num_layers):
        solver = LPLBSolver(
            phy2log=metadata.physical_to_logical_map[lid],
            log2phy=metadata.logical_to_all_physical_map[lid],
            num_gpus=metadata.ep_size,
            ep_group=ep_group,
            logical_to_all_physical_map_num_valid=(
                metadata.logical_to_all_physical_map_num_valid[lid]
            ),
        )
        set_global_lplb_solver(lid, solver)
    logger.info(f"Initialized LPLB solvers for {metadata.num_layers} layers")


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
