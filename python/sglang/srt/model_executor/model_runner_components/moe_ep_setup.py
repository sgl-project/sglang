from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.runtime_context import get_context, get_exec
from sglang.srt.utils import get_bool_env_var, is_hip, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()


def create_moe_load_balancer(*, model_config: ModelConfig):
    """Build the target's EvenMoE MoELoadBalancer, or None without the flag.

    It is published to ``resources.moe_load_balancer`` before the model is built,
    so each target (Hash)TopK binds it at construction.
    """
    if get_exec().moe.moe_load_balancer_algorithm is None:
        return None

    model_info = ModelConfigForExpertLocation.from_model_config(model_config)
    if model_info is None:
        raise ValueError(
            "--moe-load-balancer-algorithm requires a model with routed "
            f"experts; {model_config.hf_config.architectures} has none."
        )

    from evenmoe import MoELoadBalancer
    from evenmoe.adapters.sglang import to_load_balancer_kwargs

    context = get_context()
    context.resources.mlb_model_info = {
        "num_logical_experts": model_info.num_logical_experts,
        "num_groups": model_info.num_groups,
    }
    moe_load_balancer = MoELoadBalancer.from_algorithm(
        **to_load_balancer_kwargs(context)
    )
    context.resources.moe_load_balancer = moe_load_balancer
    return moe_load_balancer


def commit_initial_moe_placement(*, model, moe_load_balancer) -> None:
    """Commit the loaded placement for every layer whose router uses the balancer."""
    layer_ids = [
        module.layer_id
        for module in model.modules()
        if isinstance(module, (TopK, HashTopK)) and module.moe_load_balancer is not None
    ]
    if not layer_ids:
        return
    if None in layer_ids:
        raise RuntimeError("EvenMoE routing requires every MoE TopK to have layer_id.")
    from evenmoe.adapters.sglang import commit_placement

    commit_placement(moe_load_balancer, get_context(), layer_ids)


def prebuild_deepep_v2_buffer(*, model) -> None:
    """Build the deepep_v2 ElasticBuffer at deployment time.

    No-op unless the a2a backend is deepep_v2. The buffer is process-wide and
    shared by every MoE layer, so the first deepep_v2 dispatcher builds it. The
    per-rank cap is validated in validate_deepep_v2_dispatch_token_budget at
    server-args time.
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import DeepEPv2Dispatcher
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    if not get_moe_a2a_backend().is_deepep_v2():
        return

    for module in model.modules():
        if isinstance(module, FusedMoE) and isinstance(
            module.dispatcher, DeepEPv2Dispatcher
        ):
            module.dispatcher.prebuild()
            log_info_on_rank0(
                logger, "Prebuilt the DeepEP-V2 ElasticBuffer at startup."
            )
            return


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
