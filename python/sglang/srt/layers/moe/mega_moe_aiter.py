# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ROCm MegaMoE via AITER FlyDSL MegaMoEV2.

Ports ATOM's ``flydsl_mega_experts.py`` pattern into SGLang: dispatch, both
grouped GEMMs, and combine run inside ``aiter.ops.flydsl.kernels.mega_moe.MegaMoEV2``
(ROCm/aiter#4439). Selected when ``--moe-a2a-backend megamoe`` is set on HIP
with ``SGLANG_USE_AITER=1``. NVIDIA Blackwell still uses DeepGEMM in
``mega_moe.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.distributed.parallel_state import get_moe_ep_group
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_use_aiter = bool(envs.SGLANG_USE_AITER.get()) and _is_hip

_MEGA_CACHE: dict = {}
_MEGA_BUILD_LOGGED = False
_MORI_SHMEM_READY = False
_MEGA_RUNTIME_READY = False


def is_mega_moe_aiter_enabled() -> bool:
    return _is_hip and _use_aiter and get_moe_a2a_backend().is_megamoe()


def mega_moe_aiter_available() -> bool:
    if not is_mega_moe_aiter_enabled():
        return False
    try:
        from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2  # noqa: F401

        return True
    except ImportError:
        return False


def get_mega_moe_mtpr() -> int:
    return int(envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get())


def _validate_mtpr(mtpr: int) -> None:
    if mtpr <= 0 or (mtpr & (mtpr - 1)) != 0:
        raise ValueError(
            "MegaMoE on ROCm requires "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to be a "
            f"positive power of two, got {mtpr}"
        )


def _ensure_mori_shmem(ep_group) -> tuple[int, int]:
    """Initialize MegaMoEV2's process-global mori heap exactly once.

    MegaMoEV2 allocates through ``mori_shmem_create_tensor``, whose reference
    harness registers the bootstrap group as ``default``.  Use SGLang's EP
    Gloo group: unlike the device/RCCL WORLD group it supports the object
    broadcast mori uses to distribute its unique id, and it remains correct
    when EP is a proper subset of WORLD.
    """
    global _MORI_SHMEM_READY

    parallel = get_parallel()
    rank = int(parallel.moe_ep_rank)
    world_size = int(parallel.moe_ep_size)

    if not _MORI_SHMEM_READY:
        import mori
        import torch._C._distributed_c10d as c10d

        try:
            c10d._register_process_group("default", ep_group.cpu_group)
        except Exception as exc:
            if "already registered" not in str(exc):
                raise

        torch.distributed.barrier(group=ep_group.cpu_group)
        mori.shmem.shmem_torch_process_group_init("default")
        mori.shmem.shmem_barrier_all()
        torch.distributed.barrier(group=ep_group.cpu_group)
        _MORI_SHMEM_READY = True
        logger.info(
            "MegaMoE-ROCm: initialized mori shmem heap 'default' on EP Gloo "
            "group (rank=%s world=%s)",
            rank,
            world_size,
        )

    return rank, world_size


def initialize_mega_moe_aiter_runtime(model) -> None:
    """Collectively build and warm MegaMoEV2 before CUDA graph capture."""
    global _MEGA_RUNTIME_READY

    if not is_mega_moe_aiter_enabled() or _MEGA_RUNTIME_READY:
        return

    ep_group = get_moe_ep_group()
    rank, world_size = _ensure_mori_shmem(ep_group)

    moe = next(
        (
            module
            for module in model.modules()
            if hasattr(module, "experts")
            and getattr(module.experts, "_mega_moe_weights_built", False)
        ),
        None,
    )
    if moe is None:
        raise RuntimeError(
            "MegaMoE-ROCm runtime initialization found no prepared MoE layer. "
            "It must run after model weight loading and before graph capture."
        )

    experts = moe.experts
    mtpr = int(getattr(experts, "_mega_moe_mtpr", get_mega_moe_mtpr()))
    _validate_mtpr(mtpr)
    topk = int(moe.config.num_experts_per_tok + moe.num_fused_shared_experts)

    _get_or_build_mega_moe(
        rank=rank,
        world_size=world_size,
        model_dim=int(getattr(experts, "_mega_moe_model_dim", experts.hidden_size)),
        inter_dim=int(
            getattr(experts, "_mega_moe_inter_dim", experts.w13_weight.shape[1] // 2)
        ),
        experts=int(experts.num_experts),
        topk=topk,
        mtpr=mtpr,
        swiglu_limit=float(getattr(experts, "_mega_moe_swiglu_limit", 0.0)),
        w1=experts._mega_w1,
        w1_scale=experts._mega_w1_scale,
        w2=experts._mega_w2,
        w2_scale=experts._mega_w2_scale,
    )

    import mori

    mori.shmem.shmem_barrier_all()
    torch.distributed.barrier(group=ep_group.cpu_group)
    _MEGA_RUNTIME_READY = True
    logger.info(
        "MegaMoE-ROCm runtime ready before graph capture "
        "(rank=%s world=%s mtpr=%s topk=%s)",
        rank,
        world_size,
        mtpr,
        topk,
    )


def build_mega_moe_aiter_weights(experts) -> None:
    """Shuffle raw MXFP4 expert weights into MegaMoEV2 layout.

    Must run on the pre-AITER-shuffle tensors (same contract as ATOM).
    """
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    local_e = int(w13.shape[0])
    if local_e != int(experts.num_local_experts):
        raise RuntimeError(
            "MegaMoE local weight width disagrees with the dispatch layout: "
            f"weights={local_e}, dispatch={experts.num_local_experts}."
        )

    s1 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    s2 = experts.w2_weight_scale_inv.data

    experts._mega_w1 = shuffle_weight_a16w4(w13, 16, True).contiguous()
    experts._mega_w1_scale = shuffle_scale_a16w4(
        s1.reshape(local_e * s1.shape[1], s1.shape[2]), local_e, True
    ).contiguous()
    experts._mega_w2 = shuffle_weight_a16w4(w2, 16, False).contiguous()
    experts._mega_w2_scale = shuffle_scale_a16w4(
        s2.reshape(local_e * s2.shape[1], s2.shape[2]), local_e, False
    ).contiguous()

    experts._mega_moe_mtpr = get_mega_moe_mtpr()
    experts._mega_moe_model_dim = int(experts.hidden_size)
    experts._mega_moe_inter_dim = int(w13.shape[1] // 2)
    experts._mega_moe_swiglu_limit = float(
        getattr(experts.moe_runner_config, "swiglu_limit", None) or 0.0
    )

    global _MEGA_BUILD_LOGGED
    if not _MEGA_BUILD_LOGGED:
        _MEGA_BUILD_LOGGED = True
        logger.info(
            "[MegaMoE-ROCm] Prepared MegaMoEV2 weights: E=%d model_dim=%d "
            "inter_dim=%d mtpr=%d swiglu_limit=%s",
            local_e,
            experts._mega_moe_model_dim,
            experts._mega_moe_inter_dim,
            experts._mega_moe_mtpr,
            experts._mega_moe_swiglu_limit,
        )

    experts._mega_moe_weights_built = True


def _get_or_build_mega_moe(
    *,
    rank: int,
    world_size: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    mtpr: int,
    swiglu_limit: float,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    quant: str = "a8w4",
):
    from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

    key = (
        rank,
        world_size,
        model_dim,
        inter_dim,
        experts,
        topk,
        quant,
        mtpr,
        swiglu_limit,
    )
    mega = _MEGA_CACHE.get(key)
    if mega is None:
        with torch.inference_mode(False), torch.no_grad():
            mega = MegaMoEV2(
                rank=rank,
                world_size=world_size,
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                quant=quant,
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                max_tok_per_rank=mtpr,
                swiglu_limit=swiglu_limit,
            )
        _MEGA_CACHE[key] = mega

    mega._s1_w1 = w1.view(torch.uint8)
    mega._s1_w1_scale = w1_scale.view(torch.uint8)
    mega.w2 = w2
    mega.w2_scale = w2_scale
    return mega


def run_mega_moe_aiter_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch],
    input_ids_global: Optional[torch.Tensor],
    num_tokens: int,
) -> torch.Tensor:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo

    experts = moe.experts
    mtpr = int(getattr(experts, "_mega_moe_mtpr", get_mega_moe_mtpr()))
    _validate_mtpr(mtpr)
    if num_tokens > mtpr:
        raise ValueError(
            f"[MegaMoE-ROCm] num_tokens={num_tokens} exceeds mtpr={mtpr}; "
            "raise SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK or "
            "reduce batch size / cuda-graph max bs"
        )

    if num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=(
                forward_batch.num_token_non_padded
                if forward_batch is not None
                else None
            ),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id,
            ),
            **topk_kwargs,
        )
        topk_ids = topk_output.topk_ids.to(torch.int32).contiguous()
        topk_weights = topk_output.topk_weights.to(torch.float32).contiguous()
        topk = int(topk_ids.shape[1])
    else:
        topk = int(moe.config.num_experts_per_tok + moe.num_fused_shared_experts)
        topk_ids = hidden_states.new_empty((0, topk), dtype=torch.int32)
        topk_weights = hidden_states.new_empty((0, topk), dtype=torch.float32)

    if not hasattr(experts, "_mega_w1"):
        raise RuntimeError("MegaMoE-ROCm weights were not prepared")
    if not _MEGA_RUNTIME_READY:
        raise RuntimeError(
            "MegaMoE-ROCm runtime was not initialized before forward. "
            "BaseRunner.warmup() must run before eager execution or graph capture."
        )
    is_empty = hidden_states.shape[0] == 0
    if is_empty:
        # MegaMoEV2 is an EP collective: idle DP-attention ranks still must
        # participate, but its config selector rejects zero tokens. Dispatch a
        # zero-weight dummy token and discard its output after the collective.
        hidden_states = hidden_states.new_zeros((1, hidden_states.shape[1]))
        topk_weights = topk_weights.new_zeros((1, topk_weights.shape[1]))
        topk_ids = topk_ids.new_zeros((1, topk_ids.shape[1]))

    parallel = get_parallel()
    rank = int(parallel.moe_ep_rank)
    world_size = int(parallel.moe_ep_size)
    mega = _get_or_build_mega_moe(
        rank=rank,
        world_size=world_size,
        model_dim=int(getattr(experts, "_mega_moe_model_dim", experts.hidden_size)),
        inter_dim=int(
            getattr(experts, "_mega_moe_inter_dim", experts.w13_weight.shape[1] // 2)
        ),
        experts=int(experts.num_experts),
        topk=topk,
        mtpr=mtpr,
        swiglu_limit=float(getattr(experts, "_mega_moe_swiglu_limit", 0.0)),
        w1=experts._mega_w1,
        w1_scale=experts._mega_w1_scale,
        w2=experts._mega_w2,
        w2_scale=experts._mega_w2_scale,
    )

    with torch.inference_mode(False), torch.no_grad():
        y = mega.forward(hidden_states.contiguous(), topk_weights, topk_ids)

    if is_empty:
        return y[:0]
    if not experts.should_fuse_routed_scaling_factor_in_topk:
        y.mul_(moe.routed_scaling_factor)
    return y
