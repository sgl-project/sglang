"""Capacity planning for the DeepEP low_latency path.

Single source of truth for the dispatch bound (num_max_dispatch_tokens_per_rank)
and for the GPU memory that must be set aside for allocations landing after the
KV pool: the nvshmem RDMA buffer (outside the torch allocator) and the decode
cuda-graph capture + deep_gemm grouped-GEMM warmup footprint.

The plan is built once per target ModelRunner right before KV-pool sizing —
where the loaded model config and deep_ep itself are available — and its
reservation is subtracted from the KV budget in bytes. mem_fraction_static is
never modified, so an auto-memory serve keeps exactly the runtime headroom main
ships (the invariant whose violation got #28884 reverted).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DEEPEP_LOW_LATENCY_MAX_DISPATCH_TOKENS,
    estimate_low_latency_rdma_size_bytes,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_MIB = 1024**2
_NUM_MAX_TIERS = (1024, 512, 256, 128)
# Headroom for the transient deep_gemm warmup that runs right after capture.
_SAFETY_MIB = 2 * 1024


class DeepEPCapacityPlan(msgspec.Struct):
    # Upper bound on the dispatch bound; the memory reservation is sized for it
    # and the decode-concurrency clamp keeps runtime dispatches within it.
    ceiling: int
    # num_max counts tokens, not requests: spec/MTP verify packs
    # num_tokens_per_bs tokens per request.
    tokens_per_req: int
    # True when the auto reservation sized the ceiling. False (user-set env /
    # kill switch / user mem_fraction) keeps num_max at the static value so no
    # buffer is allocated that nobody budgeted for.
    auto_sized: bool = False
    rdma_mib: float = 0.0
    capture_mib: float = 0.0
    slack_mib: float = 0.0
    reserve_mib: float = 0.0
    # Final dispatch bound, resolved once decode concurrency is known.
    num_max: Optional[int] = None


def is_deepep_low_latency(server_args: ServerArgs) -> bool:
    from sglang.srt.layers.moe.utils import MoeA2ABackend

    return (
        MoeA2ABackend(server_args.moe_a2a_backend).is_deepep()
        and server_args.deepep_mode != "normal"
    )


def deepep_tokens_per_req(server_args: ServerArgs) -> int:
    return (
        server_args.max_speculative_num_draft_tokens
        or server_args.speculative_num_draft_tokens
        or 1
    )


def _extract_num_experts(hf_config) -> Optional[int]:
    for attr in (
        "n_routed_experts",
        "num_experts",
        "num_local_experts",
        "moe_num_experts",
    ):
        value = getattr(hf_config, attr, None)
        if value:
            return int(value)
    return None


def rdma_size_mib(
    num_max: int, hidden: int, num_experts: int, moe_ep_size: int
) -> float:
    # Prefer DeepEP's own size hint; the Python replica covers builds without
    # deep_ep and is checked byte-for-byte against native in CI.
    try:
        from deep_ep import Buffer

        return (
            Buffer.get_low_latency_rdma_size_hint(
                num_max, hidden, max(moe_ep_size, 1), num_experts
            )
            / _MIB
        )
    except Exception:
        return estimate_low_latency_rdma_size_bytes(num_max, hidden, num_experts) / (
            _MIB
        )


def plan_deepep_capacity(
    server_args: ServerArgs,
    model_config: ModelConfig,
    gpu_total_mib: Optional[float],
    moe_ep_size: int,
) -> Optional[DeepEPCapacityPlan]:
    """Build the capacity plan; None when not on the deepep low_latency path."""
    if not is_deepep_low_latency(server_args):
        return None
    tokens_per_req = deepep_tokens_per_req(server_args)
    env = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
    static_plan = DeepEPCapacityPlan(ceiling=env.get(), tokens_per_req=tokens_per_req)

    if not envs.SGLANG_ENABLE_DEEPEP_AUTO_MEM_RESERVE.get():
        return static_plan
    slack_mib = server_args.auto_mem_chunked_slack_mib
    if slack_mib is None or gpu_total_mib is None:
        # mem_fraction_static was user-set (or capacity unknown): the auto
        # formula budgeted nothing for deepep, stay at the static bound.
        return static_plan

    hidden = model_config.hidden_size
    num_experts = _extract_num_experts(model_config.hf_config)
    if not hidden or not num_experts:
        logger.warning(
            "DeepEP low_latency requested but the model config has no "
            "hidden_size/num_experts; skipping the auto mem reservation."
        )
        return static_plan

    hf_config = model_config.hf_config
    moe_intermediate = (
        getattr(hf_config, "moe_intermediate_size", None)
        or getattr(hf_config, "intermediate_size", None)
        or hidden
    )
    # Capture + deep_gemm warmup footprint scales with the grouped-GEMM size
    # (num_experts * moe_intermediate * hidden), NOT num_layers * hidden — a
    # layer-based estimate inverts the GLM-5.2 vs DeepSeek-V4 ordering.
    capture_mib = (
        envs.SGLANG_DEEPEP_CAPTURE_COEF.get()
        * num_experts
        * moe_intermediate
        * hidden
        / _MIB
    )
    cap_mib = envs.SGLANG_DEEPEP_MAX_RESERVE_FRACTION.get() * gpu_total_mib

    if server_args.max_running_requests is not None:
        user_cap = max(
            1,
            (server_args.max_running_requests // max(1, server_args.dp_size))
            * tokens_per_req,
        )
    else:
        user_cap = DEEPEP_LOW_LATENCY_MAX_DISPATCH_TOKENS

    if env.is_set():
        # User pinned the dispatch bound; reserve for exactly that buffer.
        ceiling = env.get()
    else:
        # Largest tier whose buffer + capture fits under the cap. A clamped
        # reservation starves capture (the V3.2 tp8/dp8/TBO OOM), so tier the
        # bound down instead — the concurrency clamp only trims decode batch.
        ceiling = min(user_cap, env.get())
        for candidate in _NUM_MAX_TIERS:
            if candidate > user_cap:
                continue
            required = max(
                0.0,
                rdma_size_mib(candidate, hidden, num_experts, moe_ep_size)
                + capture_mib
                + _SAFETY_MIB
                - slack_mib,
            )
            if required <= cap_mib:
                ceiling = candidate
                break

    rdma_mib = rdma_size_mib(ceiling, hidden, num_experts, moe_ep_size)
    reserve_mib = max(0.0, rdma_mib + capture_mib + _SAFETY_MIB - slack_mib)
    if reserve_mib > cap_mib:
        logger.warning(
            "DeepEP auto mem reserve %.1f GiB exceeds the %.0f%% cap (%.1f GiB); "
            "clamping. Raise SGLANG_DEEPEP_MAX_RESERVE_FRACTION or set "
            "--mem-fraction-static if capture OOMs.",
            reserve_mib / 1024,
            envs.SGLANG_DEEPEP_MAX_RESERVE_FRACTION.get() * 100,
            cap_mib / 1024,
        )
        reserve_mib = cap_mib
    return DeepEPCapacityPlan(
        ceiling=ceiling,
        tokens_per_req=tokens_per_req,
        auto_sized=True,
        rdma_mib=rdma_mib,
        capture_mib=capture_mib,
        slack_mib=slack_mib,
        reserve_mib=reserve_mib,
    )


def resolve_deepep_num_max(plan: DeepEPCapacityPlan, req_pool_size: int) -> int:
    """Resolve the final dispatch bound and export it through the env var (the
    read point for the dispatcher and NPU fuseep)."""
    env = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
    if env.is_set() or not plan.auto_sized:
        plan.num_max = env.get()
        return plan.num_max
    num_max = min(
        plan.ceiling,
        max(1, req_pool_size * plan.tokens_per_req),
        DEEPEP_LOW_LATENCY_MAX_DISPATCH_TOKENS,
    )
    env.set(num_max)
    plan.num_max = num_max
    return num_max
