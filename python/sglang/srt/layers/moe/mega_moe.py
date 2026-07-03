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
"""Mega-MoE forward path and expert-weight prep shared by Deepseek V2/V4."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.dsv4 import (
    mega_moe_pre_dispatch,
    mega_moe_pre_dispatch_waterfill_rank2,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner import get_is_capture_mode

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


logger = logging.getLogger(__name__)
_MEGA_MOE_SYMM_BUFFER: dict = {}
_MEGA_MOE_DG_ENV_APPLIED = False
_MEGA_MOE_SYMM_MEM_BACKEND_APPLIED = False
_MEGA_MOE_TOPK_STATS_CALLS = 0
_MEGA_MOE_TIMING_CALLS = 0
_MEGA_MOE_CAP_BUCKETS_LOGGED = False
_MEGA_MOE_CAP_BUCKET_FREE_GUARD_LOGGED = False
_MEGA_MOE_WATERFILL_FUSE_LOG_CALLS = 0


def _apply_mega_moe_dg_env() -> None:
    """Forward sglang's FP4/MXF4 opt-in flags to DeepGEMM via env vars.

    DeepGEMM reads `DG_USE_FP4_ACTS` (and `DG_USE_MXF4_KIND`) at host-function
    call time — both `get_symm_buffer_for_mega_moe` and `fp8_fp4_mega_moe`.
    Forwarding once at first use is sufficient (these are static config
    flags, not per-request state) and matches the `setdefault` pattern so
    explicit `DG_USE_*` overrides from outside still win.
    """
    global _MEGA_MOE_DG_ENV_APPLIED
    if _MEGA_MOE_DG_ENV_APPLIED:
        return
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get():
        os.environ.setdefault("DG_USE_FP4_ACTS", "1")
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND.get():
        os.environ.setdefault("DG_USE_MXF4_KIND", "1")
    _MEGA_MOE_DG_ENV_APPLIED = True


def _apply_mega_moe_symm_mem_backend() -> None:
    global _MEGA_MOE_SYMM_MEM_BACKEND_APPLIED
    if _MEGA_MOE_SYMM_MEM_BACKEND_APPLIED:
        return

    backend = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_SYMM_MEM_BACKEND.get().strip().upper()
    if not backend and torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability()[0] >= 10:
                backend = "NCCL"
        except RuntimeError:
            pass

    if backend:
        if backend not in ("NCCL", "NVSHMEM"):
            raise ValueError(
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_SYMM_MEM_BACKEND must be one of "
                f"NCCL, NVSHMEM, or empty; got {backend!r}"
            )
        if backend == "NCCL" and os.environ.get("NCCL_CUMEM_ENABLE") != "1":
            old_value = os.environ.get("NCCL_CUMEM_ENABLE", "<unset>")
            os.environ["NCCL_CUMEM_ENABLE"] = "1"
            logger.info(
                "Set NCCL_CUMEM_ENABLE=1 for Mega-MoE NCCL symmetric-memory "
                "backend (was %s).",
                old_value,
            )
        os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

        import torch.distributed._symmetric_memory as symm_mem

        symm_mem.set_backend(backend)
        logger.info("Set torch symmetric-memory backend for Mega-MoE to %s.", backend)

    _MEGA_MOE_SYMM_MEM_BACKEND_APPLIED = True


def _get_mega_moe_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> SymmBuffer:
    import deep_gemm

    _apply_mega_moe_dg_env()
    _apply_mega_moe_symm_mem_backend()

    key = (
        id(group),
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    buf = _MEGA_MOE_SYMM_BUFFER.get(key)
    if buf is None:
        logger.info(
            "Creating DeepGEMM Mega-MoE symmetric buffer cap=%s experts=%s "
            "topk=%s hidden=%s intermediate=%s.",
            num_max_tokens_per_rank,
            num_experts,
            num_topk,
            hidden,
            intermediate_hidden,
        )
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        _MEGA_MOE_SYMM_BUFFER[key] = buf
    return buf


def _has_mega_moe_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> bool:
    key = (
        id(group),
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    return key in _MEGA_MOE_SYMM_BUFFER


def _configured_mega_moe_token_caps() -> list[int]:
    max_cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    raw_buckets = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK_BUCKETS.get().strip()
    )
    if not raw_buckets:
        return [max_cap]

    caps = set()
    for item in raw_buckets.replace(":", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            cap = int(item)
        except ValueError as exc:
            raise ValueError(
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK_BUCKETS "
                f"must contain integers, got {item!r} in {raw_buckets!r}"
            ) from exc
        if cap <= 0:
            raise ValueError(
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK_BUCKETS "
                f"must contain positive integers, got {cap}"
            )
        if cap <= max_cap:
            caps.add(cap)

    caps.add(max_cap)
    return sorted(caps)


def _sync_mega_moe_token_count(
    num_tokens: int,
    group,
    device: torch.device,
) -> int:
    """Return the max token requirement across the Mega-MoE EP group.

    DeepGEMM Mega-MoE is a collective kernel over the symmetric buffer. All EP
    ranks must enter it with the same padded cap; otherwise one rank can wait on
    a different barrier instance and time out. Bucket selection is only enabled
    for eager experiments, so the tiny all-reduce cost is acceptable there.
    """
    if group is None or not torch.distributed.is_initialized():
        return num_tokens

    value = torch.tensor([num_tokens], dtype=torch.int64, device=device)
    torch.distributed.all_reduce(
        value,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )
    return int(value.item())


def _sync_mega_moe_bool_or(
    value: bool,
    group,
    device: torch.device,
) -> bool:
    if group is None or not torch.distributed.is_initialized():
        return value

    value = torch.tensor(
        [1 if value else 0],
        dtype=torch.int32,
        device=device,
    )
    torch.distributed.all_reduce(
        value,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )
    return bool(value.item())


def _select_mega_moe_token_cap(
    num_tokens: int,
    *,
    caps: Optional[list[int]] = None,
    group=None,
    device: Optional[torch.device] = None,
    free_hbm_guard: bool = True,
) -> int:
    """Select the smallest configured Mega-MoE cap that can hold num_tokens."""
    global _MEGA_MOE_CAP_BUCKETS_LOGGED, _MEGA_MOE_CAP_BUCKET_FREE_GUARD_LOGGED
    caps = caps or _configured_mega_moe_token_caps()
    if len(caps) > 1 and not _MEGA_MOE_CAP_BUCKETS_LOGGED:
        logger.info("Using DeepGEMM Mega-MoE cap buckets: %s.", caps)
        _MEGA_MOE_CAP_BUCKETS_LOGGED = True
    max_cap = caps[-1]
    for cap in caps:
        if num_tokens <= cap:
            min_free_gb = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_CAP_BUCKET_MIN_FREE_GB.get()
            if (
                free_hbm_guard
                and cap < max_cap
                and min_free_gb > 0
                and torch.cuda.is_available()
            ):
                try:
                    free_bytes, _ = torch.cuda.mem_get_info()
                    free_gb = free_bytes / float(1024**3)
                except RuntimeError:
                    free_gb = min_free_gb
                local_low_free_hbm = free_gb < min_free_gb
                low_free_hbm = _sync_mega_moe_bool_or(
                    local_low_free_hbm,
                    group,
                    device or torch.device("cuda", torch.cuda.current_device()),
                )
                if low_free_hbm:
                    if not _MEGA_MOE_CAP_BUCKET_FREE_GUARD_LOGGED:
                        logger.info(
                            "Skip DeepGEMM Mega-MoE cap bucket %s for "
                            "num_tokens=%s because at least one EP rank has "
                            "free HBM below %.2fGB (local free %.2fGB); using "
                            "max cap %s.",
                            cap,
                            num_tokens,
                            min_free_gb,
                            free_gb,
                            max_cap,
                        )
                        _MEGA_MOE_CAP_BUCKET_FREE_GUARD_LOGGED = True
                    return max_cap
            return cap
    return max_cap


def _preinit_mega_moe_token_caps() -> list[int]:
    caps = _configured_mega_moe_token_caps()
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_PREINIT_ALL_CAP_BUCKETS.get():
        return caps
    return [caps[-1]]


def _current_max_tokens_per_rank(num_tokens: int) -> int:
    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens:
        return max(global_num_tokens)
    return num_tokens


def pre_initialize_mega_moe_symm_buffers(model: torch.nn.Module) -> None:
    """Pre-create DeepGEMM Mega-MoE symmetric buffers on every EP rank.

    DeepGEMM's `get_symm_buffer_for_mega_moe` calls torch symmetric-memory
    rendezvous under the hood. If this happens lazily on the first real
    request, only the rank that receives work may enter the rendezvous while
    its EP peers are idle, causing the first request to hang. During model
    startup all ranks are active, so initialize the buffers there instead.
    """
    if not get_moe_a2a_backend().is_megamoe():
        return

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    ep_group = get_moe_ep_group()
    caps = _preinit_mega_moe_token_caps()
    initialized = 0

    ep_group.barrier()
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None or not getattr(experts, "_mega_moe_weights_built", False):
            continue

        for cap in caps:
            _get_mega_moe_symm_buffer(
                ep_group.device_group,
                num_experts=experts.num_experts,
                num_max_tokens_per_rank=cap,
                num_topk=module.config.num_experts_per_tok
                + module.num_fused_shared_experts,
                hidden=module.config.hidden_size,
                intermediate_hidden=module.config.moe_intermediate_size,
            )
            initialized += 1
    torch.cuda.synchronize()
    ep_group.barrier()

    if initialized > 0 and ep_group.rank_in_group == 0:
        logger.info(
            "Pre-initialized DeepGEMM Mega-MoE symmetric buffers for %s "
            "local MoE layer/cap pairs with caps=%s.",
            initialized,
            caps,
        )


def pre_initialize_mega_moe_symm_buffers_from_config(model_config) -> None:
    """Pre-create the common Mega-MoE symmetric buffer before model loading.

    On Blackwell, torch's NCCL symmetric-memory rendezvous is sensitive to the
    process CUDA/NCCL state. Creating the DeepGEMM buffer immediately after the
    distributed groups are initialized avoids doing the first rendezvous after
    weight loading has imported and initialized many CUDA extension libraries.
    """
    if not get_moe_a2a_backend().is_megamoe():
        return

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    hf_config = getattr(model_config, "hf_config", model_config)
    num_experts = getattr(
        hf_config, "n_routed_experts", getattr(hf_config, "num_experts", None)
    )
    num_topk = getattr(hf_config, "num_experts_per_tok", None)
    hidden = getattr(hf_config, "hidden_size", None)
    intermediate_hidden = getattr(hf_config, "moe_intermediate_size", None)
    if intermediate_hidden is None:
        intermediate_hidden = getattr(hf_config, "intermediate_size", None)
    if any(v is None for v in (num_experts, num_topk, hidden, intermediate_hidden)):
        logger.warning(
            "Skip early DeepGEMM Mega-MoE symmetric buffer initialization because "
            "the model config is missing required MoE dimensions."
        )
        return

    ep_group = get_moe_ep_group()
    caps = _preinit_mega_moe_token_caps()
    ep_group.barrier()
    for cap in caps:
        _get_mega_moe_symm_buffer(
            ep_group.device_group,
            num_experts=num_experts,
            num_max_tokens_per_rank=cap,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
        )
    torch.cuda.synchronize()
    ep_group.barrier()

    if ep_group.rank_in_group == 0:
        logger.info(
            "Early pre-initialized DeepGEMM Mega-MoE symmetric buffer from "
            "config with caps=%s, experts=%s, topk=%s, hidden=%s, intermediate=%s.",
            caps,
            num_experts,
            num_topk,
            hidden,
            intermediate_hidden,
        )


def should_use_mega_moe(moe: DeepseekV2MoE, hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False
    if get_is_capture_mode():
        return True

    max_tokens_per_rank = _current_max_tokens_per_rank(hidden_states.shape[0])
    return max_tokens_per_rank <= _configured_mega_moe_token_caps()[-1]


def forward_mega_moe(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    should_allreduce_fusion: bool = False,
    use_reduce_scatter: bool = False,
    input_ids_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]

    sbo_overlap_flag = (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )

    if sbo_overlap_flag:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = torch.cuda.stream(moe.alt_stream)
        with mega_stream_ctx:
            y = _run_mega_routed(
                moe, hidden_states, forward_batch, input_ids_global, num_tokens
            )
        current_stream.wait_stream(moe.alt_stream)
    else:
        y = _run_mega_routed(
            moe, hidden_states, forward_batch, input_ids_global, num_tokens
        )
        shared_output = moe._forward_shared_experts(hidden_states)

    if shared_output is not None and not getattr(moe, "_shared_expert_tp1", False):
        y.add_(shared_output)
    if moe.tp_size > 1:
        from sglang.srt.distributed import tensor_model_parallel_all_reduce
        from sglang.srt.layers.moe import should_skip_post_experts_all_reduce

        if not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            y = tensor_model_parallel_all_reduce(y)
    if shared_output is not None and getattr(moe, "_shared_expert_tp1", False):
        y.add_(shared_output)
    return y


def _run_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch],
    input_ids_global: Optional[torch.Tensor],
    num_tokens: int,
) -> torch.Tensor:
    import deep_gemm

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    hidden_size = moe.config.hidden_size
    timing_call = _should_log_mega_moe_timing()
    timing_events = []
    use_fp4_acts = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get()
    moe_ep_group = get_moe_ep_group()
    ep_group = moe_ep_group.device_group
    fused_waterfill_predispatch = False
    waterfill_dispatch_plan = None
    waterfill_balancer = None
    one_way_remote_shared_for_fused_waterfill = False

    def mark_timing(name: str) -> None:
        if not timing_call:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        timing_events.append((name, event))

    mark_timing("start")

    if num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        # Static redundant-expert placement can be active with enable_eplb=False.
        # Let the helper decide from ep_dispatch_algorithm instead of gating here.
        dispatch_info = ExpertLocationDispatchInfo.init_new(layer_id=moe.layer_id)
        fuse_env_field = envs.SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH
        fuse_env_raw = os.environ.get(fuse_env_field.name)
        fuse_env_explicit = fuse_env_raw not in (None, "")
        fuse_env = fuse_env_field.get() if fuse_env_explicit else False
        topk_stats_interval = envs.SGLANG_MEGA_MOE_LOG_TOPK_STATS_INTERVAL.get()
        waterfill_enabled = bool(getattr(moe.topk, "enable_deepep_waterfill", False))
        has_waterfill_balancer = (
            getattr(moe.topk, "deepep_waterfill_balancer", None) is not None
        )
        force_local_shared = envs.SGLANG_WATERFILL_FORCE_LOCAL_SHARED.get()
        has_no_waterfill_topk = hasattr(moe.topk, "forward_without_deepep_waterfill")
        # Mega-MoE normally materializes Waterfill's expanded [N, routed_topk + 1]
        # tensors, then immediately copies/quantizes them into DeepGEMM's
        # symmetric buffer.  For the common DSV4 EP=TP=2 FP8 path, the
        # pre-dispatch kernel can append the shared-expert column directly while
        # writing that buffer.  Keep the env as an explicit override, but default
        # to the fused path when all correctness constraints are met.
        can_fuse_waterfill_predispatch = (
            not use_fp4_acts
            and topk_stats_interval <= 0
            and waterfill_enabled
            and has_waterfill_balancer
            and moe_ep_group.world_size == 2
            and not force_local_shared
            and has_no_waterfill_topk
        )
        if fuse_env_explicit:
            can_fuse_waterfill_predispatch = fuse_env and can_fuse_waterfill_predispatch
        else:
            fuse_env = can_fuse_waterfill_predispatch
        if can_fuse_waterfill_predispatch:
            topk_output = moe.topk.forward_without_deepep_waterfill(
                hidden_states,
                router_logits,
                expert_location_dispatch_info=dispatch_info,
                **topk_kwargs,
            )
            waterfill_balancer = moe.topk.deepep_waterfill_balancer
            waterfill_dispatch_plan = waterfill_balancer.build_dispatch_plan_for_topk(
                topk_output.topk_ids,
                num_tokens,
            )
            fused_waterfill_predispatch = waterfill_dispatch_plan is not None
            one_way_env_field = envs.SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED
            one_way_env_raw = os.environ.get(one_way_env_field.name)
            one_way_env_explicit = one_way_env_raw not in (None, "")
            # The measured DSV4 Mega-MoE fused path does not convert two-way
            # shared-expert traffic into a reliable kernel speedup: it keeps the
            # same DeepGEMM padded shape and can add remote shared work on both
            # ranks.  For the fused rank-2 path, default to one-way remote shared
            # so only the routed-heavy source rank sends shared tokens remotely.
            # Keep the env as an explicit override.
            one_way_remote_shared_for_fused_waterfill = (
                one_way_env_field.get() if one_way_env_explicit else True
            )
            if not fused_waterfill_predispatch:
                topk_output = waterfill_balancer.expand_topk(topk_output, num_tokens)
        else:
            topk_output = moe.topk(
                hidden_states,
                router_logits,
                expert_location_dispatch_info=dispatch_info,
                **topk_kwargs,
            )
        _maybe_log_waterfill_fuse_predispatch(
            moe=moe,
            num_tokens=num_tokens,
            fuse_env=fuse_env,
            use_fp4_acts=use_fp4_acts,
            timing_call=timing_call,
            topk_stats_interval=topk_stats_interval,
            waterfill_enabled=waterfill_enabled,
            has_waterfill_balancer=has_waterfill_balancer,
            ep_world_size=moe_ep_group.world_size,
            force_local_shared=force_local_shared,
            has_no_waterfill_topk=has_no_waterfill_topk,
            can_fuse=can_fuse_waterfill_predispatch,
            plan_ready=waterfill_dispatch_plan is not None,
            fused=fused_waterfill_predispatch,
            topk_type=type(moe.topk).__name__,
        )
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
    else:
        topk_ids = None
        topk_weights = None

    mark_timing("topk")

    num_experts = moe.experts.num_experts
    top_k = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    intermediate_size = moe.config.moe_intermediate_size
    max_tokens_per_rank = _current_max_tokens_per_rank(num_tokens)
    caps = _configured_mega_moe_token_caps()
    if get_is_capture_mode():
        # Keep graph capture on the configured max shape. The bucketed path is
        # for eager prefill experiments where changing the DeepGEMM padded cap
        # can reduce the profiled Mega-MoE span.
        num_max_tokens_per_rank = caps[-1]
    else:
        if len(caps) > 1:
            max_tokens_per_rank = _sync_mega_moe_token_count(
                max_tokens_per_rank,
                ep_group,
                hidden_states.device,
            )
        candidate_cap = _select_mega_moe_token_cap(
            max_tokens_per_rank,
            caps=caps,
            free_hbm_guard=False,
        )
        needs_bucket_create = not _has_mega_moe_symm_buffer(
            ep_group,
            num_experts=num_experts,
            num_max_tokens_per_rank=candidate_cap,
            num_topk=top_k,
            hidden=hidden_size,
            intermediate_hidden=intermediate_size,
        )
        if len(caps) > 1:
            needs_bucket_create = _sync_mega_moe_bool_or(
                needs_bucket_create,
                ep_group,
                hidden_states.device,
            )
        num_max_tokens_per_rank = _select_mega_moe_token_cap(
            max_tokens_per_rank,
            caps=caps,
            group=ep_group if len(caps) > 1 else None,
            device=hidden_states.device,
            free_hbm_guard=needs_bucket_create,
        )
    assert max_tokens_per_rank <= num_max_tokens_per_rank, (
        f"mega MoE: max_tokens_per_rank={max_tokens_per_rank} "
        f"(local num_tokens={num_tokens}) exceeds cap "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
        f"{num_max_tokens_per_rank}; raise the env var or shrink "
        f"cuda_graph_max_bs / chunked_prefill_size accordingly"
    )

    buf = _get_mega_moe_symm_buffer(
        ep_group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=top_k,
        hidden=hidden_size,
        intermediate_hidden=intermediate_size,
    )
    mark_timing("buffer")

    if num_tokens > 0:
        topk_ids_in = topk_ids.to(torch.int32)
        topk_weights_in = topk_weights.to(torch.float32)
    else:
        topk_ids_in = hidden_states.new_empty((0, top_k), dtype=torch.int32)
        topk_weights_in = hidden_states.new_empty((0, top_k), dtype=torch.float32)

    mark_timing("cast")
    _maybe_log_mega_moe_topk_stats(moe, topk_ids_in, num_experts, num_tokens)

    if fused_waterfill_predispatch:
        assert waterfill_dispatch_plan is not None
        assert waterfill_balancer is not None
        rank_load = waterfill_dispatch_plan.rank_load
        if rank_load.device != hidden_states.device:
            rank_load = rank_load.to(device=hidden_states.device, non_blocking=True)
        old_experts_per_rank = waterfill_balancer.old_experts_per_rank
        shared_replicas_per_rank = waterfill_balancer.shared_replicas_per_rank
        new_experts_per_rank = old_experts_per_rank + shared_replicas_per_rank
        mega_moe_pre_dispatch_waterfill_rank2(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            rank_load,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            source_rank=moe_ep_group.rank_in_group,
            shared_weight=waterfill_balancer.shared_weight,
            local_pref_numer=max(envs.SGLANG_WATERFILL_LOCAL_PREF_NUMER.get(), 1),
            local_pref_denom=max(envs.SGLANG_WATERFILL_LOCAL_PREF_DENOM.get(), 1),
            remote_cost_tokens=max(envs.SGLANG_WATERFILL_REMOTE_COST_TOKENS.get(), 0),
            allow_all_ranks=waterfill_dispatch_plan.allow_all_ranks,
            one_way_remote_shared=one_way_remote_shared_for_fused_waterfill,
            old_experts_per_rank=old_experts_per_rank,
            new_experts_per_rank=new_experts_per_rank,
            shared_replicas_per_rank=shared_replicas_per_rank,
            quant_group_size=32,
        )
    elif use_fp4_acts:
        # FP4 path goes through DeepGEMM's mega_moe_pre_dispatch which
        # handles the E2M1 packing variant. The jit implementation
        # only emits FP8.
        deep_gemm.mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=num_tokens,
            group_size=32,
            use_fp4_acts=True,
        )
    else:
        mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            quant_group_size=32,
        )
    mark_timing("pre_dispatch")

    # Allocate at least one row so y has a non-null CUDA data_ptr;
    # the DeepGEMM tvm-ffi binding rejects nullptr in convert_to_torch_tensor().
    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    swiglu_limit = getattr(moe.config, "swiglu_limit", None)
    deep_gemm.fp8_fp4_mega_moe(
        y,
        moe.experts.mega_l1_weights,
        moe.experts.mega_l2_weights,
        buf,
        recipe=(1, 1, 32),
        activation="swiglu",
        activation_clamp=swiglu_limit,
        fast_math=True,
    )
    mark_timing("fp8_fp4")
    y = y[:num_tokens]

    if not moe.experts.should_fuse_routed_scaling_factor_in_topk:
        y.mul_(moe.routed_scaling_factor)
    mark_timing("done")
    _log_mega_moe_timing(moe, topk_ids_in, num_experts, num_tokens, timing_events)
    return y


def _should_log_mega_moe_timing() -> bool:
    interval = envs.SGLANG_MEGA_MOE_LOG_TIMING_INTERVAL.get()
    if interval <= 0:
        return False

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    ep_rank = get_moe_ep_group().rank_in_group
    if ep_rank != 0 and os.environ.get("SGLANG_MEGA_MOE_LOG_ALL_RANKS") != "1":
        return False

    global _MEGA_MOE_TIMING_CALLS
    _MEGA_MOE_TIMING_CALLS += 1
    return _MEGA_MOE_TIMING_CALLS % interval == 0


def _maybe_log_waterfill_fuse_predispatch(
    *,
    moe: DeepseekV2MoE,
    num_tokens: int,
    fuse_env: bool,
    use_fp4_acts: bool,
    timing_call: bool,
    topk_stats_interval: int,
    waterfill_enabled: bool,
    has_waterfill_balancer: bool,
    ep_world_size: int,
    force_local_shared: bool,
    has_no_waterfill_topk: bool,
    can_fuse: bool,
    plan_ready: bool,
    fused: bool,
    topk_type: str,
) -> None:
    interval = envs.SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH_LOG_INTERVAL.get()
    if interval <= 0 or not fuse_env:
        return

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    ep_rank = get_moe_ep_group().rank_in_group
    if ep_rank != 0 and os.environ.get("SGLANG_MEGA_MOE_LOG_ALL_RANKS") != "1":
        return

    global _MEGA_MOE_WATERFILL_FUSE_LOG_CALLS
    _MEGA_MOE_WATERFILL_FUSE_LOG_CALLS += 1
    if _MEGA_MOE_WATERFILL_FUSE_LOG_CALLS % interval != 0:
        return

    logger.info(
        "MEGA_MOE_WATERFILL_FUSE_PREDISPATCH ep_rank=%s layer=%s tokens=%s "
        "topk_type=%s fuse_env=%s use_fp4_acts=%s timing_call=%s "
        "topk_stats_interval=%s waterfill_enabled=%s has_balancer=%s "
        "ep_world_size=%s force_local_shared=%s has_no_waterfill_topk=%s "
        "can_fuse=%s plan_ready=%s fused=%s",
        ep_rank,
        moe.layer_id,
        num_tokens,
        topk_type,
        fuse_env,
        use_fp4_acts,
        timing_call,
        topk_stats_interval,
        waterfill_enabled,
        has_waterfill_balancer,
        ep_world_size,
        force_local_shared,
        has_no_waterfill_topk,
        can_fuse,
        plan_ready,
        fused,
    )


def _rank_count_summary(
    topk_ids: torch.Tensor,
    num_experts: int,
    ep_size: int,
) -> tuple[list[int], float]:
    experts_per_rank = max(num_experts // ep_size, 1)
    valid = topk_ids >= 0
    ranks = torch.clamp(topk_ids // experts_per_rank, min=0, max=ep_size - 1)
    counts = torch.bincount(ranks[valid].reshape(-1), minlength=ep_size)[:ep_size]
    counts_cpu = counts.detach().cpu().tolist()
    max_count = max(counts_cpu) if counts_cpu else 0
    min_count = min(counts_cpu) if counts_cpu else 0
    ratio = float(max_count) / float(min_count) if min_count else float("inf")
    return counts_cpu, ratio


def _percentile_from_sorted(values: list[int], p: float) -> int:
    if not values:
        return 0
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return values[idx]


def _expert_block_count(counts: torch.Tensor, block_m: int) -> int:
    if counts.numel() == 0:
        return 0
    return int(((counts + block_m - 1) // block_m).sum().item())


def _max_expert_block_count(counts: torch.Tensor, block_m: int) -> int:
    if counts.numel() == 0:
        return 0
    return int(((counts + block_m - 1) // block_m).max().item())


def _mega_moe_shape_summary(
    moe: DeepseekV2MoE,
    topk_ids: torch.Tensor,
    num_experts: int,
    source_rank: int,
) -> dict[str, int | float | list[int]]:
    ep_size = moe.moe_ep_size
    experts_per_rank = max(num_experts // ep_size, 1)
    valid = topk_ids >= 0
    if topk_ids.numel() == 0 or not bool(valid.any().item()):
        return {
            "routed_counts": [0 for _ in range(ep_size)],
            "shared_counts": [0 for _ in range(ep_size)],
            "shared_replicas_per_rank": getattr(
                moe.experts, "num_fused_shared_expert_replicas_per_rank", 1
            ),
            "expert_counts": [0 for _ in range(num_experts)],
            "active_experts": 0,
            "active_local_experts": 0,
            "max_expert_tokens": 0,
            "max_local_expert_tokens": 0,
            "local_expert_tokens_sum": 0,
            "local_expert_blocks_64": 0,
            "local_expert_blocks_128": 0,
            "local_expert_blocks_256": 0,
            "max_local_expert_blocks_64": 0,
            "max_local_expert_blocks_128": 0,
            "max_local_expert_blocks_256": 0,
            "p95_nonzero_expert_tokens": 0,
            "remote_routed_entries": 0,
            "remote_shared_entries": 0,
            "shared_remote_new_rank": 0,
            "tokens_multi_routed_rank": 0,
            "tokens_multi_full_rank": 0,
            "mean_distinct_routed_ranks": 0.0,
            "mean_distinct_full_ranks": 0.0,
        }

    ranks = torch.clamp(topk_ids // experts_per_rank, min=0, max=ep_size - 1)
    shared_cols = min(
        max(getattr(moe, "num_fused_shared_experts", 0), 0), topk_ids.shape[1]
    )
    routed_cols = topk_ids.shape[1] - shared_cols
    routed_valid = valid[:, :routed_cols] if routed_cols > 0 else valid[:, :0]
    routed_ranks = ranks[:, :routed_cols] if routed_cols > 0 else ranks[:, :0]
    shared_valid = valid[:, routed_cols:] if shared_cols > 0 else valid[:, :0]
    shared_ranks = ranks[:, routed_cols:] if shared_cols > 0 else ranks[:, :0]

    def rank_counts(rank_tensor: torch.Tensor, valid_tensor: torch.Tensor) -> list[int]:
        if valid_tensor.numel() == 0 or not bool(valid_tensor.any().item()):
            return [0 for _ in range(ep_size)]
        counts = torch.bincount(
            rank_tensor[valid_tensor].reshape(-1), minlength=ep_size
        )[:ep_size]
        return counts.detach().cpu().tolist()

    expert_counts = torch.bincount(topk_ids[valid].reshape(-1), minlength=num_experts)[
        :num_experts
    ]
    local_start = source_rank * experts_per_rank
    local_end = min(local_start + experts_per_rank, num_experts)
    local_expert_counts = expert_counts[local_start:local_end]
    expert_counts_cpu = expert_counts.detach().cpu().tolist()
    nonzero_counts = expert_counts[expert_counts > 0].detach().cpu().tolist()
    nonzero_counts.sort()

    distinct_routed = torch.zeros(
        topk_ids.shape[0], dtype=torch.int32, device=topk_ids.device
    )
    distinct_full = torch.zeros_like(distinct_routed)
    for rank in range(ep_size):
        if routed_cols > 0:
            distinct_routed += (
                ((routed_ranks == rank) & routed_valid).any(dim=1).to(torch.int32)
            )
        distinct_full += ((ranks == rank) & valid).any(dim=1).to(torch.int32)

    shared_remote_new_rank = 0
    if shared_cols > 0 and routed_cols > 0:
        first_shared_rank = shared_ranks[:, 0]
        first_shared_valid = shared_valid[:, 0]
        shared_rank_in_routed = (
            (routed_ranks == first_shared_rank[:, None]) & routed_valid
        ).any(dim=1)
        shared_remote_new_rank = int(
            (
                first_shared_valid
                & (first_shared_rank != source_rank)
                & ~shared_rank_in_routed
            )
            .sum()
            .item()
        )

    remote_routed_entries = (
        int(((routed_ranks != source_rank) & routed_valid).sum().item())
        if routed_cols > 0
        else 0
    )
    remote_shared_entries = (
        int(((shared_ranks != source_rank) & shared_valid).sum().item())
        if shared_cols > 0
        else 0
    )

    return {
        "routed_counts": rank_counts(routed_ranks, routed_valid),
        "shared_counts": rank_counts(shared_ranks, shared_valid),
        "shared_replicas_per_rank": getattr(
            moe.experts, "num_fused_shared_expert_replicas_per_rank", 1
        ),
        "expert_counts": expert_counts_cpu,
        "active_experts": int((expert_counts > 0).sum().item()),
        "active_local_experts": int((local_expert_counts > 0).sum().item()),
        "max_expert_tokens": int(expert_counts.max().item()),
        "max_local_expert_tokens": (
            int(local_expert_counts.max().item()) if local_expert_counts.numel() else 0
        ),
        "local_expert_tokens_sum": int(local_expert_counts.sum().item()),
        "local_expert_blocks_64": _expert_block_count(local_expert_counts, 64),
        "local_expert_blocks_128": _expert_block_count(local_expert_counts, 128),
        "local_expert_blocks_256": _expert_block_count(local_expert_counts, 256),
        "max_local_expert_blocks_64": _max_expert_block_count(local_expert_counts, 64),
        "max_local_expert_blocks_128": _max_expert_block_count(
            local_expert_counts, 128
        ),
        "max_local_expert_blocks_256": _max_expert_block_count(
            local_expert_counts, 256
        ),
        "p95_nonzero_expert_tokens": _percentile_from_sorted(nonzero_counts, 0.95),
        "remote_routed_entries": remote_routed_entries,
        "remote_shared_entries": remote_shared_entries,
        "shared_remote_new_rank": shared_remote_new_rank,
        "tokens_multi_routed_rank": int((distinct_routed > 1).sum().item()),
        "tokens_multi_full_rank": int((distinct_full > 1).sum().item()),
        "mean_distinct_routed_ranks": round(
            float(distinct_routed.float().mean().item()), 4
        ),
        "mean_distinct_full_ranks": round(
            float(distinct_full.float().mean().item()), 4
        ),
    }


def _log_mega_moe_timing(
    moe: DeepseekV2MoE,
    topk_ids: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    timing_events: list[tuple[str, torch.cuda.Event]],
) -> None:
    if not timing_events:
        return
    timing_events[-1][1].synchronize()
    elapsed = {
        f"{timing_events[i - 1][0]}_to_{timing_events[i][0]}_ms": timing_events[i - 1][
            1
        ].elapsed_time(timing_events[i][1])
        for i in range(1, len(timing_events))
    }
    counts_cpu, ratio = _rank_count_summary(topk_ids, num_experts, moe.moe_ep_size)
    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    ep_rank = get_moe_ep_group().rank_in_group
    shape = _mega_moe_shape_summary(moe, topk_ids, num_experts, ep_rank)
    logger.info(
        "MEGA_MOE_TIMING ep_rank=%s layer=%s tokens=%s waterfill=%s force_local=%s "
        "counts=%s ratio=%.4f shape=%s timing=%s",
        ep_rank,
        moe.layer_id,
        num_tokens,
        bool(getattr(moe.topk, "enable_deepep_waterfill", False)),
        envs.SGLANG_WATERFILL_FORCE_LOCAL_SHARED.get(),
        counts_cpu,
        ratio,
        shape,
        {k: round(v, 4) for k, v in elapsed.items()},
    )


def _maybe_log_mega_moe_topk_stats(
    moe: DeepseekV2MoE,
    topk_ids: torch.Tensor,
    num_experts: int,
    num_tokens: int,
) -> None:
    interval = envs.SGLANG_MEGA_MOE_LOG_TOPK_STATS_INTERVAL.get()
    if interval <= 0:
        return

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    ep_rank = get_moe_ep_group().rank_in_group
    if ep_rank != 0 and os.environ.get("SGLANG_MEGA_MOE_LOG_ALL_RANKS") != "1":
        return

    global _MEGA_MOE_TOPK_STATS_CALLS
    _MEGA_MOE_TOPK_STATS_CALLS += 1
    if _MEGA_MOE_TOPK_STATS_CALLS % interval != 0:
        return

    ep_size = moe.moe_ep_size
    experts_per_rank = max(num_experts // ep_size, 1)
    counts_cpu, ratio = _rank_count_summary(topk_ids, num_experts, ep_size)
    max_count = max(counts_cpu) if counts_cpu else 0
    min_count = min(counts_cpu) if counts_cpu else 0
    logger.info(
        "MEGA_MOE_TOPK_STATS ep_rank=%s layer=%s tokens=%s topk=%s ep_size=%s "
        "experts_per_rank=%s waterfill=%s force_local=%s counts=%s "
        "max_min=%s/%s ratio=%.4f",
        get_moe_ep_group().rank_in_group,
        moe.layer_id,
        num_tokens,
        topk_ids.shape[1],
        ep_size,
        experts_per_rank,
        bool(getattr(moe.topk, "enable_deepep_waterfill", False)),
        envs.SGLANG_WATERFILL_FORCE_LOCAL_SHARED.get(),
        counts_cpu,
        max_count,
        min_count,
        ratio,
    )


def build_mega_moe_experts_weights(experts) -> None:
    from deep_gemm import (
        transform_sf_into_required_layout,
        transform_weights_for_mega_moe,
    )
    from deep_gemm.mega import _interleave_l1_weights, _transpose_sf_for_utccp

    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    num_groups, n1, packed_k1 = w13.shape
    _, n2, packed_k2 = w2.shape

    quant_method = getattr(experts, "quant_method", None)
    is_fp4_packed = bool(
        getattr(quant_method, "is_fp4_expert", False)
        or getattr(experts, "_mega_moe_weights_are_fp4", False)
    )
    scales_format_ue8m0 = bool(
        getattr(experts.w13_weight_scale_inv, "format_ue8m0", False)
        and getattr(experts.w2_weight_scale_inv, "format_ue8m0", False)
    )
    if is_fp4_packed:
        k1 = packed_k1 * 2
        k2 = packed_k2 * 2
    else:
        k1 = packed_k1
        k2 = packed_k2

    if scales_format_ue8m0 and not is_fp4_packed:
        w13_sf = w13_sf_fp32
        w2_sf = w2_sf_fp32
    else:
        w13_sf = transform_sf_into_required_layout(
            w13_sf_fp32,
            mn=n1,
            k=k1,
            recipe=(1, 32),
            num_groups=num_groups,
            disable_ue8m0_cast=False,
        )
        w2_sf = transform_sf_into_required_layout(
            w2_sf_fp32,
            mn=n2,
            k=k2,
            recipe=(1, 32),
            num_groups=num_groups,
            disable_ue8m0_cast=False,
        )

    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        # Build the interleaved L1 weight + scale once; share the weight buffer
        # between `w13_weight.data` (normal deep-ep path) and `mega_l1_weights[0]`
        # (mega moe path). Mega moe additionally needs a UTCCP-transposed scale;
        # the deep-ep path consumes the non-transposed interleaved scale and a
        # swizzle-aware activation kernel. L2 weight is untouched by the mega
        # transform, so the existing `w2_weight.data` is shared directly.
        w13_interleaved, w13_sf_interleaved = _interleave_l1_weights((w13, w13_sf))
        w13_sf_utccp = _transpose_sf_for_utccp(w13_sf_interleaved)
        w2_sf_utccp = _transpose_sf_for_utccp(w2_sf)

        experts.w13_weight.data = w13_interleaved
        experts.w13_weight_scale_inv.data = w13_sf_interleaved
        experts.w2_weight_scale_inv.data = w2_sf
        experts.w13_weight_scale_inv.format_ue8m0 = True
        experts.w2_weight_scale_inv.format_ue8m0 = True

        experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)
        experts.mega_l2_weights = (experts.w2_weight.data, w2_sf_utccp)
    else:
        l1_pair, l2_pair = transform_weights_for_mega_moe((w13, w13_sf), (w2, w2_sf))

        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_weights_built = True


def convert_fp8_experts_to_fp4_for_mega_moe(experts, weight_block_size) -> None:
    """Convert block-FP8 checkpoint expert weights to DeepGEMM Mega-MoE FP4."""
    from deep_gemm.utils import per_token_cast_to_fp4

    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

    if getattr(experts, "_mega_moe_weights_are_fp4", False):
        return
    assert weight_block_size == [128, 128], weight_block_size

    def convert_pair(weight: torch.nn.Parameter, scale: torch.nn.Parameter):
        num_groups, n, k = weight.shape
        fp4_weight = torch.empty(
            (num_groups, n, k // 2),
            device=weight.device,
            dtype=torch.int8,
        )
        fp4_scale = torch.empty(
            (num_groups, n, k // 32),
            device=weight.device,
            dtype=torch.float32,
        )
        for group_idx in range(num_groups):
            bf16_weight = block_quant_dequant(
                weight[group_idx],
                scale[group_idx],
                weight_block_size,
                torch.bfloat16,
            )
            fp4_weight[group_idx], fp4_scale[group_idx] = per_token_cast_to_fp4(
                bf16_weight, use_ue8m0=True, gran_k=32
            )
            del bf16_weight
        return fp4_weight, fp4_scale

    w13_weight, w13_scale = convert_pair(
        experts.w13_weight, experts.w13_weight_scale_inv
    )
    w2_weight, w2_scale = convert_pair(experts.w2_weight, experts.w2_weight_scale_inv)

    experts.w13_weight.data = w13_weight
    experts.w2_weight.data = w2_weight
    experts.w13_weight_scale_inv.data = w13_scale
    experts.w2_weight_scale_inv.data = w2_scale
    experts.w13_weight_scale_inv.format_ue8m0 = False
    experts.w2_weight_scale_inv.format_ue8m0 = False
    experts._mega_moe_weights_are_fp4 = True
