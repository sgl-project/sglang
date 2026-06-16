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

import os
import logging
import time
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.dsv4 import mega_moe_pre_dispatch
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.server_args import get_global_server_args

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
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    initialized = 0

    ep_group.barrier()
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None or not getattr(experts, "_mega_moe_weights_built", False):
            continue

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
            "local MoE layers with cap=%s.",
            initialized,
            cap,
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
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    ep_group.barrier()
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
            "config with cap=%s, experts=%s, topk=%s, hidden=%s, intermediate=%s.",
            cap,
            num_experts,
            num_topk,
            hidden,
            intermediate_hidden,
        )


def should_use_mega_moe(moe: "DeepseekV2MoE", hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False
    if get_is_capture_mode():
        return True

    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens:
        max_tokens_per_rank = max(global_num_tokens)
    else:
        max_tokens_per_rank = hidden_states.shape[0]
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    return max_tokens_per_rank <= cap


def forward_mega_moe(
    moe: "DeepseekV2MoE",
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
    moe: "DeepseekV2MoE",
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
        server_args = get_global_server_args()
        dispatch_info = (
            ExpertLocationDispatchInfo.init_new(layer_id=moe.layer_id)
            if server_args.enable_eplb
            else None
        )
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            expert_location_dispatch_info=dispatch_info,
            **topk_kwargs,
        )
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
    else:
        topk_ids = None
        topk_weights = None

    mark_timing("topk")

    ep_group = get_moe_ep_group().device_group
    num_experts = moe.experts.num_experts
    top_k = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    intermediate_size = moe.config.moe_intermediate_size
    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    assert num_tokens <= num_max_tokens_per_rank, (
        f"mega MoE: num_tokens={num_tokens} exceeds cap "
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

    use_fp4_acts = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get()
    if use_fp4_acts:
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


def _log_mega_moe_timing(
    moe: "DeepseekV2MoE",
    topk_ids: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    timing_events: list[tuple[str, torch.cuda.Event]],
) -> None:
    if not timing_events:
        return
    timing_events[-1][1].synchronize()
    elapsed = {
        f"{timing_events[i - 1][0]}_to_{timing_events[i][0]}_ms": timing_events[
            i - 1
        ][1].elapsed_time(timing_events[i][1])
        for i in range(1, len(timing_events))
    }
    counts_cpu, ratio = _rank_count_summary(topk_ids, num_experts, moe.moe_ep_size)
    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    logger.info(
        "MEGA_MOE_TIMING ep_rank=%s layer=%s tokens=%s waterfill=%s force_local=%s "
        "counts=%s ratio=%.4f timing=%s",
        get_moe_ep_group().rank_in_group,
        moe.layer_id,
        num_tokens,
        bool(getattr(moe.topk, "enable_deepep_waterfill", False)),
        envs.SGLANG_WATERFILL_FORCE_LOCAL_SHARED.get(),
        counts_cpu,
        ratio,
        {k: round(v, 4) for k, v in elapsed.items()},
    )


def _maybe_log_mega_moe_topk_stats(
    moe: "DeepseekV2MoE",
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

    build_tic = time.perf_counter()
    logger.info(
        "MEGA_MOE_BUILD_WEIGHTS_BEGIN dtype=%s is_fp4_packed=%s "
        "scales_format_ue8m0=%s "
        "w13_shape=%s w13_sf_shape=%s w2_shape=%s w2_sf_shape=%s "
        "k1=%s k2=%s",
        w13.dtype,
        is_fp4_packed,
        scales_format_ue8m0,
        tuple(w13.shape),
        tuple(w13_sf_fp32.shape),
        tuple(w2.shape),
        tuple(w2_sf_fp32.shape),
        k1,
        k2,
    )

    if scales_format_ue8m0 and not is_fp4_packed:
        w13_sf = w13_sf_fp32
        w2_sf = w2_sf_fp32
        logger.info(
            "MEGA_MOE_BUILD_WEIGHTS_STEP reuse_ue8m0_scales "
            "w13_sf_shape=%s w13_sf_dtype=%s w2_sf_shape=%s w2_sf_dtype=%s",
            tuple(w13_sf.shape),
            w13_sf.dtype,
            tuple(w2_sf.shape),
            w2_sf.dtype,
        )
    else:
        tic = time.perf_counter()
        w13_sf = transform_sf_into_required_layout(
            w13_sf_fp32,
            mn=n1,
            k=k1,
            recipe=(1, 32),
            num_groups=num_groups,
            disable_ue8m0_cast=False,
        )
        logger.info(
            "MEGA_MOE_BUILD_WEIGHTS_STEP w13_sf elapsed=%.3fs out_shape=%s dtype=%s",
            time.perf_counter() - tic,
            tuple(w13_sf.shape),
            w13_sf.dtype,
        )
        tic = time.perf_counter()
        w2_sf = transform_sf_into_required_layout(
            w2_sf_fp32,
            mn=n2,
            k=k2,
            recipe=(1, 32),
            num_groups=num_groups,
            disable_ue8m0_cast=False,
        )
        logger.info(
            "MEGA_MOE_BUILD_WEIGHTS_STEP w2_sf elapsed=%.3fs out_shape=%s dtype=%s",
            time.perf_counter() - tic,
            tuple(w2_sf.shape),
            w2_sf.dtype,
        )

    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        # Build the interleaved L1 weight + scale once; share the weight buffer
        # between `w13_weight.data` (normal deep-ep path) and `mega_l1_weights[0]`
        # (mega moe path). Mega moe additionally needs a UTCCP-transposed scale;
        # the deep-ep path consumes the non-transposed interleaved scale and a
        # swizzle-aware activation kernel. L2 weight is untouched by the mega
        # transform, so the existing `w2_weight.data` is shared directly.
        tic = time.perf_counter()
        w13_interleaved, w13_sf_interleaved = _interleave_l1_weights((w13, w13_sf))
        logger.info(
            "MEGA_MOE_BUILD_WEIGHTS_STEP interleave_l1 elapsed=%.3fs "
            "weight_shape=%s sf_shape=%s",
            time.perf_counter() - tic,
            tuple(w13_interleaved.shape),
            tuple(w13_sf_interleaved.shape),
        )
        tic = time.perf_counter()
        w13_sf_utccp = _transpose_sf_for_utccp(w13_sf_interleaved)
        logger.info(
            "MEGA_MOE_BUILD_WEIGHTS_STEP transpose_l1_sf elapsed=%.3fs out_shape=%s",
            time.perf_counter() - tic,
            tuple(w13_sf_utccp.shape),
        )
        tic = time.perf_counter()
        w2_sf_utccp = _transpose_sf_for_utccp(w2_sf)
        logger.info(
            "MEGA_MOE_BUILD_WEIGHTS_STEP transpose_l2_sf elapsed=%.3fs out_shape=%s",
            time.perf_counter() - tic,
            tuple(w2_sf_utccp.shape),
        )

        experts.w13_weight.data = w13_interleaved
        experts.w13_weight_scale_inv.data = w13_sf_interleaved
        experts.w2_weight_scale_inv.data = w2_sf
        experts.w13_weight_scale_inv.format_ue8m0 = True
        experts.w2_weight_scale_inv.format_ue8m0 = True

        experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)
        experts.mega_l2_weights = (experts.w2_weight.data, w2_sf_utccp)
    else:
        tic = time.perf_counter()
        l1_pair, l2_pair = transform_weights_for_mega_moe((w13, w13_sf), (w2, w2_sf))
        logger.info(
            "MEGA_MOE_BUILD_WEIGHTS_STEP transform_weights elapsed=%.3fs",
            time.perf_counter() - tic,
        )

        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_weights_built = True
    logger.info(
        "MEGA_MOE_BUILD_WEIGHTS_DONE elapsed=%.3fs",
        time.perf_counter() - build_tic,
    )


def convert_fp8_experts_to_fp4_for_mega_moe(experts, weight_block_size) -> None:
    """Convert block-FP8 checkpoint expert weights to DeepGEMM Mega-MoE FP4."""
    from deep_gemm.utils import per_token_cast_to_fp4

    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

    if getattr(experts, "_mega_moe_weights_are_fp4", False):
        return
    assert weight_block_size == [128, 128], weight_block_size

    def convert_pair(weight: torch.nn.Parameter, scale: torch.nn.Parameter):
        tic = time.perf_counter()
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
        logger.info(
            "MEGA_MOE_CONVERT_FP8_TO_FP4_STEP in_weight_shape=%s "
            "out_weight_shape=%s out_scale_shape=%s elapsed=%.3fs",
            tuple(weight.shape),
            tuple(fp4_weight.shape),
            tuple(fp4_scale.shape),
            time.perf_counter() - tic,
        )
        return fp4_weight, fp4_scale

    convert_tic = time.perf_counter()
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
    logger.info(
        "MEGA_MOE_CONVERT_FP8_TO_FP4_DONE elapsed=%.3fs",
        time.perf_counter() - convert_tic,
    )
