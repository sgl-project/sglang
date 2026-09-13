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

import functools
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.dsv4 import mega_moe_pre_dispatch
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.mega_moe_sm90 import (
    is_sm90_fp8_mega_moe_available,
    run_sm90_mega_routed,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.models.deepseek_common.utils import _device_sm
from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


_MEGA_MOE_SYMM_BUFFER: dict = {}


def _mega_moe_mma_type() -> str:
    return "mxf4xmxf4" if get_exec().moe.enable_w4a4_mxfp4_megamoe else "fp8xfp4"


@functools.lru_cache(maxsize=1)
def _mega_moe_max_num_sms() -> Optional[int]:
    if _device_sm < 100:
        # The SM90 MegaMoE implementation does not use the whole-grid clustered
        # launch that needs a residency margin.
        return None

    # Physical count, not deep_gemm.get_num_sms(): two-batch overlap and the DSA
    # indexer reconfigure that process-wide, so reserving on top would compound.
    num_sms = torch.cuda.get_device_properties(device="cuda").multi_processor_count
    reserved_num_sms = max(envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.get(), 0)
    return max(2, num_sms - reserved_num_sms)


@contextmanager
def _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
    max_num_sms = _mega_moe_max_num_sms()
    if max_num_sms is None:
        yield
        return

    current_num_sms = deep_gemm.get_num_sms()
    # Stay under an outer context's budget instead of claiming SMs back from it.
    target_num_sms = min(max_num_sms, current_num_sms)
    # Round down: the clustered launch needs an even CTA count.
    target_num_sms -= target_num_sms % 2
    if target_num_sms == current_num_sms:
        yield
        return

    deep_gemm.set_num_sms(target_num_sms)
    try:
        yield
    finally:
        deep_gemm.set_num_sms(current_num_sms)


def _get_mega_moe_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
) -> SymmBuffer:
    import deep_gemm

    mma_type = _mega_moe_mma_type()
    key = (
        id(group),
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
        num_shared_experts,
        mma_type,
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
            num_shared_experts=num_shared_experts,
            mma_type=mma_type,
            activation="swiglu",
        )
        _MEGA_MOE_SYMM_BUFFER[key] = buf
    return buf


def _can_fuse_native_mega_moe_shared_experts(moe: DeepseekV2MoE) -> bool:
    """Whether DeepGEMM can preserve DSV4's routed-FP4/shared-FP8 split."""
    return bool(
        get_moe_a2a_backend().is_megamoe()
        and _device_sm >= 100
        and getattr(moe, "is_deepseek_v4", False)
        and _mega_moe_mma_type() == "fp8xfp4"
        and moe.num_fused_shared_experts == 0
        and moe.n_shared_experts == 1
        and hasattr(moe, "shared_experts")
        and moe.shared_experts_is_fp8
        and moe.shared_experts_weight_block_size == [128, 128]
    )


def build_mega_moe_shared_expert_weights(
    moe: DeepseekV2MoE, *, force: bool = False
) -> bool:
    """Build DeepGEMM's separate FP8 shared-weight layout without remapping it.

    The original block-FP8 weights stay on ``moe.shared_experts`` for the
    non-MegaMoE fallback. DeepGEMM receives per-32 UE8M0 copies, with L1
    gate/up rows interleaved and both scale tensors transformed for UTCCP.
    """
    if getattr(moe, "_mega_moe_shared_weights_built", False) and not force:
        return True
    if not _can_fuse_native_mega_moe_shared_experts(moe):
        return False

    import deep_gemm

    from sglang.srt.layers.quantization.fp8_utils import requant_weight_ue8m0

    def as_ue8m0_pair(proj):
        scale = proj.weight_scale_inv
        if getattr(scale, "format_ue8m0", False):
            return proj.weight.data, scale.data
        return requant_weight_ue8m0(
            proj.weight.data,
            scale.data,
            moe.shared_experts_weight_block_size,
        )

    l1_pair = as_ue8m0_pair(moe.shared_experts.gate_up_proj)
    l2_pair = as_ue8m0_pair(moe.shared_experts.down_proj)
    shared_l1, shared_l2 = deep_gemm.transform_weights_for_mega_moe(
        l1_pair,
        l2_pair,
        mma_type="fp8xfp4",
    )
    moe.mega_shared_l1_weights = shared_l1
    moe.mega_shared_l2_weights = shared_l2
    moe._mega_moe_shared_weights_built = True
    return True


def _has_native_mega_moe_shared_experts(moe: DeepseekV2MoE) -> bool:
    return bool(getattr(moe, "_mega_moe_shared_weights_built", False))


def should_use_mega_moe(moe: DeepseekV2MoE, hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False
    if _device_sm == 90:
        if not is_sm90_fp8_mega_moe_available(moe.experts):
            return False
    if get_is_capture_mode():
        return True

    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens and not is_dsa_enable_prefill_cp():
        max_tokens_per_rank = max(global_num_tokens)
    else:
        max_tokens_per_rank = hidden_states.shape[0]
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    return max_tokens_per_rank <= cap


def forward_mega_moe(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    input_ids_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    has_native_shared = _has_native_mega_moe_shared_experts(moe)

    sbo_overlap_flag = (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0
        and not has_native_shared
        and num_tokens > 0
        and get_is_capture_mode()
    )

    if has_native_shared:
        shared_output = None
        mega_stream_ctx = nullcontext()
    elif sbo_overlap_flag:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = torch.cuda.stream(moe.alt_stream)
    else:
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = nullcontext()

    with mega_stream_ctx:
        y = _run_mega_routed(
            moe, hidden_states, forward_batch, input_ids_global, num_tokens
        )

    if sbo_overlap_flag:
        current_stream.wait_stream(moe.alt_stream)

    if shared_output is not None:
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
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
    else:
        topk_ids = None
        topk_weights = None

    ep_group = get_moe_ep_group().device_group
    num_experts = moe.experts.num_experts
    top_k = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    has_native_shared = _has_native_mega_moe_shared_experts(moe)
    num_native_shared_experts = moe.n_shared_experts if has_native_shared else 0
    intermediate_size = moe.config.moe_intermediate_size
    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    assert num_tokens <= num_max_tokens_per_rank, (
        f"mega MoE: num_tokens={num_tokens} exceeds cap "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
        f"{num_max_tokens_per_rank}; raise the env var or lower "
        f"--cuda-graph-max-bs-decode / --chunked-prefill-size accordingly"
    )

    buf = _get_mega_moe_symm_buffer(
        ep_group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=top_k,
        hidden=hidden_size,
        intermediate_hidden=intermediate_size,
        num_shared_experts=num_native_shared_experts,
    )

    if num_tokens > 0:
        topk_ids_in = topk_ids.to(torch.int32)
        topk_weights_in = topk_weights.to(torch.float32)
        if (
            has_native_shared
            and not moe.experts.should_fuse_routed_scaling_factor_in_topk
        ):
            topk_weights_in = topk_weights_in * moe.routed_scaling_factor
    else:
        topk_ids_in = hidden_states.new_empty((0, top_k), dtype=torch.int32)
        topk_weights_in = hidden_states.new_empty((0, top_k), dtype=torch.float32)

    if _device_sm == 90:
        return run_sm90_mega_routed(
            moe,
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf,
            num_tokens,
        )

    mma_type = _mega_moe_mma_type()
    if mma_type == "mxf4xmxf4":
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
            mma_type=mma_type,
        )
    else:
        shared_block_m = (
            deep_gemm.get_block_m_for_mega_moe(
                ep_group.size(),
                num_experts,
                buf.num_max_tokens_per_rank,
                num_tokens,
                top_k,
                mma_type,
            )
            if has_native_shared
            else 0
        )
        mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            quant_group_size=32,
            shared_l1_acts_sf=(buf.shared_l1_acts_sf if has_native_shared else None),
            shared_block_m=shared_block_m,
        )

    # Allocate at least one row so y has a non-null CUDA data_ptr;
    # the DeepGEMM tvm-ffi binding rejects nullptr in convert_to_torch_tensor().
    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    swiglu_limit = getattr(moe.config, "swiglu_limit", None)
    with _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
        deep_gemm.fp8_fp4_mega_moe(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            buf,
            shared_l1_weights=(
                moe.mega_shared_l1_weights if has_native_shared else None
            ),
            shared_l2_weights=(
                moe.mega_shared_l2_weights if has_native_shared else None
            ),
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=swiglu_limit,
            fast_math=True,
        )
    y = y[:num_tokens]

    if (
        not has_native_shared
        and not moe.experts.should_fuse_routed_scaling_factor_in_topk
    ):
        y.mul_(moe.routed_scaling_factor)
    return y


def _interleave_mega_moe_gate_up(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # Match DeepGEMM's L1 gate/up layouts. FP8 activations use contiguous
    # gran-8 chunks; packed MXFP4 activations use even/odd gran-16 chunks.
    num_groups, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(num_groups, half // gran, gran, *rest)
    up = t[:, half:].reshape(num_groups, half // gran, gran, *rest)
    if gran == 16:
        result = torch.cat(
            [gate[:, :, 0::2], up[:, :, 0::2], gate[:, :, 1::2], up[:, :, 1::2]],
            dim=2,
        ).reshape(num_groups, n, *rest)
    else:
        result = torch.stack([gate, up], dim=2).reshape(num_groups, n, *rest)
    return torch.empty_like(t).copy_(result)


def _interleave_mega_moe_l1_weights(
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    mma_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    gran = 16 if mma_type == "mxf4xmxf4" else 8
    return (
        _interleave_mega_moe_gate_up(l1_weights[0], gran=gran),
        _interleave_mega_moe_gate_up(l1_weights[1], gran=gran),
    )


def _transpose_mega_moe_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (
        sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
        .transpose(2, 3)
        .reshape(num_groups, mn, packed_sf_k)
    )
    return torch.empty_like(sf).copy_(result)


def build_mega_moe_experts_weights(experts) -> None:
    from deep_gemm import (
        transform_sf_into_required_layout,
    )

    if getattr(experts, "_mega_moe_weights_built", False):
        return

    mma_type = _mega_moe_mma_type()
    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    num_groups, n1, half_k1 = w13.shape
    k1 = half_k1 * 2
    _, n2, half_k2 = w2.shape
    k2 = half_k2 * 2

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

    # Build the interleaved L1 weight + scale once; share the weight buffer
    # between `w13_weight.data` (normal deep-ep path) and `mega_l1_weights[0]`
    # (mega moe path). Mega moe additionally needs a UTCCP-transposed scale;
    # the deep-ep path consumes the non-transposed interleaved scale and a
    # swizzle-aware activation kernel. L2 weight is untouched by the mega
    # transform, so the existing `w2_weight.data` is shared directly.
    w13_interleaved, w13_sf_interleaved = _interleave_mega_moe_l1_weights(
        (w13, w13_sf), mma_type
    )
    w13_sf_utccp = _transpose_mega_moe_sf_for_utccp(w13_sf_interleaved)
    w2_sf_utccp = _transpose_mega_moe_sf_for_utccp(w2_sf)

    experts.w13_weight.data = w13_interleaved
    experts.w13_weight_scale_inv.data = w13_sf_interleaved
    experts.w2_weight_scale_inv.data = w2_sf
    experts.w13_weight_scale_inv.format_ue8m0 = True
    experts.w2_weight_scale_inv.format_ue8m0 = True

    experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)
    experts.mega_l2_weights = (experts.w2_weight.data, w2_sf_utccp)

    experts._mega_moe_weights_built = True
