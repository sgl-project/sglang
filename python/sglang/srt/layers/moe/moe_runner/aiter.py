from __future__ import annotations

import functools
import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var, get_int_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.base import CombineInput
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        MoriEPLLDispatchOutput,
        MoriEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


logger = logging.getLogger(__name__)


class AiterQuantType(str, Enum):
    NONE = "No"
    PER_TOKEN = "per_Token"
    PER_128X128 = "per_128x128"
    PER_1X32 = "per_1x32"


@dataclass
class AiterMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    quant_type: AiterQuantType = AiterQuantType.NONE
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    expert_mask: Optional[torch.Tensor] = None
    doweight_stage1: bool = False
    hidden_pad: int = 0
    intermediate_pad: int = 0
    swiglu_limit: float = 0.0
    fused_moe_kwargs: Optional[dict[str, Any]] = None


@dataclass
class AiterRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor  # int32
    topk_weights: torch.Tensor  # float32
    # Effective activation quant_type (may differ from quant_info.quant_type
    # after the dispatch-aware decision in mori pre_permute).
    quant_type: AiterQuantType
    # Per-token activation scale produced by an EP dispatcher (mori). Falls
    # back to quant_info.a13_scale when None.
    a1_scale: Optional[torch.Tensor] = None
    # Mori-only fused_moe kwargs.
    num_local_tokens: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


@dataclass
class AiterRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


_AITER_ACTIVATIONS = {
    "silu": "Silu",
    "swiglu": "Swiglu",
    "situ": "Situv2",
}


def _aiter_activation(activation: str):
    from aiter import ActivationType

    return getattr(ActivationType, _AITER_ACTIVATIONS.get(activation, "Gelu"))


def _aiter_quant_type(quant_type: AiterQuantType):
    from aiter import QuantType

    return getattr(QuantType, quant_type.value)


@functools.cache
def _aiter_fused_moe_supports_no_combine() -> bool:
    """Probe whether the installed aiter.fused_moe accepts a `no_combine` kwarg.

    Older wheels don't expose it, so feature-detect once and forward
    conditionally, matching the existing `**extra` conditional-kwarg pattern
    used for `num_local_tokens` / `dtype`.
    """
    from aiter.fused_moe import fused_moe

    return "no_combine" in inspect.signature(fused_moe).parameters


_RECV_BOUND_LOGGED: set[int] = set()
_RECV_BOUND_WARNED = False


def _warn_recv_bound_unavailable() -> None:
    global _RECV_BOUND_WARNED
    if not _RECV_BOUND_WARNED:
        _RECV_BOUND_WARNED = True
        logger.warning(
            "SGLANG_MORI_RECV_BOUND is set but the per-rank DP token counts do "
            "not cover every mori sender, so the receive fan-in is unknown; "
            "leaving the receive buffer unbounded."
        )


def _mori_decode_recv_bound(recv_rows: int, topk: int) -> int:
    """Live rows mori's receive buffer can hold in decode, or 0 for "do not bound".

    Worst case fan-in is every rank routing all of its tokens to this one, so
    `sum(per-rank tokens) * topk`, where topk already includes the fused shared
    expert. The per-rank counts come from the DP sync, so this is the fan-in for
    the batch actually being run rather than an upper bound over all batches.

    That is only sound because enabling this gate also makes
    `require_mlp_tp_gather()` true for mori, which gives every rank the same
    cuda-graph bucket. The value is baked into a captured graph and has to hold
    for every later replay; with per-rank buckets a rank on a narrow tier could
    be handed rows by a peer on a wider one, and the only bound valid under that
    is the widest tier's -- 4-16x looser than the batch being run, which costs
    more in expert-GEMM tiles (M 32/64 -> 128) than the trim saves.

    Two cases stay unbounded, because a bound below the real fan-in silently
    drops rows from the all-to-all -- wrong output rather than an error:

    * Prefill, whose per-rank counts are uneven and not knowable here.
    * Anything that leaves the per-rank counts unpopulated, or where the EP world
      is wider than the DP world so the counts do not cover every sender.
    """
    if not get_bool_env_var("SGLANG_MORI_RECV_BOUND", "false"):
        return 0

    from sglang.srt.layers.dp_attention import (
        get_dp_global_num_tokens,
        get_is_extend_in_batch,
    )

    if get_is_extend_in_batch():
        return 0

    per_rank_tokens = get_dp_global_num_tokens()
    ep_size = get_parallel().moe_ep_size
    if not per_rank_tokens or len(per_rank_tokens) < ep_size:
        # Either the DP sync did not publish counts, or they do not cover every
        # mori sender. Both mean the fan-in is unknown here.
        _warn_recv_bound_unavailable()
        return 0

    max_tokens = sum(per_rank_tokens)
    bound = max_tokens * topk
    # Never grow the tensor, and nothing to do when there is nothing to trim.
    if not 0 < bound < recv_rows:
        return 0

    # One INFO line the first time it engages, so an inert bound is not mistaken
    # for an active one in the results. Per-tier values go to DEBUG: capture
    # visits every tier, and at INFO on every rank that is dozens of lines.
    if get_parallel().tp_rank == 0 and bound not in _RECV_BOUND_LOGGED:
        first = not _RECV_BOUND_LOGGED
        _RECV_BOUND_LOGGED.add(bound)
        if first:
            logger.info(
                "mori recv bound active: %d rows -> %d for this tier "
                "(dp_tokens=%d ep=%d topk=%d); per-tier values at DEBUG",
                recv_rows,
                bound,
                max_tokens,
                get_parallel().moe_ep_size,
                topk,
            )
        else:
            logger.debug(
                "mori recv bound: %d rows -> %d (dp_tokens=%d ep=%d topk=%d)",
                recv_rows,
                bound,
                max_tokens,
                get_parallel().moe_ep_size,
                topk,
            )
    return bound


class AiterRunnerCore(MoeRunnerCore):
    def run(
        self,
        runner_input: AiterRunnerInput,
        quant_info: AiterMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> AiterRunnerOutput:
        if self.config.no_combine and not _aiter_fused_moe_supports_no_combine():
            raise NotImplementedError(
                "no_combine=True requested but the installed aiter.fused_moe does "
                "not accept a `no_combine` kwarg. Install an aiter build that "
                "supports fused_moe no_combine output."
            )

        if runner_input.hidden_states.shape[0] == 0:
            if self.config.no_combine:
                topk = runner_input.topk_ids.shape[-1]
                hidden_size = runner_input.hidden_states.shape[-1]
                return AiterRunnerOutput(
                    hidden_states=runner_input.hidden_states.new_empty(
                        (0, topk, hidden_size)
                    )
                )
            return AiterRunnerOutput(hidden_states=runner_input.hidden_states)

        from aiter.fused_moe import fused_moe

        from sglang.srt.environ import envs

        a1_scale = (
            runner_input.a1_scale
            if runner_input.a1_scale is not None
            else quant_info.a13_scale
        )

        extra: dict = {}
        if quant_info.fused_moe_kwargs:
            extra.update(quant_info.fused_moe_kwargs)
        if runner_input.num_local_tokens is not None:
            extra["num_local_tokens"] = runner_input.num_local_tokens
        if runner_input.output_dtype is not None:
            extra["dtype"] = runner_input.output_dtype
        if self.config.activation == "situ":
            from aiter.ops.flydsl.moe_common import GateMode

            extra["gate_mode"] = GateMode.SEPARATED.value
            if self.config.gemm1_alpha is not None:
                extra["beta"] = float(self.config.gemm1_alpha)
            if self.config.gemm1_clamp_limit is not None:
                extra["linear_beta"] = float(self.config.gemm1_clamp_limit)
        elif quant_info.swiglu_limit > 0:
            # GateMode is only needed for the gpt-oss MXFP4 swiglu_limit path.
            # Import lazily so models that don't use it (e.g. DeepSeek-V3 fp8,
            # swiglu_limit==0) still run on aiter builds where this module
            # lives elsewhere / is absent.
            from aiter.ops.flydsl.moe_common import GateMode

            # Default (INTERLEAVE) preserves the pre-fix behavior for paths
            # that prepare weights in the gate/up-interleaved layout. Set
            # `SGLANG_USE_AITER_MOE_GU_ITLV=0` to switch to SEPARATED, which
            # matches the layout produced by `Mxfp4MoEMethod` (gpt-oss
            # MXFP4) and the gptoss_fp4 tuned FlyDSL kernels.
            extra["gate_mode"] = (
                GateMode.INTERLEAVE.value
                if envs.SGLANG_USE_AITER_MOE_GU_ITLV.get()
                else GateMode.SEPARATED.value
            )
            extra["swiglu_limit"] = quant_info.swiglu_limit
        if self.config.no_combine:
            extra["no_combine"] = True

        output = fused_moe(
            hidden_states=runner_input.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_weight=runner_input.topk_weights,
            topk_ids=runner_input.topk_ids,
            quant_type=_aiter_quant_type(runner_input.quant_type),
            activation=_aiter_activation(self.config.activation),
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            a1_scale=a1_scale,
            a2_scale=quant_info.a2_scale,
            bias1=quant_info.b13,
            bias2=quant_info.b2,
            expert_mask=quant_info.expert_mask,
            doweight_stage1=quant_info.doweight_stage1,
            hidden_pad=quant_info.hidden_pad,
            intermediate_pad=quant_info.intermediate_pad,
            **extra,
        )
        return AiterRunnerOutput(hidden_states=output)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


# ---------------------------------------------------------------------------
# Pre-permute: dispatch_output -> AiterRunnerInput
# ---------------------------------------------------------------------------


@register_pre_permute("standard", "aiter")
def pre_permute_standard_to_aiter(
    dispatch_output: StandardDispatchOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AiterRunnerInput:
    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(torch.float32)

    if runner_config.apply_router_weight_on_input and not quant_info.doweight_stage1:
        # Pre-scale at the Python level for kernels that don't honor doweight_stage1.
        assert (
            topk_weights.dim() == 2 and topk_weights.shape[-1] == 1
        ), "apply_router_weight_on_input requires topk=1"
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        topk_weights = torch.ones_like(topk_weights)

    return AiterRunnerInput(
        hidden_states=hidden_states,
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights,
        quant_type=quant_info.quant_type,
    )


def _is_mori_dispatch_output(dispatch_output: Any) -> bool:
    # MoriEP{Normal,LL}DispatchOutput carry the post-mori-permute origin_topk_*
    # tensors that the standard DeepEP outputs lack.
    return hasattr(dispatch_output, "origin_topk_ids")


def _resolve_mori_quant_type(
    dispatch_a1_dtype: torch.dtype,
    dispatch_scale: Optional[torch.Tensor],
    weight_quant: AiterQuantType,
) -> AiterQuantType:
    """Pick the activation quant_type for AITER when the dispatch path may have
    pre-quantized hidden_states. Mirrors the original MoriEPMoE.run_moe_core
    decision tree."""
    is_fp8_quant = weight_quant in (
        AiterQuantType.PER_128X128,
        AiterQuantType.PER_TOKEN,
    )
    is_w4a4 = weight_quant == AiterQuantType.PER_1X32
    is_fp4_dispatch = dispatch_a1_dtype == torch.float4_e2m1fn_x2
    has_dispatch_scale = dispatch_scale is not None

    if is_w4a4:
        # W4A4 weights always run as per_1x32; FP8 dispatch is upscaled to BF16
        # before this point so dispatch_scale won't conflict.
        return AiterQuantType.PER_1X32
    if is_fp8_quant:
        return weight_quant
    # BF16 weights: lift to the dispatch-side quant type when scales are provided.
    if has_dispatch_scale and is_fp4_dispatch:
        return AiterQuantType.PER_1X32
    if has_dispatch_scale and not is_fp4_dispatch:
        return AiterQuantType.PER_128X128
    return AiterQuantType.NONE


def _pre_permute_deepep_to_aiter(
    dispatch_output: Union[
        DeepEPNormalDispatchOutput,
        DeepEPLLDispatchOutput,
        MoriEPNormalDispatchOutput,
        MoriEPLLDispatchOutput,
    ],
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AiterRunnerInput:
    is_mori = _is_mori_dispatch_output(dispatch_output)

    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_ids.to(torch.int32)
    topk_weights = dispatch_output.topk_weights.to(torch.float32)
    a1_scale: Optional[torch.Tensor] = None
    num_local_tokens: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None
    quant_type = quant_info.quant_type

    if is_mori:
        from sglang.kernels.ops.moe.rocm_moe_utils import upscale, upscale_mxfp4

        a1_scale = dispatch_output.hidden_states_scale
        num_local_tokens = dispatch_output.num_recv_tokens_per_expert
        output_dtype = dispatch_output.out_dtype

        # Truncate dispatch tensors to the configured cap; mori combine only
        # reads [0, totalRecvTokenNum), so the truncated result needs no
        # padding back.
        mori_max = get_int_env_var("SGLANG_MORI_MOE_MAX_INPUT_TOKENS", 0)
        if mori_max <= 0:
            mori_max = _mori_decode_recv_bound(
                hidden_states.shape[0], topk_ids.shape[-1]
            )
        if mori_max > 0:
            hidden_states = hidden_states[:mori_max]
            if a1_scale is not None:
                a1_scale = a1_scale[:mori_max]
            topk_ids = topk_ids[:mori_max]
            topk_weights = topk_weights[:mori_max]

        # Upscale dispatched activations when there is no AITER kernel for the
        # weight/activation dtype pair.
        weight_quant = quant_info.quant_type
        is_fp8_quant = weight_quant in (
            AiterQuantType.PER_128X128,
            AiterQuantType.PER_TOKEN,
        )
        is_w4a4 = weight_quant == AiterQuantType.PER_1X32
        is_fp4_dispatch = hidden_states.dtype == torch.float4_e2m1fn_x2

        # AITER fused_moe Clamped-SwiGLU is dispatched with
        # gate_mode=INTERLEAVE, for which AITER picks a bf16/fp8 `q_dtype_a`
        # Refer to https://github.com/ROCm/aiter/blob/a2617c366dc7271a1662ecda2023d19f6ccefcec/aiter/fused_moe.py#L406-L412
        swiglu_interleave = quant_info.swiglu_limit > 0 and get_bool_env_var(
            "SGLANG_USE_AITER_MOE_GU_ITLV", "true"
        )

        # MXFP8 dispatch already carries fp8 data with group-32 e8m0 scales,
        # which is what per_1x32 wants, so hand it straight to fused_moe. Only
        # fp8 dispatch's group-128/fp32 scales need the dequant round trip; it is
        # distinguishable by scale dtype (fp32 there, e8m0 here).
        is_mx_fp8_dispatch = (
            a1_scale is not None
            and a1_scale.dtype == torch.float8_e8m0fnu
            and not is_fp4_dispatch
        )
        if (
            is_w4a4
            and a1_scale is not None
            and not is_fp4_dispatch
            and not is_mx_fp8_dispatch
        ):
            # W4A4 weights with FP8 dispatch: dequant FP8->BF16 first; the
            # FP4 per_1x32 path needs BF16 input.
            hidden_states = upscale(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None
        elif is_w4a4 and is_fp4_dispatch and a1_scale is not None and swiglu_interleave:
            # W4A4 weights + FP4 dispatch on the clamped-SwiGLU/INTERLEAVE
            # path: AITER expects a bf16/fp8 activation here, not fp4x2.
            # Dequant FP4->BF16 and let fused_moe re-quantize internally.
            hidden_states = upscale_mxfp4(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None
        elif is_fp8_quant and is_fp4_dispatch and a1_scale is not None:
            # FP8 weights + FP4 dispatch: no kernel for the fp4x2/fp8 pair;
            # dequant FP4->BF16 and let fused_moe re-quantize to FP8.
            hidden_states = upscale_mxfp4(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None

        quant_type = _resolve_mori_quant_type(
            hidden_states.dtype, a1_scale, weight_quant
        )

        running_state["aiter_combine_topk_ids"] = dispatch_output.origin_topk_ids
        running_state["aiter_combine_topk_weights"] = (
            dispatch_output.origin_topk_weights
        )
    else:
        # DeepEP marks invalid topk slots with idx == -1; AITER cannot accept
        # negative ids, so reroute them to the sink slot at index
        # num_local_experts (masked off by quant_info.expert_mask which has
        # shape (num_local_experts + 1,)).
        topk_ids = torch.where(
            topk_ids == -1,
            torch.full_like(topk_ids, runner_config.num_local_experts),
            topk_ids,
        )
        running_state["aiter_combine_topk_ids"] = dispatch_output.topk_ids
        running_state["aiter_combine_topk_weights"] = dispatch_output.topk_weights

    running_state["aiter_combine_is_mori"] = is_mori

    return AiterRunnerInput(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        quant_type=quant_type,
        a1_scale=a1_scale,
        num_local_tokens=num_local_tokens,
        output_dtype=output_dtype,
    )


register_pre_permute("deepep_normal", "aiter")(_pre_permute_deepep_to_aiter)
register_pre_permute("deepep_ll", "aiter")(_pre_permute_deepep_to_aiter)


# ---------------------------------------------------------------------------
# Post-permute: AiterRunnerOutput -> CombineInput
# ---------------------------------------------------------------------------


@register_post_permute("aiter", "standard")
def post_permute_aiter_to_standard(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)


def _post_permute_aiter_to_deepep(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
    is_normal: bool,
) -> CombineInput:
    if running_state.get("aiter_combine_is_mori"):
        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            MoriEPLLCombineInput,
            MoriEPNormalCombineInput,
        )

        cls = MoriEPNormalCombineInput if is_normal else MoriEPLLCombineInput
    else:
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPLLCombineInput,
            DeepEPNormalCombineInput,
        )

        cls = DeepEPNormalCombineInput if is_normal else DeepEPLLCombineInput

    return cls(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["aiter_combine_topk_ids"],
        topk_weights=running_state["aiter_combine_topk_weights"],
    )


@register_post_permute("aiter", "deepep_normal")
def post_permute_aiter_to_deepep_normal(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> CombineInput:
    return _post_permute_aiter_to_deepep(
        runner_output, quant_info, runner_config, running_state, is_normal=True
    )


@register_post_permute("aiter", "deepep_ll")
def post_permute_aiter_to_deepep_ll(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> CombineInput:
    return _post_permute_aiter_to_deepep(
        runner_output, quant_info, runner_config, running_state, is_normal=False
    )
