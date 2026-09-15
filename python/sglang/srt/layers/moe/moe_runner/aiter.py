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
class AiterFusedRouterInput:
    """Routing left to the fused preamble: the logits, not the selection.

    `topk_output` is the untouched BypassedTopKOutput, kept so the runner can
    materialize routing via `to_standard()` if the per-call check refuses the fused
    path. `topk` is the ROUTED width only: sglang's `top_k` includes the fused shared
    experts, while the aiter entry takes the two separately.
    """

    router_logits: torch.Tensor
    correction_bias: torch.Tensor
    topk_output: Any
    topk: int
    num_fused_shared_experts: int
    num_expert_group: int
    topk_group: int
    renormalize: bool
    routed_scaling_factor: float
    shared_expert_weight: float


@dataclass
class AiterRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    # None only on the fused-router path, which has no precomputed selection.
    topk_ids: Optional[torch.Tensor]  # int32
    topk_weights: Optional[torch.Tensor]  # float32
    # Effective activation quant_type (may differ from quant_info.quant_type
    # after the dispatch-aware decision in mori pre_permute).
    quant_type: AiterQuantType
    # Per-token activation scale produced by an EP dispatcher (mori). Falls
    # back to quant_info.a13_scale when None.
    a1_scale: Optional[torch.Tensor] = None
    # Set when TopK bypassed selection and the fused preamble will do the routing.
    fused_router: Optional[AiterFusedRouterInput] = None
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


# The fused routing preamble is validated at DECODE token counts only: the tuned rows
# and the op-level sweep cover M in {1, 2, 4, 8, 16}. Above this the stage-by-stage chain
# runs, so prefill's chunk-sized router calls deliberately keep the incumbent path.
#
# The bound has to live here because aiter's own predicate cannot express it: a tuned-row
# miss does not refuse, it falls through to default heuristics, so an untuned token count
# can still be reported supported. One named constant tied to the validated set, in the
# style the kernel side asks for -- no environment variable, no model-name dispatch.
# Raising it is the same procedure as introducing it: run the tuner at the new shapes,
# validate, then raise the constant.
#
# The widths this caller is verified at, and the cap derived from them.
#
# Verification is per WIDTH, not per range: the gate below is a maximum, and the entry's
# own predicate cannot narrow it -- a tuned-metadata miss there falls through to default
# heuristics instead of refusing -- so every width at or below the cap reaches the fused
# kernel and every one of them has to have been checked.
#
# Checked on routing captured from a running server. The final MoE output is byte-identical
# to the unfused chain at every width. Where the intermediate slot buffers differ, the
# difference is a row permutation within an expert and nothing else: the permutation was
# reconstructed from the two paths' own slot buffers and applied, and it reproduces the
# other path's bytes exactly -- 0 of 152,064 e8m0 bytes differ under it. The one remaining
# difference is a single pick of 144 at a bit-exact score tie, where the two paths select
# among candidates with identical weights.
FUSED_MOE_ROUTER_VALIDATED_TOKENS = (4, 8, 12, 16, 20, 24, 28, 32)
AITER_FUSED_ROUTER_MAX_VALIDATED_TOKENS = max(FUSED_MOE_ROUTER_VALIDATED_TOKENS)


@dataclass(frozen=True)
class _FusedRouterEntry:
    """The fused routing preamble's entry points, resolved once per process."""

    call: Any
    supported: Any
    config_supported: Any
    hidden_dims: tuple
    max_topk: int
    max_experts: int
    max_tokens: int


@functools.cache
def _aiter_fused_router() -> Optional[_FusedRouterEntry]:
    """The installed aiter's fused MoE routing preamble, or None if it has none.

    ROCm/aiter#5150 folds the router's score/top-k/slot/sort/quantize chain into one
    kernel. Most builds do not carry it, so every use here is feature-detected: absent
    or inapplicable, the caller takes the ordinary stage-by-stage path. The detection is
    the only gate -- there is no model-name dispatch and no environment threshold.
    """
    try:
        from aiter.fused_moe import (
            FUSED_MOE_ROUTER_HIDDEN_DIMS,
            FUSED_MOE_ROUTER_MAX_EXPERTS,
            FUSED_MOE_ROUTER_MAX_TOKENS,
            FUSED_MOE_ROUTER_MAX_TOPK,
            fused_moe_router,
            fused_moe_router_arch_supported,
            fused_moe_router_config_supported,
            fused_moe_router_supported,
        )
    except ImportError:
        return None
    try:
        if not fused_moe_router_arch_supported():
            return None
    except Exception:  # a probe must never be the reason a server fails to start
        logger.debug("aiter fused router arch probe failed", exc_info=True)
        return None
    return _FusedRouterEntry(
        call=fused_moe_router,
        supported=fused_moe_router_supported,
        config_supported=fused_moe_router_config_supported,
        hidden_dims=tuple(FUSED_MOE_ROUTER_HIDDEN_DIMS),
        max_topk=int(FUSED_MOE_ROUTER_MAX_TOPK),
        max_experts=int(FUSED_MOE_ROUTER_MAX_EXPERTS),
        max_tokens=int(FUSED_MOE_ROUTER_MAX_TOKENS),
    )


def _aiter_enabled() -> bool:
    """Whether this deployment actually resolves MoE work to the aiter runner.

    `auto` is not sufficient on its own: the quantization method consults this flag when
    it creates the runner, so with it off the same configuration resolves to Triton.
    """
    return get_bool_env_var("SGLANG_USE_AITER")


def fused_router_can_bypass_topk(
    topk_config, hidden_dim: int, num_experts: int
) -> bool:
    """Whether TopK may hand this MoE its router logits instead of computing top-k.

    Deliberately coarse: it reads only static configuration and the hidden dim, so the
    answer cannot change between CUDA-graph capture and replay. The authoritative
    per-call check (`fused_moe_router_supported`) needs the tensors and runs in the
    runner, where a negative answer costs only the ordinary path via `to_standard()`.
    """
    entry = _aiter_fused_router()
    if entry is None:
        return False
    from sglang.srt.layers.moe.utils import (
        MoeA2ABackend,
        get_moe_a2a_backend,
        get_moe_runner_backend,
    )

    # The bypassed output has a different arity from the standard one, so only the aiter
    # runner can read it. `auto` does not establish that this layer resolves to that
    # runner: with SGLANG_USE_AITER off, an MXFP4 MoE resolves to Triton even on gfx950
    # with the entry installed, and Triton's three-field unpack raises on the bypassed
    # output. The runner's own fallback cannot save that, because the aiter runner was
    # never selected -- and because format selection happens before the token cap is
    # checked, it would fail above the cap too.
    backend = get_moe_runner_backend()
    if not (backend.is_aiter() or (backend.value == "auto" and _aiter_enabled())):
        return False
    # FusedMoE dispatches *before* the runner, so an all-to-all dispatcher reads routing
    # that a bypassed output has not produced: MORI raises on its first routing read,
    # ahead of any fallback. Expert parallelism fails differently and more quietly -- the
    # standard dispatcher's first call initialises the local expert mapping and the aiter
    # expert mask from a standard top-k result, and materialising top-k later does not
    # revisit dispatch, so the mask stays unset. Both stay on the ordinary path until the
    # dispatchers support deferred routing.
    if get_moe_a2a_backend() is not MoeA2ABackend.NONE:
        return False
    if get_parallel().moe_ep_size != 1:
        return False
    # Standard routing remaps logical expert ids onto physical slots. The fused entry
    # does not, so under a nontrivial placement -- initial placement or EPLB -- it would
    # pair logical-order logits with physically placed weights and silently address the
    # wrong expert's weights. This is the one failure here that produces a wrong result
    # rather than an exception.
    if topk_config.expert_location_dispatch_info is not None:
        return False
    if hidden_dim not in entry.hidden_dims or num_experts > entry.max_experts:
        return False
    # The kernel selects with sigmoid+bias and weights with sigmoid alone, so a softmax
    # router or a router without a correction bias is a different function.
    if topk_config.scoring_func != "sigmoid" or topk_config.correction_bias is None:
        return False
    # Grouped routing collapses to flat biased top-k only at one group; above that the
    # kernel would compute a different selection.
    if (topk_config.num_expert_group or 1) != 1 or (topk_config.topk_group or 1) != 1:
        return False
    if topk_config.custom_routing_function is not None:
        return False
    # The entry folds routed_scaling_factor into the top-k weights itself. A caller that
    # wants it applied to the MoE output instead would have it applied twice.
    if topk_config.apply_routed_scaling_factor_on_output:
        return False
    if topk_config.top_k > entry.max_topk:
        return False
    return True


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

        if runner_input.fused_router is not None and (
            runner_input.hidden_states.shape[0] > 0
        ):
            out = self._run_fused_router(runner_input, quant_info)
            if out is not None:
                return AiterRunnerOutput(hidden_states=out)
            # Refused this call: materialize the selection and take the stage chain.
            runner_input = _materialize_bypassed(runner_input, self.config)

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

        assert runner_input.topk_ids is not None, (
            "the stage-by-stage path needs a materialized selection"
        )
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

    def _run_fused_router(
        self,
        runner_input: AiterRunnerInput,
        quant_info: AiterMoeQuantInfo,
    ) -> Optional[torch.Tensor]:
        """Run the fused routing preamble, or return None to fall back.

        Returning None rather than raising is deliberate: the per-call limits (token
        count, expert-mask width) are only knowable here, and the ordinary path is
        always a correct answer.
        """
        from sglang.srt.environ import envs

        entry = _aiter_fused_router()
        if entry is None:
            return None
        fr = runner_input.fused_router
        hidden_states = runner_input.hidden_states
        tokens = hidden_states.shape[0]

        if self.config.no_combine or self.config.apply_router_weight_on_input:
            _census("fallback: no_combine/router-weight-on-input", tokens)
            return None
        if tokens > min(AITER_FUSED_ROUTER_MAX_VALIDATED_TOKENS, entry.max_tokens):
            # Outside the validated envelope (prefill, or a large decode batch).
            _census("fallback: above the validated envelope", tokens)
            return None
        if quant_info.b13 is not None or quant_info.b2 is not None:
            # The entry does not fuse a swiglu bias.
            _census("fallback: swiglu bias", tokens)
            return None

        kwargs = dict(
            hidden_states=hidden_states,
            gating_output=fr.router_logits,
            correction_bias=fr.correction_bias,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk=fr.topk,
            num_expert_group=fr.num_expert_group,
            topk_group=fr.topk_group,
            need_renorm=fr.renormalize,
            # Applied INSIDE the kernel. Nothing downstream may scale again.
            routed_scaling_factor=fr.routed_scaling_factor,
            expert_mask=quant_info.expert_mask,
            activation=_aiter_activation(self.config.activation),
            quant_type=_aiter_quant_type(runner_input.quant_type),
            doweight_stage1=quant_info.doweight_stage1,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            a1_scale=(
                runner_input.a1_scale
                if runner_input.a1_scale is not None
                else quant_info.a13_scale
            ),
            a2_scale=quant_info.a2_scale,
            hidden_pad=quant_info.hidden_pad,
            intermediate_pad=quant_info.intermediate_pad,
            num_fused_shared_experts=fr.num_fused_shared_experts,
            shared_expert_weight=fr.shared_expert_weight,
        )
        # `supported` takes a narrower surface than the call itself, and a future build
        # may narrow or widen it again, so pass only what its signature names rather
        # than the call's whole kwargs. Getting this wrong is silent: it raises
        # TypeError, which reads as "unsupported" and would disable the fused path
        # everywhere without a single line in the log.
        try:
            probe_params = set(inspect.signature(entry.supported).parameters)
        except (TypeError, ValueError):
            _census("fallback: entry support predicate not introspectable", tokens)
            return None
        probe_kwargs = {k: v for k, v in kwargs.items() if k in probe_params}
        try:
            if not entry.supported(**probe_kwargs):
                _census("fallback: entry refused this call", tokens)
                return None
        except TypeError:
            # A build whose predicate takes a different surface than the frozen one:
            # treat as unsupported rather than guessing, but say so.
            _census("fallback: entry support predicate rejected our arguments", tokens)
            logger.debug(
                "aiter fused router support probe rejected kwargs", exc_info=True
            )
            return None

        _census("fused entry invoked", tokens)
        fused_out = entry.call(**kwargs)
        if envs.SGLANG_AITER_FUSED_ROUTER_VERIFY.get():
            self._assert_fused_router_equivalent(runner_input, quant_info, fused_out)
        return fused_out

    def _assert_fused_router_equivalent(
        self,
        runner_input: AiterRunnerInput,
        quant_info: AiterMoeQuantInfo,
        fused_out: torch.Tensor,
    ) -> None:
        """Refuse to serve a mis-wired fused router, once per layer.

        The entry documents itself as producing the same result as
        `biased_grouped_topk` -> `fused_moe_`, so that is what is compared. It returns
        only the MoE output, not routing tensors, so routing equality is asserted
        through the output it determines -- which is the point: the three ways a caller
        can get this wrong (double-applying routed_scaling_factor, passing top_k with
        the shared experts still in it, narrowing fp32 logits to bf16) all change the
        output, and all of them would otherwise surface as an acceptance regression
        rather than as an error.
        """
        layer_id = getattr(self.config, "layer_id", None)
        if layer_id in _FUSED_ROUTER_VERIFIED:
            return
        _FUSED_ROUTER_VERIFIED.add(layer_id)

        reference_input = _materialize_bypassed(runner_input, self.config)
        reference_out = self.run(
            reference_input, quant_info, running_state={}
        ).hidden_states

        fused_f = fused_out.float()
        ref_f = reference_out.float()
        if fused_f.shape != ref_f.shape:
            raise AssertionError(
                "aiter fused router equivalence FAILED on shape: "
                f"fused {tuple(fused_f.shape)} vs stage chain {tuple(ref_f.shape)}"
            )
        denom = ref_f.abs().amax().clamp_min(1e-6)
        max_abs = (fused_f - ref_f).abs().amax()
        max_rel = (max_abs / denom).item()
        # The two paths differ in reduction order and by a sigmoid ulp, so they are
        # close rather than bitwise; a routing difference is orders of magnitude larger
        # than that, which is what this threshold separates.
        if max_rel > 2e-2:
            raise AssertionError(
                "aiter fused router equivalence FAILED: max relative output deviation "
                f"{max_rel:.3e} from the stage-by-stage chain (layer {layer_id}). "
                "Suspect, in order: routed_scaling_factor applied twice, top_k passed "
                "with the fused shared experts still included, or bf16 router logits."
            )
        logger.info(
            "aiter fused router equivalence PASS (layer %s): max relative output "
            "deviation %.3e vs the stage-by-stage chain",
            layer_id,
            max_rel,
        )

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


# ---------------------------------------------------------------------------
# Pre-permute: dispatch_output -> AiterRunnerInput
# ---------------------------------------------------------------------------


# One line the first time each routing decision is taken, so an operator can see which
# path is actually serving instead of inferring it from timing. Counts are kept for the
# same reason: "fused at decode, stage chain at prefill" is a claim that should be
# readable off a log, not assumed from a code path.
_FUSED_ROUTER_CENSUS: dict[str, int] = {}


def _census(event: str, tokens: int) -> None:
    if event not in _FUSED_ROUTER_CENSUS:
        _FUSED_ROUTER_CENSUS[event] = 0
        logger.info("aiter fused router census: first %s at M=%d", event, tokens)
    _FUSED_ROUTER_CENSUS[event] += 1


# Layers whose fused router has already been checked against the stage chain. The
# check is per layer and once: it runs both paths, so it is a gate/debug cost.
_FUSED_ROUTER_VERIFIED: set = set()


def _materialize_bypassed(
    runner_input: AiterRunnerInput,
    config: MoeRunnerConfig,
) -> AiterRunnerInput:
    """Turn a fused-router input back into an ordinary one by running the selection."""
    fr = runner_input.fused_router
    assert fr is not None
    standard = fr.topk_output.to_standard(layer_id=config.layer_id)
    topk_weights, topk_ids, _ = standard
    return AiterRunnerInput(
        hidden_states=runner_input.hidden_states,
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights.to(torch.float32),
        quant_type=runner_input.quant_type,
        a1_scale=runner_input.a1_scale,
        fused_router=None,
    )


def _pre_permute_bypassed_to_aiter(
    dispatch_output: StandardDispatchOutput,
    quant_info: AiterMoeQuantInfo,
) -> AiterRunnerInput:
    """Build a runner input that carries router logits rather than a selection."""
    topk_output = dispatch_output.topk_output
    cfg = topk_output.topk_config
    shared = cfg.num_fused_shared_experts or 0
    # sglang's top_k counts the fused shared experts; the aiter entry wants the routed
    # width and the shared count separately (topk.py does the same subtraction).
    routed_topk = cfg.top_k - shared
    assert routed_topk > 0, (
        f"routed top-k must be positive: top_k={cfg.top_k}, "
        f"num_fused_shared_experts={shared}"
    )
    return AiterRunnerInput(
        hidden_states=dispatch_output.hidden_states,
        topk_ids=None,
        topk_weights=None,
        quant_type=quant_info.quant_type,
        fused_router=AiterFusedRouterInput(
            router_logits=topk_output.router_logits,
            correction_bias=cfg.correction_bias,
            topk_output=topk_output,
            topk=routed_topk,
            num_fused_shared_experts=shared,
            num_expert_group=cfg.num_expert_group or 1,
            topk_group=cfg.topk_group or 1,
            renormalize=cfg.renormalize,
            routed_scaling_factor=float(cfg.routed_scaling_factor or 1.0),
            shared_expert_weight=float(cfg.fused_shared_experts_scaling_factor or 1.0),
        ),
    )


@register_pre_permute("standard", "aiter")
def pre_permute_standard_to_aiter(
    dispatch_output: StandardDispatchOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AiterRunnerInput:
    hidden_states = dispatch_output.hidden_states

    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if TopKOutputChecker.format_is_bypassed(dispatch_output.topk_output):
        return _pre_permute_bypassed_to_aiter(dispatch_output, quant_info)

    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(torch.float32)

    if runner_config.apply_router_weight_on_input and not quant_info.doweight_stage1:
        # Pre-scale at the Python level for kernels that don't honor doweight_stage1.
        assert topk_weights.dim() == 2 and topk_weights.shape[-1] == 1, (
            "apply_router_weight_on_input requires topk=1"
        )
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
