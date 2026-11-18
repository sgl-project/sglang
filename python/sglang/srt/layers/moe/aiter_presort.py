"""Hand AITER's fused_moe the BM=16 sort the radix-4 router writes on ROCm.

At decode and verify batch sizes AITER's fused_moe spends one or two launches on
moe_sorting, which also zeroes its output. route_radix4_with_sort emits both from
the routing launch; register_presorted_moe (ROCm/aiter#6484) hands that sort to
the next fused_moe call, which uses it when the call's topk tensors, M and
block_m match and ignores it otherwise.

AITER takes the sort only where the kernel config it resolves for the batch is
its BM=16 aux-sort pipeline, and sorts again elsewhere, so each layer's
PresortGate asks AITER's own config lookup (get_2stage_cfgs) per batch size.

Only a Kimi-K3 MoE layer whose experts take the sort asks for it, through the
consume-once stash in route_quant_handoff, so the shared routing functions keep
their signatures:

    KimiK3MoE.forward           stage(gate) before routing, clear() after the layer
    biased_grouped_topk_gpu     try_route_and_register() replaces route_radix4
                                when the staged gate takes the batch
    AiterRunnerCore.run         hands the gate its fused_moe arguments
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Callable, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.moe import moe_route_radix4
from sglang.srt.layers import zero_copy_context
from sglang.srt.layers.moe import route_quant_handoff
from sglang.srt.runtime_context import get_buffer
from sglang.srt.utils import is_device_stream_capturing

logger = logging.getLogger(__name__)


@cache_once
def _registrar() -> Optional[Callable[[Optional[dict]], None]]:
    try:
        from aiter.fused_moe import register_presorted_moe
    except ImportError:
        return None
    return register_presorted_moe


@cache_once
def _cfg_api() -> Optional[SimpleNamespace]:
    """The AITER calls fused_moe resolves its kernel config with."""
    try:
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import (
            _normalize_mxfp4_activation_params,
            get_2stage_cfgs,
            get_inter_dim,
            get_padded_M,
            quant_remap,
            resolve_activation_dtype,
        )
        from aiter.ops.flydsl.moe_common import GateMode
        from aiter.presorted_moe import PRESORTED_MOE_BLOCK_M
    except ImportError as e:
        logger.warning(
            f"radix-4 fused MoE sort off: AITER lacks its config lookup ({e})"
        )
        return None
    return SimpleNamespace(
        ActivationType=ActivationType,
        QuantType=QuantType,
        GateMode=GateMode,
        get_2stage_cfgs=get_2stage_cfgs,
        get_inter_dim=get_inter_dim,
        get_padded_M=get_padded_M,
        normalize_activation_params=_normalize_mxfp4_activation_params,
        quant_remap=quant_remap,
        resolve_activation_dtype=resolve_activation_dtype,
        presort_block_m=PRESORTED_MOE_BLOCK_M,
    )


def available() -> bool:
    """Whether this AITER can take a presort (register_presorted_moe) and tell
    which batch sizes will use it (get_2stage_cfgs)."""
    return _registrar() is not None and _cfg_api() is not None


def stage(gate: PresortGate) -> None:
    """Ask the next radix-4 routing call for the fused sort, on the batch sizes
    the layer's gate takes. The caller pairs this with clear() after the layer
    (try/finally)."""
    route_quant_handoff.stage_aiter_presort(gate)


def clear() -> None:
    """Drop the request and a registration that no fused_moe consumed, as
    AITER's caller contract requires."""
    route_quant_handoff.clear()
    _registrar()(None)


# fused_moe arguments that do not change the config get_2stage_cfgs resolves, or
# that _cfg_lookup() passes on the way _fused_moe_impl does. A call with any
# other argument (block_size_M, stage2_scatter, shared experts, ...) is not one
# the lookup mirrors.
_MIRRORED_ARGS = frozenset(
    (
        "hidden_states",
        "w1",
        "w2",
        "topk_weight",
        "topk_ids",
        "expert_mask",
        "activation",
        "quant_type",
        "doweight_stage1",
        "w1_scale",
        "w2_scale",
        "a1_scale",
        "a2_scale",
        "num_local_tokens",
        "moe_sorting_dispatch_policy",
        "dtype",
        "hidden_pad",
        "intermediate_pad",
        "bias1",
        "bias2",
        "swiglu_limit",
        "beta",
        "linear_beta",
        "gate_mode",
        "output",
        "quant_dtype_a",
        "quant_dtype_a2",
    )
)


def _cfg_lookup(call: dict) -> Tuple[dict, Optional[str]]:
    """The batch-size-independent part of the get_2stage_cfgs call that
    _fused_moe_impl makes for fused_moe(**call), or why AITER would not take a
    presort on this call at any batch size."""
    api = _cfg_api()
    extra = sorted(set(call) - _MIRRORED_ARGS)
    if extra:
        return {}, f"fused_moe gets {extra}"
    # match_presorted_moe drops the sort under an expert mask or a row limit.
    if call.get("expert_mask") is not None or call.get("num_local_tokens") is not None:
        return {}, "expert_mask or num_local_tokens is set"
    w1, w2, hidden = call["w1"], call["w2"], call["hidden_states"]
    # fused_moe passes the enums on as .value, which _fused_moe_impl converts back.
    activation = api.ActivationType(
        call.get("activation", api.ActivationType.Silu).value
    )
    swiglu_limit = call.get("swiglu_limit")
    if (
        swiglu_limit not in (None, 0, math.inf)
        and activation == api.ActivationType.Silu
    ):
        return {}, "a SiLU clamp limit, which rewrites fused_moe's quant contract"
    quant_type = api.QuantType(call.get("quant_type", api.QuantType.No).value)
    quant_type = api.quant_remap.get(quant_type, quant_type)
    gate_mode = api.GateMode(call.get("gate_mode", api.GateMode.SEPARATED.value))
    expert, model_dim, inter_dim = api.get_inter_dim(w1.shape, w2.shape)
    situ_beta, situ_linear_beta, _ = api.normalize_activation_params(
        activation, call.get("beta"), call.get("linear_beta"), swiglu_limit
    )
    # fused_moe reads these optional weight tags with getattr, so do the same.
    w1_shuffled = getattr(w1, "is_shuffled", False)
    w2_shuffled = getattr(w2, "is_shuffled", False)
    gate_up_interleave = getattr(w1, "gate_up_interleave", None)
    a1_scale, a2_scale = call.get("a1_scale"), call.get("a2_scale")
    return (
        dict(
            model_dim=model_dim,
            inter_dim=inter_dim,
            expert=expert,
            topk=call["topk_ids"].shape[1],
            dtype=hidden.dtype if call.get("dtype") is None else call["dtype"],
            q_dtype_w=w1.dtype,
            q_type=quant_type,
            use_g1u1=inter_dim != w1.shape[1],
            activation=activation,
            doweight_stage1=call.get("doweight_stage1", False),
            hidden_pad=call.get("hidden_pad", 0),
            intermediate_pad=call.get("intermediate_pad", 0),
            is_shuffled=w1_shuffled or w2_shuffled,
            gate_mode=gate_mode,
            is_ep=False,
            has_stage1_bias=call.get("bias1") is not None,
            has_stage2_bias=call.get("bias2") is not None,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            swiglu_limit=swiglu_limit,
            opus_weights_shuffled=w1_shuffled and w2_shuffled,
            gate_up_interleave=gate_up_interleave,
            config_file=None,
            q_dtype_a2=call.get("quant_dtype_a2"),
            input_dtype=hidden.dtype,
            has_stage2_scatter=False,
            has_activation_scales=a1_scale is not None or a2_scale is not None,
            _has_input_activation_scales=a1_scale is not None,
            has_num_local_tokens=False,
            # Not get_2stage_cfgs arguments: what resolves q_dtype_a per batch.
            quant_dtype_a=call.get("quant_dtype_a"),
            has_a1_scale=a1_scale is not None,
        ),
        None,
    )


def _cfg_args(lookup: dict, m: int) -> dict:
    """get_2stage_cfgs's keyword arguments for an M-token fused_moe call."""
    api = _cfg_api()
    args = dict(lookup)
    quant_dtype_a = args.pop("quant_dtype_a")
    has_a1_scale = args.pop("has_a1_scale")
    if quant_dtype_a is None:
        quant_dtype_a = api.resolve_activation_dtype(
            args["q_type"],
            args["q_dtype_w"],
            activation=args["activation"],
            gate_mode=args["gate_mode"],
            M=m,
            hidden_dtype=args["input_dtype"],
            has_a1_scale=has_a1_scale,
            gate_up_interleave=args["gate_up_interleave"],
        )
    return dict(token=api.get_padded_M(m), q_dtype_a=quant_dtype_a, **args)


class PresortGate:
    """Whether a layer's fused_moe takes the router's sort at a batch size.

    AITER uses a registered sort only where the config it resolves for the batch
    runs its aux-sort pipeline at block_m == PRESORTED_MOE_BLOCK_M (16); a BM=32
    row, say, sorts again and the fused sort is wasted work. The gate asks
    get_2stage_cfgs with the arguments _fused_moe_impl derives from the layer's
    fused_moe call, which it learns from the first call (observe()); until then
    it declines. A config fused_moe re-resolves at call time (an unsupported
    tuned row) is not seen here; AITER then ignores the sort and logs it.

    Answers are cached per M and are host-only, so a CUDA graph captures a fixed
    one for its batch size.
    """

    def __init__(self) -> None:
        self._lookup: Optional[dict] = None
        self._unsupported = False
        self._by_m: dict[int, bool] = {}

    def observe(self, call: dict) -> None:
        """Learn the layer's fused_moe arguments from its call (keyword form)."""
        if self._lookup is not None or self._unsupported:
            return
        lookup, reason = _cfg_lookup(call)
        if reason is not None:
            self._decline(reason)
        else:
            self._lookup = lookup

    def takes_presort(self, m: int) -> bool:
        if self._lookup is None or m == 0:
            return False
        hit = self._by_m.get(m)
        if hit is None:
            hit = self._by_m[m] = self._resolve(m)
        return hit

    def cfg_args(self, m: int) -> dict:
        """The get_2stage_cfgs arguments of the layer's fused_moe at M tokens."""
        return _cfg_args(self._lookup, m)

    def _resolve(self, m: int) -> bool:
        api = _cfg_api()
        try:
            metadata = api.get_2stage_cfgs(**self.cfg_args(m))
        except (NotImplementedError, ValueError) as e:
            # fused_moe itself would raise here; nothing to hand it.
            self._decline(f"get_2stage_cfgs rejects M={m}: {e}")
            return False
        return (
            metadata.full_impl is None
            and bool(metadata.output_aux)
            and int(metadata.block_m) == api.presort_block_m
        )

    def _decline(self, reason: str) -> None:
        self._lookup = None
        self._unsupported = True
        _warn_declined(reason)


@cache_once
def _warn_declined(reason: str) -> None:
    # Every K3 layer shares the reason, so say it once.
    logger.warning(f"radix-4 fused MoE sort off: {reason}")


class _SortWorkspaces:
    """The sort's arrival counters per (device, stream): launches in flight on
    two streams must not share them. A captured graph keeps its capture stream's
    counters, so it must not replay alongside eager launches on that stream."""

    def __init__(self) -> None:
        self.by_stream: dict[tuple[torch.device, int], torch.Tensor] = {}

    def get(self, device: torch.device) -> Optional[torch.Tensor]:
        """The current stream's counters, or None when they would first be
        allocated inside a graph capture, where they are only zeroed when that
        one graph replays."""
        key = (device, torch.cuda.current_stream(device).cuda_stream)
        ws = self.by_stream.get(key)
        if ws is None:
            if is_device_stream_capturing(device):
                _warn_captured_without_workspace()
                return None
            ws = moe_route_radix4.new_sort_workspace(device)
            self.by_stream[key] = ws
        return ws


@cache_once
def _warn_captured_without_workspace() -> None:
    # A graph captured here keeps the unfused router (and AITER's own sort) on
    # every replay, which otherwise only shows up as a slower step.
    logger.warning(
        "radix-4 fused MoE sort skipped: first use on this stream is a CUDA graph "
        "capture; run the router once eagerly on the capture stream first"
    )


def _published_moe_output(n_tokens: int) -> Optional[torch.Tensor]:
    """The MoE output buffer the model published, when the kernel can zero it.

    AITER only trusts the zeroing when this is also the output fused_moe gets.
    """
    out = zero_copy_context.ctx.moe_output
    if (
        out is not None
        and out.dim() == 2
        and out.shape[0] == n_tokens
        and out.shape[1] % 8 == 0
        and out.dtype == torch.bfloat16
        and out.is_contiguous()
        and out.data_ptr() % 16 == 0
    ):
        return out
    return None


def try_route_and_register(
    scores: torch.Tensor,
    bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Route with the fused sort and register it for the next fused_moe call
    when the staged layer's gate takes this batch size. Returns (weights, ids) on a hit,
    None otherwise (the caller runs route_radix4).

    Caller must have checked moe_route_radix4.covered().
    """
    gate, num_token_non_padded = route_quant_handoff.take_aiter_presort()
    if gate is None or not gate.takes_presort(scores.shape[0]):
        return None
    sort_ws = get_buffer("aiter_presort_sort_ws", _SortWorkspaces).get(scores.device)
    if sort_ws is None:
        return None
    weights, ids, aux = moe_route_radix4.route_radix4_with_sort(
        scores=scores,
        bias=bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        sort_ws=sort_ws,
        moe_buf=_published_moe_output(scores.shape[0]),
        num_token_non_padded=num_token_non_padded,
    )
    _registrar()(aux)
    return weights, ids
