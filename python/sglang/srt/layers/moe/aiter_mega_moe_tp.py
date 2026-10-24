"""Opt-in AITER MegaMoE-TP for the eager extend forwards of TP MoE layers.

``--enable-aiter-mega-moe-tp`` runs a layer's routed experts, shared expert and TP
all-reduce as one persistent AITER kernel (``aiter.ops.flydsl.mega_moe_tp``, ``ar``
mode); its docstring states the collective, whole-GPU and watchdog contract. Only eager
extend forwards of up to ``SGLANG_AITER_MEGA_MOE_TP_MAX_TOKENS`` tokens take it, and
every gate reads values equal on all TP ranks. Decode, CUDA graphs, torch.compile and
larger forwards keep the stock path. A timed-out kernel wait is raised, ending the
server, by a later eligible forward on the rank that observes it. The kernel binds the
expert weights at load, so online weight updates are rejected.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils.common import ceil_div, log_info_on_rank0, print_info_once

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

logger = logging.getLogger(__name__)

_EXTEND_MODES = (ForwardMode.EXTEND, ForwardMode.MIXED)
# Launch epochs are int32 in the kernel; reset() (collective) long before they wrap.
_RESET_PERIOD = 2**30
_WEIGHT_NAMES = {
    "w1": "w13_weight",
    "w1_scale": "w13_weight_scale",
    "w2": "w2_weight",
    "w2_scale": "w2_weight_scale",
}
# Read by WeightUpdater through `_derived_weight_cache_error` on any module.
_WEIGHT_UPDATE_ERROR = (
    "Online weight updates are not supported with --enable-aiter-mega-moe-tp: the "
    "kernel is bound to the expert weights loaded at startup. Restart without it to "
    "use online weight updates."
)


def comm_options(comm_dtype: str) -> tuple[str, str]:
    """``(comm_dtype, ar_gather)`` of ``MegaMoeTPConfig`` for the env's value."""
    if comm_dtype == "fp8":
        return "fp8", "auto"
    if comm_dtype == "bf16":
        return "bf16", "bf16"
    raise ValueError(
        f"SGLANG_AITER_MEGA_MOE_TP_COMM_DTYPE must be 'fp8' or 'bf16', got {comm_dtype!r}"
    )


def capture_reason() -> Optional[str]:
    if (
        get_is_capture_mode()
        or is_in_tc_piecewise_cuda_graph()
        or torch.cuda.is_current_stream_capturing()
    ):
        return "CUDA graph capture or replay"
    return None


def stock_reason(
    num_tokens: int, dtype: torch.dtype, is_extend: bool, max_tokens: int
) -> Optional[str]:
    """Why a forward takes the stock path (a fixed string per reason), or None."""
    if not is_extend:
        return "a decode or other non-extend forward"
    if num_tokens == 0:
        return "an empty forward"
    if num_tokens > max_tokens:
        return "a forward above SGLANG_AITER_MEGA_MOE_TP_MAX_TOKENS"
    if dtype != torch.bfloat16:
        return "a non-bf16 MoE input"
    return None


class _ArRows:
    """Row buffers for ``ar`` mode, which takes a multiple of tp rows: tp - 1 pad rows
    (zero input and weights, distinct valid ids) sit in front of the token rows, so a
    call is one contiguous slice and the pad rows are written once."""

    def __init__(self, tp, max_tokens, hidden_size, topk, shared_id, device):
        self.tp, self.first = tp, tp - 1
        rows = self.first + max_tokens
        self.x = torch.zeros(rows, hidden_size, dtype=torch.bfloat16, device=device)
        self.ids = torch.arange(topk, dtype=torch.int32, device=device).repeat(rows, 1)
        self.weights = torch.zeros(rows, topk, dtype=torch.float32, device=device)
        if shared_id is not None:
            self.ids[self.first :, -1] = shared_id
            self.weights[self.first :, -1] = 1.0

    def fill(self, x, topk_ids, topk_weights) -> tuple[int, slice]:
        """Stage a forward; returns its pad rows and the slice of every buffer to launch."""
        num_tokens, k = topk_ids.shape
        end = self.first + num_tokens
        pad = -num_tokens % self.tp  # rows that complete a multiple of tp
        self.ids[self.first : end, :k].copy_(topk_ids)
        self.weights[self.first : end, :k].copy_(topk_weights)
        if pad:
            self.x[self.first : end].copy_(x)
        return pad, slice(self.first - pad, end)


class _Runner:
    """One MegaMoeTP shared by all hosted layers (one symmetric arena per rank)."""

    def __init__(self, kernel, rows, max_tokens, layer_ids, err_host, err_event):
        self.kernel, self.rows, self.max_tokens = kernel, rows, max_tokens
        self.first_layer, self.last_layer = min(layer_ids), max(layer_ids)
        self.err_host, self.err_event = err_host, err_event
        self.launches = 0

    def run(self, layer_id, weights, x, topk_ids, topk_weights) -> torch.Tensor:
        # The flag is copied to pinned memory after the last layer: reading it here
        # neither synchronizes nor blocks, and a copy still in flight is checked later.
        if layer_id == self.first_layer and self.err_event.query() and self.err_host[0]:
            raise RuntimeError(
                "AITER MegaMoE-TP: a kernel wait timed out (ranks out of step); MoE "
                "outputs since then are invalid"
            )
        self.launches += 1
        if self.launches % _RESET_PERIOD == 0:
            self.kernel.reset()
        self.kernel.set_weights(**weights)
        pad, rows = self.rows.fill(x, topk_ids, topk_weights)
        route = (self.rows.weights[rows], self.rows.ids[rows])
        out = x.new_empty(x.shape)
        if pad:
            out.copy_(self.kernel(self.rows.x[rows], *route)[pad:])
        else:
            self.kernel(x, *route, out=out)
        if layer_id == self.last_layer:
            self.err_host.copy_(self.kernel.error_flag(), non_blocking=True)
            self.err_event.record()
        return out


class _Hosted(msgspec.Struct):
    """Per-block state: the shared runner and this layer's kernel weights."""

    runner: Optional[_Runner] = None
    weights: Optional[dict[str, torch.Tensor]] = None


def _refuse(moe, reason: str):
    raise ValueError(
        f"--enable-aiter-mega-moe-tp is not supported for MoE layer {moe.layer_id}: "
        f"{reason}."
    )


def _aiter_mega_moe_tp():
    try:
        from aiter.ops.flydsl import mega_moe_tp
    except ImportError as e:
        raise ValueError(
            "--enable-aiter-mega-moe-tp needs an AITER with "
            f"aiter.ops.flydsl.mega_moe_tp (ROCm/aiter#5937): {e}"
        ) from e
    if not mega_moe_tp.mega_moe_tp_supported():
        raise ValueError("--enable-aiter-mega-moe-tp needs a gfx950 GPU")
    return mega_moe_tp


def register(moe: DeepseekV2MoE) -> None:
    """Host a sparse block (decoder ``__init__``); ValueError if it cannot run the kernel."""
    if not get_exec().moe.enable_aiter_mega_moe_tp:
        moe._mega_moe_tp = None
        return
    _aiter_mega_moe_tp()
    experts, fused = moe.experts, moe.num_fused_shared_experts
    runner_config, backend = experts.moe_runner_config, experts.runner_backend
    for refused, reason in (
        (backend is None or not backend.is_aiter(), "its MoE runner is not aiter"),
        (moe.is_hash, "hash routing"),
        (moe.gate.e_score_correction_bias_vl is not None, "vision routing bias"),
        (moe.topk.enable_waterfill, "waterfill top-k"),
        (
            not moe.routed_scaling_factor_fused_in_topk,
            "the routed scaling factor is not folded into the top-k weights",
        ),
        (
            runner_config.activation != "silu"
            or not runner_config.is_gated
            or experts.with_bias,
            "the experts are not bias-free gated SiLU",
        ),
        (
            fused > 1
            or (fused == 0 and not hasattr(moe, "shared_experts"))
            or (fused == 0 and moe._shared_expert_tp1),
            "it needs exactly one shared expert, fused or TP-sharded",
        ),
    ):
        if refused:
            _refuse(moe, reason)
    moe._mega_moe_tp = _Hosted()


def _shared_expert_shard(moe) -> dict[str, torch.Tensor]:
    """This rank's unfused shared expert as one slot, preshuffled like the routed ones."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    gate_up, down = moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj
    # These paths store the weight in another layout than the checkpoint's.
    if any(
        getattr(proj, flag, False)
        for proj in (gate_up, down)
        for flag in ("use_aiter_asm_fp4_gemm", "dequantized_bf16")
    ):
        _refuse(moe, "the shared expert is not on the MXFP4 Triton GEMM path")
    shard = {}
    for (key, scale_key), proj in (
        (("w1", "w1_scale"), gate_up),
        (("w2", "w2_scale"), down),
    ):
        scale = proj.weight_scale.data
        shard[key] = shuffle_weight(proj.weight.data.contiguous()[None], (16, 16))
        shard[scale_key] = e8m0_shuffle(scale.contiguous()).view(1, *scale.shape)
    return shard


def _kernel_weights(moe) -> dict[str, torch.Tensor]:
    """The kernel's four weight tensors, with the shared expert as the last slot.

    A fused shared expert already is that slot. An unfused one is re-homed once into a
    one-slot-larger buffer; the parameters view its first slots, so the stock path reads
    the same values."""
    experts = moe.experts
    params = {key: getattr(experts, name) for key, name in _WEIGHT_NAMES.items()}
    if not getattr(experts.w13_weight, "is_shuffled", False) or any(
        p.dtype != torch.uint8 for p in params.values()
    ):
        _refuse(moe, "the experts are not preshuffled MXFP4 (uint8) tensors")
    if moe.num_fused_shared_experts:
        return {key: p.data for key, p in params.items()}
    shared = _shared_expert_shard(moe)
    weights = {}
    for key, param in params.items():
        slot = shared[key]
        if slot.shape[1:] != param.shape[1:] or slot.dtype != param.dtype:
            _refuse(
                moe,
                f"the shared expert's {_WEIGHT_NAMES[key]} is {tuple(slot.shape[1:])} "
                f"{slot.dtype}, the routed experts' {tuple(param.shape[1:])} {param.dtype}",
            )
        full = torch.empty(
            (param.shape[0] + 1, *param.shape[1:]),
            dtype=param.dtype,
            device=param.device,
        )
        full[:-1].copy_(param.data)
        full[-1:].copy_(slot)
        param.data = full[:-1]
        weights[key] = full
    # The old blocks are too small for the next layer's larger ones: return them now.
    torch.cuda.empty_cache()
    return weights


def _build_runner(blocks, weights) -> _Runner:
    from aiter import ActivationType

    mega_moe_tp = _aiter_mega_moe_tp()
    tp = get_parallel().tp_group
    experts = blocks[0].experts
    runner_config = experts.moe_runner_config
    appended = not blocks[0].num_fused_shared_experts
    num_experts, inter_dim2, packed_hidden = weights[0]["w1"].shape
    topk = runner_config.top_k + int(appended)
    max_tokens = envs.SGLANG_AITER_MEGA_MOE_TP_MAX_TOKENS.get()
    local_tokens = ceil_div(max_tokens, tp.world_size)
    comm_dtype, ar_gather = comm_options(envs.SGLANG_AITER_MEGA_MOE_TP_COMM_DTYPE.get())
    try:
        kernel = mega_moe_tp.MegaMoeTP(
            mega_moe_tp.MegaMoeTPConfig(
                rank=tp.rank_in_group,
                world_size=tp.world_size,
                model_dim=packed_hidden * 2,
                inter_dim=inter_dim2 // 2,
                experts=num_experts,
                topk=topk,
                max_local_tokens=local_tokens,
                activation=ActivationType.Silu,
                swiglu_limit=runner_config.swiglu_limit,
                comm_mode="ar",
                comm_dtype=comm_dtype,
                ar_gather=ar_gather,
            ),
            **weights[0],
            group=tp.cpu_group,
            device=experts.w13_weight.device,
        )
    except ValueError as e:  # also MegaMoeTPLDSError: the window does not fit
        raise ValueError(
            f"--enable-aiter-mega-moe-tp: {e} (tp_size={tp.world_size}, "
            f"SGLANG_AITER_MEGA_MOE_TP_MAX_TOKENS={max_tokens})"
        ) from e
    # One launch config per distinct schedule, by the smallest local token count.
    configs = {kernel.launch_config(m): m for m in range(local_tokens, 0, -1)}
    kernel.prepare(sorted(configs.values()))
    rows = _ArRows(
        tp.world_size,
        max_tokens,
        packed_hidden * 2,
        topk,
        num_experts - 1 if appended else None,
        experts.w13_weight.device,
    )
    log_info_on_rank0(
        logger,
        f"AITER MegaMoE-TP enabled: {len(blocks)} MoE layers, {num_experts} experts "
        f"(shared expert {'appended' if appended else 'fused'}), top-k {topk}, eager "
        f"extend forwards of 1-{max_tokens} tokens, comm {comm_dtype}, {len(configs)} "
        "launch configs prepared",
    )
    return _Runner(
        kernel,
        rows,
        max_tokens,
        [b.layer_id for b in blocks],
        torch.zeros(1, dtype=torch.int32, pin_memory=True),
        torch.cuda.Event(),
    )


def finalize(model: torch.nn.Module) -> None:
    """Collective on every TP rank, after weight postprocessing and before graph capture."""
    if not get_exec().moe.enable_aiter_mega_moe_tp:
        return
    blocks = [
        m for m in model.modules() if getattr(m, "_mega_moe_tp", None) is not None
    ]
    if not blocks:
        raise ValueError(
            "--enable-aiter-mega-moe-tp: the model has no MoE layer hosted by a "
            "MegaMoE-TP-aware decoder"
        )
    weights = [_kernel_weights(block) for block in blocks]
    runner = _build_runner(blocks, weights)
    for block, block_weights in zip(blocks, weights, strict=True):
        block._mega_moe_tp.runner, block._mega_moe_tp.weights = runner, block_weights
        block._derived_weight_cache_error = _WEIGHT_UPDATE_ERROR


def maybe_forward(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    gemm_output_zero_allocator=None,
) -> Optional[torch.Tensor]:
    """The block's TP-reduced output, or None for the stock path (logged once a reason)."""
    # The compiled graph holds the stock path; a logger call here would break the trace.
    if torch.compiler.is_compiling():
        return None
    hosted = moe._mega_moe_tp
    # No runner: the block belongs to a model that was never finalized (a draft).
    if hosted is None or hosted.runner is None:
        return None
    runner = hosted.runner
    reason = capture_reason() or stock_reason(
        hidden_states.shape[0],
        hidden_states.dtype,
        forward_batch.forward_mode in _EXTEND_MODES,
        runner.max_tokens,
    )
    if reason is not None:
        print_info_once(f"AITER MegaMoE-TP: stock MoE path for {reason}")
        return None
    # The layers package must not import models at load time.
    from sglang.srt.models.deepseek_common.amd import deepseek_v2_hip_moe

    # Routing as DeepseekV2MoE.forward_normal; hash and vision routing are refused.
    router_logits, partials = deepseek_v2_hip_moe.forward_gate(
        moe, hidden_states, gemm_output_zero_allocator, fused_gate=True
    )
    kwargs = {} if partials is None else {"router_logits_partials": partials}
    topk_output = moe.topk(hidden_states, router_logits, **kwargs)
    return runner.run(
        moe.layer_id,
        hosted.weights,
        hidden_states,
        topk_output.topk_ids,
        topk_output.topk_weights,
    )
