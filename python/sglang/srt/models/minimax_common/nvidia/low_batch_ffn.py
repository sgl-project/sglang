# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM vllm/models/minimax_m3/nvidia/specialized_decode/moe/__init__.py @ddd43a54
"""SGLang binding of the MiniMax-M3.1-Flash low-batch FFN
(``sglang.kernels.ops.moe.minimax_m3_low_batch``), opt-in with
``SGLANG_MINIMAX_M3_LOW_BATCH_FFN=1``.

``bind_low_batch_ffns`` runs after weight processing, before CUDA graph
capture: it gives every deferring MoE layer it supports a ``LowBatchFfn``
(compiling the kernels) and logs the layers it skips and why. In a captured
decode or target-verify graph of 1-32 rows whose layer boundary defers the
MoE finalize, ``LowBatchFfn.maybe_run`` replaces the stock routed chain
(router top-k, per-token NVFP4, TRT-LLM-gen FC1/FC2) and the MXFP8 shared MLP
and returns the same ``MoeDeferredFinalize`` handoff, so the next layer's
fused finalize + all-reduce + norm consumes it unchanged. Every other call
returns None and runs the stock path.

Streams: the shared expert runs on the layer's alt stream and the routed
chain on the current stream (the stock capture places them the other way).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable, Optional

import torch

from sglang.kernels.ops.moe.minimax_m3_low_batch import shapes
from sglang.kernels.ops.moe.minimax_m3_low_batch.shapes import (
    EXPERTS,
    HIDDEN,
    MAX_ROWS,
    TOP_K,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layer_boundary.fusions.moe_finalize import MoeDeferredFinalize
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmDeferredFinalizeOutput,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend, sum_post_experts_output
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsFusedMoEMethod,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.minimax_m3_nvfp4 import MiniMaxM3Nvfp4MoE
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.runtime_context import get_forward, get_parallel, get_server_args
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)


class LowBatchFfn:
    """The low-batch FFN bound to one ``MiniMaxM3MoE`` layer's processed
    weights (read in place). Build it only for a layer ``unsupported_reason``
    accepts, outside any CUDA graph capture."""

    def __init__(self, moe: torch.nn.Module) -> None:
        experts = moe.experts
        self.columns = int(experts.intermediate_size_per_partition)
        self.routed_scaling_factor = float(moe.routed_scaling_factor)
        self.bias = moe.e_score_correction_bias
        self.w13, self.w13_scale = experts.w13_weight, experts.w13_weight_scale
        self.w2, self.w2_scale = experts.w2_weight, experts.w2_weight_scale
        self.alpha1, self.alpha2 = experts.g1_alphas, experts.g2_alphas
        gate_up = moe.shared_experts.gate_up_proj
        down = moe.shared_experts.down_proj
        self.gate_up, self.gate_up_scale = (
            gate_up.weight,
            gate_up.weight_scale_inv_swizzled,
        )
        self.down, self.down_scale = down.weight, down.weight_scale_inv_swizzled
        # CUDA graphs bake these addresses; a reload that moves one falls back.
        self.bound_ptrs = _bound_ptrs(moe)
        # Imported here: cutlass loads only once a layer binds.
        from sglang.kernels.ops.moe.minimax_m3_low_batch import ffn

        self._routed_ffn = ffn.low_batch_routed_ffn
        self._shared_ffn = ffn.low_batch_shared_ffn
        device = self.w13.device
        self.fc1_records, self.fc2_records = ffn.shared_records(device)
        ffn.precompile_low_batch_ffn(device.index, self.columns)
        _warm_up_once(self)

    def maybe_run(
        self,
        moe: torch.nn.Module,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Optional[MoeDeferredFinalize]:
        """The layer's deferred-finalize handoff from the low-batch kernels,
        or None to run the stock path."""
        forward = get_forward()
        if not _per_call_engages(
            defer_finalize=forward.defer_moe_finalize,
            decode_or_verify=forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify(),
            capture_mode=get_is_capture_mode(),
            breakable_graph=is_in_breakable_cuda_graph(),
            stream_capturing=torch.cuda.is_current_stream_capturing(),
            can_run_tbo=forward_batch.can_run_tbo,
            sp_active=forward.sp_active,
            rows=hidden_states.shape[0],
        ):
            return None
        if not _accepts_input(hidden_states):
            return None
        if _bound_ptrs(moe) != self.bound_ptrs:
            logger.warning_once(
                "MiniMax-M3 low-batch FFN: layer weights moved since binding "
                "(weight reload?); running the stock MoE"
            )
            return None
        return self._run(moe, hidden_states)

    def _run(self, moe: torch.nn.Module, x: torch.Tensor) -> MoeDeferredFinalize:
        main = torch.cuda.current_stream()
        alt = moe.alt_stream
        if alt is not None:
            alt.wait_stream(main)
            with torch.cuda.stream(alt):
                shared = self._shared(x)
        else:
            shared = self._shared(x)
        # FP32 [rows, 128] from the stock router GEMM.
        logits = moe._compute_router_logits(x).float().contiguous()
        routed, weights, permuted = self._routed_ffn(
            x,
            logits,
            bias=self.bias,
            routed_scaling_factor=self.routed_scaling_factor,
            w13=self.w13,
            w13_scale=self.w13_scale,
            w2=self.w2,
            w2_scale=self.w2_scale,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            columns=self.columns,
        )
        if alt is not None:
            main.wait_stream(alt)
        return MoeDeferredFinalize.from_flashinfer(
            FlashInferTrtllmDeferredFinalizeOutput(
                gemm2_out=routed,
                expert_weights=weights,
                expanded_idx_to_permuted_idx=permuted,
                top_k=TOP_K,
            ),
            gated_shared_output=shared,
            m=x.shape[0],
            reduce=sum_post_experts_output,
        )

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        return self._shared_ffn(
            x,
            gate_up=self.gate_up,
            gate_up_scale=self.gate_up_scale,
            down=self.down,
            down_scale=self.down_scale,
            fc1_records=self.fc1_records,
            fc2_records=self.fc2_records,
            columns=self.columns,
        )


def bind_low_batch_ffns(layers: Iterable[torch.nn.Module]) -> None:
    """Give each deferring MoE layer of ``layers`` that supports it a
    ``LowBatchFfn`` (``layer.mlp.low_batch_ffn``) and log, once, which layers
    took it and why the others did not. A no-op unless
    ``SGLANG_MINIMAX_M3_LOW_BATCH_FFN`` is set. Every reason derives from the
    config, so all TP ranks bind the same layers."""
    if not envs.SGLANG_MINIMAX_M3_LOW_BATCH_FFN.get():
        return
    bound, columns = [], set()
    skipped: dict[str, list[int]] = defaultdict(list)
    for layer in layers:
        if not layer.is_layer_sparse:
            continue
        moe = layer.mlp
        reason = unsupported_reason(moe)
        if reason is None:
            moe.low_batch_ffn = LowBatchFfn(moe)
            bound.append(moe.layer_id)
            columns.add(moe.low_batch_ffn.columns)
        else:
            skipped[reason].append(moe.layer_id)
    total = len(bound) + sum(len(ids) for ids in skipped.values())
    message = (
        f"MiniMax-M3 low-batch FFN: bound {len(bound)}/{total} MoE layers"
        + (f" at {sorted(columns)} intermediate columns per rank" if bound else "")
        + "".join(
            f"; stock for layers {_ranges(ids)}: {reason}"
            for reason, ids in skipped.items()
        )
    )
    log_info_on_rank0(logger, message)


def unsupported_reason(moe: torch.nn.Module) -> Optional[str]:
    """Why ``moe`` (a ``MiniMaxM3MoE`` after weight processing) cannot take
    the low-batch FFN, or None."""
    for check in (
        _platform_reason,
        _server_reason,
        _parallel_reason,
        lambda: _layer_reason(moe),
        lambda: _routed_reason(moe.experts),
        lambda: _shared_reason(moe.shared_experts, moe.experts),
    ):
        reason = check()
        if reason is not None:
            return reason
    return None


def _per_call_engages(
    *,
    defer_finalize: bool,
    decode_or_verify: bool,
    capture_mode: bool,
    breakable_graph: bool,
    stream_capturing: bool,
    can_run_tbo: bool,
    sp_active: bool,
    rows: int,
) -> bool:
    """Engage only while capturing a full decode / target-verify CUDA graph of
    1-32 rows whose boundary takes the deferred finalize; warm-up, eager and
    prefill calls (the prefill graph runner also sets capture mode) and
    larger graphs run stock. All host state, equal on every TP rank."""
    return (
        defer_finalize
        and decode_or_verify
        and capture_mode
        and not breakable_graph
        and stream_capturing
        and not can_run_tbo
        and not sp_active
        and 1 <= rows <= MAX_ROWS
    )


def _accepts_input(x: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and x.size(1) == HIDDEN
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
    )


def _bound_ptrs(moe: torch.nn.Module) -> tuple[int, ...]:
    experts, shared = moe.experts, moe.shared_experts
    return tuple(
        tensor.data_ptr()
        for tensor in (
            moe.e_score_correction_bias,
            experts.w13_weight,
            experts.w13_weight_scale,
            experts.w2_weight,
            experts.w2_weight_scale,
            experts.g1_alphas,
            experts.g2_alphas,
            shared.gate_up_proj.weight,
            shared.gate_up_proj.weight_scale_inv_swizzled,
            shared.down_proj.weight,
            shared.down_proj.weight_scale_inv_swizzled,
        )
    )


_WARMED_UP: set[tuple[int, int]] = set()


def _warm_up_once(ffn: LowBatchFfn) -> None:
    """Launch each compiled kernel once per (device, columns) on zero rows, so
    that no module load happens inside a CUDA graph capture."""
    device = ffn.w13.device
    key = (device.index, ffn.columns)
    if key in _WARMED_UP:
        return
    for rows in (1, 16, 32):  # front widths 8 / 16 / 32, FC1/FC2 both overlaps
        x = torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        logits = torch.zeros((rows, EXPERTS), dtype=torch.float32, device=device)
        ffn._routed_ffn(
            x,
            logits,
            bias=ffn.bias,
            routed_scaling_factor=ffn.routed_scaling_factor,
            w13=ffn.w13,
            w13_scale=ffn.w13_scale,
            w2=ffn.w2,
            w2_scale=ffn.w2_scale,
            alpha1=ffn.alpha1,
            alpha2=ffn.alpha2,
            columns=ffn.columns,
        )
        ffn._shared(x)
    torch.cuda.synchronize(device)
    _WARMED_UP.add(key)


def _platform_reason() -> Optional[str]:
    if not torch.cuda.is_available():
        return "no CUDA device"
    major, minor = torch.cuda.get_device_capability()
    if major != 10 or minor > 3:
        return f"needs an SM100-family GPU, got sm_{major}{minor}"
    if envs.FLASHINFER_NVFP4_4OVER6.get():
        return "FLASHINFER_NVFP4_4OVER6 changes the stock activation quantization"
    return None


def _server_reason() -> Optional[str]:
    args = get_server_args()
    for enabled, what in (
        (args.enable_nvidia_megakernel, "the NVIDIA megakernel"),
        (args.enable_amd_monokernel, "the AMD monokernel"),
        (args.enable_lora or args.lora_paths, "LoRA"),
        (args.enable_eplb, "EPLB"),
        (args.ep_num_redundant_experts, "redundant experts"),
        (args.expert_distribution_recorder_mode, "the expert distribution recorder"),
        (args.enable_return_routed_experts, "routed-expert capture"),
        (
            args.enable_two_batch_overlap or args.enable_single_batch_overlap,
            "two-batch / single-batch overlap",
        ),
        (
            args.enable_deterministic_inference or args.rl_on_policy_target,
            "deterministic inference",
        ),
        (args.enable_torch_compile, "torch.compile"),
    ):
        if enabled:
            return f"{what} is enabled"
    return None


def _parallel_reason() -> Optional[str]:
    parallel = get_parallel()
    if not (
        parallel.moe_ep_size == 1
        and parallel.moe_tp_size == parallel.tp_size
        and parallel.attn_tp_size == parallel.tp_size
        and parallel.pp_size == 1
        and parallel.attn_cp_size == 1
        and not is_dp_attention_enabled()
        and get_moe_a2a_backend().is_none()
    ):
        return "needs plain TP (no EP, PP, CP, DP attention or MoE all-to-all)"
    return None


def _layer_reason(moe: torch.nn.Module) -> Optional[str]:
    if not moe.supports_deferred_finalize:
        return "the layer does not defer its MoE finalize"
    if moe.trtllm_decode_with_mega_moe:
        return "MegaMoE"
    if moe.shared_experts is None or moe.num_fused_shared_experts:
        return "no separate shared expert"
    if moe.topk.is_draft:
        return "a draft model's MoE layer"
    if moe.topk.enable_waterfill:
        return "waterfill shared-expert routing"
    config = moe.topk.topk_config
    if not (
        config.scoring_func == "sigmoid"
        and config.top_k == TOP_K
        and config.renormalize
        and not config.use_grouped_topk
        and config.custom_routing_function is None
        and not config.sqrtsoftplus_log1p
        and config.apply_routed_scaling_factor_on_output
        and config.routed_scaling_factor == moe.routed_scaling_factor
    ):
        return "routing is not renormalized sigmoid top-4"
    bias = moe.e_score_correction_bias
    if (
        bias is None
        or config.correction_bias is None
        or bias.dtype != torch.float32
        or tuple(bias.shape) != (EXPERTS,)
    ):
        return "no FP32 [128] score correction bias"
    if moe.experts._minimax_activation != "swigluoai":
        return "activation is not SwiGLU-OAI"
    return None


def _routed_reason(experts: torch.nn.Module) -> Optional[str]:
    if not (
        isinstance(experts.quant_method, CompressedTensorsFusedMoEMethod)
        and isinstance(experts.scheme, MiniMaxM3Nvfp4MoE)
        and experts.scheme.use_flashinfer_trtllm
        and not experts.scheme.use_mega_moe_prefill
    ):
        return "routed experts are not NVFP4 on flashinfer_trtllm_routed"
    columns = experts.intermediate_size_per_partition
    if columns not in shapes.COLUMNS:
        return f"{columns} intermediate columns per rank (built for 768 and 384)"
    if (
        experts.num_experts != EXPERTS
        or experts.num_local_experts != EXPERTS
        or experts.hidden_size != HIDDEN
    ):
        return "layer shape is not 128 local experts at hidden 6144"
    runner = experts.moe_runner_config
    clamp = runner.gemm1_clamp_limit or runner.swiglu_limit
    if (runner.gemm1_alpha, runner.gemm1_beta, clamp) != (
        shapes.SWIGLU_ALPHA,
        shapes.SWIGLU_BETA,
        shapes.SWIGLU_LIMIT,
    ):
        return "routed SwiGLU-OAI constants"
    for name, tensor, numel in (
        ("w13", experts.w13_weight, EXPERTS * 2 * columns * HIDDEN // 2),
        ("w13 scales", experts.w13_weight_scale, EXPERTS * 2 * columns * HIDDEN // 16),
        ("w2", experts.w2_weight, EXPERTS * HIDDEN * columns // 2),
        ("w2 scales", experts.w2_weight_scale, EXPERTS * HIDDEN * columns // 16),
    ):
        if (
            tensor.element_size() != 1
            or tensor.numel() != numel
            or not tensor.is_contiguous()
            or tensor.data_ptr() % 32
        ):
            return f"{name} layout"
    for alphas in (experts.g1_alphas, experts.g2_alphas):
        if (
            alphas.dtype != torch.float32
            or tuple(alphas.shape) != (EXPERTS,)
            or not alphas.is_contiguous()
        ):
            return "per-expert global scales are not FP32 [128]"
    return None


def _shared_reason(shared: torch.nn.Module, experts: torch.nn.Module) -> Optional[str]:
    columns = experts.intermediate_size_per_partition
    if (shared.swiglu_alpha, shared.swiglu_limit) != (
        shapes.SWIGLU_ALPHA,
        shapes.SWIGLU_LIMIT,
    ):
        return "shared expert SwiGLU-OAI constants"
    for linear, n, k in (
        (shared.gate_up_proj, 2 * columns, HIDDEN),
        (shared.down_proj, HIDDEN, columns),
    ):
        method = linear.quant_method
        if not (
            isinstance(method, Fp8LinearMethod)
            and method.use_mxfp8
            and not method.convert_mxfp8_to_block
            and method.mxfp8_dense_backend is not None
            # FLASHINFER_TRTLLM row-shuffles the weight in place.
            and (
                method.mxfp8_dense_backend.is_flashinfer_cutedsl()
                or method.mxfp8_dense_backend.is_flashinfer_cutlass()
            )
        ):
            return "shared expert is not MXFP8 on a FlashInfer CuTe DSL / CUTLASS GEMM"
        weight, scale = linear.weight, linear.weight_scale_inv_swizzled
        if (
            weight.dtype != torch.float8_e4m3fn
            or tuple(weight.shape) != (n, k)
            or not weight.is_contiguous()
            or scale is None
            or scale.element_size() != 1
            or scale.numel() != n * k // 32
            or not scale.is_contiguous()
        ):
            return "shared expert MXFP8 layout"
    return None


def _ranges(ids: list[int]) -> str:
    """``[0, 1, 2, 5]`` as ``0-2,5``."""
    spans: list[list[int]] = []
    for i in sorted(ids):
        if spans and i == spans[-1][1] + 1:
            spans[-1][1] = i
        else:
            spans.append([i, i])
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in spans)
