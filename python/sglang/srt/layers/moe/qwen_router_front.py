"""Single-kernel router front end for Qwen MoE blocks on the CuTe DSL NVFP4 runner.

At decode token counts the stock chain after the router GEMM (top-k, NVFP4
input quant, moe_sort) is replaced by one kernel that writes the same tensors
the grouped GEMMs consume; FC1 / FC2 / finalize then run as in FlashInfer's
``_moe_core_impl``. Enabled by SGLANG_Q35MF_ROUTE_FUSED (SGLANG_Q35MF_ROUTE_GEMV
also moves the router GEMM into the kernel); every unsupported configuration
keeps the stock path. On the trtllm runner, SGLANG_Q35MF_ROUTE_TABLES_MAX_TOKENS
replaces FlashInfer's routing kernels by one top-k + routing-tables kernel and
runs FlashInfer's FP4 body from those tables.
"""

# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by NVIDIA: _run_experts_from_tables is the post-moe_sort part of
# flashinfer.fused_moe.cute_dsl.fused_moe._moe_core_impl (FlashInfer 0.7.1),
# restricted to W4A4 with a scalar input scale and fed externally built routing
# tables; _choose_tactic and the runner inputs in forward_routed follow
# CuteDslMoEWrapper.run so the autotuner cache is shared.
# _stock_runner_and_tactic, _fp4_body_inputs, _fp4_output and _fp4_runner_kwargs follow
# flashinfer.fused_moe.core.trtllm_fp4_block_scale_moe_op (FlashInfer 0.7.1) so the
# precomputed-routing call launches the stock call's tactic.

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

logger = logging.getLogger(__name__)

_HIDDEN = 4096
_NUM_EXPERTS = 512
_TOP_K = 10
_MAX_TOKENS = 16
# Upper end of the parallel-quantization front (TP2 verify at batch 8 is 56 tokens).
_MAX_PARQ_TOKENS = 64

logger = logging.getLogger(__name__)
_parq_logged: set = set()


def _parq_max_tokens() -> int:
    # SGLANG_Q35MF_K4_PARQ_MAX_TOKENS widens the parallel-quant front and its routing tables up to 128
    # tokens (TP2 verify at batch 12 is 84); unset keeps 64.
    return max(_MAX_PARQ_TOKENS, min(envs.SGLANG_Q35MF_K4_PARQ_MAX_TOKENS.get(), 128))


# Precomputed-routing FP4 MoE state, one set for all layers: each layer's body and finalize consume
# it before the next layer's router writes it. Keys: (tokens, finalize) -> (runner, stock tactic);
# (tokens, tile) -> RouterTables; (tokens, tile, config, finalize) -> prepared body buffers.
_tables_tactics: Dict[Tuple[int, bool], Tuple[Any, Any]] = {}
_tables_by_tile: Dict[Tuple[int, int], Any] = {}
_tables_bodies: Dict[Tuple[int, int, int, bool], Tuple[torch.Tensor, ...]] = {}
_tables_logged: set = set()


def enabled() -> bool:
    return envs.SGLANG_Q35MF_ROUTE_FUSED.get()


def _static_supported(block: Qwen2MoeSparseMoeBlock) -> bool:
    from sglang.srt.layers.moe.topk import precomputed_topk_postprocess_is_noop
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    experts = block.experts
    cfg = block.topk.topk_config
    gate_w = block.gate.weight
    quant_method = experts.quant_method
    return (
        isinstance(quant_method, ModelOptNvFp4FusedMoEMethod)
        and not quant_method._is_cutedsl_v1_deepep
        and get_moe_runner_backend().is_flashinfer_cutedsl()
        and get_moe_a2a_backend().is_none()
        and not block.enable_shared_expert_fusion
        and block.shared_expert is not None
        and block.shared_expert_gate is not None
        and not block.reduce_results
        and not experts.reduce_results
        and experts.num_experts == _NUM_EXPERTS
        and experts.num_local_experts == _NUM_EXPERTS
        and experts.top_k == _TOP_K
        and experts.hidden_size == _HIDDEN
        and cfg.top_k == _TOP_K
        and cfg.renormalize
        and cfg.scoring_func == "softmax"
        and cfg.correction_bias is None
        and cfg.custom_routing_function is None
        and not cfg.use_grouped_topk
        and cfg.num_fused_shared_experts == 0
        and not cfg.apply_routed_scaling_factor_on_output
        and gate_w.dtype == torch.bfloat16
        and tuple(gate_w.shape) == (_NUM_EXPERTS, _HIDDEN)
        and gate_w.is_contiguous()
        and precomputed_topk_postprocess_is_noop(cfg)
        and not block.topk.enable_waterfill
    )


def supported(block: Qwen2MoeSparseMoeBlock, hidden_states: torch.Tensor) -> bool:
    """Per-call check; the static part is evaluated once per block."""
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        is_in_tc_piecewise_cuda_graph,
    )

    ok = block._route_front_static_ok
    if ok is None:
        ok = block._route_front_static_ok = _static_supported(block)
    m = hidden_states.shape[0]
    return (
        ok
        and not torch.compiler.is_compiling()
        and not is_in_tc_piecewise_cuda_graph()
        and 0 < m <= _MAX_TOKENS
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and hidden_states.data_ptr() % 32 == 0
        and _cutedsl_ready(block.experts)
    )


def _static_supported_rt(block: Qwen2MoeSparseMoeBlock) -> bool:
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    experts = block.experts
    cfg = block.topk.topk_config
    gate_w = block.gate.weight
    return (
        isinstance(experts.quant_method, ModelOptNvFp4FusedMoEMethod)
        and get_moe_runner_backend().is_flashinfer_trtllm()
        and get_moe_a2a_backend().is_none()
        and not experts.quant_method.quant_config.use_per_token_activation
        and experts.trtllm_padded_hidden_size is None
        and not experts._dwdp_bound
        and not experts.reduce_results
        and not block.enable_shared_expert_fusion
        and experts.hidden_size == _HIDDEN
        and experts.w13_input_scale_quant.numel() == 1
        and cfg.output_format is None
        and gate_w.dtype == torch.bfloat16
        and tuple(gate_w.shape) == (_NUM_EXPERTS, _HIDDEN)
        and gate_w.is_contiguous()
    )


def supported_rt(block: Qwen2MoeSparseMoeBlock, hidden_states: torch.Tensor) -> bool:
    """Per-call check for the trtllm-runner path; the static part is evaluated once per block."""
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        is_in_tc_piecewise_cuda_graph,
    )

    m = hidden_states.shape[0]
    if m > _MAX_TOKENS and not (
        envs.SGLANG_Q35MF_ROUTE_PARQ.get()
        and m <= _parq_max_tokens()
        and block.alt_stream is not None
        and get_is_capture_mode()
    ):
        return False
    ok = block._route_rt_static_ok
    if ok is None:
        ok = block._route_rt_static_ok = _static_supported_rt(block)
    return (
        ok
        and not torch.compiler.is_compiling()
        and not is_in_tc_piecewise_cuda_graph()
        and m > 0
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and hidden_states.data_ptr() % 32 == 0
    )


def _static_supported_shared(block: Qwen2MoeSparseMoeBlock) -> bool:
    """The shared expert's gate/up is a static per-tensor FP8 projection the front-end kernel can compute."""
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

    mlp = block.shared_expert
    if not isinstance(mlp, Qwen2MoeMLP):
        return False
    gu = mlp.gate_up_proj
    return (
        isinstance(gu.quant_method, ModelOptFp8LinearMethod)
        and gu.use_flashinfer_bmm
        and gu.bias is None
        and gu.weight.dtype == torch.float8_e4m3fn
        and gu.weight.shape[0] == _HIDDEN
        and gu.weight.shape[1] % 8 == 0
        and gu.weight.shape[1] <= 512
        and gu.weight.t().is_contiguous()
        and gu.weight_scale.numel() == 1
        and gu.input_scale.numel() == 1
        and isinstance(mlp.act_fn, SiluAndMul)
    )


def _use_shared(block: Qwen2MoeSparseMoeBlock, m: int, defer_finalize: bool) -> bool:
    """Whether this launch also computes the shared gate/up: only where the routed path is issued
    before the shared expert (dual-stream deferred finalize), so the side stream can consume it."""
    from sglang.srt.model_executor.runner import get_is_capture_mode

    if not (
        defer_finalize
        and m <= 8
        and block.alt_stream is not None
        and get_is_capture_mode()
    ):
        return False
    ok = block._route_sh_static_ok
    if ok is None:
        ok = block._route_sh_static_ok = _static_supported_shared(block)
    return ok  # False unless SGLANG_Q35MF_ROUTE_SHARED is set (Qwen2MoeSparseMoeBlock.__init__)


def shared_expert_tail(mlp, gate_up: torch.Tensor) -> torch.Tensor:
    """Qwen2MoeMLP.forward after its gate_up projection."""
    if mlp._enable_silu_fp4_quant_fusion and not isinstance(gate_up, tuple):
        x, _ = mlp.down_proj(mlp._silu_fp4_quant_fused(gate_up))
        return x
    x = mlp.act_fn(gate_up)
    x, _ = mlp.down_proj(x)
    return x


_MKM_MAX_TOKENS = 7
_MKM_M16_MAX_TOKENS = 16  # SGLANG_Q35MF_MKM_FUSED_M16, TP4 (intermediate 256) only
_MKM_M21_MAX_TOKENS = 21  # SGLANG_Q35MF_MKM_FUSED_M21, TP4 (intermediate 256) only
_MKM_INTERMEDIATES = (128, 256)  # per rank: TP8, TP4
# Keyed by id(block): blocks live for the whole process.
_mkm_static_by_block: dict[int, bool] = {}
_mkm_expanded_idx_by_device: dict[torch.device, torch.Tensor] = {}
_mkm_engaged: set[tuple[int, int]] = set()


def _mkm_static_supported(block: Qwen2MoeSparseMoeBlock) -> bool:
    """The fused engine's fixed shapes: TP8 or TP4 Qwen3.5 NVFP4 on the trtllm-gen weight layout, no EP."""
    from sglang.srt.layers.moe.utils import RoutingMethodType

    experts = block.experts
    cfg = block.topk.topk_config
    runner_config = experts.moe_runner_config
    expert_scales = (experts.g1_alphas, experts.g1_scale_c, experts.g2_alphas)
    inter = experts.intermediate_size_per_partition
    return (
        experts.num_experts == _NUM_EXPERTS
        and experts.num_local_experts == _NUM_EXPERTS
        and experts.moe_ep_rank == 0
        and inter in _MKM_INTERMEDIATES
        and experts.top_k == _TOP_K
        and cfg.top_k == _TOP_K
        and cfg.renormalize
        and cfg.scoring_func == "softmax"
        and cfg.correction_bias is None
        and not cfg.use_grouped_topk
        and cfg.num_fused_shared_experts == 0
        and experts.routing_method_type == RoutingMethodType.RenormalizeNaive
        and runner_config.activation == "silu"
        and runner_config.is_gated
        and runner_config.routed_scaling_factor in (None, 1.0)
        and runner_config.gemm1_alpha is None
        and runner_config.gemm1_clamp_limit is None
        and runner_config.swiglu_limit is None
        and experts.w13_weight.dtype == torch.uint8
        and tuple(experts.w13_weight.shape) == (_NUM_EXPERTS, 2 * inter, _HIDDEN // 2)
        and experts.w2_weight.dtype == torch.uint8
        and tuple(experts.w2_weight.shape) == (_NUM_EXPERTS, _HIDDEN, inter // 2)
        and experts.w13_weight_scale.dtype == torch.float8_e4m3fn
        and tuple(experts.w13_weight_scale.shape)
        == (_NUM_EXPERTS, 2 * inter, _HIDDEN // 16)
        and experts.w2_weight_scale.dtype == torch.float8_e4m3fn
        and tuple(experts.w2_weight_scale.shape) == (_NUM_EXPERTS, _HIDDEN, inter // 16)
        and all(
            t.dtype == torch.float32 and tuple(t.shape) == (_NUM_EXPERTS,)
            for t in expert_scales
        )
    )


def _mkm_supported(block: Qwen2MoeSparseMoeBlock, m: int) -> bool:
    """SGLANG_Q35MF_MKM_FUSED and a call the fused engine covers; the static part is evaluated once per block."""
    if not envs.SGLANG_Q35MF_MKM_FUSED.get() or m > _MKM_M21_MAX_TOKENS:
        return False
    if m > _MKM_MAX_TOKENS:
        wide = (
            envs.SGLANG_Q35MF_MKM_FUSED_M16
            if m <= _MKM_M16_MAX_TOKENS
            else envs.SGLANG_Q35MF_MKM_FUSED_M21
        )
        if not (wide.get() and block.experts.intermediate_size_per_partition == 256):
            return False
    ok = _mkm_static_by_block.get(id(block))
    if ok is None:
        ok = _mkm_static_by_block[id(block)] = _mkm_static_supported(block)
    return ok


def _forward_mkm_fused(
    block: Qwen2MoeSparseMoeBlock,
    *,
    logits: torch.Tensor,
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
):
    """Top-10 routing, FC1, SwiGLU and FC2 in one launch; returns the trtllm deferred-finalize output."""
    from sglang.kernels.ops.moe.qwen_mkm_moe import mkm_moe
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    experts = block.experts
    m = logits.shape[0]
    gemm2_out, expert_weights = mkm_moe(
        logits,
        x_fp4,
        x_sf,
        experts.w13_weight,
        experts.w13_weight_scale.view(torch.uint8),
        experts.w2_weight,
        experts.w2_weight_scale.view(torch.uint8),
        experts.g1_alphas,
        experts.g1_scale_c,
        experts.g2_alphas,
    )
    # gemm2_out row t * 10 + k holds expert k of token t, so the permutation is the identity.
    expanded_idx = _mkm_expanded_idx_by_device.get(logits.device)
    if expanded_idx is None:
        expanded_idx = _mkm_expanded_idx_by_device[logits.device] = torch.arange(
            _MKM_M21_MAX_TOKENS * _TOP_K, dtype=torch.int32, device=logits.device
        )
    key = (block.topk.layer_id, m)
    if key not in _mkm_engaged:
        _mkm_engaged.add(key)
        logger.info("SGLANG_Q35MF_MKM_FUSED engaged: layer %s, %d tokens", key[0], m)
    deferred = FlashInferTrtllmDeferredFinalizeOutput(
        gemm2_out=gemm2_out,
        expert_weights=expert_weights,
        expanded_idx_to_permuted_idx=expanded_idx[: m * _TOP_K],
        top_k=_TOP_K,
    )
    return experts.dispatcher.combine(
        combine_input=StandardCombineInput(hidden_states=deferred)
    )


def mkm_wide_supported(
    block: Qwen2MoeSparseMoeBlock, hidden_states: torch.Tensor
) -> bool:
    """A 17..21-token call, past the router front, that SGLANG_Q35MF_MKM_FUSED_M21 takes from the block gate's logits."""
    m = hidden_states.shape[0]
    return (
        _MKM_M16_MAX_TOKENS < m <= _MKM_M21_MAX_TOKENS
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and block.supports_deferred_finalize
        # the stock runner quantizes such inputs with quantize_hidden_states_fp4, as forward_mkm_wide does
        and not block.experts.quant_method.quant_config.use_per_token_activation
        and block.experts.trtllm_padded_hidden_size is None
        and _mkm_supported(block, m)
    )


def forward_mkm_wide(
    block: Qwen2MoeSparseMoeBlock,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
):
    """The stock runner's input quantization, then the fused engine (the stock path's inputs, bit for bit)."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        quantize_hidden_states_fp4,
    )

    x_fp4, x_sf = quantize_hidden_states_fp4(
        hidden_states, block.experts.w13_input_scale_quant
    )
    return _forward_mkm_fused(
        block, logits=router_logits, x_fp4=x_fp4, x_sf=x_sf.view(torch.uint8)
    )


def forward_routed_rt(
    block: Qwen2MoeSparseMoeBlock, hidden_states: torch.Tensor, *, defer_finalize: bool
):
    """Routed experts on the trtllm runner with router logits and NVFP4 input from one launch.

    Same runner call as FusedMoE.forward_impl / forward_deferred_finalize, with the
    activation pre-quantized (the runner's hidden_states_scale passthrough).
    """
    from sglang.kernels.ops.moe.qwen_router_front import (
        router_gemm_quant,
        router_gemm_quant_shared,
    )

    experts = block.experts
    if hidden_states.shape[0] > _MAX_TOKENS:
        logits, x_fp4, x_sf = _logits_and_quant_parallel(block, hidden_states)
        return _run_rt_experts(
            block, hidden_states, logits, x_fp4, x_sf, defer_finalize
        )
    # The runner keeps the input scale as a 0-d tensor; the kernel takes shape [1].
    in_scale = experts.w13_input_scale_quant.reshape(1)
    if _use_shared(block, hidden_states.shape[0], defer_finalize):
        gu = block.shared_expert.gate_up_proj
        logits, x_fp4, x_sf, gate_up = router_gemm_quant_shared(
            hidden_states,
            block.gate.weight,
            in_scale,
            gu.weight.t(),
            gu.input_scale.reshape(1),
            gu.weight_scale.reshape(1),
        )
        gate_up.record_stream(block.alt_stream)
        block._route_sh_gate_up = gate_up
        block._route_sh_event.record()
    else:
        logits, x_fp4, x_sf = router_gemm_quant(
            hidden_states, block.gate.weight, in_scale
        )
    if defer_finalize and _mkm_supported(block, hidden_states.shape[0]):
        return _forward_mkm_fused(block, logits=logits, x_fp4=x_fp4, x_sf=x_sf)
    return _run_rt_experts(block, hidden_states, logits, x_fp4, x_sf, defer_finalize)


def _logits_and_quant_parallel(
    block: Qwen2MoeSparseMoeBlock, hidden_states: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stock router GEMM on this stream and the runner's own NVFP4 input quantization on the side
    stream, so the quantization leaves the critical path (same kernels and outputs as stock)."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        quantize_hidden_states_fp4,
    )

    m = hidden_states.shape[0]
    if m not in _parq_logged:
        _parq_logged.add(m)
        logger.info(
            "ROUTE_PARQ: NVFP4 input quantization on the side stream at %d tokens", m
        )
    current = torch.cuda.current_stream()
    side = block.alt_stream
    if envs.SGLANG_Q35MF_K4_PARQ_STREAM.get():
        from sglang.srt.runtime_context import get_stream

        side = get_stream("moe_input_quant")
    side.wait_stream(current)
    with torch.cuda.stream(side):
        x_fp4, x_sf = quantize_hidden_states_fp4(
            hidden_states, block.experts.w13_input_scale_quant
        )
    logits, _ = block.gate(hidden_states)
    current.wait_stream(side)
    x_fp4.record_stream(current)
    x_sf.record_stream(current)
    return logits, x_fp4, x_sf


def _run_rt_experts(
    block: Qwen2MoeSparseMoeBlock,
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    defer_finalize: bool,
):
    """FusedMoE.forward_impl / forward_deferred_finalize with the runner's pre-quantized input passthrough."""
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        flashinfer_trtllm_deferred_finalize_context,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.runtime_context import get_parallel

    experts = block.experts
    topk_output = block.topk(hidden_states, logits)
    if not TopKOutputChecker.format_is_bypassed(topk_output):
        raise RuntimeError("router_gemm_quant path requires logits-based bypassed TopK")
    dispatch_output = experts.dispatcher.dispatch(
        hidden_states=hidden_states, topk_output=topk_output
    )._replace(hidden_states=x_fp4, hidden_states_scale=x_sf.view(torch.float8_e4m3fn))
    if defer_finalize:
        with (
            flashinfer_trtllm_deferred_finalize_context(),
            _tables_context(hidden_states.shape[0]),
        ):
            combine_input = experts.run_moe_core(dispatch_output=dispatch_output)
        return experts.dispatcher.combine(combine_input=combine_input)
    with _tables_context(hidden_states.shape[0]):
        combine_input = experts.run_moe_core(dispatch_output=dispatch_output)
    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        final_hidden_states = experts.dispatcher.combine(combine_input=combine_input)
        final_hidden_states = final_hidden_states[..., :_HIDDEN].contiguous()
    return final_hidden_states


def _tables_context(m: int):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        flashinfer_trtllm_fp4_from_tables_context,
    )

    max_tokens = min(
        envs.SGLANG_Q35MF_ROUTE_TABLES_MAX_TOKENS.get(), _parq_max_tokens()
    )
    if _MAX_TOKENS < m <= max_tokens or (
        envs.SGLANG_Q35MF_ROUTE2_SMALL_TABLES_MIN_TOKENS.get() <= m <= _MAX_TOKENS
        and envs.SGLANG_Q35MF_ROUTE2_SMALL_TABLES.get()
    ):
        return flashinfer_trtllm_fp4_from_tables_context(_fp4_from_tables)
    return contextlib.nullcontext()


def _fp4_from_tables(kw: dict) -> Optional[List[torch.Tensor]]:
    """trtllm_fp4_block_scale_moe(**kw) with the top-k and routing tables from router_topk_tables:
    FlashInfer's FP4 body from precomputed routing metadata, at the tactic the stock call picks.
    None keeps the stock call."""
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.shared.inputs import RoutingInputMode

    from sglang.kernels.ops.moe.qwen_router_tables import (
        RouterTables,
        router_topk_tables,
        router_topk_tables_small,
    )

    m, fin = kw["hidden_states"].shape[0], kw["do_finalize"]
    if AutoTuner.get().is_tuning_mode or not _fp4_from_tables_supported(kw):
        return None
    # Graph capture replays the buffers the eager warmup forwards allocated.
    capturing = torch.cuda.is_current_stream_capturing()
    stock = _tables_tactics.get((m, fin))
    if stock is None:
        if capturing:
            return _tables_fallback(m, "first call under graph capture")
        stock = _tables_tactics[(m, fin)] = _stock_runner_and_tactic(kw)
    runner, tactic = stock
    # -1 = untuned (the launch picks its own config); a tuned tactic is a (tile_n, config) FFI array.
    if tactic == -1:
        return _tables_fallback(m, "no tuned tactic")
    tile_n, config = int(tactic[0]), int(tactic[1])
    body_key = (m, tile_n, config, fin)
    body = _tables_bodies.get(body_key)
    if body is None and capturing:
        return _tables_fallback(m, "first call under graph capture")
    tables = _tables_by_tile.get((m, tile_n))
    if tables is None:
        tables = _tables_by_tile[(m, tile_n)] = RouterTables(
            m, tile_n, device=kw["hidden_states"].device
        )
    if m <= _MAX_TOKENS:
        router_topk_tables_small(
            kw["routing_logits"],
            tables,
            early_trigger=envs.SGLANG_Q35MF_ROUTE2_EARLY_TRIGGER.get(),
        )
    else:
        router_topk_tables(kw["routing_logits"], tables)
    inputs = _fp4_body_inputs(kw, tables)
    run_kw = _fp4_runner_kwargs(kw, RoutingInputMode.PackedPrecomputed)
    if body is None:
        prepared = runner.forward(
            inputs,
            tactic=[tile_n, config],
            do_preparation=True,
            da_routing_metadata=tables.metadata(),
            **run_kw,
        )
        body = _tables_bodies[body_key] = tuple(_as_torch(t) for t in prepared)
    runner.forward(
        inputs,
        tactic=[tile_n, config],
        da_routing_metadata=tables.metadata(),
        da_body_workspace=body,
        **run_kw,
    )
    if body_key not in _tables_logged:
        _tables_logged.add(body_key)
        logger.info(
            "ROUTE_TABLES: %s + FlashInfer FP4 body from precomputed routing "
            "at %d tokens (tile %d, config %d, finalize %s, early trigger %s)",
            "router_topk_tables_small" if m <= _MAX_TOKENS else "router_topk_tables",
            m,
            tile_n,
            config,
            fin,
            m <= _MAX_TOKENS and envs.SGLANG_Q35MF_ROUTE2_EARLY_TRIGGER.get(),
        )
    if fin:
        return [kw["output"]]
    # FlashInfer's do_finalize=False result: GEMM2 rows, expert weights, expanded -> permuted.
    return [body[4], tables.expert_weights, tables.expanded_idx_to_permuted_idx]


def _fp4_from_tables_supported(kw: dict) -> bool:
    """The routing router_topk_tables reproduces: softmax over the top-10 of 512 experts, no bias
    or scaling, all experts local, NVFP4 input with block scales."""
    from sglang.srt.layers.moe.utils import RoutingMethodType

    return (
        kw["routing_method_type"] == RoutingMethodType.RenormalizeNaive
        and kw["routing_bias"] is None
        and kw["routed_scaling_factor"] is None
        and not kw["n_group"]
        and not kw["topk_group"]
        and kw["top_k"] == _TOP_K
        and kw["num_experts"] == _NUM_EXPERTS
        and kw["local_num_experts"] == _NUM_EXPERTS
        and kw["local_expert_offset"] == 0
        and kw["per_token_scale"] is None
        and kw["hidden_states"].dtype == torch.uint8
        and kw["hidden_states_scale"] is not None
        and kw["routing_logits"].dtype == torch.bfloat16
        and kw["routing_logits"].is_contiguous()
    )


def _tables_fallback(m: int, reason: str) -> None:
    if (m, reason) not in _tables_logged:
        _tables_logged.add((m, reason))
        logger.warning("ROUTE_TABLES: stock routing at %d tokens: %s", m, reason)
    return None


def _stock_runner_and_tactic(kw: dict) -> Tuple[Any, Any]:
    """FlashInfer's runner for trtllm_fp4_block_scale_moe(**kw) and the tactic that call launches:
    the same tuner query as flashinfer.fused_moe.core.trtllm_fp4_block_scale_moe_op (0.7.1)."""
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs, RoutingInputMode
    from flashinfer.tllm_enums import (
        Fp8QuantizationType,
        WeightLayout,
        deduce_trtllm_gen_tensor_dtype,
    )

    hs, hs_scale = kw["hidden_states"], kw["hidden_states_scale"]
    m, hidden = hs.shape[0], hs.shape[-1] * 2
    mod = get_trtllm_moe_sm100_module()
    runner = mod.MoERunner(
        mod.moe_op,
        top_k=kw["top_k"],
        num_local_experts=kw["local_num_experts"],
        dtype_act=deduce_trtllm_gen_tensor_dtype(hs, hs_scale),
        dtype_weights=deduce_trtllm_gen_tensor_dtype(
            kw["gemm1_weights"], kw["gemm1_weights_scale"]
        ),
        fp8_quantization_type=Fp8QuantizationType.NoneFp8,
        hidden_size=hidden,
        hidden_size_output=hidden,
        intermediate_size=kw["intermediate_size"],
        activation_type=kw["activation_type"],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        use_per_token_scaling=False,
        num_experts=kw["num_experts"],
        num_fused_shared_experts=0,
    )
    inputs = MoeRunnerInputs(
        output=_fp4_output(kw),
        routing_logits=kw["routing_logits"],
        topk_ids=hs.new_empty((m, kw["top_k"]), dtype=torch.int32),
        expert_weights=hs.new_empty((m, kw["top_k"]), dtype=torch.bfloat16),
        hidden_states=hs,
        hidden_states_scale=hs_scale,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )
    tuning_config = runner._make_tuning_config(
        inputs,
        tune_max_num_tokens=kw["tune_max_num_tokens"],
        routing_input_mode=RoutingInputMode.FromLogits,
        hidden_states_scale_layout=None,
        use_cold_l2_cache=True,
        use_cuda_graph=True,
    )
    _, tactic = AutoTuner.get().choose_one(
        "flashinfer::trtllm_fp4_block_scale_moe",
        [runner],
        tuning_config,
        inputs.to_list(),
        **_fp4_runner_kwargs(kw, RoutingInputMode.FromLogits),
    )
    return runner, tactic


def _fp4_output(kw: dict) -> torch.Tensor:
    if kw["do_finalize"]:
        return kw["output"]
    # FlashInfer's zero-width placeholder for do_finalize=False (alloc_trtllm_moe_output).
    m = kw["hidden_states"].shape[0]
    return kw["hidden_states"].new_empty((m, 0), dtype=torch.bfloat16)


def _fp4_body_inputs(kw: dict, tables) -> List[Optional[torch.Tensor]]:
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs

    # topk_ids only satisfies the packed-routing ABI; the body reads the routing metadata.
    return MoeRunnerInputs(
        output=_fp4_output(kw),
        routing_logits=None,
        topk_ids=tables.topk_ids,
        expert_weights=tables.expert_weights,
        hidden_states=kw["hidden_states"],
        hidden_states_scale=kw["hidden_states_scale"],
        gemm1_lora_delta=None,
        per_token_scale=None,
    ).to_list()


def _fp4_runner_kwargs(kw: dict, routing_input_mode) -> dict:
    """The runner kwargs trtllm_fp4_block_scale_moe_op (FlashInfer 0.7.1) builds from kw."""
    from flashinfer.fused_moe.core import _device_support_moe_pdl

    return {
        "routing_input_mode": routing_input_mode,
        "num_experts": kw["num_experts"],
        "routing_bias": kw["routing_bias"],
        "gemm1_weights": kw["gemm1_weights"],
        "gemm1_weights_scale": kw["gemm1_weights_scale"],
        "gemm1_bias": kw["gemm1_bias"],
        "gemm1_alpha": kw["gemm1_alpha"],
        "gemm1_beta": kw["gemm1_beta"],
        "gemm1_clamp_limit": kw["gemm1_clamp_limit"],
        "gemm2_weights": kw["gemm2_weights"],
        "gemm2_weights_scale": kw["gemm2_weights_scale"],
        "gemm2_bias": kw["gemm2_bias"],
        "output1_scale_scalar": kw["output1_scale_scalar"],
        "output1_scale_gate_scalar": kw["output1_scale_gate_scalar"],
        "output2_scale_scalar": kw["output2_scale_scalar"],
        "n_group": kw["n_group"],
        "topk_group": kw["topk_group"],
        "local_expert_offset": kw["local_expert_offset"],
        "routed_scaling_factor": kw["routed_scaling_factor"],
        "routing_method_type": kw["routing_method_type"],
        "enable_pdl": kw["enable_pdl"]
        and _device_support_moe_pdl(kw["hidden_states"].device),
        "do_finalize": kw["do_finalize"],
        "activation_type": kw["activation_type"],
        "num_fused_shared_experts": 0,
        "norm_topk_prob": True,
        "routing_replay_out": None,
        "valid_hidden_size": None,
        "valid_intermediate_size": None,
    }


def _as_torch(t) -> torch.Tensor:
    return t if isinstance(t, torch.Tensor) else torch.from_dlpack(t)


def _cutedsl_ready(experts) -> bool:
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
        ensure_cutedsl_wrapper,
    )

    ensure_cutedsl_wrapper(experts)
    wrapper = experts._cutedsl_wrapper
    return (
        wrapper.quant_mode == "w4a4"
        and not experts.quant_method.quant_config.use_per_token_activation
        and experts._cutedsl_input_scale.numel() == 1
    )


def _choose_tactic(wrapper, inputs) -> Tuple:
    """The tactic wrapper.run would pick for these inputs (same op name and cache)."""
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.cute_dsl.tuner import _get_default_tactic

    runner = wrapper._runner
    activation_name = (
        "Situ" if wrapper.situ_beta is not None else wrapper.activation_type.name
    )
    op_name = f"CuteDslMoEWrapper::run::w4a4::{activation_name}"
    _, tactic = AutoTuner.get().choose_one(
        op_name, [runner], runner.tuning_config, inputs
    )
    if tactic is None or tactic == -1:
        tactic = _get_default_tactic()
    return tactic


def forward_routed(
    block: Qwen2MoeSparseMoeBlock, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Routed-expert output [M, hidden] for ``hidden_states`` [M, hidden] (bf16)."""
    from flashinfer.autotuner import AutoTuner

    from sglang.kernels.ops.moe.qwen_router_front import (
        RouterFrontBuffers,
        router_front,
        router_tail,
    )
    from sglang.srt.layers.moe.topk import build_precomputed_topk_output

    experts = block.experts
    wrapper = experts._cutedsl_wrapper
    w1_alpha, fc2_input_scale, w2_alpha = experts._cutedsl_scales
    w13_sf = experts.w13_blockscale_mma
    w2_sf = experts.w2_blockscale_mma
    m = hidden_states.shape[0]

    # Tile size depends on the tuned tactic: size the permutation table for the largest.
    bufs = RouterFrontBuffers(
        m, tile_size=256, device=hidden_states.device, arrival=block._route_arrival
    )
    x_sf = bufs.x_sf.view(torch.float8_e4m3fn)
    moe_output = torch.empty(
        (m, experts.hidden_size),
        dtype=wrapper.output_dtype,
        device=hidden_states.device,
    )
    inputs = [
        bufs.x_fp4,
        x_sf,
        bufs.topk_ids,
        bufs.topk_weights,
        experts.w13_weight,
        w13_sf,
        w1_alpha,
        fc2_input_scale,
        experts.w2_weight,
        w2_sf,
        w2_alpha,
        moe_output,
    ]
    tactic = _choose_tactic(wrapper, inputs)
    bufs.set_tile_size(tactic[0])
    if envs.SGLANG_Q35MF_ROUTE_GEMV.get():
        router_front(
            hidden_states, block.gate.weight, experts._cutedsl_input_scale, bufs
        )
    else:
        router_logits, _ = block.gate(hidden_states)
        router_tail(router_logits, hidden_states, experts._cutedsl_input_scale, bufs)
    # Capture hook and expert-distribution recorder, as select_experts would run them.
    build_precomputed_topk_output(
        bufs.topk_weights, bufs.topk_ids, block.topk.topk_config, block.topk.layer_id
    )
    return _run_experts_from_tables(
        wrapper=wrapper,
        bufs=bufs,
        x_sf=x_sf,
        w1_weight=experts.w13_weight,
        w1_weight_sf=w13_sf,
        w1_alpha=w1_alpha,
        fc2_input_scale=fc2_input_scale,
        w2_weight=experts.w2_weight,
        w2_weight_sf=w2_sf,
        w2_alpha=w2_alpha,
        tactic=tactic,
        moe_output=moe_output,
        use_async_memset=not AutoTuner.get().is_tuning_mode,
    )


def _run_experts_from_tables(
    *,
    wrapper,
    bufs,
    x_sf: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    fc2_input_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    tactic: Tuple,
    moe_output: torch.Tensor,
    use_async_memset: bool,
) -> torch.Tensor:
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
        blockscaled_contiguous_gather_grouped_gemm_act_fusion,
    )
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
        blockscaled_contiguous_grouped_gemm_finalize_fusion,
    )
    from flashinfer.fused_moe.cute_dsl.fused_moe import _get_cuda_graph_resources
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        moe_output_memset_inplace,
        moe_unpermute,
    )
    from flashinfer.fused_moe.cute_dsl.tuner import _extract_tactic_params

    params = _extract_tactic_params(tactic)
    num_tokens = bufs.num_tokens
    top_k = _TOP_K
    hidden_size = w2_weight.size(1)
    use_fused_finalize = wrapper.use_fused_finalize
    enable_pdl = wrapper.enable_pdl

    aux_stream: Optional[torch.cuda.Stream] = wrapper._aux_stream
    main_event = wrapper._main_event
    memset_event = wrapper._memset_event
    if use_async_memset and use_fused_finalize:
        if aux_stream is None or main_event is None or memset_event is None:
            resources = _get_cuda_graph_resources()
            aux_stream = aux_stream or resources["aux_stream"]
            main_event = main_event or resources["main_event"]
            memset_event = memset_event or resources["memset_event"]
        main_event.record()
        moe_output.record_stream(aux_stream)

    intermediate, intermediate_sf = (
        blockscaled_contiguous_gather_grouped_gemm_act_fusion(
            a=bufs.x_fp4,
            b=w1_weight,
            a_scale=x_sf,
            b_scale=w1_weight_sf,
            alpha=w1_alpha,
            tile_idx_to_expert_idx=bufs.tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=bufs.tile_idx_to_mn_limit,
            token_id_mapping=bufs.permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=bufs.num_non_exiting_tiles,
            out=None,
            out_scale=None,
            global_scale=fc2_input_scale,
            a_per_token_scale=None,
            c_dtype="float4_e2m1fn",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            sf_vec_size=16,
            quantize_output=True,
            topk=top_k,
            mma_tiler_mn=params["gemm1_mma_tiler_mn"],
            cluster_shape_mn=params["gemm1_cluster_shape_mn"],
            mma_tiler=params["gemm1_mma_tiler"],
            mma_inst_shape=params["gemm1_mma_inst_shape"],
            enable_pdl=enable_pdl,
            activation_type=wrapper.activation_type.value,
            swiglu_alpha=wrapper.swiglu_alpha,
            swiglu_beta=wrapper.swiglu_beta,
            swiglu_limit=wrapper.swiglu_limit,
            situ_beta=wrapper.situ_beta,
            situ_linear_beta=wrapper.situ_linear_beta,
            gated=wrapper.gated,
        )
    )

    if use_fused_finalize:
        if use_async_memset:
            with torch.cuda.stream(aux_stream):
                main_event.wait()
                moe_output_memset_inplace(moe_output)
                memset_event.record()
            memset_event.wait()
        else:
            moe_output_memset_inplace(moe_output)
        gemm2_output = moe_output
    else:
        gemm2_output = torch.empty(
            (num_tokens * top_k, hidden_size),
            dtype=wrapper.output_dtype,
            device=moe_output.device,
        )

    blockscaled_contiguous_grouped_gemm_finalize_fusion(
        a=intermediate,
        b=w2_weight,
        a_scale=intermediate_sf,
        b_scale=w2_weight_sf,
        alpha=w2_alpha,
        tile_idx_to_expert_idx=bufs.tile_idx_to_expert_idx,
        num_non_exiting_tiles=bufs.num_non_exiting_tiles,
        tile_idx_to_mn_limit=bufs.tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=bufs.permuted_idx_to_expanded_idx,
        token_final_scales=bufs.topk_weights,
        out=gemm2_output,
        a_per_token_scale=None,
        a_dtype="float4_e2m1fn",
        b_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        sf_vec_size=16,
        out_dtype="bfloat16",
        mma_tiler_mn=params["gemm2_mma_tiler_mn"],
        mma_tiler=params["gemm2_mma_tiler"],
        mma_inst_shape=params["gemm2_mma_inst_shape"],
        cluster_shape_mn=params["gemm2_cluster_shape_mn"],
        enable_pdl=enable_pdl,
        use_fused_finalize=use_fused_finalize,
    )

    if not use_fused_finalize:
        moe_unpermute(
            permuted_input=gemm2_output,
            output=moe_output,
            expanded_idx_to_permuted_idx=bufs.expanded_idx_to_permuted_idx,
            topk_scales=bufs.topk_weights,
            num_tokens=num_tokens,
            top_k=top_k,
            input_is_expanded=True,
            enable_pdl=enable_pdl,
        )
    return moe_output
