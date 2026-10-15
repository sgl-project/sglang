"""Single-kernel router front end for Qwen MoE blocks on the CuTe DSL NVFP4 runner.

At decode token counts the stock chain after the router GEMM (top-k, NVFP4
input quant, moe_sort) is replaced by one kernel that writes the same tensors
the grouped GEMMs consume; FC1 / FC2 / finalize then run as in FlashInfer's
``_moe_core_impl``. Enabled by SGLANG_Q35MF_ROUTE_FUSED (SGLANG_Q35MF_ROUTE_GEMV
also moves the router GEMM into the kernel); every unsupported configuration
keeps the stock path.
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
# tables.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

_HIDDEN = 4096
_NUM_EXPERTS = 512
_TOP_K = 10
_MAX_TOKENS = 16


def enabled() -> bool:
    return envs.SGLANG_Q35MF_ROUTE_FUSED.get()


def _static_supported(block: "Qwen2MoeSparseMoeBlock") -> bool:
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


def supported(block: "Qwen2MoeSparseMoeBlock", hidden_states: torch.Tensor) -> bool:
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
    block: "Qwen2MoeSparseMoeBlock", hidden_states: torch.Tensor
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
        (m, experts.hidden_size), dtype=wrapper.output_dtype, device=hidden_states.device
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

    intermediate, intermediate_sf = blockscaled_contiguous_gather_grouped_gemm_act_fusion(
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
