"""Prepared-routing decode path for Qwen3.8's NVFP4 TRT-LLM MoE (SGLANG_ENABLE_Q38_FUSED_ROUTING).

At 1-32 rows of a deferred-finalize call (up to 512 with
SGLANG_Q38_FUSED_ROUTING_MAX_ROWS raised), one fused kernel
(kernels/ops/moe/q38_fused_routing.py) computes the top-10 routing, the routing
metadata and the NVFP4 activation quant, and the routed MoE runs only its GEMMs
from that metadata through FlashInfer's prepared-routing (DA) body entry, so
FlashInfer's routing kernel and the separate quant kernel are skipped. The body
runs the tactic FlashInfer's tuner returns for the unfused call at this M,
looked up with the same runner, inputs and tuning config, so its GEMM kernels and
outputs are unchanged. Adapted from select_trtllm_routed_body and
_run_trtllm_prepared_routing_moe (sgl-project/sglang #48586).
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_ENABLED = envs.SGLANG_ENABLE_Q38_FUSED_ROUTING.get()
# 1-32 rows run the single-CTA kernel, 33-512 the cluster kernel; the default cap keeps 1-32.
_MAX_ROWS = min(envs.SGLANG_Q38_FUSED_ROUTING_MAX_ROWS.get(), 512)
_MOE_OP = "flashinfer::trtllm_fp4_block_scale_moe"
# Rows above DynBlock's 16 take FlashInfer's cluster kernel only when the
# block-per-token split path is forced off; the split path's math is not checked here.
_DYNBLOCK_MAX_ROWS = 16
_ENGAGED_LOGGED = False
_ENGAGED_CLUSTER_LOGGED = False


def wants(num_tokens: int) -> bool:
    """The decode MoE at this row count takes the fused path, so the caller skips its prequant."""
    return _ENABLED and 1 <= num_tokens <= _MAX_ROWS


@functools.lru_cache(maxsize=None)
def _log_once(message: str) -> None:
    logger.info(message)


def _quant_recipe_matches() -> bool:
    # The fused quant reproduces the CuTe-DSL fp4_quantize standard recipe only.
    from sglang.srt.layers.quantization import fp4_utils

    flags = ("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "FLASHINFER_NVFP4_4OVER6")
    return fp4_utils._flashinfer_fp4_quantize_backend == "cute-dsl" and all(
        os.environ.get(flag, "") in ("", "0") for flag in flags
    )


def _miss_reason(hidden_states, topk_output, quant_info) -> Optional[str]:
    import flashinfer
    from flashinfer.autotuner import AutoTuner

    from sglang.kernels.ops.moe import q38_fused_routing as op
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.runtime_context import get_parallel

    config = topk_output.topk_config
    num_tokens = hidden_states.shape[0]
    checks = (
        # The tuner-key mirror, the default tile and the PDL gate follow FlashInfer 0.7.2's sources.
        (flashinfer.__version__ == "0.7.2", "flashinfer version"),
        (config.top_k == op.TOP_K and config.correction_bias is None, "routing config"),
        (not config.num_expert_group and not config.topk_group, "grouped routing"),
        (quant_info.routing_method_type == RoutingMethodType.RenormalizeNaive, "routing method"),
        (
            quant_info.global_num_experts == op.NUM_EXPERTS
            and quant_info.local_num_experts == op.NUM_EXPERTS
            and quant_info.local_expert_offset == 0,
            "expert layout",
        ),
        (get_parallel().tp_size == 1, "tp_size"),
        (_quant_recipe_matches(), "fp4 quant recipe"),
        (
            num_tokens <= _DYNBLOCK_MAX_ROWS
            or os.environ.get("FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN") in ("0", "off", "OFF"),
            "routing kernel above 16 rows",
        ),
        (
            op.covered(topk_output.router_logits, hidden_states, quant_info.w13_input_scale_quant),
            "tensor layout",
        ),
        (not AutoTuner.get().is_tuning_mode, "autotuning"),
        (op.available(), "kernel unavailable"),
    )
    return next((reason for ok, reason in checks if not ok), None)


def _default_tile_n(num_tokens: int, num_local_experts: int, rubin: bool) -> int:
    # FlashInfer 0.7.2 selectDefaultTileN over FP4BlockScaleLauncher::getSupportedTileNums
    # (E2m1 x E2m1; the Rubin build has no 192 tile).
    tiles = (8, 16, 32, 64, 128) + (() if rubin else (192,)) + (256,)
    average = num_tokens * 10 / num_local_experts
    i = next((i for i, t in enumerate(tiles) if t >= average), len(tiles) - 1)
    return tiles[max(i - 1, 0)]


def _stock_runner(moe_kwargs: dict, inputs):
    """The MoERunner, tuning config and runner kwargs trtllm_fp4_block_scale_moe_op
    builds for this call (FlashInfer 0.7.2 core.py), so the tuner key is the same."""
    from flashinfer.fused_moe.backends.trtllm import MoERunner
    from flashinfer.fused_moe.core import (
        _device_support_moe_pdl,
        get_trtllm_moe_sm100_module,
    )
    from flashinfer.tllm_enums import (
        Fp8QuantizationType,
        RoutingInputMode,
        WeightLayout,
        deduce_trtllm_gen_tensor_dtype,
    )

    def trtllm_fp4_block_scale_moe_op(device) -> bool:
        # Named after FlashInfer's op: SGLANG_TRTLLM_MOE_SM107_PDL's gate keys on the caller.
        return _device_support_moe_pdl(device)

    hidden_size = inputs.hidden_states.shape[-1] * 2
    runner = MoERunner(
        get_trtllm_moe_sm100_module().moe_op,
        top_k=moe_kwargs["top_k"],
        num_local_experts=moe_kwargs["local_num_experts"],
        dtype_act=deduce_trtllm_gen_tensor_dtype(inputs.hidden_states, inputs.hidden_states_scale),
        dtype_weights=deduce_trtllm_gen_tensor_dtype(
            moe_kwargs["gemm1_weights"], moe_kwargs["gemm1_weights_scale"]
        ),
        fp8_quantization_type=Fp8QuantizationType.NoneFp8,
        hidden_size=hidden_size,
        hidden_size_output=hidden_size,
        intermediate_size=moe_kwargs["intermediate_size"],
        activation_type=moe_kwargs["activation_type"],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        use_per_token_scaling=False,
        num_experts=moe_kwargs["num_experts"],
        num_fused_shared_experts=0,
    )
    tuning_config = runner._make_tuning_config(
        inputs,
        tune_max_num_tokens=moe_kwargs["tune_max_num_tokens"],
        routing_input_mode=RoutingInputMode.FromLogits,
        hidden_states_scale_layout=None,
        use_cold_l2_cache=True,
        use_cuda_graph=True,
    )
    device = inputs.hidden_states.device
    runner_kwargs = {
        "routing_input_mode": int(RoutingInputMode.FromLogits),
        "num_experts": moe_kwargs["num_experts"],
        "routing_bias": moe_kwargs["routing_bias"],
        "gemm1_weights": moe_kwargs["gemm1_weights"],
        "gemm1_weights_scale": moe_kwargs["gemm1_weights_scale"],
        "gemm1_bias": moe_kwargs["gemm1_bias"],
        "gemm1_alpha": moe_kwargs["gemm1_alpha"],
        "gemm1_beta": moe_kwargs["gemm1_beta"],
        "gemm1_clamp_limit": moe_kwargs["gemm1_clamp_limit"],
        "gemm2_weights": moe_kwargs["gemm2_weights"],
        "gemm2_weights_scale": moe_kwargs["gemm2_weights_scale"],
        "gemm2_bias": moe_kwargs["gemm2_bias"],
        "output1_scale_scalar": moe_kwargs["output1_scale_scalar"],
        "output1_scale_gate_scalar": moe_kwargs["output1_scale_gate_scalar"],
        "output2_scale_scalar": moe_kwargs["output2_scale_scalar"],
        "n_group": moe_kwargs["n_group"],
        "topk_group": moe_kwargs["topk_group"],
        "local_expert_offset": moe_kwargs["local_expert_offset"],
        "routed_scaling_factor": moe_kwargs["routed_scaling_factor"],
        "routing_method_type": moe_kwargs["routing_method_type"],
        "enable_pdl": bool(moe_kwargs["enable_pdl"]) and trtllm_fp4_block_scale_moe_op(device),
        "do_finalize": False,
        "activation_type": moe_kwargs["activation_type"],
        "num_fused_shared_experts": 0,
        "norm_topk_prob": True,
        "routing_replay_out": None,
        "valid_hidden_size": None,
        "valid_intermediate_size": None,
    }
    return runner, tuning_config, runner_kwargs


def run(*, hidden_states, topk_output, quant_info, moe_kwargs: dict) -> Optional[list]:
    """The stock call's do_finalize=False result, or None (the caller then quantizes
    and runs the stock call).

    ``moe_kwargs`` is the stock call's argument dict; its hidden_states and
    hidden_states_scale are unfilled buffers from alloc_quant_outputs, which the
    fused kernel fills."""
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs

    from sglang.kernels.ops.moe import q38_fused_routing as op

    reason = _miss_reason(hidden_states, topk_output, quant_info)
    if reason is None and moe_kwargs.get("output") is None:
        reason = "no full-width output buffer"
    if reason is not None:
        if reason != "autotuning":  # the tuning pass runs the stock call by design
            _log_once(f"Q38 fused routing not taken: {reason} (SGLANG_ENABLE_Q38_FUSED_ROUTING)")
        return None

    num_tokens = hidden_states.shape[0]
    router_logits = topk_output.router_logits
    device = hidden_states.device
    # Same shapes and dtypes as the stock op's own topk buffers: part of the tuner key.
    ids = torch.empty((num_tokens, op.TOP_K), dtype=torch.int32, device=device)
    weights = torch.empty((num_tokens, op.TOP_K), dtype=torch.bfloat16, device=device)
    inputs = MoeRunnerInputs(
        output=moe_kwargs["output"],
        routing_logits=router_logits,
        topk_ids=ids,
        expert_weights=weights,
        hidden_states=moe_kwargs["hidden_states"],
        hidden_states_scale=moe_kwargs["hidden_states_scale"],
        gemm1_lora_delta=None,
        per_token_scale=None,
    )
    runner, tuning_config, runner_kwargs = _stock_runner(moe_kwargs, inputs)
    _, tactic = AutoTuner.get().choose_one(
        _MOE_OP, [runner], tuning_config, inputs.to_list(), **runner_kwargs
    )
    tactic = [-1, -1] if isinstance(tactic, int) else [int(v) for v in tactic]
    if min(tactic) < 0:
        rubin = torch.cuda.get_device_capability(device) == (10, 7)
        tile_n = _default_tile_n(num_tokens, moe_kwargs["local_num_experts"], rubin)
    else:
        tile_n = tactic[0]

    routed = op.fused_routing(
        router_logits,
        hidden_states,
        quant_info.w13_input_scale_quant,
        x_fp4=moe_kwargs["hidden_states"],
        x_sf=moe_kwargs["hidden_states_scale"],
        ids=ids,
        weights=weights,
        tile_tokens=tile_n,
    )
    metadata = list(routed.metadata.tensors())
    flat_inputs = inputs.to_list()
    workspace = runner.forward(
        flat_inputs, tactic=tactic, do_preparation=True, da_routing_metadata=metadata, **runner_kwargs
    )
    gemm2_out = runner.forward(
        flat_inputs, tactic=tactic, da_routing_metadata=metadata, da_body_workspace=workspace, **runner_kwargs
    )[0]
    if not isinstance(gemm2_out, torch.Tensor):
        gemm2_out = torch.from_dlpack(gemm2_out)
    # Once per process for each kernel.
    global _ENGAGED_LOGGED, _ENGAGED_CLUSTER_LOGGED
    if num_tokens <= op.MAX_TOKENS and not _ENGAGED_LOGGED:
        _ENGAGED_LOGGED = True
        logger.info(
            "Q38 fused routing engaged: rows=%d max_rows=%d tile_n=%d tactic=%s pdl=%s "
            "(SGLANG_ENABLE_Q38_FUSED_ROUTING)",
            num_tokens, _MAX_ROWS, tile_n, tactic, runner_kwargs["enable_pdl"],
        )
    if num_tokens > op.MAX_TOKENS and not _ENGAGED_CLUSTER_LOGGED:
        _ENGAGED_CLUSTER_LOGGED = True
        logger.info(
            "Q38 fused routing engaged: rows=%d max_rows=%d tile_n=%d tactic=%s pdl=%s "
            "kernel=cluster%d (SGLANG_ENABLE_Q38_FUSED_ROUTING)",
            num_tokens, _MAX_ROWS, tile_n, tactic, runner_kwargs["enable_pdl"], op.cluster_ctas(),
        )
    return [gemm2_out, weights, routed.metadata.expanded_idx_to_permuted_idx]
